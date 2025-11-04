#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <cuda_runtime.h>
#include <pthread.h>
#include <cstdio>
#include <signal.h>
#include <execinfo.h>

// Global completion flag for synchronization
std::atomic<bool> transfer_completed{false};

// Track main thread ID for comparison
pthread_t main_thread_id;

// Signal handler for segfaults
void segfault_handler(int sig)
{
    std::cout << "\n=== SEGMENTATION FAULT DETECTED ===" << std::endl;
    std::cout << "Signal: " << sig << std::endl;

    void *array[10];
    size_t size = backtrace(array, 10);
    char **strings = backtrace_symbols(array, size);

    std::cout << "Stack trace:" << std::endl;
    for (size_t i = 0; i < size; i++)
    {
        std::cout << "  " << strings[i] << std::endl;
    }

    free(strings);
    exit(1);
}

void MyCompletionCallback(void *ctx1, void *ctx2, int32_t status, size_t bytes)
{
    // Execute callback directly to avoid thread deadlock during cleanup
    std::cout << "\n=== GPU Transfer Completion Callback TRIGGERED ===" << std::endl;
    std::cout << "Status: " << status << ", Bytes: " << bytes << std::endl;

    if (status == -734011) {
        std::cout << "Transfer was cancelled (session likely closed prematurely)" << std::endl;
        transfer_completed = true;
        return;
    }

    if (status != easyrdma_Error_Success) {
        std::cout << "Transfer failed with error status: " << status << std::endl;
        transfer_completed = true;
        return;
    }

    if (!ctx1) {
        std::cout << "Error: GPU buffer pointer is null!" << std::endl;
        transfer_completed = true;
        return;
    }

    if (status == easyrdma_Error_Success && bytes > 0) {
        std::cout << "Successfully received " << bytes << " bytes into GPU memory!" << std::endl;

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cout << "GPU sync error: " << cudaGetErrorString(err) << std::endl;
            transfer_completed = true;
            return;
        }

        std::vector<char> hostBuffer(bytes);
        err = cudaMemcpy(hostBuffer.data(), ctx1, bytes, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
            if (bytes > 0 && hostBuffer[bytes - 1] == '\0') {
                std::cout << "Received message: \"" << std::string(hostBuffer.data()) << "\"" << std::endl;
            } else {
                std::cout << "Received " << bytes << " bytes of binary data" << std::endl;
            }
        } else {
            std::cout << "Failed to copy GPU data to host: " << cudaGetErrorString(err) << std::endl;
        }

        std::cout << "GPU buffer processing complete" << std::endl;
    } else if (status == easyrdma_Error_Success && bytes == 0) {
        std::cout << "Received empty transfer (0 bytes)" << std::endl;
    } else {
        std::cout << "GPU transfer failed with status: " << status << std::endl;
    }

    std::cout << "=== End Transfer Completion ===" << std::endl;
    transfer_completed = true;
}

int main(int argc, char *argv[])
{
    // Install signal handler for better debugging
    signal(SIGSEGV, segfault_handler);
    signal(SIGABRT, segfault_handler);

    // Capture main thread ID for debugging
    main_thread_id = pthread_self();
    std::cout << "GPUDirect RDMA Server (Main Thread ID: " << main_thread_id << ")" << std::endl;

    // Default settings
    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.1"; // FIX: Use working interface
    uint16_t port = (argc > 2) ? std::stoi(argv[2]) : 12345;

    std::cout << "Server: " << server_ip << ":" << port << std::endl;
    int32_t results = 0;

    // Step 3: CUDA device check
    int32_t cuda_device_count = 0;
    results = easyrdma_GetCUDADeviceCount(&cuda_device_count);
    if (results != easyrdma_Error_Success || cuda_device_count == 0)
    {
        std::cout << "No CUDA devices found: " << results << std::endl;
        return 1;
    }
    std::cout << "Found " << cuda_device_count << " CUDA device(s)" << std::endl;

    // Step 4: GPU memory allocation with proper alignment
    void *gpuBuffer = nullptr;
    size_t gpuBufferSize = 4096; // 4KB buffer, page-aligned for better GPU Direct compatibility

    results = easyrdma_AllocateGpuMemory(&gpuBuffer, gpuBufferSize);
    if (results != easyrdma_Error_Success || !gpuBuffer)
    {
        std::cout << "Failed to allocate GPU memory: " << results << std::endl;
        return 1;
    }
    std::cout << "GPU Memory allocated: " << gpuBufferSize << " bytes at " << gpuBuffer << std::endl;
    
    // Test GPU memory accessibility
    std::cout << "Testing GPU memory accessibility..." << std::endl;
    cudaError_t cuda_err = cudaMemset(gpuBuffer, 0x42, gpuBufferSize);
    if (cuda_err != cudaSuccess) {
        std::cout << "WARNING: GPU memory not accessible via CUDA: " << cudaGetErrorString(cuda_err) << std::endl;
    } else {
        std::cout << "GPU memory accessible via CUDA - OK" << std::endl;
    }

    // Step 1: Create listener session
    easyrdma_Session server_session;
    int32_t result = easyrdma_CreateListenerSession(server_ip, port, &server_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to create listener: " << result << std::endl;
        return 1;
    }
    std::cout << "Created listener session on " << server_ip << ":" << port << std::endl;

    // Step 2: Wait for client connection
    std::cout << "Waiting for client connection (timeout: 30 seconds)..." << std::endl;
    easyrdma_Session connected_session;
    result = easyrdma_Accept(server_session, easyrdma_Direction_Receive, 30000, &connected_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to accept connection: " << result << std::endl;
        if (result == -734001)
        {
            std::cout << " This is a timeout error. Make sure client is connecting to the same IP:port" << std::endl;
            std::cout << " Server is listening on: " << server_ip << ":" << port << std::endl;
        }
        easyrdma_CloseSession(server_session);
        return 1;
    }
    std::cout << "Client connected!" << std::endl;

    std::cout << "Configuring GPU buffer for RDMA..." << std::endl;
    result = easyrdma_ConfigureExternalBuffer(connected_session, gpuBuffer, gpuBufferSize, 1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to configure GPU buffer: " << result << std::endl;
        std::cout << "This indicates GPU Direct RDMA is not working properly" << std::endl;
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }
    std::cout << "Successfully configured session to use GPU memory buffer" << std::endl;
    std::cout << "GPU Direct RDMA setup appears to be working..." << std::endl;

    // Setup callback for GPU transfer completion
    easyrdma_BufferCompletionCallbackData callback;
    callback.callbackFunction = MyCompletionCallback;
    callback.context1 = gpuBuffer;      // Pass GPU buffer pointer as context
    callback.context2 = &gpuBufferSize; // Pass buffer size as context

    result = easyrdma_QueueExternalBufferRegion(connected_session, gpuBuffer, gpuBufferSize, &callback, 10000);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to queue GPU buffer region: " << result << std::endl;
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }

    std::cout << "GPU buffer queued successfully" << std::endl;
    std::cout << "Ready to receive data into GPU memory..." << std::endl;

    // Wait for data to arrive
    std::cout << "Waiting for RDMA data to arrive..." << std::endl;
    std::cout << "Server is now listening for incoming RDMA transfers..." << std::endl;
    
    // Wait for actual completion or timeout
    auto start_time = std::chrono::steady_clock::now();
    const auto timeout_duration = std::chrono::seconds(10);
    
    while (!transfer_completed && 
           (std::chrono::steady_clock::now() - start_time) < timeout_duration) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (transfer_completed) {
        std::cout << "Transfer completed - proceeding with cleanup..." << std::endl;
    } else {
        std::cout << "Timeout reached - proceeding with cleanup..." << std::endl;
    }
    
    // Safe cleanup sequence to avoid thread deadlocks
    std::cout << "Cleaning up sessions..." << std::endl;
    
    // Allow MLX5 hardware to complete any pending operations
    std::cout << "Waiting for MLX5 hardware completion..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Close sessions with deferred cleanup to prevent thread joining deadlock
    std::cout << "Closing RDMA sessions..." << std::endl;
    easyrdma_CloseSession(connected_session, 1); // Use deferred cleanup
    easyrdma_CloseSession(server_session, 1);    // Use deferred cleanup
    
    // Final wait for cleanup threads to complete
    std::cout << "Final wait for cleanup completion..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    easyrdma_FreeGpuMemory(gpuBuffer);

    std::cout << "Server completed successfully." << std::endl;

    return 0;
}
