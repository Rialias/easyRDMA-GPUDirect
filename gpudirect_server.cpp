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
    std::cout << std::endl
              << "=== GPU Transfer Completion Callback ===" << std::endl;
    std::cout << "Callback called with status: " << status << ", bytes: " << bytes << std::endl;
    std::cout << "ctx1 (GPU buffer): " << ctx1 << ", ctx2 (buffer size ptr): " << ctx2 << std::endl;

    // Check for operation cancelled error
    if (status == -734011)
    { // easyrdma_Error_OperationCancelled
        std::cout << "Transfer was cancelled (session likely closed prematurely)" << std::endl;
        std::cout << "This is normal if the client disconnected early" << std::endl;
        transfer_completed = true;
        return;
    }

    // Check for other error conditions that might indicate connection issues
    if (status != easyrdma_Error_Success)
    {
        std::cout << "Transfer failed with error status: " << status << std::endl;
        transfer_completed = true;
        return;
    }

    // Validate pointers before using them
    if (!ctx1)
    {
        std::cout << "Error: GPU buffer pointer is null!" << std::endl;
        transfer_completed = true;
        return;
    }

    if (status == easyrdma_Error_Success && bytes > 0)
    {
        std::cout << "Successfully received " << bytes << " bytes into GPU memory!" << std::endl;

        // Ensure all GPU operations are complete
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            std::cout << "GPU sync error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Copy GPU data to host to verify reception
        std::vector<char> hostBuffer(bytes);
        err = cudaMemcpy(hostBuffer.data(), ctx1, bytes, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess)
        {
            // Ensure null termination for string data
            if (bytes > 0 && hostBuffer[bytes - 1] == '\0')
            {
                std::cout << "Received message: \"" << std::string(hostBuffer.data()) << "\"" << std::endl;
            }
            else
            {
                std::cout << "Received " << bytes << " bytes of binary data" << std::endl;
            }
        }
        else
        {
            std::cout << "Failed to copy GPU data to host: " << cudaGetErrorString(err) << std::endl;
        }

        std::cout << "GPU buffer processing complete" << std::endl;
    }
    else if (status == easyrdma_Error_Success && bytes == 0)
    {
        std::cout << "Received empty transfer (0 bytes)" << std::endl;
    }
    else
    {
        std::cout << "GPU transfer failed with status: " << status << std::endl;
    }
    std::cout << "=== End Transfer Completion ===" << std::endl;

    // Signal completion (regardless of success or failure)
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
    // Step 1: Interface enumeration
    /*size_t num_interfaces = 0;
    int32_t results = easyrdma_Enumerate(nullptr, &num_interfaces, easyrdma_AddressFamily_AF_INET);
    if (results != easyrdma_Error_Success || num_interfaces == 0)
    {
        std::cout << "Failed to enumerate interfaces: " << results << std::endl;
        return 1;
    }

    std::cout << "Found " << num_interfaces << " RDMA interface(s)" << std::endl;

    std::vector<easyrdma_AddressString> interfaces(num_interfaces);
    results = easyrdma_Enumerate(interfaces.data(), &num_interfaces, easyrdma_AddressFamily_AF_INET);

    std::cout << "Available RDMA interfaces:" << std::endl;
    for (size_t i = 0; i < num_interfaces; i++)
    {
        std::cout << "  [" << i << "] " << interfaces[i].addressString << std::endl;
    } */

    // Step 3: CUDA device check
    int32_t cuda_device_count = 0;
    results = easyrdma_GetCUDADeviceCount(&cuda_device_count);
    if (results != easyrdma_Error_Success || cuda_device_count == 0)
    {
        std::cout << "No CUDA devices found: " << results << std::endl;
        return 1;
    }
    std::cout << "Found " << cuda_device_count << " CUDA device(s)" << std::endl;

    // Step 4: GPU memory allocation
    void *gpuBuffer = nullptr;
    size_t gpuBufferSize = 1024; // 1KB buffer should work with GPU Direct

    results = easyrdma_AllocateGpuMemory(&gpuBuffer, gpuBufferSize);
    if (results != easyrdma_Error_Success || !gpuBuffer)
    {
        std::cout << "Failed to allocate GPU memory: " << results << std::endl;
        return 1;
    }
    std::cout << "GPU Memory allocated: " << gpuBufferSize << " bytes at " << gpuBuffer << std::endl;

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

    result = easyrdma_ConfigureExternalBuffer(connected_session, gpuBuffer, gpuBufferSize, 1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to configure GPU buffer: " << result << std::endl;
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }
    std::cout << "Configured session to use GPU memory buffer" << std::endl;

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

    // Wait for callback to complete with proper timeout
    std::cout << "Waiting for callback to complete..." << std::endl;
    // Proper cleanup to avoid double-free corruption
    std::cout << "Cleaning up sessions..." << std::endl;
    easyrdma_CloseSession(connected_session);
    easyrdma_CloseSession(server_session);
    
    std::cout << "Server completed successfully." << std::endl;

    return 0;
}
