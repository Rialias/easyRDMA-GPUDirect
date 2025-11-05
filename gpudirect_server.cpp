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
    // NOTE: This callback runs in the RDMA worker thread context
    std::cout << "\n=== GPU Transfer Completion Callback TRIGGERED ===" << std::endl;
    std::cout << "Status: " << status << ", Bytes: " << bytes << std::endl;
    std::cout << "Thread ID: " << pthread_self() << " (Main: " << main_thread_id << ")" << std::endl;

    if (status == -734011) {
        std::cout << "Transfer was cancelled (session likely closed prematurely)" << std::endl;
        return;
    }

    if (status != easyrdma_Error_Success) {
        std::cout << "Transfer failed with error status: " << status << std::endl;
        return;
    }

    if (!ctx1) {
        std::cout << "Error: GPU buffer pointer is null!" << std::endl;
        return;
    }

    if (status == easyrdma_Error_Success && bytes > 0) {
        std::cout << "Successfully received " << bytes << " bytes into GPU memory!" << std::endl;

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cout << "GPU sync error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        char *verifyBuffer = new char[bytes + 1];  // Allocate exactly what we need + null terminator
        memset(verifyBuffer, 0, bytes + 1);
        cudaError_t cudaErr = cudaMemcpy(verifyBuffer, ctx1, bytes, cudaMemcpyDeviceToHost);
        if (cudaErr == cudaSuccess) {
            verifyBuffer[bytes] = '\0';  // Ensure null termination
            std::cout << "Received message: \"" << std::string(verifyBuffer) << "\"" << std::endl;
        } else {
            std::cout << "Failed to copy GPU data to host: " << cudaGetErrorString(cudaErr) << std::endl;  // Fixed: use cudaErr, not err
        }
        delete[] verifyBuffer;  // Don't forget to free the allocated memory

        std::cout << "GPU buffer processing complete" << std::endl;
    } else if (status == easyrdma_Error_Success && bytes == 0) {
        std::cout << "Received empty transfer (0 bytes)" << std::endl;
    } else {
        std::cout << "GPU transfer failed with status: " << status << std::endl;
    }

    std::cout << "=== End Transfer Completion ===" << std::endl;
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
    size_t gpuBufferSize = 2048; // 2KB buffer, page-aligned for better GPU Direct compatibility

    results = easyrdma_AllocateGpuMemory(&gpuBuffer, gpuBufferSize);
    if (results != easyrdma_Error_Success || !gpuBuffer)
    {
        std::cout << "Failed to allocate GPU memory: " << results << std::endl;
        return 1;
    }
    std::cout << "GPU Memory allocated: " << gpuBufferSize << " bytes at " << gpuBuffer << std::endl;
    
    // Test GPU memory accessibility and GPUDirect capability
    std::cout << "Testing GPU memory accessibility..." << std::endl;
    cudaError_t cuda_err = cudaMemset(gpuBuffer, 0x42, gpuBufferSize);
    if (cuda_err != cudaSuccess) {
        std::cout << "WARNING: GPU memory not accessible via CUDA: " << cudaGetErrorString(cuda_err) << std::endl;
    } else {
        std::cout << "GPU memory accessible via CUDA - OK" << std::endl;
    }

    // Check GPU Direct RDMA capabilities
    std::cout << "\n=== GPUDirect RDMA Diagnostics ===" << std::endl;
    
    // Get CUDA device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "GPU Device: " << prop.name << std::endl;
    std::cout << "GPU Memory Type: " << (prop.canMapHostMemory ? "Mappable" : "Non-mappable") << std::endl;
    std::cout << "GPU PCIe: Bus " << prop.pciBusID << ", Device " << prop.pciDeviceID << std::endl;
    
    // Check if this is device memory (should be for GPUDirect)
    cudaPointerAttributes attributes;
    cudaError_t ptr_err = cudaPointerGetAttributes(&attributes, gpuBuffer);
    if (ptr_err == cudaSuccess) {
        std::cout << "Memory Type: " << (attributes.type == cudaMemoryTypeDevice ? "GPU Device Memory" : 
                     attributes.type == cudaMemoryTypeHost ? "Host Memory" : "Unknown") << std::endl;
        std::cout << "Device Pointer: " << attributes.devicePointer << std::endl;
        std::cout << "Host Pointer: " << attributes.hostPointer << std::endl;
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

    std::cout << "\n=== Configuring GPU Buffer for RDMA ===" << std::endl;
    std::cout << "GPU Buffer: " << gpuBuffer << ", Size: " << gpuBufferSize << std::endl;
    std::cout << "Calling easyrdma_ConfigureExternalBuffer..." << std::endl;
    
    result = easyrdma_ConfigureExternalBuffer(connected_session, gpuBuffer, gpuBufferSize, 1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to configure GPU buffer: " << result << std::endl;
        
        // Get more detailed error information
        char error_buffer[512];
        easyrdma_GetLastErrorString(error_buffer, sizeof(error_buffer));
        std::cout << "Detailed error: " << error_buffer << std::endl;
        
        // Try to understand what went wrong
        if (result == -734003) {
            std::cout << "ERROR: Invalid argument - GPU buffer may not be compatible with RDMA" << std::endl;
        } else if (result == -734006) {
            std::cout << "ERROR: Operating system error - Check CUDA/RDMA drivers" << std::endl;
        } else if (result == -734008) {
            std::cout << "ERROR: Out of memory during GPU buffer registration" << std::endl;
        }
        
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }
    
    std::cout << "✓ Successfully configured session to use GPU memory buffer" << std::endl;
    std::cout << "✓ GPU Direct RDMA setup completed successfully" << std::endl;
    
    // Verify the buffer is properly registered
    std::cout << "Verifying GPU buffer registration..." << std::endl;
    
    // The library should have logged "GPU Direct RDMA memory" registration
    // Let's add a small test to ensure the buffer is accessible
    cudaMemset(gpuBuffer, 0x55, 64); // Set first 64 bytes to 0x55
    cudaDeviceSynchronize();
    
    std::vector<char> verify_buffer(64);
    cudaMemcpy(verify_buffer.data(), gpuBuffer, 64, cudaMemcpyDeviceToHost);
    bool verification_ok = true;
    for (int i = 0; i < 64; i++) {
        if ((unsigned char)verify_buffer[i] != 0x55) {
            verification_ok = false;
            break;
        }
    }
    
    std::cout << "GPU buffer accessibility: " << (verification_ok ? "✓ OK" : "✗ FAILED") << std::endl;

    // Setup callback for GPU transfer completion
    std::cout << "\n=== Setting Up GPU Buffer Queue ===" << std::endl;

    easyrdma_BufferCompletionCallbackData callback;
    callback.callbackFunction = &MyCompletionCallback;
    callback.context1 = gpuBuffer;      // Pass GPU buffer pointer as context
    callback.context2 = &gpuBufferSize; // Pass buffer size as context// Pass buffer size as context

    std::cout << "Queuing GPU buffer for receive operations..." << std::endl;
    std::cout << "Buffer: " << gpuBuffer << ", Size: " << gpuBufferSize << ", Timeout: 10000ms" << std::endl;
    
    result = easyrdma_QueueExternalBufferRegion(connected_session, gpuBuffer, gpuBufferSize, &callback, 10000);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to queue GPU buffer region: " << result << std::endl;
        
        // Get detailed error for queue operation
        char error_buffer[512];
        easyrdma_GetLastErrorString(error_buffer, sizeof(error_buffer));
        std::cout << "Queue error details: " << error_buffer << std::endl;
        
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }

    std::cout << "✓ GPU buffer queued successfully for receive operations" << std::endl;
    std::cout << "✓ Ready to receive data into GPU memory via GPUDirect RDMA" << std::endl;
    
    
    // Step 3: Wait for client to send data to GPU buffer
    std::cout << "Waiting for client to send data to GPU buffer..." << std::endl;
    std::cout << "Server is now ready to receive RDMA data into GPU memory" << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(5));
    // Step 4: Close sessions (this is where the crash typically occurs)
    std::cout << "Closing RDMA sessions..." << std::endl;
    std::cout << "*** CRITICAL SECTION: Closing connected_session ***" << std::endl;
    
    // Set signal handlers to catch crashes during cleanup
    signal(SIGSEGV, segfault_handler);
    signal(SIGABRT, segfault_handler);
    
    easyrdma_CloseSession(connected_session);
    std::cout << "Connected session closed successfully" << std::endl;
    
    std::cout << "*** CRITICAL SECTION: Closing server_session ***" << std::endl;
    easyrdma_CloseSession(server_session);
    std::cout << "Server session closed successfully" << std::endl;

    easyrdma_FreeGpuMemory(gpuBuffer);

    std::cout << "Server completed successfully." << std::endl;


    return 0;
}
