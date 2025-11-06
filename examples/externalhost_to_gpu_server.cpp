#include "easyrdma.h"
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


void MyCompletionCallback(void *ctx1, void *ctx2, int32_t status, size_t bytes)
{
    std::cout << "\n=== GPU Transfer Completion Callback ===" << std::endl;
    std::cout << "Status: " << status << ", Bytes: " << bytes << std::endl;

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

        char *verifyBuffer = new char[bytes + 1]; 
        memset(verifyBuffer, 0, bytes + 1);
        cudaError_t cudaErr = cudaMemcpy(verifyBuffer, ctx1, bytes, cudaMemcpyDeviceToHost);
        if (cudaErr == cudaSuccess) {
            verifyBuffer[bytes] = '\0';  // Ensure null termination
            std::cout << "Received message: \"" << std::string(verifyBuffer) << "\"" << std::endl;
        } else {
            std::cout << "Failed to copy GPU data to host: " << cudaGetErrorString(cudaErr) << std::endl; 
        }
        delete[] verifyBuffer;

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
    std::cout << "=== GPUDirect RDMA Server (Host-to-GPU Receiver) ===" << std::endl;

    // Default settings
    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.1"; // FIX: Use working interface
    uint16_t port = (argc > 2) ? std::stoi(argv[2]) : 12345;

    std::cout << "Server: " << server_ip << ":" << port << std::endl;
    int32_t results = 0;

    // Step 1: Interface enumeration
    size_t num_interfaces = 0;
    results = easyrdma_Enumerate(nullptr, &num_interfaces, easyrdma_AddressFamily_AF_INET);
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
    }

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
    size_t gpuBufferSize = 2048; // 2KB buffer

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
    
    // Check if this is device memory (should be for GPUDirect)
    cudaPointerAttributes attributes;
    cudaError_t ptr_err = cudaPointerGetAttributes(&attributes, gpuBuffer);
    if (ptr_err == cudaSuccess) {
        std::cout << "Memory Type: " << (attributes.type == cudaMemoryTypeDevice ? "GPU Device Memory" : 
                     attributes.type == cudaMemoryTypeHost ? "Host Memory" : "Unknown") << std::endl;
        std::cout << "Device Pointer: " << attributes.devicePointer << std::endl;
        std::cout << "Host Pointer: " << attributes.hostPointer << std::endl;
    }
    

    // Step 5: Create listener session
    easyrdma_Session server_session;
    int32_t result = easyrdma_CreateListenerSession(server_ip, port, &server_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to create listener: " << result << std::endl;
        return 1;
    }
    std::cout << "Created listener session on " << server_ip << ":" << port << std::endl;

    // Step 6: Wait for client connection
    std::cout << "Waiting for client connection (timeout: 30 seconds)..." << std::endl;
    easyrdma_Session connected_session;
    result = easyrdma_Accept(server_session, easyrdma_Direction_Receive, 30000, &connected_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to accept connection: " << result << std::endl;
        easyrdma_CloseSession(server_session);
        return 1;
    }
    std::cout << "Client connected!" << std::endl;

    std::cout << "\n=== Configuring GPU Buffer for RDMA ===" << std::endl;

    // Step 7: Configure external buffer
    result = easyrdma_ConfigureExternalBuffer(connected_session, gpuBuffer, gpuBufferSize, 1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to configure GPU buffer: " << result << std::endl;       
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }
    
    std::cout << "✓ Successfully configured session to use GPU memory buffer" << std::endl;

    // Setup callback for GPU transfer completion
    easyrdma_BufferCompletionCallbackData callback;
    callback.callbackFunction = &MyCompletionCallback;
    callback.context1 = gpuBuffer;      // Pass GPU buffer pointer as context
    callback.context2 = &gpuBufferSize; // Pass buffer size as context

    // Step 8: Queue GPU buffer for receive
    std::cout << "Queuing GPU buffer for receive operations..." << std::endl;
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
    
    std::this_thread::sleep_for(std::chrono::seconds(5));
    // Step 9: Close sessions
    std::cout << "Closing RDMA sessions..." << std::endl;

    easyrdma_CloseSession(connected_session);
    easyrdma_CloseSession(server_session);
    easyrdma_FreeGpuMemory(gpuBuffer);

    std::cout << "Server completed successfully." << std::endl;

    return 0;
}
