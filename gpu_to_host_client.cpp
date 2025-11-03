#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

// Completion callback
void send_completion_callback(void* context1, void* context2, int32_t status, size_t bytes) {
    if (status == easyrdma_Error_Success) {
        std::cout << "GPU data sent successfully!" << std::endl;
    } else {
        std::cout << "Send failed: " << status << std::endl;
    }
}

int main(int argc, char *argv[])
{
    std::cout << "GPUDirect RDMA GPU TO HOST Client" << std::endl;

    std::cout << "=== GPUDirect RDMA Client (Host-to-GPU Sender) ===" << std::endl;

    // Connection settings
    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.6";
    uint16_t server_port = (argc > 2) ? std::stoi(argv[2]) : 12345;
    const char *local_ip = (argc > 3) ? argv[3] : "192.168.30.1";

    std::cout << "Connecting to server: " << server_ip << ":" << server_port << std::endl;
    std::cout << "Using local interface: " << local_ip << std::endl;

    // Step 3: CUDA device check (your code was correct)
    int32_t cuda_device_count = 0;
    int32_t results = easyrdma_GetCUDADeviceCount(&cuda_device_count);
    if (results != easyrdma_Error_Success || cuda_device_count == 0)
    {
        std::cout << "No CUDA devices found: " << results << std::endl;
        return 1;
    }
    std::cout << "Found " << cuda_device_count << " CUDA device(s)" << std::endl;

    // Step 4: GPU memory allocation (your code was correct)
    void *gpuBuffer = nullptr;
    size_t gpuBufferSize = 1024; // 1KB buffer should work with GPU Direct

    results = easyrdma_AllocateGpuMemory(&gpuBuffer, gpuBufferSize);
    if (results != easyrdma_Error_Success || !gpuBuffer)
    {
        std::cout << "Failed to allocate GPU memory: " << results << std::endl;
        return 1;
    }
    std::cout << "GPU Memory allocated: " << gpuBufferSize << " bytes at " << gpuBuffer << std::endl;

    const char* message = "Hello from GPU via RDMA!";
    cudaMemcpy(gpuBuffer, message, strlen(message) + 1, cudaMemcpyHostToDevice);

    // Create connector session
    easyrdma_Session client_session;
    int32_t result = easyrdma_CreateConnectorSession(local_ip, 0, &client_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to create connector session: " << result << std::endl;
        return 1;
    }
    std::cout << "Connector session created" << std::endl;

    // Connect to server
    std::cout << "Connecting to server..." << std::endl;
    result = easyrdma_Connect(client_session, easyrdma_Direction_Send, server_ip, server_port, 10000);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to connect to server: " << result << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << "Connected to server!" << std::endl;

    result = easyrdma_ConfigureExternalBuffer(client_session, gpuBuffer, gpuBufferSize, 1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to configure GPU buffer: " << result << std::endl;
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << "Configured session to use GPU memory buffer" << std::endl;

    // 5. Send data using callback approach
    easyrdma_BufferCompletionCallbackData callbackData;
    callbackData.callbackFunction = send_completion_callback;
    callbackData.context1 = gpuBuffer;
    result = easyrdma_QueueExternalBufferRegion(client_session, gpuBuffer, gpuBufferSize, &callbackData, 10000);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to queue GPU buffer region: " << result << std::endl;
        easyrdma_FreeGpuMemory(gpuBuffer);
        easyrdma_CloseSession(client_session);
        return 1;
    }

    std::cout << "Ready to send data from GPU memory..." << std::endl;
    std::cout << "Waiting for send completion..." << std::endl;

    // Wait a bit for the completion callback to be called
    // In a real application, you'd want to use proper synchronization
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "Cleaning up..." << std::endl;
    easyrdma_CloseSession(client_session);
    easyrdma_FreeGpuMemory(gpuBuffer);
    return 0;
}

