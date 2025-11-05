#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <cuda_runtime.h>
#include <signal.h>

// Global completion flag for synchronization
std::atomic<bool> transfer_completed{false};
std::atomic<bool> server_running{true};

// Safer signal handler
void signal_handler(int sig)
{
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    server_running = false;
    transfer_completed = true;
}

// Simplified, thread-safe callback
void GPUCompletionCallback(void *ctx1, void *ctx2, int32_t status, size_t bytes)
{
    std::cout << "\n=== GPU Completion: status=" << status << ", bytes=" << bytes << " ===" << std::endl;

    if (status != easyrdma_Error_Success) {
        std::cout << "Transfer failed with status: " << status << std::endl;
        transfer_completed = true;
        return;
    }

    if (bytes == 0) {
        std::cout << "Empty transfer (likely credit message)" << std::endl;
        return; // Don't set completion flag for credit messages
    }

    if (!ctx1) {
        std::cout << "Error: GPU buffer pointer is null!" << std::endl;
        transfer_completed = true;
        return;
    }

    std::cout << "Successfully received " << bytes << " bytes into GPU memory!" << std::endl;

    // Copy GPU data to host for verification (safely)
    try {
        std::vector<char> hostBuffer(bytes);
        cudaError_t err = cudaMemcpy(hostBuffer.data(), ctx1, bytes, cudaMemcpyDeviceToHost);
        
        if (err == cudaSuccess) {
            // Safely handle the received data
            hostBuffer[bytes-1] = '\0'; // Ensure null termination
            std::cout << "GPU->Host copy successful" << std::endl;
            std::cout << "Received message: \"" << std::string(hostBuffer.data()) << "\"" << std::endl;
        } else {
            std::cout << "GPU->Host copy failed: " << cudaGetErrorString(err) << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error processing GPU data: " << e.what() << std::endl;
    }

    transfer_completed = true;
}

int main(int argc, char *argv[])
{
    // Install safer signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "=== Safe GPUDirect RDMA Server ===" << std::endl;

    // Server settings
    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.1";
    uint16_t port = (argc > 2) ? std::stoi(argv[2]) : 12345;

    std::cout << "Server: " << server_ip << ":" << port << std::endl;

    // Check CUDA devices
    int32_t cuda_device_count = 0;
    int32_t result = easyrdma_GetCUDADeviceCount(&cuda_device_count);
    if (result != easyrdma_Error_Success || cuda_device_count == 0) {
        std::cout << "No CUDA devices found: " << result << std::endl;
        return 1;
    }
    std::cout << "Found " << cuda_device_count << " CUDA device(s)" << std::endl;

    // Allocate GPU memory
    void *gpuBuffer = nullptr;
    size_t gpuBufferSize = 4096;
    result = easyrdma_AllocateGpuMemory(&gpuBuffer, gpuBufferSize);
    if (result != easyrdma_Error_Success || !gpuBuffer) {
        std::cout << "Failed to allocate GPU memory: " << result << std::endl;
        return 1;
    }
    std::cout << "GPU memory allocated: " << gpuBufferSize << " bytes" << std::endl;

    // Test GPU memory
    cudaError_t cuda_err = cudaMemset(gpuBuffer, 0x0, gpuBufferSize);
    if (cuda_err != cudaSuccess) {
        std::cout << "GPU memory test failed: " << cudaGetErrorString(cuda_err) << std::endl;
        easyrdma_FreeGpuMemory(gpuBuffer);
        return 1;
    }
    std::cout << "GPU memory test passed" << std::endl;

    // Create listener session
    easyrdma_Session server_session;
    result = easyrdma_CreateListenerSession(server_ip, port, &server_session);
    if (result != easyrdma_Error_Success) {
        std::cout << "Failed to create listener: " << result << std::endl;
        easyrdma_FreeGpuMemory(gpuBuffer);
        return 1;
    }
    std::cout << "Listener session created" << std::endl;

    // Accept client connection
    std::cout << "Waiting for client connection..." << std::endl;
    easyrdma_Session connected_session;
    result = easyrdma_Accept(server_session, easyrdma_Direction_Receive, 30000, &connected_session);
    if (result != easyrdma_Error_Success) {
        std::cout << "Failed to accept connection: " << result << std::endl;
        easyrdma_CloseSession(server_session);
        easyrdma_FreeGpuMemory(gpuBuffer);
        return 1;
    }
    std::cout << "Client connected!" << std::endl;

    // Configure GPU buffer for RDMA
    std::cout << "Configuring GPU buffer for RDMA..." << std::endl;
    result = easyrdma_ConfigureExternalBuffer(connected_session, gpuBuffer, gpuBufferSize, 1);
    if (result != easyrdma_Error_Success) {
        std::cout << "Failed to configure GPU buffer: " << result << std::endl;
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        easyrdma_FreeGpuMemory(gpuBuffer);
        return 1;
    }
    std::cout << "GPU buffer configured successfully" << std::endl;

    // Queue GPU buffer for receiving
    easyrdma_BufferCompletionCallbackData callback;
    callback.callbackFunction = GPUCompletionCallback;
    callback.context1 = gpuBuffer;
    callback.context2 = &gpuBufferSize;

    std::cout << "Queuing GPU buffer for receive..." << std::endl;
    result = easyrdma_QueueExternalBufferRegion(connected_session, gpuBuffer, gpuBufferSize, &callback, 10000);
    if (result != easyrdma_Error_Success) {
        std::cout << "Failed to queue GPU buffer: " << result << std::endl;
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        easyrdma_FreeGpuMemory(gpuBuffer);
        return 1;
    }
    std::cout << "GPU buffer queued - ready to receive data!" << std::endl;

    // Wait for completion or shutdown signal
    std::cout << "Waiting for RDMA data..." << std::endl;
    int wait_count = 0;
    while (server_running && !transfer_completed && wait_count < 100) { // 10 second timeout
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    if (transfer_completed) {
        std::cout << "Data transfer completed!" << std::endl;
    } else if (!server_running) {
        std::cout << "Server shutdown requested" << std::endl;
    } else {
        std::cout << "Timeout waiting for data" << std::endl;
    }

    // Graceful cleanup
    std::cout << "Shutting down..." << std::endl;
    
    // Close sessions first (this stops background threads)
    easyrdma_CloseSession(connected_session);
    easyrdma_CloseSession(server_session);
    
    // Small delay for cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Free GPU memory last
    easyrdma_FreeGpuMemory(gpuBuffer);

    std::cout << "Server shutdown complete." << std::endl;
    return 0;
}