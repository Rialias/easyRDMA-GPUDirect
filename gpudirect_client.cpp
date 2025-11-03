// Simple GPUDirect Client to test the server
#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cuda_runtime.h>

int main(int argc, char *argv[])
{
    std::cout << "=== GPUDirect RDMA Client (Host-to-GPU Sender) ===" << std::endl;

    // Connection settings
    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.1";
    uint16_t server_port = (argc > 2) ? std::stoi(argv[2]) : 12345;
    const char *local_ip = (argc > 3) ? argv[3] : "192.168.30.1";

    std::cout << "Connecting to server: " << server_ip << ":" << server_port << std::endl;
    std::cout << "Using local interface: " << local_ip << std::endl;

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
    result = easyrdma_Connect(client_session, easyrdma_Direction_Send, server_ip, server_port, 100000);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to connect to server: " << result << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << "Connected to server!" << std::endl;

    // Step 3: Configure host memory buffers for sending
    size_t buffer_size = 1024; // 1KB buffers
    size_t num_buffers = 1;    // Just 1 buffer
    result = easyrdma_ConfigureBuffers(client_session, buffer_size, num_buffers);
    if (result != easyrdma_Error_Success)
    {
        std::cout << " Failed to configure buffers: " << result << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << " Configured buffers (" << buffer_size << " bytes x " << num_buffers << ")" << std::endl;

    // Step 4: Send messages from host memory
    std::cout << "\n Sending messages from host..." << std::endl;

    const char *messages[] = {"Hello from host client!"};

    easyrdma_InternalBufferRegion send_region;
    result = easyrdma_AcquireSendRegion(client_session, 10000, &send_region); // 10 second timeout
    if (result != easyrdma_Error_Success)
    {
        std::cout << " Failed to acquire send region: " << result << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }

    // Copy message into send buffer
    std::memcpy(send_region.buffer, messages[0], std::strlen(messages[0]) + 1);
    send_region.usedSize = std::strlen(messages[0]) + 1;

    // Queue the buffer for sending
    result = easyrdma_QueueBufferRegion(client_session, &send_region, nullptr);
    if (result != easyrdma_Error_Success)
    {
        std::cout << " Failed to queue send buffer: " << result << std::endl;
        easyrdma_ReleaseUserBufferRegionToIdle(client_session, &send_region);
        easyrdma_CloseSession(client_session);
        return 1;
    }

    std::cout << " Sent message from host: " << messages[0] << std::endl;

    std::cout << " Messages sent. Waiting for server to process..." << std::endl;

    easyrdma_ReleaseUserBufferRegionToIdle(client_session, &send_region);

    // Wait longer to ensure server has time to receive data into GPU memory 
    // Host-to-GPU transfers may take longer to complete
    std::this_thread::sleep_for(std::chrono::seconds(10)); // Wait for server to process
    // Cleanup
    std::cout << "\n Shutting down..." << std::endl;
    easyrdma_CloseSession(client_session);

    std::cout << "Client finished cleanly" << std::endl;
    return 0;
}
