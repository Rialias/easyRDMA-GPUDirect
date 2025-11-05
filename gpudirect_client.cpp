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

    // Allocate host memory for sender
    char *hostBuffer = new char[buffer_size];
    const char *message = "GPU Direct RDMA WRITE: Host→GPU Direct!";
    strcpy(hostBuffer, message);
    int messageLength = strlen(message) + 1;
    std::cout << "Host buffer allocated and initialized for sending" << std::endl;

    easyrdma_InternalBufferRegion send_region;
    send_region.buffer = hostBuffer;
    send_region.bufferSize = buffer_size;
    send_region.usedSize = messageLength;
    result = easyrdma_ConfigureBuffers(client_session, send_region.bufferSize, num_buffers);
    if (result != easyrdma_Error_Success)
    {
        std::cout << " Failed to configure buffers: " << result << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << " Configured buffers (" << buffer_size << " bytes x " << num_buffers << ")" << std::endl;

    // Step 4: Wait for server to configure and send credits
    std::cout << "\n Waiting for server to configure GPU buffer and send credits..." << std::endl;
    
    // Give server time to configure GPU buffer and queue receive buffer
    // This allows server to send credit notification back to client
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Step 5: Send messages from host memory
    std::cout << "\n Sending messages from host..." << std::endl;


    //easyrdma_InternalBufferRegion send_region;
    result = easyrdma_AcquireSendRegion(client_session, 30000, &send_region); // 30 second timeout to wait for credits
    if (result != easyrdma_Error_Success)
    {
        std::cout << " Failed to acquire send region (no credits available?): " << result << std::endl;
        if (result == easyrdma_Error_Timeout) {
            std::cout << " This likely means server hasn't sent credit notification yet" << std::endl;
            std::cout << " Make sure server has configured GPU buffer and queued receive buffer" << std::endl;
        }
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << " Acquired send region - server credits available!" << std::endl;

    // Copy message into send buffer
    /*size_t message_len = std::strlen(messages[0]) + 1; // Include null terminator
    std::memcpy(send_region.buffer, messages[0], message_len);
    send_region.usedSize = message_len;

    std::cout << " Prepared message (" << message_len << " bytes): \"" << messages[0] << "\"" << std::endl; */

    // Queue the buffer for sending (this will use the credits from server)
    std::cout << " Queuing send buffer (using server credits)..." << std::endl;
    result = easyrdma_QueueBufferRegion(client_session, &send_region, nullptr);
    if (result != easyrdma_Error_Success)
    {
        std::cout << " Failed to queue send buffer: " << result << std::endl;
        if (result == easyrdma_Error_SendTooLargeForRecvBuffer) {
            std::cout << " Message too large for server's GPU buffer!" << std::endl;
        }
        easyrdma_ReleaseUserBufferRegionToIdle(client_session, &send_region);
        easyrdma_CloseSession(client_session);
        return 1;
    }

    std::cout << " ✓ Successfully sent message to server's GPU memory!" << std::endl;
    std::cout << " Message: \"" << hostBuffer << "\" (" << messageLength << " bytes)" << std::endl;

    std::cout << " Messages sent. Waiting for server to process..." << std::endl;

    // Wait for server to process the GPU data
    // The RDMA transfer to GPU is nearly instantaneous, but give server time to process
    std::this_thread::sleep_for(std::chrono::seconds(10)); // Wait for server to process
    // Cleanup
    std::cout << "\n Shutting down..." << std::endl;
    easyrdma_CloseSession(client_session);

    std::cout << "Client finished cleanly" << std::endl;
    return 0;
}
