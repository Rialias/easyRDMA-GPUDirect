// Robust GPUDirect Client with proper credit handling
#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <algorithm>
#include <atomic>

// Completion callback to track send completion
std::atomic<bool> send_completed{false};
std::atomic<int32_t> send_status{0};

void SendCompletionCallback(void *ctx1, void *ctx2, int32_t status, size_t bytes)
{
    std::cout << "Send completion: status=" << status << ", bytes=" << bytes << std::endl;
    send_status = status;
    send_completed = true;
}

int main(int argc, char *argv[])
{
    std::cout << "=== Robust GPUDirect RDMA Client ===" << std::endl;

    // Connection settings
    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.1";
    uint16_t server_port = (argc > 2) ? std::stoi(argv[2]) : 12345;
    const char *local_ip = (argc > 3) ? argv[3] : "192.168.30.1";

    std::cout << "Connecting to server: " << server_ip << ":" << server_port << std::endl;

    // Create and connect
    easyrdma_Session client_session;
    int32_t result = easyrdma_CreateConnectorSession(local_ip, 0, &client_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to create connector session: " << result << std::endl;
        return 1;
    }

    result = easyrdma_Connect(client_session, easyrdma_Direction_Send, server_ip, server_port, 10000);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to connect: " << result << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << "Connected to server!" << std::endl;

    // Configure buffers
    size_t buffer_size = 1024;
    result = easyrdma_ConfigureBuffers(client_session, buffer_size, 1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to configure buffers: " << result << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }
    std::cout << "Configured send buffers" << std::endl;

    // Wait for server to configure and send credits
    std::cout << "Waiting for server to configure GPU buffer..." << std::endl;
    
    // Try to acquire send region with increasing timeouts
    easyrdma_InternalBufferRegion send_region;
    int attempts = 0;
    const int max_attempts = 10;
    
    while (attempts < max_attempts) {
        attempts++;
        std::cout << "Attempt " << attempts << " to acquire send region..." << std::endl;
        
        result = easyrdma_AcquireSendRegion(client_session, 2000, &send_region); // 2 sec timeout per attempt
        
        if (result == easyrdma_Error_Success) {
            std::cout << "✓ Send region acquired! Server has sent credits." << std::endl;
            break;
        } else if (result == easyrdma_Error_Timeout) {
            std::cout << "  Timeout waiting for credits (server may not be ready yet)" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } else {
            std::cout << "Failed to acquire send region: " << result << std::endl;
            easyrdma_CloseSession(client_session);
            return 1;
        }
    }
    
    if (attempts >= max_attempts) {
        std::cout << "ERROR: Server never sent credits after " << max_attempts << " attempts" << std::endl;
        std::cout << "Make sure server is running and has configured GPU buffer properly" << std::endl;
        easyrdma_CloseSession(client_session);
        return 1;
    }

    // Prepare and send message
    const char *message = "Hello GPU from robust client!";
    size_t message_len = std::strlen(message) + 1;
    
    std::memcpy(send_region.buffer, message, message_len);
    send_region.usedSize = message_len;

    // Send with completion callback
    easyrdma_BufferCompletionCallbackData callback;
    callback.callbackFunction = SendCompletionCallback;
    callback.context1 = nullptr;
    callback.context2 = nullptr;

    std::cout << "Sending message to GPU: \"" << message << "\"" << std::endl;
    result = easyrdma_QueueBufferRegion(client_session, &send_region, &callback);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to queue send buffer: " << result << std::endl;
        easyrdma_ReleaseUserBufferRegionToIdle(client_session, &send_region);
        easyrdma_CloseSession(client_session);
        return 1;
    }

    // Wait for send completion
    std::cout << "Waiting for send completion..." << std::endl;
    int wait_count = 0;
    while (!send_completed && wait_count < 50) { // 5 second timeout
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }
    
    if (send_completed) {
        if (send_status == 0) {
            std::cout << "✓ Message successfully sent to server's GPU memory!" << std::endl;
        } else {
            std::cout << "✗ Send completed with error: " << send_status << std::endl;
        }
    } else {
        std::cout << "⚠ Send completion timeout" << std::endl;
    }

    // Give server time to process
    std::cout << "Waiting for server to process GPU data..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10000));

    // Cleanup
    std::cout << "Closing connection..." << std::endl;
    easyrdma_CloseSession(client_session);
    
    std::cout << "Client completed successfully!" << std::endl;
    return 0;
}