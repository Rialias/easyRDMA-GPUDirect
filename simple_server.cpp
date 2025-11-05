// Simple easyRDMA Server Example
// Demonstrates basic server-side RDMA functionality
// Copyright (c) 2022 National Instruments
// SPDX-License-Identifier: MIT

#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>

int main(int argc, char *argv[])
{
    std::cout << "Simple easyRDMA Server" << std::endl;
    std::cout << "======================" << std::endl;

    // Default settings
    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.6";
    uint16_t port = (argc > 2) ? std::stoi(argv[2]) : 12345;

    std::cout << "Server: " << server_ip << ":" << port << std::endl;

    size_t num_interfaces = 0;
    int32_t results = easyrdma_Enumerate(nullptr, &num_interfaces, 4); 
    if (results != easyrdma_Error_Success)
    {
        std::cout << "Failed to enumerate interfaces: " << results << std::endl;
        return 1;
    }

    if (num_interfaces == 0)
    {
        std::cout << "No RDMA interfaces found!" << std::endl;
        return 1;
    }

    std::cout << "✓ Found " << num_interfaces << " RDMA interface(s)" << std::endl;

    // Second call: Get the actual interface addresses
    std::vector<easyrdma_AddressString> interfaces(num_interfaces);
    results = easyrdma_Enumerate(interfaces.data(), &num_interfaces, 4);
    if (results != easyrdma_Error_Success)
    {
        std::cout << "Failed to get interface addresses: " << results << std::endl;
        return 1;
    }

    // Display available interfaces
    std::cout << "Available RDMA interfaces:" << std::endl;
    for (size_t i = 0; i < num_interfaces; i++)
    {
        std::cout << "  [" << i << "] " << interfaces[i].addressString << std::endl;
    }
    // Step 1: Create listener session
    easyrdma_Session listen_session;
    int32_t result = easyrdma_CreateListenerSession(server_ip, port, &listen_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to create listener: " << result << std::endl;
        return 1;
    }
    std::cout << " Created listener session on " << server_ip << ":" << port << std::endl;

    // Step 2: Wait for client connection
    std::cout << "Waiting for client connection (timeout: 30 seconds)..." << std::endl;
    easyrdma_Session connected_session;
    result = easyrdma_Accept(listen_session, easyrdma_Direction_Receive, 30000, &connected_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to accept connection: " << result << std::endl;
        if (result == -734001)
        {
            std::cout << "   This is a timeout error. Make sure client is connecting to the same IP:port" << std::endl;
            std::cout << "   Server is listening on: " << server_ip << ":" << port << std::endl;
        }
        easyrdma_CloseSession(listen_session);
        return 1;
    }
    std::cout << " Client connected!" << std::endl;

    // Step 3: Configure buffers
    size_t buffer_size = 1024; // 1KB buffers
    size_t num_buffers = 5;    // Just 5 buffers
    result = easyrdma_ConfigureBuffers(connected_session, buffer_size, num_buffers);
    if (result != easyrdma_Error_Success)
    {
        std::cout << " Failed to configure buffers: " << result << std::endl;
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(listen_session);
        return 1;
    }
    std::cout << "Configured buffers (" << buffer_size << " bytes x " << num_buffers << ")" << std::endl;

    // Step 4: Simple receive loop
    std::cout << "\n Ready to receive messages! (Ctrl+C to exit)" << std::endl;

    for (int msg_count = 0; msg_count < 10; msg_count++)
    { // Receive up to 10 messages
        // Get a buffer to receive data
        easyrdma_InternalBufferRegion receive_region;
        result = easyrdma_AcquireReceivedRegion(connected_session, 5000, &receive_region);
        if (result != easyrdma_Error_Success)
        {
            if (result == easyrdma_Error_Timeout)
            {
                std::cout << " No data received (timeout)" << std::endl;
                break;
            }
            std::cout << " Failed to get receive buffer: " << result << std::endl;
            break;
        }

        // Display received message
        std::string message(static_cast<char *>(receive_region.buffer), receive_region.usedSize);
        std::cout << "Message " << (msg_count + 1) << ": \"" << message << "\" (" << receive_region.usedSize << " bytes)" << std::endl;

        //Send echo response
        easyrdma_InternalBufferRegion send_region;
        result = easyrdma_AcquireSendRegion(connected_session, 1000, &send_region);
        if (result == easyrdma_Error_Success)
        {
            std::string echo = "ECHO: " + message;
            size_t echo_len = std::min(echo.length(), send_region.bufferSize);
            memcpy(send_region.buffer, echo.c_str(), echo_len);
            send_region.usedSize = echo_len;

            // Simple synchronous send (blocks until complete)
            result = easyrdma_QueueBufferRegion(connected_session, &send_region, nullptr);
            if (result == easyrdma_Error_Success)
            {
                std::cout << " Sent echo: \"" << echo << "\"" << std::endl;
            }
            else
            {
                std::cout << " Failed to send echo: " << result << std::endl;
            }
        } 

        // Release the receive buffer
        easyrdma_ReleaseReceivedBufferRegion(connected_session, &receive_region);
    }

    // Cleanup
    std::cout << "\n✓ Shutting down..." << std::endl;
    easyrdma_CloseSession(connected_session);
    easyrdma_CloseSession(listen_session);

    std::cout << "✓ Server finished cleanly" << std::endl;
    return 0;
}