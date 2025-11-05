// Simple easyRDMA Client Example
// Demonstrates basic client-side RDMA functionality
// Copyright (c) 2022 National Instruments
// SPDX-License-Identifier: MIT

#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>

int main(int argc, char* argv[]) {
    std::cout << "Simple easyRDMA Client" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Default settings
    const char* server_ip = "192.168.30.6";
    uint16_t port = 12345;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--server" && i + 1 < argc) {
            server_ip = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options] [server_ip] [port]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --server <ip>    Server IP address" << std::endl;
            std::cout << "  --port <port>    Server port" << std::endl;
            std::cout << "  --help          Show this help" << std::endl;
            std::cout << "Positional:" << std::endl;
            std::cout << "  server_ip       Server IP (default: 192.168.30.1)" << std::endl;
            std::cout << "  port           Server port (default: 12345)" << std::endl;
            return 0;
        } else if (arg[0] != '-') {
            // Positional arguments
            if (i == 1) {
                server_ip = argv[i];
            } else if (i == 2) {
                port = static_cast<uint16_t>(std::stoi(argv[i]));
            }
        }
    }
    

    // First call: Get the number of available interfaces
    size_t num_interfaces = 0;
    int32_t results = easyrdma_Enumerate(nullptr, &num_interfaces, 4); // AF_INET for IPv4
    if (results != easyrdma_Error_Success) {
        std::cout << "Failed to enumerate interfaces: " << results << std::endl;
        return 1;
    }
    
    if (num_interfaces == 0) {
        std::cout << "No RDMA interfaces found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << num_interfaces << " RDMA interface(s)" << std::endl;
    
    // Second call: Get the actual interface addresses
    std::vector<easyrdma_AddressString> interfaces(num_interfaces);
    results = easyrdma_Enumerate(interfaces.data(), &num_interfaces, 4);
    if (results != easyrdma_Error_Success) {
        std::cout << " Failed to get interface addresses: " << results << std::endl;
        return 1;
    }
    
    // Display available interfaces
    std::cout << "Available RDMA interfaces:" << std::endl;
    for (size_t i = 0; i < num_interfaces; i++) {
        std::cout << "  [" << i << "] " << interfaces[i].addressString << std::endl;
    }
    
    std::cout << "Connecting to: " << server_ip << ":" << port << std::endl;
    
    // Step 1: Choose appropriate local interface
    const char* local_ip = nullptr;
    
    // For RDMA, we can actually bind to the same interface as server (different ports)
    std::string server_network = std::string(server_ip).substr(0, 11);
    
    // First try: Use same interface as server (RDMA allows this with different ports)
    for (size_t i = 0; i < num_interfaces; i++) {
        std::string iface_ip = interfaces[i].addressString;
        if (iface_ip == server_ip) {
            local_ip = interfaces[i].addressString;
            std::cout << "🔧 Selected local interface: " << local_ip << " (same as server, different port)" << std::endl;
            break;
        }
    }
    
    // Second try: Use interface on same subnet but different IP
    if (!local_ip) {
        for (size_t i = 0; i < num_interfaces; i++) {
            std::string iface_ip = interfaces[i].addressString;
            std::string iface_network = iface_ip.substr(0, 11);
            
            if (iface_network == server_network && iface_ip != server_ip) {
                local_ip = interfaces[i].addressString;
                std::cout << "🔧 Selected local interface: " << local_ip << " (same subnet as server)" << std::endl;
                break;
            }
        }
    }
    
    // Fallback: Use any available interface
    if (!local_ip && num_interfaces > 0) {
        local_ip = interfaces[0].addressString;
        std::cout << " Selected local interface: " << local_ip << " (fallback)" << std::endl;
    }
    
    if (!local_ip) {
        std::cout << " No suitable local RDMA interface found!" << std::endl;
        std::cout << " Available interfaces:" << std::endl;
        for (size_t i = 0; i < num_interfaces; i++) {
            std::cout << "     " << interfaces[i].addressString << std::endl;
        }
        std::cout << "   Server IP: " << server_ip << std::endl;
        return 1;
    }
    
    // Step 2: Create connector session with explicit local binding
    easyrdma_Session session;
    int32_t result = easyrdma_CreateConnectorSession(local_ip, 0, &session);  // Use port 0 for auto-assign
    if (result != easyrdma_Error_Success) {
        std::cout << " Failed to create connector on " << local_ip << ": " << result << std::endl;
        return 1;
    }
    std::cout << "Created connector session on " << local_ip << std::endl;
    
    // Step 3: Connect to server
    std::cout << "Connecting to server..." << server_ip << ":" << port << std::endl;
    result = easyrdma_Connect(session, easyrdma_Direction_Send, server_ip, port, 5000);
    if (result != easyrdma_Error_Success) {
        std::cout << " Failed to connect: " << result << std::endl;
        if (result == -734010) {
            std::cout << "   This is 'Invalid Address' error. Possible causes:" << std::endl;
            std::cout << "   • Server is not running on " << server_ip << ":" << port << std::endl;
            std::cout << "   • Network interface " << server_ip << " not available for RDMA" << std::endl;
            std::cout << "   • Check: ibv_devinfo and show_gids for available RDMA interfaces" << std::endl;
        } else if (result == -734015) {
            std::cout << " This is 'Unable to Connect' error - server may not be listening" << std::endl;
        }
        easyrdma_CloseSession(session);
        return 1;
    }
    std::cout << "✓ Connected to server!" << std::endl;
    
    // Step 3: Configure buffers
    size_t buffer_size = 1024;  // 1KB buffers
    size_t num_buffers = 5;     // Just 2 buffers
    result = easyrdma_ConfigureBuffers(session, buffer_size, num_buffers);
    if (result != easyrdma_Error_Success) {
        std::cout << " Failed to configure buffers: " << result << std::endl;
        easyrdma_CloseSession(session);
        return 1;
    }
    std::cout << " Configured buffers (" << buffer_size << " bytes x " << num_buffers << ")" << std::endl;
    
    // Step 4: Send some messages
    std::cout << "\n Sending messages..." << std::endl;
    
    const char* messages[] = {
        "Hello from client!",
        "How are you?",
        "This is RDMA!",
        "Message number 4",
        "Final message"
    };
    
    for (int i = 0; i < 5; i++) {  
        // Get send buffer
        easyrdma_InternalBufferRegion send_region;
        result = easyrdma_AcquireSendRegion(session, 1000, &send_region);
        if (result != easyrdma_Error_Success) {
            std::cout << "Failed to get send buffer: " << result << std::endl;
            break;
        }
        
        // Copy message to buffer
        size_t msg_len = strlen(messages[i]);
        if (msg_len > send_region.bufferSize) {
            msg_len = send_region.bufferSize;
        }
        memcpy(send_region.buffer, messages[i], msg_len);
        send_region.usedSize = msg_len;
        
        // Send message (synchronous - blocks until complete)
        std::cout << " Sending: \"" << messages[i] << "\"" << std::endl;
        result = easyrdma_QueueBufferRegion(session, &send_region, nullptr);
        if (result != easyrdma_Error_Success) {
            std::cout << "Failed to send: " << result << std::endl;
            break;
        }
        
        // Wait for echo response
        /*easyrdma_InternalBufferRegion receive_region;
        result = easyrdma_AcquireReceivedRegion(session, 3000, &receive_region);
        if (result == easyrdma_Error_Success) {
            std::string response(static_cast<char*>(receive_region.buffer), receive_region.usedSize);
            std::cout << "📨 Server replied: \"" << response << "\"" << std::endl;
            easyrdma_ReleaseReceivedBufferRegion(session, &receive_region);
        } else if (result == easyrdma_Error_Timeout) {
            std::cout << " No response from server (timeout)" << std::endl;
        } else {
            std::cout << " Failed to receive response: " << result << std::endl;
        } */
    }
    
    // Cleanup
    std::cout << "\n Shutting down..." << std::endl;
    easyrdma_CloseSession(session);
    
    std::cout << "Client finished cleanly" << std::endl;
    return 0;
}