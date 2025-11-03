#include "core/api/easyrdma.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>

void MyCompletionCallback(void *ctx1, void *ctx2, int32_t status, size_t bytes)
{
    std::cout << "Transfer status: " << status << ", bytes: " << bytes << std::endl;

    if (status == easyrdma_Error_Success)
    {
        std::cout << "Received " << bytes << " bytes into system memory!" << std::endl;
        // Print received data
        char* buffer = (char*)ctx1;
        std::cout << "Data: " << std::string(buffer, bytes) << std::endl;
    }
    else
    {
        std::cout << "Transfer failed with status: " << status << std::endl;
    }
}

int main(int argc, char *argv[])
{
    std::cout << "System Memory RDMA Server (Debug Test)" << std::endl;

    const char *server_ip = (argc > 1) ? argv[1] : "192.168.30.1";
    uint16_t port = (argc > 2) ? std::stoi(argv[2]) : 12345;

    std::cout << "Server: " << server_ip << ":" << port << std::endl;

    // Allocate system memory instead of GPU memory
    size_t bufferSize = 1024; // Match client buffer size
    void* systemBuffer = malloc(bufferSize);
    if (!systemBuffer) {
        std::cout << "Failed to allocate system memory" << std::endl;
        return 1;
    }
    memset(systemBuffer, 0, bufferSize);
    std::cout << "System Memory allocated: " << bufferSize << " bytes at " << systemBuffer << std::endl;

    // Create listener session
    easyrdma_Session server_session;
    int32_t result = easyrdma_CreateListenerSession(server_ip, port, &server_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to create listener: " << result << std::endl;
        free(systemBuffer);
        return 1;
    }
    std::cout << "Created listener session on " << server_ip << ":" << port << std::endl;

    // Wait for client connection
    std::cout << "Waiting for client connection (timeout: 30 seconds)..." << std::endl;
    easyrdma_Session connected_session;
    result = easyrdma_Accept(server_session, easyrdma_Direction_Receive, 30000, &connected_session);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to accept connection: " << result << std::endl;
        easyrdma_CloseSession(server_session);
        free(systemBuffer);
        return 1;
    }
    std::cout << "Client connected!" << std::endl;

    // Configure external buffer with system memory
    result = easyrdma_ConfigureExternalBuffer(connected_session, systemBuffer, bufferSize, 1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to configure system buffer: " << result << std::endl;
        free(systemBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }
    std::cout << "Configured session to use system memory buffer" << std::endl;

    easyrdma_BufferCompletionCallbackData callbackData;
    callbackData.callbackFunction = MyCompletionCallback;
    callbackData.context1 = systemBuffer;
    callbackData.context2 = nullptr;

    // Queue the entire buffer for receiving
    result = easyrdma_QueueExternalBufferRegion(connected_session, systemBuffer, bufferSize, &callbackData, -1);
    if (result != easyrdma_Error_Success)
    {
        std::cout << "Failed to queue system buffer region: " << result << std::endl;
        free(systemBuffer);
        easyrdma_CloseSession(connected_session);
        easyrdma_CloseSession(server_session);
        return 1;
    }

    std::cout << "Ready to receive data into system memory..." << std::endl;
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();

    // Cleanup
    easyrdma_CloseSession(connected_session);
    easyrdma_CloseSession(server_session);
    free(systemBuffer);

    return 0;
}