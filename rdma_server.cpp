#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#define BUFFER_SIZE 4096
#define TCP_PORT 12345

struct connection_info {
    uint32_t qp_num;
    uint16_t lid;
    ibv_gid gid;
    uint64_t gpu_addr;
    uint32_t gpu_rkey;
};

int main()
{
    std::cout << "Starting RDMA Server (GPU Receiver)..." << std::endl;

    // Declare variables that might be used with goto to avoid initialization crossing issues
    connection_info client_info = {};
    bool connection_success = false;

    // Check for CUDA devices
    int deviceCount;
    cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaErr) << std::endl;
        return -1;
    }
    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;

    // Allocate GPU memory for receiver
    void *gpuBuffer;
    cudaErr = cudaMalloc(&gpuBuffer, BUFFER_SIZE);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(cudaErr) << std::endl;
        return -1;
    }
    std::cout << "GPU memory allocated successfully" << std::endl;

    // Initialize GPU memory with zeros
    cudaErr = cudaMemset(gpuBuffer, 0, BUFFER_SIZE);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "Failed to initialize GPU memory: " << cudaGetErrorString(cudaErr) << std::endl;
        cudaFree(gpuBuffer);
        return -1;
    }
    std::cout << "GPU memory initialized" << std::endl;

    // Get RDMA device
    std::cout << "Checking for RDMA devices..." << std::endl;
    ibv_device **dev_list = ibv_get_device_list(nullptr);
    if (!dev_list || !dev_list[0])
    {
        std::cerr << "ERROR: No RDMA devices found!" << std::endl;
        cudaFree(gpuBuffer);
        return -1;
    }

    std::cout << "Found RDMA device: " << ibv_get_device_name(dev_list[0]) << std::endl;

    ibv_device *ib_dev = dev_list[0];
    ibv_context *ctx = ibv_open_device(ib_dev);
    if (!ctx)
    {
        std::cerr << "ERROR: Failed to open RDMA device!" << std::endl;
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }
    std::cout << "Device opened successfully" << std::endl;

    // Query port attributes
    ibv_port_attr port_attr;
    if (ibv_query_port(ctx, 1, &port_attr))
    {
        std::cerr << "ERROR: Failed to query port!" << std::endl;
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }

    std::cout << "Port state: " << port_attr.state << " (Active=4)" << std::endl;
    std::cout << "Link layer: " << (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ? "Ethernet (RoCE)" : "InfiniBand") << std::endl;

    if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET)
    {
        std::cerr << "ERROR: This program is designed for RoCE (Ethernet). Your device is configured for InfiniBand." << std::endl;
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }

    // Get GID for RoCE addressing
    ibv_gid my_gid;
    if (ibv_query_gid(ctx, 1, 0, &my_gid))
    {
        std::cerr << "ERROR: Failed to query GID!" << std::endl;
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }

    printf("Server GID: %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x\n",
           my_gid.raw[0], my_gid.raw[1], my_gid.raw[2], my_gid.raw[3],
           my_gid.raw[4], my_gid.raw[5], my_gid.raw[6], my_gid.raw[7],
           my_gid.raw[8], my_gid.raw[9], my_gid.raw[10], my_gid.raw[11],
           my_gid.raw[12], my_gid.raw[13], my_gid.raw[14], my_gid.raw[15]);

    ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd)
    {
        std::cerr << "ERROR: Failed to allocate protection domain!" << std::endl;
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }
    std::cout << "Protection domain allocated" << std::endl;

    // Register GPU memory region
    std::cout << "Registering GPU memory region..." << std::endl;
    ibv_mr *gpu_mr = ibv_reg_mr(pd, gpuBuffer, BUFFER_SIZE,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!gpu_mr)
    {
        std::cerr << "ERROR: Failed to register GPU memory region!" << std::endl;
        std::cerr << "GPU Direct RDMA might not be supported or enabled." << std::endl;
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }
    std::cout << "GPU memory region registered" << std::endl;

    // Create Completion Queues
    ibv_cq *send_cq = ibv_create_cq(ctx, 10, nullptr, nullptr, 0);
    ibv_cq *recv_cq = ibv_create_cq(ctx, 10, nullptr, nullptr, 0);
    if (!send_cq || !recv_cq)
    {
        std::cerr << "ERROR: Failed to create completion queues!" << std::endl;
        ibv_dereg_mr(gpu_mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }
    std::cout << "Completion queues created" << std::endl;

    // Create QP
    ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = recv_cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 10;
    qp_init_attr.cap.max_recv_wr = 10;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp)
    {
        std::cerr << "ERROR: Failed to create queue pair!" << std::endl;
        ibv_destroy_cq(send_cq);
        ibv_destroy_cq(recv_cq);
        ibv_dereg_mr(gpu_mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        return -1;
    }
    std::cout << "Queue pair created: " << qp->qp_num << std::endl;

    // Transition QP to INIT
    std::cout << "Transitioning QP to INIT state..." << std::endl;
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    {
        std::cerr << "ERROR: Failed to modify QP to INIT!" << std::endl;
        goto cleanup;
    }
    std::cout << "QP transitioned to INIT" << std::endl;

    // Setup TCP server and exchange connection info
    std::cout << "Setting up TCP server for connection exchange..." << std::endl;
    
    // Use a separate function-like block to handle TCP connection
    do {
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0)
        {
            std::cerr << "ERROR: Failed to create TCP socket!" << std::endl;
            break;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(TCP_PORT);

        if (bind(server_fd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0)
        {
            std::cerr << "ERROR: Failed to bind TCP socket!" << std::endl;
            close(server_fd);
            break;
        }

        if (listen(server_fd, 1) < 0)
        {
            std::cerr << "ERROR: Failed to listen on TCP socket!" << std::endl;
            close(server_fd);
            break;
        }

        std::cout << "Waiting for client connection on port " << TCP_PORT << "..." << std::endl;

        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0)
        {
            std::cerr << "ERROR: Failed to accept client connection!" << std::endl;
            close(server_fd);
            break;
        }
        std::cout << "Client connected!" << std::endl;

        // Prepare server connection info
        connection_info server_info = {};
        server_info.qp_num = qp->qp_num;
        server_info.lid = port_attr.lid;
        server_info.gid = my_gid;
        server_info.gpu_addr = (uint64_t)gpuBuffer;
        server_info.gpu_rkey = gpu_mr->rkey;

        // Send server info to client
        if (send(client_fd, &server_info, sizeof(server_info), 0) != sizeof(server_info))
        {
            std::cerr << "ERROR: Failed to send server info!" << std::endl;
            close(client_fd);
            close(server_fd);
            break;
        }

        // Receive client info
        if (recv(client_fd, &client_info, sizeof(client_info), MSG_WAITALL) != sizeof(client_info))
        {
            std::cerr << "ERROR: Failed to receive client info!" << std::endl;
            close(client_fd);
            close(server_fd);
            break;
        }

        close(client_fd);
        close(server_fd);
        std::cout << "Connection info exchanged successfully!" << std::endl;
        connection_success = true;
    } while (false);
    
    if (!connection_success)
    {
        goto cleanup;
    }

    // Transition QP to RTR (Ready to Receive)
    std::cout << "Transitioning QP to RTR state..." << std::endl;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = client_info.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;

    // RoCE addressing (GID-based)
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = client_info.gid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.sgid_index = 0;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;

    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    {
        std::cerr << "ERROR: Failed to modify QP to RTR!" << std::endl;
        goto cleanup;
    }

    // Transition QP to RTS (Ready to Send)
    std::cout << "Transitioning QP to RTS state..." << std::endl;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;

    if (ibv_modify_qp(qp, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
    {
        std::cerr << "ERROR: Failed to modify QP to RTS!" << std::endl;
        goto cleanup;
    }
    std::cout << "QP transitioned to RTS successfully!" << std::endl;

    std::cout << "Server ready! Waiting for RDMA operations..." << std::endl;
    std::cout << "GPU buffer address: " << gpuBuffer << std::endl;
    std::cout << "GPU buffer rkey: " << gpu_mr->rkey << std::endl;

    // Just wait for client to perform RDMA operations
    // The client will write directly to our GPU memory
    std::cout << "Press Enter to check GPU buffer contents after client operations..." << std::endl;
    std::cin.get();

    // Check GPU buffer contents - moved allocation to avoid goto issues
    {
        char *verifyBuffer = new char[BUFFER_SIZE];
        memset(verifyBuffer, 0, BUFFER_SIZE);
        cudaErr = cudaMemcpy(verifyBuffer, gpuBuffer, BUFFER_SIZE, cudaMemcpyDeviceToHost);
        if (cudaErr == cudaSuccess)
        {
            std::cout << "GPU buffer contents: " << verifyBuffer << std::endl;
            
            if (strlen(verifyBuffer) > 0)
            {
                std::cout << "SUCCESS: Data received in GPU memory via RDMA!" << std::endl;
            }
            else
            {
                std::cout << "No data found in GPU buffer" << std::endl;
            }
        }
        else
        {
            std::cerr << "Failed to copy data from GPU: " << cudaGetErrorString(cudaErr) << std::endl;
        }
        delete[] verifyBuffer;
    }

cleanup:
    // Cleanup
    std::cout << "Cleaning up resources..." << std::endl;
    if (qp)
        ibv_destroy_qp(qp);
    if (send_cq)
        ibv_destroy_cq(send_cq);
    if (recv_cq)
        ibv_destroy_cq(recv_cq);
    if (gpu_mr)
        ibv_dereg_mr(gpu_mr);
    if (pd)
        ibv_dealloc_pd(pd);
    if (ctx)
        ibv_close_device(ctx);
    if (dev_list)
        ibv_free_device_list(dev_list);
    cudaFree(gpuBuffer);

    std::cout << "Server completed." << std::endl;
    return 0;
}