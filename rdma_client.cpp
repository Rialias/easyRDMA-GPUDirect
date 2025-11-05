#include <infiniband/verbs.h>
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

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <server_ip>" << std::endl;
        return -1;
    }

    std::cout << "Starting RDMA Client (Host Sender)..." << std::endl;
    std::cout << "Connecting to server: " << argv[1] << std::endl;

    // Declare all variables that might be used with goto at the top to avoid crossing initialization
    connection_info client_info = {};
    connection_info server_info = {};
    ibv_sge sge = {};
    ibv_send_wr wr = {};
    ibv_wc wc = {};
    int timeout = 0;
    int client_fd = -1;
    sockaddr_in server_addr = {};

    // Allocate host memory for sender
    char *hostBuffer = new char[BUFFER_SIZE];
    const char *message = "GPU Direct RDMA WRITE: Host→GPU Direct via Client-Server!";
    strcpy(hostBuffer, message);
    int messageLength = strlen(message) + 1;
    std::cout << "Host buffer allocated and initialized with message: " << message << std::endl;

    // Get RDMA device
    std::cout << "Checking for RDMA devices..." << std::endl;
    ibv_device **dev_list = ibv_get_device_list(nullptr);
    if (!dev_list || !dev_list[0])
    {
        std::cerr << "ERROR: No RDMA devices found!" << std::endl;
        delete[] hostBuffer;
        return -1;
    }

    std::cout << "Found RDMA device: " << ibv_get_device_name(dev_list[0]) << std::endl;

    ibv_device *ib_dev = dev_list[0];
    ibv_context *ctx = ibv_open_device(ib_dev);
    if (!ctx)
    {
        std::cerr << "ERROR: Failed to open RDMA device!" << std::endl;
        ibv_free_device_list(dev_list);
        delete[] hostBuffer;
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
        delete[] hostBuffer;
        return -1;
    }

    std::cout << "Port state: " << port_attr.state << " (Active=4)" << std::endl;
    std::cout << "Link layer: " << (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET ? "Ethernet (RoCE)" : "InfiniBand") << std::endl;

    if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET)
    {
        std::cerr << "ERROR: This program is designed for RoCE (Ethernet). Your device is configured for InfiniBand." << std::endl;
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        delete[] hostBuffer;
        return -1;
    }

    // Get GID for RoCE addressing
    ibv_gid my_gid;
    if (ibv_query_gid(ctx, 1, 0, &my_gid))
    {
        std::cerr << "ERROR: Failed to query GID!" << std::endl;
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        delete[] hostBuffer;
        return -1;
    }

    printf("Client GID: %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x\n",
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
        delete[] hostBuffer;
        return -1;
    }
    std::cout << "Protection domain allocated" << std::endl;

    // Register host memory region
    std::cout << "Registering host memory region..." << std::endl;
    ibv_mr *host_mr = ibv_reg_mr(pd, hostBuffer, BUFFER_SIZE,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!host_mr)
    {
        std::cerr << "ERROR: Failed to register host memory region!" << std::endl;
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        delete[] hostBuffer;
        return -1;
    }
    std::cout << "Host memory region registered" << std::endl;

    // Create Completion Queues
    ibv_cq *send_cq = ibv_create_cq(ctx, 10, nullptr, nullptr, 0);
    ibv_cq *recv_cq = ibv_create_cq(ctx, 10, nullptr, nullptr, 0);
    if (!send_cq || !recv_cq)
    {
        std::cerr << "ERROR: Failed to create completion queues!" << std::endl;
        ibv_dereg_mr(host_mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        delete[] hostBuffer;
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
        ibv_dereg_mr(host_mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        delete[] hostBuffer;
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

    // Connect to server via TCP to exchange connection info
    std::cout << "Connecting to server via TCP..." << std::endl;
    client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd < 0)
    {
        std::cerr << "ERROR: Failed to create TCP socket!" << std::endl;
        goto cleanup;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(TCP_PORT);
    if (inet_pton(AF_INET, argv[1], &server_addr.sin_addr) <= 0)
    {
        std::cerr << "ERROR: Invalid server IP address!" << std::endl;
        close(client_fd);
        goto cleanup;
    }

    if (connect(client_fd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0)
    {
        std::cerr << "ERROR: Failed to connect to server!" << std::endl;
        close(client_fd);
        goto cleanup;
    }
    std::cout << "Connected to server successfully!" << std::endl;

    // Prepare client connection info
    client_info.qp_num = qp->qp_num;
    client_info.lid = port_attr.lid;
    client_info.gid = my_gid;
    client_info.gpu_addr = 0; // Client doesn't have GPU memory
    client_info.gpu_rkey = 0;

    // Receive server info first
    if (recv(client_fd, &server_info, sizeof(server_info), MSG_WAITALL) != sizeof(server_info))
    {
        std::cerr << "ERROR: Failed to receive server info!" << std::endl;
        close(client_fd);
        goto cleanup;
    }

    // Send client info to server
    if (send(client_fd, &client_info, sizeof(client_info), 0) != sizeof(client_info))
    {
        std::cerr << "ERROR: Failed to send client info!" << std::endl;
        close(client_fd);
        goto cleanup;
    }

    close(client_fd);
    std::cout << "Connection info exchanged successfully!" << std::endl;
    std::cout << "Server QP: " << server_info.qp_num << std::endl;
    std::cout << "Server GPU address: 0x" << std::hex << server_info.gpu_addr << std::dec << std::endl;
    std::cout << "Server GPU rkey: " << server_info.gpu_rkey << std::endl;

    // Transition QP to RTR (Ready to Receive)
    std::cout << "Transitioning QP to RTR state..." << std::endl;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = server_info.qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;

    // RoCE addressing (GID-based)
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = server_info.gid;
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

    // Perform RDMA WRITE to server's GPU memory
    std::cout << "Performing RDMA WRITE to server's GPU memory..." << std::endl;
    
    sge.addr = (uintptr_t)hostBuffer;
    sge.length = messageLength;
    sge.lkey = host_mr->lkey;

    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = server_info.gpu_addr;
    wr.wr.rdma.rkey = server_info.gpu_rkey;

    ibv_send_wr *bad_wr;
    if (ibv_post_send(qp, &wr, &bad_wr))
    {
        std::cerr << "ERROR: Failed to post RDMA WRITE!" << std::endl;
        goto cleanup;
    }
    std::cout << "RDMA WRITE posted successfully" << std::endl;

    // Poll for completion
    std::cout << "Polling for completion..." << std::endl;
    int poll_result;
    timeout = 0;

    while ((poll_result = ibv_poll_cq(send_cq, 1, &wc)) == 0)
    {
        timeout++;
        if (timeout > 1000000)
        {
            std::cerr << "ERROR: RDMA WRITE timeout!" << std::endl;
            goto cleanup;
        }
    }

    if (poll_result < 0)
    {
        std::cerr << "ERROR: Polling failed!" << std::endl;
        goto cleanup;
    }

    if (wc.status == IBV_WC_SUCCESS)
    {
        std::cout << "SUCCESS: RDMA WRITE completed successfully!" << std::endl;
        std::cout << "Data written directly from host to server's GPU memory!" << std::endl;
        std::cout << "Message sent: " << message << std::endl;
        std::cout << "Message length: " << messageLength << " bytes" << std::endl;
    }
    else
    {
        std::cerr << "ERROR: RDMA WRITE failed with status: " << wc.status << std::endl;
    }

cleanup:
    // Cleanup
    std::cout << "Cleaning up resources..." << std::endl;
    if (client_fd >= 0)
        close(client_fd);
    if (qp)
        ibv_destroy_qp(qp);
    if (send_cq)
        ibv_destroy_cq(send_cq);
    if (recv_cq)
        ibv_destroy_cq(recv_cq);
    if (host_mr)
        ibv_dereg_mr(host_mr);
    if (pd)
        ibv_dealloc_pd(pd);
    if (ctx)
        ibv_close_device(ctx);
    if (dev_list)
        ibv_free_device_list(dev_list);
    delete[] hostBuffer;

    std::cout << "Client completed." << std::endl;
    return 0;
}