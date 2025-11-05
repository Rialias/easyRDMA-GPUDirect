#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#define BUFFER_SIZE 4096

int main()
{
    std::cout << "Starting RDMA Host->GPU RoCE loopback test..." << std::endl;

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
    std::cout << "GPU memory initialized with message" << std::endl;

    // Allocate host memory for sender
    char *hostBuffer = new char[BUFFER_SIZE];
    const char *message = "GPU Direct RDMA WRITE: Host→GPU Direct!";
    strcpy(hostBuffer, message);
    int messageLength = strlen(message) + 1;
    std::cout << "Host buffer allocated and initialized for sending" << std::endl;

    // Get RDMA device
    std::cout << "Checking for RDMA devices..." << std::endl;
    ibv_device **dev_list = ibv_get_device_list(nullptr);
    if (!dev_list || !dev_list[0])
    {
        std::cerr << "ERROR: No RDMA devices found!" << std::endl;
        cudaFree(gpuBuffer);
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
        cudaFree(gpuBuffer);
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
        cudaFree(gpuBuffer);
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
        cudaFree(gpuBuffer);
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
        cudaFree(gpuBuffer);
        delete[] hostBuffer;
        return -1;
    }

    printf("Using GID: %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x\n",
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
        delete[] hostBuffer;
        return -1;
    }
    std::cout << "Protection domain allocated" << std::endl;

    // Register memory regions
    std::cout << "Registering memory regions..." << std::endl;
    ibv_mr *recv_mr = ibv_reg_mr(pd, gpuBuffer, BUFFER_SIZE,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!recv_mr)
    {
        std::cerr << "ERROR: Failed to register GPU receive memory region!" << std::endl;
        std::cerr << "GPU Direct RDMA might not be supported or enabled." << std::endl;
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        delete[] hostBuffer;
        return -1;
    }
    std::cout << "GPU memory region registered" << std::endl;

    ibv_mr *send_mr = ibv_reg_mr(pd, hostBuffer, BUFFER_SIZE,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!send_mr)
    {
        std::cerr << "ERROR: Failed to register host send memory region!" << std::endl;
        ibv_dereg_mr(recv_mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
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
        ibv_dereg_mr(send_mr);
        ibv_dereg_mr(recv_mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        delete[] hostBuffer;
        return -1;
    }
    std::cout << "Completion queues created" << std::endl;

    // Create two QPs
    ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = recv_cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 10;
    qp_init_attr.cap.max_recv_wr = 10;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    ibv_qp *qp_sender = ibv_create_qp(pd, &qp_init_attr);
    ibv_qp *qp_receiver = ibv_create_qp(pd, &qp_init_attr);

    if (!qp_sender || !qp_receiver)
    {
        std::cerr << "ERROR: Failed to create queue pairs!" << std::endl;
        if (send_cq)
            ibv_destroy_cq(send_cq);
        if (recv_cq)
            ibv_destroy_cq(recv_cq);
        ibv_dereg_mr(send_mr);
        ibv_dereg_mr(recv_mr);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        cudaFree(gpuBuffer);
        delete[] hostBuffer;
        return -1;
    }
    std::cout << "Queue pairs created: sender=" << qp_sender->qp_num << ", receiver=" << qp_receiver->qp_num << std::endl;

    {
        // Transition QPs to INIT
        std::cout << "Transitioning QPs to INIT state..." << std::endl;
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = 1;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

        if (ibv_modify_qp(qp_sender, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
        {
            std::cerr << "ERROR: Failed to modify sender QP to INIT!" << std::endl;
            goto cleanup;
        }

        if (ibv_modify_qp(qp_receiver, &attr,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
        {
            std::cerr << "ERROR: Failed to modify receiver QP to INIT!" << std::endl;
            goto cleanup;
        }
        std::cout << "QPs transitioned to INIT" << std::endl;

        // Transition QPs to RTR (using GID addressing for RoCE)
        std::cout << "Transitioning QPs to RTR state..." << std::endl;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024;
        attr.dest_qp_num = qp_receiver->qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;

        // RoCE addressing (GID-based)
        attr.ah_attr.is_global = 1;     // MUST be 1 for RoCE
        attr.ah_attr.grh.dgid = my_gid; // Use our own GID for loopback
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.sgid_index = 0;
        attr.ah_attr.grh.hop_limit = 1; // Local network
        attr.ah_attr.grh.traffic_class = 0;
        attr.ah_attr.dlid = 0; // Not used in RoCE
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = 1;

        if (ibv_modify_qp(qp_sender, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        {
            std::cerr << "ERROR: Failed to modify sender QP to RTR!" << std::endl;
            goto cleanup;
        }

        attr.dest_qp_num = qp_sender->qp_num;
        if (ibv_modify_qp(qp_receiver, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        {
            std::cerr << "ERROR: Failed to modify receiver QP to RTR!" << std::endl;
            goto cleanup;
        }
        std::cout << "QPs transitioned to RTR successfully!" << std::endl;

        // Transition QPs to RTS
        std::cout << "Transitioning QPs to RTS state..." << std::endl;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;

        if (ibv_modify_qp(qp_sender, &attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
        {
            std::cerr << "ERROR: Failed to modify sender QP to RTS!" << std::endl;
            goto cleanup;
        }

        if (ibv_modify_qp(qp_receiver, &attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
        {
            std::cerr << "ERROR: Failed to modify receiver QP to RTS!" << std::endl;
            goto cleanup;
        }
        std::cout << "QPs transitioned to RTS successfully!" << std::endl;

        // Post receive (GPU receives the data)
        std::cout << "Posting receive work request to GPU..." << std::endl;
        ibv_sge recv_sge = {};
        recv_sge.addr = (uintptr_t)gpuBuffer;
        recv_sge.length = BUFFER_SIZE;
        recv_sge.lkey = recv_mr->lkey;

        ibv_recv_wr recv_wr = {};
        recv_wr.wr_id = 1;
        recv_wr.sg_list = &recv_sge;
        recv_wr.num_sge = 1;

        ibv_recv_wr *bad_recv_wr;
        if (ibv_post_recv(qp_receiver, &recv_wr, &bad_recv_wr))
        {
            std::cerr << "ERROR: Failed to post receive!" << std::endl;
            goto cleanup;
        }
        std::cout << "GPUreceive posted successfully" << std::endl;

        // Post send (HOST sends the data)
        std::cout << "Posting send work request from host..." << std::endl;
        ibv_sge send_sge = {};
        send_sge.addr = (uintptr_t)hostBuffer;
        send_sge.length = messageLength;
        send_sge.lkey = send_mr->lkey;

        ibv_send_wr send_wr = {};
        send_wr.wr_id = 2;
        send_wr.sg_list = &send_sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;

        ibv_send_wr *bad_send_wr;
        if (ibv_post_send(qp_sender, &send_wr, &bad_send_wr))
        {
            std::cerr << "ERROR: Failed to post send!" << std::endl;
            goto cleanup;
        }
        std::cout << "Send posted successfully" << std::endl;

        // Poll for completion with timeout
        std::cout << "Polling for completion..." << std::endl;

        // First check send completion
        ibv_wc send_wc = {};
        int send_poll_result;
        int send_timeout = 0;

        std::cout << "Checking send completion..." << std::endl;
        while ((send_poll_result = ibv_poll_cq(send_cq, 1, &send_wc)) == 0)
        {
            send_timeout++;
            if (send_timeout > 1000000)
            {
                std::cerr << "ERROR: Send timeout!" << std::endl;
                goto cleanup;
            }
        }

        if (send_poll_result > 0)
        {
            if (send_wc.status == IBV_WC_SUCCESS)
            {
                std::cout << "Send completed successfully" << std::endl;
            }
            else
            {
                std::cerr << "Send failed with status: " << send_wc.status << std::endl;
                goto cleanup;
            }
        }

        // Then check receive completion
        ibv_wc recv_wc = {};
        int recv_poll_result;
        int recv_timeout = 0;

        std::cout << "Checking receive completion..." << std::endl;
        while ((recv_poll_result = ibv_poll_cq(recv_cq, 1, &recv_wc)) == 0)
        {
            recv_timeout++;
            if (recv_timeout > 1000000)
            {
                std::cerr << "ERROR: Receive timeout!" << std::endl;
                goto cleanup;
            }
        }

        if (recv_poll_result < 0)
        {
            std::cerr << "ERROR: Polling failed!" << std::endl;
            goto cleanup;
        }

        if (recv_wc.status == IBV_WC_SUCCESS)
        {
            std::cout << "SUCCESS: Host -> GPU RoCE RDMA transfer complete!" << std::endl;

            // Copy and display original GPU message for verification

            char *verifyBuffer = new char[BUFFER_SIZE];
            memset(verifyBuffer, 0, BUFFER_SIZE);
            cudaErr = cudaMemcpy(verifyBuffer, gpuBuffer, BUFFER_SIZE, cudaMemcpyDeviceToHost);
            if (cudaErr == cudaSuccess)
            {
                std::cout << "Original host message: " << hostBuffer << std::endl;
                std::cout << "GPU Direct RDMA result: " << verifyBuffer << std::endl;

                // Verify data integrity
                bool match = (strncmp(hostBuffer, verifyBuffer, messageLength) == 0);
                std::cout << "Data integrity check: " << (match ? "PASSED " : "FAILED ") << std::endl;

                if (match)
                {
                    std::cout << "Data was transferred DIRECTLY from host to GPU via RDMA!" << std::endl;
                }
            }
            else
            {
                std::cerr << "Failed to copy original message from GPU: " << cudaGetErrorString(cudaErr) << std::endl;
            }
            delete[] verifyBuffer;
        }
        else
        {
            std::cerr << "ERROR: RDMA transfer failed!" << std::endl;
            std::cerr << "Work completion status: " << recv_wc.status << std::endl;
        }
    }

cleanup:
    // Cleanup
    std::cout << "Cleaning up resources..." << std::endl;
    if (qp_sender)
        ibv_destroy_qp(qp_sender);
    if (qp_receiver)
        ibv_destroy_qp(qp_receiver);
    if (send_cq)
        ibv_destroy_cq(send_cq);
    if (recv_cq)
        ibv_destroy_cq(recv_cq);
    if (send_mr)
        ibv_dereg_mr(send_mr);
    if (recv_mr)
        ibv_dereg_mr(recv_mr);
    if (pd)
        ibv_dealloc_pd(pd);
    if (ctx)
        ibv_close_device(ctx);
    if (dev_list)
        ibv_free_device_list(dev_list);
    cudaFree(gpuBuffer);
    delete[] hostBuffer;

    std::cout << "Program completed." << std::endl;
    return 0;
}