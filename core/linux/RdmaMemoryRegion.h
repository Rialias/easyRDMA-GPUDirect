// Copyright (c) 2022 National Instruments
// SPDX-License-Identifier: MIT

#pragma once

#include "RdmaCommon.h"
#include "rdma/rdma_verbs.h"
#include <cstdio>
#include <unistd.h>
#include <iostream>

#ifdef __has_include
  #if __has_include(<cuda_runtime.h>)
    #include <cuda_runtime.h>
    #define HAVE_CUDA 1
  #endif
#endif

class RdmaMemoryRegion
{
public:
    // Automatic memory type detection and registration
    RdmaMemoryRegion(rdma_cm_id* _cm_id, void* buffer, size_t length) :
        cm_id(_cm_id)
    {
        // Automatically detect if this is GPU memory
        if (IsGpuMemory(buffer)) {
            std::cout << "RdmaMemoryRegion: Auto-detected GPU memory, using GPU Direct RDMA" << std::endl;
            mr = RegisterGpuMemory(buffer, length);
        } else {
            mr = rdma_reg_msgs(cm_id, buffer, length);
        }
        HandleErrorFromPointer(mr);
    }
    
    ~RdmaMemoryRegion()
    {
        if (mr) {
            rdma_dereg_mr(mr);
        }
    }
    
    ibv_mr* GetMR()
    {
        return mr;
    }

private:
    bool IsGpuMemory(void* buffer)
    {
        if (!buffer) return false;
        
#ifdef HAVE_CUDA
        // Use CUDA runtime to check if this is device memory
        cudaPointerAttributes attrs;
        cudaError_t result = cudaPointerGetAttributes(&attrs, buffer);
        
        if (result == cudaSuccess) {
            // Check memory type
#if CUDART_VERSION >= 10000
            return (attrs.type == cudaMemoryTypeDevice);
#else
            return (attrs.memoryType == cudaMemoryTypeDevice);
#endif
        }
        
        // Clear CUDA error state if query failed
        cudaGetLastError();
#endif
        
        return false;
    }
    
    ibv_mr* RegisterGpuMemory(void* gpuBuffer, size_t length)
    {
        if (!cm_id || !cm_id->pd) {
            return nullptr;
        }
    
        
        // Check if nvidia_peermem is available
        if (access("/sys/module/nvidia_peermem", F_OK) == 0) {            
            // GPU Direct RDMA flags
            int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
            
            // Try to register GPU memory directly
            ibv_mr* gpu_mr = ibv_reg_mr(cm_id->pd, gpuBuffer, length, access_flags);
            return gpu_mr; // Return the result (could be nullptr if failed)
        }
        else
        {
            return nullptr;
        }
    }

protected:
    rdma_cm_id* cm_id;
    ibv_mr* mr;
};