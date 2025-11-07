# Copyright (c) 2022 National Instruments
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_STANDARD 11)

# Collect all example source files
file(GLOB EXAMPLE_SOURCES *.cpp)

# Platform-specific definitions
if(WIN32)
    add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -DWIN32_LEAN_AND_MEAN)
endif()

# Include directories
include_directories(. .. ../core ../core/api)

# Find CUDA libraries (simple approach to avoid compiler conflicts)
find_library(CUDA_LIB cuda PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
find_library(CUDART_LIB cudart PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

# Find InfiniBand libraries
find_library(IBV_LIB ibverbs PATHS /usr/lib/x86_64-linux-gnu /usr/lib64)
find_library(MLX5_LIB mlx5 PATHS /usr/lib/x86_64-linux-gnu /usr/lib64)

# Create executables for each example source file
foreach(source_file ${EXAMPLE_SOURCES})
    get_filename_component(exe_name ${source_file} NAME_WE)
    
    add_executable(${exe_name} ${source_file})
    target_link_libraries(${exe_name} easyrdma)
    
    # Platform-specific linking
    if(WIN32)
        target_link_libraries(${exe_name} ws2_32)
    elseif(UNIX)
        # Add CUDA libraries for GPU examples (simple library linking)
        if(CUDA_LIB AND CUDART_LIB)
            target_link_libraries(${exe_name} ${CUDA_LIB} ${CUDART_LIB})
        endif()
        
        # Add InfiniBand libraries for direct ibv usage
        if(IBV_LIB)
            target_link_libraries(${exe_name} ${IBV_LIB})
        endif()
        if(MLX5_LIB)
            target_link_libraries(${exe_name} ${MLX5_LIB})
        endif()
        
        # Add runtime library path and allow shared library dependencies
        target_link_libraries(${exe_name} -Wl,--allow-shlib-undefined -lrt -lpthread)
        
        # Set RPATH for runtime library discovery
        set_target_properties(${exe_name} PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endforeach()
