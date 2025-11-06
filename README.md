<!--
Copyright (c) 2022 National Instruments
SPDX-License-Identifier: MIT
-->

# easyRDMA

An easy-to-use, cross-platform, MIT-licensed [RDMA](https://en.wikipedia.org/wiki/Remote_direct_memory_access) library from NI.

Features include:

- Cross-platform support for Windows and Linux
- Zero-copy send and receive
- Synchronous and asynchronous operation modes
- Internally-allocated or externally-provided buffer models

easyRDMA is a part of [NI-RDMA](https://www.ni.com/en-us/support/documentation/release-notes/product.ni-rdma.html).

## Getting Started

To learn how to build, debug, and test easyRDMA, please read [CONTRIBUTING.md](./CONTRIBUTING.md).

## Examples

There are various examples available to get started with the easyRDMA library.

### Basic RDMA Examples

**`easyRDMA_simple_client.cpp`** and **`easyRDMA_simple_server.cpp`** demonstrate the basic functionality of the easyRDMA library. These examples provide details on the various RDMA interfaces available and show how to get started with easyRDMA send and receive operations. Both the server and client use host memory for data transfer.

### GPUDirect RDMA Examples

The GPU examples demonstrate the integration of GPUDirect RDMA with easyRDMA:

- **`gpu_to_host_client.cpp`**: Uses GPU memory to send data from the client
- **`gpu_to_host_server.cpp`**: Receives data into internal host memory at the server

- **`externalhost_to_gpu_client.cpp`**: Uses external system memory to send messages from the client  
- **`externalhost_to_gpu_server.cpp`**: Receives data into GPU memory at the server

**GPU-to-GPU Transfer**: You can test GPU-to-GPU transfers by pairing `gpu_to_host_client.cpp` (GPU sender) with `externalhost_to_gpu_server.cpp` (GPU receiver).

## Compile and Run

All example files are included in the CMake build system. To build and run:

1. **Build the examples**:
   ```bash
   cd build
   make
   ```

2. **Run the examples** (start server first):
   ```bash
   # Terminal 1 - Start server
   ./gpu_to_host_server 192.168.30.6
   
   # Terminal 2 - Run client
   ./gpu_to_host_client 192.168.30.6
   ```

**Network Configuration**: 
- `192.168.30.6` is the server IP address (listening interface)
- Client automatically uses `192.168.30.1` as the local interface (if not specified explicitly)
- These IPs correspond to your RDMA-capable network interfaces in a loopback configuration

## Community

We welcome feedback and contributions!

For significant effort or feature work, please start by filing an issue to discuss your approach.

Not sure where to start? The fastest way is to start a discussion on GitHub on the Discussions tab.

## License and Copyright

Copyright © 2022 National Instruments (NI)

Licensed under the [MIT License](LICENSE.txt).
