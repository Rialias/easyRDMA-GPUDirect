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

To compile:

Server: g++ -g -O0 -Wall gpudirect_server.cpp -o gpudirect_server -I. -Icore -Lbuild/core -leasyrdma -lcuda -lcudart -std=c++11 -Wl,-rpath,build/core

Client: g++ -fdiagnostics-color=always -g3 -O0 -DDEBUG -Wall -Wextra gpudirect_client.cpp -o gpudirect_client -I. -Icore -Lbuild/core -leasyrdma -lcuda -lcudart -std=c++11 -Wl,-rpath,build/core

To run the code: Start the server first with

Server:  ./gpudirect_server 192.168.30.1
Client:  ./gpudirect_client 192.168.30.1

## Community

We welcome feedback and contributions!

For significant effort or feature work, please start by filing an issue to discuss your approach.

Not sure where to start? The fastest way is to start a discussion on GitHub on the Discussions tab.

## License and Copyright

Copyright © 2022 National Instruments (NI)

Licensed under the [MIT License](LICENSE.txt).
