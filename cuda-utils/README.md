# Cuda Utils

Supporting material for blogpost [agramunt.me/posts/cuda-utils](https://agramunt.me/posts/cuda-utils/).

## Build & Installation
Install utils in `$HOME/.local/bin` with

```bash
rm -rf build && mkdir build && cd build
cmake \
  -DGPU_INFO_OUT_NAME=gpu_info \
  -DGPU_ALLOC_OUT_NAME=gpu_allocate \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80" \
  -DCMAKE_INSTALL_PREFIX=${HOME}/.local \
  ..
cmake --build .
cmake --install .
```

Feel free to change `gpu_info` and `gpu_allocate` names in cmake to whatever name you want to call them. If you have path `$HOME/.local/bin` in your `$PATH` then the execs should be available by just calling in command line

```bash
gpu_info
gpu_allocate
```

If you don't want to install them and just build insdide the project just run

```bash
rm -rf build && mkdir build && cd build
cmake ..
cmake --build .
```

## Usage

### tool gpu_info

Gives you the basic information of all your GPUs in the current host machine. Just run `gpu_info` executable and get a result similar to:

```
Detected 1 CUDA Capable Device(s)

Device 0: NVIDIA A100-PCIE-40GB
  PCI Domain/Bus/Device ID: 0/7/0
  Compute capability: 8.0
  Total global memory: 40442.4 MB
  Free memory (current): 40019.6 MB
  Total allocatable memory (current): 40442.4 MB
  Memory clock rate: 1215 MHz
  Memory bus width: 5120 bits
  L2 cache size: 40960 KB
  Max shared memory per block: 48 KB
  Total constant memory: 64 KB
  Warp size: 32
  Max threads per block: 1024
  Max threads per multiprocessor: 2048
  Multiprocessor count: 108
  Max grid dimensions: [2147483647, 65535, 65535]
  Max block dimensions: [1024, 1024, 64]
  Clock rate: 1410 MHz
  Concurrent kernels: Yes
  ECC enabled: Yes
  Integrated device: No
  Can map host memory: Yes
  Compute mode: Default
  Unified addressing: Yes
  Async engines: 3
  Device overlap: Yes
  PCI bus ID: 7
  PCI device ID: 0
```

for an A100 Nvidia GPU.

## tool gpu_allocate

A tool that allocates memory in the gpu for a certain amount of time, usage is

```bash
<gpu_id> <memory_amount (e.g., 512M, 1G, or bytes)> <duration (e.g., 10s, 5m, 1h)>
```

For instance, run `gpu_allocate 1 5G 10h` to allocate 5 gigabytes on GPU with id 1 for 10 hours.