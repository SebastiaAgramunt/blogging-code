#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Detected " << deviceCount << " CUDA Capable Device(s)\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        // Select device
        cudaSetDevice(dev);

        // Query device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        // Query memory info
        size_t freeBytes = 0, totalBytes = 0;
        cudaMemGetInfo(&freeBytes, &totalBytes);

        std::cout << "Device " << dev << ": " << prop.name << "\n";
        std::cout << "  PCI Domain/Bus/Device ID: " 
                  << prop.pciDomainID << "/" 
                  << prop.pciBusID    << "/" 
                  << prop.pciDeviceID << "\n";
        std::cout << "  Compute capability: " 
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " 
                  << (prop.totalGlobalMem  / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Free memory (current): " 
                  << (freeBytes  / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Total allocatable memory (current): " 
                  << (totalBytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Memory clock rate: " 
                  << (prop.memoryClockRate * 1e-3) << " MHz\n";
        std::cout << "  Memory bus width: " 
                  << prop.memoryBusWidth << " bits\n";
        std::cout << "  L2 cache size: " 
                  << prop.l2CacheSize / 1024 << " KB\n";
        std::cout << "  Max shared memory per block: " 
                  << prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "  Total constant memory: " 
                  << prop.totalConstMem / 1024 << " KB\n";
        std::cout << "  Warp size: " 
                  << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " 
                  << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per multiprocessor: " 
                  << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Multiprocessor count: " 
                  << prop.multiProcessorCount << "\n";
        std::cout << "  Max grid dimensions: [" 
                  << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " 
                  << prop.maxGridSize[2] << "]\n";
        std::cout << "  Max block dimensions: [" 
                  << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " 
                  << prop.maxThreadsDim[2] << "]\n";
        std::cout << "  Clock rate: " 
                  << (prop.clockRate * 1e-3) << " MHz\n";
        std::cout << "  Concurrent kernels: " 
                  << (prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << "  ECC enabled: " 
                  << (prop.ECCEnabled ? "Yes" : "No") << "\n";
        std::cout << "  Integrated device: " 
                  << (prop.integrated ? "Yes" : "No") << "\n";
        std::cout << "  Can map host memory: " 
                  << (prop.canMapHostMemory ? "Yes" : "No") << "\n";
        std::cout << "  Compute mode: ";
        switch (prop.computeMode) {
            case cudaComputeModeDefault:      std::cout << "Default\n"; break;
            case cudaComputeModeExclusive:    std::cout << "Exclusive\n"; break;
            case cudaComputeModeProhibited:   std::cout << "Prohibited\n"; break;
            case cudaComputeModeExclusiveProcess:
                                              std::cout << "Exclusive Process\n"; break;
            default:                          std::cout << "Unknown\n"; break;
        }
        std::cout << "  Unified addressing: " 
                  << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
        std::cout << "  Async engines: " 
                  << prop.asyncEngineCount << "\n";
        std::cout << "  Device overlap: " 
                  << (prop.deviceOverlap ? "Yes" : "No") << "\n";
        std::cout << "  PCI bus ID: " 
                  << prop.pciBusID << "\n";
        std::cout << "  PCI device ID: " 
                  << prop.pciDeviceID << "\n";
        std::cout << "\n";
    }

    return 0;
}