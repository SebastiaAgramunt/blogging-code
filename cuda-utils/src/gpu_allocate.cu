#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <thread>

// Helper to parse size strings like 1024, 100M, 2G, etc.
size_t parseSize(const std::string& s) {
    char unit = s.back();
    std::string num = s;
    size_t multiplier = 1;
    if (unit == 'K' || unit == 'k') {
        multiplier = 1024ULL;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'M' || unit == 'm') {
        multiplier = 1024ULL * 1024ULL;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'G' || unit == 'g') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
        num = s.substr(0, s.size() - 1);
    }
    return static_cast<size_t>(std::stoull(num) * multiplier);
}

// Helper to parse time strings like 10s, 5m, 1h, or raw seconds
long parseTime(const std::string& s) {
    char unit = s.back();
    std::string num = s;
    long multiplier = 1;
    if (unit == 's' || unit == 'S') {
        multiplier = 1;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'm' || unit == 'M') {
        multiplier = 60;
        num = s.substr(0, s.size() - 1);
    } else if (unit == 'h' || unit == 'H') {
        multiplier = 3600;
        num = s.substr(0, s.size() - 1);
    }
    return static_cast<long>(std::stol(num) * multiplier);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <gpu_id> <memory_amount (e.g., 512M, 1G, or bytes)> <duration (e.g., 10s, 5m, 1h)>" << std::endl;
        return EXIT_FAILURE;
    }

    int gpuId = std::stoi(argv[1]);
    size_t bytes = parseSize(argv[2]);
    long duration = parseTime(argv[3]);

    cudaError_t err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
        std::cerr << "Error setting GPU device " << gpuId << ": " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    void* d_ptr = nullptr;
    err = cudaMalloc(&d_ptr, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating " << bytes << " bytes on GPU " << gpuId
                  << ": " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Successfully allocated " << bytes << " bytes on GPU " << gpuId
              << ", holding for " << duration << " seconds..." << std::endl;

    // Keep the allocation alive for the specified duration
    std::this_thread::sleep_for(std::chrono::seconds(duration));

    // Free the allocation and exit
    cudaFree(d_ptr);
    std::cout << "Freed memory and exiting." << std::endl;
    return EXIT_SUCCESS;
}