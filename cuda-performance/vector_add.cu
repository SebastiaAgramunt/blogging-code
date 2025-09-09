#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    std::srand(std::time(0));
    std::ofstream csv("timings.csv");
    csv << "N,Mb,allocateTime(ms),loadTime(ms),calcTime(ms),loadTimeBack(ms),totalTime(ms),cpuTime(ms)\n";

    // 2^0 to 2^19 floats
    // each float is 4 bytes, so sizes range from 4B to ~2MB
    int sizes[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
        1048576,    // 4 MB
        2097152,    // 8 MB
        4194304,    // 16 MB
        8388608,    // 32 MB
        16777216,   // 64 MB
        33554432,   // 128 MB
        67108864,   // 256 MB
        134217728,  // 512 MB
        268435456,  // 1 GB
        536870912,  // 2 GB
        1073741824  // 4 GB
    };

    // Warm up CUDA, allocate and free a small array to initialize the GPU
    // and avoid measuring initialization overhead in timing
    float *dummy_a, *dummy_b, *dummy_c;
    cudaMalloc(&dummy_a, sizeof(float));
    cudaMalloc(&dummy_b, sizeof(float));
    cudaMalloc(&dummy_c, sizeof(float));
    vectorAdd<<<1, 1>>>(dummy_a, dummy_b, dummy_c, 1);
    cudaFree(dummy_a);
    cudaFree(dummy_b);
    cudaFree(dummy_c);

    for (int s = 0; s < sizeof(sizes)/sizeof(sizes[0]); ++s) {
        int N = sizes[s];
        std::vector<float> h_a(N), h_b(N), h_c(N), h_c_cpu(N);

        for (int i = 0; i < N; ++i) {
            h_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
            h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // CPU calculation timing
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            h_c_cpu[i] = h_a[i] + h_b[i];
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpuTime = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // GPU calcualtion timing
        auto gpu_start = std::chrono::high_resolution_clock::now();

        // Create CUDA events for timing
        cudaEvent_t startAllocate, stopAllocate, startLoad, stopLoad, startCalc, stopCalc;
        cudaEventCreate(&startAllocate);
        cudaEventCreate(&stopAllocate);
        cudaEventCreate(&startLoad);
        cudaEventCreate(&stopLoad);
        cudaEventCreate(&startCalc);
        cudaEventCreate(&stopCalc);

        // Allocate device memory
        float *d_a, *d_b, *d_c;
        cudaEventRecord(startAllocate);
        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_c, N * sizeof(float));
        cudaEventRecord(stopAllocate);
        cudaEventSynchronize(stopAllocate);

        float allocateTime = 0;
        cudaEventElapsedTime(&allocateTime, startAllocate, stopAllocate);
    
        // Measure loading time (host to device + device to host)
        cudaEventRecord(startLoad);
        cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(stopLoad);
        cudaEventSynchronize(stopLoad);

        float loadTime = 0;
        cudaEventElapsedTime(&loadTime, startLoad, stopLoad);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Measure calculation time (kernel execution)
        cudaEventRecord(startCalc);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stopCalc);
        cudaEventSynchronize(stopCalc);

        float calcTime = 0;
        cudaEventElapsedTime(&calcTime, startCalc, stopCalc);

        // Measure loading time for device to host
        cudaEventRecord(startLoad);
        cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stopLoad);
        cudaEventSynchronize(stopLoad);

        float loadTimeBack = 0;
        cudaEventElapsedTime(&loadTimeBack, startLoad, stopLoad);

        // Total loading time = H2D + D2H
        float totalTime = allocateTime + loadTime + calcTime + loadTimeBack;
        float sizeMB = N * sizeof(float) / (1024.0 * 1024.0);

        csv << N << "," << sizeMB << "," << allocateTime << "," << loadTime << "," << calcTime << "," << loadTimeBack << "," << totalTime << "," << cpuTime << "\n";

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(startLoad);
        cudaEventDestroy(stopLoad);
        cudaEventDestroy(startCalc);
        cudaEventDestroy(stopCalc);
    }

    csv.close();
    std::cout << "Timing data saved to timings.csv\n";
    return 0;
}

// nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add