#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
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
    csv << "N,LoadingTime(ms),CalculationTime(ms)\n";

    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576}; // Example sizes

    for (int s = 0; s < sizeof(sizes)/sizeof(sizes[0]); ++s) {
        int N = sizes[s];
        std::vector<float> h_a(N), h_b(N), h_c(N);

        for (int i = 0; i < N; ++i) {
            h_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
            h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_c, N * sizeof(float));

        cudaEvent_t startLoad, stopLoad, startCalc, stopCalc;
        cudaEventCreate(&startLoad);
        cudaEventCreate(&stopLoad);
        cudaEventCreate(&startCalc);
        cudaEventCreate(&stopCalc);

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
        float totalLoadTime = loadTime + loadTimeBack;

        csv << N << "," << totalLoadTime << "," << calcTime << "\n";

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