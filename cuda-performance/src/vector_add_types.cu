#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <filesystem>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Addition kernel: Adds two vectors a, b and stores the result in c
// The parameter m is used to increase the computation time per thread
// by repeating the addition operation m times. This is useful for performance
// testing with different computation loads.
template<typename T>
__global__ void vectorAdd(const T* a, const T* b, T* c, int n, int m=1) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // this operation is silly, but read the function docs
        for(int j = 0; j < m; ++j) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

// Function to perform a dummy calculation to warm up the GPU
// This is only used to boot up the GPU and avoid measuring initialization overhead in timing
template<typename T>
void dummyCalculation(){
    T *dummy_a, *dummy_b, *dummy_c;
    CHECK_CUDA_ERROR(cudaMalloc(&dummy_a, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dummy_b, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dummy_c, sizeof(T)));
    vectorAdd<T><<<1, 1>>>(dummy_a, dummy_b, dummy_c, 1);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaFree(dummy_a));
    CHECK_CUDA_ERROR(cudaFree(dummy_b));
    CHECK_CUDA_ERROR(cudaFree(dummy_c));
}


std::ofstream create_csv(int threadsPerBlock,
                         int m,
                         const std::string& dtype) {
    namespace fs = std::filesystem;

    // Build filename
    std::string filename = "performance_time_threads_" +
                           std::to_string(threadsPerBlock) +
                           "_m_" + std::to_string(m) +
                           "_dtype_" + dtype + ".csv";

    // Define "outputs" directory in current working dir
    fs::path dir_path = fs::current_path() / "outputs";

    // Ensure directory exists
    if (!fs::exists(dir_path)) {
        if (!fs::create_directories(dir_path)) {
            std::cerr << "Error: could not create directory " << dir_path << "\n";
            std::exit(1);
        }
    }

    // Full path
    fs::path file_path = dir_path / filename;

    // Open file
    std::ofstream csv(file_path);
    if (!csv.is_open()) {
        std::cerr << "Error: could not open file " << file_path << "\n";
        std::exit(1);
    }

    // Write header
    csv << "N,Mb,allocateTime(ms),loadTime(ms),calcTime(ms),loadTimeBack(ms),"
           "totalTime(ms),cpuTime(ms)\n";

    return csv;
}



int main(int argc, char **argv){

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <threadsPerBlock> <m> <dtype>\n";
        std::cerr << "  dtype must be: int | float | double\n";
        return 1;
    }

    int threadsPerBlock = std::atoi(argv[1]);
    int m               = std::atoi(argv[2]);
    std::string dtype   = argv[3];

    using T = float;
    if (dtype == "int") {
        using T = int;
    } else if (dtype == "float") {
        using T = float;
    } else if (dtype == "double") {
        using T = double;
    } else {
        std::cerr << "Error: dtype must be one of: int | float | double\n";
        return 1;
    }

    std::srand(std::time(0));
    std::ofstream csv = create_csv(threadsPerBlock, m, dtype);

    // 2^0 to 2^19 elements of type T
    // if float or int, each element is 4 bytes
    // if double, each element is 8 bytes
    // commented out sizes for int or float
    long int sizes[] = {
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

    // dummy calculation to warm up the GPU
    dummyCalculation<int>();
    dummyCalculation<float>();
    dummyCalculation<double>();

    for (int s = 0; s < sizeof(sizes)/sizeof(sizes[0]); ++s) {
        long int N = sizes[s];

        // ceate host vectors and randomize
        std::vector<float> h_a(N), h_b(N), h_c(N), h_c_cpu(N);
        for (int i = 0; i < N; ++i) {
            h_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
            h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // CPU calculation timing
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            for(int j = 0; j < m; ++j){
                h_c_cpu[i] = h_a[i] + h_b[i];
            }
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
        T *d_a, *d_b, *d_c;
        cudaEventRecord(startAllocate);
        cudaMalloc(&d_a, N * sizeof(T));
        cudaMalloc(&d_b, N * sizeof(T));
        cudaMalloc(&d_c, N * sizeof(T));
        cudaEventRecord(stopAllocate);
        cudaEventSynchronize(stopAllocate);

        float allocateTime = 0;
        cudaEventElapsedTime(&allocateTime, startAllocate, stopAllocate);
    
        // Measure loading time (host to device + device to host)
        cudaEventRecord(startLoad);
        cudaMemcpy(d_a, h_a.data(), N * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), N * sizeof(T), cudaMemcpyHostToDevice);
        cudaEventRecord(stopLoad);
        cudaEventSynchronize(stopLoad);

        float loadTime = 0;
        cudaEventElapsedTime(&loadTime, startLoad, stopLoad);
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Measure calculation time (kernel execution)
        cudaEventRecord(startCalc);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, m);
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
        float sizeMB = N * sizeof(T) / (1024.0 * 1024.0);

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
    return 0;
}

// compile with:
// nvcc -O3 -arch=sm_80 -o vector_add_types vector_add_types.cu
// run with:
// ./vector_add_types  256 1 float