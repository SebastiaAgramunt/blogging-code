#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "simpleMultiply.h"
#include "tiledMultiply.h"
#include "cuBLASMultiply.h"
#include "utils.h"


int main(int argc, char **argv) {

    // Matrix dimensions
    // A[M, K], B[K, N], C[M, N]
    size_t M, K, N;
    std::string filename;
    bool use_cpu = true;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matrix_size" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        }
        if (arg == "--output_file" && i + 1 < argc) {
            filename = argv[++i];
        }
        if (arg == "--no_cpu" && i + 1 < argc) {
            use_cpu = false;
        }
        if (arg == "--help") {
            std::cerr << "Usage: " << argv[0] << " --matrix_size <int> --output_file <string> --no_cpu\n";
            return 1;
        }
    }

    // square matrices
    K = M;
    N = M;

    // writing on csv file
    bool new_file = !std::filesystem::exists(filename);
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        return 1;
    }

    // If the file was just created, write the first line
    if (new_file) {
        file << "M,K,N,memory_usage_MB,allocation_time,copy_time,compute_time_gpu,compute_time_cpu,copy_time_back,free_time,total_time_gpu,total_time_gpu_tiled,total_time_gpu_cublas\n";
    }
    file.close();

    std::cout << "Calculating for M = " << M << " K = " << K << " N = " << N << std::endl;

    // Host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_cpu(M * N);

    // stream to use in CuBLAS
    cudaStream_t stream = 0;

    // Fill A and B with random values
    GenerateRandomMatrices(M, K, N, h_A, h_B);

    // Calculate time for CPU multiplication
    auto start = std::chrono::high_resolution_clock::now();
    if (use_cpu) simpleMatrixMultiplication_cpp(h_A.data(), h_B.data(), h_C_cpu.data(), M, K, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> compute_time_cpu = end - start;

    // Dummy allocation to avoid first allocation overhead
    // calculate time first allocation
    DummyAllocation();

    // Allocate GPU resources
    float *d_A, *d_B, *d_C;
    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> allocation_time = end - start;

    // copy A and B to GPU
    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copy_time = end - start;

    // Multiply matrices
    // simple
    start = std::chrono::high_resolution_clock::now();
    simpleMatrixMultiplication_call(d_A, d_B, d_C, M, K, N);
    CHECK_LAST_CUDA_ERROR();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> compute_time_gpu = end - start;

    // tiled kernel
    start = std::chrono::high_resolution_clock::now();
    tiledMultiply_call(d_A, d_B, d_C, N, K, M);
    CHECK_LAST_CUDA_ERROR();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> compute_time_gpu_tiled = end - start;

    // cublas function
    start = std::chrono::high_resolution_clock::now();
    cuBLASmultiply_call(d_A, d_B, d_C, N, K, M, stream);
    CHECK_LAST_CUDA_ERROR();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> compute_time_gpu_cublas = end - start;

    // copy C from GPU to host
    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copy_back_time = end - start;

    // Free resources
    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> free_time = end - start;

    // check cpu vs gpu results for cuBLAS, the others have been tested manually
    if (use_cpu) {
        for (size_t i = 0; i < h_C.size(); ++i) {
            if (std::abs(h_C[i] - h_C_cpu[i]) > 1e-3) {
                std::cerr << "Error: results do not match at index " << i << " " << h_C[i] << " " << h_C_cpu[i] << std::endl;
                break;
            }
        }
    }

    // write times to csv file
    file.open(filename, std::ios::app);
    size_t total_memory = ( h_A.size() * sizeof(float) + h_B.size() * sizeof(float) + h_C.size() * sizeof(float) ) / 1024 / 1024;
    auto total_time_gpu = allocation_time + copy_time + compute_time_gpu + copy_back_time + free_time;
    auto total_time_gpu_tiled = allocation_time + copy_time + compute_time_gpu_tiled + copy_back_time + free_time;
    auto total_time_cublas = allocation_time + copy_time + compute_time_gpu_cublas + copy_back_time + free_time;
    file << M << ','
     << K << ','
     << N << ','
     << total_memory << ','                  // still an integer (bytes), that’s fine
     << allocation_time.count() << ','
     << copy_time.count() << ','
     << compute_time_gpu.count() << ','
     << compute_time_cpu.count() << ','
     << copy_back_time.count() << ','
     << free_time.count() << ','
     << total_time_gpu.count() << ',' 
     << compute_time_gpu_tiled.count() << ','
     << compute_time_gpu_cublas.count() << '\n';
    file.close();

    return 0;
}
