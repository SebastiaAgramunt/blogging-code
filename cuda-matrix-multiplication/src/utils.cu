#include "utils.h"

void GenerateRandomMatrices(const size_t M, const size_t K, const size_t N, std::vector<float> &A,
    std::vector<float> &B){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < M * K; ++i) {
        A[i] = dis(gen);
    }
    for (size_t i = 0; i < K * N; ++i) {
        B[i] = dis(gen);
    }
}

void DummyAllocation(){
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 1024 * 1024 * sizeof(float));
    cudaMalloc(&d_B, 1024 * 1024 * sizeof(float));
    cudaMalloc(&d_C, 1024 * 1024 * sizeof(float));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
