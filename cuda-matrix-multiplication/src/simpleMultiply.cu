#include "simpleMultiply.h"

__global__ void simpleMatrixMultiplication(const float* __restrict__ A,
    const float* __restrict__ B, float* __restrict__ C, const size_t M, const size_t K,
    const size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


void simpleMatrixMultiplication_call(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, const size_t M, const size_t K, const size_t N) {
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    simpleMatrixMultiplication<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}


void simpleMatrixMultiplication_cpp(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, const size_t M, const size_t K, const size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}