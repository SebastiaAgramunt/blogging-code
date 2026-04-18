#include <cstdio>

#include "cuda_check.cuh"
#include "kernels.cuh"
#include "timer.h"
#include "utils.h"
#include "benchmarks.h"

#define WARMUP 3
#define ITERS  5

float benchmark_naive(int S, const float* h_A, const float* h_B)
{
    size_t szA = (size_t)S * S * sizeof(float);
    size_t szB = (size_t)S * S * sizeof(float);
    size_t szC = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, szA));
    CUDA_CHECK(cudaMalloc(&d_B, szB));
    CUDA_CHECK(cudaMalloc(&d_C, szC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, szC));

    dim3 gridDim(CEIL_DIV(S, 32), CEIL_DIV(S, 32));
    dim3 blockDim(32, 32, 1);

    for (int i = 0; i < WARMUP; ++i)
        sgemm_naive<<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        sgemm_naive<<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        total_ms += timer.stop();
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}

float benchmark_tiled(int S, const float* h_A, const float* h_B)
{
    size_t szA = (size_t)S * S * sizeof(float);
    size_t szB = (size_t)S * S * sizeof(float);
    size_t szC = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, szA));
    CUDA_CHECK(cudaMalloc(&d_B, szB));
    CUDA_CHECK(cudaMalloc(&d_C, szC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, szC));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(CEIL_DIV(S, blockDim.x), CEIL_DIV(S, blockDim.y));

    for (int i = 0; i < WARMUP; ++i)
        sgemm_tiled<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        sgemm_tiled<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        total_ms += timer.stop();
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}

float benchmark_coalesced(int S, const float* h_A, const float* h_B)
{
    size_t szA = (size_t)S * S * sizeof(float);
    size_t szB = (size_t)S * S * sizeof(float);
    size_t szC = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, szA));
    CUDA_CHECK(cudaMalloc(&d_B, szB));
    CUDA_CHECK(cudaMalloc(&d_C, szC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, szC));

    dim3 block(32 * 32);
    dim3 grid((S + block.x / 16 - 1) / (block.x / 16),
              (S + block.x / 16 - 1) / (block.x / 16));

    for (int i = 0; i < WARMUP; ++i)
        sgemm_coalesced<16><<<grid, block>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        sgemm_coalesced<16><<<grid, block>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        total_ms += timer.stop();
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}
