#include <cstdio>
#include <chrono>

#include <cblas.h>
#include <cublas_v2.h>
#include "cuda_check.cuh"
#include "kernels.cuh"
#include "timer.h"
#include "utils.h"
#include "benchmarks.h"

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t _s = (call);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", _s,                \
                    __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

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

    dim3 gridDim(CEIL_DIV(S, 32), CEIL_DIV(S, 32));
    dim3 blockDim(32 * 32);

    for (int i = 0; i < WARMUP; ++i)
        sgemm_coalesced<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        sgemm_coalesced<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        total_ms += timer.stop();
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}

float benchmark_cublas(int S, const float* h_A, const float* h_B)
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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS is column-major. For row-major C=A*B, use the identity
    // C^T = B^T * A^T, so pass B first with leading dimension S.
    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < WARMUP; ++i)
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            S, S, S,
            &alpha,
            d_B, S,
            d_A, S,
            &beta,
            d_C, S));
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            S, S, S,
            &alpha,
            d_B, S,
            d_A, S,
            &beta,
            d_C, S));
        total_ms += timer.stop();
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}

float benchmark_cblas(int S, const float* h_A, const float* h_B)
{
    size_t sz = (size_t)S * S;
    float* h_C = new float[sz]();

    using clock = std::chrono::high_resolution_clock;

    for (int i = 0; i < WARMUP; ++i)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    S, S, S, 1.0f, h_A, S, h_B, S, 0.0f, h_C, S);

    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        std::fill(h_C, h_C + sz, 0.0f);
        auto t0 = clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    S, S, S, 1.0f, h_A, S, h_B, S, 0.0f, h_C, S);
        auto t1 = clock::now();
        total_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }

    delete[] h_C;
    return total_ms / ITERS;
}
