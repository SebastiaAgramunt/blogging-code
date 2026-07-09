#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>

#include "cuda_check.cuh"
#include "kernels.cuh"
#include "profiling.h"
#include "utils.h"

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", _s,                \
                    __FILE__, __LINE__);                                     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Untimed launches to warm up clocks/caches before the profiled region.
#define WARMUP 3
// Launches inside the NVTX range, so the timeline/report shows more than
// one occurrence. `ncu` defaults to profiling all of them; pass
// `--launch-count 1` (done by the Makefile) to only profile the first.
#define REPEAT 3

void profile_naive(int S, const float* h_A, const float* h_B)
{
    size_t sz = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    dim3 gridDim(CEIL_DIV(S, 32), CEIL_DIV(S, 32));
    dim3 blockDim(32, 32, 1);

    for (int i = 0; i < WARMUP; ++i)
        sgemm_naive<<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        NvtxRange range("naive");
        for (int i = 0; i < REPEAT; ++i)
            sgemm_naive<<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void profile_coalesced(int S, const float* h_A, const float* h_B)
{
    size_t sz = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    dim3 gridDim(CEIL_DIV(S, 32), CEIL_DIV(S, 32));
    dim3 blockDim(32 * 32);

    for (int i = 0; i < WARMUP; ++i)
        sgemm_coalesced<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        NvtxRange range("coalesced");
        for (int i = 0; i < REPEAT; ++i)
            sgemm_coalesced<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void profile_tiled(int S, const float* h_A, const float* h_B)
{
    size_t sz = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(CEIL_DIV(S, blockDim.x), CEIL_DIV(S, blockDim.y));

    for (int i = 0; i < WARMUP; ++i)
        sgemm_tiled<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        NvtxRange range("tiled");
        for (int i = 0; i < REPEAT; ++i)
            sgemm_tiled<32><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void profile_coarsened(int S, const float* h_A, const float* h_B)
{
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;

    size_t sz = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    dim3 gridDim(CEIL_DIV(S, BN), CEIL_DIV(S, BM));
    dim3 blockDim(BM * BN / TM);   // 512 threads

    for (int i = 0; i < WARMUP; ++i)
        sgemm_coarsened<BM, BN, BK, TM><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        NvtxRange range("coarsened");
        for (int i = 0; i < REPEAT; ++i)
            sgemm_coarsened<BM, BN, BK, TM><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Persistent handle for cuBLAS
static cublasHandle_t cublas_handle()
{
    static cublasHandle_t h = []() {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
        return handle;
    }();
    return h;
}

void profile_cublas(int S, const float* h_A, const float* h_B)
{
    size_t sz = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    cublasHandle_t handle = cublas_handle();

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

    {
        NvtxRange range("cublas");
        for (int i = 0; i < REPEAT; ++i)
            CUBLAS_CHECK(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                S, S, S,
                &alpha,
                d_B, S,
                d_A, S,
                &beta,
                d_C, S));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

const ProfileEntry PROFILE_TABLE[] = {
    {"naive",       profile_naive},
    {"coalesced",   profile_coalesced},
    {"tiled",       profile_tiled},
    {"coarsened",   profile_coarsened},
    {"cublas",      profile_cublas},
    {"cutlass_fp32", profile_cutlass_fp32},
    {"cutlass_tf32", profile_cutlass_tf32},
    {"cutlass_fp16", profile_cutlass_fp16},
};
const int PROFILE_TABLE_SIZE = sizeof(PROFILE_TABLE) / sizeof(PROFILE_TABLE[0]);
