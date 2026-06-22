#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>

#include "cuda_check.cuh"
#include "timer.h"
#include "utils.h"
#include "benchmarks.h"

#define WARMUP 3
#define ITERS  5


float benchmark_cutlass_fp16(int S, const float* h_A, const float* h_B)
{
    // SGEMM via CUTLASS using FP16 Tensor Cores on Ampere (sm_80).
    using CutlassGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,   // A (fp16 half precision) and row major
        cutlass::half_t, cutlass::layout::RowMajor,   // B (fp16 half precision) and row major
        float,           cutlass::layout::RowMajor,   // C / D (full float output)
        float,                                        // accumulator
        cutlass::arch::OpClassTensorOp,               // FP16 tensor cores
        cutlass::arch::Sm80                           // Ampere architecture for our A100 GPU
    >;

    size_t n = (size_t)S * S;
    size_t szAB = n * sizeof(cutlass::half_t);
    size_t szC  = n * sizeof(float);

    // Narrow the float operands to half on the host (outside the timed region).
    std::vector<cutlass::half_t> h_A_half(n), h_B_half(n);
    for (size_t i = 0; i < n; ++i) {
        h_A_half[i] = cutlass::half_t(h_A[i]);
        h_B_half[i] = cutlass::half_t(h_B[i]);
    }

    cutlass::half_t *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, szAB));
    CUDA_CHECK(cudaMalloc(&d_B, szAB));
    CUDA_CHECK(cudaMalloc(&d_C, szC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_half.data(), szAB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_half.data(), szAB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, szC));

    const float alpha = 1.0f, beta = 0.0f;

    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {S, S, S},          // problem size M, N, K
        {d_A, S},           // A ref: pointer + leading dim
        {d_B, S},           // B ref
        {d_C, S},           // C ref (source for beta*C)
        {d_C, S},           // D ref (output destination)
        {alpha, beta}       // epilogue: alpha*A*B + beta*C
    );

    // Allocate workspace (zero for split_k_slices=1)
    size_t workspace_bytes = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_bytes)
        CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    for (int i = 0; i < WARMUP; ++i) {
        cutlass::Status s = gemm_op(args, workspace);
        if (s != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS error %d\n", (int)s);
            exit(EXIT_FAILURE);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        gemm_op(args, workspace);
        total_ms += timer.stop();
    }

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}

float benchmark_cutlass_fp32(int S, const float* h_A, const float* h_B)
{
    using CutlassGemmFP32 = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,   // CUDA cores, not tensor cores
        cutlass::arch::Sm80
    >;
    size_t n = (size_t)S * S;
    size_t sz = n * sizeof(float);

    // No precision narrowing needed — upload float directly.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    const float alpha = 1.0f, beta = 0.0f;

    CutlassGemmFP32 gemm_op;
    CutlassGemmFP32::Arguments args(
        {S, S, S},
        {d_A, S},
        {d_B, S},
        {d_C, S},
        {d_C, S},
        {alpha, beta}
    );

    size_t workspace_bytes = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_bytes)
        CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    for (int i = 0; i < WARMUP; ++i) {
        cutlass::Status s = gemm_op(args, workspace);
        if (s != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS error %d\n", (int)s);
            exit(EXIT_FAILURE);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, sz));
        timer.start();
        gemm_op(args, workspace);
        total_ms += timer.stop();
    }

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}