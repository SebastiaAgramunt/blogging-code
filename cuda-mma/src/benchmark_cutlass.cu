#include <cstdio>
#include <cstdlib>

#include <cutlass/gemm/device/gemm.h>

#include "cuda_check.cuh"
#include "timer.h"
#include "utils.h"
#include "benchmarks.h"

#define WARMUP 3
#define ITERS  5

// SGEMM via CUTLASS using TF32 Tensor Cores on Ampere (sm_80).
// CUTLASS 4.x requires tfloat32_t element types to select TF32 MMA instructions;
// float inputs are bitcast to tf32 precision (rounds the mantissa to 10 bits).
using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,   // A (tf32 ~ float with 10-bit mantissa)
    cutlass::tfloat32_t, cutlass::layout::RowMajor,   // B
    float,               cutlass::layout::RowMajor,   // C / D (full float output)
    float,                                            // accumulator
    cutlass::arch::OpClassTensorOp,                   // TF32 tensor cores
    cutlass::arch::Sm80,                              // Ampere
    cutlass::gemm::GemmShape<128, 128, 32>,           // threadblock tile
    cutlass::gemm::GemmShape<64,  64,  32>,           // warp tile
    cutlass::gemm::GemmShape<16,   8,   8>            // TF32 MMA instruction shape
>;

float benchmark_cutlass(int S, const float* h_A, const float* h_B)
{
    size_t sz = (size_t)S * S * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    const float alpha = 1.0f, beta = 0.0f;

    // Reinterpret float* as tfloat32_t* — same bit width, bitcast is intentional.
    auto* tf32_A = reinterpret_cast<cutlass::tfloat32_t*>(d_A);
    auto* tf32_B = reinterpret_cast<cutlass::tfloat32_t*>(d_B);

    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {S, S, S},          // problem size M, N, K
        {tf32_A, S},        // A ref: pointer + leading dim
        {tf32_B, S},        // B ref
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
