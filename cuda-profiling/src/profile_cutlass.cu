#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>

#include "cuda_check.cuh"
#include "profiling.h"

#define WARMUP 3
#define REPEAT 3

// CUTLASS version using FP16 Tensor Cores; A/B are narrowed to half precision on host before upload.
void profile_cutlass_fp16(int S, const float* h_A, const float* h_B)
{
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

    // Narrow the float operands to half on the host (outside the profiled region).
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

    {
        NvtxRange range("cutlass_fp16");
        for (int i = 0; i < REPEAT; ++i)
            gemm_op(args, workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// CUTLASS version using TF32 Tensor Cores; A/B are bitcast (still 4 bytes), not narrowed.
void profile_cutlass_tf32(int S, const float* h_A, const float* h_B)
{
    // tfloat32_t is bit-identical width to float (4 bytes); float inputs are
    // bitcast, not copied, which truncates the mantissa to 10 bits for the MMA.
    // Explicit tile shapes are required: CUTLASS's generic Sm80/tf32 default
    // config hits a broken epilogue dispatch (FragmentIteratorComplexTensorOp)
    // in this CUTLASS version.
    using CutlassGemmTF32 = cutlass::gemm::device::Gemm<
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

    size_t n = (size_t)S * S;
    size_t sz = n * sizeof(float);

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

    CutlassGemmTF32 gemm_op;
    CutlassGemmTF32::Arguments args(
        {S, S, S},
        {tf32_A, S},
        {tf32_B, S},
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

    {
        NvtxRange range("cutlass_tf32");
        for (int i = 0; i < REPEAT; ++i)
            gemm_op(args, workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// CUTLASS version using regular fp32 CUDA cores (no precision narrowing, no tensor cores).
void profile_cutlass_fp32(int S, const float* h_A, const float* h_B)
{
    using CutlassGemmFP32 = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor, // A and row major
        float, cutlass::layout::RowMajor, // B and row major
        float, cutlass::layout::RowMajor, // C / D (full float output)
        float,                            // accumulator
        cutlass::arch::OpClassSimt,       // CUDA cores, not tensor cores
        cutlass::arch::Sm80               // Ampere architecture for our A100 GPU
    >;
    size_t n = (size_t)S * S;
    size_t sz = n * sizeof(float);

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

    {
        NvtxRange range("cutlass_fp32");
        for (int i = 0; i < REPEAT; ++i)
            gemm_op(args, workspace);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
