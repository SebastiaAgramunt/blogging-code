#include "kernels.h"

// ─────────────────────────────────────────────────────────────────────────────
// Vector-add + FMA benchmark kernel
//
// For small m  (low AI)  →  memory-bandwidth-bound
// For large m  (high AI) →  compute-throughput-bound
//
// Key: the GPU hides FMA latency (~4-cycle pipeline) by scheduling other warps
// while a given warp waits.  With enough active warps (large N), the throughput
// asymptotically approaches 2 FMAs/cycle even with a single accumulator chain
// per thread.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void vector_add_fma_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int N, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // One memory-bound load+add (1 FLOP, 8 bytes read)
    float val = a[idx] + b[idx];

    // m compute-intensive FMAs: val = val * alpha + beta  (2 FLOPs each)
    // alpha/beta chosen so val stays bounded for any starting value in [0,2]:
    //   val → converges to beta/(1-alpha) = 0.0001/(0.00001) = 10
    // #pragma unroll 1 prevents the compiler from unrolling (keeps m general)
    #pragma unroll 1
    for (int j = 0; j < m; ++j) {
        val = fmaf(val, 1.0f - 1e-5f, 1e-4f);
    }

    // One write (4 bytes written)
    c[idx] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure-copy kernel — measures empirical peak memory bandwidth
// ─────────────────────────────────────────────────────────────────────────────
__global__ void bandwidth_test_kernel(
    const float* __restrict__ a,
    float* __restrict__ b,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) b[idx] = a[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiny kernel just to force driver/context init before benchmarking
// ─────────────────────────────────────────────────────────────────────────────
__global__ void warmup_kernel(float* x, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) x[idx] = static_cast<float>(idx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Host wrappers
// ─────────────────────────────────────────────────────────────────────────────

void vector_add_fma(
    const float* d_a, const float* d_b, float* d_c,
    int N, int m, int threads_per_block,
    cudaEvent_t start, cudaEvent_t stop, float& elapsed_ms)
{
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vector_add_fma_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N, m);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));
}

void bandwidth_test(
    const float* d_a, float* d_b,
    int N, int threads_per_block,
    cudaEvent_t start, cudaEvent_t stop, float& elapsed_ms)
{
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    bandwidth_test_kernel<<<blocks, threads_per_block>>>(d_a, d_b, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));
}

void warmup()
{
    float* d;
    CHECK_CUDA_ERROR(cudaMalloc(&d, 4096 * sizeof(float)));
    warmup_kernel<<<16, 256>>>(d, 4096);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d));
}
