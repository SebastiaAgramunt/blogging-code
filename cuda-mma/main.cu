#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "cuda_check.cuh"
#include "kernels.cuh"
#include "utils.h"
#include "timer.h"

// ---------------------------------------------------------------------------
// Benchmark config
// ---------------------------------------------------------------------------
#define WARMUP 3
#define ITERS  5

// ---------------------------------------------------------------------------
// Benchmark helper
// Allocates device memory for a square S×S problem, runs the naive kernel
// WARMUP+ITERS times, and returns the average elapsed milliseconds.
// Host data is filled once at startup; each timed run resets d_C to zero so
// that C += A*B doesn't accumulate across iterations.
// ---------------------------------------------------------------------------
static float benchmark_naive(int S,
                              const float* h_A,
                              const float* h_B)
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

    // Warmup — not measured.
    for (int i = 0; i < WARMUP; ++i)
        sgemm_naive<<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs.
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

static float benchmark_tiled(int S,
                            const float* h_A,
                            const float* h_B)
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
    dim3 gridDim(CEIL_DIV(S, blockDim.x / 32), CEIL_DIV(S, blockDim.y / 32));

    // Warmup — not measured.
    for (int i = 0; i < WARMUP; ++i)
        sgemm_tiled<16><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs.
    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        sgemm_tiled<16><<<gridDim, blockDim>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
        total_ms += timer.stop();
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return total_ms / ITERS;
}


void benchmark_coalesced(int S,
                            const float* h_A,
                            const float* h_B)
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

    // Warmup — not measured.
    for (int i = 0; i < WARMUP; ++i)
        sgemm_coalesced<16><<<grid, block>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs.
    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_C, 0, szC));
        timer.start();
        total_ms += timer.stop();
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("Coalesced kernel: %9.3f ms\n", total_ms / ITERS);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    srand(42);

    // // ── Correctness check (small fixed size) ────────────────────────────────
    // // The CPU triple-loop is O(S^3), so we only run it for a small S.
    // const int S_verify = 256;
    // {
    //     float* h_A     = new float[S_verify * S_verify];
    //     float* h_B     = new float[S_verify * S_verify];
    //     float* h_C     = new float[S_verify * S_verify]();  // zero-init
    //     float* h_C_ref = new float[S_verify * S_verify]();

    //     fill_random(h_A, S_verify * S_verify);
    //     fill_random(h_B, S_verify * S_verify);
    //     matmul_cpu(h_A, h_B, h_C_ref, S_verify, S_verify, S_verify);

    //     float *d_A, *d_B, *d_C;
    //     size_t sz = (size_t)S_verify * S_verify * sizeof(float);
    //     CUDA_CHECK(cudaMalloc(&d_A, sz));
    //     CUDA_CHECK(cudaMalloc(&d_B, sz));
    //     CUDA_CHECK(cudaMalloc(&d_C, sz));
    //     CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemset(d_C, 0, sz));

    //     constexpr int BS = 32;
    //     dim3 block(BS * BS);   // BLOCKSIZE² threads: threadIdx.x / BS = row, threadIdx.x % BS = col
    //     dim3 grid((S_verify + BS - 1) / BS, (S_verify + BS - 1) / BS);
    //     sgemm_coalesced<BS><<<grid, block>>>(S_verify, S_verify, S_verify, 1.0f, d_A, d_B, 0.0f, d_C);
    //     CHECK_LAST_ERROR();
    //     CHECK_SYNC();
    //     CUDA_CHECK(cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost));

    //     printf("Correctness check (S=%d): ", S_verify);
    //     if (verify(h_C_ref, h_C, S_verify * S_verify))
    //         printf("PASSED\n\n");
    //     else
    //         printf("FAILED\n\n");

    //     CUDA_CHECK(cudaFree(d_A));
    //     CUDA_CHECK(cudaFree(d_B));
    //     CUDA_CHECK(cudaFree(d_C));
    //     delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    // }


    const int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    const int N_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    // Allocate host matrices for the largest size once.
    const int S_max = sizes[N_SIZES - 1];
    float* h_A = new float[(size_t)S_max * S_max];
    float* h_B = new float[(size_t)S_max * S_max];
    fill_random(h_A, S_max * S_max);
    fill_random(h_B, S_max * S_max);

    printf("Naive kernel roofline sweep (%d iters, %d warmup)\n", ITERS, WARMUP);
    printf("%-6s  %9s  %9s  %11s  %9s\n",
           "Size", "Time(ms)", "GFLOP/s", "BW(GB/s)", "AI(F/B)");
    printf("------  ---------  ---------  -----------  ---------\n");

    for (int i = 0; i < N_SIZES; ++i) {
        int S = sizes[i];

        float avg_ms = benchmark_naive(S, h_A, h_B);
        float avg_ms_tiled = benchmark_tiled(S, h_A, h_B);

        // Each term is promoted to double before multiplying to prevent int32
        // overflow at large S (see note above).
        double flops = 2.0 * S * S * S;
        double bytes = ((double)S * S        // read A
                      + (double)S * S        // read B
                      + 2.0 * S * S)         // read + write C
                     * sizeof(float);
        double ai        = flops / bytes;
        double gflops    = flops / (avg_ms * 1e-3) / 1e9;
        double bandwidth = bytes / (avg_ms * 1e-3) / 1e9;

        printf("%-6d  %9.3f  %9.1f  %11.1f  %9.2f\n",
               S, avg_ms, gflops, bandwidth, ai);

        // printf("  Tiled kernel: %9.3f ms\n", avg_ms_tiled);
    }

    delete[] h_A;
    delete[] h_B;

    return 0;
}
