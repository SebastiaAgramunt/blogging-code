#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"
#include <cuda_runtime.h>

// ─── Benchmark kernel ────────────────────────────────────────────────────────
//
// Each thread:
//   1. Loads a[i] and b[i]  →  2 × sizeof(float) = 8 bytes read
//   2. Computes val = a[i] + b[i]                  (1 FLOP)
//   3. Runs m FMA iterations: val = val * alpha + beta  (2 FLOPs each)
//   4. Stores c[i]           →  1 × sizeof(float) = 4 bytes written
//
// Totals per element
//   Bytes:  3 × 4 = 12
//   FLOPs:  1 + 2·m
//   Arithmetic intensity:  (1 + 2·m) / 12  [FLOPs / byte]
//
// The GPU hides FMA latency (~4 cycles) through inter-warp parallelism, so for
// large N the kernel will approach peak FP32 throughput once AI exceeds the
// ridge point.
void vector_add_fma(
    const float* d_a, const float* d_b, float* d_c,
    int N, int m, int threads_per_block,
    cudaEvent_t start, cudaEvent_t stop, float& elapsed_ms);

// ─── Peak-bandwidth kernel ────────────────────────────────────────────────────
//
// Pure memory copy: c[i] = a[i]
//   Bytes:  2 × 4 = 8 per element
//   FLOPs:  0  →  arithmetic intensity ≈ 0
//
// Used to measure empirical peak HBM / GDDR bandwidth.
void bandwidth_test(
    const float* d_a, float* d_b,
    int N, int threads_per_block,
    cudaEvent_t start, cudaEvent_t stop, float& elapsed_ms);

// GPU warm-up (first kernel launch incurs driver-init overhead)
void warmup();

#endif
