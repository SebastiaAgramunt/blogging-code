#pragma once

#include <cuda_runtime.h>


__global__ void sgemm_naive(
    size_t M,             // Number of rows in A and C
    size_t N,             // Number of columns in B and C
    size_t K,             // Number of columns in A and rows in B
    float alpha,       // Scaling factor for the product of A and B
    const float *A,    // [M x K] row-major
    const float *B,    // [K x N] row-major
    float beta,        // Scaling factor for C
    float *C);         // [M x N] row-major  (in-out: C = alpha*A*B + beta*C)


template <int BLOCKSIZE>
__global__ void sgemm_coalesced(
    size_t M,             // Number of rows in A and C
    size_t N,             // Number of columns in B and C
    size_t K,             // Number of columns in A and rows in B
    float alpha,       // Scaling factor for the product of A and B
    const float *A,    // [M x K] row-major
    const float *B,    // [K x N] row-major
    float beta,        // Scaling factor for C
    float *C);         // [M x N] row-major  (in-out: C = alpha*A*B + beta*C)


template <int TILE_SIZE>
__global__ void sgemm_tiled(
    size_t M,             // Number of rows in A and C
    size_t N,             // Number of columns in B and C
    size_t K,             // Number of columns in A and rows in B
    float alpha,       // Scaling factor for the product of A and B
    const float *A,    // [M x K] row-major
    const float *B,    // [K x N] row-major
    float beta,        // Scaling factor for C
    float *C);         // [M x N] row-major  (in-out: C = alpha*A*B + beta*C)


// Thread-coarsened tiling: each thread computes TM consecutive rows of one C column.
// Block tile BM×BN, K-tile BK. Threads per block: BM*BN/TM.
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_coarsened(
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float *A,    // [M x K] row-major
    const float *B,    // [K x N] row-major
    float beta,
    float *C);         // [M x N] row-major  (in-out: C = alpha*A*B + beta*C)