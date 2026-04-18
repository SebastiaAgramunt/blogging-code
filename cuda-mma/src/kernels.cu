#include <cuda_runtime.h>

#include "kernels.cuh"

// Single Precision Matrix Multiplication Kernels SGEMM: C = alpha * A * B + beta * C


// Naive implementation: 1 thread per output element, no shared memory, non-coalesced accesses.
__global__ void sgemm_naive(
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float *A,              // [M x K] row-major
    const float *B,              // [K x N] row-major
    float beta, 
    float *C)                    // [M x N] row-major  (in-out: C = alpha*A*B + beta*C)
    {                   
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0 .. M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. N-1

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (size_t i = 0; i < K; ++i)
        acc += A[row * K + i] * B[i * N + col];

    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}


// Coalesced access version: 1 thread per output element, no shared memory, but coalesced accesses to A and B.
template <int BLOCKSIZE>
__global__ void sgemm_coalesced(
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (size_t i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

// Tiled version: 1 thread per output element, shared memory for tiles of A and B.
template <int TILE_SIZE>
__global__ void sgemm_tiled(
    size_t M,             // Number of rows in A and C
    size_t N,             // Number of columns in B and C
    size_t K,             // Number of columns in A and rows in B
    float alpha,       // Scaling factor for the product of A and B
    const float *A,    // [M x K] row-major
    const float *B,    // [K x N] row-major
    float beta,        // Scaling factor for C
    float *C)          // [M x N] row-major  (in-out: C = alpha*A*B + beta*C)
{
    // Declaring the tiles 
    __shared__ float A_tile[TILE_SIZE * TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE * TILE_SIZE];

    // the element of C for this specific thread
    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // we multiply along the K dimension, iterate over this
    size_t n_tiles = (K + TILE_SIZE -1)/TILE_SIZE;

    float acc = .0f;
    for(size_t tile=0; tile<n_tiles; tile++){

        // get tile col from A and tile row for B
        size_t aCol = tile * TILE_SIZE + threadIdx.x;
        size_t bRow = tile * TILE_SIZE + threadIdx.y;

        A_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        B_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            acc += A_tile[threadIdx.y * TILE_SIZE + i] * B_tile[i * TILE_SIZE + threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
}





































// // Tiled version: 1 thread per output element, shared memory for tiles of A and B.
// template <int TILE_SIZE>
// __global__ void sgemm_tiled(
//     size_t M,            // Number of rows in A and C
//     size_t N,             // Number of columns in B and C
//     size_t K,             // Number of columns in A and rows in B
//     float alpha,       // Scaling factor for the product of A and B
//     const float *A,    // [M x K] row-major
//     const float *B,    // [K x N] row-major
//     float beta,        // Scaling factor for C
//     float *C)         // [M x N] row-major  (in-out: C = alpha*A*B + beta*C)
// {
//     __shared__ float sA[TILE_SIZE][TILE_SIZE];
//     __shared__ float sB[TILE_SIZE][TILE_SIZE];

//     int row = blockIdx.y * TILE_SIZE + threadIdx.y;
//     int col = blockIdx.x * TILE_SIZE + threadIdx.x;

//     float acc = 0.0f;

//     // Iterate over tiles along the K dimension.
//     for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
//         size_t aCol = t * TILE_SIZE + threadIdx.x;  // column of A this thread loads
//         size_t bRow = t * TILE_SIZE + threadIdx.y;  // row    of B this thread loads

//         // Boundary-safe loads: pad with 0 for out-of-bounds tiles.
//         sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
//         sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
//         __syncthreads();

//         // Accumulate partial dot product from shared memory.
//         #pragma unroll
//         for (int i = 0; i < TILE_SIZE; ++i)
//             acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
//         __syncthreads();
//     }

//     if (row < M && col < N)
//         C[row * N + col] = alpha * acc + beta * C[row * N + col];
// }

template __global__ void sgemm_tiled<16>(size_t M, size_t N, size_t K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C);
template __global__ void sgemm_tiled<32>(size_t M, size_t N, size_t K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C);
template __global__ void sgemm_coalesced<16>(size_t M, size_t N, size_t K, float alpha,
                                              const float *A, const float *B,
                                              float beta, float *C);
template __global__ void sgemm_coalesced<32>(size_t M, size_t N, size_t K, float alpha,
                                              const float *A, const float *B,
                                              float beta, float *C);
