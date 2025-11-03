#include "tiledMultiply.h"

__global__ void tiledMultiply(const float* __restrict__ A, // M x K
                              const float* __restrict__ B, // K x N
                              float* __restrict__ C,       // M x N
                              std::size_t M,
                              std::size_t K,
                              std::size_t N) {

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // global row/col this thread is responsible for
    int i = by * TILE + ty;  // row in C
    int j = bx * TILE + tx;  // col in C

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float value = 0.0f;

    // number of tiles along K
    int numTiles = (K + TILE - 1) / TILE;

    for (int ph = 0; ph < numTiles; ++ph) {
        // column in A, row in B that this thread wants to load
        int aCol = ph * TILE + tx;  // along K
        int bRow = ph * TILE + ty;  // along K

        // load A tile (row = i, col = aCol)
        if (i < M && aCol < K)
            As[ty][tx] = A[i * K + aCol];
        else
            As[ty][tx] = 0.0f;

        // load B tile (row = bRow, col = j)
        if (bRow < K && j < N)
            Bs[ty][tx] = B[bRow * N + j];
        else
            Bs[ty][tx] = 0.0f;

        // sync all threads to make sure the tiles are loaded
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            value += As[ty][t] * Bs[t][tx];
        }

        // sync before loading the next tile
        __syncthreads();
    }

    // write back only if in-bounds
    if (i < (int)M && j < (int)N) {
        C[i * N + j] = value;
    }
}

void tiledMultiply_call(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N){
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiledMultiply<<<blocks, threads>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
