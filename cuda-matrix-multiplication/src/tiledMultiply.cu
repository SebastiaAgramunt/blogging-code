#include "tiledMultiply.h"

__global__ void tiledMultiply(const float* __restrict__ A, // M x K
                                const float* __restrict__ B, // K x N
                                float* __restrict__ C,       // M x N
                                std::size_t M,
                                std::size_t K,
                                std::size_t N){

    const int row = blockIdx.y * TILE + threadIdx.y; // [0, M)
    const int col = blockIdx.x * TILE + threadIdx.x; // [0, N)

    // +1 padding to mitigate shared-memory bank conflicts
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    float acc = 0.0f;

    // Sweep across K in TILE-sized chunks
    for (int k0 = 0; k0 < static_cast<int>(K); k0 += TILE) {
        const int aCol = k0 + threadIdx.x; // along K
        const int bRow = k0 + threadIdx.y; // along K

        // Guard each cooperative load
        As[threadIdx.y][threadIdx.x] =
            (row < static_cast<int>(M) && aCol < static_cast<int>(K))
                ? A[row * K + aCol] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (bRow < static_cast<int>(K) && col < static_cast<int>(N))
                ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < static_cast<int>(M) && col < static_cast<int>(N)) {
        C[row * N + col] = acc;
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
