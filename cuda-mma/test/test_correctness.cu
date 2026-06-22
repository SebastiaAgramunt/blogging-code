#include <cstdio>
#include <cstdlib>

#include "cuda_check.cuh"
#include "kernels.cuh"
#include "utils.h"

int main() {
    srand(42);

    const int S = 256;
    float* h_A     = new float[S * S];
    float* h_B     = new float[S * S];
    float* h_C     = new float[S * S]();
    float* h_C_ref = new float[S * S]();

    fill_random(h_A, S * S);
    fill_random(h_B, S * S);
    matmul_cpu(h_A, h_B, h_C_ref, S, S, S);

    float *d_A, *d_B, *d_C;
    size_t sz = (size_t)S * S * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_C, sz));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sz));

    constexpr int BS = 32;
    dim3 block(BS, BS);
    dim3 grid((S + BS - 1) / BS, (S + BS - 1) / BS);
    sgemm_tiled<BS><<<grid, block>>>(S, S, S, 1.0f, d_A, d_B, 0.0f, d_C);
    CHECK_LAST_ERROR();
    CHECK_SYNC();
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost));

    printf("Correctness check (S=%d): ", S);
    if (verify(h_C_ref, h_C, S * S))
        printf("PASSED\n");
    else
        printf("FAILED\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;

    return 0;
}
