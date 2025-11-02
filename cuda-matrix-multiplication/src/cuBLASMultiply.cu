

#include<cuBLASMultiply.h>

void cuBLASmultiply_call(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N,
                    cudaStream_t stream){

    
    cublasHandle_t handle;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // A: M x K (row-major)
    // B: K x N (row-major)
    // C: M x N (row-major)

    // We ask cuBLAS to compute: C^T = (B^T) * (A^T)
    CHECK_CUBLAS_ERROR(
        cublasSgemm(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            B, N,
            A, K,
            &beta,
            C, N)
        );
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}