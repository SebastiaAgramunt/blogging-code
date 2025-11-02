#ifndef CUBLASMATMULTIPLY_H
#define CUBLASMATMULTIPLY_H

#include "utils.h"
#include<cublas_v2.h>
#include <cuda_runtime.h>


void cuBLASmultiply_call(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    std::size_t M,
                    std::size_t K,
                    std::size_t N,
                    cudaStream_t stream);

#endif