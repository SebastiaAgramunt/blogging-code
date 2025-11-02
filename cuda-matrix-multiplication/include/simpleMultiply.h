#ifndef SIMPLEMULTIPLY_H
#define SIMPLEMULTIPLY_H

#include <iostream>
#include <cuda_runtime.h>

void simpleMatrixMultiplication_call(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, const size_t M, const size_t K, const size_t N);

void simpleMatrixMultiplication_cpp(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, const size_t M, const size_t K, const size_t N);

#endif
