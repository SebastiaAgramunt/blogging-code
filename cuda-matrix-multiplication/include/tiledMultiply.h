#ifndef TILEDMULTIPLY_H
#define TILEDMULTIPLY_H

#include <iostream>
#include <cuda_runtime.h>

# define TILE 16
void tiledMultiply_call(const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const std::size_t M,
    const std::size_t K,
    const std::size_t N);

#endif