#pragma once

#include <cstdlib>

// Utility macros and functions for the CUDA MMA example.
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Fill an array with uniform random floats in [-1, 1].
void fill_random(float* data, int n);

// CPU reference: C += A * B  (row-major, m×k × k×n → m×n).
void matmul_cpu(const float* A, const float* B, float* C, int m, int n, int k);

// Element-wise comparison with absolute + relative tolerance.
// Prints the first few mismatches to stderr; returns true if all match.
bool verify(const float* ref, const float* gpu, int total_elements,
            float atol = 1e-5f, float rtol = 1e-5f);
