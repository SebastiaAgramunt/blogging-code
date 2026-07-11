#pragma once

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Fill an array with uniform random floats in [-1, 1].
void fill_random(float* data, int n);
