#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include "utils.h"

template <typename T>
void vectorAdd_wrapper(const T* __restrict__ d_a,
                       const T* __restrict__ d_b,
                       T* __restrict__ d_c,
                       int N,
                       int m = 1,
                       int ThreadsPerBlock = 256);

template <typename T>
void dummyCalculation();

#endif