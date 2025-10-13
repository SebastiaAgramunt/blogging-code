
#include "vector_add.h"

// Addition kernel: Adds two vectors a, b and stores the result in c
// The parameter m is used to increase the computation time per thread
// by repeating the addition operation m times. This is useful for performance
// testing with different computation loads.
template<typename T>
__global__ void vectorAdd(
    const T* __restrict__ d_a,
    const T* __restrict__ d_b,
    T* __restrict__ d_c,
    int N,
    int m=1) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
        T s = d_a[idx] + d_b[idx];
        T acc = T(0);
        for (int j = 0; j < m -1; ++j) {
            acc = acc + s;
        }
        d_c[idx] = acc;
}

// Wrapper to the cuda kernel call
template<typename T>
void vectorAdd_wrapper(
    const T* __restrict__ d_a,
    const T* __restrict__ d_b,
    T* __restrict__ d_c,
    int N,
    int m,
    int ThreadsPerBlock){
    
    int blocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    vectorAdd<<<blocksPerGrid, ThreadsPerBlock>>>(d_a, d_b, d_c, N, m);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 
}

template void vectorAdd_wrapper<float>(const float*, const float*, float*, int, int, int);
template void vectorAdd_wrapper<double>(const double*, const double*, double*, int, int, int);


// Function to perform a dummy calculation to warm up the GPU
// This is only used to boot up the GPU and avoid measuring initialization overhead in timing
template<typename T>
void dummyCalculation(){
    T *dummy_a, *dummy_b, *dummy_c;
    CHECK_CUDA_ERROR(cudaMalloc(&dummy_a, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dummy_b, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&dummy_c, sizeof(T)));
    vectorAdd<T><<<1, 1>>>(dummy_a, dummy_b, dummy_c, 1);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaFree(dummy_a));
    CHECK_CUDA_ERROR(cudaFree(dummy_b));
    CHECK_CUDA_ERROR(cudaFree(dummy_c));
}

template void dummyCalculation<int>();
template void dummyCalculation<float>();
template void dummyCalculation<double>();