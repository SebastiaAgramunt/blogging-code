#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void GenerateRandomMatrices(const size_t M, const size_t K, const size_t N, std::vector<float> &A,
    std::vector<float> &B);

void DummyAllocation();

inline void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Converts cublasStatus_t to a human-readable string
inline const char* cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "<unknown>";
    }
}

// Error-checking function, same style as your CUDA one
inline void checkCublas(cublasStatus_t status,
                        const char* const func,
                        const char* const file,
                        const int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS error at: " << file << ":" << line << std::endl;
        std::cerr << cublasGetErrorString(status) << " in " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_CUDA_ERROR(expr)  (check((expr), #expr, __FILE__, __LINE__))
#define CHECK_LAST_CUDA_ERROR() (checkLast(__FILE__, __LINE__))
#define CHECK_CUBLAS_ERROR(expr) (checkCublas((expr), #expr, __FILE__, __LINE__))

#endif
