#include <iostream>
#include "matmul.h"

// Matrices are indexed row-major in this example. E.g. if A is [M x N]
// If i,j are the row and column indices, the element A[i, j] is
// A[i, j] = A[i * N + j] // if row-major
// A[i, j] = A[j * M + i] // if column-major

void matmul(const float* A,
            const float* B,
            float* C,
            std::size_t M,
            std::size_t N,
            std::size_t K) {
    // C[M x K] = A[M x N] * B[N x K]
    // C[m, k] = sum_n A[m, n] * B[n, k]

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t k = 0; k < K; ++k) {
            float acc = 0.0f;
            for (std::size_t n = 0; n < N; ++n) {
                acc += A[m * N + n] * B[n * K + k];
            }
            C[m * K + k] = acc;
        }
    }
}


void printmatrix(const float* A, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}