#include <iostream>
#include "matmul.h"

// Matrices are indexed row-major in this example. E.g. if A is [M x N]
// If i,j are the row and column indices, the element A[i, j] is
// A[i, j] = A[i * N + j] // if row-index
// A[i, j] = A[j * M + i] // if column-index

void matmul(const int* A, const int* B, int* C, int M, int N, int K){
// Matrix multiplication, C[M x K] = A[M x N] * B[N x K]
// Multiplication is $\sum_n A[m, n] * B[n, k]$
    for(int m=0; m<M; m++){
        for(int k=0; k<K; k++){
            C[m * K + k] = 0;
            for(int n=0; n<N; n++){
                C[m * K + k] += A[m * N + n] * B[n * K + k];
            }
        }
    }
}

void printmatrix(const int* A, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}
