#include <iostream>
#include "matmul.h"

#define M 32
#define N 64
#define K 32

void initializeMatrices(float* A, float* B) {
    srand(7);

    for (int i = 0; i < M * N; ++i)
        A[i] = (rand() % 100) / 10.0f;  // Random float in range [0,10]

    for (int i = 0; i < N * K; ++i)
        B[i] = (rand() % 100) / 10.0f;  // Random float in range [0,10]
}

int main() {

    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    initializeMatrices(A, B);
    matmul(A, B, C, M, N, K);

    std::cout << "C = A x B:" << std::endl;
    printmatrix(C, M, K);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}