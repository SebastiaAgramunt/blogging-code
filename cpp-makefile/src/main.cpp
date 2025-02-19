#include <iostream>
#include "matmul.h"

int main(void){

    // A[M x N]
    int M = 2;
    int N = 3; 
    int* A = new int[M * N];

    for(int i=0; i < M * N; i++){
        A[i] = i;
    }
    std::cout << std::endl << "A:" << std::endl;
    printmatrix(A, M, N);

    // B[N x K]
    int K = 4;
    int* B = new int[N * K];

    for(int i=0; i < N * K; i++){
        B[i] = i;
    }
    std::cout << std::endl << "B:" << std::endl;
    printmatrix(B, N, K);

    // C[M x K]
    int* C = new int[M * K];
    matmul(A, B, C, M, N, K);

    std::cout << std::endl << "C = A x B: " << std::endl;
    printmatrix(C, M, K);

    return 0;
}
