// cblas_example.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include <cblas.h>

int main() {

    // Simple dot product: x·y
    std::cout << "======== Example 1: Dot product ========" << std::endl;

    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {4, 5, 6};

    // BLAS double dot operation
    double dot = cblas_ddot(3, x.data(), 1, y.data(), 1);

    // output result.
    std::cout << "dot(x,y) = " << dot << " (expected 32)\n";


    std::cout << "======== Example 2: Matrix Multiplication ========" << std::endl;
    // Matrix multiplication
    // A (MxK), B (KxN) -> C (MxN)
    const int M = 2, K = 3, N = 2;

    // Row-major layout
    std::vector<double> A = {
        1, 2, 3,
        4, 5, 6
    }; // 2x3

    std::vector<double> B = {
         7,  8,
         9, 10,
        11, 12
    }; // 3x2

    // C: matrix of zeroes, we won't use it since we weant 
    // A (MxK), B (KxN) -> C (MxN)
    std::vector<double> C(M * N, 0.0); // 2x2

    // C := alpha * Op(A) * Op(B) + beta * C
    cblas_dgemm(
        CblasRowMajor,    // Matrix order
        CblasNoTrans,     // Transpose matrix A
        CblasNoTrans,     // Transpose matrix B
        M,                // number of rows of op(A) and C
        N,                // number of columns of op(B) and C
        K,                // number of columns of op(A) and rows of op(B)
        1.0,              // alpha
        A.data(),         // A
        K,                // for row-major, lda = #cols of A
        B.data(),         // B
        N,                // for row-major, ldb = #cols of B
        0.0,              // beta
        C.data(),         // C
        N                 // for row-major, ldc = #cols of C
    );

    std::cout << "C = A*B:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            std::cout << std::setw(6) << C[i * N + j] << " ";
        std::cout << "\n";
    }
    // Expected:
    // [ 58  64 ]
    // [139 154 ]

    return 0;
}