// cblas_example.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include <cblas.h>  // For OpenBLAS or Netlib CBLAS

int main() {
    // A (2x3), B (3x2) -> C (2x2)
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

    std::vector<double> C(M * N, 0.0); // 2x2

    // C := 1.0 * A * B + 0.0 * C
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0,
        A.data(), /*lda=*/K,   // for row-major, lda = #cols of A
        B.data(), /*ldb=*/N,   // for row-major, ldb = #cols of B
        0.0,
        C.data(), /*ldc=*/N    // for row-major, ldc = #cols of C
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

    // Also show a simple dot product: x·y
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {4, 5, 6};
    double dot = cblas_ddot(3, x.data(), 1, y.data(), 1);
    std::cout << "dot(x,y) = " << dot << " (expected 32)\n";
    return 0;
}