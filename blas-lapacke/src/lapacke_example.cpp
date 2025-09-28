#include <cstdio>
#include <vector>
#include <lapacke.h>

// Solve A x = b for x, overwriting b with the solution.
// Uses LAPACKE_dgesv (LU factorization with partial pivoting).
int main() {
    // Example 3x3 system
    // A =
    // [ 3  1  2 ]
    // [ 6  3  4 ]
    // [ 3  1  5 ]
    // b = [ 0, 1, 3 ]^T
    const int n = 3;          // order of A
    const int nrhs = 1;       // number of right-hand sides
    const int lda = n;        // leading dimension of A (row-major -> lda = n)
    const int ldb = nrhs;     // leading dimension of B (row-major -> ldb = nrhs)

    // Row-major storage (C style)
    std::vector<double> A = {
        3.0, 1.0, 2.0,
        6.0, 3.0, 4.0,
        3.0, 1.0, 5.0
    };
    std::vector<double> b = { 0.0, 1.0, 3.0 };

    // Pivot indices
    std::vector<lapack_int> ipiv(n);

    // Call LAPACKE (row-major)
    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR,
                                    n, nrhs,
                                    A.data(), lda,
                                    ipiv.data(),
                                    b.data(), ldb);
    if (info > 0) {
        std::fprintf(stderr, "U(%d,%d) is exactly zero; singular matrix.\n", info, info);
        return 1;
    } else if (info < 0) {
        std::fprintf(stderr, "Argument %d to dgesv had an illegal value.\n", -info);
        return 1;
    }

    // b now contains the solution x
    std::printf("Solution x:\n");
    for (int i = 0; i < n; ++i) {
        std::printf("x[%d] = %.10f\n", i, b[i]);
    }
    return 0;
}
