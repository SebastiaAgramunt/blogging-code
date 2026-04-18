#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "utils.h"

void fill_random(float* p, int n) {
    for (int i = 0; i < n; ++i)
        p[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

void matmul_cpu(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i)
                acc += A[row * k + i] * B[i * n + col];
            C[row * n + col] += acc;
        }
}

bool verify(const float* ref, const float* gpu, int total_elements,
            float atol, float rtol) {
    int mismatches = 0;
    for (int i = 0; i < total_elements; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        float tol  = atol + rtol * fabsf(ref[i]);
        if (diff > tol) {
            if (mismatches < 5)
                fprintf(stderr, "  Mismatch at [%d]: ref=%.6f  gpu=%.6f  diff=%.2e\n",
                        i, ref[i], gpu[i], diff);
            ++mismatches;
        }
    }
    if (mismatches > 0)
        fprintf(stderr, "  Total mismatches: %d / %d\n", mismatches, total_elements);
    return mismatches == 0;
}
