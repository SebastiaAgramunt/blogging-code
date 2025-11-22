#ifndef MATMUL_H
#define MATMUL_H

void matmul(const float* A, const float* B, float* C, std::size_t M, std::size_t N, std::size_t K);
void printmatrix(const float* A, int M, int N);

#endif