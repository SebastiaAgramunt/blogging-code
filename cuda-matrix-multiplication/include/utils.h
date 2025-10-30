#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <random>

void GenerateRandomMatrices(const size_t M, const size_t K, const size_t N, std::vector<float> &A,
    std::vector<float> &B);

void DummyAllocation();

#endif
