#pragma once

float benchmark_naive(int S, const float* h_A, const float* h_B);
float benchmark_tiled(int S, const float* h_A, const float* h_B);
float benchmark_coalesced(int S, const float* h_A, const float* h_B);
