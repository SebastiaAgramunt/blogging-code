#pragma once

float benchmark_naive(int S, const float* h_A, const float* h_B);
float benchmark_tiled(int S, const float* h_A, const float* h_B);
float benchmark_coalesced(int S, const float* h_A, const float* h_B);
float benchmark_cublas(int S, const float* h_A, const float* h_B);
float benchmark_cblas(int S, const float* h_A, const float* h_B);
float benchmark_coarsened(int S, const float* h_A, const float* h_B);
float benchmark_cutlass_fp16(int S, const float* h_A, const float* h_B);
float benchmark_cutlass_fp32(int S, const float* h_A, const float* h_B);

