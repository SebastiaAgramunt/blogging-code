// cuda_check.cuh
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ── Host-side API checks ──────────────────────────────────────────────
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
      fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n  → %s\n",              \
              __FILE__, __LINE__, #call,                               \
              cudaGetErrorString(_e));                                 \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

// ── Kernel launch checks ──────────────────────────────────────────────
#define CHECK_LAST_ERROR()                                             \
  do {                                                                 \
    cudaError_t _e = cudaGetLastError();                               \
    if (_e != cudaSuccess) {                                           \
      fprintf(stderr, "[KERNEL LAUNCH ERROR] %s:%d  → %s\n",           \
              __FILE__, __LINE__, cudaGetErrorString(_e));             \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

// ── Post-kernel execution checks ──────────────────────────────────────
#define CHECK_SYNC()                                                   \
  do {                                                                 \
    cudaError_t _e = cudaDeviceSynchronize();                          \
    if (_e != cudaSuccess) {                                           \
      fprintf(stderr, "[KERNEL EXEC ERROR] %s:%d  → %s\n",             \
              __FILE__, __LINE__, cudaGetErrorString(_e));             \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)
