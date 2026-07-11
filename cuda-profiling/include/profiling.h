#pragma once

#include <nvtx3/nvToolsExt.h>

// RAII NVTX range. Pushed/popped around the region you want visible as a
// named block in the Nsight Systems timeline, and selectable on its own
// with `ncu --nvtx-include "<name>/"` (see Makefile targets `nsys`/`ncu`).
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

// One profiling entry point per kernel implementation. Each function mallocs
// its own device buffers, runs a few untimed warmup launches, then wraps the
// launch(es) to actually profile in an NvtxRange named after `name` below.
void profile_naive(int S, const float* h_A, const float* h_B);
void profile_coalesced(int S, const float* h_A, const float* h_B);
void profile_tiled(int S, const float* h_A, const float* h_B);
void profile_coarsened(int S, const float* h_A, const float* h_B);
void profile_cublas(int S, const float* h_A, const float* h_B);
void profile_cutlass_fp32(int S, const float* h_A, const float* h_B);
void profile_cutlass_tf32(int S, const float* h_A, const float* h_B);
void profile_cutlass_fp16(int S, const float* h_A, const float* h_B);

struct ProfileEntry {
    const char* name;   // also the NVTX range label used by --nvtx-include
    void (*fn)(int, const float*, const float*);
};

extern const ProfileEntry PROFILE_TABLE[];
extern const int PROFILE_TABLE_SIZE;
