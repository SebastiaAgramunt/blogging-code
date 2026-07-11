#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA event-based timer.  Usage:
//
//   GpuTimer t;
//   t.start();
//   kernel<<<...>>>(...);
//   float ms = t.stop();   // blocks until kernel finishes
// ---------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t _start, _stop;

    GpuTimer()  { cudaEventCreate(&_start); cudaEventCreate(&_stop); }
    ~GpuTimer() { cudaEventDestroy(_start); cudaEventDestroy(_stop);  }

    void start() { cudaEventRecord(_start); }

    // Returns elapsed milliseconds.
    float stop() {
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, _start, _stop);
        return ms;
    }
};
