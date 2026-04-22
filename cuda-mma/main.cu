#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "utils.h"
#include "benchmarks.h"

#define WARMUP 3
#define ITERS  5

using BenchmarkFn = float (*)(int, const float*, const float*);

static void run_sweep(const char* label, BenchmarkFn fn,
                      const int* sizes, int n_sizes,
                      const float* h_A, const float* h_B,
                      const char* csv_path)
{
    printf("%s roofline sweep (%d iters, %d warmup)\n", label, ITERS, WARMUP);
    printf("%-6s  %9s  %9s  %11s  %9s\n",
           "Size", "Time(ms)", "GFLOP/s", "BW(GB/s)", "AI(F/B)");
    printf("------  ---------  ---------  -----------  ---------\n");

    FILE* csv = fopen(csv_path, "w");
    fprintf(csv, "size,time_ms,gflops,bandwidth_gbs,arithmetic_intensity\n");

    for (int i = 0; i < n_sizes; ++i) {
        int S = sizes[i];

        float avg_ms = fn(S, h_A, h_B);

        double flops = 2.0 * S * S * S;
        double bytes = ((double)S * S
                      + (double)S * S
                      + 2.0 * S * S)
                     * sizeof(float);
        double ai        = flops / bytes;
        double gflops    = flops / (avg_ms * 1e-3) / 1e9;
        double bandwidth = bytes / (avg_ms * 1e-3) / 1e9;

        printf("%-6d  %9.3f  %9.1f  %11.1f  %9.2f\n",
               S, avg_ms, gflops, bandwidth, ai);
        fprintf(csv, "%d,%.3f,%.3f,%.3f,%.4f\n",
                S, avg_ms, gflops, bandwidth, ai);
    }
    printf("\n");
    fclose(csv);
}

int main() {
    srand(42);

    std::filesystem::create_directories("output");

    const int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    const int N_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    const int S_max = sizes[N_SIZES - 1];
    float* h_A = new float[(size_t)S_max * S_max];
    float* h_B = new float[(size_t)S_max * S_max];
    fill_random(h_A, S_max * S_max);
    fill_random(h_B, S_max * S_max);

    run_sweep("Naive",     benchmark_naive,     sizes, N_SIZES, h_A, h_B, "output/naive.csv");
    run_sweep("Tiled",     benchmark_tiled,     sizes, N_SIZES, h_A, h_B, "output/tiled.csv");
    run_sweep("Coalesced", benchmark_coalesced, sizes, N_SIZES, h_A, h_B, "output/coalesced.csv");
    run_sweep("cuBLAS",    benchmark_cublas,    sizes, N_SIZES, h_A, h_B, "output/cublas.csv");
    run_sweep("Coarsened", benchmark_coarsened, sizes, N_SIZES, h_A, h_B, "output/coarsened.csv");

    // CBLAS runs on CPU; cap at 4096 to keep runtime reasonable
    const int cblas_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int N_CBLAS = sizeof(cblas_sizes) / sizeof(cblas_sizes[0]);
    run_sweep("CBLAS",     benchmark_cblas,     cblas_sizes, N_CBLAS, h_A, h_B, "output/cblas.csv");

    delete[] h_A;
    delete[] h_B;

    return 0;
}
