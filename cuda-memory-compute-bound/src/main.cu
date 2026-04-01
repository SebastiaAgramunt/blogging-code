//
// main.cu — Memory-Bound vs Compute-Bound: A GPU Deep Dive with Vector Addition
//
// Benchmarks two related quantities as the arithmetic intensity (AI) rises:
//   • Achieved memory bandwidth  (GB/s)
//   • Achieved FP32 throughput   (GFLOP/s)
//
// The kernel does:
//   val = a[i] + b[i]                     ← 1 FLOP,  8 bytes read
//   for j in 0..m: val = fma(val,α,β)    ← 2m FLOPs, no extra memory
//   c[i] = val                            ← 0 FLOPs,  4 bytes written
//
//   AI = (1 + 2m) / 12  [FLOPs / byte]
//
// Sweeps:
//   1. ROOFLINE  — fixed large N, m ∈ {1,2,4,…,16384}
//      Produces the roofline-model scatter plot.
//   2. BW_SAT    — fixed m=1, N ∈ {2^10 … 2^27}
//      Shows how bandwidth saturates as the working set grows.
//
// Output: two CSV files in <results_dir>/
//   roofline.csv
//   bw_saturation.csv
//
// Usage:
//   ./main --results_dir ../results [--threads_per_block 256]
//          [--N_roofline 33554432] [--bw_repeats 10]

#include "kernels.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <random>

// ─────────────────────────────────────────────────────────────────────────────
// Theoretical peak values from device properties
// ─────────────────────────────────────────────────────────────────────────────

// CUDA cores per SM keyed on (major, minor) compute capability
static int cores_per_sm(int major, int minor)
{
    struct Entry { int major, minor, cores; };
    static constexpr Entry table[] = {
        {3, 0, 192}, {3, 2, 192}, {3, 5, 192}, {3, 7, 192},
        {5, 0, 128}, {5, 2, 128},
        {6, 0,  64}, {6, 1, 128}, {6, 2, 128},
        {7, 0,  64}, {7, 2,  64}, {7, 5,  64},
        {8, 0,  64}, {8, 6, 128}, {8, 7, 128}, {8, 9, 128},
        {9, 0, 128},
    };
    for (auto& e : table)
        if (e.major == major && e.minor == minor) return e.cores;
    return 128; // safe default
}

// Theoretical peak HBM / GDDR bandwidth in GB/s
static double theoretical_bw_gbs(const cudaDeviceProp& p)
{
    // memoryClockRate is in kHz; memoryBusWidth is in bits
    // Bandwidth = clock_Hz × (bus_width_bits / 8) × 2 (DDR) / 1e9
    return static_cast<double>(p.memoryClockRate) * 1e3   // → Hz
         * (p.memoryBusWidth / 8.0)                        // → bytes/cycle
         * 2.0                                              // DDR
         / 1e9;
}

// Theoretical peak FP32 in GFLOP/s
static double theoretical_fp32_gflops(const cudaDeviceProp& p)
{
    // Each CUDA core can issue 1 FMA (= 2 FLOPs) per clock cycle
    return 2.0
         * p.multiProcessorCount
         * cores_per_sm(p.major, p.minor)
         * static_cast<double>(p.clockRate) * 1e3   // → Hz
         / 1e9;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static float time_kernel_ms(
    const float* d_a, const float* d_b, float* d_c,
    int N, int m, int tpb, int repeats,
    cudaEvent_t ev0, cudaEvent_t ev1)
{
    float best = 1e30f;
    for (int r = 0; r < repeats; ++r) {
        float t;
        vector_add_fma(d_a, d_b, d_c, N, m, tpb, ev0, ev1, t);
        best = std::min(best, t);
    }
    return best;
}

static float time_bw_kernel_ms(
    const float* d_a, float* d_c,
    int N, int tpb, int repeats,
    cudaEvent_t ev0, cudaEvent_t ev1)
{
    float best = 1e30f;
    for (int r = 0; r < repeats; ++r) {
        float t;
        bandwidth_test(d_a, d_c, N, tpb, ev0, ev1, t);
        best = std::min(best, t);
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Arg parsing
// ─────────────────────────────────────────────────────────────────────────────

struct Config {
    std::string results_dir   = "../results";
    int  threads_per_block    = 256;
    long N_roofline           = 1L << 25;   // 33 554 432 floats = 128 MB / array
    int  bw_repeats           = 10;
};

static Config parse_args(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--results_dir"      && i + 1 < argc) cfg.results_dir        = argv[++i];
        if (a == "--threads_per_block"&& i + 1 < argc) cfg.threads_per_block  = std::stoi(argv[++i]);
        if (a == "--N_roofline"       && i + 1 < argc) cfg.N_roofline         = std::stol(argv[++i]);
        if (a == "--bw_repeats"       && i + 1 < argc) cfg.bw_repeats         = std::stoi(argv[++i]);
        if (a == "--help") {
            std::cout << "Usage: ./main [--results_dir DIR] [--threads_per_block INT]\n"
                      << "             [--N_roofline INT] [--bw_repeats INT]\n";
            std::exit(0);
        }
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    Config cfg = parse_args(argc, argv);

    // ── Device info ─────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));

    double th_bw    = theoretical_bw_gbs(prop);
    double th_flops = theoretical_fp32_gflops(prop);
    double ridge_pt = th_flops / th_bw;   // FLOPs/byte at the ridge

    std::cout << "Device           : " << prop.name << "\n"
              << "Compute cap.     : " << prop.major << "." << prop.minor << "\n"
              << "SMs              : " << prop.multiProcessorCount << "\n"
              << "CUDA cores/SM    : " << cores_per_sm(prop.major, prop.minor) << "\n"
              << "GPU clock        : " << prop.clockRate / 1e6 << " GHz\n"
              << "Mem clock        : " << prop.memoryClockRate / 1e6 << " GHz\n"
              << "Bus width        : " << prop.memoryBusWidth << " bits\n"
              << "Theoretical BW   : " << th_bw    << " GB/s\n"
              << "Theoretical FLOPS: " << th_flops << " GFLOP/s\n"
              << "Ridge point      : " << ridge_pt << " FLOP/byte\n\n";

    // ── Make results directory ───────────────────────────────────────────────
    std::filesystem::create_directories(cfg.results_dir);

    // ── Shared CUDA events ───────────────────────────────────────────────────
    cudaEvent_t ev0, ev1;
    CHECK_CUDA_ERROR(cudaEventCreate(&ev0));
    CHECK_CUDA_ERROR(cudaEventCreate(&ev1));

    // ── Warm up ──────────────────────────────────────────────────────────────
    warmup();
    warmup();

    // ── Measure empirical peak bandwidth (pure-copy kernel) ──────────────────
    {
        const long N_bw = cfg.N_roofline;
        float *d_a, *d_b;
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, N_bw * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, N_bw * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_a, 1, N_bw * sizeof(float)));

        float t = time_bw_kernel_ms(d_a, d_b, static_cast<int>(N_bw),
                                    cfg.threads_per_block, cfg.bw_repeats, ev0, ev1);

        // copy reads 1 array + writes 1 array = 2 × N × sizeof(float) bytes
        double bytes  = 2.0 * N_bw * sizeof(float);
        double emp_bw = bytes / (t * 1e-3) / 1e9;

        std::cout << "Empirical peak BW: " << emp_bw << " GB/s  (vs "
                  << th_bw << " GB/s theoretical)\n\n";

        // Save device info + measured peak to a small metadata file
        std::ofstream meta(cfg.results_dir + "/device_info.csv");
        meta << "key,value\n"
             << "device_name,"          << prop.name << "\n"
             << "compute_capability,"   << prop.major << "." << prop.minor << "\n"
             << "num_sms,"              << prop.multiProcessorCount << "\n"
             << "cores_per_sm,"         << cores_per_sm(prop.major, prop.minor) << "\n"
             << "gpu_clock_ghz,"        << prop.clockRate / 1e6 << "\n"
             << "mem_clock_ghz,"        << prop.memoryClockRate / 1e6 << "\n"
             << "mem_bus_width_bits,"   << prop.memoryBusWidth << "\n"
             << "theoretical_bw_gbs,"  << th_bw << "\n"
             << "theoretical_fp32_gflops," << th_flops << "\n"
             << "ridge_point_flop_per_byte," << ridge_pt << "\n"
             << "empirical_bw_gbs,"    << emp_bw << "\n";

        CHECK_CUDA_ERROR(cudaFree(d_a));
        CHECK_CUDA_ERROR(cudaFree(d_b));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SWEEP 1: ROOFLINE — fixed N, sweep m
    //
    // For each m we record:
    //   • kernel_time_ms
    //   • achieved_bandwidth_gbs   = 12·N / (time_s · 1e9)
    //   • achieved_gflops          = (1+2m)·N / (time_s · 1e9)
    //   • arithmetic_intensity     = (1+2m) / 12
    // ═════════════════════════════════════════════════════════════════════════
    {
        const long N = cfg.N_roofline;
        const double bytes_per_elem  = 3.0 * sizeof(float);  // 2 reads + 1 write

        // m values span several orders of magnitude to cross the ridge point
        std::vector<int> m_values;
        for (int m = 1; m <= 16384; m *= 2) m_values.push_back(m);
        // add fine-grained points near the expected ridge
        for (int m : {3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288})
            m_values.push_back(m);
        std::sort(m_values.begin(), m_values.end());
        m_values.erase(std::unique(m_values.begin(), m_values.end()), m_values.end());

        float *d_a, *d_b, *d_c;
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_c, N * sizeof(float)));

        // Fill with random data (on host, then copy)
        std::vector<float> h(N);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::generate(h.begin(), h.end(), [&]{ return dist(rng); });
        CHECK_CUDA_ERROR(cudaMemcpy(d_a, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        std::ofstream csv(cfg.results_dir + "/roofline.csv");
        csv << "m,N,kernel_time_ms,arithmetic_intensity,achieved_bw_gbs,achieved_gflops,"
               "theoretical_bw_gbs,theoretical_fp32_gflops,ridge_point\n";

        std::cout << "=== ROOFLINE SWEEP (N=" << N << ") ===\n";
        std::cout << "  m       AI(FLOPs/B)   BW(GB/s)   GFLOP/s\n";

        for (int m : m_values) {
            float t_ms = time_kernel_ms(d_a, d_b, d_c, static_cast<int>(N),
                                        m, cfg.threads_per_block,
                                        cfg.bw_repeats, ev0, ev1);

            double t_s   = t_ms * 1e-3;
            double flops  = static_cast<double>(N) * (1.0 + 2.0 * m);
            double bytes  = static_cast<double>(N) * bytes_per_elem;
            double ai     = flops / bytes;
            double bw_gbs = bytes  / t_s / 1e9;
            double gflops  = flops  / t_s / 1e9;

            csv << m << "," << N << "," << t_ms << ","
                << ai << "," << bw_gbs << "," << gflops << ","
                << th_bw << "," << th_flops << "," << ridge_pt << "\n";

            std::cout << "  m=" << m
                      << "  AI=" << ai
                      << "  BW=" << bw_gbs << " GB/s"
                      << "  GFLOP/s=" << gflops << "\n";
        }

        csv.close();
        CHECK_CUDA_ERROR(cudaFree(d_a));
        CHECK_CUDA_ERROR(cudaFree(d_b));
        CHECK_CUDA_ERROR(cudaFree(d_c));
        std::cout << "\n";
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SWEEP 2: BANDWIDTH SATURATION — m=1, sweep N
    //
    // Shows that small vectors can't saturate HBM bandwidth (the GPU doesn't
    // have enough warps in flight to fill the memory controllers), while large
    // vectors approach the empirical peak.
    // ═════════════════════════════════════════════════════════════════════════
    {
        const int m = 1;   // minimal compute — purely memory-bound
        const double bytes_per_elem = 3.0 * sizeof(float);

        // N from 1024 to N_roofline in powers of 2
        std::vector<long> n_values;
        for (long n = 1024; n <= cfg.N_roofline; n *= 2)
            n_values.push_back(n);

        std::ofstream csv(cfg.results_dir + "/bw_saturation.csv");
        csv << "m,N,sizeMB,kernel_time_ms,arithmetic_intensity,"
               "achieved_bw_gbs,achieved_gflops,"
               "theoretical_bw_gbs,theoretical_fp32_gflops\n";

        std::cout << "=== BANDWIDTH SATURATION SWEEP (m=" << m << ") ===\n";
        std::cout << "  N           sizeMB   BW(GB/s)\n";

        for (long N : n_values) {
            float *d_a, *d_b, *d_c;
            CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_c, N * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_a, 1, N * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_b, 2, N * sizeof(float)));

            float t_ms = time_kernel_ms(d_a, d_b, d_c, static_cast<int>(N),
                                        m, cfg.threads_per_block,
                                        cfg.bw_repeats, ev0, ev1);

            double t_s    = t_ms * 1e-3;
            double flops  = static_cast<double>(N) * (1.0 + 2.0 * m);
            double bytes  = static_cast<double>(N) * bytes_per_elem;
            double ai     = flops / bytes;
            double bw_gbs = bytes  / t_s / 1e9;
            double gflops = flops  / t_s / 1e9;
            double sizeMB = static_cast<double>(N) * sizeof(float) / (1024.0 * 1024.0);

            csv << m << "," << N << "," << sizeMB << "," << t_ms << ","
                << ai << "," << bw_gbs << "," << gflops << ","
                << th_bw << "," << th_flops << "\n";

            std::cout << "  N=" << N << "  " << sizeMB << " MB   BW=" << bw_gbs << " GB/s\n";

            CHECK_CUDA_ERROR(cudaFree(d_a));
            CHECK_CUDA_ERROR(cudaFree(d_b));
            CHECK_CUDA_ERROR(cudaFree(d_c));
        }

        csv.close();
    }

    CHECK_CUDA_ERROR(cudaEventDestroy(ev0));
    CHECK_CUDA_ERROR(cudaEventDestroy(ev1));

    std::cout << "\nResults written to: " << cfg.results_dir << "/\n";
    return 0;
}
