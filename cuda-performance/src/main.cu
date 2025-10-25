#include "utils.h"
#include "vector_add.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <chrono>

void open_new_csv_file(const std::string& filename) {
    bool new_file = !std::filesystem::exists(filename);
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        return;
    }

    if (new_file) {
        file << "threads_per_block,number_of_additions,N,sizeMB,allocate_time,copy_H2D_time,compute_time,copy_D2H_time,free_time,total_GPU_time,total_CPU_time\n";
    }

    file.close();
}

// Append a row with timing results
void append_to_csv(const std::string& filename,
                   int threads_per_block,
                   int number_of_additions,
                   size_t N,
                   double sizeMB,
                   float allocateTime,
                   float loadH2D,
                   float calcTime,
                   float loadD2H,
                   float freeTime,
                   float totalGPUTime,
                   float totalCPUTime)
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << filename << " for appending.\n";
        return;
    }

    file << threads_per_block << ","
         << number_of_additions << ","
         << N << ","
         << sizeMB << ","
         << allocateTime << ","
         << loadH2D << ","
         << calcTime << ","
         << loadD2H << ","
         << freeTime << ","
         << totalGPUTime << ","
         << totalCPUTime
         << "\n";

    file.close();
}

int main(int argc, char **argv) {

    int threads_per_block;       // threads per block
    int m;                       // number of additions to perform
    std::string filename;        // file to save results
    bool use_cpu = true;         // use cpu by default

    using T = float;

    // arg parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads_per_block" && i + 1 < argc) {
            threads_per_block = std::stoi(argv[++i]);
        }
        if (arg == "--number_of_additions" && i + 1 < argc) {
            m = std::stoi(argv[++i]);
        }
        if (arg == "--output_file" && i + 1 < argc) {
            filename = argv[++i];
        }
        if (arg == "--use_cpu" && i + 1 < argc){
            std::string val = argv[++i];
            use_cpu = (val == "1" || val == "true" || val == "True");
        }
        if (arg == "--help") {
            std::cerr << "Usage: " << argv[0] << " --threads_per_block <int> --vector_size <int> --number_of_additions <int> --use_cpu <bool> --output_file <string>\n";
            return 1;
        }
    }

    std::cout << "threads_per_block: " << threads_per_block << ", number_of_additions: " << m << ", use_cpu:: " << use_cpu << std::endl;

    // vector sizes to consider
    long int sizes[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
        1048576,    // 4 MB
        2097152,    // 8 MB
        4194304,    // 16 MB
        8388608,    // 32 MB
        16777216,   // 64 MB
        33554432,   // 128 MB
        67108864,   // 256 MB
        134217728,  // 512 MB
        268435456,  // 1 GB
        536870912,  // 2 GB
        1073741824  // 4 GB
    };

    // create file if it doesn't exist
    open_new_csv_file(filename);

    // Create events to log CUDA times
    cudaEvent_t startAllocate, stopAllocate, startH2D, stopH2D, startCalc, stopCalc, startD2H, stopD2H, startFree, stopFree;
    cudaEventCreate(&startAllocate);
    cudaEventCreate(&stopAllocate);
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startCalc);
    cudaEventCreate(&stopCalc);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);
    cudaEventCreate(&startFree);
    cudaEventCreate(&stopFree);

    float allocateTime = 0;
    float loadH2D = 0;
    float calcTime = 0;
    float loadD2H = 0;
    float freeTime = 0;

    // dummy calculation to warm up the GPU
    dummyCalculation<int>();
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 

    dummyCalculation<float>();
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 

    dummyCalculation<double>();
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 

    // for every size calculate the addition and record the results
    for(int s=0; s < sizeof(sizes)/sizeof(sizes[0]); s++){
        long int N = sizes[s];

        // ceate host vectors and randomize
        std::vector<float> h_a(N), h_b(N), h_c(N), h_c_cpu(N);
        for (int i = 0; i < N; ++i) {
            h_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
            h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // CPU time
        auto cpu_start = std::chrono::high_resolution_clock::now();
        if(use_cpu){
            for (int i = 0; i < N; ++i) {
                double acc = 0;
                double s = h_a[i] + h_b[i];
                for (int j = 0; j < m-1; ++j) {
                    acc = acc + s;
                }
                h_c_cpu[i] = acc;
            }
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpuTime = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // CUDA times
        // Allocate device memory
        T *d_a, *d_b, *d_c;
        CHECK_CUDA_ERROR(cudaEventRecord(startAllocate));
        CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_c, N * sizeof(T)));
        CHECK_CUDA_ERROR(cudaEventRecord(stopAllocate));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stopAllocate));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&allocateTime, startAllocate, stopAllocate));

        // loading data from host to device
        CHECK_CUDA_ERROR(cudaEventRecord(startH2D));
        CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), N * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), N * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventRecord(stopH2D));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stopH2D));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&loadH2D, startH2D, stopH2D));

        // perform calculation on GPU
        CHECK_CUDA_ERROR(cudaEventRecord(startCalc));
        vectorAdd_wrapper<T>(d_a, d_b, d_c, N, m, threads_per_block);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); 
        CHECK_CUDA_ERROR(cudaEventRecord(stopCalc));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stopCalc));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&calcTime, startCalc, stopCalc));

        // load back data from device to host
        CHECK_CUDA_ERROR(cudaEventRecord(startD2H));
        CHECK_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaEventRecord(stopD2H));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stopD2H));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&loadD2H, startD2H, stopD2H));

        // free cuda memory
        CHECK_CUDA_ERROR(cudaEventRecord(startFree));
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        CHECK_CUDA_ERROR(cudaEventRecord(stopFree));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stopFree));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&freeTime, startFree, stopFree));

        float totalTime = allocateTime + loadH2D + calcTime + loadD2H + freeTime;
        float sizeMB = N * sizeof(T) / (1024.0 * 1024.0);

        append_to_csv(filename, threads_per_block, m, N, sizeMB, allocateTime, loadH2D, calcTime, loadD2H, freeTime, totalTime, cpuTime);
    }
    return 0;
}