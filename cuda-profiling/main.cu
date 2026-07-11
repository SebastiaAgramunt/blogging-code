#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "profiling.h"
#include "utils.h"

// Usage: cuda_profiling [SIZE] [KERNEL]
//   SIZE   square matrix dimension (default 4096)
//   KERNEL one of the names in PROFILE_TABLE; if omitted, all run in sequence.
//
// Each kernel's profiled launches are wrapped in an NVTX range named after
// it, so a single run can be:
//   - traced end-to-end with `nsys profile` (every kernel + memcpy on one
//     timeline), or
//   - narrowed to one kernel with `ncu --nvtx-include "<KERNEL>/"`.
int main(int argc, char** argv) {
    srand(42);

    int S = (argc > 1) ? atoi(argv[1]) : 4096;
    const char* kernel = (argc > 2) ? argv[2] : nullptr;

    float* h_A = new float[(size_t)S * S];
    float* h_B = new float[(size_t)S * S];
    fill_random(h_A, S * S);
    fill_random(h_B, S * S);

    bool ran_any = false;
    for (int i = 0; i < PROFILE_TABLE_SIZE; ++i) {
        const ProfileEntry& entry = PROFILE_TABLE[i];
        if (kernel && strcmp(kernel, entry.name) != 0)
            continue;
        printf("Profiling %-14s S=%d\n", entry.name, S);
        entry.fn(S, h_A, h_B);
        ran_any = true;
    }

    if (!ran_any) {
        fprintf(stderr, "Unknown kernel '%s'. Available: ", kernel);
        for (int i = 0; i < PROFILE_TABLE_SIZE; ++i)
            fprintf(stderr, "%s%s", PROFILE_TABLE[i].name,
                    i + 1 < PROFILE_TABLE_SIZE ? ", " : "\n");
        delete[] h_A;
        delete[] h_B;
        return EXIT_FAILURE;
    }

    delete[] h_A;
    delete[] h_B;
    return 0;
}
