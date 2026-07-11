#include "utils.h"

void fill_random(float* p, int n) {
    for (int i = 0; i < n; ++i)
        p[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}
