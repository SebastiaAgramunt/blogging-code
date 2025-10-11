#ifndef RGB_TO_GRAYSCALE_CUH
#define RGB_TO_GRAYSCALE_CUH

__global__ void rgb_to_grayscale_kernel(const unsigned char* rgb, unsigned char* gray, int width, int height);
void img_to_gray(unsigned char* input, int width, int height);
void cpu_grayscale(unsigned char* input, int width, int height);

#endif