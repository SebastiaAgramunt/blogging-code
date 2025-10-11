#ifndef BLUR_IMAGE_CUH
#define BLUR_IMAGE_CUH

__global__ void blur_kernel(unsigned char* input, unsigned char *output, int width, int height);
void img_to_blur(unsigned char* input, int width, int height);

#endif
