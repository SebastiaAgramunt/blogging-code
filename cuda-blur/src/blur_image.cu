#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "blur_image.cuh"

#define BLUR_SIZE 10

__global__ void blur_kernel(unsigned char* input, unsigned char *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixVal[3] = {0, 0, 0};
        int pixels = 0;

        for(int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1; blurRow++){
            for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1; blurCol++){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    int offset = (curRow * width + curCol) * 3;
                    pixVal[0] += input[offset];     // R
                    pixVal[1] += input[offset + 1]; // G
                    pixVal[2] += input[offset + 2]; // B
                    pixels++;
                }
            }
        }
        int out_offset = (row * width + col) * 3;
        output[out_offset]     = static_cast<unsigned char>(pixVal[0] / pixels);
        output[out_offset + 1] = static_cast<unsigned char>(pixVal[1] / pixels);
        output[out_offset + 2] = static_cast<unsigned char>(pixVal[2] / pixels);
    }
}

void img_to_blur(unsigned char* input, int width, int height) {
    unsigned char* d_rgb;
    unsigned char* d_blur;

    int n_pixels = width * height * 3;
    unsigned char* h_blur = new unsigned char[n_pixels];

    cudaMalloc(&d_rgb, n_pixels * sizeof(unsigned char));
    cudaMalloc(&d_blur, n_pixels * sizeof(unsigned char));
    cudaMemcpy(d_rgb, input, n_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((height + 15) / 16, (width + 15) / 16);
    blur_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_blur, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_blur, d_blur, n_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_rgb);
    cudaFree(d_blur);

    cv::Mat blur_image(width, height, CV_8UC3, h_blur);
    cv::imwrite("blur_output_cuda.jpg", blur_image);

    delete[] h_blur;
}