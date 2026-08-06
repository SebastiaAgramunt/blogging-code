#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "rgb_to_grayscale.cuh"


__global__ void rgb_to_grayscale_kernel(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int rgb_idx = idx * 3;
        unsigned char r = rgb[rgb_idx];
        unsigned char g = rgb[rgb_idx + 1];
        unsigned char b = rgb[rgb_idx + 2];

        gray[idx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}


void img_to_gray(unsigned char* input, int width, int height) {
    unsigned char* image_data_gray = new unsigned char[width * height];

    unsigned char* d_rgb;
    unsigned char* d_gray;
    unsigned char* h_gray = new unsigned char[width * height];

    cudaMalloc(&d_rgb, width * height * 3);
    cudaMalloc(&d_gray, width * height);

    cudaMemcpy(d_rgb, input, width * height * 3, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    rgb_to_grayscale_kernel<<<blocks, threads>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gray, d_gray, width * height, cudaMemcpyDeviceToHost);
    cudaFree(d_rgb);
    cudaFree(d_gray);

    cv::Mat gray_image(width, height, CV_8UC1, h_gray);
    cv::imwrite("gray_output_cuda.jpg", gray_image);
    delete[] h_gray;    
}

void cpu_grayscale(unsigned char* input, int width, int height) {
    unsigned char* image_data_gray = new unsigned char[width * height];

    for (int i = 0; i < width * height; i++) {
        image_data_gray[i] = 0.299 * input[i * 3] + 0.587 * input[i * 3 + 1] + 0.114 * input[i * 3 + 2];
    }

    cv::Mat gray_image(width, height, CV_8UC1, image_data_gray);
    cv::imwrite("gray_output_cpu.jpg", gray_image);
    delete[] image_data_gray;
}