#include <opencv2/opencv.hpp>
#include <iostream>

#include "blur_image.cuh"
#include "rgb_to_grayscale.cuh"
#include "utils.h"


int main(int argc, char* argv[]) {
    // Check if the user provided an argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Get the image path from the command-line argument
    std::string imagePath = argv[1];

    // Read the image
    cv::Mat image = cv::imread(imagePath);

    // Check if the image was successfully loaded
    if (image.empty()) {
        std::cerr << "Error: Unable to load image at " << imagePath << std::endl;
        return -1;
    }

    // cast image to 1d array
    int width = image.size().width;
    int height = image.size().height;
    int channels = image.channels();
    int bitsPerChannel = getBitsPerChannel(image.depth());
    int num_pixels = width * height;

    // print size of image in bytes
    std::cout << "Image size: " << image.total() << " bytes" << std::endl;
    std::cout << "Image size: " << image.total() / 1024 << " KB" << std::endl;
    std::cout << "Image size: " << image.total() / 1024 / 1024 << " MB" << std::endl;

    // print image shape
    std::cout << "Image shape: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "Number of pixels: " << num_pixels << std::endl;
    std::cout << "Bits per channel per pixel: " << bitsPerChannel << std::endl;
    std::cout << "Bytes per channel per pixel: " << bitsPerChannel / 8 << std::endl;

    unsigned char* image_data = image.data;

    cpu_grayscale(image_data, height, width);
    img_to_gray(image_data, height, width);
    img_to_blur(image_data, height, width);

    return 0;
}