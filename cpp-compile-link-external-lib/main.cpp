#include <opencv2/opencv.hpp>
#include <iostream>

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

    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(15, 15), 5.0);

    // Display the original and processed images
    cv::imshow("Original Image", image);
    cv::imshow("Blurred Image", blurredImage);

    // Wait for a key press
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    // // Save the processed image
    std::string outputPath = "blurred_image.jpg";
    cv::imwrite(outputPath, blurredImage);
    std::cout << "Processed image saved to " << outputPath << std::endl;

    return 0;
}

