#include <opencv2/opencv.hpp>
#include <iostream>

int getBitsPerChannel(int depth) {
    switch (depth) {
        case CV_8U:
        case CV_8S:  return 8;
        case CV_16U:
        case CV_16S: return 16;
        case CV_32S:
        case CV_32F: return 32;
        case CV_64F: return 64;
        default:     return -1; // unknown type
    }
}