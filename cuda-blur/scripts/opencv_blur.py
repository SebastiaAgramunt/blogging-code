#!/usr/bin/env python

import numpy as np
import cv2 as cv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


if __name__ == "__main__":
    img = cv.imread(ROOT_DIR / "raw_img.jpeg")
    if img is None:
        raise ValueError("Failed to load image. Check the file path.")

    window=51
    kernel = np.ones((window, window),np.float32)/(window * window)
    filtered_img = cv.filter2D(img, -1, kernel)
    cv.imwrite(str(ROOT_DIR / "blur_output_python.jpeg"), filtered_img)
    
    kernel = gaussian_kernel(window, window/2.0)
    filtered_img = cv.filter2D(img, -1, kernel)
    cv.imwrite(str(ROOT_DIR / "blur_output_gaussian_python.jpeg"), filtered_img)