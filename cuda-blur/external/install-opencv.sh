#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})
OPENCV_VERSION=4.10.0


# library installed in this directory/lib
LIBDIR=${THIS_DIR}/lib

# download and untar
wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz -O ${THIS_DIR}/opencv.tar.gz
cd ${THIS_DIR} && tar -xzf ${THIS_DIR}/opencv.tar.gz

# build the library
cd ${THIS_DIR}/opencv-${OPENCV_VERSION}
mkdir -p build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_INSTALL_PREFIX=$LIBDIR/opencv \
	-D BUILD_EXAMPLES=ON ..

make -j$(nproc)
make install

# remove temporary files
cd ${THIS_DIR} && rm -rf opencv-${OPENCV_VERSION}
cd ${THIS_DIR} && rm -rf opencv.tar.gz