#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

recreate_dirs(){
    # removing build directory
    echo "Removing ${ROOT_DIR}/build and recreating..."
    rm -rf ${ROOT_DIR}/build
    mkdir ${ROOT_DIR}/build

    # creating directories for the build
    mkdir ${ROOT_DIR}/build/obj
    mkdir ${ROOT_DIR}/build/bin
    mkdir ${ROOT_DIR}/build/lib
}

compile_exec(){
    recreate_dirs

    # compile to objects
    echo "Compiling objects for executable..."

    nvcc -std=c++17 \
        -Xcudafe --diag_suppress=611 \
        -I${ROOT_DIR}/external/lib/opencv/include/opencv4 \
        -I${ROOT_DIR}/include \
        -c ${ROOT_DIR}/src/main.cu \
        -o ${ROOT_DIR}/build/obj/main.o

    nvcc -std=c++17 \
        -Xcudafe --diag_suppress=611 \
        -I${ROOT_DIR}/external/lib/opencv/include/opencv4 \
        -I${ROOT_DIR}/include \
        -I/usr/local/cuda/include \
        -L/usr/local/cuda/lib64 \
        -c ${ROOT_DIR}/src/utils.cpp \
        -o ${ROOT_DIR}/build/obj/utils.o

    nvcc -std=c++17 \
        -Xcudafe --diag_suppress=611 \
        -I${ROOT_DIR}/external/lib/opencv/include/opencv4 \
        -I${ROOT_DIR}/include \
        -I/usr/local/cuda/include \
        -L/usr/local/cuda/lib64 \
        -c ${ROOT_DIR}/src/rgb_to_grayscale.cu \
        -o ${ROOT_DIR}/build/obj/rgb_to_grayscale.o

    nvcc -std=c++17 \
        -Xcudafe --diag_suppress=611 \
        -I${ROOT_DIR}/external/lib/opencv/include/opencv4 \
        -I${ROOT_DIR}/include \
        -I/usr/local/cuda/include \
        -L/usr/local/cuda/lib64 \
        -c ${ROOT_DIR}/src/blur_image.cu \
        -o ${ROOT_DIR}/build/obj/blur_image.o


    # link all the objects
    g++ ${ROOT_DIR}/build/obj/main.o \
        ${ROOT_DIR}/build/obj/rgb_to_grayscale.o \
        ${ROOT_DIR}/build/obj/blur_image.o \
        ${ROOT_DIR}/build/obj/utils.o \
        -I${ROOT_DIR}/external/lib/opencv/include/opencv4 \
        -I/usr/local/cuda/include \
        -L${ROOT_DIR}/external/lib/opencv/lib \
        -L/usr/local/cuda/lib64 \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lcudart \
        -Wl,-rpath,${ROOT_DIR}/external/lib/opencv/lib \
        -o ${ROOT_DIR}/build/bin/main
}


croak(){
    echo "[ERROR] $*" > /dev/stderr
    exit 1
}

main(){
    if [[ -z "$TASK" ]]; then
        croak "No TASK specified."
    fi
    echo "[INFO] running $TASK $*"
    $TASK "$@"
}

main "$@"

