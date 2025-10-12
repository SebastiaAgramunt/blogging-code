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
    echo "Compiling..."

    # project includes
    INCLUDES="-I${ROOT_DIR}/include"

    # cuda specific includes (modify to your cuda installation)
    CUDA_INCLUDES="-I/usr/include"
    CUDA_LIB_DIRS="-L/usr/lib/x86_64-linux-gnu/"

    # cuda libraries to link
    CUDA_LIB="-lcudart"

    # flags. Using sm_80 for A100 GPU (compute cabability 8.0)
    FLAGS="-O3 -arch=sm_80"

    # compile to objects
    nvcc ${FLAGS} \
        -c ${ROOT_DIR}/src/vector_add.cu \
        ${INCLUDES} \
        ${CUDA_INCLUDES} \
        -o ${ROOT_DIR}/build/obj/vector_add.o

    nvcc ${FLAGS} \
        -c ${ROOT_DIR}/src/vector_add.cu \
        ${INCLUDES} \
        ${CUDA_INCLUDES} \
        -o ${ROOT_DIR}/build/obj/vector_add.o

    nvcc ${FLAGS} \
        -c ${ROOT_DIR}/src/main.cu \
        ${INCLUDES} \
        ${CUDA_INCLUDES} \
        -o ${ROOT_DIR}/build/obj/main.o
    
    
    # link all the objects
    nvcc ${FLAGS} \
        ${ROOT_DIR}/build/obj/vector_add.o \
        ${ROOT_DIR}/build/obj/main.o \
        ${CUDA_LIB_DIRS} \
        ${CUDA_LIB} \
        -o ${ROOT_DIR}/build/bin/main
}

compile_exec