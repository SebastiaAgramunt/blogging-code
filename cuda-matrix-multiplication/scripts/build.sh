#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})


recreate_dirs(){
    # removing build directory
    echo "Removing ${ROOT_DIR}/build and recreating..."
    rm -rf ${ROOT_DIR}/data
    rm -rf ${ROOT_DIR}/build
    mkdir ${ROOT_DIR}/build

    # creating directories for the build
    mkdir ${ROOT_DIR}/build/obj
    mkdir ${ROOT_DIR}/build/bin
    mkdir ${ROOT_DIR}/build/lib
}

compile(){
    echo "Compiling..."

    COMMON_INC="-I${ROOT_DIR}/include -I/usr/local/cuda/include"
    COMMON_LIB="-L/usr/local/cuda/lib64"
    COMMON_LIBS="-lcublas -lcudart"
    GENCODE="-gencode arch=compute_90,code=compute_90"
    OPT="-O3 -std=c++17 ${GENCODE} -Xptxas -O3"

    nvcc -c ${ROOT_DIR}/src/simpleMultiply.cu ${COMMON_INC}${OPT} \
        -o ${ROOT_DIR}/build/obj/simpleMultiply.o

    nvcc -c ${ROOT_DIR}/src/main.cu ${COMMON_INC} ${OPT} \
        -o ${ROOT_DIR}/build/obj/main.o

    nvcc -c ${ROOT_DIR}/src/utils.cu ${COMMON_INC} ${OPT} \
        -o ${ROOT_DIR}/build/obj/utils.o

    nvcc -c ${ROOT_DIR}/src/tiledMultiply.cu ${COMMON_INC} ${OPT} \
        -o ${ROOT_DIR}/build/obj/tiledMultiply.o

    nvcc ${OPT} -o ${ROOT_DIR}/build/bin/main \
        -std=c++17 \
        ${COMMON_LIB} \
        ${COMMON_LIBS} \
        ${ROOT_DIR}/build/obj/main.o \
        ${ROOT_DIR}/build/obj/simpleMultiply.o \
        ${ROOT_DIR}/build/obj/tiledMultiply.o \
        ${ROOT_DIR}/build/obj/utils.o
}

recreate_dirs
compile