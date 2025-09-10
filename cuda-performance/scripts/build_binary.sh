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
    echo "Compiling objects for executable..."

    # compile to objects
    nvcc -O3 -arch=sm_80 -c ${ROOT_DIR}/src/vector_add_types.cu -o ${ROOT_DIR}/build/obj/vector_add_types.o 
    
    # link all the objects
    nvcc -O3 -arch=sm_80 ${ROOT_DIR}/build/obj/vector_add_types.o -o ${ROOT_DIR}/build/bin/vector_add_types
}

compile_exec