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

compile(){
    OPENBLAS_INC="${ROOT_DIR}/external/lib/openblas/include/openblas"
    OPENBLAS_LIB="${ROOT_DIR}/external/lib/openblas/lib"

    LAPACKE_INC="${ROOT_DIR}/external/lib/lapack/include"
    LAPACKE_LIB="${ROOT_DIR}/external/lib/lapack/lib"

    # OPENBLAS example
    # compile object
    g++ -O3 \
        -std=c++17 \
        -c ${ROOT_DIR}/src/cblas_example.cpp \
        -I${OPENBLAS_INC} \
        -o ${ROOT_DIR}/build/obj/cblas_example.o

    # compile binary
    g++ -O3 \
        ${ROOT_DIR}/build/obj/cblas_example.o \
        -L${OPENBLAS_LIB} \
        -lopenblas \
        -Wl,-rpath,${OPENBLAS_LIB} \
        -o ${ROOT_DIR}/build/bin/cblas_example

    # LAPACKE example
    # compile object
    g++ -O3 \
        -std=c++17 \
        -c ${ROOT_DIR}/src/lapacke_example.cpp \
        -I${LAPACKE_INC} \
        -o ${ROOT_DIR}/build/obj/lapacke_example.o

    # # compile binary
    g++ -O3 \
    "${ROOT_DIR}/build/obj/lapacke_example.o" \
    -L"${LAPACKE_LIB}" \
    -L"${OPENBLAS_LIB}" \
    -llapacke -lopenblas \
    -Wl,-rpath,"${LAPACKE_LIB}" \
    -Wl,-rpath,"${OPENBLAS_LIB}" \
    -o "${ROOT_DIR}/build/bin/lapacke_example"

}

recreate_dirs
compile