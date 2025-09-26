#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})
INSTALL_DIR=${ROOT_DIR}/external
LIB_DIR=${INSTALL_DIR}/lib
OPENBLAS_VERSION="0.3.30"
LAPACK_VERSION="3.12.1"
OPENBLAS_URL="https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
LAPACK_URL="https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v${LAPACK_VERSION}.tar.gz"

check-requisites(){
    if ! command -v cmake &>/dev/null; then
        echo "Error: cmake is not installed. Please install cmake first." >&2
        exit 1
    fi
    if ! command -v wget &>/dev/null; then
        echo "Error: wget is not installed. Please install wget first." >&2
        exit 1
    fi
    if ! command -v gfortran &>/dev/null; then
        echo "Error: gfortran is not installed. Please install gfortran first." >&2
        exit 1
    fi
}
create-dirs(){
    if [ ! -d ${ROOT_DIR}/external ]; then
        mkdir -p ${ROOT_DIR}/external
    fi
    if  [ ! -d ${LIB_DIR} ]; then
        mkdir -p ${LIB_DIR}
    fi
    if [ ! -d ${INSTALL_DIR}/tmp ]; then
        mkdir -p ${INSTALL_DIR}/tmp
    fi
}
compile-install-openblas(){
    cd ${INSTALL_DIR}/tmp
    # download
    if [ ! -f OpenBLAS-${OPENBLAS_VERSION}.tar.gz ]; then
        wget ${OPENBLAS_URL}
    fi
    # untar
    if [ ! -d OpenBLAS-${OPENBLAS_VERSION} ]; then
        tar -xvzf OpenBLAS-${OPENBLAS_VERSION}.tar.gz
    fi
    cd OpenBLAS-${OPENBLAS_VERSION}
    if [ ! -d build ]; then
        mkdir -p build && cd build

        if [[ "$(uname)" == "Linux" && "$(uname -m)" == "aarch64"  ]]; then
            cmake -DCMAKE_INSTALL_PREFIX="${LIB_DIR}/openblas" \
                 -DCMAKE_BUILD_TYPE=Release \
                 -DBUILD_SHARED_LIBS=ON \
                 -DDYNAMIC_ARCH=OFF \
                 -DTARGET=ARMV8 \
                 ..
            cmake --build . -j
            cmake --install .
            cd ${THIS_DIR}
        else
            cmake -DCMAKE_INSTALL_PREFIX=${LIB_DIR}/openblas \
                -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_SHARED_LIBS=ON \
                ..

            make -j 64
            make install
            cd ${THIS_DIR}
        fi
    fi
}
compile-install-lapack(){
    cd ${INSTALL_DIR}/tmp
    # download
    if [ ! -f v${LAPACK_VERSION}.tar.gz ]; then
        wget ${LAPACK_URL}
    fi
    # untar
    if [ ! -d lapack-${LAPACK_VERSION} ]; then
        tar -xvzf v${LAPACK_VERSION}.tar.gz
    fi
    cd lapack-${LAPACK_VERSION}
    if [ ! -d build ]; then
        # if linux OS
        if [[ "$(uname)" == "Linux"  ]]; then
            mkdir -p build && cd build
            cmake -DCMAKE_INSTALL_PREFIX=${LIB_DIR}/lapack \
              -DCBLAS=ON \
              -DBUILD_SHARED_LIBS=ON \
              -DLAPACKE=ON \
              -DBLAS_LIBRARIES="${LIB_DIR}/openblas/lib/libopenblas.so" \
              -DCMAKE_BUILD_TYPE=Release \
              ..
            cd ..
        fi

        # if macOS and new architecture  arm64
        if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
            rm -rf build
            export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
            cmake -S . -B build \
            -DCMAKE_INSTALL_PREFIX="${LIB_DIR}/lapack" \
            -DBUILD_SHARED_LIBS=ON -DCBLAS=ON -DLAPACKE=ON \
            -DCMAKE_C_COMPILER=/usr/bin/clang \
            -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
            -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran \
            -DCMAKE_OSX_SYSROOT="$SDKROOT" \
            -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-syslibroot,${SDKROOT}" \
            -DCMAKE_EXE_LINKER_FLAGS="-Wl,-syslibroot,${SDKROOT}"
        fi
        cmake --build build -j 8
        cmake --install build
        cd ${THIS_DIR}
    fi
}
check-requisites
create-dirs
compile-install-openblas
compile-install-lapack