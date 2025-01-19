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
    g++ -std=c++17 -I${ROOT_DIR}/external/lib/opencv/include/opencv4 -c ${ROOT_DIR}/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link all the objects
    g++ ${ROOT_DIR}/build/obj/main.o \
        -I${ROOT_DIR}/external/lib/opencv/include/opencv4 \
        -L${ROOT_DIR}/external/lib/opencv/lib \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -Wl,-rpath,external/lib/opencv/lib \
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

