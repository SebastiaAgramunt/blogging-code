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
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/matmul.cpp -o ${ROOT_DIR}/build/obj/matmul.o
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link all the objects
    g++ ${ROOT_DIR}/build/obj/matmul.o \
        ${ROOT_DIR}/build/obj/main.o \
        -o ${ROOT_DIR}/build/bin/main
}

compile_static(){
    recreate_dirs
    echo "Compiling objects for executable using static library..."

    # compile shared library
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/matmul.cpp -o build/obj/matmul.o
    ar rcs ${ROOT_DIR}/build/lib/libmatmul.a ${ROOT_DIR}/build/obj/matmul.o

    # compile main object
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link
    g++ ${ROOT_DIR}/build/obj/main.o -o ${ROOT_DIR}/build/bin/main_static -L${ROOT_DIR}/build/lib -lmatmul
}


compile_dynamic(){
    recreate_dirs
    g++ -std=c++17 -Iinclude -c ${ROOT_DIR}/src/matmul.cpp -o ${ROOT_DIR}/build/obj/matmul.o
    g++ -std=c++17 -shared -fPIC -Iinclude ${ROOT_DIR}/build/obj/matmul.o -o ${ROOT_DIR}/build/lib/libmatmul.so

    g++ -std=c++17 -Iinclude -c ${ROOT_DIR}/src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    g++ ${ROOT_DIR}/build/obj/main.o \
    -I${ROOT_DIR}/include \
    -L${ROOT_DIR}/build/lib \
    -lmatmul \
    -Wl,-rpath,${ROOT_DIR}/build/lib \
    -o ${ROOT_DIR}/build/bin/main_dynamic
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

