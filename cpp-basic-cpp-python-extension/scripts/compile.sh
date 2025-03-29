#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

VENV_BUILD=.venv_build
VENV=.venv

compile_exec(){

    echo "Compiling objects for executable..."
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/matmul.cpp -o ${ROOT_DIR}/build/obj/matmul.o
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/main.cpp -o ${ROOT_DIR}/build/obj/main.o
    
    # link all the objects
    g++ ${ROOT_DIR}/build/obj/matmul.o \
        ${ROOT_DIR}/build/obj/main.o \
        -o ${ROOT_DIR}/build/bin/main
}

compile_python_lib(){

    # creating new building environment
    rm -rf ${ROOT_DIR}/${VENV_BUILD}
    python -m venv ${ROOT_DIR}/${VENV_BUILD}
    ${ROOT_DIR}/${VENV_BUILD}/bin/pip install --upgrade pip

    # install pybind (a header only library to create the bindings)
    ${ROOT_DIR}/${VENV_BUILD}/bin/python -m pip install pybind11

    # pybind includes. The headers of python and pybind11 in my mac
    # -I/Users/sebas/.pyenv/versions/3.12.4/include/python3.12 \
    # -I/Users/sebas/tmp/blogging-code/cpp-basic-cpp-python-extension/.venv_build/lib/python3.12/site-packages/pybind11/include
    pybind_includes=$(${VENV_BUILD}/bin/python -m pybind11 --includes)

    # ld flags for default python. In my mac
    # -lintl -ldl -L/Users/sebas/.pyenv/versions/3.12.4/lib \
    # -Wl,-rpath,/Users/sebas/.pyenv/versions/3.12.4/lib \
    # -framework CoreFoundation
    python_ldflags=$(python3-config --ldflags)

    # get python library, should be something like python3.12
    # we will add this to the linker as the library -lpython3.12 which is python library
    # libpython3.12.dylib on MacOS / libpython3.12.so on Linux
    python_library=python$(${VENV_BUILD}/bin/python --version | awk '{print $2}' | awk -F. '{print $1"."$2}')

    # compile matmul and bindings. Recall we add the python and pybind includes in the $pybind_includes library
    g++ -std=c++17 -I${ROOT_DIR}/include -c ${ROOT_DIR}/src/matmul.cpp -o ${ROOT_DIR}/build/obj/matmul.o
    g++ -std=c++17 -I${ROOT_DIR}/include $pybind_includes -c ${ROOT_DIR}/src/bindings.cpp -o ${ROOT_DIR}/build/obj/bindings.o

    # compile the shared object (the python library) and place it in build/lib
    g++ -O3 -Wall -shared -std=c++17 -fPIC \
        $python_ldflags \
        -l${python_library} \
        ${ROOT_DIR}/build/obj/matmul.o \
        ${ROOT_DIR}/build/obj/bindings.o \
        -o build/lib/matrix_mul$(python3-config --extension-suffix)
}

test(){
    # create a fresh virtual environment without pybind11, is not needed as user
    rm -rf ${ROOT_DIR}/${VENV}
    python -m venv ${ROOT_DIR}/${VENV}
    ${ROOT_DIR}/${VENV}/bin/pip install --upgrade pip

    # install numpy, required for the script tests/test_matmul.py
    ${ROOT_DIR}/${VENV}/bin/pip install numpy

    # run the test
    ${ROOT_DIR}/${VENV}/bin/python ${ROOT_DIR}/tests/test_matmul.py
}

clean(){
    # removing build directory
    echo "Removing ${ROOT_DIR}/build and recreating..."
    rm -rf ${ROOT_DIR}/build
    rm -rf  ${ROOT_DIR}/${VENV_BUILD}
    rm -rf  ${ROOT_DIR}/${VENV}

    # creating directories for the build
    mkdir -p ${ROOT_DIR}/build/obj
    mkdir ${ROOT_DIR}/build/bin
    mkdir ${ROOT_DIR}/build/lib
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