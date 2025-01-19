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
    g++ -std=c++17 -Iinclude -c src/module1/module1c1.cpp -o ${ROOT_DIR}/build/obj/moudle1c1.o
    g++ -std=c++17 -Iinclude -c src/module1/module1c2.cpp -o ${ROOT_DIR}/build/obj/moudle1c2.o
    g++ -std=c++17 -Iinclude -c src/module2/module2c1.cpp -o ${ROOT_DIR}/build/obj/moudle2c1.o
    g++ -std=c++17 -Iinclude -c src/module2/module2c2.cpp -o ${ROOT_DIR}/build/obj/moudle2c2.o

    # compile the main to object
    g++ -std=c++17 -Iinclude -c src/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link all the objects
    g++ ${ROOT_DIR}/build/obj/moudle1c1.o \
        ${ROOT_DIR}/build/obj/moudle1c2.o \
        ${ROOT_DIR}/build/obj/moudle2c1.o \
        ${ROOT_DIR}/build/obj/moudle2c2.o \
        ${ROOT_DIR}/build/obj/main.o \
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