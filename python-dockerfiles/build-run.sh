#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")

# # Dockerfile name
# DOCKERFILE=python-3.12
# DOCKERFILE=python-3.12-slim
# DOCKERFILE=python-3.12-build
# DOCKERFILE=mamba
DOCKERFILE=uv


build_image(){
    docker build -f Docker/Dockerfile-${DOCKERFILE} \
                --build-arg USERNAME=$(whoami) \
                --build-arg UID=$(id -u) \
                --build-arg GID=$(id -g) \
                 -t ${DOCKERFILE}-image .
}

run_image(){
    docker run \
    -v ${THIS_DIR}:/home/$(whoami) \
    -it \
    ${DOCKERFILE}-image \
     /bin/bash
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

# TASK=build_image ./build-run.sh
# TASK=run_image ./build-run.sh