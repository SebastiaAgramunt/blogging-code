#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})


${ROOT_DIR}/build/bin/vector_add_types  256 1 float
${ROOT_DIR}/build/bin/vector_add_types  256 4 float
${ROOT_DIR}/build/bin/vector_add_types  256 20 float
${ROOT_DIR}/build/bin/vector_add_types  256 40 float