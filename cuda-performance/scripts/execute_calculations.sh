#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

DIRECTORY=${ROOT_DIR}/results
if [ ! -d ${DIRECTORY} ]; then
    mkdir ${DIRECTORY}
fi

FILENAME=${DIRECTORY}/results.csv
# rm ${FILENAME}

${ROOT_DIR}/build/bin/main --threads_per_block 512 --number_of_additions 1 --output_file ${FILENAME}
