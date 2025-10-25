#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

DIRECTORY=${ROOT_DIR}/results
if [ ! -d ${DIRECTORY} ]; then
    mkdir ${DIRECTORY}
fi

FILENAME=${DIRECTORY}/results.csv
rm ${FILENAME}

"${ROOT_DIR}/build/bin/main" --threads_per_block 512 --number_of_additions 1 --use_cpu 1 --output_file "${FILENAME}"
"${ROOT_DIR}/build/bin/main" --threads_per_block 512 --number_of_additions 256 --use_cpu 1 --output_file "${FILENAME}"

# calculate n_additions without CPU compute (takes too long)
n_additions=(512 1024 2048 4096 8192 16384)

for n in "${n_additions[@]}"; do
    "${ROOT_DIR}/build/bin/main" \
        --threads_per_block 512 \
        --number_of_additions "$n" \
        --use_cpu 0 \
        --output_file "${FILENAME}"
done
