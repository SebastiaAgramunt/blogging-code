#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

DIRECTORY=${ROOT_DIR}/data
if [ ! -d ${DIRECTORY} ]; then
    mkdir ${DIRECTORY}
fi

FILENAME=${DIRECTORY}/output.csv
rm ${FILENAME}

# for loop from 2 to 1024 in increments of 2
for ((i=2;i<=512;i+=2)); do
    ${ROOT_DIR}/build/bin/main  --matrix_size ${i} --output_file ${FILENAME}
done


# ${ROOT_DIR}/build/bin/main  --matrix_size 256    --output_file ${FILENAME} # 2^8
# ${ROOT_DIR}/build/bin/main  --matrix_size 512    --output_file ${FILENAME} # 2^9
${ROOT_DIR}/build/bin/main  --matrix_size 1024   --output_file ${FILENAME} # 2^10

# # now in increments of 1024 without cpu (takes too long!)
${ROOT_DIR}/build/bin/main  --matrix_size 2048   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 3072   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 4096   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 5120   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 6144   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 7168   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 8192   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 9216   --no_cpu --output_file ${FILENAME}
${ROOT_DIR}/build/bin/main  --matrix_size 10240  --no_cpu --output_file ${FILENAME}
