#!/bin/bash

# not overcomplicating python environment. Use default python in the system
THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

rm -rf ${ROOT_DIR}/.venv
python -m venv ${ROOT_DIR}/.venv
${ROOT_DIR}/.venv/bin/pip install --upgrade pip
${ROOT_DIR}/.venv/bin/pip install numpy matplotlib pandas nvitop
