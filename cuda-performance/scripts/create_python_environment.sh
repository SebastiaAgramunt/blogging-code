#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})


# remove env and recreate
rm -rf ${THIS_DIR}/.venv

python -m venv ${THIS_DIR}/.venv
${THIS_DIR}/.venv/bin/python -m pip install --upgrade pip
${THIS_DIR}/.venv/bin/python -m pip install -r ${THIS_DIR}/requirements.txt