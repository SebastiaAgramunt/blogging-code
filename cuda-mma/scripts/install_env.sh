#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")

rm -rf "$THIS_DIR"/.venv
uv venv "$THIS_DIR"/.venv --python 3.12

cd "$THIS_DIR" && uv pip install matplotlib path pandas