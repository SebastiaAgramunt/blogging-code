#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

mkdir ${ROOT_DIR}/img
wget "https://upload.wikimedia.org/wikipedia/commons/2/28/20100723_Miyajima_4904.jpg" -O ${ROOT_DIR}/img/raw_img.jpeg
