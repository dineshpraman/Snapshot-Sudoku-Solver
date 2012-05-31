#!/bin/bash
PKG_CONFIG_PATH=/usr/local/lib/pkg-config:${PKG_CONFIG_PATH}
export PKG_CONFIG_PATH
g++ `pkg-config opencv --libs --cflags` sudoku.cpp  basicOCR.cpp preprocessing.c -o code
./code sudoku.jpg
