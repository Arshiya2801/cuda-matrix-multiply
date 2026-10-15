#!/bin/bash

BUILD_DIR=build

echo "=== CUDA Matrix Multiply: Build & Run ==="

mkdir -p $BUILD_DIR
cd $BUILD_DIR || exit

echo "--- Generating build files with CMake ---"
cmake ..

echo "--- Building project ---"
cmake --build .

if [ -f Debug/matrix_mul.exe ]; then
    echo "--- Running main ---"
    ./Debug/matrix_mul.exe
elif [ -f Release/matrix_mul.exe ]; then
    echo "--- Running main ---"
    ./Release/matrix_mul.exe
elif [ -f matrix_mul ]; then
    echo "--- Running main ---"
    ./matrix_mul
else
    echo "--- Running main ---"
    ./matrix_mul.exe
fi

if [ -f Debug/matrix_mul.exe ]; then
    echo "--- Running tests ---"
    ./Debug/matrix_mul.exe --test
elif [ -f Release/matrix_mul.exe ]; then
    echo "--- Running tests ---"
    ./Release/matrix_mul.exe --test
elif [ -f matrix_mul ]; then
    echo "--- Running tests ---"
    ./matrix_mul --test
fi

echo "=== Done ==="
