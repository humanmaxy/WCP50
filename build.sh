#!/bin/bash

# Build script for YOLO Plugin compatible with TensorRT 10.12

# Set TensorRT path (adjust according to your installation)
export TENSORRT_ROOT="/usr/local/TensorRT"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT=${TENSORRT_ROOT} \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Build
make -j$(nproc)

echo "Build completed. Library created: build/libyolo_plugin.so"
echo "To use this plugin, link against libyolo_plugin.so and include yololayer.h"