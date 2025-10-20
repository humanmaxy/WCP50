cmake_minimum_required(VERSION 3.18)
project(YoloPlugin LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

# TensorRT paths (adjust these paths according to your TensorRT installation)
set(TENSORRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT root directory")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

# Include directories
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# Link directories
link_directories(${TENSORRT_LIB_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

# Source files
set(PLUGIN_SOURCES
    yolo_plugin.cpp
    yolo_kernel.cu
    plugin_registry.cpp
)

# Create shared library
add_library(yolo_plugin SHARED ${PLUGIN_SOURCES})

# Link libraries
target_link_libraries(yolo_plugin
    nvinfer
    nvonnxparser
    cudart
    cublas
    curand
)

# Set output properties
set_target_properties(yolo_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create test executable
add_executable(test_plugin test_plugin.cpp)
target_link_libraries(test_plugin yolo_plugin nvinfer)

# Create example executable
add_executable(example_usage example_usage.cpp)
target_link_libraries(example_usage yolo_plugin nvinfer)

# Install targets
install(TARGETS yolo_plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES yololayer.h
    DESTINATION include
)