#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "yololayer.h"

#define CUDA_CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            printf("Cuda failure: %s at line %d in file %s\n", cudaGetErrorString(ret), __LINE__, __FILE__); \
        } \
    } while (0)

__device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void yoloKernel(const float *input, float *output, int noElements, 
                          int yoloWidth, int yoloHeight, const float *anchors, int classes, 
                          int outputElem, int netWidth, int netHeight, int maxOut) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= noElements) return;

    int total_grid = yoloWidth * yoloHeight;
    int bnIdx = idx / total_grid;
    idx = idx - total_grid * bnIdx;
    int info_len_i = 5 + classes;
    const float* curInput = input + bnIdx * (info_len_i * total_grid * 3);

    for (int k = 0; k < 3; ++k) {
        float box_prob = sigmoidGPU(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
        if (box_prob < 0.1f) continue;

        int class_id = 0;
        float max_cls_prob = 0.0f;
        for (int i = 5; i < info_len_i; ++i) {
            float p = sigmoidGPU(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
            if (p > max_cls_prob) {
                max_cls_prob = p;
                class_id = i - 5;
            }
        }
        
        float *res_count = output + bnIdx * outputElem;
        int count = (int)atomicAdd(res_count, 1.0f);
        if (count >= maxOut) return;
        
        char* data = (char*)res_count + sizeof(float) + count * sizeof(Yolo::Detection);
        Yolo::Detection* det = (Yolo::Detection*)(data);

        int row = idx / yoloWidth;
        int col = idx % yoloWidth;

        // Decode bounding box
        det->bbox[0] = (col + sigmoidGPU(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netWidth / yoloWidth;
        det->bbox[1] = (row + sigmoidGPU(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netHeight / yoloHeight;
        det->bbox[2] = expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2 * k];
        det->bbox[3] = expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2 * k + 1];
        det->conf = box_prob * max_cls_prob;
        det->class_id = class_id;
    }
}

extern "C" int yoloLayerV8(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, 
                          int netWidth, int netHeight, int maxOutObject, bool is_segmentation, 
                          void* workspace, cudaStream_t stream, void** anchors, int* anchor_grid, int anchor_len) {
    
    int outputElem = 1 + maxOutObject * sizeof(Yolo::Detection) / sizeof(float);
    
    for (int idx = 0; idx < batchSize; ++idx) {
        CUDA_CHECK(cudaMemsetAsync((float*)outputs[0] + idx * outputElem, 0, sizeof(float), stream));
    }
    
    int numElem = 0;
    for (int i = 0; i < anchor_len; ++i) {
        const float* input = (const float*)inputs[i];
        float* output = (float*)outputs[0];
        
        // Calculate grid dimensions based on input tensor dimensions
        // This is a simplified version - you may need to adjust based on your specific model
        int yoloWidth = netWidth / (8 << i);  // Assuming stride of 8, 16, 32
        int yoloHeight = netHeight / (8 << i);
        
        numElem = yoloWidth * yoloHeight * batchSize;
        
        // Get anchors for this scale
        float* scale_anchors;
        CUDA_CHECK(cudaMemcpy(&scale_anchors, (void**)anchors + i, sizeof(void*), cudaMemcpyDeviceToHost));
        
        if (numElem < 512) {
            yoloKernel<<<1, numElem, 0, stream>>>(
                input, output, numElem, yoloWidth, yoloHeight, 
                (float*)scale_anchors, Yolo::CLASS_NUM, outputElem, 
                netWidth, netHeight, maxOutObject);
        } else {
            yoloKernel<<<(numElem + 512 - 1) / 512, 512, 0, stream>>>(
                input, output, numElem, yoloWidth, yoloHeight, 
                (float*)scale_anchors, Yolo::CLASS_NUM, outputElem, 
                netWidth, netHeight, maxOutObject);
        }
    }
    
    return 0;
}