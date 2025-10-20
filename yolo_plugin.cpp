#include "yololayer.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace nvinfer1;

// CUDA kernel declarations
extern "C" {
    int yoloLayerV8(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, 
                    int netWidth, int netHeight, int maxOutObject, bool is_segmentation, 
                    void* workspace, cudaStream_t stream, void** anchors, int* anchor_grid, int anchor_len);
}

namespace nvinfer1 {

// Static plugin field collection for YoloPluginCreator
PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

// YoloLayerPlugin implementation
YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, bool is_segmentation, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    : mClassCount(classCount), mYoloV5NetWidth(netWidth), mYoloV5NetHeight(netHeight), 
      mMaxOutObject(maxOut), is_segmentation_(is_segmentation), mYoloKernel(vYoloKernel)
{
    mKernelCount = vYoloKernel.size();
    
    // Allocate GPU memory for anchors
    CUDA_CHECK(cudaMalloc(&mAnchor, mKernelCount * sizeof(void*)));
    
    size_t kernelSize = sizeof(Yolo::YoloKernel);
    for (int i = 0; i < mKernelCount; ++i) {
        void* anchor;
        CUDA_CHECK(cudaMalloc(&anchor, kernelSize));
        CUDA_CHECK(cudaMemcpy(anchor, &mYoloKernel[i], kernelSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void**)mAnchor + i, &anchor, sizeof(void*), cudaMemcpyHostToDevice));
    }
}

YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    
    // Deserialize parameters
    mClassCount = read<int>(d);
    mThreadCount = read<int>(d);
    mKernelCount = read<int>(d);
    mYoloV5NetWidth = read<int>(d);
    mYoloV5NetHeight = read<int>(d);
    mMaxOutObject = read<int>(d);
    is_segmentation_ = read<bool>(d);
    
    // Deserialize kernels
    mYoloKernel.resize(mKernelCount);
    for (int i = 0; i < mKernelCount; ++i) {
        mYoloKernel[i] = read<Yolo::YoloKernel>(d);
    }
    
    // Allocate GPU memory for anchors
    CUDA_CHECK(cudaMalloc(&mAnchor, mKernelCount * sizeof(void*)));
    
    size_t kernelSize = sizeof(Yolo::YoloKernel);
    for (int i = 0; i < mKernelCount; ++i) {
        void* anchor;
        CUDA_CHECK(cudaMalloc(&anchor, kernelSize));
        CUDA_CHECK(cudaMemcpy(anchor, &mYoloKernel[i], kernelSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void**)mAnchor + i, &anchor, sizeof(void*), cudaMemcpyHostToDevice));
    }
}

YoloLayerPlugin::~YoloLayerPlugin()
{
    if (mAnchor) {
        // Free individual anchor arrays
        for (int i = 0; i < mKernelCount; ++i) {
            void* anchor;
            CUDA_CHECK(cudaMemcpy(&anchor, (void**)mAnchor + i, sizeof(void*), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(anchor));
        }
        // Free anchor pointer array
        CUDA_CHECK(cudaFree(mAnchor));
        mAnchor = nullptr;
    }
}

IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT
{
    YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, 
                                           mMaxOutObject, is_segmentation_, mYoloKernel);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
{
    // Output dimensions: [batch_size, 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float)]
    return Dims2(1, 1 + mMaxOutObject * sizeof(Yolo::Detection) / sizeof(float));
}

bool YoloLayerPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
{
    // Plugin configuration
}

int YoloLayerPlugin::initialize() TRT_NOEXCEPT
{
    return 0;
}

int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
{
    return yoloLayerV8(batchSize, inputs, outputs, mYoloV5NetWidth, mYoloV5NetHeight, 
                       mMaxOutObject, is_segmentation_, workspace, stream, 
                       (void**)mAnchor, nullptr, mKernelCount);
}

size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT
{
    return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + 
           sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject) + 
           sizeof(is_segmentation_) + mKernelCount * sizeof(Yolo::YoloKernel);
}

void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT
{
    char* d = reinterpret_cast<char*>(buffer);
    
    write(d, mClassCount);
    write(d, mThreadCount);
    write(d, mKernelCount);
    write(d, mYoloV5NetWidth);
    write(d, mYoloV5NetHeight);
    write(d, mMaxOutObject);
    write(d, is_segmentation_);
    
    for (int i = 0; i < mKernelCount; ++i) {
        write(d, mYoloKernel[i]);
    }
}

const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT
{
    return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT
{
    return "1";
}

void YoloLayerPlugin::destroy() TRT_NOEXCEPT
{
    delete this;
}

void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
{
    mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT
{
    return mPluginNamespace;
}

DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
{
    return DataType::kFLOAT;
}

bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
{
    return false;
}

bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
{
    return false;
}

void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
{
    // No additional context needed
}

void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT
{
    // No context to detach
}

// YoloPluginCreator implementation
YoloPluginCreator::YoloPluginCreator()
{
    mPluginAttributes.clear();
    
    mPluginAttributes.emplace_back(PluginField("netinfo", nullptr, PluginFieldType::kFLOAT32, 5));
    mPluginAttributes.emplace_back(PluginField("kernels", nullptr, PluginFieldType::kFLOAT32, 1));
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const TRT_NOEXCEPT
{
    return "YoloLayer_TRT";
}

const char* YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT
{
    return "1";
}

const PluginFieldCollection* YoloPluginCreator::getFieldNames() TRT_NOEXCEPT
{
    return &mFC;
}

IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
{
    const PluginField* fields = fc->fields;
    
    int* netinfo = nullptr;
    Yolo::YoloKernel* kernels = nullptr;
    int kernels_size = 0;
    
    for (int i = 0; i < fc->nbFields; ++i) {
        if (strcmp(fields[i].name, "netinfo") == 0) {
            netinfo = (int*)fields[i].data;
        } else if (strcmp(fields[i].name, "kernels") == 0) {
            kernels = (Yolo::YoloKernel*)fields[i].data;
            kernels_size = fields[i].length;
        }
    }
    
    if (!netinfo || !kernels) {
        return nullptr;
    }
    
    int classCount = netinfo[0];
    int netWidth = netinfo[1];
    int netHeight = netinfo[2];
    int maxOut = netinfo[3];
    bool is_segmentation = (bool)netinfo[4];
    
    std::vector<Yolo::YoloKernel> vYoloKernel(kernels, kernels + kernels_size);
    
    YoloLayerPlugin* obj = new YoloLayerPlugin(classCount, netWidth, netHeight, maxOut, is_segmentation, vYoloKernel);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
{
    YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

// Helper template functions for serialization
template<typename T>
void YoloLayerPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T>
T YoloLayerPlugin::read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

} // namespace nvinfer1

// Plugin registration for TensorRT 10.12+
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);