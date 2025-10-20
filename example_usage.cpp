#include "yololayer.h"
#include <NvInfer.h>
#include <iostream>
#include <memory>

using namespace nvinfer1;

// Example of how to use the YOLO plugin in your TensorRT network
class YoloPluginExample {
public:
    bool buildNetwork() {
        // Create builder, network, and config
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger_));
        auto network = std::unique_ptr<INetworkDefinition>(
            builder->createNetworkV2(0U));
        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());

        // Add your network layers here...
        // For example, add input layer
        auto input = network->addInput("input", DataType::kFLOAT, 
                                     Dims3{3, Yolo::INPUT_H, Yolo::INPUT_W});
        
        // Add your backbone network layers (ResNet, DarkNet, etc.)
        // ...
        
        // Add YOLO detection heads (assuming you have 3 detection layers)
        std::vector<IConvolutionLayer*> detectionLayers;
        // ... create your detection layers and add to detectionLayers
        
        // Add YOLO plugin layer
        auto yoloLayer = addYoloLayer(network.get(), detectionLayers);
        if (!yoloLayer) {
            std::cerr << "Failed to add YOLO layer" << std::endl;
            return false;
        }
        
        // Mark output
        yoloLayer->getOutput(0)->setName("output");
        network->markOutput(*yoloLayer->getOutput(0));
        
        // Build engine
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB
        auto engine = std::unique_ptr<ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config));
        
        if (!engine) {
            std::cerr << "Failed to build engine" << std::endl;
            return false;
        }
        
        std::cout << "Successfully built TensorRT engine with YOLO plugin" << std::endl;
        return true;
    }

private:
    IPluginV2Layer* addYoloLayer(INetworkDefinition* network, 
                                const std::vector<IConvolutionLayer*>& detLayers) {
        // Get plugin creator
        auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
        if (!creator) {
            std::cerr << "Could not find YoloLayer_TRT plugin creator" << std::endl;
            return nullptr;
        }
        
        // Prepare plugin fields
        PluginField pluginFields[2];
        
        // Network info: [class_num, input_w, input_h, max_output_bbox_count, is_segmentation]
        int netInfo[5] = {Yolo::CLASS_NUM, Yolo::INPUT_W, Yolo::INPUT_H, 
                         Yolo::MAX_OUTPUT_BBOX_COUNT, 0};
        pluginFields[0].data = netInfo;
        pluginFields[0].length = 5;
        pluginFields[0].name = "netinfo";
        pluginFields[0].type = PluginFieldType::kINT32;
        
        // YOLO kernels (anchors and grid sizes)
        std::vector<Yolo::YoloKernel> kernels;
        // Example kernels for 3 scales (you should use your actual anchor values)
        Yolo::YoloKernel kernel1 = {80, 80, {10, 13, 16, 30, 33, 23}};
        Yolo::YoloKernel kernel2 = {40, 40, {30, 61, 62, 45, 59, 119}};
        Yolo::YoloKernel kernel3 = {20, 20, {116, 90, 156, 198, 373, 326}};
        
        kernels.push_back(kernel1);
        kernels.push_back(kernel2);
        kernels.push_back(kernel3);
        
        pluginFields[1].data = kernels.data();
        pluginFields[1].length = kernels.size();
        pluginFields[1].name = "kernels";
        pluginFields[1].type = PluginFieldType::kFLOAT32;
        
        // Create plugin field collection
        PluginFieldCollection pluginData;
        pluginData.nbFields = 2;
        pluginData.fields = pluginFields;
        
        // Create plugin
        auto plugin = creator->createPlugin("yolo", &pluginData);
        if (!plugin) {
            std::cerr << "Failed to create YOLO plugin" << std::endl;
            return nullptr;
        }
        
        // Prepare input tensors
        std::vector<ITensor*> inputTensors;
        for (auto detLayer : detLayers) {
            inputTensors.push_back(detLayer->getOutput(0));
        }
        
        // Add plugin layer to network
        auto yoloLayer = network->addPluginV2(inputTensors.data(), 
                                            inputTensors.size(), *plugin);
        
        return yoloLayer;
    }
    
    class Logger : public ILogger {
    public:
        void log(Severity severity, const char* msg) TRT_NOEXCEPT override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    } logger_;
};

int main() {
    std::cout << "YOLO Plugin Example for TensorRT 10.12+" << std::endl;
    
    YoloPluginExample example;
    if (example.buildNetwork()) {
        std::cout << "Example completed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "Example failed!" << std::endl;
        return 1;
    }
}