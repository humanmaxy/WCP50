#include "yololayer.h"
#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;

class SimpleLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) TRT_NOEXCEPT override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

int main() {
    std::cout << "Testing YOLO Plugin Registration for TensorRT 10.12+" << std::endl;
    
    SimpleLogger logger;
    
    try {
        // Test plugin registry access
        auto registry = getPluginRegistry();
        if (!registry) {
            std::cerr << "Failed to get plugin registry" << std::endl;
            return 1;
        }
        std::cout << "✓ Plugin registry accessible" << std::endl;
        
        // Test plugin creator lookup
        auto creator = registry->getPluginCreator("YoloLayer_TRT", "1");
        if (!creator) {
            std::cerr << "Failed to find YoloLayer_TRT plugin creator" << std::endl;
            std::cerr << "Make sure the plugin library is properly linked and loaded" << std::endl;
            return 1;
        }
        std::cout << "✓ YoloLayer_TRT plugin creator found" << std::endl;
        
        // Test plugin creator properties
        std::cout << "Plugin name: " << creator->getPluginName() << std::endl;
        std::cout << "Plugin version: " << creator->getPluginVersion() << std::endl;
        std::cout << "Plugin namespace: " << creator->getPluginNamespace() << std::endl;
        
        // Test field names
        auto fieldCollection = creator->getFieldNames();
        if (fieldCollection) {
            std::cout << "Number of plugin fields: " << fieldCollection->nbFields << std::endl;
            for (int i = 0; i < fieldCollection->nbFields; ++i) {
                std::cout << "  Field " << i << ": " << fieldCollection->fields[i].name << std::endl;
            }
        }
        
        // Test plugin creation with minimal parameters
        PluginField pluginFields[2];
        
        // Network info
        int netInfo[5] = {15, 640, 640, 1000, 0}; // class_num, input_w, input_h, max_output, is_segmentation
        pluginFields[0].data = netInfo;
        pluginFields[0].length = 5;
        pluginFields[0].name = "netinfo";
        pluginFields[0].type = PluginFieldType::kINT32;
        
        // Simple kernel
        Yolo::YoloKernel kernel = {80, 80, {10, 13, 16, 30, 33, 23}};
        pluginFields[1].data = &kernel;
        pluginFields[1].length = 1;
        pluginFields[1].name = "kernels";
        pluginFields[1].type = PluginFieldType::kFLOAT32;
        
        PluginFieldCollection pluginData;
        pluginData.nbFields = 2;
        pluginData.fields = pluginFields;
        
        // Create plugin instance
        auto plugin = creator->createPlugin("test_yolo", &pluginData);
        if (!plugin) {
            std::cerr << "Failed to create plugin instance" << std::endl;
            return 1;
        }
        std::cout << "✓ Plugin instance created successfully" << std::endl;
        
        // Test plugin properties
        std::cout << "Plugin type: " << plugin->getPluginType() << std::endl;
        std::cout << "Plugin version: " << plugin->getPluginVersion() << std::endl;
        std::cout << "Number of outputs: " << plugin->getNbOutputs() << std::endl;
        
        // Test serialization
        size_t serialSize = plugin->getSerializationSize();
        std::cout << "Serialization size: " << serialSize << " bytes" << std::endl;
        
        // Test cloning
        auto clonedPlugin = plugin->clone();
        if (!clonedPlugin) {
            std::cerr << "Failed to clone plugin" << std::endl;
            return 1;
        }
        std::cout << "✓ Plugin cloning successful" << std::endl;
        
        // Cleanup
        clonedPlugin->destroy();
        plugin->destroy();
        
        std::cout << "\n🎉 All tests passed! YOLO plugin is working correctly with TensorRT 10.12+" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return 1;
    }
}