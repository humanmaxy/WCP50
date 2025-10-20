#include "yololayer.h"
#include <NvInfer.h>
#include <memory>

// TensorRT 10.12 compatible plugin registration
namespace nvinfer1 {

// Global plugin registry instance
class PluginRegistry {
public:
    static PluginRegistry& getInstance() {
        static PluginRegistry instance;
        return instance;
    }
    
    void registerCreator(IPluginCreator* creator, const std::string& pluginNamespace = "") {
        if (creator) {
            creators_.push_back(std::unique_ptr<IPluginCreator>(creator));
            // Register with TensorRT's plugin registry
            getPluginRegistry()->registerCreator(*creator, pluginNamespace.c_str());
        }
    }
    
private:
    std::vector<std::unique_ptr<IPluginCreator>> creators_;
};

// Plugin registrar template for TensorRT 10.12+
template<typename T>
class PluginRegistrar {
public:
    PluginRegistrar() {
        static_assert(std::is_base_of<IPluginCreator, T>::value, 
                     "T must inherit from IPluginCreator");
        
        // Create plugin creator instance
        T* creator = new T();
        
        // Register with our registry (which will handle TensorRT registration)
        PluginRegistry::getInstance().registerCreator(creator, "");
    }
};

} // namespace nvinfer1

// Macro for plugin registration compatible with TensorRT 10.12
#undef REGISTER_TENSORRT_PLUGIN
#define REGISTER_TENSORRT_PLUGIN(name) \
    static nvinfer1::PluginRegistrar<name> pluginRegistrar##name{};

// Register the YoloPluginCreator
REGISTER_TENSORRT_PLUGIN(nvinfer1::YoloPluginCreator);