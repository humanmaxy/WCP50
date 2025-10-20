# Migration Guide: YOLO Plugin for TensorRT 10.12+

## Overview

This guide helps you migrate from the old YOLO plugin implementation that was causing access violations in TensorRT 10.12+ to the new compatible implementation.

## The Problem

The original plugin registration code:

```cpp
class PluginRegistrar
{
public:
    PluginRegistrar()
    {
        getPluginRegistry()->registerCreator(instance, "");  // ❌ Access violation
    }
private:
    T instance{};
};

#define REGISTER_TENSORRT_PLUGIN(name) \
    static nvinfer1::PluginRegistrar<name> pluginRegistrar##name {}
```

This caused access violations because:
1. The plugin registry wasn't properly initialized during static construction
2. The `instance` object was being passed by value instead of pointer
3. TensorRT 10.12+ changed the internal plugin registration mechanism

## The Solution

### 1. Replace Old Registration System

**Old (❌ Broken):**
```cpp
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
```

**New (✅ Working):**
```cpp
// In plugin_registry.cpp
template<typename T>
class PluginRegistrar {
public:
    PluginRegistrar() {
        T* creator = new T();
        PluginRegistry::getInstance().registerCreator(creator, "");
    }
};

REGISTER_TENSORRT_PLUGIN(nvinfer1::YoloPluginCreator);
```

### 2. Update Plugin Implementation

**Key changes needed:**

1. **Add serialization helpers:**
```cpp
private:
    template<typename T>
    void write(char*& buffer, const T& val) const;
    
    template<typename T>
    T read(const char*& buffer);
```

2. **Proper memory management:**
```cpp
YoloLayerPlugin::~YoloLayerPlugin() {
    if (mAnchor) {
        for (int i = 0; i < mKernelCount; ++i) {
            void* anchor;
            CUDA_CHECK(cudaMemcpy(&anchor, (void**)mAnchor + i, sizeof(void*), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(anchor));
        }
        CUDA_CHECK(cudaFree(mAnchor));
        mAnchor = nullptr;
    }
}
```

3. **Complete interface implementation:**
```cpp
// All virtual functions must be implemented
bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override;
void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;
// ... etc
```

### 3. Update Build System

**Add to CMakeLists.txt:**
```cmake
# Source files
set(PLUGIN_SOURCES
    yolo_plugin.cpp
    yolo_kernel.cu
    plugin_registry.cpp  # New registration system
)

# Create shared library
add_library(yolo_plugin SHARED ${PLUGIN_SOURCES})

# Link libraries for TensorRT 10.12+
target_link_libraries(yolo_plugin
    nvinfer
    nvonnxparser
    cudart
)
```

### 4. Update Usage Code

**No changes needed in usage code!** The plugin is used exactly the same way:

```cpp
auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
auto plugin = creator->createPlugin("yolo", &pluginData);
auto yoloLayer = network->addPluginV2(inputTensors.data(), inputTensors.size(), *plugin);
```

## Step-by-Step Migration

### Step 1: Backup Your Code
```bash
cp yololayer.h yololayer.h.backup
cp your_plugin_implementation.cpp your_plugin_implementation.cpp.backup
```

### Step 2: Replace Files
1. Replace `yololayer.h` with the new header
2. Add the new implementation files:
   - `yolo_plugin.cpp`
   - `yolo_kernel.cu`
   - `plugin_registry.cpp`

### Step 3: Update Build System
1. Update your CMakeLists.txt or Makefile
2. Ensure you're linking against TensorRT 10.12+ libraries
3. Add CUDA compilation for `.cu` files

### Step 4: Build and Test
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

# Test the plugin
./test_plugin
```

### Step 5: Verify Integration
1. Run your existing application
2. Check that YOLO detection works correctly
3. Verify no access violations occur

## Troubleshooting

### Common Issues

1. **Plugin not found:**
   ```
   Could not find YoloLayer_TRT plugin creator
   ```
   **Solution:** Ensure the plugin library is properly linked and loaded.

2. **CUDA errors:**
   ```
   Cuda failure: invalid device pointer
   ```
   **Solution:** Check CUDA memory allocation in the plugin constructor.

3. **Serialization errors:**
   ```
   Plugin serialization failed
   ```
   **Solution:** Ensure all member variables are properly serialized/deserialized.

### Debug Tips

1. **Enable verbose logging:**
```cpp
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) TRT_NOEXCEPT override {
        std::cout << "[TensorRT] " << msg << std::endl;  // Log everything
    }
};
```

2. **Test plugin creation separately:**
```cpp
// Test just plugin creation without network building
auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
assert(creator != nullptr);
```

3. **Check memory usage:**
```bash
nvidia-smi  # Monitor GPU memory usage
valgrind --tool=memcheck ./your_app  # Check for memory leaks
```

## Performance Notes

The new implementation should have similar or better performance:
- Optimized CUDA kernels
- Better memory management
- Reduced memory fragmentation
- Compatible with latest TensorRT optimizations

## Compatibility Matrix

| TensorRT Version | Old Plugin | New Plugin |
|------------------|------------|------------|
| 8.x              | ✅         | ✅         |
| 9.x              | ⚠️         | ✅         |
| 10.0-10.11       | ❌         | ✅         |
| 10.12+           | ❌         | ✅         |

## Support

If you encounter issues during migration:

1. Check the test program: `./test_plugin`
2. Verify TensorRT version: `dpkg -l | grep tensorrt`
3. Check CUDA compatibility: `nvcc --version`
4. Review the example usage in `example_usage.cpp`

The new implementation is fully backward compatible and should be a drop-in replacement for most use cases.