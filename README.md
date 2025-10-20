# YOLO Plugin for TensorRT 10.12+

This repository provides a TensorRT 10.12+ compatible implementation of the YOLO plugin that was previously causing access violations due to deprecated plugin registration methods.

## Problem Solved

The original error was caused by the deprecated `PluginRegistrar` template and `REGISTER_TENSORRT_PLUGIN` macro in TensorRT 10.12+. The old registration system:

```cpp
template<class T>
class PluginRegistrar
{
public:
    PluginRegistrar()
    {
        getPluginRegistry()->registerCreator(instance, "");  // This causes access violation
    }
private:
    T instance{};
};
```

This implementation caused a null pointer dereference because the plugin registry wasn't properly initialized when the static constructor was called.

## Solution

This implementation provides:

1. **Compatible Plugin Registration**: New registration system that works with TensorRT 10.12+
2. **Proper Memory Management**: Correct CUDA memory allocation and deallocation
3. **Complete Plugin Implementation**: Full implementation of IPluginV2IOExt interface
4. **CUDA Kernel**: Optimized CUDA kernel for YOLO inference

## Files Structure

- `yololayer.h` - Updated header with TensorRT 10.12+ compatibility
- `yolo_plugin.cpp` - Complete plugin implementation
- `yolo_kernel.cu` - CUDA kernel for YOLO inference
- `plugin_registry.cpp` - New plugin registration system
- `CMakeLists.txt` - Build configuration
- `build.sh` - Build script

## Building

1. Ensure you have TensorRT 10.12+ installed
2. Update the TensorRT path in `build.sh` if needed
3. Run the build script:

```bash
./build.sh
```

## Usage

1. Link against the generated `libyolo_plugin.so`
2. Include the header file:

```cpp
#include "yololayer.h"
```

3. The plugin will be automatically registered when the library is loaded
4. Use it in your TensorRT network:

```cpp
auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
// ... create plugin as usual
```

## Key Changes from Original

1. **Plugin Registration**: Uses new registration system compatible with TensorRT 10.12+
2. **Memory Safety**: Proper CUDA memory management with error checking
3. **Template Functions**: Added serialization helper templates
4. **Complete Implementation**: All virtual functions properly implemented
5. **CUDA Kernel**: Optimized kernel implementation for better performance

## Compatibility

- TensorRT 10.12+
- CUDA 11.0+
- C++14 or later

## Notes

- The plugin automatically handles different input scales (typical YOLO multi-scale detection)
- Memory allocation is properly managed to prevent leaks
- Error checking is implemented for all CUDA operations
- The plugin supports both segmentation and detection modes

This implementation resolves the access violation error and provides a robust, production-ready YOLO plugin for modern TensorRT versions.