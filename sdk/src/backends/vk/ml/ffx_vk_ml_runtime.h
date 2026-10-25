// SPDX-FileCopyrightText: Copyright 2026 Neural Graphics SDK universal contributors
// SPDX-License-Identifier: MIT

// Native compute execution of TOSA data graphs for devices without VK_ARM_data_graph.
//
// The runtime lowers the (int8, NHWC) TOSA graphs shipped with the SDK to a short list of compute dispatches and picks
// the kernel that best matches the vendor's own ML acceleration:
//
//   CoopMat  VK_KHR_cooperative_matrix implicit GEMM. NVIDIA tensor cores, AMD RDNA3+ WMMA, Intel Xe XMX. The matrix
//            element type is chosen from what the driver reports and the app enabled: int8, then fp16, then bf16.
//            All three are bit exact for the int8 networks; lossy types such as fp8 are deliberately not used.
//   Dot4     VK_KHR_shader_integer_dot_product packed int8 dot products (e.g. AMD RDNA2, NVIDIA Pascal, Intel Xe-LP).
//   Scalar   Plain 32-bit integer math, needs no optional device features.
//
// Devices that do expose VK_ARM_data_graph keep using the data-graph pipeline and never reach this code.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ffx_ml
{
    // Device features the application enabled on its VkDevice. Vulkan cannot report this, so it is passed in.
    // The values match FFX_VK_ML_FEATURE_* in ffx_vk.h.
    enum FeatureFlags : uint32_t
    {
        FEATURE_SHADER_INT8               = 1u << 0,  // VkPhysicalDeviceShaderFloat16Int8Features::shaderInt8
        FEATURE_STORAGE_BUFFER_8BIT       = 1u << 1,  // VkPhysicalDevice8BitStorageFeatures::storageBuffer8BitAccess
        FEATURE_SHADER_FLOAT16            = 1u << 2,  // VkPhysicalDeviceShaderFloat16Int8Features::shaderFloat16
        FEATURE_STORAGE_BUFFER_16BIT      = 1u << 3,  // VkPhysicalDevice16BitStorageFeatures::storageBuffer16BitAccess
        FEATURE_INTEGER_DOT_PRODUCT       = 1u << 4,  // shaderIntegerDotProduct
        FEATURE_COOPERATIVE_MATRIX        = 1u << 5,  // VK_KHR_cooperative_matrix + cooperativeMatrix
        FEATURE_SHADER_BFLOAT16           = 1u << 6,  // VK_KHR_shader_bfloat16 shaderBFloat16Type + shaderBFloat16CooperativeMatrix
        FEATURE_SUBGROUP_SIZE_CONTROL     = 1u << 8,  // subgroupSizeControl + computeFullSubgroups
        FEATURE_VULKAN_MEMORY_MODEL       = 1u << 9,  // vulkanMemoryModel (required by the coop-matrix shaders)
    };

    enum class Path : uint32_t
    {
        Auto = 0,
        CoopMat,
        Dot4,
        Scalar,
    };

    enum class MatType : uint32_t
    {
        None = 0,
        Int8,
        Fp16,
        Bf16,
    };

    enum class Vendor : uint32_t
    {
        Unknown = 0,
        Nvidia,
        Amd,
        Intel,
        Arm,
        Qualcomm,
        Other,
    };

    struct DeviceContext
    {
        VkInstance                instance;
        VkPhysicalDevice          physicalDevice;
        VkDevice                  device;
        PFN_vkGetInstanceProcAddr getInstanceProcAddr;
        PFN_vkGetDeviceProcAddr   getDeviceProcAddr;
        uint32_t                  enabledFeatures;  // FeatureFlags
    };

    // The kernel configuration picked for a device.
    struct Selection
    {
        Path     path         = Path::Scalar;
        MatType  matType      = MatType::None;
        Vendor   vendor       = Vendor::Unknown;
        uint32_t vendorId     = 0;
        uint32_t tileM        = 0;
        uint32_t tileN        = 0;
        uint32_t tileK        = 0;
        uint32_t numSubgroups = 0;   // subgroups per workgroup (coop-matrix)
        uint32_t subgroupSize = 0;   // subgroup size the kernel is built for
        bool     requireSubgroupSize = false;
        uint32_t workgroupSize = 0;  // threads per workgroup (dot4 / scalar)
        std::string deviceName;

        std::string Describe() const;
    };

    const char* ToString(Path path);
    const char* ToString(MatType type);
    const char* ToString(Vendor vendor);

    // Pick the fastest supported path for the device. forcedPath / forcedType restrict the choice (used for testing
    // and the FFX_ML_PATH / FFX_ML_MATTYPE environment overrides); an impossible request falls back down the ladder.
    Selection SelectPath(const DeviceContext& context, Path forcedPath = Path::Auto, MatType forcedType = MatType::None);

    // FeatureFlags the physical device supports, plus the device extensions to enable for them.
    uint32_t QueryDeviceFeatures(VkInstance instance, VkPhysicalDevice physicalDevice, PFN_vkGetInstanceProcAddr getInstanceProcAddr, std::vector<const char*>* extensions);

    // Apply the FFX_ML_PATH (coopmat|dot4|scalar) and FFX_ML_MATTYPE (int8|fp16|bf16) environment overrides.
    void ReadEnvironmentOverrides(Path& path, MatType& type);

    struct GraphConstant
    {
        uint32_t             id;
        uint32_t             vkFormat;
        std::vector<int64_t> shape;
        const uint8_t*       data;
        size_t               dataSize;
    };

    struct GraphDesc
    {
        const uint32_t*                          spirv;       // TOSA graph module (as shipped in the VGF)
        size_t                                   spirvWords;
        std::vector<GraphConstant>               constants;
        std::map<uint32_t, std::vector<int64_t>> inputShapes; // binding -> NHWC shape; spatial dims resolved
    };

    struct ExternalTensor
    {
        uint32_t     binding;
        VkBuffer     buffer;
        VkDeviceSize offset;
    };

    class Graph
    {
    public:
        static std::unique_ptr<Graph> Create(const DeviceContext& context, const Selection& selection, const GraphDesc& desc, std::string* error);
        ~Graph();

        // Records the whole graph. The caller makes inputs visible to compute shader reads before, and synchronises
        // compute shader writes to the outputs after. Bindings follow the graph's descriptor bindings.
        void Record(VkCommandBuffer commandBuffer, const ExternalTensor* tensors, uint32_t tensorCount);

        // NHWC shape of a graph input or output, by binding.
        const std::vector<int64_t>* GetTensorShape(uint32_t binding) const;

        const Selection& GetSelection() const { return m_selection; }
        uint32_t         GetDispatchCount() const;

        // Profiling: when set, Record writes a timestamp before the first and after every dispatch into
        // queryPool[firstQuery .. firstQuery + GetDispatchCount()] (the caller resets the queries).
        void        SetProfiling(VkQueryPool queryPool, uint32_t firstQuery);
        std::string DescribeLayer(uint32_t index) const;  // e.g. "conv3x3 s2 12->16 272x480->136x240"

        struct Impl;

    private:
        Graph() = default;
        Selection             m_selection;
        std::unique_ptr<Impl> m_impl;
    };

    // CPU reference execution of the same lowered graph (bit exact with TOSA int8 semantics). inputs/outputs are keyed
    // by binding and hold packed NHWC int8 data. Used by the conformance test.
    bool RunReference(const GraphDesc& desc, const std::map<uint32_t, std::vector<int8_t>>& inputs, std::map<uint32_t, std::vector<int8_t>>& outputs, std::string* error);
}  // namespace ffx_ml
