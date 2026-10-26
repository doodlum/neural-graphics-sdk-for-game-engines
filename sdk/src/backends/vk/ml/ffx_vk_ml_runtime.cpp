// SPDX-FileCopyrightText: Copyright 2026 Neural Graphics SDK universal contributors
// SPDX-License-Identifier: MIT

#include "ffx_vk_ml_runtime.h"
#include "ffx_vk_ml_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Kernels, compiled from shaders/ffx_ml_conv2d.comp at build time.
#include "ffx_ml_conv2d_scalar_spv.h"
#include "ffx_ml_conv2d_dot4_spv.h"
#include "ffx_ml_conv2d_dot4_tiled_spv.h"
#include "ffx_ml_conv2d_coopmat_int8_spv.h"
#include "ffx_ml_conv2d_coopmat_fp16_spv.h"
#include "ffx_ml_conv2d_coopmat_bf16_spv.h"

namespace ffx_ml
{
    namespace
    {
        struct Functions
        {
            PFN_vkGetPhysicalDeviceProperties2                      GetPhysicalDeviceProperties2;
            PFN_vkGetPhysicalDeviceMemoryProperties                 GetPhysicalDeviceMemoryProperties;
            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR   GetPhysicalDeviceCooperativeMatrixPropertiesKHR;
            PFN_vkCreateBuffer                                      CreateBuffer;
            PFN_vkDestroyBuffer                                     DestroyBuffer;
            PFN_vkGetBufferMemoryRequirements                       GetBufferMemoryRequirements;
            PFN_vkAllocateMemory                                    AllocateMemory;
            PFN_vkFreeMemory                                        FreeMemory;
            PFN_vkBindBufferMemory                                  BindBufferMemory;
            PFN_vkMapMemory                                         MapMemory;
            PFN_vkUnmapMemory                                       UnmapMemory;
            PFN_vkCreateShaderModule                                CreateShaderModule;
            PFN_vkDestroyShaderModule                               DestroyShaderModule;
            PFN_vkCreateDescriptorSetLayout                         CreateDescriptorSetLayout;
            PFN_vkDestroyDescriptorSetLayout                        DestroyDescriptorSetLayout;
            PFN_vkCreatePipelineLayout                              CreatePipelineLayout;
            PFN_vkDestroyPipelineLayout                             DestroyPipelineLayout;
            PFN_vkCreateComputePipelines                            CreateComputePipelines;
            PFN_vkDestroyPipeline                                   DestroyPipeline;
            PFN_vkCreateDescriptorPool                              CreateDescriptorPool;
            PFN_vkDestroyDescriptorPool                             DestroyDescriptorPool;
            PFN_vkAllocateDescriptorSets                            AllocateDescriptorSets;
            PFN_vkUpdateDescriptorSets                              UpdateDescriptorSets;
            PFN_vkCmdBindPipeline                                   CmdBindPipeline;
            PFN_vkCmdBindDescriptorSets                             CmdBindDescriptorSets;
            PFN_vkCmdDispatch                                       CmdDispatch;
            PFN_vkCmdPipelineBarrier                                CmdPipelineBarrier;
            PFN_vkCmdCopyBuffer                                     CmdCopyBuffer;
            PFN_vkCmdWriteTimestamp                                 CmdWriteTimestamp;

            bool Load(const DeviceContext& ctx)
            {
                bool ok = true;
#define FFX_ML_INSTANCE_FN(name) ok &= (name = reinterpret_cast<PFN_vk##name>(ctx.getInstanceProcAddr(ctx.instance, "vk" #name))) != nullptr
#define FFX_ML_DEVICE_FN(name) ok &= (name = reinterpret_cast<PFN_vk##name>(ctx.getDeviceProcAddr(ctx.device, "vk" #name))) != nullptr
                FFX_ML_INSTANCE_FN(GetPhysicalDeviceProperties2);
                FFX_ML_INSTANCE_FN(GetPhysicalDeviceMemoryProperties);
                GetPhysicalDeviceCooperativeMatrixPropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                    ctx.getInstanceProcAddr(ctx.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
                if (ctx.device == VK_NULL_HANDLE)
                    return ok;
                FFX_ML_DEVICE_FN(CreateBuffer);
                FFX_ML_DEVICE_FN(DestroyBuffer);
                FFX_ML_DEVICE_FN(GetBufferMemoryRequirements);
                FFX_ML_DEVICE_FN(AllocateMemory);
                FFX_ML_DEVICE_FN(FreeMemory);
                FFX_ML_DEVICE_FN(BindBufferMemory);
                FFX_ML_DEVICE_FN(MapMemory);
                FFX_ML_DEVICE_FN(UnmapMemory);
                FFX_ML_DEVICE_FN(CreateShaderModule);
                FFX_ML_DEVICE_FN(DestroyShaderModule);
                FFX_ML_DEVICE_FN(CreateDescriptorSetLayout);
                FFX_ML_DEVICE_FN(DestroyDescriptorSetLayout);
                FFX_ML_DEVICE_FN(CreatePipelineLayout);
                FFX_ML_DEVICE_FN(DestroyPipelineLayout);
                FFX_ML_DEVICE_FN(CreateComputePipelines);
                FFX_ML_DEVICE_FN(DestroyPipeline);
                FFX_ML_DEVICE_FN(CreateDescriptorPool);
                FFX_ML_DEVICE_FN(DestroyDescriptorPool);
                FFX_ML_DEVICE_FN(AllocateDescriptorSets);
                FFX_ML_DEVICE_FN(UpdateDescriptorSets);
                FFX_ML_DEVICE_FN(CmdBindPipeline);
                FFX_ML_DEVICE_FN(CmdBindDescriptorSets);
                FFX_ML_DEVICE_FN(CmdDispatch);
                FFX_ML_DEVICE_FN(CmdPipelineBarrier);
                FFX_ML_DEVICE_FN(CmdCopyBuffer);
                FFX_ML_DEVICE_FN(CmdWriteTimestamp);
#undef FFX_ML_INSTANCE_FN
#undef FFX_ML_DEVICE_FN
                return ok;
            }
        };

        Vendor VendorFromId(uint32_t id)
        {
            switch (id)
            {
            case 0x10DE:
                return Vendor::Nvidia;
            case 0x1002:
            case 0x1022:
                return Vendor::Amd;
            case 0x8086:
                return Vendor::Intel;
            case 0x13B5:
                return Vendor::Arm;
            case 0x5143:
                return Vendor::Qualcomm;
            default:
                return Vendor::Other;
            }
        }

        // Proton reports AMD GPUs as NVIDIA (vendor 0x10DE) so games enable DLSS. The tuning must follow the real
        // hardware, so the driver ID and the device name take precedence over the vendor ID.
        Vendor VendorFromDevice(uint32_t vendorId, VkDriverId driverId, const char* deviceName)
        {
            switch (driverId)
            {
            case VK_DRIVER_ID_MESA_RADV:
            case VK_DRIVER_ID_AMD_PROPRIETARY:
            case VK_DRIVER_ID_AMD_OPEN_SOURCE:
                return Vendor::Amd;
            case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
            case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
                return Vendor::Intel;
            default:
                break;
            }
            if (deviceName && (strstr(deviceName, "RADV") || strstr(deviceName, "AMD") || strstr(deviceName, "Radeon")))
                return Vendor::Amd;
            return VendorFromId(vendorId);
        }

        // Per-vendor kernel tuning. The tile shape itself always comes from what the driver reports.
        struct VendorTuning
        {
            uint32_t coopSubgroups;       // subgroups per workgroup for the coop-matrix kernel
            uint32_t preferredSubgroup;   // subgroup size to request when subgroup size control is enabled (0 = default)
            uint32_t preferredTileM;      // preferred M for the coop-matrix tile (0 = largest)
            uint32_t vectorWorkgroup;     // workgroup size for dot4 / scalar kernels
            MatType  matOrder[3];         // coop-matrix element types to try, fastest first (all exact)
            uint32_t splitTargetGroups;   // coop layers with fewer workgroups split their output channels (grid z)
            // Register-tiled dot4 kernel (ML_DOT4_TILED). 0 keeps the one-thread-per-(pixel, 4 channels) kernel.
            uint32_t dot4Tiled;           // 1: use the tiled kernel
            uint32_t dot4Subgroup;        // subgroup size to require for it (0 = driver default)
            uint32_t dot4Acc;             // accumulators per thread: pixels per thread = dot4Acc / OCB (1..8)
            uint32_t dot4Ocb;             // max output channels per thread
            uint32_t dot4WgX, dot4WgY;    // workgroup shape in threads
            uint32_t dot4TargetThreads;   // layers with fewer threads get narrower tiles (fewer pixels / channels per thread)
            uint32_t dot4KsplitMax;       // max lanes splitting one tile's input words (1 = off)
        };

        VendorTuning TuningFor(Vendor vendor)
        {
            switch (vendor)
            {
            case Vendor::Nvidia:
                // warp32 tensor cores, 16x16xK tiles (mma.sync m16n8k32 pairs). On the RTX 4080 fp16 beat int8 by ~5-8% and
                // a split target of 128 workgroups (~2 per SM) was best; see docs/nss_gpu_tuning.md.
                return {4, 32, 16, 128, {MatType::Fp16, MatType::Bf16, MatType::Int8}, 128};
            case Vendor::Amd:
                // RDNA3+ WMMA runs in wave32, 16x16x16 tiles (untuned). RDNA2 (no WMMA) takes the tiled dot4 kernel, tuned on the
                // Steam Deck (Van Gogh); see docs/nss_gpu_tuning.md. Channel blocks up to 40 keep the 36-channel KPN layer in one block.
                return {4, 32, 16, 64, {MatType::Int8, MatType::Fp16, MatType::Bf16}, 128, 1, 64, 32, 40, 8, 8, 4096, 4};
            case Vendor::Intel:
                return {8, 16, 8, 64, {MatType::Int8, MatType::Fp16, MatType::Bf16}, 128};    // Xe XMX systolic arrays, SIMD16 with 8xNxK tiles (untuned)
            default:
                return {4, 0, 16, 64, {MatType::Int8, MatType::Fp16, MatType::Bf16}, 128};
            }
        }

        VkComponentTypeKHR MatComponentType(MatType type)
        {
            switch (type)
            {
            case MatType::Int8:
                return VK_COMPONENT_TYPE_SINT8_KHR;
            case MatType::Fp16:
                return VK_COMPONENT_TYPE_FLOAT16_KHR;
            case MatType::Bf16:
                return VK_COMPONENT_TYPE_BFLOAT16_KHR;
            default:
                return VK_COMPONENT_TYPE_MAX_ENUM_KHR;
            }
        }

        bool MatTypeEnabled(MatType type, uint32_t features)
        {
            switch (type)
            {
            case MatType::Int8:
                return (features & FEATURE_SHADER_INT8) && (features & FEATURE_STORAGE_BUFFER_8BIT);
            case MatType::Fp16:
                return (features & FEATURE_SHADER_FLOAT16) && (features & FEATURE_STORAGE_BUFFER_16BIT);
            case MatType::Bf16:
                return (features & FEATURE_SHADER_BFLOAT16) && (features & FEATURE_STORAGE_BUFFER_16BIT);
            default:
                return false;
            }
        }

        uint32_t MatElementSize(MatType type)
        {
            return type == MatType::Int8 ? 1u : 2u;
        }

        uint32_t RoundUp(uint32_t v, uint32_t m)
        {
            return (v + m - 1) / m * m;
        }

        uint16_t FloatToHalf(float f)
        {
            // Inputs are small integers, always exactly representable.
            uint32_t x;
            memcpy(&x, &f, 4);
            const uint32_t sign = (x >> 16) & 0x8000u;
            int32_t        exp  = int32_t((x >> 23) & 0xFF) - 127 + 15;
            uint32_t       mant = x & 0x7FFFFFu;
            if (f == 0.0f)
                return uint16_t(sign);
            if (exp <= 0)
                return uint16_t(sign);
            return uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
        }

        uint16_t FloatToBf16(float f)
        {
            uint32_t x;
            memcpy(&x, &f, 4);
            x += 0x7FFFu + ((x >> 16) & 1u);  // round to nearest even
            return uint16_t(x >> 16);
        }

    }  // namespace

    const char* ToString(Path path)
    {
        switch (path)
        {
        case Path::CoopMat:
            return "coopmat";
        case Path::Dot4:
            return "dot4";
        case Path::Scalar:
            return "scalar";
        default:
            return "auto";
        }
    }

    const char* ToString(MatType type)
    {
        switch (type)
        {
        case MatType::Int8:
            return "int8";
        case MatType::Fp16:
            return "fp16";
        case MatType::Bf16:
            return "bf16";
        default:
            return "none";
        }
    }

    const char* ToString(Vendor vendor)
    {
        switch (vendor)
        {
        case Vendor::Nvidia:
            return "NVIDIA";
        case Vendor::Amd:
            return "AMD";
        case Vendor::Intel:
            return "Intel";
        case Vendor::Arm:
            return "Arm";
        case Vendor::Qualcomm:
            return "Qualcomm";
        case Vendor::Other:
            return "other";
        default:
            return "unknown";
        }
    }

    std::string Selection::Describe() const
    {
        std::string s = std::string(ToString(vendor)) + " " + deviceName + ": " + ToString(path);
        if (path == Path::CoopMat)
        {
            s += " " + std::string(ToString(matType)) + " " + std::to_string(tileM) + "x" + std::to_string(tileN) + "x" + std::to_string(tileK) +
                 ", " + std::to_string(numSubgroups) + " subgroups of " + std::to_string(subgroupSize);
        }
        else if (dot4Tiled)
        {
            s += " tiled, subgroup " + std::to_string(subgroupSize);
        }
        else
        {
            s += ", workgroup " + std::to_string(workgroupSize);
        }
        return s;
    }

    uint32_t QueryDeviceFeatures(VkInstance instance, VkPhysicalDevice physicalDevice, PFN_vkGetInstanceProcAddr getInstanceProcAddr, std::vector<const char*>* extensions)
    {
        auto getFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(getInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
        auto enumerateExtensions =
            reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(getInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties"));
        if (!getFeatures2 || !enumerateExtensions)
            return 0;
    
        uint32_t available = 0;
        enumerateExtensions(physicalDevice, nullptr, &available, nullptr);
        std::vector<VkExtensionProperties> properties(available);
        enumerateExtensions(physicalDevice, nullptr, &available, properties.data());
        auto hasExtension = [&](const char* name) {
            for (const VkExtensionProperties& p : properties)
                if (strcmp(p.extensionName, name) == 0)
                    return true;
            return false;
        };
    
        VkPhysicalDeviceShaderBfloat16FeaturesKHR     bfloat16    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR  coopMatrix  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceVulkan13Features              vulkan13    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features              vulkan12    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features              vulkan11    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2                     features2   = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &vulkan11;
        vulkan11.pNext  = &vulkan12;
        vulkan12.pNext  = &vulkan13;
        void** tail     = &vulkan13.pNext;
        const bool coopExt     = hasExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        const bool bfloat16Ext = hasExtension(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
        if (coopExt)
        {
            *tail = &coopMatrix;
            tail  = &coopMatrix.pNext;
        }
        if (bfloat16Ext)
        {
            *tail = &bfloat16;
            tail  = &bfloat16.pNext;
        }
        getFeatures2(physicalDevice, &features2);
    
        uint32_t                 flags = 0;
        std::vector<const char*> required;
        if (vulkan12.shaderInt8)
            flags |= FEATURE_SHADER_INT8;
        if (vulkan12.storageBuffer8BitAccess)
            flags |= FEATURE_STORAGE_BUFFER_8BIT;
        if (vulkan12.shaderFloat16)
            flags |= FEATURE_SHADER_FLOAT16;
        if (vulkan11.storageBuffer16BitAccess)
            flags |= FEATURE_STORAGE_BUFFER_16BIT;
        if (vulkan13.shaderIntegerDotProduct)
            flags |= FEATURE_INTEGER_DOT_PRODUCT;
        if (vulkan13.subgroupSizeControl && vulkan13.computeFullSubgroups)
            flags |= FEATURE_SUBGROUP_SIZE_CONTROL;
        if (vulkan12.vulkanMemoryModel)
            flags |= FEATURE_VULKAN_MEMORY_MODEL;
        if (coopExt && coopMatrix.cooperativeMatrix)
        {
            flags |= FEATURE_COOPERATIVE_MATRIX;
            required.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }
        if (bfloat16Ext && bfloat16.shaderBFloat16Type && bfloat16.shaderBFloat16CooperativeMatrix)
        {
            flags |= FEATURE_SHADER_BFLOAT16;
            required.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
        }
    
        if (extensions)
            *extensions = required;
        return flags;
    }

    void ReadEnvironmentOverrides(Path& path, MatType& type)
    {
        if (const char* p = getenv("FFX_ML_PATH"))
        {
            if (!strcmp(p, "coopmat"))
                path = Path::CoopMat;
            else if (!strcmp(p, "dot4"))
                path = Path::Dot4;
            else if (!strcmp(p, "scalar"))
                path = Path::Scalar;
        }
        if (const char* t = getenv("FFX_ML_MATTYPE"))
        {
            if (!strcmp(t, "int8"))
                type = MatType::Int8;
            else if (!strcmp(t, "fp16"))
                type = MatType::Fp16;
            else if (!strcmp(t, "bf16"))
                type = MatType::Bf16;
        }
    }

    Selection SelectPath(const DeviceContext& context, Path forcedPath, MatType forcedType)
    {
        Functions fn = {};
        DeviceContext instanceOnly = context;
        instanceOnly.device        = VK_NULL_HANDLE;
        fn.Load(instanceOnly);

        Selection sel;

        VkPhysicalDeviceDriverProperties              driver      = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceSubgroupSizeControlProperties sizeControl = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, &driver};
        VkPhysicalDeviceSubgroupProperties            subgroup    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &sizeControl};
        VkPhysicalDeviceProperties2                   props       = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
        fn.GetPhysicalDeviceProperties2(context.physicalDevice, &props);

        sel.vendorId   = props.properties.vendorID;
        sel.vendor     = VendorFromDevice(sel.vendorId, driver.driverID, props.properties.deviceName);
        sel.deviceName = props.properties.deviceName;
        const VendorTuning tuning = TuningFor(sel.vendor);
        const uint32_t     features = context.enabledFeatures;

        auto allowed = [&](Path p) { return forcedPath == Path::Auto || forcedPath == p; };

        // Is the coop-matrix kernel buildable on this device? Needs the extension plus at least one matrix type.
        if ((allowed(Path::CoopMat) || forcedPath == Path::Auto) && (features & FEATURE_COOPERATIVE_MATRIX) &&
            (features & FEATURE_VULKAN_MEMORY_MODEL) && fn.GetPhysicalDeviceCooperativeMatrixPropertiesKHR && forcedPath != Path::Dot4 &&
            forcedPath != Path::Scalar)
        {
            uint32_t count = 0;
            fn.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(context.physicalDevice, &count, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> shapes(count, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
            fn.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(context.physicalDevice, &count, shapes.data());

            // Exact types only: int8 natively; fp16/bf16 hold every int8 value and fp32 accumulation stays exact. The order
            // is the vendor's measured preference; FFX_ML_MATTYPE forces one.
            for (MatType type : tuning.matOrder)
            {
                if (forcedType != MatType::None && forcedType != type)
                    continue;
                if (!MatTypeEnabled(type, features))
                    continue;
                const VkComponentTypeKHR ab  = MatComponentType(type);
                const VkComponentTypeKHR acc = (type == MatType::Int8) ? VK_COMPONENT_TYPE_SINT32_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR;

                const VkCooperativeMatrixPropertiesKHR* best      = nullptr;
                int64_t                                 bestScore = -1;
                for (const auto& s : shapes)
                {
                    if (s.AType != ab || s.BType != ab || s.CType != acc || s.ResultType != acc || s.scope != VK_SCOPE_SUBGROUP_KHR || s.saturatingAccumulation)
                        continue;
                    if (s.KSize % 4 != 0 || s.NSize > 16 || s.MSize > 16)
                        continue;
                    // Prefer N = 16 (fewer passes over the small OC counts), the vendor's preferred M, then the deepest K.
                    int64_t score = int64_t(s.NSize) * 1000000 + (s.MSize == tuning.preferredTileM ? 100000 : int64_t(s.MSize) * 1000) + s.KSize;
                    if (score > bestScore)
                    {
                        bestScore = score;
                        best      = &s;
                    }
                }
                if (!best)
                    continue;

                sel.path         = Path::CoopMat;
                sel.matType      = type;
                sel.tileM        = best->MSize;
                sel.tileN        = best->NSize;
                sel.tileK        = best->KSize;
                sel.numSubgroups = tuning.coopSubgroups;
                sel.subgroupSize = subgroup.subgroupSize;
                if ((features & FEATURE_SUBGROUP_SIZE_CONTROL) && tuning.preferredSubgroup != 0 && tuning.preferredSubgroup >= sizeControl.minSubgroupSize &&
                    tuning.preferredSubgroup <= sizeControl.maxSubgroupSize && (sizeControl.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
                {
                    sel.subgroupSize        = tuning.preferredSubgroup;
                    sel.requireSubgroupSize = true;
                }
                sel.workgroupSize = sel.numSubgroups * sel.subgroupSize;
                return sel;
            }
        }

        if (allowed(Path::Dot4) && (features & FEATURE_INTEGER_DOT_PRODUCT))
        {
            sel.path          = Path::Dot4;
            sel.workgroupSize = tuning.vectorWorkgroup;
            sel.dot4Tiled     = tuning.dot4Tiled != 0;
            if (const char* e = getenv("FFX_ML_DOT4_TILED"))
                sel.dot4Tiled = atoi(e) != 0;
            uint32_t sgSize = tuning.dot4Subgroup;
            if (const char* e = getenv("FFX_ML_DOT4_SUBGROUP"))
                sgSize = uint32_t(atoi(e));
            sel.subgroupSize = subgroup.subgroupSize;
            if (sel.dot4Tiled && (features & FEATURE_SUBGROUP_SIZE_CONTROL) && sgSize != 0 && sgSize >= sizeControl.minSubgroupSize &&
                sgSize <= sizeControl.maxSubgroupSize && (sizeControl.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
            {
                sel.subgroupSize        = sgSize;
                sel.requireSubgroupSize = true;
            }
            return sel;
        }

        sel.path          = Path::Scalar;
        sel.workgroupSize = tuning.vectorWorkgroup;
        return sel;
    }

    // -------------------------------------------------------------------------------------------------------------
    // Graph
    // -------------------------------------------------------------------------------------------------------------
    struct Graph::Impl
    {
        static constexpr uint32_t kSetGroups = 4;  // descriptor set groups kept for distinct external bindings

        struct Allocation
        {
            VkBuffer       buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize   size   = 0;
        };

        struct Layer
        {
            VkPipeline   pipeline = VK_NULL_HANDLE;
            VkDeviceSize weightsOffset = 0, weightsSize = 0;
            VkDeviceSize paramsOffset = 0, paramsSize = 0;
            uint32_t     groups[3] = {1, 1, 1};
            uint32_t     src[2] = {0, 0};  // StoredTensor indices
            uint32_t     numSources = 1;
            uint32_t     dst = 0;
        };

        struct SetGroup
        {
            std::vector<ExternalTensor>  externals;
            std::vector<VkDescriptorSet> sets;
            uint64_t                     lastUse = 0;
        };

        DeviceContext   ctx;
        Functions       fn;
        Selection       selection;
        tosa::Graph     graph;
        LoweredGraph    lowered;

        VkDescriptorSetLayout setLayout      = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorPool      pool           = VK_NULL_HANDLE;
        VkShaderModule        module         = VK_NULL_HANDLE;
        std::vector<Layer>    layers;
        std::vector<Allocation> internals;
        Allocation            constantsDevice;
        Allocation            constantsStaging;
        bool                  constantsUploaded = false;
        std::vector<SetGroup> setGroups;
        uint64_t              frame = 0;
        VkQueryPool           profilePool  = VK_NULL_HANDLE;
        uint32_t              profileFirst = 0;
        std::map<uint32_t, std::vector<int64_t>> shapes;  // external bindings

        VkPhysicalDeviceMemoryProperties memoryProperties = {};

        ~Impl()
        {
            if (ctx.device == VK_NULL_HANDLE)
                return;
            for (Layer& l : layers)
                if (l.pipeline)
                    fn.DestroyPipeline(ctx.device, l.pipeline, nullptr);
            for (Allocation& a : internals)
                Free(a);
            Free(constantsDevice);
            Free(constantsStaging);
            if (module)
                fn.DestroyShaderModule(ctx.device, module, nullptr);
            if (pool)
                fn.DestroyDescriptorPool(ctx.device, pool, nullptr);
            if (pipelineLayout)
                fn.DestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
            if (setLayout)
                fn.DestroyDescriptorSetLayout(ctx.device, setLayout, nullptr);
        }

        void Free(Allocation& a)
        {
            if (a.buffer)
                fn.DestroyBuffer(ctx.device, a.buffer, nullptr);
            if (a.memory)
                fn.FreeMemory(ctx.device, a.memory, nullptr);
            a = {};
        }

        bool Allocate(Allocation& a, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required)
        {
            VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size               = std::max<VkDeviceSize>(size, 16);
            info.usage              = usage;
            info.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
            if (fn.CreateBuffer(ctx.device, &info, nullptr, &a.buffer) != VK_SUCCESS)
                return false;
            VkMemoryRequirements req;
            fn.GetBufferMemoryRequirements(ctx.device, a.buffer, &req);
            uint32_t typeIndex = UINT32_MAX;
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
            {
                if ((req.memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & required) == required)
                {
                    typeIndex = i;
                    break;
                }
            }
            if (typeIndex == UINT32_MAX)
                return false;
            VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            alloc.allocationSize       = req.size;
            alloc.memoryTypeIndex      = typeIndex;
            if (fn.AllocateMemory(ctx.device, &alloc, nullptr, &a.memory) != VK_SUCCESS)
                return false;
            a.size = info.size;
            return fn.BindBufferMemory(ctx.device, a.buffer, a.memory, 0) == VK_SUCCESS;
        }

        struct Dot4Tile
        {
            uint32_t px, ocb, blocks, wgX, wgY, ksplit;
        };

        Dot4Tile ChooseDot4Tile(const FusedConv& fc, const tosa::Node& conv) const
        {
            (void)conv;
            const VendorTuning t   = TuningFor(selection.vendor);
            auto               env = [](const char* name, uint32_t def) {
                const char* e = getenv(name);
                return e ? uint32_t(atoi(e)) : def;
            };
            Dot4Tile tile;
            tile.wgX                = std::max(1u, env("FFX_ML_DOT4_WGX", t.dot4WgX ? t.dot4WgX : 8));
            tile.wgY                = std::max(1u, env("FFX_ML_DOT4_WGY", t.dot4WgY ? t.dot4WgY : 8));
            const uint32_t ocbMax   = std::max(4u, env("FFX_ML_DOT4_OCB", t.dot4Ocb ? t.dot4Ocb : 16)) & ~3u;
            const uint32_t target   = env("FFX_ML_DOT4_TARGET_THREADS", t.dot4TargetThreads);
            const uint32_t oc4      = RoundUp(uint32_t(fc.outC), 4);
            tile.blocks             = (oc4 + ocbMax - 1) / ocbMax;
            tile.ocb                = RoundUp((oc4 + tile.blocks - 1) / tile.blocks, 4);
            // Narrow layers (few output channels) take several pixels per thread so each input word still feeds enough dot4s.
            // Capped at 4: the 4-channel output layer (16 -> 4 at full resolution) measured faster at 4 than at 8 pixels.
            uint32_t px = std::min(env("FFX_ML_DOT4_PX_MAX", 4u), std::max(1u, env("FFX_ML_DOT4_ACC", t.dot4Acc ? t.dot4Acc : 64) / tile.ocb));
            if (const char* e = getenv("FFX_ML_DOT4_PX"))
                px = std::max(1u, uint32_t(atoi(e)));
            auto threads            = [&](uint32_t p, uint32_t blocks) { return ((uint32_t(fc.outW) + p - 1) / p) * uint32_t(fc.outH) * blocks; };
            // Small layers: fewer pixels per thread first, then narrower channel blocks, until the target thread count is met.
            // FFX_ML_DOT4_KSPLIT_FIRST=1 leaves the channel blocks alone and lets the KSPLIT below take the shortfall.
            while (target && px > 1 && threads(px, tile.blocks) < target)
                px /= 2;
            while (target && !env("FFX_ML_DOT4_KSPLIT_FIRST", 0) && tile.ocb > 4 && threads(px, tile.blocks) < target)
            {
                const uint32_t ocb = tile.ocb / 2 >= 4 ? RoundUp(tile.ocb / 2, 4) : 4;
                tile.blocks        = (oc4 + ocb - 1) / ocb;
                tile.ocb           = RoundUp((oc4 + tile.blocks - 1) / tile.blocks, 4);
            }
            tile.px = px;
            // Input-word split across KSPLIT subgroups of one workgroup (partial sums reduced exactly in shared memory):
            // shortens each thread's serial chain on layers too small to fill the GPU. Needs one subgroup per z slice
            // (WG_X * WG_Y == required subgroup size), PX * OCB / KSPLIT a multiple of 4 and the partial sums in 32 KB.
            uint32_t       k       = 1;
            const uint32_t kMax    = env("FFX_ML_DOT4_KSPLIT_MAX", t.dot4KsplitMax);
            const uint32_t icWords = (uint32_t(fc.inC) + 3) / 4;
            auto           kOk     = [&](uint32_t kk) {
                return kk <= icWords && (tile.px * tile.ocb) % (4 * kk) == 0 && kk * tile.px * tile.ocb * tile.wgX * tile.wgY * 4 <= 32768 &&
                       selection.requireSubgroupSize && tile.wgX * tile.wgY == selection.subgroupSize;
            };
            while (k < kMax && kOk(k * 2) && threads(px, tile.blocks) * k < target)
                k *= 2;
            if (const char* e = getenv("FFX_ML_DOT4_KSPLIT"))
            {
                k = 1;
                while (k * 2 <= uint32_t(atoi(e)) && kOk(k * 2))
                    k *= 2;
            }
            tile.ksplit = k;
            return tile;
        }

        const uint32_t* KernelCode(size_t& bytes) const
        {
            switch (selection.path)
            {
            case Path::CoopMat:
                switch (selection.matType)
                {
                case MatType::Int8:
                    bytes = sizeof(ffx_ml_conv2d_coopmat_int8);
                    return ffx_ml_conv2d_coopmat_int8;
                case MatType::Fp16:
                    bytes = sizeof(ffx_ml_conv2d_coopmat_fp16);
                    return ffx_ml_conv2d_coopmat_fp16;
                case MatType::Bf16:
                    bytes = sizeof(ffx_ml_conv2d_coopmat_bf16);
                    return ffx_ml_conv2d_coopmat_bf16;
                default:
                    return nullptr;
                }
            case Path::Dot4:
                if (selection.dot4Tiled)
                {
                    bytes = sizeof(ffx_ml_conv2d_dot4_tiled);
                    return ffx_ml_conv2d_dot4_tiled;
                }
                bytes = sizeof(ffx_ml_conv2d_dot4);
                return ffx_ml_conv2d_dot4;
            default:
                bytes = sizeof(ffx_ml_conv2d_scalar);
                return ffx_ml_conv2d_scalar;
            }
        }

        bool Build(std::string* error)
        {
            auto fail = [&](const std::string& m) {
                if (error)
                    *error = m;
                return false;
            };

            fn.GetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memoryProperties);
            VkPhysicalDeviceProperties2 deviceProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            fn.GetPhysicalDeviceProperties2(ctx.physicalDevice, &deviceProperties);
            const uint32_t sharedMemoryLimit = deviceProperties.properties.limits.maxComputeSharedMemorySize;

            // Descriptor layout: src0, src1, weights, params, dst.
            VkDescriptorSetLayoutBinding bindings[5] = {};
            for (uint32_t i = 0; i < 5; ++i)
                bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo setInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setInfo.bindingCount                    = 5;
            setInfo.pBindings                       = bindings;
            if (fn.CreateDescriptorSetLayout(ctx.device, &setInfo, nullptr, &setLayout) != VK_SUCCESS)
                return fail("vkCreateDescriptorSetLayout failed");
            VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount             = 1;
            layoutInfo.pSetLayouts                = &setLayout;
            if (fn.CreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
                return fail("vkCreatePipelineLayout failed");

            size_t          codeBytes = 0;
            const uint32_t* code      = KernelCode(codeBytes);
            VkShaderModuleCreateInfo moduleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            moduleInfo.codeSize                 = codeBytes;
            moduleInfo.pCode                    = code;
            if (fn.CreateShaderModule(ctx.device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
                return fail("vkCreateShaderModule failed");

            // Internal tensors.
            internals.resize(lowered.internalTensorCount);
            for (const StoredTensor& t : lowered.tensors)
            {
                if (t.external)
                {
                    shapes[t.binding] = {1, t.h, t.w, t.c};
                    continue;
                }
                const VkDeviceSize bytes = VkDeviceSize(t.h) * t.w * t.channelStride;
                if (!Allocate(internals[t.internalIndex], bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                    return fail("failed to allocate an intermediate tensor");
            }

            // Pack weights and epilogue parameters for every layer into one constants buffer.
            std::vector<uint8_t> constants;
            auto                 align = [&](size_t a) { constants.resize(RoundUp(uint32_t(constants.size()), uint32_t(a)), 0); };

            const bool coop = selection.path == Path::CoopMat;
            const bool rawInput = !coop || selection.matType == MatType::Int8;

            layers.resize(lowered.convs.size());
            for (size_t li = 0; li < lowered.convs.size(); ++li)
            {
                const FusedConv&  fc   = lowered.convs[li];
                const tosa::Node& conv = *fc.conv;
                const tosa::Node& rs   = *fc.rescale;
                Layer&            layer = layers[li];

                const uint32_t IC   = uint32_t(fc.inC);
                const uint32_t OC   = uint32_t(fc.outC);
                const uint32_t taps = uint32_t(conv.kernelH * conv.kernelW);
                const uint32_t icPad = coop ? RoundUp(IC, selection.tileK) : RoundUp(IC, 4);
                const uint32_t ocPad = coop ? RoundUp(OC, selection.tileN) : OC;

                auto weightAt = [&](uint32_t oc, uint32_t tap, uint32_t ic) -> int32_t {
                    return conv.weights[(size_t(oc) * taps + tap) * IC + ic];
                };

                // Tiled dot4: pixels and output channels per thread, output-channel blocks (grid z), workgroup shape.
                const bool tiled = selection.path == Path::Dot4 && selection.dot4Tiled;
                Dot4Tile   tile  = {};
                if (tiled)
                    tile = ChooseDot4Tile(fc, conv);

                // Weights
                align(64);
                layer.weightsOffset = constants.size();
                if (coop)
                {
                    const uint32_t elem = MatElementSize(selection.matType);
                    const size_t   base = constants.size();
                    constants.resize(base + size_t(taps) * icPad * ocPad * elem, 0);
                    for (uint32_t tap = 0; tap < taps; ++tap)
                        for (uint32_t ic = 0; ic < IC; ++ic)
                            for (uint32_t oc = 0; oc < OC; ++oc)
                            {
                                const int32_t w   = weightAt(oc, tap, ic);
                                const size_t  idx = (size_t(tap) * icPad + ic) * ocPad + oc;
                                uint8_t*      dst = &constants[base + idx * elem];
                                switch (selection.matType)
                                {
                                case MatType::Int8:
                                    dst[0] = uint8_t(int8_t(w));
                                    break;
                                case MatType::Fp16:
                                {
                                    const uint16_t h = FloatToHalf(float(w));
                                    memcpy(dst, &h, 2);
                                    break;
                                }
                                case MatType::Bf16:
                                {
                                    const uint16_t h = FloatToBf16(float(w));
                                    memcpy(dst, &h, 2);
                                    break;
                                }
                                default:
                                    break;
                                }
                            }
                }
                else if (tiled)
                {
                    // [ocBlock][tap][icWord][OCB] words, zero padded past OC.
                    const uint32_t icWords = icPad / 4;
                    const size_t   base    = constants.size();
                    constants.resize(base + size_t(tile.blocks) * tile.ocb * taps * icWords * 4, 0);
                    for (uint32_t oc = 0; oc < OC; ++oc)
                    {
                        const uint32_t blk = oc / tile.ocb, o = oc % tile.ocb;
                        for (uint32_t tap = 0; tap < taps; ++tap)
                            for (uint32_t ic = 0; ic < IC; ++ic)
                            {
                                const size_t word = ((size_t(blk) * taps + tap) * icWords + ic / 4) * tile.ocb + o;
                                constants[base + word * 4 + (ic & 3)] = uint8_t(int8_t(weightAt(oc, tap, ic)));
                            }
                    }
                }
                else
                {
                    const uint32_t icWords = icPad / 4;
                    const size_t   base    = constants.size();
                    constants.resize(base + size_t(OC) * taps * icWords * 4, 0);
                    for (uint32_t oc = 0; oc < OC; ++oc)
                        for (uint32_t tap = 0; tap < taps; ++tap)
                            for (uint32_t ic = 0; ic < IC; ++ic)
                                constants[base + ((size_t(oc) * taps + tap) * icWords) * 4 + ic] = uint8_t(int8_t(weightAt(oc, tap, ic)));
                }
                layer.weightsSize = constants.size() - layer.weightsOffset;

                // Params: bias (with the input zero point folded in for raw int8 paths), multiplier, shift, table.
                align(64);
                layer.paramsOffset     = constants.size();
                const uint32_t stride  = RoundUp(OC, 4);
                std::vector<int32_t> p(size_t(stride) * 3 + 64, 0);
                for (uint32_t oc = 0; oc < OC; ++oc)
                {
                    int64_t b = conv.bias[oc];
                    if (rawInput)
                    {
                        int64_t sum = 0;
                        for (uint32_t tap = 0; tap < taps; ++tap)
                            for (uint32_t ic = 0; ic < IC; ++ic)
                                sum += weightAt(oc, tap, ic);
                        b -= int64_t(conv.inputZp) * sum;
                    }
                    p[oc] = int32_t(b);
                }
                const size_t mulCount = rs.perChannel ? OC : 1;
                int32_t      minShift = 64, maxShift = 0;
                for (size_t i = 0; i < mulCount; ++i)
                {
                    p[stride + i]     = rs.multiplier[i];
                    p[2 * stride + i] = rs.shift[i];
                    minShift          = std::min(minShift, rs.shift[i]);
                    maxShift          = std::max(maxShift, rs.shift[i]);
                }
                if (getenv("FFX_ML_DEBUG_LAYERS"))
                    printf("[ffx_ml] layer %zu: IC %u OC %u %dx%d->%dx%d, shift %d..%d%s%s, scale32 %d, tile px %u ocb %u blocks %u ksplit %u\n", li, IC, OC, fc.inH,
                           fc.inW, fc.outH, fc.outW, minShift, maxShift, rs.doubleRound ? " double-round" : "", fc.table ? " table" : "", rs.scale32 ? 1 : 0,
                           tile.px, tile.ocb, tile.blocks, tile.ksplit);
                if (fc.table)
                {
                    for (uint32_t i = 0; i < 256; ++i)
                        reinterpret_cast<uint8_t*>(&p[3 * stride])[i] = uint8_t(fc.table->table[i]);
                }
                const size_t pBytes = p.size() * 4;
                constants.resize(constants.size() + pBytes);
                memcpy(&constants[layer.paramsOffset], p.data(), pBytes);
                layer.paramsSize = pBytes;

                // Sources / destination.
                layer.numSources = uint32_t(fc.sources.size());
                layer.src[0]     = fc.sources[0].tensor;
                layer.src[1]     = fc.sources.size() > 1 ? fc.sources[1].tensor : fc.sources[0].tensor;
                layer.dst        = fc.output;

                // Specialization constants.
                std::vector<VkSpecializationMapEntry> entries;
                std::vector<int32_t>                  values;
                auto spec = [&](uint32_t id, int32_t value) {
                    entries.push_back({id, uint32_t(values.size() * 4), 4});
                    values.push_back(value);
                };
                // Coop-matrix tile height: as many rows per subgroup (up to 4) as the input patch lets fit in shared memory.
                // Coop-matrix tile: one row per subgroup (keeps occupancy high on the small NSS layers).
                uint32_t tileRows      = getenv("FFX_ML_TILE_ROWS") ? uint32_t(atoi(getenv("FFX_ML_TILE_ROWS"))) : 1;
                uint32_t weightsShared  = 0;
                uint32_t ntPerWorkgroup = ocPad / std::max(selection.tileN, 1u);
                if (coop)
                {
                    // Output-channel split: each workgroup computes a subset of the N tiles (grid z), which shortens the
                    // per-subgroup MMA chain and multiplies the workgroup count. Small, low-resolution layers are split until
                    // they have at least VendorTuning::splitTargetGroups workgroups; large layers are left whole (splitting them re-stages
                    // the same input patch per split and measured slower). FFX_ML_N_SPLIT forces a split for experiments.
                    const uint32_t numNt          = ntPerWorkgroup;
                    const uint32_t tileH          = selection.numSubgroups * tileRows;
                    const uint32_t spatialGroups  = ((uint32_t(fc.outW) + selection.tileM - 1) / selection.tileM) * ((uint32_t(fc.outH) + tileH - 1) / tileH);
                    const uint32_t splitTarget    = getenv("FFX_ML_N_SPLIT_TARGET") ? uint32_t(atoi(getenv("FFX_ML_N_SPLIT_TARGET"))) : TuningFor(selection.vendor).splitTargetGroups;
                    uint32_t       split          = 1;
                    while (split < numNt && spatialGroups * split < splitTarget)
                        ++split;
                    if (getenv("FFX_ML_N_SPLIT"))
                        split = uint32_t(atoi(getenv("FFX_ML_N_SPLIT")));
                    split = std::max(1u, std::min(split, numNt));
                    while (numNt % split)
                        ++split;
                    ntPerWorkgroup = numNt / split;
                    const uint32_t elem        = MatElementSize(selection.matType);
                    const uint32_t weightBytes = taps * icPad * ocPad * elem;
                    auto sharedBytes           = [&](uint32_t rows, bool withWeights) {
                        const uint32_t tileH  = selection.numSubgroups * rows;
                        const uint32_t patchW = (selection.tileM - 1) * conv.stride[1] + (conv.kernelW - 1) * conv.dilation[1] + 1;
                        const uint32_t patchH = (tileH - 1) * conv.stride[0] + (conv.kernelH - 1) * conv.dilation[0] + 1;
                        return patchW * patchH * icPad * elem + selection.numSubgroups * selection.tileM * ocPad * 4 + (withWeights ? weightBytes : 16);
                    };
                    while (tileRows > 1 && sharedBytes(tileRows, false) > sharedMemoryLimit)
                        --tileRows;
                    if (sharedBytes(tileRows, false) > sharedMemoryLimit)
                        return fail("layer " + std::to_string(li) + " does not fit in shared memory");
                    // Staging the weights per workgroup measured slower than reading them through the cache on the RTX 4080,
                    // so it is opt-in (FFX_ML_WEIGHTS_SHARED_LIMIT=<bytes>) for experiments on other GPUs.
                    const uint32_t weightBudget = getenv("FFX_ML_WEIGHTS_SHARED_LIMIT") ? uint32_t(atoi(getenv("FFX_ML_WEIGHTS_SHARED_LIMIT"))) : 0u;
                    weightsShared = sharedBytes(tileRows, true) <= std::min(sharedMemoryLimit, weightBudget) ? 1u : 0u;
                }
                spec(0, int32_t(coop ? selection.numSubgroups * selection.subgroupSize : tiled ? tile.wgX : selection.workgroupSize));
                if (tiled)
                {
                    spec(53, int32_t(tile.wgY));
                    spec(54, int32_t(tile.px));
                    spec(55, int32_t(tile.ocb));
                    spec(56, int32_t(tile.ksplit));
                    // Cheaper exact rescale when no shift needs the low product word (the NSS models: shifts 35..49).
                    const bool hiOnly = rs.scale32 && !rs.doubleRound && minShift >= 33 && maxShift <= 62 && !getenv("FFX_ML_NO_SCALE_HI_ONLY");
                    spec(58, hiOnly ? 1 : 0);
                }
                if (coop)
                {
                    spec(1, int32_t(selection.tileM));
                    spec(2, int32_t(selection.tileN));
                    spec(3, int32_t(selection.tileK));
                    spec(4, int32_t(selection.numSubgroups));
                    spec(5, int32_t(ocPad / selection.tileN));
                    spec(6, int32_t(ocPad));
                    spec(7, int32_t(icPad));
                    spec(8, int32_t(tileRows));
                    spec(9, int32_t(weightsShared));
                    spec(52, int32_t(ntPerWorkgroup));
                }
                spec(10, fc.inH);
                spec(11, fc.inW);
                spec(12, fc.outH);
                spec(13, fc.outW);
                spec(14, fc.outC);
                spec(15, lowered.tensors[fc.output].channelStride);
                spec(16, conv.kernelH);
                spec(17, conv.kernelW);
                spec(18, conv.stride[0]);
                spec(19, conv.stride[1]);
                spec(20, conv.pad[0]);
                spec(21, conv.pad[2]);
                spec(22, conv.dilation[0]);
                spec(23, conv.dilation[1]);
                spec(24, conv.inputZp);
                spec(25, int32_t(icPad / 4));
                spec(26, rs.rescaleInZp);
                spec(27, rs.rescaleOutZp);
                spec(28, (rs.doubleRound ? 1 : 0) | (rs.perChannel ? 2 : 0) | (fc.table ? 4 : 0));
                spec(29, int32_t(fc.sources.size()));
                for (size_t s = 0; s < 2; ++s)
                {
                    const uint32_t base = s == 0 ? 30 : 41;
                    if (s >= fc.sources.size())
                    {
                        spec(base + 2, 0);  // S1_C = 0
                        continue;
                    }
                    const ConvSource&   src = fc.sources[s];
                    const StoredTensor& t   = lowered.tensors[src.tensor];
                    spec(base + 0, t.h);
                    spec(base + 1, t.w);
                    spec(base + 2, t.c);
                    spec(base + 3, t.channelStride);
                    spec(base + 4, src.resize ? 1 : 0);
                    spec(base + 5, src.scale[0]);
                    spec(base + 6, src.scale[1]);
                    spec(base + 7, src.offset[0]);
                    spec(base + 8, src.scale[2]);
                    spec(base + 9, src.scale[3]);
                    spec(base + 10, src.offset[1]);
                }

                VkSpecializationInfo specInfo = {uint32_t(entries.size()), entries.data(), values.size() * 4, values.data()};

                VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSize = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
                requiredSize.requiredSubgroupSize = selection.subgroupSize;

                VkComputePipelineCreateInfo pipeInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                pipeInfo.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pipeInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
                pipeInfo.stage.module                = module;
                pipeInfo.stage.pName                 = "main";
                pipeInfo.stage.pSpecializationInfo   = &specInfo;
                if (coop && selection.requireSubgroupSize)
                {
                    pipeInfo.stage.pNext = &requiredSize;
                    pipeInfo.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
                }
                else if (tiled && selection.requireSubgroupSize)
                {
                    // Full subgroups need the workgroup X size to be a multiple of the subgroup size.
                    pipeInfo.stage.pNext = &requiredSize;
                    if (tile.wgX % selection.subgroupSize == 0)
                        pipeInfo.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
                }
                pipeInfo.layout = pipelineLayout;
                if (fn.CreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &layer.pipeline) != VK_SUCCESS)
                    return fail("vkCreateComputePipelines failed for layer " + std::to_string(li));

                const uint32_t pixels = uint32_t(fc.outH * fc.outW);
                if (coop)
                {
                    const uint32_t tileH = selection.numSubgroups * tileRows;
                    layer.groups[0]      = (uint32_t(fc.outW) + selection.tileM - 1) / selection.tileM;
                    layer.groups[1]      = (uint32_t(fc.outH) + tileH - 1) / tileH;
                    layer.groups[2]      = (ocPad / selection.tileN) / ntPerWorkgroup;
                }
                else if (tiled)
                {
                    const uint32_t threadsX = (uint32_t(fc.outW) + tile.px - 1) / tile.px;
                    layer.groups[0]         = (threadsX + tile.wgX - 1) / tile.wgX;
                    layer.groups[1]         = (uint32_t(fc.outH) + tile.wgY - 1) / tile.wgY;
                    layer.groups[2]         = tile.blocks;
                }
                else
                {
                    layer.groups[0] = (pixels + selection.workgroupSize - 1) / selection.workgroupSize;
                    layer.groups[1] = (OC + 3) / 4;
                }
            }

            align(64);
            if (!Allocate(constantsDevice, constants.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
                !Allocate(constantsStaging, constants.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                return fail("failed to allocate the weight buffers");
            void* mapped = nullptr;
            if (fn.MapMemory(ctx.device, constantsStaging.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
                return fail("vkMapMemory failed");
            memcpy(mapped, constants.data(), constants.size());
            fn.UnmapMemory(ctx.device, constantsStaging.memory);

            VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uint32_t(5 * layers.size() * kSetGroups)};
            VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets                    = uint32_t(layers.size() * kSetGroups);
            poolInfo.poolSizeCount              = 1;
            poolInfo.pPoolSizes                 = &poolSize;
            if (fn.CreateDescriptorPool(ctx.device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
                return fail("vkCreateDescriptorPool failed");
            return true;
        }

        bool FindExternal(const ExternalTensor* tensors, uint32_t count, uint32_t binding, ExternalTensor& out) const
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                if (tensors[i].binding == binding)
                {
                    out = tensors[i];
                    return true;
                }
            }
            return false;
        }

        VkDescriptorBufferInfo TensorInfo(uint32_t tensorIndex, const ExternalTensor* tensors, uint32_t count) const
        {
            const StoredTensor& t = lowered.tensors[tensorIndex];
            if (!t.external)
                return {internals[t.internalIndex].buffer, 0, VK_WHOLE_SIZE};
            ExternalTensor ext = {};
            FindExternal(tensors, count, t.binding, ext);
            return {ext.buffer, ext.offset, VK_WHOLE_SIZE};
        }

        SetGroup* GetSetGroup(const ExternalTensor* tensors, uint32_t count)
        {
            std::vector<ExternalTensor> key(tensors, tensors + count);
            std::sort(key.begin(), key.end(), [](const ExternalTensor& a, const ExternalTensor& b) { return a.binding < b.binding; });
            auto same = [&](const SetGroup& g) {
                if (g.externals.size() != key.size())
                    return false;
                for (size_t i = 0; i < key.size(); ++i)
                    if (g.externals[i].binding != key[i].binding || g.externals[i].buffer != key[i].buffer || g.externals[i].offset != key[i].offset)
                        return false;
                return true;
            };
            for (SetGroup& g : setGroups)
            {
                if (same(g))
                {
                    g.lastUse = frame;
                    return &g;
                }
            }

            SetGroup* group = nullptr;
            if (setGroups.size() < kSetGroups)
            {
                setGroups.emplace_back();
                group = &setGroups.back();
                group->sets.resize(layers.size());
                std::vector<VkDescriptorSetLayout> layouts(layers.size(), setLayout);
                VkDescriptorSetAllocateInfo        alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                alloc.descriptorPool                     = pool;
                alloc.descriptorSetCount                 = uint32_t(layers.size());
                alloc.pSetLayouts                        = layouts.data();
                if (fn.AllocateDescriptorSets(ctx.device, &alloc, group->sets.data()) != VK_SUCCESS)
                    return nullptr;
            }
            else
            {
                // Recycle the least recently used group; it is several frames old by now.
                group = &*std::min_element(setGroups.begin(), setGroups.end(), [](const SetGroup& a, const SetGroup& b) { return a.lastUse < b.lastUse; });
            }
            group->externals = key;
            group->lastUse   = frame;

            std::vector<VkDescriptorBufferInfo> infos(layers.size() * 5);
            std::vector<VkWriteDescriptorSet>   writes(layers.size() * 5, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
            for (size_t li = 0; li < layers.size(); ++li)
            {
                const Layer& l = layers[li];
                infos[li * 5 + 0] = TensorInfo(l.src[0], tensors, count);
                infos[li * 5 + 1] = TensorInfo(l.src[1], tensors, count);
                infos[li * 5 + 2] = {constantsDevice.buffer, l.weightsOffset, l.weightsSize};
                infos[li * 5 + 3] = {constantsDevice.buffer, l.paramsOffset, l.paramsSize};
                infos[li * 5 + 4] = TensorInfo(l.dst, tensors, count);
                for (uint32_t b = 0; b < 5; ++b)
                {
                    VkWriteDescriptorSet& w = writes[li * 5 + b];
                    w.dstSet                = group->sets[li];
                    w.dstBinding            = b;
                    w.descriptorCount       = 1;
                    w.descriptorType        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    w.pBufferInfo           = &infos[li * 5 + b];
                }
            }
            fn.UpdateDescriptorSets(ctx.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
            return group;
        }

        void Record(VkCommandBuffer cmd, const ExternalTensor* tensors, uint32_t count)
        {
            ++frame;
            if (!constantsUploaded)
            {
                VkBufferCopy copy = {0, 0, constantsStaging.size};
                fn.CmdCopyBuffer(cmd, constantsStaging.buffer, constantsDevice.buffer, 1, &copy);
                VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
                fn.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                constantsUploaded = true;
            }

            SetGroup* group = GetSetGroup(tensors, count);
            if (!group)
                return;

            if (profilePool)
                fn.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profilePool, profileFirst);
            for (size_t li = 0; li < layers.size(); ++li)
            {
                const Layer& l = layers[li];
                fn.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, l.pipeline);
                fn.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &group->sets[li], 0, nullptr);
                fn.CmdDispatch(cmd, l.groups[0], l.groups[1], l.groups[2]);
                if (profilePool)
                    fn.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profilePool, profileFirst + uint32_t(li) + 1);
                static const bool noBarriers = getenv("FFX_ML_EXPERIMENT_NO_BARRIERS") != nullptr;  // timing experiments only: wrong results
                if (li + 1 < layers.size() && !noBarriers)
                {
                    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
                    fn.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                }
            }
        }
    };

    std::unique_ptr<Graph> Graph::Create(const DeviceContext& context, const Selection& selection, const GraphDesc& desc, std::string* error)
    {
        std::unique_ptr<Graph> graph(new Graph());
        graph->m_selection   = selection;
        graph->m_impl        = std::make_unique<Impl>();
        Impl& impl           = *graph->m_impl;
        impl.ctx             = context;
        impl.selection       = selection;
        if (!impl.fn.Load(context))
        {
            if (error)
                *error = "failed to load Vulkan entry points";
            impl.ctx.device = VK_NULL_HANDLE;
            return nullptr;
        }
        if (!tosa::Parse(desc, impl.graph, error) || !Lower(impl.graph, impl.lowered, error) || !impl.Build(error))
            return nullptr;
        return graph;
    }

    Graph::~Graph() = default;

    void Graph::Record(VkCommandBuffer commandBuffer, const ExternalTensor* tensors, uint32_t tensorCount)
    {
        m_impl->Record(commandBuffer, tensors, tensorCount);
    }

    const std::vector<int64_t>* Graph::GetTensorShape(uint32_t binding) const
    {
        auto it = m_impl->shapes.find(binding);
        return it == m_impl->shapes.end() ? nullptr : &it->second;
    }

    uint32_t Graph::GetDispatchCount() const
    {
        return uint32_t(m_impl->layers.size());
    }

    void Graph::SetProfiling(VkQueryPool queryPool, uint32_t firstQuery)
    {
        m_impl->profilePool  = queryPool;
        m_impl->profileFirst = firstQuery;
    }

    std::string Graph::DescribeLayer(uint32_t index) const
    {
        const FusedConv&  fc   = m_impl->lowered.convs[index];
        const tosa::Node& conv = *fc.conv;
        std::string s = "conv" + std::to_string(conv.kernelH) + "x" + std::to_string(conv.kernelW) + " s" + std::to_string(conv.stride[0]) + " " +
                        std::to_string(fc.inC) + "->" + std::to_string(fc.outC) + " " + std::to_string(fc.inH) + "x" + std::to_string(fc.inW) + "->" +
                        std::to_string(fc.outH) + "x" + std::to_string(fc.outW);
        for (const ConvSource& src : fc.sources)
            if (src.resize)
                s += " (resize in)";
        if (fc.sources.size() > 1)
            s += " (concat in)";
        return s;
    }
}  // namespace ffx_ml
