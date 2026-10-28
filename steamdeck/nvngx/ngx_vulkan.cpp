// Vulkan NGX entry points: NSS records directly into the game's command buffer on the game's device.
// SPDX-License-Identifier: MIT
#include "cost_model.h"
#include "ngx_common.h"

#include "common.h"

#include <windows.h>

#include <ffx_api/vk/ffx_api_vk.h>

#include <cstring>
#include <mutex>
#include <vector>

namespace
{
std::mutex   g_mutex;
VulkanDevice g_vk;
bool         g_inited = false;

PFN_vkGetInstanceProcAddr LoaderGipa()
{
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    if (!vulkan)
        vulkan = LoadLibraryA("vulkan-1.dll");
    return vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")) : nullptr;
}

NVSDK_NGX_Result InitDevice(VkInstance instance, VkPhysicalDevice pd, VkDevice device, PFN_vkGetInstanceProcAddr gipa,
                            PFN_vkGetDeviceProcAddr gdpa)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!instance || !pd || !device)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    CostSetApi(CostApi::Vulkan);
    VulkanDevice vk;
    vk.instance       = instance;
    vk.physicalDevice = pd;
    vk.device         = device;
    vk.gipa           = gipa ? gipa : LoaderGipa();
    if (!vk.gipa)
        return NVSDK_NGX_Result_FAIL_PlatformError;
    vk.gdpa = gdpa ? gdpa : reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk.gipa(instance, "vkGetDeviceProcAddr"));
    // A Vulkan device's enabled features cannot be queried. The aiss Vulkan layer (AISS_LAYER=1) enables every ML
    // feature the GPU has on every device; without it only the scalar ML path is safe.
    vk.mlFeatures = Config().mlFeaturesForced ? ffxQueryMlDeviceFeaturesVK(instance, pd, vk.gipa, nullptr, nullptr) : 0;
    g_vk          = vk;
    g_inited      = true;
    LOGI("Vulkan init: VkDevice %p, ML features 0x%x%s", reinterpret_cast<void*>(device), vk.mlFeatures,
         Config().mlFeaturesForced ? "" : " (set AISS_LAYER=1 for the fast ML path)");
    return NVSDK_NGX_Result_Success;
}

// Extensions the upscalers can use. Only those the device has are requested.
const char* const kNssDeviceExtensions[] = {
    VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
    VK_KHR_8BIT_STORAGE_EXTENSION_NAME,               VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
    VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,      VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
    // FSR 4 through D3D12 interop (fsr4_interop.cpp): shared D3D12 textures and fence on the game's device.
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,      VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    // FSR 4 replacement shaders with v_dot2_f32_f16 (fsr4_vk.cpp).
    "VK_VALVE_shader_mixed_float_dot_product",
};

// NGX returns extension lists either by pointing the caller at its own array or by filling the caller's array.
NVSDK_NGX_Result ReturnExtensions(const std::vector<VkExtensionProperties>& list, uint32_t* count, VkExtensionProperties** props)
{
    static std::vector<VkExtensionProperties> storage;
    if (!count)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (props && *props)
    {
        const uint32_t n = std::min<uint32_t>(*count, static_cast<uint32_t>(list.size()));
        std::memcpy(*props, list.data(), n * sizeof(VkExtensionProperties));
        *count = n;
        return NVSDK_NGX_Result_Success;
    }
    storage = list;
    *count  = static_cast<uint32_t>(storage.size());
    if (props)
        *props = storage.empty() ? nullptr : storage.data();
    return NVSDK_NGX_Result_Success;
}

UpscalerImage Describe(const NgxParams& p, const char* name, VkImageLayout layout, bool& ok)
{
    UpscalerImage img;
    auto* res = static_cast<const NVSDK_NGX_Resource_VK*>(p.Ptr(name));
    if (!res || res->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW || !res->Resource.ImageViewInfo.Image)
    {
        ok = false;
        return img;
    }
    const auto& info = res->Resource.ImageViewInfo;
    img.image        = info.Image;
    img.format       = info.Format;
    img.width        = info.Width;
    img.height       = info.Height;
    img.layout       = layout;
    return img;
}

// Layout contract for NGX on Vulkan: inputs are shader-readable, the output is a storage image. AISS_VK_INPUT_LAYOUT
// overrides the input layout (the VkImageLayout number) for titles that keep inputs in GENERAL.
VkImageLayout InputLayout()
{
    static const VkImageLayout layout = [] {
        const char* v = getenv("AISS_VK_INPUT_LAYOUT");
        return v && *v ? static_cast<VkImageLayout>(atoi(v)) : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }();
    return layout;
}
} // namespace

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_RequiredExtensions(unsigned int* instCount, const char*** instExts,
                                                                unsigned int* devCount, const char*** devExts)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_RequiredExtensions");
    static const char* none[1] = {nullptr};
    if (instCount) *instCount = 0;
    if (instExts) *instExts = none;
    if (devCount) *devCount = 0;
    if (devExts) *devExts = none;
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init(unsigned long long, const wchar_t*, VkInstance inst, VkPhysicalDevice pd,
                                                  VkDevice dev, PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa,
                                                  const NVSDK_NGX_FeatureCommonInfo*, int)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Init");
    return InitDevice(inst, pd, dev, gipa, gdpa);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_Ext(unsigned long long, const wchar_t*, VkInstance inst, VkPhysicalDevice pd,
                                                      VkDevice dev, int, const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Init_Ext");
    return InitDevice(inst, pd, dev, nullptr, nullptr);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_Ext2(unsigned long long, const wchar_t*, VkInstance inst, VkPhysicalDevice pd,
                                                       VkDevice dev, PFN_vkGetInstanceProcAddr gipa,
                                                       PFN_vkGetDeviceProcAddr gdpa, int, const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Init_Ext2");
    return InitDevice(inst, pd, dev, gipa, gdpa);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_ProjectID(const char*, int, const char*, const wchar_t*, VkInstance inst,
                                                            VkPhysicalDevice pd, VkDevice dev, PFN_vkGetInstanceProcAddr gipa,
                                                            PFN_vkGetDeviceProcAddr gdpa, int, const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Init_ProjectID");
    return InitDevice(inst, pd, dev, gipa, gdpa);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_ProjectID_Ext(const char*, int, const char*, const wchar_t*, VkInstance inst,
                                                                VkPhysicalDevice pd, VkDevice dev, PFN_vkGetInstanceProcAddr gipa,
                                                                PFN_vkGetDeviceProcAddr gdpa, int,
                                                                const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Init_ProjectID_Ext");
    return InitDevice(inst, pd, dev, gipa, gdpa);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_with_ProjectID(const char*, int, const char*, const wchar_t*, VkInstance inst,
                                                                 VkPhysicalDevice pd, VkDevice dev, PFN_vkGetInstanceProcAddr gipa,
                                                                 PFN_vkGetDeviceProcAddr gdpa, const NVSDK_NGX_FeatureCommonInfo*, int)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Init_with_ProjectID");
    return InitDevice(inst, pd, dev, gipa, gdpa);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Shutdown()
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Shutdown");
    // NGX shutdown comes before the device is destroyed and after the game has idled it: free everything now.
    Registry().CollectGarbage(true);
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_Shutdown1(VkDevice)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_Shutdown1");
    return NVSDK_NGX_VULKAN_Shutdown();
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_VULKAN_GetParameters"); return NgxGetParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetCapabilityParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_VULKAN_GetCapabilityParameters"); return NgxAllocateParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_AllocateParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_VULKAN_AllocateParameters"); return NgxAllocateParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_DestroyParameters(NVSDK_NGX_Parameter* p) { Log(3, "%s", "NVSDK_NGX_VULKAN_DestroyParameters"); return NgxDestroyParameters(p); }

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_PopulateParameters_Impl(NVSDK_NGX_Parameter* p)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_PopulateParameters_Impl");
    if (!p)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    NgxParams::From(p)->FillCapabilities();
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetScratchBufferSize(NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, size_t* size)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_GetScratchBufferSize");
    if (!size)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *size = 0;
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureRequirements(VkInstance, VkPhysicalDevice,
                                                                    const NVSDK_NGX_FeatureDiscoveryInfo* info,
                                                                    NVSDK_NGX_FeatureRequirement* out)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_GetFeatureRequirements");
    return NgxFeatureRequirement(info ? info->FeatureID : NVSDK_NGX_Feature(0), out);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(const NVSDK_NGX_FeatureDiscoveryInfo*,
                                                                                     uint32_t* count, VkExtensionProperties** props)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements");
    return ReturnExtensions({}, count, props);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(VkInstance instance, VkPhysicalDevice pd,
                                                                                   const NVSDK_NGX_FeatureDiscoveryInfo* info,
                                                                                   uint32_t* count, VkExtensionProperties** props)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements");
    std::vector<VkExtensionProperties> wanted;
    PFN_vkGetInstanceProcAddr gipa = LoaderGipa();
    auto enumerate = gipa ? reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
                                gipa(instance, "vkEnumerateDeviceExtensionProperties"))
                          : nullptr;
    if (info && info->FeatureID == NVSDK_NGX_Feature_SuperSampling && enumerate && pd)
    {
        uint32_t n = 0;
        enumerate(pd, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> available(n);
        enumerate(pd, nullptr, &n, available.data());
        for (const char* name : kNssDeviceExtensions)
            for (const auto& e : available)
                if (!strcmp(e.extensionName, name))
                    wanted.push_back(e);
    }
    return ReturnExtensions(wanted, count, props);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_CreateFeature1(VkDevice, VkCommandBuffer, NVSDK_NGX_Feature feature,
                                                            NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_CreateFeature1");
    if (feature != NVSDK_NGX_Feature_SuperSampling)
    {
        LOGI("CreateFeature: feature %d is not provided", feature);
        return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    }
    if (!params || !outHandle)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (!g_inited)
        return NVSDK_NGX_Result_FAIL_NotInitialized;

    Registry().CollectGarbage();
    auto upscaler = CreateUpscaler(g_vk, NgxRemapVulkan(NgxCreateInfo(*NgxParams::From(params))));
    if (!upscaler)
        return NVSDK_NGX_Result_FAIL_PlatformError;
    *outHandle = NgxNewHandle(Registry().Add(std::move(upscaler)));
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_CreateFeature(VkCommandBuffer cmd, NVSDK_NGX_Feature feature,
                                                           NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_CreateFeature");
    return NVSDK_NGX_VULKAN_CreateFeature1(g_vk.device, cmd, feature, params, outHandle);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_ReleaseFeature(NVSDK_NGX_Handle* handle)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_ReleaseFeature");
    if (!handle)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    Registry().Release(handle->Id);
    NgxFreeHandle(handle);
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_EvaluateFeature(VkCommandBuffer cmd, const NVSDK_NGX_Handle* handle,
                                                             NVSDK_NGX_Parameter* params, PFN_NVSDK_NGX_ProgressCallback)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_EvaluateFeature");
    if (!cmd || !handle || !params)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    Upscaler* upscaler = Registry().Get(handle->Id);
    if (upscaler)
        upscaler = NgxApplyLiveSettings(handle->Id, upscaler, {}, NgxRemapVulkan);
    if (!upscaler)
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;

    const NgxParams& p  = *NgxParams::From(params);
    bool             ok = true;
    UpscalerDispatch d;
    d.color         = Describe(p, ngxp::Color, InputLayout(), ok);
    d.depth         = Describe(p, ngxp::Depth, InputLayout(), ok);
    d.motionVectors = Describe(p, ngxp::MotionVectors, InputLayout(), ok);
    d.output        = Describe(p, ngxp::Output, VK_IMAGE_LAYOUT_GENERAL, ok);
    if (!ok)
    {
        LOGI("EvaluateFeature: missing input resource");
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NgxFillDispatch(p, upscaler->Info(), d);
    return NgxDispatch(upscaler, cmd, d) ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_VULKAN_EvaluateFeature_C(VkCommandBuffer cmd, const NVSDK_NGX_Handle* handle,
                                                               const NVSDK_NGX_Parameter* params, void*)
{
    Log(3, "%s", "NVSDK_NGX_VULKAN_EvaluateFeature_C");
    return NVSDK_NGX_VULKAN_EvaluateFeature(cmd, handle, const_cast<NVSDK_NGX_Parameter*>(params), nullptr);
}
