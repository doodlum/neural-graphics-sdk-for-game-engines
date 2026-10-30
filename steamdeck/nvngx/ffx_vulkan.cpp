// AMD FidelityFX API front end on Vulkan: a game's FSR upscaling, routed to the proxy's upscalers. The game keeps
// loading its own amd_fidelityfx_vk.dll name; the plugin's forwarder in its place (steamdeck/ffxfwd) sends every upscale
// context here and everything else (frame generation, queries without a context) to AMD's original DLL.
// SPDX-License-Identifier: MIT
#include "cost_model.h"
#include "ngx_common.h"

#include "common.h"

#include <windows.h>

#include "amd_ffx/ffx_api.h"
#include "amd_ffx/ffx_api_vk.h"
#include "amd_ffx/ffx_upscale.h"

#include <cstdio>
#include <cstring>
#include <mutex>

namespace
{
std::mutex g_mutex;

// vulkan-1.dll's own exports (vkGetInstanceProcAddr, vkGetDeviceProcAddr).
PFN_vkVoidFunction VKAPI_CALL ExportGipa(VkInstance instance, const char* name)
{
    static HMODULE vulkan = [] {
        HMODULE m = GetModuleHandleA("vulkan-1.dll");
        return m ? m : LoadLibraryA("vulkan-1.dll");
    }();
    if (!vulkan)
        return nullptr;
    if (auto fn = reinterpret_cast<PFN_vkVoidFunction>(GetProcAddress(vulkan, name)))
        return fn;
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr"));
    return gipa && instance ? gipa(instance, name) : nullptr;
}

// The instance of a physical device, as the loader shim recorded it (AISS_VK_PHYSDEV, see loader_shim.cpp).
VkInstance InstanceFor(VkPhysicalDevice pd)
{
    char list[1024] = {};
    if (!GetEnvironmentVariableA("AISS_VK_PHYSDEV", list, sizeof(list)))
        return VK_NULL_HANDLE;
    for (char* entry = list; *entry;)
    {
        char* end = strchr(entry, ';');
        if (end)
            *end = 0;
        void *dev = nullptr, *inst = nullptr;
        if (sscanf(entry, "%p=%p", &dev, &inst) == 2 && dev == static_cast<void*>(pd))
            return static_cast<VkInstance>(inst);
        if (!end)
            break;
        entry = end + 1;
    }
    return VK_NULL_HANDLE;
}

VkFormat VkFormatFor(uint32_t format, uint32_t usage)
{
    const bool depth = (usage & FFX_API_RESOURCE_USAGE_DEPTHTARGET) != 0;
    switch (format)
    {
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R32G32_FLOAT:
    case FFX_API_SURFACE_FORMAT_R32G32_TYPELESS: return VK_FORMAT_R32G32_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS:
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case FFX_API_SURFACE_FORMAT_R16G16_FLOAT:
    case FFX_API_SURFACE_FORMAT_R16G16_TYPELESS: return VK_FORMAT_R16G16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
    case FFX_API_SURFACE_FORMAT_R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;  // id Tech's HDR colour
    case FFX_API_SURFACE_FORMAT_R32_FLOAT:
    case FFX_API_SURFACE_FORMAT_R32_TYPELESS: return depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R32_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

VkImageLayout VkLayoutFor(uint32_t state)
{
    if (state & FFX_API_RESOURCE_STATE_UNORDERED_ACCESS)
        return VK_IMAGE_LAYOUT_GENERAL;
    if (state & (FFX_API_RESOURCE_STATE_COMPUTE_READ | FFX_API_RESOURCE_STATE_PIXEL_READ))
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (state & FFX_API_RESOURCE_STATE_COPY_SRC)
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    if (state & FFX_API_RESOURCE_STATE_COPY_DEST)
        return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    if (state & FFX_API_RESOURCE_STATE_RENDER_TARGET)
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (state & FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT)
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    return VK_IMAGE_LAYOUT_GENERAL;
}

UpscalerImage Describe(const FfxApiResource& r, bool& ok)
{
    UpscalerImage img;
    img.image  = static_cast<VkImage>(r.resource);
    img.format = VkFormatFor(r.description.format, r.description.usage);
    img.width  = r.description.width;
    img.height = r.description.height;
    img.layout = VkLayoutFor(r.state);
    if (!img.image || img.format == VK_FORMAT_UNDEFINED)
        ok = false;
    return img;
}
}  // namespace

// Creates an upscaler for an FFX upscale context on the game's Vulkan device; returns its handle, 0 on failure.
extern "C" __declspec(dllexport) uint32_t AISS_FfxCreateUpscaleVK(const ffxCreateContextDescUpscale* desc, VkDevice device,
                                                                  VkPhysicalDevice physicalDevice, PFN_vkGetDeviceProcAddr gdpa)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!desc || !device || !physicalDevice)
        return 0;
    CostSetApi(CostApi::Vulkan);
    VulkanDevice vk;
    vk.instance       = InstanceFor(physicalDevice);
    vk.physicalDevice = physicalDevice;
    vk.device         = device;
    vk.gipa           = reinterpret_cast<PFN_vkGetInstanceProcAddr>(ExportGipa(VK_NULL_HANDLE, "vkGetInstanceProcAddr"));
    vk.gdpa           = gdpa ? gdpa : reinterpret_cast<PFN_vkGetDeviceProcAddr>(ExportGipa(VK_NULL_HANDLE, "vkGetDeviceProcAddr"));
    vk.mlFeatures     = Config().mlFeaturesForced ? VkMlFeatures(vk.instance, physicalDevice, vk.gipa) : 0;

    UpscalerCreateInfo info;
    info.renderWidth   = desc->maxRenderSize.width;
    info.renderHeight  = desc->maxRenderSize.height;
    info.displayWidth  = desc->maxUpscaleSize.width;
    info.displayHeight = desc->maxUpscaleSize.height;
    info.hdr           = (desc->flags & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE) != 0;
    info.depthInverted = (desc->flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) != 0;
    info.autoExposure  = (desc->flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) != 0;
    info.highResMv     = (desc->flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    info.nssQuality    = NssQualityForDlss(NVSDK_NGX_PerfQuality_Value_MaxPerf);
    info.fastMath      = FastMathEnabled();
    info.backend       = SelectedBackend(&info.fsr4Model);
    if (!vk.instance)
    {
        // The SDK's Vulkan backend and XeSS need the VkInstance, which an FFX context does not hand over; the loader
        // shim records it. Without it the game keeps AMD's upscaler.
        LOGI("FSR (FFX) create: no VkInstance recorded for VkPhysicalDevice %p; leaving this context to AMD's FSR",
             reinterpret_cast<void*>(physicalDevice));
        return 0;
    }
    LOGI("FSR (FFX) create: %ux%u -> %ux%u, flags 0x%x, VkDevice %p, instance %p", info.renderWidth, info.renderHeight,
         info.displayWidth, info.displayHeight, desc->flags, reinterpret_cast<void*>(device),
         reinterpret_cast<void*>(vk.instance));
    Registry().CollectGarbage();
    auto upscaler = CreateUpscaler(vk, NgxRemapVulkan(info));
    if (!upscaler)
        return 0;
    return Registry().Add(std::move(upscaler));
}

extern "C" __declspec(dllexport) uint32_t AISS_FfxDispatchUpscaleVK(uint32_t id, const ffxDispatchDescUpscale* desc)
{
    if (!desc || !desc->commandList)
        return FFX_API_RETURN_ERROR_PARAMETER;
    Upscaler* upscaler = Registry().Get(id);
    if (upscaler)
        upscaler = NgxApplyLiveSettings(id, upscaler, {}, NgxRemapVulkan);
    if (!upscaler)
        return FFX_API_RETURN_ERROR_PARAMETER;
    bool             ok = true;
    UpscalerDispatch d;
    d.color         = Describe(desc->color, ok);
    d.depth         = Describe(desc->depth, ok);
    d.motionVectors = Describe(desc->motionVectors, ok);
    d.output        = Describe(desc->output, ok);
    if (!ok)
    {
        static bool warned = false;
        if (!warned)
            LOGI("FSR (FFX) dispatch: missing input or unknown format (colour %u, depth %u, mv %u, output %u)",
                 desc->color.description.format, desc->depth.description.format, desc->motionVectors.description.format,
                 desc->output.description.format);
        warned = true;
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    d.renderWidth  = desc->renderSize.width ? desc->renderSize.width : upscaler->Info().renderWidth;
    d.renderHeight = desc->renderSize.height ? desc->renderSize.height : upscaler->Info().renderHeight;
    d.jitterX      = desc->jitterOffset.x * Config().jitterSign[0];
    d.jitterY      = desc->jitterOffset.y * Config().jitterSign[1];
    d.mvScaleX     = desc->motionVectorScale.x * Config().mvSign[0];
    d.mvScaleY     = desc->motionVectorScale.y * Config().mvSign[1];
    d.reset        = desc->reset;
    d.frameTimeMs  = desc->frameTimeDelta > 0.0f ? desc->frameTimeDelta : 16.6f;
    d.exposure     = Config().exposure > 0.0f ? Config().exposure : 1.0f;
    return NgxDispatch(upscaler, static_cast<VkCommandBuffer>(desc->commandList), d) ? FFX_API_RETURN_OK
                                                                                    : FFX_API_RETURN_ERROR;
}

extern "C" __declspec(dllexport) void AISS_FfxDestroyUpscale(uint32_t id)
{
    Registry().Release(id);
}
