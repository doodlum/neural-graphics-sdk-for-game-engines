// AMD FSR 4 (Vulkan) as an alternative to NSS, from amdvkffx64.dll: AMD's Vulkan driver component, redistributed
// unmodified and loaded at runtime from next to the proxy. The proxy talks to it only through the FidelityFX API
// provider it hands out (UpdateFfxApiProvider), plus a few in-memory redirections, never by changing the file:
//
//  - The DLL only hands out providers after checking that signed FidelityFX SDK modules are loaded, looked up by name
//    with GetModuleHandleW. The proxy points the DLL's GetModuleHandleW import at a lookup that answers those names with
//    the DLL itself, which AMD signed with the same certificate (its file version then counts as the SDK version).
//    Wine's WinVerifyTrust accepts AMD's signature, so nothing else is needed for the check.
//  - It asks the Windows kernel driver for the GPU family and revision (D3DKMTQueryAdapterInfo with the AMD driver's
//    private data), which Wine does not have, so on its own it rejects every FSR 4 network under Proton. The proxy points
//    its GetProcAddress import at a lookup that hands it a D3DKMT pair answering with the GPU's amdgpu family and
//    revision (Van Gogh: 144 / 1).
//  - Its shaders declare images with 16-bit float texels, which only AMD's own Vulkan driver accepts; RADV crashes on
//    them. The DLL gets a vkGetDeviceProcAddr whose vkCreateShaderModule rewrites them (spirv_half_images.cpp).
//  - Every FSR 4 network is a provider class of its own whose device check FidelityFX runs in order until one accepts;
//    the checks are wrapped (vtable entries) so that the model the user picked is the one that accepts.
//
// The ABI facts this relies on (provider data block layout, the Vulkan backend description type, the upscaler version
// description, the per-model provider classes and their check slot) were learnt from reading OptiScaler's notes and
// from disassembling amdvkffx64.dll; none of OptiScaler's code is used here.
// SPDX-License-Identifier: MIT
#include "nss_upscaler.h"

#include "common.h"
#include "fsr4_common.h"
#include "spirv_half_images.h"

#include <windows.h>

// AMD's own FidelityFX API types (the Arm fork used for NSS numbers formats and resource states differently).
#include "amd_ffx/ffx_api.h"
#include "amd_ffx/ffx_upscale.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace
{
// Vulkan backend create-desc type amdvkffx64's FidelityFX runtime understands, and the upscaler version desc it wants.
constexpr uint64_t kDriverBackendVk    = 0x01000003u;
constexpr uint64_t kUpscaleVersionDesc = 0x0001000Bu;
constexpr uint32_t kUpscaleVersion     = (4u << 22) | (1u << 12) | 2u;

// The Vulkan backend description the DLL's FidelityFX runtime understands (laid out like the SDK 1.x
// ffxCreateBackendVKDesc: device, physical device, device proc addr).
struct DriverVkDevice
{
    VkDevice                device;
    VkPhysicalDevice        physicalDevice;
    PFN_vkGetDeviceProcAddr gdpa;
};
struct DriverBackendVkDesc
{
    ffxCreateContextDescHeader header;
    DriverVkDevice             vk;
};

// The DLL logs an error without the upscaler version desc and accepts 4.0.0 to 4.1.2.
struct UpscaleVersionDesc
{
    ffxCreateContextDescHeader header;
    uint32_t                   version;
};

// ---------------------------------------------------------------------------------------------------------------------
// Shader modules the DLL creates pass through the proxy (it hands the DLL a vkGetDeviceProcAddr of its own).
struct DeviceShaderFns
{
    VkDevice                device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr gdpa   = nullptr;
    PFN_vkCreateShaderModule create = nullptr;
};
std::mutex                   g_deviceMutex;
std::vector<DeviceShaderFns> g_devices;
std::atomic<uint32_t>        g_shaderCount{0};

DeviceShaderFns DeviceFns(VkDevice device)
{
    std::lock_guard<std::mutex> lock(g_deviceMutex);
    for (const DeviceShaderFns& d : g_devices)
        if (d.device == device)
            return d;
    return {};
}

std::vector<uint32_t> AdaptSpirv(const uint32_t* code, size_t words);  // below

VKAPI_ATTR VkResult VKAPI_CALL ShaderModuleThunk(VkDevice device, const VkShaderModuleCreateInfo* info,
                                                         const VkAllocationCallbacks* allocator, VkShaderModule* module)
{
    const DeviceShaderFns fns = DeviceFns(device);
    if (!fns.create)
        return VK_ERROR_INITIALIZATION_FAILED;
    const uint32_t n = g_shaderCount++;
    if (const char* dir = getenv("AISS_FSR4_DUMP_SPIRV"))
    {
        char path[MAX_PATH];
        snprintf(path, sizeof(path), "%s\\fsr4_%03u.spv", dir, n);
        if (FILE* f = fopen(path, "wb"))
        {
            fwrite(info->pCode, 1, info->codeSize, f);
            fclose(f);
        }
    }
    if (!info || !info->pCode || info->codeSize % 4)
        return fns.create(device, info, allocator, module);
    std::vector<uint32_t> adapted = AdaptSpirv(info->pCode, info->codeSize / 4);
    if (adapted.empty())
        return fns.create(device, info, allocator, module);
    VkShaderModuleCreateInfo copy = *info;
    copy.pCode                    = adapted.data();
    copy.codeSize                 = adapted.size() * 4;
    Log(2, "FSR 4: shader %u adapted (%u -> %u bytes)", n, unsigned(info->codeSize), unsigned(copy.codeSize));
    return fns.create(device, &copy, allocator, module);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL DeviceProcAddrThunk(VkDevice device, const char* name)
{
    if (name && !strcmp(name, "vkCreateShaderModule"))
        return reinterpret_cast<PFN_vkVoidFunction>(&ShaderModuleThunk);
    const DeviceShaderFns fns = DeviceFns(device);
    return fns.gdpa ? fns.gdpa(device, name) : nullptr;
}

void RegisterDevice(const VulkanDevice& dev)
{
    std::lock_guard<std::mutex> lock(g_deviceMutex);
    for (const DeviceShaderFns& d : g_devices)
        if (d.device == dev.device)
            return;
    DeviceShaderFns d;
    d.device = dev.device;
    d.gdpa   = dev.gdpa;
    d.create = reinterpret_cast<PFN_vkCreateShaderModule>(dev.gdpa(dev.device, "vkCreateShaderModule"));
    g_devices.push_back(d);
}

std::vector<uint32_t> AdaptSpirv(const uint32_t* code, size_t words)
{
    if (getenv("AISS_FSR4_NO_SPIRV_FIX"))
        return {};
    HalfImageStats        st;
    std::vector<uint32_t> out = PromoteHalfImages(code, words, &st);
    if (!out.empty())
        Log(2, "FSR 4: shader: %u half-float image types -> 32-bit, %u reads, %u writes converted%s", st.images, st.reads,
            st.writes, st.unhandled ? " (sparse image reads left alone)" : "");
    return out;
}

void FfxMessage(uint32_t type, const wchar_t* message)
{
    LOGI("FSR 4 runtime %s: %ls", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", message);
}

bool IsDepth(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_X8_D24_UNORM_PACK32 ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

bool HasStencil(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

uint32_t SurfaceFormat(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_R32G32B32A32_SFLOAT: return FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT;
    case VK_FORMAT_R32G32B32A32_UINT: return FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT;
    case VK_FORMAT_R32G32B32_SFLOAT: return FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_R32G32_SFLOAT: return FFX_API_SURFACE_FORMAT_R32G32_FLOAT;
    case VK_FORMAT_R32G32_UINT: return FFX_API_SURFACE_FORMAT_R32G32_UINT;
    case VK_FORMAT_R8G8B8A8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM: return FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
    case VK_FORMAT_R16G16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16G16_FLOAT;
    case VK_FORMAT_R16G16_UINT: return FFX_API_SURFACE_FORMAT_R16G16_UINT;
    case VK_FORMAT_R16G16_SINT: return FFX_API_SURFACE_FORMAT_R16G16_SINT;
    case VK_FORMAT_R16_SFLOAT: return FFX_API_SURFACE_FORMAT_R16_FLOAT;
    case VK_FORMAT_R16_UINT: return FFX_API_SURFACE_FORMAT_R16_UINT;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT: return FFX_API_SURFACE_FORMAT_R16_UNORM;
    case VK_FORMAT_R16_SNORM: return FFX_API_SURFACE_FORMAT_R16_SNORM;
    case VK_FORMAT_R8_UNORM: return FFX_API_SURFACE_FORMAT_R8_UNORM;
    case VK_FORMAT_R8_UINT: return FFX_API_SURFACE_FORMAT_R8_UINT;
    case VK_FORMAT_R8_SNORM: return FFX_API_SURFACE_FORMAT_R8_SNORM;
    case VK_FORMAT_R8G8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8_UNORM;
    case VK_FORMAT_R8G8_UINT: return FFX_API_SURFACE_FORMAT_R8G8_UINT;
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D24_UNORM_S8_UINT: return FFX_API_SURFACE_FORMAT_R32_UINT;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP;
    default: return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}

FfxApiResource Resource(const UpscalerImage& img, uint32_t usage, uint32_t state)
{
    FfxApiResource r       = {};
    r.resource             = reinterpret_cast<void*>(img.image);
    r.state                = state;
    r.description.type     = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format   = SurfaceFormat(img.format);
    r.description.width    = img.width;
    r.description.height   = img.height;
    r.description.depth    = 1;
    r.description.mipCount = 1;
    r.description.flags    = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage    = usage | (IsDepth(img.format) ? FFX_API_RESOURCE_USAGE_DEPTHTARGET : 0u) |
                          (HasStencil(img.format) ? FFX_API_RESOURCE_USAGE_STENCILTARGET : 0u);
    return r;
}

// amdvkffx64.dll, loaded once. The wrapped device checks keep a pointer to it, so it must outlive every upscaler.
std::mutex g_mutex;
bool       g_loadTried = false, g_loadOk = false;
bool       g_modelFailed[kFsr4ModelCount] = {};
Fsr4Driver g_drv;

Fsr4Gpu GpuForDevice(const VulkanDevice& device)
{
    VkPhysicalDeviceProperties props = {};
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(device.gipa(device.instance, "vkGetPhysicalDeviceProperties"))(
        device.physicalDevice, &props);
    return Fsr4KnownGpu(props.vendorID, props.deviceID, props.deviceName);
}

const Fsr4Driver* LoadVkDriver(const VulkanDevice& device)
{
    if (g_loadTried)
        return g_loadOk ? &g_drv : nullptr;
    g_loadTried = true;
    VkPhysicalDeviceProperties props = {};
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(device.gipa(device.instance, "vkGetPhysicalDeviceProperties"))(
        device.physicalDevice, &props);
    const Fsr4Gpu gpu = Fsr4KnownGpu(props.vendorID, props.deviceID, props.deviceName);
    if (!gpu.family)
        LOGI("FSR 4: GPU %04x:%04x (%s) has no known amdgpu family; AMD's automatic model choice will refuse it "
             "(set AISS_FSR4_GPU=family,revision)", props.vendorID, props.deviceID, props.deviceName);
    g_loadOk = Fsr4LoadDriver(L"amdvkffx64.dll", gpu, g_drv);
    return g_loadOk ? &g_drv : nullptr;
}

// VK_EXT_shader_float8 + VK_KHR_cooperative_matrix, which the FP8 network's shaders need (RADV crashes without them).
bool DeviceHasFp8(const VulkanDevice& device)
{
    auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        device.gipa(device.instance, "vkEnumerateDeviceExtensionProperties"));
    uint32_t n = 0;
    enumerate(device.physicalDevice, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    enumerate(device.physicalDevice, nullptr, &n, exts.data());
    bool fp8 = false, coop = false;
    for (const VkExtensionProperties& e : exts)
    {
        fp8 |= !strcmp(e.extensionName, "VK_EXT_shader_float8");
        coop |= !strcmp(e.extensionName, "VK_KHR_cooperative_matrix");
    }
    return fp8 && coop;
}

class Fsr4Upscaler : public Upscaler
{
public:
    ~Fsr4Upscaler() override;
    bool Init(const VulkanDevice& device, const UpscalerCreateInfo& info);
    bool Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d) override;

private:
    const Fsr4Driver*        m_drv    = nullptr;
    ffxContext               m_ctx    = nullptr;
    int                      m_model  = kFsr4Auto;  // the network that runs
    uint64_t                 m_dispatchCount = 0;
    PFN_vkCmdPipelineBarrier m_barrier = nullptr;
    PFN_vkCmdClearColorImage m_clear   = nullptr;

    // FSR 4 needs an exposure: its own auto exposure, or a 1x1 texture. Without DLSS auto exposure, the proxy's
    // R32_SFLOAT texture is cleared to the exposure (1, or AISS_EXPOSURE) before every dispatch.
    bool CreateExposure();
    VkImage        m_exposure       = VK_NULL_HANDLE;
    VkDeviceMemory m_exposureMemory = VK_NULL_HANDLE;
    bool           m_autoExposure   = false;
};

bool Fsr4Upscaler::CreateExposure()
{
    auto fn = [&](const char* name) { return m_dev.gdpa(m_dev.device, name); };
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType         = VK_IMAGE_TYPE_2D;
    ci.format            = VK_FORMAT_R32_SFLOAT;
    ci.extent            = {1, 1, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples                    = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling                     = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (reinterpret_cast<PFN_vkCreateImage>(fn("vkCreateImage"))(m_dev.device, &ci, nullptr, &m_exposure) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req = {};
    reinterpret_cast<PFN_vkGetImageMemoryRequirements>(fn("vkGetImageMemoryRequirements"))(m_dev.device, m_exposure, &req);
    VkPhysicalDeviceMemoryProperties mp = {};
    reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(m_dev.gipa(m_dev.instance, "vkGetPhysicalDeviceMemoryProperties"))(
        m_dev.physicalDevice, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            type = i;
    if (type == UINT32_MAX)
        return false;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize       = req.size;
    ai.memoryTypeIndex      = type;
    if (reinterpret_cast<PFN_vkAllocateMemory>(fn("vkAllocateMemory"))(m_dev.device, &ai, nullptr, &m_exposureMemory) != VK_SUCCESS)
        return false;
    return reinterpret_cast<PFN_vkBindImageMemory>(fn("vkBindImageMemory"))(m_dev.device, m_exposure, m_exposureMemory, 0) ==
           VK_SUCCESS;
}

bool Fsr4Upscaler::Init(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    m_dev     = device;
    m_info    = info;
    m_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(device.gdpa(device.device, "vkCmdPipelineBarrier"));
    m_clear   = reinterpret_cast<PFN_vkCmdClearColorImage>(device.gdpa(device.device, "vkCmdClearColorImage"));

    m_drv = LoadVkDriver(device);
    if (!m_drv)
        return false;
    const int wanted = info.fsr4Model;
    if (wanted == kFsr4Fp8 && !getenv("AISS_FSR4_FORCE_FP8") && !DeviceHasFp8(device))
    {
        LOGI("FSR 4: the FP8 network needs VK_EXT_shader_float8 and VK_KHR_cooperative_matrix (RDNA 4), which this "
             "GPU does not have");
        return false;
    }
    if (wanted != kFsr4Auto && !m_drv->HasModel(wanted))
    {
        LOGI("FSR 4: amdvkffx64.dll has no %s network", Fsr4ModelName(wanted));
        return false;
    }

    DriverBackendVkDesc backend = {};
    backend.header.type         = kDriverBackendVk;
    RegisterDevice(device);
    backend.vk                  = {device.device, device.physicalDevice, &DeviceProcAddrThunk};

    ffxCreateContextDescUpscale create = {};
    create.header.type                 = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags                       = FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
    if (info.hdr)
        create.flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    if (info.depthInverted)
        create.flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    if (info.highResMv)
        create.flags |= FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    // Like NSS and XeSS: FSR's own auto exposure when the game asked DLSS for it, unless AISS_EXPOSURE forces a value.
    m_autoExposure = info.autoExposure && Config().exposure <= 0.0f;
    if (m_autoExposure)
        create.flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    else if (!CreateExposure())
    {
        LOGI("FSR 4: cannot create the exposure texture");
        return false;
    }
    if (LogLevel() >= 2)
        create.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    create.maxRenderSize  = {std::min(info.renderWidth, info.displayWidth), std::min(info.renderHeight, info.displayHeight)};
    create.maxUpscaleSize = {info.displayWidth, info.displayHeight};
    create.fpMessage      = &FfxMessage;

    UpscaleVersionDesc version = {};
    version.header.type        = kUpscaleVersionDesc;
    version.version            = kUpscaleVersion;
    create.header.pNext        = &backend.header;
    backend.header.pNext       = &version.header;

    const auto t0 = GetTickCount64();
    m_model = Fsr4CreateWithModel(*m_drv, wanted, GpuForDevice(device), &m_ctx, &create.header);
    if (m_model < 0)
    {
        m_ctx = nullptr;
        return false;
    }
    LOGI("FSR 4 (Vulkan) context %ux%u -> %ux%u, flags 0x%x, created in %llu ms", create.maxRenderSize.width,
         create.maxRenderSize.height, info.displayWidth, info.displayHeight, create.flags,
         (unsigned long long)(GetTickCount64() - t0));
    m_info.fsr4Model  = wanted;
    m_info.fsr4Active = m_model;
    return true;
}

Fsr4Upscaler::~Fsr4Upscaler()
{
    if (m_ctx)
        m_drv->provider.destroyContext(&m_ctx, nullptr);
    if (m_exposure)
        reinterpret_cast<PFN_vkDestroyImage>(m_dev.gdpa(m_dev.device, "vkDestroyImage"))(m_dev.device, m_exposure, nullptr);
    if (m_exposureMemory)
        reinterpret_cast<PFN_vkFreeMemory>(m_dev.gdpa(m_dev.device, "vkFreeMemory"))(m_dev.device, m_exposureMemory, nullptr);
}

bool Fsr4Upscaler::Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d)
{
    // Inputs in SHADER_READ_ONLY_OPTIMAL (FFX "compute read"), the output in GENERAL (FFX "unordered access"); the
    // FidelityFX backend moves them to its working layouts and back.
    VkImageMemoryBarrier before[4], after[4];
    uint32_t             count = 0;
    auto transition = [&](const UpscalerImage& img, VkImageLayout want, VkAccessFlags access) {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask        = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask        = access;
        b.oldLayout            = img.layout;
        b.newLayout            = want;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image                = img.image;
        const VkImageAspectFlags aspect =
            IsDepth(img.format) ? VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(img.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
                                : VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange = {aspect, 0, 1, 0, 1};
        before[count]      = b;
        std::swap(b.oldLayout, b.newLayout);
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        after[count++]  = b;
    };
    for (const UpscalerImage* img : {&d.color, &d.motionVectors, &d.depth})
        transition(*img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    transition(d.output, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    if (m_exposure)
    {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask        = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image                = m_exposure;
        b.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        m_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkClearColorValue value = {};
        value.float32[0]        = d.exposure;
        m_clear(cmd, m_exposure, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &b.subresourceRange);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        m_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
    m_barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
              count, before);

    ffxDispatchDescUpscale up = {};
    up.header.type            = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    up.commandList            = cmd;
    up.color         = Resource(d.color, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    up.depth         = Resource(d.depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    up.motionVectors = Resource(d.motionVectors, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    up.output        = Resource(d.output, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    if (m_exposure)
        up.exposure = Resource({m_exposure, VK_FORMAT_R32_SFLOAT, 1, 1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                               FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    // DLSS and FSR share the jitter convention (sample offset in render pixels); AISS_JITTER_SIGN still applies.
    up.jitterOffset      = {d.jitterX * Config().jitterSign[0], d.jitterY * Config().jitterSign[1]};
    // Motion vectors scaled to render pixels, current -> previous, as DLSS's.
    up.motionVectorScale = {d.mvScaleX * Config().mvSign[0], d.mvScaleY * Config().mvSign[1]};
    // Dynamic resolution: the per-frame render size (DLSS render subrect), at most the textures' size.
    up.renderSize  = {std::min(d.renderWidth, d.color.width), std::min(d.renderHeight, d.color.height)};
    up.upscaleSize = {d.output.width, d.output.height};
    up.enableSharpening        = false;
    up.sharpness               = 0.0f;
    up.frameTimeDelta          = d.frameTimeMs;
    up.preExposure             = 1.0f;  // the game's colour is not pre-exposed (DLSS has no pre-exposure)
    up.reset                   = d.reset || m_dispatchCount == 0;
    up.cameraNear              = Config().cameraNear;
    up.cameraFar               = Config().cameraFar;
    up.cameraFovAngleVertical  = Config().cameraFovY;
    up.viewSpaceToMetersFactor = 1.0f;
    if (m_info.depthInverted)  // FidelityFX convention for inverted depth: near and far swapped
        std::swap(up.cameraNear, up.cameraFar);
    const uint32_t rc = m_drv->provider.dispatch(&m_ctx, &up.header);
    ++m_dispatchCount;

    m_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
              count, after);
    if ((m_dispatchCount % 600) == 1 || LogLevel() >= 2)
        LOGI("FSR 4 (%s) evaluate %llu: render %ux%u -> %ux%u, jitter %.3f,%.3f, mv scale %.2f,%.2f%s",
             Fsr4ModelName(m_model), (unsigned long long)m_dispatchCount, up.renderSize.width, up.renderSize.height,
             up.upscaleSize.width, up.upscaleSize.height, up.jitterOffset.x, up.jitterOffset.y, up.motionVectorScale.x,
             up.motionVectorScale.y, m_autoExposure ? " (auto exposure)" : "");
    if (rc != FFX_API_RETURN_OK)
    {
        LOGI("FSR 4: dispatch failed (%u)", rc);
        return false;
    }
    return true;
}
}  // namespace

const char* Fsr4ModelName(int model)
{
    switch (model)
    {
    case kFsr4Auto: return "auto";
    case kFsr4Lite: return "Lite";
    case kFsr4Int8: return "INT8";
    case kFsr4Rdna2: return "INT8 RDNA 2";
    case kFsr4Fp8: return "FP8";
    default: return "?";
    }
}

int Fsr4ModelFromName(const char* name)
{
    if (!name)
        return -1;
    std::string s(name);
    for (char& c : s)
        c = char(tolower(c));
    if (s == "auto" || s.empty())
        return kFsr4Auto;
    if (s == "lite" || s == "int8lite" || s == "int8-lite")
        return kFsr4Lite;
    if (s == "int8")
        return kFsr4Int8;
    if (s == "rdna2" || s == "navi2" || s == "int8-rdna2" || s == "int8rdna2")
        return kFsr4Rdna2;
    if (s == "fp8")
        return kFsr4Fp8;
    return -1;
}

bool Fsr4Unavailable(int model)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loadTried && !g_loadOk)
        return true;
    return model >= 0 && model < kFsr4ModelCount && g_modelFailed[model];
}

std::unique_ptr<Upscaler> CreateFsr4Upscaler(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    const int model = info.fsr4Model >= 0 && info.fsr4Model < kFsr4ModelCount ? info.fsr4Model : kFsr4Auto;
    if ((g_loadTried && !g_loadOk) || g_modelFailed[model])
        return nullptr;
    UpscalerCreateInfo fixed = info;
    fixed.fsr4Model          = model;
    std::unique_ptr<Fsr4Upscaler> up(new Fsr4Upscaler());
    if (up->Init(device, fixed))
        return up;
    g_modelFailed[model] = true;
    return nullptr;
}
