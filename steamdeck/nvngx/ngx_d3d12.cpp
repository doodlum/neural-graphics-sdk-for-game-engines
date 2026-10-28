// D3D12 NGX entry points. On Proton, D3D12 is vkd3d-proton, so NSS runs on the game's own VkDevice and records
// straight into the game's command list through vkd3d-proton's interop interface. No copies, no extra submissions.
// SPDX-License-Identifier: MIT
#include "cost_model.h"
#include "ngx_common.h"

#include "common.h"
#include "fsr4_dx12.h"

#include <d3d12.h>
#include <windows.h>

#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
#include <memory>
#include <mutex>

// vkd3d-proton interop interfaces (include/vkd3d_device_vkd3d_ext.idl).
struct ID3D12DXVKInteropDevice : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetDXGIAdapter(REFIID iid, void** object)                                     = 0;
    virtual HRESULT STDMETHODCALLTYPE GetInstanceExtensions(UINT* count, const char** extensions)                   = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceExtensions(UINT* count, const char** extensions)                     = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceFeatures(const VkPhysicalDeviceFeatures2** features)                 = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHandles(VkInstance* instance, VkPhysicalDevice* pd, VkDevice* dev)   = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanQueueInfo(ID3D12CommandQueue* queue, VkQueue* vk, UINT32* family)    = 0;
    virtual void STDMETHODCALLTYPE    GetVulkanImageLayout(ID3D12Resource* r, D3D12_RESOURCE_STATES s, VkImageLayout* l) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo(ID3D12Resource* r, UINT64* handle, UINT64* offset)      = 0;
    virtual HRESULT STDMETHODCALLTYPE LockCommandQueue(ID3D12CommandQueue* queue)                                   = 0;
    virtual HRESULT STDMETHODCALLTYPE UnlockCommandQueue(ID3D12CommandQueue* queue)                                 = 0;
};

struct ID3D12DXVKInteropDevice1 : public ID3D12DXVKInteropDevice
{
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo1(ID3D12Resource* r, UINT64* handle, UINT64* offset, VkFormat* format) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, UINT32 family, ID3D12CommandQueue** queue) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandAllocator(D3D12_COMMAND_LIST_TYPE type, UINT32 family, ID3D12CommandAllocator** alloc) = 0;
    virtual HRESULT STDMETHODCALLTYPE BeginVkCommandBufferInterop(ID3D12CommandList* list, VkCommandBuffer* cmd) = 0;
    virtual HRESULT STDMETHODCALLTYPE EndVkCommandBufferInterop(ID3D12CommandList* list)                       = 0;
};

static const GUID IID_ID3D12DXVKInteropDevice1 = {0x902d8115, 0x59eb, 0x4406, {0x95, 0x18, 0xfe, 0x00, 0xf9, 0x91, 0xee, 0x65}};

namespace
{
std::mutex                g_mutex;
ID3D12DXVKInteropDevice1* g_interop = nullptr;
VulkanDevice              g_vk;
ID3D12Device*             g_device = nullptr;

// Carrier for the D3D12-native FSR 4 path: an Upscaler (so it lives in the registry and switches like the others) whose
// work runs on the game's D3D12 device, not the Vulkan one. Its Vulkan Dispatch is never called.
class Dx12Carrier : public Upscaler
{
public:
    static std::unique_ptr<Dx12Carrier> Create(const VulkanDevice& vk, const UpscalerCreateInfo& info, ID3D12Device* device,
                                               uint32_t outW, uint32_t outH)
    {
        Fsr4Dx12* f = Fsr4Dx12::Create(device, info.renderWidth, info.renderHeight, info.displayWidth ? info.displayWidth : outW,
                                       info.displayHeight ? info.displayHeight : outH, info.hdr, info.depthInverted,
                                       info.highResMv, info.autoExposure, info.fsr4Model);
        if (!f)
            return nullptr;
        std::unique_ptr<Dx12Carrier> c(new Dx12Carrier());
        c->m_dev        = vk;
        c->m_info       = info;
        c->m_info.fsr4Active = f->Model();
        c->m_fsr4.reset(f);
        return c;
    }
    bool       Dispatch(VkCommandBuffer, const UpscalerDispatch&) override { return false; }
    Fsr4Dx12*  AsDx12() override { return m_fsr4.get(); }

private:
    std::unique_ptr<Fsr4Dx12> m_fsr4;
};

// D3D12 games run FSR 4 natively on their own device (amdxcffx64); once that has failed, NSS.
UpscalerCreateInfo RemapD3D12(const UpscalerCreateInfo& in)
{
    UpscalerCreateInfo info = in;
    if (info.backend == kBackendFsr4)
    {
        if (Fsr4Dx12Unavailable(info.fsr4Model))
        {
            info.backend   = kBackendNss;
            info.fsr4Model = kFsr4Auto;
        }
        else
            info.backend = kBackendFsr4Dx12;
    }
    return info;
}

// The upscaler for a D3D12 create: the native FSR 4 path when asked and available, else the Vulkan-device path. The
// display size may only be known from the output resource at evaluate time, so pass a fallback here.
std::unique_ptr<Upscaler> CreateD3D12Upscaler(const UpscalerCreateInfo& info, uint32_t outW, uint32_t outH)
{
    if (info.backend == kBackendFsr4Dx12 && g_device)
    {
        if (auto c = Dx12Carrier::Create(g_vk, info, g_device, outW, outH))
            return c;
        LOGI("FSR 4 DX12 (%s) unavailable; falling back to NSS", Fsr4ModelName(info.fsr4Model));
        UpscalerCreateInfo nss = info;
        nss.backend            = kBackendNss;
        nss.fsr4Model          = kFsr4Auto;
        return CreateUpscaler(g_vk, nss);
    }
    return CreateUpscaler(g_vk, info);
}

// ML device features vkd3d-proton enabled, read from its feature chain.
uint32_t MlFeaturesFromChain(const VkPhysicalDeviceFeatures2* features)
{
    uint32_t mask = 0;
    for (auto* s = reinterpret_cast<const VkBaseInStructure*>(features); s; s = s->pNext)
    {
        switch (s->sType)
        {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDeviceVulkan11Features*>(s);
            if (f->storageBuffer16BitAccess) mask |= FFX_API_VK_ML_FEATURE_STORAGE_BUFFER_16BIT;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(s);
            if (f->shaderInt8) mask |= FFX_API_VK_ML_FEATURE_SHADER_INT8;
            if (f->storageBuffer8BitAccess) mask |= FFX_API_VK_ML_FEATURE_STORAGE_BUFFER_8BIT;
            if (f->shaderFloat16) mask |= FFX_API_VK_ML_FEATURE_SHADER_FLOAT16;
            if (f->vulkanMemoryModel) mask |= FFX_API_VK_ML_FEATURE_VULKAN_MEMORY_MODEL;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(s);
            if (f->shaderIntegerDotProduct) mask |= FFX_API_VK_ML_FEATURE_INTEGER_DOT_PRODUCT;
            if (f->subgroupSizeControl && f->computeFullSubgroups) mask |= FFX_API_VK_ML_FEATURE_SUBGROUP_SIZE_CONTROL;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDeviceShaderFloat16Int8Features*>(s);
            if (f->shaderInt8) mask |= FFX_API_VK_ML_FEATURE_SHADER_INT8;
            if (f->shaderFloat16) mask |= FFX_API_VK_ML_FEATURE_SHADER_FLOAT16;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDevice8BitStorageFeatures*>(s);
            if (f->storageBuffer8BitAccess) mask |= FFX_API_VK_ML_FEATURE_STORAGE_BUFFER_8BIT;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDevice16BitStorageFeatures*>(s);
            if (f->storageBuffer16BitAccess) mask |= FFX_API_VK_ML_FEATURE_STORAGE_BUFFER_16BIT;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDeviceShaderIntegerDotProductFeatures*>(s);
            if (f->shaderIntegerDotProduct) mask |= FFX_API_VK_ML_FEATURE_INTEGER_DOT_PRODUCT;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDeviceSubgroupSizeControlFeatures*>(s);
            if (f->subgroupSizeControl && f->computeFullSubgroups) mask |= FFX_API_VK_ML_FEATURE_SUBGROUP_SIZE_CONTROL;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES: {
            auto* f = reinterpret_cast<const VkPhysicalDeviceVulkanMemoryModelFeatures*>(s);
            if (f->vulkanMemoryModel) mask |= FFX_API_VK_ML_FEATURE_VULKAN_MEMORY_MODEL;
            break;
        }
        default:
            break;
        }
    }
    return mask;
}

NVSDK_NGX_Result InitDevice(ID3D12Device* device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!device)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    CostSetApi(CostApi::D3D12);

    ID3D12DXVKInteropDevice1* interop = nullptr;
    if (FAILED(device->QueryInterface(IID_ID3D12DXVKInteropDevice1, reinterpret_cast<void**>(&interop))) || !interop)
    {
        LOGI("D3D12 device is not vkd3d-proton (no ID3D12DXVKInteropDevice1); this proxy only runs under Proton");
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
    if (g_interop)
        g_interop->Release();
    g_interop = interop;
    g_device  = device;

    VulkanDevice vk;
    interop->GetVulkanHandles(&vk.instance, &vk.physicalDevice, &vk.device);
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    if (!vulkan)
        vulkan = LoadLibraryA("vulkan-1.dll");
    vk.gipa = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")) : nullptr;
    if (!vk.gipa)
    {
        LOGI("vulkan-1.dll not found");
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
    vk.gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk.gipa(vk.instance, "vkGetDeviceProcAddr"));

    const VkPhysicalDeviceFeatures2* features = nullptr;
    interop->GetDeviceFeatures(&features);
    vk.mlFeatures = MlFeaturesFromChain(features);
    if (Config().mlFeaturesForced)
        vk.mlFeatures |= ffxQueryMlDeviceFeaturesVK(vk.instance, vk.physicalDevice, vk.gipa, nullptr, nullptr);
    g_vk = vk;

    const LUID luid = device->GetAdapterLuid();
    LOGI("D3D12 init on vkd3d-proton: VkDevice %p, adapter LUID %08lx:%08lx, ML features 0x%x",
         reinterpret_cast<void*>(vk.device), luid.HighPart, luid.LowPart, vk.mlFeatures);
    return NVSDK_NGX_Result_Success;
}

VkFormat VkFormatFromDxgi(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case DXGI_FORMAT_R11G11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
    case DXGI_FORMAT_D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_UNORM: return VK_FORMAT_R16_UNORM;
    case DXGI_FORMAT_D16_UNORM: return VK_FORMAT_D16_UNORM;
    default: return VK_FORMAT_UNDEFINED;
    }
}

bool Describe(ID3D12Resource* resource, D3D12_RESOURCE_STATES state, UpscalerImage& out)
{
    if (!resource)
        return false;
    UINT64   handle = 0, offset = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (FAILED(g_interop->GetVulkanResourceInfo1(resource, &handle, &offset, &format)) || !handle)
        return false;
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    // vkd3d-proton reports the image's own format; typeless images get a mutable representative format, which is
    // only usable for views when it has a defined numeric type, so fall back to the DXGI description otherwise.
    if (format == VK_FORMAT_UNDEFINED || ffxApiGetSurfaceFormatVK(format) == FFX_API_SURFACE_FORMAT_UNKNOWN)
    {
        const VkFormat fromDxgi = VkFormatFromDxgi(desc.Format);
        if (fromDxgi != VK_FORMAT_UNDEFINED && !(format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                                                  format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM))
            format = fromDxgi;
    }
    out.image  = reinterpret_cast<VkImage>(handle);
    out.format = format;
    out.width  = static_cast<uint32_t>(desc.Width);
    out.height = desc.Height;
    g_interop->GetVulkanImageLayout(resource, state, &out.layout);
    return true;
}
} // namespace

// ---------------------------------------------------------------------------------------------------------------------

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_Init_Ext(unsigned long long, const wchar_t*, ID3D12Device* device, int,
                                                     const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_Init_Ext");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_Init(unsigned long long, const wchar_t*, ID3D12Device* device,
                                                 const NVSDK_NGX_FeatureCommonInfo*, int)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_Init");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_Init_ProjectID(const char*, int, const char*, const wchar_t*,
                                                           ID3D12Device* device, int, const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_Init_ProjectID");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_Init_with_ProjectID(const char*, int, const char*, const wchar_t*,
                                                                ID3D12Device* device, const NVSDK_NGX_FeatureCommonInfo*, int)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_Init_with_ProjectID");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_Shutdown()
{
    Log(3, "%s", "NVSDK_NGX_D3D12_Shutdown");
    // NGX shutdown comes before the device is destroyed and after the game has idled it: free everything now.
    Registry().CollectGarbage(true);
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_Shutdown1(ID3D12Device*)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_Shutdown1");
    return NVSDK_NGX_D3D12_Shutdown();
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_GetParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_D3D12_GetParameters"); return NgxGetParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_D3D12_GetCapabilityParameters"); return NgxAllocateParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_D3D12_AllocateParameters"); return NgxAllocateParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* p) { Log(3, "%s", "NVSDK_NGX_D3D12_DestroyParameters"); return NgxDestroyParameters(p); }

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_PopulateParameters_Impl(NVSDK_NGX_Parameter* p)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_PopulateParameters_Impl");
    if (!p)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    NgxParams::From(p)->FillCapabilities();
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_GetScratchBufferSize(NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, size_t* size)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_GetScratchBufferSize");
    if (!size)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *size = 0;
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_GetFeatureRequirements(IUnknown*, const NVSDK_NGX_FeatureDiscoveryInfo* info,
                                                                   NVSDK_NGX_FeatureRequirement* out)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_GetFeatureRequirements");
    return NgxFeatureRequirement(info ? info->FeatureID : NVSDK_NGX_Feature(0), out);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature feature,
                                                          NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_CreateFeature");
    if (feature != NVSDK_NGX_Feature_SuperSampling)
    {
        LOGI("CreateFeature: feature %d is not provided", feature);
        return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    }
    if (!params || !outHandle)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    if (!g_interop)
        return NVSDK_NGX_Result_FAIL_NotInitialized;

    Registry().CollectGarbage();
    const UpscalerCreateInfo info = RemapD3D12(NgxCreateInfo(*NgxParams::From(params)));
    auto upscaler = CreateD3D12Upscaler(info, info.displayWidth, info.displayHeight);
    if (!upscaler)
        return NVSDK_NGX_Result_FAIL_PlatformError;
    *outHandle = NgxNewHandle(Registry().Add(std::move(upscaler)));
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle* handle)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_ReleaseFeature");
    if (!handle)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    Registry().Release(handle->Id);
    NgxFreeHandle(handle);
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* cmdList, const NVSDK_NGX_Handle* handle,
                                                            NVSDK_NGX_Parameter* params, PFN_NVSDK_NGX_ProgressCallback)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_EvaluateFeature");
    if (!cmdList || !handle || !params)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    Upscaler* upscaler = Registry().Get(handle->Id);
    if (upscaler)
    {
        auto create = [](const UpscalerCreateInfo& info) { return CreateD3D12Upscaler(info, info.displayWidth, info.displayHeight); };
        upscaler    = NgxApplyLiveSettings(handle->Id, upscaler, create, RemapD3D12);
    }
    if (!upscaler || !g_interop)
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;

    const NgxParams& p = *NgxParams::From(params);

    // D3D12-native FSR 4: hand amdxcffx64 the game's own ID3D12 resources and command list, no Vulkan interop.
    if (Fsr4Dx12* fsr4 = upscaler->AsDx12())
    {
        auto* color  = static_cast<ID3D12Resource*>(p.Ptr(ngxp::Color));
        auto* depth  = static_cast<ID3D12Resource*>(p.Ptr(ngxp::Depth));
        auto* motion = static_cast<ID3D12Resource*>(p.Ptr(ngxp::MotionVectors));
        auto* output = static_cast<ID3D12Resource*>(p.Ptr(ngxp::Output));
        if (!color || !depth || !motion || !output)
        {
            LOGI("EvaluateFeature: missing input resource");
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        }
        UpscalerDispatch d;  // reuse the shared scalar fill (jitter, mv scale, render subrect, exposure, reset)
        NgxFillDispatch(p, upscaler->Info(), d);
        const D3D12_RESOURCE_DESC cd = color->GetDesc(), od = output->GetDesc();
        Fsr4Dx12Dispatch fd;
        fd.color = color, fd.depth = depth, fd.motion = motion, fd.output = output;
        fd.renderWidth  = std::min<uint32_t>(d.renderWidth, uint32_t(cd.Width));
        fd.renderHeight = std::min<uint32_t>(d.renderHeight, cd.Height);
        fd.outputWidth  = uint32_t(od.Width);
        fd.outputHeight = od.Height;
        fd.jitterX = d.jitterX, fd.jitterY = d.jitterY;
        fd.mvScaleX = d.mvScaleX, fd.mvScaleY = d.mvScaleY;
        fd.frameTimeMs = d.frameTimeMs;
        fd.exposure    = d.exposure;
        fd.reset       = d.reset;
        return fsr4->Dispatch(cmdList, fd) ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
    }

    UpscalerDispatch d;
    constexpr D3D12_RESOURCE_STATES kInput = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (!Describe(static_cast<ID3D12Resource*>(p.Ptr(ngxp::Color)), kInput, d.color) ||
        !Describe(static_cast<ID3D12Resource*>(p.Ptr(ngxp::Depth)), kInput, d.depth) ||
        !Describe(static_cast<ID3D12Resource*>(p.Ptr(ngxp::MotionVectors)), kInput, d.motionVectors) ||
        !Describe(static_cast<ID3D12Resource*>(p.Ptr(ngxp::Output)), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, d.output))
    {
        LOGI("EvaluateFeature: missing or foreign input resource");
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NgxFillDispatch(p, upscaler->Info(), d);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_interop->BeginVkCommandBufferInterop(cmdList, &cmd)) || !cmd)
        return NVSDK_NGX_Result_FAIL_PlatformError;
    const bool ok = NgxDispatch(upscaler, cmd, d);
    g_interop->EndVkCommandBufferInterop(cmdList);
    return ok ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList* cmdList, const NVSDK_NGX_Handle* handle,
                                                              const NVSDK_NGX_Parameter* params, void*)
{
    Log(3, "%s", "NVSDK_NGX_D3D12_EvaluateFeature_C");
    return NVSDK_NGX_D3D12_EvaluateFeature(cmdList, handle, const_cast<NVSDK_NGX_Parameter*>(params), nullptr);
}
