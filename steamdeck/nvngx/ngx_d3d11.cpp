// D3D11 NGX entry points. On Proton, D3D11 is DXVK. Its Vulkan interop interface gives the VkDevice, the VkImage and
// layout behind each texture, and its submission queue; NSS records into the proxy's own command buffer, which is
// submitted on DXVK's queue right after DXVK has flushed the game's work, so queue order keeps everything in sequence.
// SPDX-License-Identifier: MIT
#include "ngx_common.h"

#include "common.h"

#include <d3d11.h>
#include <windows.h>

#include <ffx_api/vk/ffx_api_vk.h>

#include <mutex>

// DXVK interop interfaces (src/dxgi/dxgi_interfaces.h).
struct IDXGIVkInteropDevice;

struct IDXGIVkInteropSurface : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetDevice(IDXGIVkInteropDevice** device)                                  = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanImageInfo(VkImage* image, VkImageLayout* layout, VkImageCreateInfo* info) = 0;
};

struct IDXGIVkInteropDevice : public IUnknown
{
    virtual void STDMETHODCALLTYPE GetVulkanHandles(VkInstance* instance, VkPhysicalDevice* pd, VkDevice* device) = 0;
    virtual void STDMETHODCALLTYPE GetSubmissionQueue(VkQueue* queue, uint32_t* family)                         = 0;
    virtual void STDMETHODCALLTYPE TransitionSurfaceLayout(IDXGIVkInteropSurface* surface, const VkImageSubresourceRange* range,
                                                           VkImageLayout oldLayout, VkImageLayout newLayout)   = 0;
    virtual void STDMETHODCALLTYPE FlushRenderingCommands()                                                     = 0;
    virtual void STDMETHODCALLTYPE LockSubmissionQueue()                                                        = 0;
    virtual void STDMETHODCALLTYPE ReleaseSubmissionQueue()                                                     = 0;
};

static const GUID IID_IDXGIVkInteropDevice  = {0xe2ef5fa5, 0xdc21, 0x4af7, {0x90, 0xc4, 0xf6, 0x7e, 0xf6, 0xa0, 0x93, 0x23}};
static const GUID IID_IDXGIVkInteropSurface = {0x5546cf8c, 0x77e7, 0x4341, {0xb0, 0x5d, 0x8d, 0x4d, 0x50, 0x00, 0xe7, 0x7d}};

namespace
{
// Command buffers the proxy submits on DXVK's queue, recycled behind fences.
class Submitter
{
public:
    bool Init(const VulkanDevice& vk, VkQueue queue, uint32_t family)
    {
        m_device = vk.device;
        m_queue  = queue;
#define VK_FN(name) m_##name = reinterpret_cast<PFN_##name>(vk.gdpa(vk.device, #name));
        VK_FN(vkCreateCommandPool)
        VK_FN(vkAllocateCommandBuffers)
        VK_FN(vkBeginCommandBuffer)
        VK_FN(vkEndCommandBuffer)
        VK_FN(vkResetCommandBuffer)
        VK_FN(vkCreateFence)
        VK_FN(vkWaitForFences)
        VK_FN(vkResetFences)
        VK_FN(vkQueueSubmit)
#undef VK_FN
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex        = family;
        if (m_vkCreateCommandPool(m_device, &pci, nullptr, &m_pool) != VK_SUCCESS)
            return false;
        VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool                 = m_pool;
        cai.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount          = kRing;
        if (m_vkAllocateCommandBuffers(m_device, &cai, m_cmd) != VK_SUCCESS)
            return false;
        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags             = VK_FENCE_CREATE_SIGNALED_BIT;
        for (auto& f : m_fence)
            if (m_vkCreateFence(m_device, &fci, nullptr, &f) != VK_SUCCESS)
                return false;
        return true;
    }

    VkCommandBuffer Begin()
    {
        m_slot = (m_slot + 1) % kRing;
        m_vkWaitForFences(m_device, 1, &m_fence[m_slot], VK_TRUE, UINT64_MAX);
        m_vkResetFences(m_device, 1, &m_fence[m_slot]);
        m_vkResetCommandBuffer(m_cmd[m_slot], 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        m_vkBeginCommandBuffer(m_cmd[m_slot], &bi);
        return m_cmd[m_slot];
    }

    // Caller holds DXVK's submission queue lock. Optionally waits for / signals a timeline semaphore value.
    VkResult Submit(VkSemaphore wait = VK_NULL_HANDLE, uint64_t waitValue = 0, VkSemaphore signal = VK_NULL_HANDLE,
                    uint64_t signalValue = 0)
    {
        m_vkEndCommandBuffer(m_cmd[m_slot]);
        VkTimelineSemaphoreSubmitInfo ts   = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        VkSubmitInfo                  si   = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        const VkPipelineStageFlags    stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        si.commandBufferCount              = 1;
        si.pCommandBuffers                 = &m_cmd[m_slot];
        if (wait || signal)
        {
            si.pNext = &ts;
            if (wait)
            {
                si.waitSemaphoreCount        = 1;
                si.pWaitSemaphores           = &wait;
                si.pWaitDstStageMask         = &stage;
                ts.waitSemaphoreValueCount   = 1;
                ts.pWaitSemaphoreValues      = &waitValue;
            }
            if (signal)
            {
                si.signalSemaphoreCount      = 1;
                si.pSignalSemaphores         = &signal;
                ts.signalSemaphoreValueCount = 1;
                ts.pSignalSemaphoreValues    = &signalValue;
            }
        }
        return m_vkQueueSubmit(m_queue, 1, &si, m_fence[m_slot]);
    }

private:
    static constexpr uint32_t kRing = 4;
    VkDevice        m_device = VK_NULL_HANDLE;
    VkQueue         m_queue  = VK_NULL_HANDLE;
    VkCommandPool   m_pool   = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd[kRing]   = {};
    VkFence         m_fence[kRing] = {};
    uint32_t        m_slot         = 0;
    PFN_vkCreateCommandPool      m_vkCreateCommandPool      = nullptr;
    PFN_vkAllocateCommandBuffers m_vkAllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer     m_vkBeginCommandBuffer     = nullptr;
    PFN_vkEndCommandBuffer       m_vkEndCommandBuffer       = nullptr;
    PFN_vkResetCommandBuffer     m_vkResetCommandBuffer     = nullptr;
    PFN_vkCreateFence            m_vkCreateFence            = nullptr;
    PFN_vkWaitForFences          m_vkWaitForFences          = nullptr;
    PFN_vkResetFences            m_vkResetFences            = nullptr;
    PFN_vkQueueSubmit            m_vkQueueSubmit            = nullptr;
};

std::mutex            g_mutex;
IDXGIVkInteropDevice* g_interop = nullptr;
VulkanDevice          g_vk;
Submitter*            g_submitter = nullptr;

NVSDK_NGX_Result InitDevice(ID3D11Device* device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!device)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;

    IDXGIVkInteropDevice* interop = nullptr;
    if (FAILED(device->QueryInterface(IID_IDXGIVkInteropDevice, reinterpret_cast<void**>(&interop))) || !interop)
    {
        LOGI("D3D11 device is not DXVK (no IDXGIVkInteropDevice); this proxy only runs under Proton");
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
    if (g_interop)
        g_interop->Release();
    g_interop = interop;

    VulkanDevice vk;
    interop->GetVulkanHandles(&vk.instance, &vk.physicalDevice, &vk.device);
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    if (!vulkan)
        vulkan = LoadLibraryA("vulkan-1.dll");
    vk.gipa = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")) : nullptr;
    if (!vk.gipa)
        return NVSDK_NGX_Result_FAIL_PlatformError;
    vk.gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk.gipa(vk.instance, "vkGetDeviceProcAddr"));
    // DXVK does not report its enabled features; the aiss Vulkan layer (AISS_LAYER=1) guarantees the ML ones.
    vk.mlFeatures = Config().mlFeaturesForced ? ffxQueryMlDeviceFeaturesVK(vk.instance, vk.physicalDevice, vk.gipa, nullptr, nullptr) : 0;

    VkQueue  queue  = VK_NULL_HANDLE;
    uint32_t family = 0;
    interop->GetSubmissionQueue(&queue, &family);
    auto* submitter = new Submitter();
    if (!submitter->Init(vk, queue, family))
    {
        delete submitter;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
    g_submitter = submitter;  // a previous one (device re-init) is leaked deliberately: its fences may still be pending
    g_vk        = vk;
    LOGI("D3D11 init on DXVK: VkDevice %p, ML features 0x%x%s", reinterpret_cast<void*>(vk.device), vk.mlFeatures,
         Config().mlFeaturesForced ? "" : " (set AISS_LAYER=1 for the fast ML path)");
    return NVSDK_NGX_Result_Success;
}

bool Describe(ID3D11Resource* resource, UpscalerImage& out)
{
    if (!resource)
        return false;
    IDXGIVkInteropSurface* surface = nullptr;
    if (FAILED(resource->QueryInterface(IID_IDXGIVkInteropSurface, reinterpret_cast<void**>(&surface))) || !surface)
        return false;
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    VkImage           image = VK_NULL_HANDLE;
    VkImageLayout     layout = VK_IMAGE_LAYOUT_GENERAL;
    const HRESULT     hr = surface->GetVulkanImageInfo(&image, &layout, &info);
    surface->Release();
    if (FAILED(hr) || !image)
        return false;
    out.image  = image;
    out.format = info.format;
    out.width  = info.extent.width;
    out.height = info.extent.height;
    out.layout = layout;
    return true;
}
} // namespace

// ---------------------------------------------------------------------------------------------------------------------

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init(unsigned long long, const wchar_t*, ID3D11Device* device,
                                                 const NVSDK_NGX_FeatureCommonInfo*, int)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init_Ext(unsigned long long, const wchar_t*, ID3D11Device* device, int,
                                                     const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init_Ext");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init_ProjectID(const char*, int, const char*, const wchar_t*, ID3D11Device* device,
                                                           int, const NVSDK_NGX_FeatureCommonInfo*)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init_ProjectID");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init_with_ProjectID(const char*, int, const char*, const wchar_t*,
                                                                ID3D11Device* device, const NVSDK_NGX_FeatureCommonInfo*, int)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init_with_ProjectID");
    return InitDevice(device);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Shutdown()
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Shutdown");
    // The proxy's own submissions must retire before their resources go away.
    if (g_interop)
        g_interop->FlushRenderingCommands();
    Registry().CollectGarbage(true);
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Shutdown1(ID3D11Device*)
{
    return NVSDK_NGX_D3D11_Shutdown();
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_GetParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_D3D11_GetParameters"); return NgxGetParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_GetCapabilityParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_D3D11_GetCapabilityParameters"); return NgxAllocateParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_AllocateParameters(NVSDK_NGX_Parameter** out) { Log(3, "%s", "NVSDK_NGX_D3D11_AllocateParameters"); return NgxAllocateParameters(out); }
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_DestroyParameters(NVSDK_NGX_Parameter* p) { Log(3, "%s", "NVSDK_NGX_D3D11_DestroyParameters"); return NgxDestroyParameters(p); }

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_PopulateParameters_Impl(NVSDK_NGX_Parameter* p)
{
    if (!p)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    NgxParams::From(p)->FillCapabilities();
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_GetScratchBufferSize(NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, size_t* size)
{
    if (!size)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    *size = 0;
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_GetFeatureRequirements(IUnknown*, const NVSDK_NGX_FeatureDiscoveryInfo* info,
                                                                   NVSDK_NGX_FeatureRequirement* out)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_GetFeatureRequirements");
    return NgxFeatureRequirement(info ? info->FeatureID : NVSDK_NGX_Feature(0), out);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_CreateFeature(ID3D11DeviceContext*, NVSDK_NGX_Feature feature,
                                                          NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_CreateFeature");
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
    auto upscaler = CreateUpscaler(g_vk, NgxRemapInterop(NgxCreateInfo(*NgxParams::From(params))));
    if (!upscaler)
        return NVSDK_NGX_Result_FAIL_PlatformError;
    *outHandle = NgxNewHandle(Registry().Add(std::move(upscaler)));
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_ReleaseFeature(NVSDK_NGX_Handle* handle)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_ReleaseFeature");
    if (!handle)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    Registry().Release(handle->Id);
    NgxFreeHandle(handle);
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_EvaluateFeature(ID3D11DeviceContext* context, const NVSDK_NGX_Handle* handle,
                                                            NVSDK_NGX_Parameter* params, PFN_NVSDK_NGX_ProgressCallback)
{
    if (!context || !handle || !params)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    Upscaler* upscaler = Registry().Get(handle->Id);
    if (upscaler)
        upscaler = NgxApplyLiveSettings(handle->Id, upscaler, {}, NgxRemapInterop);
    if (!upscaler || !g_interop || !g_submitter)
        return NVSDK_NGX_Result_FAIL_FeatureNotFound;

    const NgxParams& p = *NgxParams::From(params);
    UpscalerDispatch d;
    if (!Describe(static_cast<ID3D11Resource*>(p.Ptr(ngxp::Color)), d.color) ||
        !Describe(static_cast<ID3D11Resource*>(p.Ptr(ngxp::Depth)), d.depth) ||
        !Describe(static_cast<ID3D11Resource*>(p.Ptr(ngxp::MotionVectors)), d.motionVectors) ||
        !Describe(static_cast<ID3D11Resource*>(p.Ptr(ngxp::Output)), d.output))
    {
        LOGI("EvaluateFeature: missing or foreign input resource");
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NgxFillDispatch(p, upscaler->Info(), d);

    // Everything the game recorded so far reaches the queue first; the upscale follows on the same queue.
    g_interop->FlushRenderingCommands();
    if (upscaler->SupportsSplit())
    {
        // FSR 4 via D3D12 interop: copy-in submission signals a shared timeline value, the D3D12 queue upscales, the
        // copy-out submission waits for it - all on the GPU.
        struct Split : SplitSubmitter
        {
            VkCommandBuffer Begin() override { return g_submitter->Begin(); }
            bool Submit(VkSemaphore wait, uint64_t wv, VkSemaphore signal, uint64_t sv) override
            {
                g_interop->LockSubmissionQueue();
                const VkResult r = g_submitter->Submit(wait, wv, signal, sv);
                g_interop->ReleaseSubmissionQueue();
                if (r != VK_SUCCESS)
                    LOGI("EvaluateFeature: vkQueueSubmit failed (%d)", r);
                return r == VK_SUCCESS;
            }
        } split;
        return upscaler->DispatchSplit(split, d) ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
    }
    VkCommandBuffer cmd = g_submitter->Begin();
    const bool      ok  = NgxDispatch(upscaler, cmd, d);
    g_interop->LockSubmissionQueue();
    const VkResult res = g_submitter->Submit();
    g_interop->ReleaseSubmissionQueue();
    if (res != VK_SUCCESS)
        LOGI("EvaluateFeature: vkQueueSubmit failed (%d)", res);
    return ok && res == VK_SUCCESS ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_EvaluateFeature_C(ID3D11DeviceContext* context, const NVSDK_NGX_Handle* handle,
                                                              const NVSDK_NGX_Parameter* params, void*)
{
    return NVSDK_NGX_D3D11_EvaluateFeature(context, handle, const_cast<NVSDK_NGX_Parameter*>(params), nullptr);
}
