// FFX forwarder: stands in for a game's amd_fidelityfx_vk.dll (the plugin keeps AMD's original next to it as
// amd_fidelityfx_vk.deckyupscaler-orig.dll). Upscale contexts on Vulkan go to the NSS proxy (system32\_nvngx.dll,
// nvngx/ffx_vulkan.cpp), so the game's FSR option runs the upscaler picked in the plugin; every other context (frame
// generation, other effects) and every call without a context goes to AMD's DLL unchanged. If the proxy is missing or
// refuses a context, the game gets AMD's upscaler as before. DU_FFX=0 forwards everything.
// SPDX-License-Identifier: MIT
#include <windows.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <set>

#include "../nvngx/amd_ffx/ffx_api.h"
#include "../nvngx/amd_ffx/ffx_upscale.h"

namespace
{
constexpr uint64_t kBackendVkDeprecated = 0x00000003u;  // FFX API 1.0: header, VkDevice, VkPhysicalDevice, gdpa
constexpr uint64_t kBackendVk           = 0x01000003u;  // FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, same layout

// Vulkan handles only pass through here, so they are plain pointers (no Vulkan headers needed).
struct BackendVk
{
    ffxCreateContextDescHeader header;
    void*                      device;          // VkDevice
    void*                      physicalDevice;  // VkPhysicalDevice
    void*                      gdpa;            // PFN_vkGetDeviceProcAddr
};

using PFN_Create    = uint32_t(__cdecl*)(const ffxCreateContextDescUpscale*, void* device, void* physicalDevice, void* gdpa);
using PFN_Dispatch  = uint32_t(__cdecl*)(uint32_t, const ffxDispatchDescUpscale*);
using PFN_Destroy   = void(__cdecl*)(uint32_t);

struct Apis
{
    PfnFfxCreateContext  create    = nullptr;
    PfnFfxDestroyContext destroy   = nullptr;
    PfnFfxConfigure      configure = nullptr;
    PfnFfxQuery          query     = nullptr;
    PfnFfxDispatch       dispatch  = nullptr;
    PFN_Create           proxyCreate   = nullptr;
    PFN_Dispatch         proxyDispatch = nullptr;
    PFN_Destroy          proxyDestroy  = nullptr;
};

extern "C" IMAGE_DOS_HEADER __ImageBase;

const Apis& Load()
{
    static const Apis apis = [] {
        Apis a;
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), path, MAX_PATH);
        wchar_t* slash = wcsrchr(path, L'\\');
        if (slash)
            wcscpy(slash + 1, L"amd_fidelityfx_vk.deckyupscaler-orig.dll");
        if (HMODULE amd = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH))
        {
            a.create    = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(amd, "ffxCreateContext"));
            a.destroy   = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(amd, "ffxDestroyContext"));
            a.configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(amd, "ffxConfigure"));
            a.query     = reinterpret_cast<PfnFfxQuery>(GetProcAddress(amd, "ffxQuery"));
            a.dispatch  = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(amd, "ffxDispatch"));
        }
        wchar_t off[8];
        if (GetEnvironmentVariableW(L"DU_FFX", off, 8) && !wcscmp(off, L"0"))
            return a;
        GetSystemDirectoryW(path, MAX_PATH);
        wcscat(path, L"\\_nvngx.dll");
        if (HMODULE proxy = LoadLibraryW(path))
        {
            a.proxyCreate   = reinterpret_cast<PFN_Create>(GetProcAddress(proxy, "DU_FfxCreateUpscaleVK"));
            a.proxyDispatch = reinterpret_cast<PFN_Dispatch>(GetProcAddress(proxy, "DU_FfxDispatchUpscaleVK"));
            a.proxyDestroy  = reinterpret_cast<PFN_Destroy>(GetProcAddress(proxy, "DU_FfxDestroyUpscale"));
            if (!a.proxyCreate || !a.proxyDispatch || !a.proxyDestroy)
                a.proxyCreate = nullptr;
        }
        return a;
    }();
    return apis;
}

// The proxy's contexts: a handle the forwarder owns, told apart from AMD's by address.
struct ProxyContext
{
    uint32_t id;
    uint32_t renderWidth, displayWidth;
};
std::mutex              g_mutex;
std::set<ProxyContext*> g_ours;

ProxyContext* Ours(ffxContext* context)
{
    if (!context || !*context)
        return nullptr;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_ours.find(static_cast<ProxyContext*>(*context));
    return it == g_ours.end() ? nullptr : *it;
}

// FSR's upscale queries (FidelityFX SDK documentation): quality-mode ratios, Halton(2,3) jitter.
float RatioFor(uint32_t mode)
{
    switch (mode)
    {
    case FFX_UPSCALE_QUALITY_MODE_NATIVEAA: return 1.0f;
    case FFX_UPSCALE_QUALITY_MODE_QUALITY: return 1.5f;
    case FFX_UPSCALE_QUALITY_MODE_BALANCED: return 1.7f;
    case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE: return 2.0f;
    case FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0f;
    default: return 0.0f;
    }
}

float Halton(int32_t index, int32_t base)
{
    float f = 1.0f, result = 0.0f;
    for (int32_t i = index; i > 0;)
    {
        f /= float(base);
        result += f * float(i % base);
        i /= base;
    }
    return result;
}

ffxReturnCode_t QueryOurs(ffxQueryDescHeader* desc)
{
    for (ffxQueryDescHeader* h = desc; h; h = h->pNext)
    {
        switch (h->type)
        {
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE:
        {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(h);
            if (q->pOutUpscaleRatio)
                *q->pOutUpscaleRatio = RatioFor(q->qualityMode);
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE:
        {
            auto*       q     = reinterpret_cast<ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(h);
            const float ratio = RatioFor(q->qualityMode);
            if (ratio <= 0.0f)
                return FFX_API_RETURN_ERROR_PARAMETER;
            if (q->pOutRenderWidth)
                *q->pOutRenderWidth = uint32_t(float(q->displayWidth) / ratio);
            if (q->pOutRenderHeight)
                *q->pOutRenderHeight = uint32_t(float(q->displayHeight) / ratio);
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT:
        {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(h);
            if (q->pOutPhaseCount && q->renderWidth)
                *q->pOutPhaseCount = int32_t(8.0f * std::pow(float(q->displayWidth) / float(q->renderWidth), 2.0f));
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET:
        {
            auto*         q     = reinterpret_cast<ffxQueryDescUpscaleGetJitterOffset*>(h);
            const int32_t phase = q->phaseCount > 0 ? q->phaseCount : 1;
            const int32_t i     = (q->index % phase + phase) % phase + 1;
            if (q->pOutX)
                *q->pOutX = Halton(i, 2) - 0.5f;
            if (q->pOutY)
                *q->pOutY = Halton(i, 3) - 0.5f;
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE:
        {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsage*>(h);
            if (q->gpuMemoryUsageUpscaler)
                *q->gpuMemoryUsageUpscaler = {};
            break;
        }
        default:
            return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
        }
    }
    return FFX_API_RETURN_OK;
}
}  // namespace

// Also the plugin's marker for this DLL (the export name is in the binary).
extern "C" __declspec(dllexport) const char* DU_FfxForwarderVersion()
{
    return "1";
}

extern "C" ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc,
                                            const ffxAllocationCallbacks* memCb)
{
    const Apis& a = Load();
    if (a.proxyCreate && context && desc && desc->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE)
    {
        const BackendVk* backend = nullptr;
        for (const ffxApiHeader* h = desc->pNext; h; h = h->pNext)
            if (h->type == kBackendVk || h->type == kBackendVkDeprecated)
                backend = reinterpret_cast<const BackendVk*>(h);
        if (backend)
        {
            auto*          up = reinterpret_cast<const ffxCreateContextDescUpscale*>(desc);
            const uint32_t id = a.proxyCreate(up, backend->device, backend->physicalDevice, backend->gdpa);
            if (id)
            {
                auto* ours = new ProxyContext{id, up->maxRenderSize.width, up->maxUpscaleSize.width};
                std::lock_guard<std::mutex> lock(g_mutex);
                g_ours.insert(ours);
                *context = ours;
                return FFX_API_RETURN_OK;
            }
        }
    }
    return a.create ? a.create(context, desc, memCb) : FFX_API_RETURN_ERROR;
}

extern "C" ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb)
{
    const Apis& a = Load();
    if (ProxyContext* ours = Ours(context))
    {
        a.proxyDestroy(ours->id);
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_ours.erase(ours);
        }
        delete ours;
        *context = nullptr;
        return FFX_API_RETURN_OK;
    }
    return a.destroy ? a.destroy(context, memCb) : FFX_API_RETURN_ERROR;
}

extern "C" ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc)
{
    if (Ours(context))
        return FFX_API_RETURN_OK;  // key-value tuning of AMD's upscaler: nothing to apply
    const Apis& a = Load();
    return a.configure ? a.configure(context, desc) : FFX_API_RETURN_ERROR;
}

extern "C" ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc)
{
    if (Ours(context))
        return QueryOurs(desc);
    const Apis& a = Load();
    return a.query ? a.query(context, desc) : FFX_API_RETURN_ERROR;
}

extern "C" ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc)
{
    const Apis& a = Load();
    if (ProxyContext* ours = Ours(context))
    {
        if (!desc)
            return FFX_API_RETURN_ERROR_PARAMETER;
        if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE)
            return a.proxyDispatch(ours->id, reinterpret_cast<const ffxDispatchDescUpscale*>(desc));
        if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK)
            return FFX_API_RETURN_OK;  // the proxy's upscalers take no reactive mask
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
    return a.dispatch ? a.dispatch(context, desc) : FFX_API_RETURN_ERROR;
}
