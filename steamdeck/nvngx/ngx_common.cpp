// SPDX-License-Identifier: MIT
#include "ngx_common.h"

#include "common.h"
#include "cost_model.h"

bool Fsr4Dx12Unavailable(int model);  // fsr4_dx12.cpp (its header needs d3d12.h)

#include <cstdio>
#include <cstring>
#include <string>

NVSDK_NGX_Result NgxGetParameters(NVSDK_NGX_Parameter** out)
{
    if (!out)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    static NgxParams* table = [] {
        NgxParams* p  = new NgxParams();
        p->persistent = true;
        p->FillCapabilities();
        return p;
    }();
    *out = table->Abi();
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxAllocateParameters(NVSDK_NGX_Parameter** out)
{
    if (!out)
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    NgxParams* p = new NgxParams();
    p->FillCapabilities();
    *out = p->Abi();
    return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxDestroyParameters(NVSDK_NGX_Parameter* params)
{
    NgxParams* p = NgxParams::From(params);
    if (p && !p->persistent)
        delete p;
    return NVSDK_NGX_Result_Success;
}

UpscalerCreateInfo NgxCreateInfo(const NgxParams& p)
{
    UpscalerCreateInfo info;
    const int flags    = p.I32(ngxp::CreateFlags);
    info.renderWidth   = p.U32(ngxp::Width);
    info.renderHeight  = p.U32(ngxp::Height);
    info.displayWidth  = p.U32(ngxp::OutWidth);
    info.displayHeight = p.U32(ngxp::OutHeight);
    info.hdr           = (flags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0;
    info.depthInverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
    info.autoExposure  = (flags & NVSDK_NGX_DLSS_Feature_Flags_AutoExposure) != 0;
    info.nssQuality    = NssQualityForDlss(p.I32(ngxp::PerfQuality, NVSDK_NGX_PerfQuality_Value_MaxPerf));
    info.fastMath      = FastMathEnabled();
    info.highResMv     = !(flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes);
    info.backend       = SelectedBackend(&info.fsr4Model);
    if (!(flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes))
        LOGI("warning: game passes display-resolution motion vectors (MVLowRes not set); NSS expects render resolution, "
             "XeSS takes them as high-res motion vectors");
    LOGI("DLSS create: %ux%u -> %ux%u, flags 0x%x, perf quality %d", info.renderWidth, info.renderHeight,
         info.displayWidth, info.displayHeight, flags, p.I32(ngxp::PerfQuality));
    return info;
}

void NgxFillDispatch(const NgxParams& p, const UpscalerCreateInfo& info, UpscalerDispatch& d)
{
    d.renderWidth  = p.U32(ngxp::RenderSubrectW);
    d.renderHeight = p.U32(ngxp::RenderSubrectH);
    if (!d.renderWidth || !d.renderHeight)
    {
        d.renderWidth  = info.renderWidth;
        d.renderHeight = info.renderHeight;
    }
    d.depthOffsetX = p.I32(ngxp::DepthBaseX);
    d.depthOffsetY = p.I32(ngxp::DepthBaseY);
    d.jitterX      = p.F32(ngxp::JitterX);
    d.jitterY      = p.F32(ngxp::JitterY);
    d.mvScaleX     = p.F32(ngxp::MVScaleX, 1.0f);
    d.mvScaleY     = p.F32(ngxp::MVScaleY, 1.0f);
    d.reset        = p.I32(ngxp::Reset) != 0;
    d.frameTimeMs  = p.F32(ngxp::FrameTime, 16.6f);
    if (d.frameTimeMs <= 0.0f)
        d.frameTimeMs = 16.6f;
    d.exposure = Config().exposure > 0.0f ? Config().exposure : 1.0f;

    static bool warned = false;
    if (!warned && (p.I32(ngxp::ColorBaseX) || p.I32(ngxp::ColorBaseY) || p.I32(ngxp::MVBaseX) ||
                    p.I32(ngxp::MVBaseY) || p.I32(ngxp::OutputBaseX) || p.I32(ngxp::OutputBaseY)))
    {
        warned = true;
        LOGI("warning: colour/MV/output subrect offsets are not supported and are ignored");
    }
    LOGV("evaluate: render %ux%u jitter %.3f,%.3f mv scale %.3f,%.3f reset %d dt %.2f", d.renderWidth,
         d.renderHeight, d.jitterX, d.jitterY, d.mvScaleX, d.mvScaleY, d.reset, d.frameTimeMs);
    CostNoteEvaluate(info, d.renderWidth, d.renderHeight, info.displayWidth, info.displayHeight);
}

NVSDK_NGX_Result NgxFeatureRequirement(NVSDK_NGX_Feature feature, NVSDK_NGX_FeatureRequirement* out)
{
    const bool supported = feature == NVSDK_NGX_Feature_SuperSampling;
    if (out)
    {
        out->FeatureSupported  = supported ? NVSDK_NGX_FeatureSupportResult_Supported
                                           : NVSDK_NGX_FeatureSupportResult_AdapterUnsupported;
        out->MinHWArchitecture = 0;
        strcpy(out->MinOSVersion, "10.0.10240.16384");
    }
    return supported ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_FAIL_FeatureNotSupported;
}

NVSDK_NGX_Handle* NgxNewHandle(unsigned int id)
{
    return new NVSDK_NGX_Handle{id};
}

void NgxFreeHandle(NVSDK_NGX_Handle* handle)
{
    delete handle;
}

int SelectedBackend(int* fsr4Model)
{
    const int live    = LiveUpscaler();
    int       backend = kBackendNss, model = kFsr4Auto;
    if (live >= 0)
    {
        backend = live == 3 ? kBackendXess : live >= 4 ? kBackendFsr4 : kBackendNss;
        if (backend == kBackendFsr4)
            model = Fsr4ModelForLiveCode(live);
    }
    else if (Config().fsr4Model >= 0)
    {
        backend = kBackendFsr4;
        model   = Config().fsr4Model;
    }
    else if (Config().xess)
        backend = kBackendXess;
    if (backend == kBackendXess && XessUnavailable())
        backend = kBackendNss;
    // Neither AMD's SDK Vulkan DLL nor amdxcffx64 usable (missing / not 4.1.2 / refuses the model): NSS.
    if (backend == kBackendFsr4 && Fsr4VkUnavailable() && Fsr4Dx12Unavailable(model))
    {
        static bool logged = false;
        if (!logged)
            LOGI("FSR 4 (%s) unavailable (amd_fidelityfx_upscaler_vk.dll and amdxcffx64.dll); using NSS",
                 Fsr4ModelName(model));
        logged  = true;
        backend = kBackendNss;
    }
    if (fsr4Model)
        *fsr4Model = backend == kBackendFsr4 ? model : kFsr4Auto;
    return backend;
}

const char* UpscalerName(const UpscalerCreateInfo& i)
{
    static thread_local char text[64];
    if (i.backend == kBackendXess)
        snprintf(text, sizeof(text), "XeSS");
    else if (i.backend == kBackendFsr4 || i.backend == kBackendFsr4Dx12 || i.backend == kBackendFsr4Interop ||
             i.backend == kBackendFsr4Vk)
        snprintf(text, sizeof(text), "FSR 4 %s%s%s%s%s", Fsr4ModelName(i.fsr4Model),
                 i.fsr4Model == kFsr4Auto && i.fsr4Active > 0 ? " (" : "",
                 i.fsr4Model == kFsr4Auto && i.fsr4Active > 0 ? Fsr4ModelName(i.fsr4Active) : "",
                 i.fsr4Model == kFsr4Auto && i.fsr4Active > 0 ? ")" : "",
                 i.backend == kBackendFsr4Dx12      ? " DX12"
                 : i.backend == kBackendFsr4Interop ? " DX12 interop"
                 : i.backend == kBackendFsr4Vk      ? " Vulkan"
                                                    : "");
    else
        snprintf(text, sizeof(text), "NSS quality %d%s", i.nssQuality, i.fastMath ? " fast maths" : "");
    return text;
}

UpscalerCreateInfo NgxRemapInterop(const UpscalerCreateInfo& in)
{
    // D3D11 games run FSR 4 through the D3D12 interop path; once it has failed, NSS.
    UpscalerCreateInfo info = in;
    if (info.backend == kBackendFsr4)
    {
        if (Fsr4InteropUnavailable(info.fsr4Model))
        {
            info.backend   = kBackendNss;
            info.fsr4Model = kFsr4Auto;
        }
        else
            info.backend = kBackendFsr4Interop;
    }
    return info;
}

UpscalerCreateInfo NgxRemapVulkan(const UpscalerCreateInfo& in)
{
    // Vulkan games run FSR 4 natively through AMD's FSR SDK Vulkan DLL (fsr4_vk.cpp). Without it (or when it refuses
    // the device) they use the D3D12 interop path, with the AISS layer splitting the game's command buffer at the
    // evaluate (see fsr4_interop.cpp); once that has failed too, NSS.
    UpscalerCreateInfo info = in.backend == kBackendFsr4 && !Fsr4VkUnavailable() ? in : NgxRemapInterop(in);
    if (info.backend == kBackendFsr4)
        info.backend = kBackendFsr4Vk;
    info.layerSplit = info.backend == kBackendFsr4Interop || info.backend == kBackendFsr4Vk;  // for the interop fallback
    return info;
}

UpscalerCreateInfo NgxRemapD3D11(const UpscalerCreateInfo& in)
{
    // D3D11 games (DXVK) run FSR 4 natively on DXVK's VkDevice like Vulkan games, recording into the proxy's own
    // command buffer; without the SDK DLL, through the D3D12 interop (the front end's own submissions, no split).
    if (in.backend == kBackendFsr4 && !Fsr4VkUnavailable())
    {
        UpscalerCreateInfo info = in;
        info.backend            = kBackendFsr4Vk;
        info.layerSplit         = false;
        return info;
    }
    return NgxRemapInterop(in);
}

static bool SameLiveSettings(const UpscalerCreateInfo& a, const UpscalerCreateInfo& b)
{
    if (a.backend != b.backend)
        return false;
    if (a.backend == kBackendXess)
        return true;
    if (a.backend == kBackendFsr4 || a.backend == kBackendFsr4Dx12 || a.backend == kBackendFsr4Interop ||
        a.backend == kBackendFsr4Vk)
        return a.fsr4Model == b.fsr4Model;
    return a.nssQuality == b.nssQuality && a.fastMath == b.fastMath;
}

Upscaler* NgxApplyLiveSettings(unsigned int id, Upscaler* current, const UpscalerCreate& create,
                               const UpscalerRemap& remap)
{
    const UpscalerCreateInfo& old  = current->Info();
    UpscalerCreateInfo        info = old;
    info.backend                   = SelectedBackend(&info.fsr4Model);
    if (remap)
        info = remap(info);
    if (info.backend == kBackendNss)
    {
        const int live = LiveNssQuality();
        if (live >= 0)
            info.nssQuality = live;
        info.fastMath = FastMathEnabled();
    }
    if (SameLiveSettings(info, old))
        return current;
    const std::string from = UpscalerName(old);
    auto fresh = create ? create(info) : CreateUpscaler(current->Device(), info);
    if (!fresh)
    {
        LOGI("switch from %s to %s failed; keeping it", from.c_str(), UpscalerName(info));
        return current;
    }
    LOGI("upscaler switched at runtime: %s -> %s", from.c_str(), UpscalerName(fresh->Info()));
    Registry().CollectGarbage();
    Upscaler* swapped = Registry().Replace(id, std::move(fresh));
    return swapped ? swapped : current;
}
