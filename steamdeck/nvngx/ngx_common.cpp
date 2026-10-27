// SPDX-License-Identifier: MIT
#include "ngx_common.h"

#include "common.h"

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
    if (backend == kBackendFsr4 && Fsr4Unavailable(model))
        backend = kBackendNss;
    if (fsr4Model)
        *fsr4Model = backend == kBackendFsr4 ? model : kFsr4Auto;
    return backend;
}

const char* UpscalerName(const UpscalerCreateInfo& i)
{
    static thread_local char text[64];
    if (i.backend == kBackendXess)
        snprintf(text, sizeof(text), "XeSS");
    else if (i.backend == kBackendFsr4)
        snprintf(text, sizeof(text), "FSR 4 %s%s%s%s", Fsr4ModelName(i.fsr4Model),
                 i.fsr4Model == kFsr4Auto && i.fsr4Active > 0 ? " (" : "",
                 i.fsr4Model == kFsr4Auto && i.fsr4Active > 0 ? Fsr4ModelName(i.fsr4Active) : "",
                 i.fsr4Model == kFsr4Auto && i.fsr4Active > 0 ? ")" : "");
    else
        snprintf(text, sizeof(text), "NSS quality %d%s", i.nssQuality, i.fastMath ? " fast maths" : "");
    return text;
}

Upscaler* NgxApplyLiveSettings(unsigned int id, Upscaler* current)
{
    const UpscalerCreateInfo& old  = current->Info();
    UpscalerCreateInfo        info = old;
    info.backend                   = SelectedBackend(&info.fsr4Model);
    if (info.backend == kBackendNss)
    {
        const int live = LiveNssQuality();
        if (live >= 0)
            info.nssQuality = live;
        info.fastMath = FastMathEnabled();
    }
    if (info.backend == old.backend &&
        (info.backend == kBackendXess || (info.backend == kBackendFsr4 && info.fsr4Model == old.fsr4Model) ||
         (info.backend == kBackendNss && info.nssQuality == old.nssQuality && info.fastMath == old.fastMath)))
        return current;
    const std::string from = UpscalerName(old);
    auto fresh = CreateUpscaler(current->Device(), info);
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
