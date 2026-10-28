// AMD FSR 4 on the game's D3D12 device (amdxcffx64.dll). The driver loading, GPU-family answer and per-network model
// selection are shared with the D3D12 interop path (fsr4_common). amdxcffx64 learns the GPU family through AMD's amdxc64 driver
// extension or a D3DKMT query; Proton Experimental ships a GE-Proton-derived amdxc64.dll stand-in, and the D3DKMT answer
// in fsr4_common covers the rest. The FFX DX12 backend runs the network straight on the game's device and command list,
// so there is no Vulkan interop here.
// SPDX-License-Identifier: MIT
#include "fsr4_dx12.h"

#include "common.h"
#include "fsr4_common.h"
#include "nss_upscaler.h"  // Fsr4Model, Fsr4ModelName

#include <windows.h>

#include "amd_ffx/ffx_api.h"
#include "amd_ffx/ffx_api_dx12.h"
#include "amd_ffx/ffx_upscale.h"

#include <algorithm>
#include <mutex>
#include <string>

namespace
{
constexpr uint64_t kBackendDx12        = 0x00000002u;  // FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12
constexpr uint64_t kUpscaleVersionDesc = 0x0001000Bu;
constexpr uint32_t kUpscaleVersion     = (4u << 22) | (1u << 12) | 2u;

struct BackendDx12Desc
{
    ffxCreateContextDescHeader header;
    ID3D12Device*              device;
};
struct UpscaleVersionDesc
{
    ffxCreateContextDescHeader header;
    uint32_t                   version;
};

std::mutex g_mutex;
bool       g_loadTried = false, g_loadOk = false;
bool       g_modelFailed[kFsr4ModelCount] = {};
Fsr4Driver g_drv;

Fsr4Gpu GpuForDevice(ID3D12Device* device)
{
    // amdxcffx64's own GPU query is what fsr4_common answers; here we only need the family for the auto-model fallback.
    // vkd3d-proton reports the adapter's real vendor/device through DXGI, but the simplest reliable source on a Steam
    // Deck is the DRM query fsr4_common does by name, so pass the adapter description's vendor/device and let
    // Fsr4KnownGpu map Van Gogh (its RADV-name match does not apply on D3D12, so AISS_FSR4_GPU can override).
    Fsr4Gpu gpu;
    LUID     luid = device->GetAdapterLuid();
    (void)luid;
    // Default to Van Gogh on a Steam Deck; AISS_FSR4_GPU overrides inside Fsr4KnownGpu.
    return Fsr4KnownGpu(0x1002, 0x1435, "VANGOGH");
}

const Fsr4Driver* LoadDx12Driver(ID3D12Device* device)
{
    if (g_loadTried)
        return g_loadOk ? &g_drv : nullptr;
    g_loadTried = true;
    g_loadOk    = Fsr4LoadDriver(L"amdxcffx64.dll", GpuForDevice(device), g_drv);
    return g_loadOk ? &g_drv : nullptr;
}

void FfxMessage(uint32_t type, const wchar_t* message)
{
    LOGI("FSR 4 DX12 runtime %s: %ls", type == 0 ? "error" : "warning", message);
}

uint32_t ResourceState(bool uav)
{
    return uav ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_COMPUTE_READ;
}

// AISS_GPU_TIMING=N for the D3D12 path: a pair of GPU timestamps around the FFX dispatch on the game's command list,
// resolved to a readback buffer and read back kSlots dispatches later (never waiting on the game's queue), averaged
// over N dispatches. D3D12 counterpart of gpu_timing.cpp (which works on VkCommandBuffer).
struct Dx12Timer
{
    static constexpr uint32_t kSlots = 32;
    ID3D12QueryHeap*          heap = nullptr;
    ID3D12Resource*           readback = nullptr;  // kSlots*2 UINT64 timestamps
    double                    nsPerTick = 1.0;
    uint64_t                  written = 0;
    uint8_t                   valid[kSlots] = {};
    double                    sum = 0, minUs = 1e30, maxUs = 0;
    uint32_t                  samples = 0;
    uint64_t                  reportedAt = 0;
    std::string               name;
    bool                      failed = false;

    bool Setup(ID3D12Device* device)
    {
        if (heap || failed)
            return heap != nullptr;
        failed = true;
        // Timestamp frequency from a throwaway queue (only CreateCommandQueue + GetTimestampFrequency, never submitted).
        ID3D12CommandQueue*     q    = nullptr;
        D3D12_COMMAND_QUEUE_DESC qd   = {};
        qd.Type                       = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&q))) || !q)
            return false;
        UINT64 freq = 0;
        const HRESULT fr = q->GetTimestampFrequency(&freq);
        q->Release();
        if (FAILED(fr) || !freq)
            return false;
        nsPerTick = 1e9 / double(freq);
        D3D12_QUERY_HEAP_DESC hd = {};
        hd.Type                  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        hd.Count                 = kSlots * 2;
        if (FAILED(device->CreateQueryHeap(&hd, __uuidof(ID3D12QueryHeap), reinterpret_cast<void**>(&heap))))
            return false;
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type                  = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rd   = {};
        rd.Dimension             = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width                 = sizeof(uint64_t) * kSlots * 2;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count             = 1;
        rd.Layout                       = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   __uuidof(ID3D12Resource), reinterpret_cast<void**>(&readback))))
        {
            heap->Release();
            heap = nullptr;
            return false;
        }
        failed = false;
        return true;
    }

    void Collect(uint32_t slot, const char* label)
    {
        if (!valid[slot])
            return;
        valid[slot] = 0;
        D3D12_RANGE  r   = {slot * 2 * sizeof(uint64_t), (slot * 2 + 2) * sizeof(uint64_t)};
        void*        ptr = nullptr;
        if (FAILED(readback->Map(0, &r, &ptr)) || !ptr)
            return;
        const uint64_t* t  = reinterpret_cast<const uint64_t*>(static_cast<uint8_t*>(ptr) + slot * 2 * sizeof(uint64_t));
        const double    us = double(t[1] - t[0]) * nsPerTick / 1000.0;
        readback->Unmap(0, nullptr);
        if (t[1] <= t[0])
            return;
        sum += us;
        minUs = us < minUs ? us : minUs;
        maxUs = us > maxUs ? us : maxUs;
        if (++samples >= uint32_t(GpuTimingFrames()))
        {
            const uint64_t now = GetTickCount64();
            const double   fps = reportedAt ? samples * 1000.0 / double(now - reportedAt) : 0.0;
            reportedAt         = now;
            LOGI("GPU time per upscale (%s): mean %.1f us, min %.1f, max %.1f over %u evaluations, %.1f evaluations/s",
                 label, sum / samples, minUs, maxUs, samples, fps);
            sum = 0, samples = 0, minUs = 1e30, maxUs = 0;
        }
    }
};
std::mutex g_timerMutex;
Dx12Timer  g_timer;
}  // namespace

Fsr4Dx12* Fsr4Dx12::Create(ID3D12Device* device, uint32_t maxRenderW, uint32_t maxRenderH, uint32_t displayW,
                           uint32_t displayH, bool hdr, bool depthInverted, bool highResMv, bool autoExposure, int model)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    const int wanted = model >= 0 && model < kFsr4ModelCount ? model : kFsr4Auto;
    if ((g_loadTried && !g_loadOk) || g_modelFailed[wanted])
        return nullptr;
    const Fsr4Driver* drv = LoadDx12Driver(device);
    if (!drv)
        return nullptr;
    if (wanted == kFsr4Fp8)
    {
        LOGI("FSR 4 DX12: FP8 is not offered on this GPU");
        g_modelFailed[wanted] = true;
        return nullptr;
    }
    if (wanted != kFsr4Auto && !drv->HasModel(wanted))
    {
        LOGI("FSR 4 DX12: amdxcffx64.dll has no %s network", Fsr4ModelName(wanted));
        g_modelFailed[wanted] = true;
        return nullptr;
    }

    uint64_t backendType = kBackendDx12;
    if (const char* v = getenv("AISS_FSR4_DX12_BACKEND"))
        backendType = strtoull(v, nullptr, 0);
    BackendDx12Desc backend = {};
    backend.header.type     = backendType;
    backend.device          = device;
    // amdxcffx64 rejects the upscaler version desc (createContext returns
    // ERROR_PARAMETER); on DX12 it is left out unless AISS_FSR4_DX12_VER forces a value.
    UpscaleVersionDesc version = {};
    version.header.type        = kUpscaleVersionDesc;
    version.version            = kUpscaleVersion;
    bool chainVersion = false;
    if (const char* v = getenv("AISS_FSR4_DX12_VER"))
    {
        version.version = uint32_t(strtoul(v, nullptr, 0));
        chainVersion    = version.version != 0;
    }

    ffxCreateContextDescUpscale create = {};
    create.header.type                 = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags                       = FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
    if (hdr)
        create.flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    if (depthInverted)
        create.flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    if (highResMv)
        create.flags |= FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    const bool autoExp = autoExposure && Config().exposure <= 0.0f;
    if (autoExp)
        create.flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    if (LogLevel() >= 2)
        create.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    create.maxRenderSize  = {std::min(maxRenderW, displayW), std::min(maxRenderH, displayH)};
    create.maxUpscaleSize = {displayW, displayH};
    create.fpMessage      = &FfxMessage;
    create.header.pNext   = &backend.header;
    backend.header.pNext  = chainVersion ? &version.header : nullptr;

    std::unique_ptr<Fsr4Dx12> up(new Fsr4Dx12());
    up->m_device        = device;
    up->m_requested     = wanted;
    up->m_hdr           = hdr;
    up->m_depthInverted = depthInverted;
    up->m_highResMv     = highResMv;
    up->m_autoExposure  = autoExp;
    up->m_displayW      = displayW;
    up->m_displayH      = displayH;

    const auto t0 = GetTickCount64();
    const int  accepted = Fsr4CreateWithModel(*drv, wanted, GpuForDevice(device), &up->m_ctx, &create.header);
    if (accepted < 0)
    {
        g_modelFailed[wanted] = true;
        return nullptr;
    }
    up->m_model = accepted;
    LOGI("FSR 4 (DX12) context %ux%u -> %ux%u, flags 0x%x, created in %llu ms", create.maxRenderSize.width,
         create.maxRenderSize.height, displayW, displayH, create.flags, (unsigned long long)(GetTickCount64() - t0));
    return up.release();
}

Fsr4Dx12::~Fsr4Dx12()
{
    if (m_ctx)
        g_drv.provider.destroyContext(&m_ctx, nullptr);
}

bool Fsr4Dx12::Dispatch(ID3D12GraphicsCommandList* cmd, const Fsr4Dx12Dispatch& d)
{
    ffxDispatchDescUpscale up = {};
    up.header.type            = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    up.commandList            = cmd;
    // ffxApiGetResourceDX12 reads each resource's own D3D12 desc (dimensions, format, depth/stencil usage). The game
    // leaves its inputs in NON_PIXEL_SHADER_RESOURCE and the output in UNORDERED_ACCESS (see the NGX D3D12 evaluate).
    up.color         = ffxApiGetResourceDX12(d.color, ResourceState(false));
    up.depth         = ffxApiGetResourceDX12(d.depth, ResourceState(false));
    up.motionVectors = ffxApiGetResourceDX12(d.motion, ResourceState(false));
    up.output        = ffxApiGetResourceDX12(d.output, ResourceState(true));
    up.jitterOffset      = {d.jitterX * Config().jitterSign[0], d.jitterY * Config().jitterSign[1]};
    up.motionVectorScale = {d.mvScaleX * Config().mvSign[0], d.mvScaleY * Config().mvSign[1]};
    up.renderSize  = {d.renderWidth, d.renderHeight};
    up.upscaleSize = {d.outputWidth, d.outputHeight};
    up.enableSharpening        = false;
    up.sharpness               = 0.0f;
    up.frameTimeDelta          = d.frameTimeMs;
    up.preExposure             = 1.0f;
    up.reset                   = d.reset || m_dispatchCount == 0;
    up.cameraNear              = Config().cameraNear;
    up.cameraFar               = Config().cameraFar;
    up.cameraFovAngleVertical  = Config().cameraFovY;
    up.viewSpaceToMetersFactor = 1.0f;
    if (m_depthInverted)
        std::swap(up.cameraNear, up.cameraFar);

    // Optional GPU timing: bracket the dispatch with timestamps on the game's command list.
    uint32_t timerSlot = UINT32_MAX;
    if (GpuTimingFrames() > 0)
    {
        std::lock_guard<std::mutex> lock(g_timerMutex);
        if (g_timer.Setup(m_device))
        {
            timerSlot = uint32_t(g_timer.written++ % Dx12Timer::kSlots);
            char label[64];
            snprintf(label, sizeof(label), "FSR 4 %s DX12", Fsr4ModelName(m_model));
            g_timer.Collect(timerSlot, label);  // the slot's previous use, kSlots dispatches ago (retired)
            cmd->EndQuery(g_timer.heap, D3D12_QUERY_TYPE_TIMESTAMP, timerSlot * 2);
        }
    }

    const uint32_t rc = g_drv.provider.dispatch(&m_ctx, &up.header);
    ++m_dispatchCount;

    if (timerSlot != UINT32_MAX)
    {
        std::lock_guard<std::mutex> lock(g_timerMutex);
        cmd->EndQuery(g_timer.heap, D3D12_QUERY_TYPE_TIMESTAMP, timerSlot * 2 + 1);
        cmd->ResolveQueryData(g_timer.heap, D3D12_QUERY_TYPE_TIMESTAMP, timerSlot * 2, 2, g_timer.readback,
                              timerSlot * 2 * sizeof(uint64_t));
        g_timer.valid[timerSlot] = 1;
    }
    if ((m_dispatchCount % 600) == 1 || LogLevel() >= 2)
        LOGI("FSR 4 DX12 (%s) evaluate %llu: render %ux%u -> %ux%u, jitter %.3f,%.3f, mv scale %.2f,%.2f%s",
             Fsr4ModelName(m_model), (unsigned long long)m_dispatchCount, up.renderSize.width, up.renderSize.height,
             up.upscaleSize.width, up.upscaleSize.height, up.jitterOffset.x, up.jitterOffset.y, up.motionVectorScale.x,
             up.motionVectorScale.y, m_autoExposure ? " (auto exposure)" : "");
    if (rc != FFX_API_RETURN_OK)
    {
        LOGI("FSR 4 DX12: dispatch failed (%u)", rc);
        return false;
    }
    return true;
}

bool Fsr4Dx12Unavailable(int model)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loadTried && !g_loadOk)
        return true;
    return model >= 0 && model < kFsr4ModelCount && g_modelFailed[model];
}
