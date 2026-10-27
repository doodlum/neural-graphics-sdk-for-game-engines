// AMD FSR 4 on the game's own D3D12 device, from amdxcffx64.dll (AMD's DX12 driver FSR 4 component). Driven directly by
// the D3D12 NGX front end with ID3D12 resources and command list, so there is no Vulkan interop on this path.
// SPDX-License-Identifier: MIT
#pragma once

#include <d3d12.h>

#include <cstdint>

// Per-frame inputs for the D3D12 FSR 4 dispatch (DLSS conventions, as UpscalerDispatch for the Vulkan paths).
struct Fsr4Dx12Dispatch
{
    ID3D12Resource* color   = nullptr;
    ID3D12Resource* depth   = nullptr;
    ID3D12Resource* motion  = nullptr;
    ID3D12Resource* output  = nullptr;
    uint32_t        renderWidth = 0, renderHeight = 0;
    uint32_t        outputWidth = 0, outputHeight = 0;
    float           jitterX = 0, jitterY = 0;
    float           mvScaleX = 1, mvScaleY = 1;
    float           frameTimeMs = 16.6f;
    float           exposure    = 1.0f;
    bool            reset = false;
};

// One FSR 4 context on a D3D12 device.
class Fsr4Dx12
{
public:
    // info* mirror UpscalerCreateInfo; model is a Fsr4Model. nullptr when the DLL / model cannot be used (NSS is the
    // fallback, handled by the caller). Built once per handle.
    static Fsr4Dx12* Create(ID3D12Device* device, uint32_t maxRenderW, uint32_t maxRenderH, uint32_t displayW,
                            uint32_t displayH, bool hdr, bool depthInverted, bool highResMv, bool autoExposure, int model);
    ~Fsr4Dx12();

    bool Dispatch(ID3D12GraphicsCommandList* cmd, const Fsr4Dx12Dispatch& d);

    int  Model() const { return m_model; }      // Fsr4Model that runs
    int  RequestedModel() const { return m_requested; }

private:
    Fsr4Dx12() = default;

    void*          m_ctx         = nullptr;  // ffxContext
    ID3D12Device*  m_device      = nullptr;
    int            m_model       = 0;
    int            m_requested   = 0;
    bool           m_hdr = false, m_depthInverted = false, m_highResMv = false, m_autoExposure = false;
    uint64_t       m_dispatchCount = 0;
    uint32_t       m_displayW = 0, m_displayH = 0;
};

// True once FSR 4 on D3D12 has failed for that model in this process (or for every model, when the DLL is unusable).
bool Fsr4Dx12Unavailable(int model);
