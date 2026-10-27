// Shared plumbing for the two AMD FSR 4 paths: amdvkffx64.dll (Vulkan, fsr4_upscaler.cpp) and amdxcffx64.dll (D3D12,
// fsr4_dx12.cpp). Both DLLs export the same FidelityFX provider entry point and carry the same per-network provider
// classes; only the graphics backend and the way they learn the GPU family differ. This loads one such DLL, answers
// its GPU query, wraps its per-network device checks for model selection, and hands back the upscaler provider.
// SPDX-License-Identifier: MIT
#pragma once

#include "nss_upscaler.h"  // Fsr4Model

#include <windows.h>

#include <cstdint>
#include <string>

// AMD's FidelityFX provider interface (same layout as the SDK's ffxProviderInterface).
struct Fsr4Provider
{
    uint64_t    versionId;
    const char* versionName;
    uint32_t (*canProvide)(uint64_t descType);
    uint32_t (*createContext)(void* context, void* desc, const void* allocator);
    uint32_t (*destroyContext)(void* context, const void* allocator);
    uint32_t (*configure)(void* context, const void* desc);
    uint32_t (*query)(void* context, void* desc);
    uint32_t (*dispatch)(void* context, const void* desc);
};

// One loaded FSR 4 driver DLL.
struct Fsr4Driver
{
    HMODULE      module      = nullptr;
    Fsr4Provider provider    = {};
    std::string  versionName;
    bool (*deviceCheck)(const void* backend) = nullptr;  // provider-level check (block version 3)
    int          modelsFound = 0;                        // bit per Fsr4Model found as a provider class
    const char*  name        = "";

    bool HasModel(int model) const { return (modelsFound & (1 << model)) != 0; }
};

// GPU family / revision for the DLL's own GPU query (amdgpu AMDGPU_INFO_DEV_INFO family + external revision).
struct Fsr4Gpu
{
    uint32_t family   = 0;
    uint32_t revision = 0;
};

// The family / revision for this adapter: AISS_FSR4_GPU override, else Van Gogh (144,1) by device id or RADV name.
Fsr4Gpu Fsr4KnownGpu(uint32_t vendorId, uint32_t deviceId, const char* deviceName);
// The network AMD's own device checks accept for a family / revision (empirical, from amd*ffx64 4.1.2), or -1.
int Fsr4ModelForGpu(const Fsr4Gpu& gpu);

// Loads the DLL from next to the proxy and gets its upscaler provider. backendDescType is the FidelityFX backend
// create-desc type the DLL understands (Vulkan or DX12). The GPU answered to its D3DKMT query is `gpu`. Returns false
// (and logs why) when the DLL is missing, hands out no provider, or cannot be set up. Done once per DLL.
bool Fsr4LoadDriver(const wchar_t* dllName, const Fsr4Gpu& gpu, Fsr4Driver& out);

// Runs the provider's createContext with model selection active: only `wantedModel`'s network accepts the device
// (kFsr4Auto lets AMD's own answer through). On return, *accepted is the network that accepted (or -1), and
// *nativeMask has a bit per network AMD's own check accepted. The caller passes the fully built create desc chain.
uint32_t Fsr4CreateContext(const Fsr4Driver& drv, int wantedModel, void* context, void* createDesc, int* accepted,
                           int* nativeMask);

// Turns on AMD's watermark (MLSR-WATERMARK) when AISS_FSR4_WATERMARK / FSR_WATERMARK is set. Call once per process.
void Fsr4EnableWatermark();

// "FP8" etc.: a comma list of the networks the driver has, for logging.
std::string Fsr4ModelList(const Fsr4Driver& drv);

// Runs createContext for `wanted` (kFsr4Auto = AMD's own pick). Under Proton's NVIDIA spoof AMD's checks see a non-AMD
// vendor and accept nothing; `auto` then falls back to the network Fsr4ModelForGpu(gpu) names (destroying and
// recreating the context). Returns the network that runs, or -1 (logging why), and leaves *context valid on success.
int Fsr4CreateWithModel(const Fsr4Driver& drv, int wanted, const Fsr4Gpu& gpu, void* context, void* createDesc);
