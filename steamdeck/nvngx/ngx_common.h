// Shared NGX entry-point helpers for the D3D12 and Vulkan front ends.
// SPDX-License-Identifier: MIT
#pragma once

#include "ngx_abi.h"
#include "nss_upscaler.h"
#include "params.h"

#include <functional>
#include <memory>

// Parameter tables handed out by GetParameters / GetCapabilityParameters / AllocateParameters.
NVSDK_NGX_Result NgxGetParameters(NVSDK_NGX_Parameter** out);       // persistent table, owned by the proxy
NVSDK_NGX_Result NgxAllocateParameters(NVSDK_NGX_Parameter** out);  // caller frees with DestroyParameters
NVSDK_NGX_Result NgxDestroyParameters(NVSDK_NGX_Parameter* params);

// DLSS creation parameters translated to an NSS upscaler description.
UpscalerCreateInfo NgxCreateInfo(const NgxParams& p);

// Per-frame values that do not depend on the graphics API.
void NgxFillDispatch(const NgxParams& p, const UpscalerCreateInfo& info, UpscalerDispatch& d);

NVSDK_NGX_Result NgxFeatureRequirement(NVSDK_NGX_Feature feature, NVSDK_NGX_FeatureRequirement* out);

// Optional hooks for NgxApplyLiveSettings: a creator (defaults to CreateUpscaler on the current device) and a backend
// remap applied to the resolved settings (the D3D12 front end uses it to pick the native FSR 4 path).
using UpscalerCreate = std::function<std::unique_ptr<Upscaler>(const UpscalerCreateInfo&)>;
using UpscalerRemap  = std::function<UpscalerCreateInfo(const UpscalerCreateInfo&)>;

// Rebuilds the handle's upscaler when the upscaler or NSS quality asked for at runtime (control file) differs from the
// one it was created with; returns the upscaler to dispatch.
Upscaler* NgxApplyLiveSettings(unsigned int id, Upscaler* current, const UpscalerCreate& create = {},
                               const UpscalerRemap& remap = {});

// The upscaler asked for (control file, else AISS_UPSCALER): kBackendXess or kBackendFsr4 (with the FSR 4 model in
// *fsr4Model) unless that one has failed to load, else kBackendNss.
int SelectedBackend(int* fsr4Model);

// Vulkan and D3D11 front ends: FSR 4 runs through the D3D12 interop path (NSS once that has failed).
UpscalerCreateInfo NgxRemapInterop(const UpscalerCreateInfo& info);  // D3D11: FSR 4 through D3D12 interop (else NSS)
UpscalerCreateInfo NgxRemapVulkan(const UpscalerCreateInfo& info);   // Vulkan: same, with the AISS layer split

// "NSS quality 2", "XeSS", "FSR 4 Lite", ...
const char* UpscalerName(const UpscalerCreateInfo& info);

// Records the upscale, with GPU timestamps around it when AISS_GPU_TIMING is set.
bool NgxDispatch(Upscaler* upscaler, VkCommandBuffer cmd, const UpscalerDispatch& d);

// AISS_DUMP_FRAME for XeSS and FSR 4 (NSS dumps its own output): call after each dispatch.
void NgxDumpFrame(const Upscaler* upscaler, VkCommandBuffer cmd, const UpscalerImage& output);

// Allocates a handle for a created upscaler. NGX hands handles back by pointer, so they must stay valid.
NVSDK_NGX_Handle* NgxNewHandle(unsigned int id);
void              NgxFreeHandle(NVSDK_NGX_Handle* handle);
