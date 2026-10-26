// Shared NGX entry-point helpers for the D3D12 and Vulkan front ends.
// SPDX-License-Identifier: MIT
#pragma once

#include "ngx_abi.h"
#include "nss_upscaler.h"
#include "params.h"

// Parameter tables handed out by GetParameters / GetCapabilityParameters / AllocateParameters.
NVSDK_NGX_Result NgxGetParameters(NVSDK_NGX_Parameter** out);       // persistent table, owned by the proxy
NVSDK_NGX_Result NgxAllocateParameters(NVSDK_NGX_Parameter** out);  // caller frees with DestroyParameters
NVSDK_NGX_Result NgxDestroyParameters(NVSDK_NGX_Parameter* params);

// DLSS creation parameters translated to an NSS upscaler description.
UpscalerCreateInfo NgxCreateInfo(const NgxParams& p);

// Per-frame values that do not depend on the graphics API.
void NgxFillDispatch(const NgxParams& p, const UpscalerCreateInfo& info, UpscalerDispatch& d);

NVSDK_NGX_Result NgxFeatureRequirement(NVSDK_NGX_Feature feature, NVSDK_NGX_FeatureRequirement* out);

// Rebuilds the handle's upscaler when the NSS quality asked for at runtime (control file) differs from the one it
// was created with; returns the upscaler to dispatch.
NssUpscaler* NgxApplyLiveSettings(unsigned int id, NssUpscaler* current);

// Allocates a handle for a created upscaler. NGX hands handles back by pointer, so they must stay valid.
NVSDK_NGX_Handle* NgxNewHandle(unsigned int id);
void              NgxFreeHandle(NVSDK_NGX_Handle* handle);
