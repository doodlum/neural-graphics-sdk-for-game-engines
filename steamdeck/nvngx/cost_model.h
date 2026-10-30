// Equal-cost render scale: each upscaler gets the render size at which the whole frame (the game's render-resolution
// work + the upscale) costs the same GPU time as NSS Quality at the DLSS mode's standard ratio.
// SPDX-License-Identifier: MIT
#pragma once

#include "nss_upscaler.h"

enum class CostApi : int
{
    Unknown,
    D3D11,
    D3D12,
    Vulkan,
};

// The front end the game initialised (FSR 4 costs differ: native on D3D12, interop on D3D11 / Vulkan).
void CostSetApi(CostApi api);
CostApi CostGetApi();

// Called once per evaluate with the upscaler that ran and the sizes: feeds the game-cost estimate.
void CostNoteEvaluate(const UpscalerCreateInfo& info, unsigned renderW, unsigned renderH, unsigned outW, unsigned outH);

// The render-scale divisor (output / render, per axis) for DLSS mode `perfQuality` at this output size. Returns
// `standardRatio` for DLAA, when equal cost is off (control file equal_cost=0, DU_EQUAL_COST=0) or for NSS Quality.
float EqualCostRatio(int perfQuality, float standardRatio, unsigned outW, unsigned outH);

// Whether equal cost is on: the control file's equal_cost, else DU_EQUAL_COST (default on).
bool EqualCostEnabled();
