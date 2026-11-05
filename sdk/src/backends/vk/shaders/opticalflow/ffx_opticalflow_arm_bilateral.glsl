// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// FFX_OF_IMAGE=arm (ffx_opticalflow_arm_common.h), per level, step 4 of 4: the layer's bilateral_filter.comp, a 5x5
// filter of the flow weighted by TM1's intensity differences, which the next finer level warps by. At the last level
// (level 2) also the hint (the layer's dense_warp.comp, block_match_of.comp's RAW_SAD and mv_replace.comp): TP1
// warped by the hint flow, the 5x5 SAD of TM1 against that, and where it is lower than the match's the hint is the
// flow; else the filtered flow, in colour pixels (x 4).

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : enable

#include "ffx_opticalflow_arm_common.h"

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 1)        uniform texture2D r_pyramid_tm1;
layout(set = 0, binding = 2)        uniform texture2D r_pyramid_tp1;
layout(set = 0, binding = 3)        uniform texture2D r_arm_med;
layout(set = 0, binding = 4)        uniform utexture2D r_arm_match;
layout(set = 0, binding = 5)        uniform texture2D r_flow_hint;  // the hint, colour pixels per 4x4 cell
layout(set = 0, binding = 6, rg16f) uniform writeonly image2D rw_arm_flow;
layout(set = 0, binding = 7, rg16f) uniform writeonly image2D rw_flow;  // optical flow resolution

const float16_t kSigmaIntensityDivSqrt = float16_t(5.05129933779);
const float16_t kSpatial               = float16_t(-0.00913);

// TP1's level 2 warped by the hint at p (dense_warp.comp), zero off the level as the RAW_SAD reads it.
FfxUInt32 HintWarped(FfxInt32x2 p)
{
    if (any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, ubo._LevelDims)))
        return 0u;
    // The hint image is the output's size; past it (the level padded to even) a zero flow.
    const FfxFloat32x2 hint = all(lessThan(p, ubo._HintDims)) ? texelFetch(r_flow_hint, p, 0).xy : FfxFloat32x2(0.0f);
    precise FfxFloat32x2 src = FfxFloat32x2(p) + 0.5f + hint * 0.25f;
    return ArmQuantize(ArmBilinear(r_pyramid_tp1, ubo._LevelRow, ubo._LevelDims, src).x);
}

void main()
{
    const FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, ubo._LevelDims)) || (ubo._Final != 0 && any(greaterThanEqual(pos, ubo._HintDims))))
        return;
    precise float16_t centre = float16_t(ArmTexel(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, pos));
    precise float16_t coeffSum  = float16_t(0.0);
    precise f16vec2    filterSum = f16vec2(0.0);
    [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
    {
        const FfxInt32x2   k     = pos + FfxInt32x2(i % 5, i / 5) - 2;
        precise float16_t diff  = (centre - float16_t(ArmTexel(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, k))) * kSigmaIntensityDivSqrt;
        precise float16_t coeff = clamp(float16_t(1.0) - abs(kSpatial - diff * diff), float16_t(0.0), float16_t(1.0));
        coeffSum += coeff;
        filterSum += coeff * f16vec2(texelFetch(r_arm_med, clamp(k, FfxInt32x2(0), ubo._LevelDims - FfxInt32x2(1)), 0).xy);
    }
    precise f16vec2 filtered = filterSum / coeffSum;
    if (ubo._Final == 0)
    {
        imageStore(rw_arm_flow, pos, FfxFloat32x4(FfxFloat32x2(filtered), 0.0f, 0.0f));
        return;
    }
    filtered *= f16vec2(4.0);
    FfxUInt32 hintCost = 0u;
    [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
    {
        const FfxInt32x2 b = pos + FfxInt32x2(i % 5, i / 5) - ARM_BLOCK_R;
        hintCost += FfxUInt32(abs(FfxInt32(HintWarped(b)) - FfxInt32(ArmU8Zero(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, b))));
    }
    const FfxUInt32 matchCost = texelFetch(r_arm_match, pos, 0).x;
    const f16vec2   hint      = f16vec2(texelFetch(r_flow_hint, pos, 0).xy);
    imageStore(rw_flow, pos, FfxFloat32x4(FfxFloat32x2(hintCost < matchCost ? hint : filtered), 0.0f, 0.0f));
}
