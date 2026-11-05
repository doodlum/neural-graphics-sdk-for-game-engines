// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// FFX_OF_IMAGE=arm (ffx_opticalflow_arm_common.h), per level from 5 to 2, step 1 of 4: the layer's
// mv_process_and_warp.comp. TP1's level warped by the coarser level's filtered flow (bilinear, doubled), so that the
// block match searches only the residual; at level 5 TP1's level as it is.

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

layout(set = 0, binding = 1)     uniform texture2D r_pyramid_tp1;
layout(set = 0, binding = 2)     uniform texture2D r_arm_flow;  // the coarser level's filtered flow (its texels)
layout(set = 0, binding = 3, r8) uniform writeonly image2D rw_arm_warp;

void main()
{
    const FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, ubo._LevelDims)))
        return;
    FfxFloat32x2 flow = FfxFloat32x2(0.0f);
    if (ubo._CoarseDims.x > 0)
        flow = FfxFloat32x2(f16vec2(ArmBilinear(r_arm_flow, 0, ubo._CoarseDims, (FfxFloat32x2(pos) + 0.5f) * 0.5f).xy * 2.0f));
    precise FfxFloat32x2 src = FfxFloat32x2(pos) + 0.5f + flow;
    const FfxFloat32     w   = ArmBilinear(r_pyramid_tp1, ubo._LevelRow, ubo._LevelDims, src).x;
    imageStore(rw_arm_warp, pos, FfxFloat32x4(FfxFloat32(ArmQuantize(w)) / 255.0f, 0.0f, 0.0f, 0.0f));
}
