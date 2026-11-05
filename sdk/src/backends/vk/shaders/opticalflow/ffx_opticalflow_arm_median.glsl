// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// FFX_OF_IMAGE=arm (ffx_opticalflow_arm_common.h), per level, step 3 of 4: the layer's median_filter.comp, the 3x3
// median of each component (its sorting network, edges clamped).

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

layout(set = 0, binding = 1)        uniform texture2D r_arm_sub;
layout(set = 0, binding = 2, rg16f) uniform writeonly image2D rw_arm_med;

#define SORT(a, b)                     \
    {                                  \
        const f16vec2 lo = min(a, b);  \
        const f16vec2 hi = max(a, b);  \
        a                = lo;         \
        b                = hi;         \
    }

void main()
{
    const FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, ubo._LevelDims)))
        return;
    f16vec2 v[9];
    [[unroll]] for (FfxInt32 i = 0; i < 9; ++i)
        v[i] = f16vec2(texelFetch(r_arm_sub, clamp(pos + FfxInt32x2(i % 3, i / 3) - 1, FfxInt32x2(0), ubo._LevelDims - FfxInt32x2(1)), 0).xy);
    SORT(v[1], v[2]); SORT(v[4], v[5]); SORT(v[7], v[8]);
    SORT(v[0], v[1]); SORT(v[3], v[4]); SORT(v[6], v[7]);
    SORT(v[1], v[2]); SORT(v[4], v[5]); SORT(v[7], v[8]);
    SORT(v[0], v[3]); SORT(v[5], v[8]); SORT(v[4], v[7]);
    SORT(v[3], v[6]); SORT(v[1], v[4]); SORT(v[2], v[5]);
    SORT(v[4], v[7]); SORT(v[4], v[2]); SORT(v[6], v[4]);
    SORT(v[4], v[2]);
    imageStore(rw_arm_med, pos, FfxFloat32x4(FfxFloat32x2(v[4]), 0.0f, 0.0f));
}
