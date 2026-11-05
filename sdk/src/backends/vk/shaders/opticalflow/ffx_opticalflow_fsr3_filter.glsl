// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// FFX_OF_IMAGE=fsr3, step 3 of 3, after each level's search (ffx_opticalflow_fsr3_search.glsl): FidelityFX FSR 3's
// ffx_opticalflow_filter_optical_flow_v5.h as it is, the 3x3 vector median of the 8x8 blocks' vectors (column by
// column, the first of equals; zero past the edges, as its out-of-range loads). The next finer level seeds from the
// result; level 0's is the flow: each 4x4 cell takes its block's vector, in colour pixels.

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

#include "ffx_opticalflow_callbacks_glsl.h"
#include "ffx_opticalflow_image_common.h"

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 1)        uniform texture2D r_flow_raw;
layout(set = 0, binding = 2, rg16f) uniform writeonly image2D rw_flow_levels;
layout(set = 0, binding = 3, rg16f) uniform writeonly image2D rw_flow;  // optical flow resolution (level 0's 4x4 cells)

void main()
{
    const FfxInt32x2 o = FfxInt32x2(gl_GlobalInvocationID.xy);
    const FfxInt32x2 b = ubo._Final != 0 ? o >> 1 : o;
    if (any(greaterThanEqual(b, ubo._CellDims)) || (ubo._Final != 0 && any(greaterThanEqual(o, ubo._HintDims))))
        return;
    FfxInt32x2 v[9];
    [[unroll]] for (FfxInt32 i = 0; i < 9; ++i)
    {
        const FfxInt32x2 p = b + FfxInt32x2(i / 3, i % 3) - 1;
        v[i] = any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, ubo._CellDims))
                   ? FfxInt32x2(0)
                   : FfxInt32x2(texelFetch(r_flow_raw, FfxInt32x2(p.x, ubo._FlowRow + p.y), 0).xy);
    }
    FfxUInt32 best = 0xffffffffu;
    [[unroll]] for (FfxInt32 i = 0; i < 9; ++i)
    {
        FfxUInt32 d = 0u;
        [[unroll]] for (FfxInt32 j = 0; j < 9; ++j)
        {
            const FfxInt32x2 e = v[i] - v[j];
            d += FfxUInt32(e.x * e.x + e.y * e.y);
        }
        best = min(best, (d << 4) | FfxUInt32(i));
    }
    const FfxFloat32x2 m = FfxFloat32x2(v[best & 0xfu]);
    if (ubo._Final != 0)
        imageStore(rw_flow, o, FfxFloat32x4(m * ubo._OutScale, 0.0f, 0.0f));
    else
        imageStore(rw_flow_levels, FfxInt32x2(b.x, ubo._FlowRow + b.y), FfxFloat32x4(m, 0.0f, 0.0f));
}
