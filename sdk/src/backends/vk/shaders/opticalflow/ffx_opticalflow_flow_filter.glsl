// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Image optical flow without the data graph optical flow engine, step 3 of 3, after each level's search: FidelityFX
// FSR 3's filter (ffx_opticalflow_filter_optical_flow_v5.h), the 3x3 vector median (the vector nearest the others),
// which removes single wrong blocks. The next finer level seeds from the result; level 0's is the flow, in colour
// pixels.

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
layout(set = 0, binding = 3, rg16f) uniform writeonly image2D rw_flow;  // optical flow resolution (level 0's cells)  // optical flow resolution (level 0's cells)

void main()
{
    // The finest level's filter runs per level 0 cell (the flow's texels), on the level's cell under it.
    const FfxInt32x2 o = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(ubo._Final != 0 ? o >> ubo._Level : o, ubo._CellDims)) || (ubo._Final != 0 && any(greaterThanEqual(o, ubo._HintDims))))
        return;
    const FfxInt32x2 c = ubo._Final != 0 ? o >> ubo._Level : o;
    FfxFloat32x2     v[9];
    [[unroll]] for (FfxInt32 i = 0; i < 9; ++i)
    {
        const FfxInt32x2 p = clamp(c + FfxInt32x2(i % 3, i / 3) - 1, FfxInt32x2(0), ubo._CellDims - FfxInt32x2(1));
        v[i]               = texelFetch(r_flow_raw, FfxInt32x2(p.x, ubo._FlowRow + p.y), 0).xy;
    }
    FfxFloat32   best  = 3.4e38f;
    FfxFloat32x2 bestV = v[4];
    [[unroll]] for (FfxInt32 i = 0; i < 9; ++i)
    {
        FfxFloat32 d = 0.0f;
        [[unroll]] for (FfxInt32 j = 0; j < 9; ++j)
            d += dot(v[i] - v[j], v[i] - v[j]);
        if (d < best)
            best = d, bestV = v[i];
    }
    if (ubo._Final != 0)
        imageStore(rw_flow, o, FfxFloat32x4(bestV * FfxFloat32(1 << ubo._Level) * ubo._OutScale, 0.0f, 0.0f));
    else
        imageStore(rw_flow_levels, FfxInt32x2(c.x, ubo._FlowRow + c.y), FfxFloat32x4(bestV, 0.0f, 0.0f));
}
