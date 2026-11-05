// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Flow without the data graph optical flow engine (hints only), after ffx_opticalflow_scatter.glsl: each optical
// flow texel something landed on takes the winner's flow (TM1 -> TP1, display pixels) over the camera-motion hint,
// and its key is reset for the next frame.

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : enable

#include "ffx_opticalflow_callbacks_glsl.h"

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Uniforms {
    FfxFloat32x2 _MotionVectorScale;
    FfxInt32x2   _OutputDims;          // optical flow resolution
    FfxInt32x2   _MotionVectorDims;
    FfxInt32x2   _DepthDims;
    FfxInt32     _DepthInverted;
    FfxInt32     _Pad;
    FfxInt32x2   _DisplayDims;
}ubo;

layout(set = 0, binding = 1, std430) buffer ScatterKeys_t
{
    uint64_t keys[];
}
rw_scatter_keys;
layout(set = 0, binding = 2, rg16f) uniform writeonly image2D rw_flow;  // holds the MV hints

void main()
{
    const FfxInt32x2 c = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(c, ubo._OutputDims)))
        return;
    const uint32_t i = uint32_t(c.y * ubo._OutputDims.x + c.x);
    const uint64_t k = rw_scatter_keys.keys[i];
    if (k == ~0ul)
        return;
    imageStore(rw_flow, c, FfxFloat32x4(unpackHalf2x16(uint32_t(k)), 0.0f, 0.0f));
    rw_scatter_keys.keys[i] = ~0ul;
}
