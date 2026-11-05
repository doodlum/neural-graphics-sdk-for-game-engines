// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Flow without the data graph optical flow engine (hints only), in one pass where the device has 64-bit buffer
// atomics (instead of ffx_opticalflow_scatter_depth.glsl then _flow.glsl, each reading all the motion vectors and
// depth: the Deck's bandwidth is most of their time): one TP1 pixel of every 2x2 block takes a 64-bit atomic min of
// its depth key (high word: the nearest wins) and its flow (low word, half floats: the smaller bits win a tie) in
// the optical flow texel it came from in TM1. ffx_opticalflow_scatter_resolve.glsl writes the winners' flow.

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_atomic_int64 : require

#include "ffx_opticalflow_callbacks_glsl.h"

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Uniforms {
    FfxFloat32x2 _MotionVectorScale;   // raw motion vector -> display pixels (from TP1 to where the pixel was in TM1)
    FfxInt32x2   _OutputDims;          // optical flow resolution
    FfxInt32x2   _MotionVectorDims;
    FfxInt32x2   _DepthDims;
    FfxInt32     _DepthInverted;
    FfxInt32     _Pad;
    FfxInt32x2   _DisplayDims;         // the flow's pixels (the motion vectors may be at the render size)
}ubo;

layout(set = 0, binding = 1) uniform texture2D r_game_motion_vectors;  // display resolution | RG
layout(set = 0, binding = 2) uniform texture2D r_depth;                // TP1 depth | R32F

// Each invocation takes one TP1 pixel of a 2x2 block (an optical flow texel is 4x4 pixels or more): a quarter of
// the atomics and memory traffic.
FfxUInt32 DepthKey(FfxInt32x2 p)
{
    const FfxInt32x2 d = clamp(p * ubo._DepthDims / ubo._MotionVectorDims, FfxInt32x2(0), ubo._DepthDims - FfxInt32x2(1));
    const FfxFloat32 z = clamp(texelFetch(r_depth, d, 0).r, 0.0f, 1.0f);
    return floatBitsToUint(ubo._DepthInverted != 0 ? 1.0f - z : z);  // nearer is smaller; non-negative floats order like their bits
}

// Where this TP1 pixel was in TM1, as an optical flow texel (false off screen).
bool Source(FfxInt32x2 p, out FfxInt32x2 cell, out FfxFloat32x2 flow)
{
    const FfxFloat32x2 mv = texelFetch(r_game_motion_vectors, p, 0).xy * ubo._MotionVectorScale;
    // Where it was, in motion vector pixels (the scale gives display pixels).
    const FfxFloat32x2 q  = FfxFloat32x2(p) + 0.5f + mv * FfxFloat32x2(ubo._MotionVectorDims) / FfxFloat32x2(ubo._DisplayDims);
    flow = -mv;
    cell = FfxInt32x2(floor(q * FfxFloat32x2(ubo._OutputDims) / FfxFloat32x2(ubo._MotionVectorDims)));
    return all(greaterThanEqual(q, FfxFloat32x2(0.0f))) && all(lessThan(q, FfxFloat32x2(ubo._MotionVectorDims))) &&
           all(lessThan(cell, ubo._OutputDims));
}

layout(set = 0, binding = 3, std430) coherent buffer ScatterKeys_t
{
    uint64_t keys[];  // optical flow resolution, row-major; ~0 where nothing landed
}
rw_scatter_keys;

void main()
{
    const FfxInt32x2 block = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(block * 2, ubo._MotionVectorDims)))
        return;
    const FfxInt32x2 p = block * 2;
    FfxInt32x2       cell;
    FfxFloat32x2     flow;
    if (Source(p, cell, flow))
        atomicMin(rw_scatter_keys.keys[cell.y * ubo._OutputDims.x + cell.x], (uint64_t(DepthKey(p)) << 32) | uint64_t(packHalf2x16(flow)));
}
