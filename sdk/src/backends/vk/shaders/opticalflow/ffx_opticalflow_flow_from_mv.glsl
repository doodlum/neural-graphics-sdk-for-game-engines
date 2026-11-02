// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Flow without the data graph optical flow engine (hints only): the forward flow TM1 -> TP1 on the TM1 grid is the
// previous frame's game motion vectors, negated (constant velocity), in display pixels at the optical flow
// resolution. Each invocation writes the stored flow of the previous frame and stores this frame's for the next one.
// The camera-only MV hints miss object motion: NFRU's TM1 warp then leaves moving objects where they were.

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : enable

#include "ffx_opticalflow_callbacks_glsl.h"

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Uniforms {
    FfxFloat32x2 _MotionVectorScale;   // raw motion vector -> display pixels (TP1 -> TM1 convention)
    FfxInt32x2   _OutputDims;          // optical flow resolution
    FfxInt32x2   _MotionVectorDims;
    FfxInt32     _Reset;               // no previous frame: zero flow
    FfxInt32     _Pad;
}ubo;

layout(set = 0, binding = 1)        uniform texture2D r_game_motion_vectors;           // display resolution | RG
layout(set = 0, binding = 2)        uniform texture2D r_flow_store_tm1;                // optical flow resolution | RG16F
layout(set = 0, binding = 3, rg16f) uniform writeonly image2D rw_flow;                 // optical flow resolution | RG16F
layout(set = 0, binding = 4, rg16f) uniform writeonly image2D rw_flow_store;           // optical flow resolution | RG16F

void main()
{
    const FfxInt32x2 p = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, ubo._OutputDims)))
        return;

    imageStore(rw_flow, p, ubo._Reset != 0 ? FfxFloat32x4(0.0f) : FfxFloat32x4(texelFetch(r_flow_store_tm1, p, 0).xy, 0.0f, 0.0f));

    // This frame's motion at the centre of the block, as a forward flow for the next frame.
    const FfxInt32x2   mvPixel = clamp((p * ubo._MotionVectorDims + ubo._MotionVectorDims / 2) / ubo._OutputDims, FfxInt32x2(0),
                                       ubo._MotionVectorDims - FfxInt32x2(1));
    const FfxFloat32x2 mv      = texelFetch(r_game_motion_vectors, mvPixel, 0).xy * ubo._MotionVectorScale;
    imageStore(rw_flow_store, p, FfxFloat32x4(-mv, 0.0f, 0.0f));
}
