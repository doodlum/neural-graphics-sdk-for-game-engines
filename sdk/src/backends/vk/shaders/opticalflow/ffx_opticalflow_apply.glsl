// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Flow without the data graph optical flow engine, block matching on the image, step 4 of 4: the game's motion
// vectors with ffx_opticalflow_refine.glsl's corrections written in, in their own layout and scale, for the
// interpolation and the scatter, one invocation per motion vector so the reads and writes coalesce. A correction
// covers 8x8 display pixels: choosing per pixel by depth among the neighbouring texels (joint bilateral
// upsampling) measured no better in the ground-truth test, at 1.7x the cost.

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

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform Uniforms {
    FfxInt32x2   _OutputDims;         // optical flow resolution
    FfxInt32x2   _LumaDims;           // half the colour resolution (the search's level: a quarter)
    FfxInt32x2   _ColorDims;
    FfxInt32     _PackedWidth;        // words per luma row
    FfxUInt32    _PenaltyCandidate;   // added to the block SAD (16 texels, 0..4080) of a candidate not from the motion vectors
    FfxUInt32    _PenaltyStep;        // added again for a refinement step
    FfxInt32     _MatchRatio;         // a correction needs a SAD below this percentage of the motion vector's
    FfxInt32x2   _BlockDims;          // search blocks (8x8 luma texels each)
    FfxInt32     _DepthInverted;
    FfxInt32     _PackedWidthQ;       // words per row of the search's level
    FfxInt32x2   _MotionVectorDims;
    FfxFloat32x2 _MotionVectorScale;  // raw motion vector -> display pixels (from TP1 to where the pixel was in TM1)
}ubo;

layout(set = 0, binding = 1)        uniform texture2D r_game_motion_vectors;
layout(set = 0, binding = 2)        uniform texture2D r_motion_fix;
layout(set = 0, binding = 3, rg16f) uniform writeonly image2D rw_fixed_motion_vectors;

void main()
{
    const FfxInt32x2 p = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, ubo._MotionVectorDims)))
        return;
    const FfxInt32x2   t   = min((p * ubo._ColorDims / ubo._MotionVectorDims) / 8, textureSize(r_motion_fix, 0) - FfxInt32x2(1));
    const FfxFloat32x4 fix = texelFetch(r_motion_fix, t, 0);
    imageStore(rw_fixed_motion_vectors, p,
               FfxFloat32x4(fix.z > 0.5f ? fix.xy / ubo._MotionVectorScale : texelFetch(r_game_motion_vectors, p, 0).xy, 0.0f, 0.0f));
}
