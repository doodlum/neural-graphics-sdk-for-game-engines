// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Flow without the data graph optical flow engine, block matching on the image, step 1 of 4: TP1's luma at half
// (2x2 box) and a quarter (4x4) of the resolution, 8 bits, four horizontally adjacent texels packed per 32-bit word
// (lowest byte leftmost), the layout ffx_opticalflow_search.glsl (on the quarter) and _refine.glsl (on the half) read
// a 4-texel row from with two fetches. The next frame reads them as TM1. The search's coarse level reaches far
// cheaply; the refinement's finer one tells motions of a few pixels apart.

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_KHR_shader_subgroup_clustered : require
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

layout(set = 0, binding = 1)       uniform texture2D r_color_tp1;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage2D rw_luma_tp1;
layout(set = 0, binding = 3, r32ui) uniform writeonly uimage2D rw_lumaq_tp1;

FfxUInt32 Luma(FfxInt32x2 h)
{
    const FfxInt32x2   p = clamp(h * 2, FfxInt32x2(0), ubo._ColorDims - FfxInt32x2(2));
    const FfxFloat32x3 c = texelFetch(r_color_tp1, p, 0).rgb + texelFetch(r_color_tp1, p + FfxInt32x2(1, 0), 0).rgb +
                           texelFetch(r_color_tp1, p + FfxInt32x2(0, 1), 0).rgb + texelFetch(r_color_tp1, p + FfxInt32x2(1, 1), 0).rgb;
    return FfxUInt32(clamp(dot(c, FfxFloat32x3(0.299f, 0.587f, 0.114f)) * (255.0f / 4.0f) + 0.5f, 0.0f, 255.0f));
}

shared FfxUInt32 s_luma[8][8];

// One invocation per half-resolution texel; the 4 of a word are adjacent lanes (local x is innermost), packed by a
// clustered OR. The group's 8x8 texels make one quarter-resolution word per pair of rows.
void main()
{
    const FfxInt32x2 l = FfxInt32x2(gl_LocalInvocationID.xy);
    const FfxInt32x2 h = FfxInt32x2(gl_GlobalInvocationID.xy);
    const FfxUInt32  y = Luma(h);
    const FfxUInt32  v = subgroupClusteredOr(y << (FfxUInt32(h.x & 3) * 8u), 4);
    if ((h.x & 3) == 0 && h.x / 4 < ubo._PackedWidth && h.y < ubo._LumaDims.y)
        imageStore(rw_luma_tp1, FfxInt32x2(h.x / 4, h.y), FfxUInt32x4(v));
    s_luma[l.y][l.x] = y;
    barrier();
    if (l.x == 0 && (l.y & 1) == 0)
    {
        FfxUInt32 q = 0u;
        for (FfxInt32 i = 0; i < 4; ++i)
            q |= ((s_luma[l.y][2 * i] + s_luma[l.y][2 * i + 1] + s_luma[l.y + 1][2 * i] + s_luma[l.y + 1][2 * i + 1] + 2u) / 4u) << (8u * FfxUInt32(i));
        const FfxInt32x2 w = FfxInt32x2(gl_WorkGroupID.x, h.y / 2);
        if (w.x < ubo._PackedWidthQ && w.y < ubo._LumaDims.y / 2)
            imageStore(rw_lumaq_tp1, w, FfxUInt32x4(q));
    }
}
