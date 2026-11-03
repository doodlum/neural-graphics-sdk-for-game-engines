// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Flow without the data graph optical flow engine, block matching on the image, step 2 of 4: a full search in the
// manner of FidelityFX FSR 3's optical flow (ffx_opticalflow_compute_optical_flow_v5.h: a block's luma against a
// search window in groupshared memory, four offsets per SAD step, a min reduction over the group), cut down for a
// device that has the game's motion vectors: one level at a quarter of the resolution instead of a 7-level pyramid
// from full resolution, centred on the motion vectors instead of on the coarser level's result. One group per 8x8
// texel TP1 block (32x32 pixels) searches +-8 texels (+-32 pixels) of TM1, in the motion vectors' direction
// (TP1 -> TM1), so it finds what moves without motion vectors, or with wrong ones, as far as 32 pixels from where
// they put it. ffx_opticalflow_refine.glsl takes the result as a candidate for each 8x8 pixel texel.

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_GOOGLE_include_directive : enable

#include "ffx_opticalflow_callbacks_glsl.h"

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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

layout(set = 0, binding = 1)        uniform utexture2D r_lumaq_tp1;  // quarter resolution
layout(set = 0, binding = 2)        uniform utexture2D r_lumaq_tm1;
layout(set = 0, binding = 3)        uniform texture2D r_game_motion_vectors;
layout(set = 0, binding = 4, rg16f) uniform writeonly image2D rw_block_flow;

#define BLOCK  8
#define RADIUS 8
#define WINDOW (BLOCK + 2 * RADIUS)  // 24 texels: 6 words a row

shared FfxUInt32 s_block[BLOCK][BLOCK / 4];
shared FfxUInt32 s_window[WINDOW][WINDOW / 4];
shared FfxUInt32 s_best;

// The 4 luma texels from (x, y) rightwards, packed (any alignment), clamped to the image.
FfxUInt32 Row(bool tp1, FfxInt32 x, FfxInt32 y)
{
    x = clamp(x, 0, ubo._LumaDims.x / 2 - 4);
    y = clamp(y, 0, ubo._LumaDims.y / 2 - 1);
    const FfxInt32x2 w  = FfxInt32x2(x >> 2, y);
    const FfxUInt32  s  = FfxUInt32(x & 3) * 8u;
    const FfxUInt32  w0 = tp1 ? texelFetch(r_lumaq_tp1, w, 0).r : texelFetch(r_lumaq_tm1, w, 0).r;
    const FfxUInt32  w1 = tp1 ? texelFetch(r_lumaq_tp1, w + FfxInt32x2(1, 0), 0).r : texelFetch(r_lumaq_tm1, w + FfxInt32x2(1, 0), 0).r;
    return s == 0u ? w0 : (w0 >> s) | (w1 << (32u - s));
}

// Packed 16-bit max/min/sub (v_pk_* on RDNA): the even and the odd bytes two at a time.
FfxUInt32 Sad(FfxUInt32 a, FfxUInt32 b)
{
    const u16vec2 ae = unpack16(a & 0x00ff00ffu), ao = unpack16((a >> 8) & 0x00ff00ffu);
    const u16vec2 be = unpack16(b & 0x00ff00ffu), bo = unpack16((b >> 8) & 0x00ff00ffu);
    const u16vec2 d  = (max(ae, be) - min(ae, be)) + (max(ao, bo) - min(ao, bo));
    return FfxUInt32(d.x) + FfxUInt32(d.y);
}

// The SADs of b against the 4 byte offsets of a0:a1 (FSR 3's QSad).
FfxUInt32x4 QSad(FfxUInt32 a0, FfxUInt32 a1, FfxUInt32 b)
{
    return FfxUInt32x4(Sad(a0, b), Sad((a0 >> 8) | (a1 << 24), b), Sad((a0 >> 16) | (a1 << 16), b), Sad((a0 >> 24) | (a1 << 8), b));
}

// SAD (at most 64 * 255) above, then the offset's distance from the centre (ties go to the smaller motion), then
// the offset itself.
FfxUInt32 Key(FfxUInt32 sad, FfxInt32x2 o)
{
    const FfxUInt32 dist = FfxUInt32(abs(o.x - RADIUS) + abs(o.y - RADIUS));
    return (sad << 15) | (dist << 10) | (FfxUInt32(o.y) << 5) | FfxUInt32(o.x);
}

void main()
{
    const FfxInt32x2 block = FfxInt32x2(gl_WorkGroupID.xy);
    const FfxInt32   i     = FfxInt32(gl_LocalInvocationIndex);
    const FfxInt32x2 org   = block * BLOCK;  // TP1 luma texel

    // Centred on the motion vector at the block's centre (luma texels: 4 pixels).
    const FfxInt32x2   mvPixel = clamp((org * 4 + BLOCK * 2) * ubo._MotionVectorDims / ubo._ColorDims, FfxInt32x2(0), ubo._MotionVectorDims - FfxInt32x2(1));
    const FfxFloat32x2 seedF   = texelFetch(r_game_motion_vectors, mvPixel, 0).xy * ubo._MotionVectorScale;
    const FfxInt32x2   seed    = FfxInt32x2(round(seedF * 0.25f));

    if (i < BLOCK * BLOCK / 4)
        s_block[i >> 1][i & 1] = Row(true, org.x + (i & 1) * 4, org.y + (i >> 1));
    const FfxInt32x2 base = org + seed - RADIUS;
    for (FfxInt32 k = i; k < WINDOW * (WINDOW / 4); k += 64)
        s_window[k / (WINDOW / 4)][k % (WINDOW / 4)] = Row(false, base.x + (k % (WINDOW / 4)) * 4, base.y + k / (WINDOW / 4));
    if (i == 0)
        s_best = 0xffffffffu;
    barrier();

    // 64 invocations x 4 horizontal offsets: the 16x16 offsets in [0, 2 * RADIUS).
    const FfxInt32 ox = (i & 3) * 4, oy = i >> 2;
    FfxUInt32x4    sad = FfxUInt32x4(0u);
    for (FfxInt32 r = 0; r < BLOCK; ++r)
    {
        const FfxInt32 w = ox >> 2;
        sad += QSad(s_window[oy + r][w], s_window[oy + r][w + 1], s_block[r][0]);
        sad += QSad(s_window[oy + r][w + 1], s_window[oy + r][w + 2], s_block[r][1]);
    }
    FfxUInt32 best = min(min(Key(sad.x, FfxInt32x2(ox, oy)), Key(sad.y, FfxInt32x2(ox + 1, oy))),
                         min(Key(sad.z, FfxInt32x2(ox + 2, oy)), Key(sad.w, FfxInt32x2(ox + 3, oy))));
    atomicMin(s_best, best);
    barrier();

    if (i == 0 && all(lessThan(block, ubo._BlockDims)))
    {
        const FfxInt32x2 o = FfxInt32x2(s_best & 31u, (s_best >> 5) & 31u) - RADIUS;
        imageStore(rw_block_flow, block, FfxFloat32x4(FfxFloat32x2(seed + o) * 4.0f, 0.0f, 0.0f));
    }
}
