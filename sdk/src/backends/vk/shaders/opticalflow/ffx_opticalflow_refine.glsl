// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Flow without the data graph optical flow engine, block matching on the image, step 3 of 4: where the image
// contradicts the game's motion vectors (particles, transparencies, objects drawn without motion vectors or with
// wrong ones), a correction for each 8x8 pixel TP1 texel, which ffx_opticalflow_apply.glsl writes into a copy of
// the motion vectors for the interpolation's TP1 warp and, scattered (ffx_opticalflow_scatter_depth.glsl /
// _flow.glsl), for its TM1 flow: it needs both to agree with the image. Candidate-based block matching in the style
// of 3D recursive search (de Haan et al., "True-motion estimation with 3-D recursive search block matching", IEEE
// TCSVT 1993): each texel matches its 8x8 pixels (4x4 half-resolution luma texels) against TM1 at the game's
// motion vector, its own and the nearest neighbouring blocks' full search results (ffx_opticalflow_search.glsl) and
// zero, then refines the best with a 2-step search (+-4, +-2 pixels). Pixel accuracy is not the aim: the
// interpolation tolerates +-8 pixels of error. Corrections are rare by design: the best candidate must match
// clearly better than the motion vector (below _MatchRatio percent of its SAD, after a penalty), and the texel must
// not have been hidden in TM1 where the motion vector points (TM1 depth nearer there): content just disoccluded has
// no counterpart in TM1, so anything that happens to match better there is wrong.

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

layout(set = 0, binding = 1)        uniform utexture2D r_luma_tp1;
layout(set = 0, binding = 2)        uniform utexture2D r_luma_tm1;
layout(set = 0, binding = 3)          uniform texture2D r_game_motion_vectors;
layout(set = 0, binding = 4)          uniform texture2D r_block_flow;  // display pixels TP1 -> TM1
layout(set = 0, binding = 5)          uniform texture2D r_depth;
layout(set = 0, binding = 6)          uniform texture2D r_depth_tm1;
layout(set = 0, binding = 7, rgba16f) uniform writeonly image2D rw_motion_fix;  // xy display pixels TP1 -> TM1, z 1 where corrected, w depth

// The 4 luma texels from (x, y) rightwards, packed.
FfxUInt32 Row(bool tp1, FfxInt32 x, FfxInt32 y)
{
    x = clamp(x, 0, ubo._LumaDims.x - 4);
    y = clamp(y, 0, ubo._LumaDims.y - 1);
    const FfxInt32x2 w  = FfxInt32x2(x >> 2, y);
    const FfxUInt32  s  = FfxUInt32(x & 3) * 8u;
    const FfxUInt32  w0 = tp1 ? texelFetch(r_luma_tp1, w, 0).r : texelFetch(r_luma_tm1, w, 0).r;
    const FfxUInt32  w1 = tp1 ? texelFetch(r_luma_tp1, w + FfxInt32x2(1, 0), 0).r : texelFetch(r_luma_tm1, w + FfxInt32x2(1, 0), 0).r;
    return s == 0u ? w0 : (w0 >> s) | (w1 << (32u - s));
}

// Packed 16-bit max/min/sub (v_pk_* on RDNA): the even and the odd bytes two at a time.
FfxUInt32 Sad4(FfxUInt32 a, FfxUInt32 b)
{
    const u16vec2 ae = unpack16(a & 0x00ff00ffu), ao = unpack16((a >> 8) & 0x00ff00ffu);
    const u16vec2 be = unpack16(b & 0x00ff00ffu), bo = unpack16((b >> 8) & 0x00ff00ffu);
    const u16vec2 d  = (max(ae, be) - min(ae, be)) + (max(ao, bo) - min(ao, bo));
    return FfxUInt32(d.x) + FfxUInt32(d.y);
}

FfxUInt32    g_template[4];
FfxInt32x2   g_origin;  // the block's top left luma texel
FfxUInt32    g_best;
FfxFloat32x2 g_bestFlow;

FfxUInt32 Cost(FfxFloat32x2 flow)
{
    const FfxInt32x2 d   = FfxInt32x2(round(flow * 0.5f));
    FfxUInt32        sad = 0u;
    for (FfxInt32 r = 0; r < 4; ++r)
        sad += Sad4(g_template[r], Row(false, g_origin.x + d.x, g_origin.y + d.y + r));
    return sad;
}

void Try(FfxFloat32x2 flow, FfxUInt32 penalty)
{
    const FfxUInt32 c = Cost(flow) + penalty;
    if (c < g_best)
    {
        g_best     = c;
        g_bestFlow = flow;
    }
}

FfxFloat32 Depth(bool tm1, FfxFloat32x2 pixel)
{
    const FfxInt32x2 dims = tm1 ? textureSize(r_depth_tm1, 0) : textureSize(r_depth, 0);
    const FfxInt32x2 p    = clamp(FfxInt32x2(pixel * FfxFloat32x2(dims) / FfxFloat32x2(ubo._ColorDims)), FfxInt32x2(0), dims - FfxInt32x2(1));
    const FfxFloat32 z    = tm1 ? texelFetch(r_depth_tm1, p, 0).r : texelFetch(r_depth, p, 0).r;
    return ubo._DepthInverted != 0 ? 1.0f - z : z;  // nearer is smaller
}

void main()
{
    const FfxInt32x2 t = FfxInt32x2(gl_GlobalInvocationID.xy);  // 8x8 display pixels
    if (any(greaterThanEqual(t * 8, ubo._ColorDims)))
        return;

    g_origin = t * 4;
    for (FfxInt32 r = 0; r < 4; ++r)
        g_template[r] = Row(true, g_origin.x, g_origin.y + r);

    const FfxFloat32x2 centre = FfxFloat32x2(t * 8 + 4);
    const FfxInt32x2   mc     = clamp(FfxInt32x2(centre * FfxFloat32x2(ubo._MotionVectorDims) / FfxFloat32x2(ubo._ColorDims)),
                                      FfxInt32x2(0), ubo._MotionVectorDims - FfxInt32x2(1));
    const FfxFloat32x2 game   = texelFetch(r_game_motion_vectors, mc, 0).xy * ubo._MotionVectorScale;

    // The best other candidate (see the top).
    const FfxUInt32 gameCost = Cost(game);
    g_best     = 0xffffffffu;
    g_bestFlow = game;
    const FfxInt32x2 b  = min(t / 4, ubo._BlockDims - FfxInt32x2(1));
    const FfxInt32x2 nb = clamp(b + FfxInt32x2(greaterThanEqual(t & 3, FfxInt32x2(2))) * 2 - 1, FfxInt32x2(0), ubo._BlockDims - FfxInt32x2(1));
    Try(texelFetch(r_block_flow, b, 0).xy, 0u);
    Try(texelFetch(r_block_flow, FfxInt32x2(nb.x, b.y), 0).xy, 0u);
    Try(texelFetch(r_block_flow, FfxInt32x2(b.x, nb.y), 0).xy, 0u);
    Try(FfxFloat32x2(0.0f), 0u);
    for (FfxInt32 step = 2; step >= 1; step >>= 1)
    {
        const FfxFloat32x2 c = g_bestFlow;
        const FfxFloat32   s = FfxFloat32(step * 2);
        Try(c + FfxFloat32x2(s, 0.0f), ubo._PenaltyStep);
        Try(c + FfxFloat32x2(-s, 0.0f), ubo._PenaltyStep);
        Try(c + FfxFloat32x2(0.0f, s), ubo._PenaltyStep);
        Try(c + FfxFloat32x2(0.0f, -s), ubo._PenaltyStep);
    }

    const bool clearer    = (g_best + ubo._PenaltyCandidate) * 100u < gameCost * FfxUInt32(ubo._MatchRatio);
    const FfxFloat32 z          = Depth(false, centre);
    const bool       hiddenInM1 = Depth(true, centre + game) < z - 1e-3f;
    const bool       fix        = clearer && !hiddenInM1;
    imageStore(rw_motion_fix, t, FfxFloat32x4(g_bestFlow, fix ? 1.0f : 0.0f, z));
}
