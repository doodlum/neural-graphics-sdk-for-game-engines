// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Image optical flow without the data graph optical flow engine, step 1 of 3: TP1's luma pyramid, as FidelityFX
// FSR 3's optical flow builds one (ffx_opticalflow_compute_luminance_pyramid.h, with SPD): level 0 at the colour's
// resolution, each next level a 2x2 box of the one before. 8 bits, four horizontally adjacent texels per 32-bit word
// (lowest byte leftmost), all levels one under the other in one image (level l from row _Levels[l].z). The next
// frame reads it as TM1 (ffx_opticalflow_flow_search.glsl). One dispatch: a group per 64x64 pixel tile, the coarser
// levels in groupshared memory (level 4 is 4 texels a side, one word).

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

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 1)        uniform texture2D r_color_tp1;
layout(set = 0, binding = 2, r32ui) uniform writeonly uimage2D rw_pyramid_tp1;

#define TILE 64

FfxUInt32 Luma(FfxInt32x2 p)
{
    const FfxFloat32x3 c = texelFetch(r_color_tp1, clamp(p, FfxInt32x2(0), ubo._SrcDims - FfxInt32x2(1)), 0).rgb;
    return FfxUInt32(clamp(dot(c, FfxFloat32x3(0.299f, 0.587f, 0.114f)) * 255.0f + 0.5f, 0.0f, 255.0f));
}

shared FfxUInt32 s_luma[TILE][TILE + 1];            // level 0, so that the colour reads are row-contiguous
shared FfxUInt32 s_level[2][TILE / 2][TILE / 2];  // ping-pong: level 1 (32x32), 2 (16x16), ...

// Level l's row (of the tile's n x n texels, l >= 1) from s_level[from]: one word per invocation.
void StoreLevel(FfxInt32 l, FfxInt32 from, FfxInt32 n, FfxInt32 i)
{
    const FfxInt32 words = n / 4;
    if (i >= n * words)
        return;
    const FfxInt32x2 t = FfxInt32x2((i % words) * 4, i / words);  // in the tile at level l
    const FfxInt32x2 g = FfxInt32x2(gl_WorkGroupID.xy) * n + t;
    if (g.x >= _LevelDimsAll(l).x || g.y >= _LevelDimsAll(l).y)
        return;
    const FfxUInt32 v = s_level[from][t.y][t.x] | (s_level[from][t.y][t.x + 1] << 8u) | (s_level[from][t.y][t.x + 2] << 16u) |
                        (s_level[from][t.y][t.x + 3] << 24u);
    imageStore(rw_pyramid_tp1, FfxInt32x2(g.x / 4, _LevelRows(l) + g.y), FfxUInt32x4(v));
}

void main()
{
    const FfxInt32x2 l = FfxInt32x2(gl_LocalInvocationID.xy);
    const FfxInt32   i = FfxInt32(gl_LocalInvocationIndex);
    const FfxInt32x2 p = FfxInt32x2(gl_WorkGroupID.xy) * TILE + l * 4;  // this invocation's 4x4 level 0 texels

    // Adjacent invocations read adjacent pixels; each then takes its 4x4 from groupshared memory.
    [[unroll]] for (FfxInt32 k = 0; k < 16; ++k)
    {
        const FfxInt32x2 t = FfxInt32x2(l.x + 16 * (k & 3), l.y + 16 * (k >> 2));
        s_luma[t.y][t.x]   = Luma(FfxInt32x2(gl_WorkGroupID.xy) * TILE + t);
    }
    barrier();
    FfxUInt32 v[4][4];
    [[unroll]] for (FfxInt32 y = 0; y < 4; ++y)
    {
        [[unroll]] for (FfxInt32 x = 0; x < 4; ++x)
            v[y][x] = s_luma[l.y * 4 + y][l.x * 4 + x];
        if (p.x < _LevelDimsAll(0).x && p.y + y < _LevelDimsAll(0).y)
            imageStore(rw_pyramid_tp1, FfxInt32x2(p.x / 4, p.y + y), FfxUInt32x4(v[y][0] | (v[y][1] << 8u) | (v[y][2] << 16u) | (v[y][3] << 24u)));
    }
    [[unroll]] for (FfxInt32 y = 0; y < 2; ++y)
        [[unroll]] for (FfxInt32 x = 0; x < 2; ++x)
            s_level[0][l.y * 2 + y][l.x * 2 + x] = (v[2 * y][2 * x] + v[2 * y][2 * x + 1] + v[2 * y + 1][2 * x] + v[2 * y + 1][2 * x + 1] + 2u) >> 2u;
    barrier();
    StoreLevel(1, 0, TILE / 2, i);

    FfxInt32 from = 0;
    for (FfxInt32 level = 2, n = TILE / 4; level < ubo._LevelCount; ++level, n >>= 1)
    {
        if (i < n * n)
        {
            const FfxInt32x2 t = FfxInt32x2(i % n, i / n) * 2;
            s_level[from ^ 1][t.y / 2][t.x / 2] =
                (s_level[from][t.y][t.x] + s_level[from][t.y][t.x + 1] + s_level[from][t.y + 1][t.x] + s_level[from][t.y + 1][t.x + 1] + 2u) >> 2u;
        }
        from ^= 1;
        barrier();
        StoreLevel(level, from, n, i);
    }
}
