// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// FFX_OF_IMAGE=fsr3, step 2 of 3, one dispatch per pyramid level from the coarsest: FidelityFX FSR 3's optical flow
// (SDK 2.4.0) as its own algorithm instead of the image flow's (ffx_opticalflow_flow_search.glsl). A vector per 8x8
// block of the level (not overlapping, not centred on a cell), TM1 -> TP1 on the TM1 grid (FSR 3 searches the
// previous frame for the current one's blocks; the interpolation's flow is the other way round, so the images swap):
//   - the seed: ffx_opticalflow_scale_optical_flow_advanced_v5.h, here at the start of the finer level's dispatch
//     instead of a pass of its own. Of the coarser level's four nearest blocks' vectors, the one whose 4x4 texels at
//     the coarser level (the block's area there) match best by SAD, doubled. Zero at the coarsest level.
//   - ffx_opticalflow_compute_optical_flow_v5.h: the block's SAD over offsets -8..+7 around the seed, ties to the
//     smaller |dy| then |dx| (FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS); at level 0 zero motion where it costs no more
//     (FFX_LOCAL_SEARCH_FALLBACK).
// Reads past the level's edges take its edge texel, as FSR 3's packed loads do; FSR 3's scene change detection is
// left out (a reset restarts the pyramid). RDNA has no msad4 (FSR 3 uses plain SAD there too): one invocation per
// block, the window a row at a time.

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

layout(set = 0, binding = 1)        uniform utexture2D r_pyramid_tm1;
layout(set = 0, binding = 2)        uniform utexture2D r_pyramid_tp1;
layout(set = 0, binding = 3)        uniform texture2D r_flow_levels;  // the coarser level's vectors (its texels)
layout(set = 0, binding = 4, rg16f) uniform writeonly image2D rw_flow_raw;

#define SR 8                    // FSR 3's SearchRadiusX/Y: offsets -8..+7
#define WINDOW (8 + 2 * SR - 1) // texels of a window row

// Level l's texel (x, y), the edge texel past its edges.
FfxUInt32 Texel(bool tp1, FfxInt32 l, FfxInt32 x, FfxInt32 y)
{
    const FfxInt32x2 d = _LevelDimsAll(l);
    x                  = clamp(x, 0, d.x - 1);
    const FfxInt32x2 w = FfxInt32x2(x >> 2, _LevelRows(l) + clamp(y, 0, d.y - 1));
    const FfxUInt32  v = tp1 ? texelFetch(r_pyramid_tp1, w, 0).r : texelFetch(r_pyramid_tm1, w, 0).r;
    return bitfieldExtract(v, (x & 3) * 8, 8);
}

// TM1's 8 texels of block row (x .. x + 7, y) at this level, packed.
FfxUInt32x2 BlockRow(FfxInt32 x, FfxInt32 y)
{
    FfxUInt32x2 r = FfxUInt32x2(0u);
    [[unroll]] for (FfxInt32 k = 0; k < 8; ++k)
        r[k >> 2] |= Texel(false, ubo._Level, x + k, y) << (8u * FfxUInt32(k & 3));
    return r;
}

// TP1's WINDOW texels of row y from x at this level.
void WindowRow(FfxInt32 x, FfxInt32 y, out FfxUInt32 t[WINDOW])
{
    if (x >= 0 && x + WINDOW <= ubo._LevelDims.x && y >= 0 && y < ubo._LevelDims.y)
    {
        // Inside the level: seven words, shifted to x.
        const FfxInt32 row = ubo._LevelRow + y;
        FfxUInt32      w[8];
        [[unroll]] for (FfxInt32 i = 0; i < 7; ++i)
            w[i] = texelFetch(r_pyramid_tp1, FfxInt32x2((x >> 2) + i, row), 0).r;
        w[7]              = 0u;
        const FfxUInt32 s = FfxUInt32(x & 3) * 8u;
        [[unroll]] for (FfxInt32 i = 0; i < 6; ++i)
        {
            const FfxUInt32 v = s == 0u ? w[i] : (w[i] >> s) | (w[i + 1] << (32u - s));
            [[unroll]] for (FfxInt32 k = 0; k < 4; ++k)
                if (i * 4 + k < WINDOW)
                    t[i * 4 + k] = bitfieldExtract(v, k * 8, 8);
        }
    }
    else
        [[unroll]] for (FfxInt32 i = 0; i < WINDOW; ++i)
            t[i] = Texel(true, ubo._Level, x + i, y);
}

FfxUInt32 Sad4(FfxUInt32 a, FfxUInt32 b)
{
    FfxUInt32 s = 0u;
    [[unroll]] for (FfxInt32 k = 0; k < 4; ++k)
        s += FfxUInt32(abs(FfxInt32(bitfieldExtract(a, k * 8, 8)) - FfxInt32(bitfieldExtract(b, k * 8, 8))));
    return s;
}

// The coarser level's vector of block p (zero off its blocks, as FSR 3's out-of-range loads).
FfxInt32x2 CoarseVector(FfxInt32x2 p)
{
    if (any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, ubo._CoarseDims)))
        return FfxInt32x2(0);
    return FfxInt32x2(texelFetch(r_flow_levels, FfxInt32x2(p.x, ubo._CoarseRow + p.y), 0).xy);
}

// FSR 3's ScaleOpticalFlowAdvanced for block b of this level.
FfxInt32x2 Seed(FfxInt32x2 b)
{
    const FfxInt32 lc   = ubo._Level + 1;
    FfxUInt32      best = 0xffffffffu;
    FfxInt32x2     seed = FfxInt32x2(0);
    [[unroll]] for (FfxInt32 n = 0; n < 4; ++n)
    {
        const FfxInt32x2 v = CoarseVector(b / 2 + FfxInt32x2((n % 2) - 1 + b.x % 2, (n / 2) - 1 + b.y % 2));
        FfxUInt32        sad = 0u;
        for (FfxInt32 y = 0; y < 4; ++y)
            [[unroll]] for (FfxInt32 x = 0; x < 4; ++x)
            {
                const FfxInt32x2 p = b * 4 + FfxInt32x2(x, y);
                sad += FfxUInt32(abs(FfxInt32(Texel(false, lc, p.x, p.y)) - FfxInt32(Texel(true, lc, p.x + v.x, p.y + v.y))));
            }
        if (sad < best)
            best = sad, seed = v * 2;
    }
    return seed;
}

void main()
{
    const FfxInt32x2 b = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(b, ubo._CellDims)))
        return;
    const FfxInt32x2 seed = ubo._CoarseDims.x > 0 ? Seed(b) : FfxInt32x2(0);
    const FfxInt32x2 org  = b * 8;

    FfxUInt32x2 block[8];
    [[unroll]] for (FfxInt32 r = 0; r < 8; ++r)
        block[r] = BlockRow(org.x, org.y + r);

    FfxUInt32 best = 0xffffffffu;  // SAD << 16 | |dy| << 12 | |dx| << 8 | oy << 4 | ox
    for (FfxInt32 oy = 0; oy < 2 * SR; ++oy)
    {
        FfxUInt32 sad[2 * SR];
        [[unroll]] for (FfxInt32 ox = 0; ox < 2 * SR; ++ox)
            sad[ox] = 0u;
        [[unroll]] for (FfxInt32 r = 0; r < 8; ++r)
        {
            FfxUInt32 t[WINDOW];
            WindowRow(org.x + seed.x - SR, org.y + seed.y - SR + oy + r, t);
            FfxUInt32 a[8];
            [[unroll]] for (FfxInt32 k = 0; k < 8; ++k)
                a[k] = bitfieldExtract(block[r][k >> 2], (k & 3) * 8, 8);
            [[unroll]] for (FfxInt32 ox = 0; ox < 2 * SR; ++ox)
                [[unroll]] for (FfxInt32 k = 0; k < 8; ++k)
                    sad[ox] += FfxUInt32(abs(FfxInt32(a[k]) - FfxInt32(t[ox + k])));
        }
        const FfxUInt32 ady = FfxUInt32(abs(oy - SR));
        [[unroll]] for (FfxInt32 ox = 0; ox < 2 * SR; ++ox)
            best = min(best, (sad[ox] << 16) | (ady << 12) | (FfxUInt32(abs(ox - SR)) << 8) | (FfxUInt32(oy) << 4) | FfxUInt32(ox));
    }
    FfxInt32x2 v = seed + FfxInt32x2(FfxInt32(best & 0xfu) - SR, FfxInt32((best >> 4) & 0xfu) - SR);

    if (ubo._Level == 0)
    {
        FfxUInt32 still = 0u;
        [[unroll]] for (FfxInt32 r = 0; r < 8; ++r)
        {
            FfxUInt32 t = 0u, u = 0u;
            [[unroll]] for (FfxInt32 k = 0; k < 4; ++k)
            {
                t |= Texel(true, 0, org.x + k, org.y + r) << (8u * FfxUInt32(k));
                u |= Texel(true, 0, org.x + 4 + k, org.y + r) << (8u * FfxUInt32(k));
            }
            still += Sad4(block[r].x, t) + Sad4(block[r].y, u);
        }
        if (still <= (best >> 16))
            v = FfxInt32x2(0);
    }
    imageStore(rw_flow_raw, FfxInt32x2(b.x, ubo._FlowRow + b.y), FfxFloat32x4(FfxFloat32x2(v), 0.0f, 0.0f));
}
