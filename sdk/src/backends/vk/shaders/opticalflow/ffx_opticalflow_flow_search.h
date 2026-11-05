// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// The image flow's search (see ffx_opticalflow_flow_search.glsl), for a full search of +-RADIUS texels.

#include "ffx_opticalflow_callbacks_glsl.h"
#include "ffx_opticalflow_image_common.h"

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 1)        uniform utexture2D r_pyramid_tm1;
layout(set = 0, binding = 2)        uniform utexture2D r_pyramid_tp1;
layout(set = 0, binding = 3)        uniform texture2D r_flow_levels;  // the coarser level's vectors (texels of it)
layout(set = 0, binding = 4)        uniform texture2D r_flow_hint;    // level 0 cells, colour pixels
layout(set = 0, binding = 5, rg16f) uniform writeonly image2D rw_flow_raw;

#define WINDOW (8 + 2 * RADIUS)  // texels a side; 16 bytes a row hold it

// The texels from byte s of lo:hi (s a constant after unrolling).
FfxUInt32 Funnel(FfxUInt32 lo, FfxUInt32 hi, FfxUInt32 s)
{
    return s == 0u ? lo : (lo >> (8u * s)) | (hi << (32u - 8u * s));
}

// 8 (or 16) texels of this level from (x, y), x within the row's words, y clamped to the level.
FfxUInt32x2 Load8(bool tp1, FfxInt32 x, FfxInt32 y)
{
    const FfxInt32x2 w  = FfxInt32x2(x >> 2, ubo._LevelRow + clamp(y, 0, ubo._LevelDims.y - 1));
    const FfxUInt32  w0 = tp1 ? texelFetch(r_pyramid_tp1, w, 0).r : texelFetch(r_pyramid_tm1, w, 0).r;
    const FfxUInt32  w1 = tp1 ? texelFetch(r_pyramid_tp1, w + FfxInt32x2(1, 0), 0).r : texelFetch(r_pyramid_tm1, w + FfxInt32x2(1, 0), 0).r;
    const FfxUInt32  w2 = tp1 ? texelFetch(r_pyramid_tp1, w + FfxInt32x2(2, 0), 0).r : texelFetch(r_pyramid_tm1, w + FfxInt32x2(2, 0), 0).r;
    const FfxUInt32  s  = FfxUInt32(x & 3) * 8u;
    return FfxUInt32x2(FfxUInt32(pack64(FfxUInt32x2(w0, w1)) >> s), FfxUInt32(pack64(FfxUInt32x2(w1, w2)) >> s));
}
FfxUInt32x4 Load16(FfxInt32 x, FfxInt32 y)
{
    const FfxInt32x2 w = FfxInt32x2(x >> 2, ubo._LevelRow + clamp(y, 0, ubo._LevelDims.y - 1));
    FfxUInt32        v[5];
    [[unroll]] for (FfxInt32 i = 0; i < 5; ++i)
        v[i] = texelFetch(r_pyramid_tp1, w + FfxInt32x2(i, 0), 0).r;
    const FfxUInt32 s = FfxUInt32(x & 3) * 8u;
    return FfxUInt32x4(FfxUInt32(pack64(FfxUInt32x2(v[0], v[1])) >> s), FfxUInt32(pack64(FfxUInt32x2(v[1], v[2])) >> s),
                       FfxUInt32(pack64(FfxUInt32x2(v[2], v[3])) >> s), FfxUInt32(pack64(FfxUInt32x2(v[3], v[4])) >> s));
}

FfxUInt32 Dot(FfxUInt32x2 a, FfxUInt32x2 b, FfxUInt32 acc)
{
    return dotPacked4x8AccSatEXT(a.y, b.y, dotPacked4x8AccSatEXT(a.x, b.x, acc));
}

// Window x start: the block's offset by v - RADIUS, kept inside the level (the offsets then not centred on v).
FfxInt32 WindowX(FfxInt32 x)
{
    return clamp(x, 0, max(ubo._LevelDims.x - WINDOW, 0));
}

// A candidate's block inside the level: off it the reads are clamped and would match something else. Rows clamp one
// by one (the edge row repeated), so half the block may be over the top or bottom edge.
bool Inside(FfxInt32x2 org, FfxInt32x2 v)
{
    const FfxInt32x2 p = org + v;
    return p.x >= 0 && p.x + 8 <= ubo._LevelDims.x && p.y >= -4 && p.y + 4 <= ubo._LevelDims.y;
}

FfxUInt32 L1(FfxInt32x2 d)
{
    return FfxUInt32(abs(d.x) + abs(d.y));
}

// The hint's own sub-texel vector where the search agrees with it (a camera's motion, say).
void Store(FfxInt32x2 c, FfxInt32x2 v, FfxInt32x2 hint, FfxFloat32x2 hintF)
{
    imageStore(rw_flow_raw, FfxInt32x2(c.x, ubo._FlowRow + c.y), FfxFloat32x4(all(equal(v, hint)) ? hintF : FfxFloat32x2(v), 0.0f, 0.0f));
}

void main()
{
    const FfxInt32x2 c = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(c, ubo._CellDims)))
        return;

    // The TM1 block: 8x8 texels centred on the cell (kept inside the level).
    const FfxInt32x2 org = clamp(c * 4 - 2, FfxInt32x2(0), max(ubo._LevelDims - 8, FfxInt32x2(0)));
    FfxUInt32x2      a[8];
    FfxUInt32        aa = 0u;
    [[unroll]] for (FfxInt32 i = 0; i < 8; ++i)
    {
        a[i] = Load8(false, org.x, org.y + i);
        aa   = Dot(a[i], a[i], aa);
    }

    // The hint flow at the cell's centre, in this level's texels.
    const FfxInt32     scale = 1 << ubo._Level;
    const FfxInt32x2   hc    = min(c * scale + (scale >> 1), ubo._HintDims - FfxInt32x2(1));
    const FfxFloat32x2 hintF = texelFetch(r_flow_hint, hc, 0).xy / (ubo._OutScale * FfxFloat32(scale));
    const FfxInt32x2   hint  = FfxInt32x2(round(hintF));

    // Seeds: the hint, zero (not at level 0, which refines level 1), the coarser level's four nearest cells; each
    // distinct one tried.
#ifdef FINE_LEVEL
#define SEED_FIRST_COARSE 1
#else
#define SEED_FIRST_COARSE 2
#endif
    FfxInt32x2 seeds[SEED_FIRST_COARSE + 4];
    FfxInt32   seedCount = SEED_FIRST_COARSE;
    seeds[0]             = hint;
#ifndef FINE_LEVEL
    seeds[1] = FfxInt32x2(0);
#endif
    if (ubo._CoarseDims.x > 0)
    {
        const FfxInt32x2 cc = c >> 1, o = (c & 1) * 2 - 1;
        [[unroll]] for (FfxInt32 k = 0; k < 4; ++k)
        {
            const FfxInt32x2 p = clamp(cc + FfxInt32x2(k & 1, k >> 1) * o, FfxInt32x2(0), ubo._CoarseDims - FfxInt32x2(1));
            seeds[SEED_FIRST_COARSE + k] = FfxInt32x2(round(texelFetch(r_flow_levels, FfxInt32x2(p.x, ubo._CoarseRow + p.y), 0).xy)) * 2;
        }
        seedCount = SEED_FIRST_COARSE + 4;
    }
    // Where the hint leaves the image (what scrolls out of view), nothing tells it wrong: it stands. So it does at its
    // own edges (an object's, from the motion vectors and depth): there they are exact to the pixel and a block is
    // not, and what the object covers in TP1 has no match at all (it would take the object's motion).
    bool edge = false;
    [[unroll]] for (FfxInt32 k = 0; k < 8; ++k)
    {
        const FfxInt32   r  = (k >> 2) + 1, s = (k & 1) * 2 - 1;
        const FfxInt32x2 hn = clamp(hc + ((k & 2) == 0 ? FfxInt32x2(s * r, 0) : FfxInt32x2(0, s * r)) * scale, FfxInt32x2(0), ubo._HintDims - FfxInt32x2(1));
        const FfxFloat32x2 n = texelFetch(r_flow_hint, hn, 0).xy / (ubo._OutScale * FfxFloat32(scale));
        edge = edge || any(greaterThan(abs(n - hintF), FfxFloat32x2(1.0f)));
    }
    if (edge || !Inside(org, hint))
    {
        Store(c, hint, hint, hintF);
        return;
    }
    FfxUInt32  best     = 0xffffffffu;
    FfxInt32x2 bestV    = hint;
    FfxInt32   distinct = 0;
    [[unroll]] for (FfxInt32 k = 0; k < SEED_FIRST_COARSE + 4; ++k)
    {
        bool tried = k >= seedCount;
        [[unroll]] for (FfxInt32 m = 0; m < k; ++m)
            tried = tried || all(equal(seeds[m], seeds[k]));
        const FfxInt32x2 v = seeds[k];
        if (tried || !Inside(org, v))
            continue;
        ++distinct;
        FfxUInt32 ab = 0u, bb = 0u;
        [[unroll]] for (FfxInt32 i = 0; i < 8; ++i)
        {
            const FfxUInt32x2 b = Load8(true, org.x + v.x, org.y + v.y + i);
            ab = Dot(a[i], b, ab), bb = Dot(b, b, bb);
        }
        const FfxUInt32 cost = aa + bb - 2u * ab + ubo._LambdaHint * L1(v - hint);
        if (cost < best)
            best = cost, bestV = v;
    }

    // Settled (3-D recursive search's convergence, most of a frame): the seeds agree and match; nothing to search.
    if (distinct == 1 && best <= ubo._SettledCost)
    {
        Store(c, bestV, hint, hintF);
        return;
    }

#if RADIUS > 0
    // The full search around the best seed: the window's 16-byte rows; per column offset, the rows' b^2, then the
    // SSD of each row offset.
    const FfxInt32x2 seed = bestV;
    const FfxInt32x2 ws   = FfxInt32x2(WindowX(org.x + seed.x - RADIUS), org.y + seed.y - RADIUS);
    FfxUInt32x4      win[WINDOW];
    [[unroll]] for (FfxInt32 r = 0; r < WINDOW; ++r)
        win[r] = Load16(ws.x, ws.y + r);
    [[unroll]] for (FfxInt32 j = 0; j <= 2 * RADIUS; ++j)
    {
        FfxUInt32x2 b[WINDOW];
        FfxUInt32   bb[WINDOW];
        [[unroll]] for (FfxInt32 r = 0; r < WINDOW; ++r)
        {
            const FfxUInt32 q = FfxUInt32(j) >> 2u, s = FfxUInt32(j) & 3u;
            b[r]  = FfxUInt32x2(Funnel(win[r][q], win[r][q + 1u], s), Funnel(win[r][q + 1u], win[r][q + 2u], s));
            bb[r] = Dot(b[r], b[r], 0u);
        }
        [[unroll]] for (FfxInt32 k = 0; k <= 2 * RADIUS; ++k)
        {
            FfxUInt32 ab = 0u, sb = 0u;
            [[unroll]] for (FfxInt32 i = 0; i < 8; ++i)
                ab = Dot(a[i], b[k + i], ab), sb += bb[k + i];
            const FfxInt32x2 v    = ws + FfxInt32x2(j, k) - org;
            const FfxUInt32  cost = aa + sb - 2u * ab + ubo._LambdaHint * L1(v - hint) + ubo._LambdaSeed * L1(v - seed);
            if (cost < best && Inside(org, v))
                best = cost, bestV = v;
        }
    }
#endif
    Store(c, bestV, hint, hintF);
}
