// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// FFX_OF_IMAGE=arm (ffx_opticalflow_arm_common.h), per level, step 2 of 4: the layer's block_match_of.comp and
// subpixel_me.comp in one pass. Per texel of the level, TM1's 5x5 block against the warped TP1 at the 49 offsets of
// +-3 in the Motion Engine's spiral order (equal costs to the earlier, zero off the level); then a Lucas-Kanade step
// in half precision on the block's border ring (24 texels) around the match, dropped where 1 texel or more, and the
// coarser level's flow added. Also keeps the match's cost, which the last level compares with the hint's.

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

#include "ffx_opticalflow_arm_common.h"

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 1)        uniform texture2D r_pyramid_tm1;  // template
layout(set = 0, binding = 2)        uniform texture2D r_arm_warp;     // search
layout(set = 0, binding = 3)        uniform texture2D r_arm_flow;     // the coarser level's filtered flow
layout(set = 0, binding = 4, rg16f) uniform writeonly image2D rw_arm_sub;
layout(set = 0, binding = 5, r32ui) uniform writeonly uimage2D rw_arm_match;  // the match's cost

#define SR 3
#define SW (2 * SR + 1)
const FfxInt32 kSpiral[SW * SW] = {24, 30, 31, 32, 25, 18, 17, 16, 23, 36, 37, 38, 39, 40, 33, 26, 19,
                                   12, 11, 10, 9,  8,  15, 22, 29, 42, 43, 44, 45, 46, 47, 48, 41, 34,
                                   27, 20, 13, 6,  5,  4,  3,  2,  1,  0,  7,  14, 21, 28, 35};

float16_t TemplateClamp(FfxInt32x2 p)
{
    return float16_t(ArmTexel(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, p));
}

float16_t SearchZero(FfxInt32x2 p)
{
    if (any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, ubo._LevelDims)))
        return float16_t(0.0);
    return float16_t(texelFetch(r_arm_warp, p, 0).x);
}

void main()
{
    const FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, ubo._LevelDims)))
        return;

    FfxUInt32 tmpl[25];
    [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
        tmpl[i] = ArmU8Zero(r_pyramid_tm1, ubo._LevelRow, ubo._LevelDims, pos + FfxInt32x2(i % 5, i / 5) - ARM_BLOCK_R);
    FfxUInt32 win[(5 + 2 * SR) * (5 + 2 * SR)];
    [[unroll]] for (FfxInt32 i = 0; i < (5 + 2 * SR) * (5 + 2 * SR); ++i)
    {
        const FfxInt32x2 p = pos + FfxInt32x2(i % (5 + 2 * SR), i / (5 + 2 * SR)) - ARM_BLOCK_R - SR;
        win[i]             = any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, ubo._LevelDims)) ? 0u : ArmU8(texelFetch(r_arm_warp, p, 0).x);
    }
    FfxUInt32  best    = 0xffffffffu;
    FfxInt32x2 intFlow = FfxInt32x2(0);
    [[unroll]] for (FfxInt32 si = 0; si < SW * SW; ++si)
    {
        const FfxInt32 sx = kSpiral[si] % SW, sy = kSpiral[si] / SW;  // window offset of the block
        FfxUInt32      d  = 0u;
        [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
            d += FfxUInt32(abs(FfxInt32(win[(sy + i / 5) * (5 + 2 * SR) + sx + i % 5]) - FfxInt32(tmpl[i])));
        if (d < best)
            best = d, intFlow = FfxInt32x2(sx - SR, sy - SR);
    }
    imageStore(rw_arm_match, pos, FfxUInt32x4(best, 0u, 0u, 0u));

    // subpixel_me.comp
    precise float16_t a = float16_t(0.0), b = float16_t(0.0), d = float16_t(0.0), p = float16_t(0.0), q = float16_t(0.0);
    [[unroll]] for (FfxInt32 i = 0; i < 25; ++i)
    {
        const FfxInt32x2 o = FfxInt32x2(i % 5, i / 5) - ARM_BLOCK_R;
        if (abs(o.x) != ARM_BLOCK_R && abs(o.y) != ARM_BLOCK_R)
            continue;  // the ring only
        const FfxInt32x2   bpos = pos + o;
        precise float16_t f    = TemplateClamp(bpos);
        precise float16_t dfx  = (TemplateClamp(bpos + FfxInt32x2(1, 0)) - TemplateClamp(bpos - FfxInt32x2(1, 0))) * float16_t(0.5);
        precise float16_t dfy  = (TemplateClamp(bpos + FfxInt32x2(0, 1)) - TemplateClamp(bpos - FfxInt32x2(0, 1))) * float16_t(0.5);
        const float16_t   in_  = any(lessThan(bpos, FfxInt32x2(0))) || any(greaterThanEqual(bpos, ubo._LevelDims)) ? float16_t(0.0) : float16_t(1.0);
        f *= in_, dfx *= in_, dfy *= in_;
        a += dfx * dfx;
        b += dfx * dfy;
        d += dfy * dfy;
        precise float16_t z = SearchZero(bpos + intFlow) - f;
        p += z * dfx;
        q += z * dfy;
    }
    precise FfxFloat32   detA = FfxFloat32(a) * FfxFloat32(d) - FfxFloat32(b) * FfxFloat32(b);
    precise FfxFloat32x2 num  = FfxFloat32x2(FfxFloat32(d) * FfxFloat32(p) - FfxFloat32(b) * FfxFloat32(q),
                                             FfxFloat32(a) * FfxFloat32(q) - FfxFloat32(b) * FfxFloat32(p));
    FfxFloat32x2 sub = FfxFloat32x2(0.0f);
    if (detA > 1e-7f)
    {
        sub = num / detA;
        sub = mix(sub, FfxFloat32x2(0.0f), greaterThanEqual(abs(sub), FfxFloat32x2(1.0f)));
    }
    f16vec2 flow = f16vec2(intFlow) - f16vec2(sub);
    if (ubo._CoarseDims.x > 0)  // the upscaled flow mv_process_and_warp.comp keeps
        flow += f16vec2(ArmBilinear(r_arm_flow, 0, ubo._CoarseDims, (FfxFloat32x2(pos) + 0.5f) * 0.5f).xy * 2.0f);
    imageStore(rw_arm_sub, pos, FfxFloat32x4(FfxFloat32x2(flow), 0.0f, 0.0f));
}
