// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// FFX_OF_IMAGE=arm (ffx_opticalflow_arm_common.h), TP1's luma pyramid, a dispatch per level: level 1 from the colour
// (the layer's rgb_to_y.comp: Y = (R + 2G + B) / 4 over a 2x2 box), levels 2 to 5 from the level before (downsample
// .comp: a [1 3 3 1] x [1 3 3 1] blur at 2p - 1, its last row and column repeated where the level was padded to even).
// The next frame reads it as TM1.

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

layout(set = 0, binding = 1)     uniform texture2D r_color_tp1;
layout(set = 0, binding = 2, r8) uniform image2D   rw_pyramid_tp1;

FfxFloat32x3 Rgb(FfxInt32x2 p)
{
    return texelFetch(r_color_tp1, clamp(p, FfxInt32x2(0), ubo._SrcDims - FfxInt32x2(1)), 0).rgb;
}

FfxUInt32 FromColour(FfxInt32x2 base)
{
    if (ubo._Unorm8 != 0)
    {
        FfxUInt32x3 sum = FfxUInt32x3(0u);
        [[unroll]] for (FfxInt32 i = 0; i < 4; ++i)
            sum += FfxUInt32x3(floor(Rgb(base + FfxInt32x2(i & 1, i >> 1)) * 255.0f + 0.5f));
        return (sum.r + 2u * sum.g + sum.b + 8u) / 16u;
    }
    FfxUInt32x3 sum = FfxUInt32x3(0u);
    [[unroll]] for (FfxInt32 i = 0; i < 4; ++i)
        sum += FfxUInt32x3(floor(clamp(Rgb(base + FfxInt32x2(i & 1, i >> 1)), 0.0f, 1.0f) * 65535.0f + 0.5f));
    const FfxUInt32 weighted = sum.r + 2u * sum.g + sum.b;
    return (weighted * 255u + 524280u) / 1048560u;
}

FfxUInt32 Finer(FfxInt32x2 p)  // the level before, clamped
{
    p = clamp(p, FfxInt32x2(0), ubo._CoarseDims - FfxInt32x2(1));
    return ArmU8(imageLoad(rw_pyramid_tp1, FfxInt32x2(p.x, ubo._CoarseRow + p.y)).x);
}

void main()
{
    const FfxInt32x2 pos = FfxInt32x2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pos, ubo._LevelDims)))
        return;
    FfxUInt32 v;
    if (ubo._Level == 1)
        v = FromColour(pos * 2);
    else
    {
        const bool       padX = ubo._CoarseDims.x != ubo._LevelDims.x * 2, padY = ubo._CoarseDims.y != ubo._LevelDims.y * 2;
        const FfxInt32x2 edge = FfxInt32x2(padX ? ubo._LevelDims.x - 2 : ubo._LevelDims.x - 1, padY ? ubo._LevelDims.y - 2 : ubo._LevelDims.y - 1);
        const FfxInt32x2 base = min(pos, edge) * 2 - 1;
        const FfxUInt32  w[4] = {1u, 3u, 3u, 1u};
        FfxUInt32        sum  = 0u;
        [[unroll]] for (FfxInt32 y = 0; y < 4; ++y)
            [[unroll]] for (FfxInt32 x = 0; x < 4; ++x)
                sum += w[x] * w[y] * Finer(base + FfxInt32x2(x, y));
        v = (sum + 32u) / 64u;
    }
    imageStore(rw_pyramid_tp1, FfxInt32x2(pos.x, ubo._LevelRow + pos.y), FfxFloat32x4(FfxFloat32(v) / 255.0f, 0.0f, 0.0f, 0.0f));
}
