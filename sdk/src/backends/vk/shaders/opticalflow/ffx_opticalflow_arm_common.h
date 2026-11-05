// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// Ported from Arm's ML Emulation Layer for Vulkan (github.com/arm/ai-ml-emulation-layer-for-vulkan, Apache-2.0).
//
// FFX_OF_IMAGE=arm: the optical flow engine of VK_ARM_data_graph_optical_flow as Arm's ML Emulation Layer for Vulkan
// models it (graph/optical_flow.cpp and graph/shaders/optical_flow/*.comp, Apache-2.0), as NFRU configures it: a
// 4x4 output grid (estimation ends at level 2), medium performance level (median and joint bilateral filters), the
// hint enabled, search range 3. Its luma pyramid has levels 1 to 5 (each half the one before, rounded up to even),
// one under the other in an R8 image (level l from row _LevelRow); its other images hold one level at a time.

#include "ffx_opticalflow_callbacks_glsl.h"
#include "ffx_opticalflow_image_common.h"

#define ARM_BLOCK_R 2  // the 5x5 macro block

// A pyramid texel as the layer reads its R8 images: floor(x * 255 + 0.5).
FfxUInt32 ArmU8(FfxFloat32 v)
{
    return FfxUInt32(floor(v * 255.0f + 0.5f));
}

FfxFloat32 ArmTexel(texture2D t, FfxInt32 row, FfxInt32x2 dims, FfxInt32x2 p)  // clamped to the level
{
    p = clamp(p, FfxInt32x2(0), dims - FfxInt32x2(1));
    return texelFetch(t, FfxInt32x2(p.x, row + p.y), 0).x;
}

FfxUInt32 ArmU8Zero(texture2D t, FfxInt32 row, FfxInt32x2 dims, FfxInt32x2 p)  // zero off the level
{
    if (any(lessThan(p, FfxInt32x2(0))) || any(greaterThanEqual(p, dims)))
        return 0u;
    return ArmU8(texelFetch(t, FfxInt32x2(p.x, row + p.y), 0).x);
}

// The layer's sample_bilinear: texel centres at +0.5, the corners clamped.
FfxFloat32x4 ArmBilinear(texture2D t, FfxInt32 row, FfxInt32x2 dims, FfxFloat32x2 samplePos)
{
    precise FfxFloat32x2 texelPos = samplePos - 0.5f;
    const FfxInt32x2     base     = FfxInt32x2(floor(texelPos));
    precise FfxFloat32x2 fraction = texelPos - FfxFloat32x2(base);
    const FfxInt32x2     maxPos   = dims - FfxInt32x2(1);
    const FfxFloat32x4   p00      = texelFetch(t, clamp(base, FfxInt32x2(0), maxPos) + FfxInt32x2(0, row), 0);
    const FfxFloat32x4   p10      = texelFetch(t, clamp(base + FfxInt32x2(1, 0), FfxInt32x2(0), maxPos) + FfxInt32x2(0, row), 0);
    const FfxFloat32x4   p01      = texelFetch(t, clamp(base + FfxInt32x2(0, 1), FfxInt32x2(0), maxPos) + FfxInt32x2(0, row), 0);
    const FfxFloat32x4   p11      = texelFetch(t, clamp(base + FfxInt32x2(1, 1), FfxInt32x2(0), maxPos) + FfxInt32x2(0, row), 0);
    precise FfxFloat32x4 row0     = p00 + (p10 - p00) * fraction.x;
    precise FfxFloat32x4 row1     = p01 + (p11 - p01) * fraction.x;
    precise FfxFloat32x4 value    = row0 + (row1 - row0) * fraction.y;
    return value;
}

FfxUInt32 ArmQuantize(FfxFloat32 x)  // convert_uint8
{
    return FfxUInt32(clamp(floor(x * 255.0f + 0.5f), 0.0f, 255.0f));
}
