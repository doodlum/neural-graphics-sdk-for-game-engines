/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_NSS_GENERATE_OFFSET_LUT_H
#define GPU_NSS_GENERATE_OFFSET_LUT_H

#include "nss/ffx_nss_common_glsl.h"

float2 Scale()
{
    return cbNSS._ScaleFactor.xy;
}

float2 InvScale()
{
    return cbNSS._ScaleFactor.zw;
}

#if (NSS_FILTER_MODE == 2) || (NSS_FILTER_MODE == 3)

#ifndef NSS_BIND_UAV_OFFSET_LUT
#error "NSS_BIND_UAV_OFFSET_LUT must be defined before including ffx_nss_generate_offset_lut.h"
#endif

layout(set = 0, binding = NSS_BIND_UAV_OFFSET_LUT, rgba32ui) uniform writeonly uimage2D rw_offset_lut;

#if NSS_FILTER_MODE == 3
const int32_t  kPackedTapCount                     = 4;
const int32_t  kLutGroupCount                      = 1;
const int32_t  kCandidateCount                     = 16;
const int32_t  kCandidateLinear[kCandidateCount]   = int32_t[kCandidateCount](5, 1, 4, 6, 9, 0, 2, 8, 10, 7, 13, 3, 11, 12, 14, 15);
const int32_t2 kCandidateOffsetYx[kCandidateCount] = int32_t2[kCandidateCount](int32_t2(0, 0),
                                                                               int32_t2(0, -1),
                                                                               int32_t2(-1, 0),
                                                                               int32_t2(1, 0),
                                                                               int32_t2(0, 1),
                                                                               int32_t2(-1, -1),
                                                                               int32_t2(1, -1),
                                                                               int32_t2(-1, 1),
                                                                               int32_t2(1, 1),
                                                                               int32_t2(2, 0),
                                                                               int32_t2(0, 2),
                                                                               int32_t2(2, -1),
                                                                               int32_t2(2, 1),
                                                                               int32_t2(-1, 2),
                                                                               int32_t2(1, 2),
                                                                               int32_t2(2, 2));
#else
const int32_t kPackedTapCount                   = 9;
const int32_t kLutGroupCount                    = 3;
const int32_t kCandidateCount                   = 36;
const int32_t kCandidateLinear[kCandidateCount] = int32_t[kCandidateCount](
    14, 8, 13, 15, 20, 7, 9, 19, 21, 2, 12, 16, 26, 1, 3, 6, 10, 18, 22, 25, 27, 0, 4, 24, 28, 17, 32, 11, 23, 31, 33, 5, 29, 30, 34, 35);
const int32_t2 kCandidateOffsetYx[kCandidateCount] = int32_t2[kCandidateCount](int32_t2(0, 0),
                                                                               int32_t2(0, -1),
                                                                               int32_t2(-1, 0),
                                                                               int32_t2(1, 0),
                                                                               int32_t2(0, 1),
                                                                               int32_t2(-1, -1),
                                                                               int32_t2(1, -1),
                                                                               int32_t2(-1, 1),
                                                                               int32_t2(1, 1),
                                                                               int32_t2(0, -2),
                                                                               int32_t2(-2, 0),
                                                                               int32_t2(2, 0),
                                                                               int32_t2(0, 2),
                                                                               int32_t2(-1, -2),
                                                                               int32_t2(1, -2),
                                                                               int32_t2(-2, -1),
                                                                               int32_t2(2, -1),
                                                                               int32_t2(-2, 1),
                                                                               int32_t2(2, 1),
                                                                               int32_t2(-1, 2),
                                                                               int32_t2(1, 2),
                                                                               int32_t2(-2, -2),
                                                                               int32_t2(2, -2),
                                                                               int32_t2(-2, 2),
                                                                               int32_t2(2, 2),
                                                                               int32_t2(3, 0),
                                                                               int32_t2(0, 3),
                                                                               int32_t2(3, -1),
                                                                               int32_t2(3, 1),
                                                                               int32_t2(-1, 3),
                                                                               int32_t2(1, 3),
                                                                               int32_t2(3, -2),
                                                                               int32_t2(3, 2),
                                                                               int32_t2(-2, 3),
                                                                               int32_t2(2, 3),
                                                                               int32_t2(3, 3));
#endif

struct BaseLutSample
{
    int32_t2 tile_offset;
    bool     valid;
};

int32_t PositiveMod(int32_t value, int32_t modulo)
{
    return value - modulo * int32_t(floor(float(value) / float(modulo)));
}

bool AxisSourceForHr(int32_t hr_coord, int32_t reduced_input_modulo, float scale, float inv_scale, float jitter, out int32_t lr_coord)
{
    lr_coord = int32_t(ceil(float(hr_coord) * inv_scale - jitter - 0.5 - 1e-6));
    if (lr_coord < int32_t(0) || lr_coord >= reduced_input_modulo)
    {
        return false;
    }
    int32_t projected = int32_t(floor((float(lr_coord) + jitter + 0.5) * scale));
    return projected == hr_coord;
}

BaseLutSample LoadBaseLutSample(int32_t2 hr_yx)
{
    BaseLutSample result;
    result.tile_offset = int32_t2(0);
    result.valid       = false;

    int32_t lr_y    = int32_t(0);
    int32_t lr_x    = int32_t(0);
    bool    valid_y = AxisSourceForHr(hr_yx.x, ReducedInputModulo().y, Scale().y, InvScale().y, JitterOffset().y, lr_y);
    bool    valid_x = AxisSourceForHr(hr_yx.y, ReducedInputModulo().x, Scale().x, InvScale().x, JitterOffset().x, lr_x);
    if (!(valid_y && valid_x))
    {
        return result;
    }

    int32_t back_projected_y = int32_t(floor((float(hr_yx.x) + 0.5) * InvScale().y));
    int32_t back_projected_x = int32_t(floor((float(hr_yx.y) + 0.5) * InvScale().x));
    result.tile_offset       = int32_t2(lr_x - back_projected_x, lr_y - back_projected_y);
    result.valid             = true;
    return result;
}

uint32_t EncodeI8(int32_t value)
{
    return uint32_t(value) & uint32_t(0xFF);
}

uint32_t PackTap(int32_t2 tile_xy, int32_t2 tap_offset_yx, int32_t tap_channel, bool valid, int32_t tile_x, int32_t tile_y)
{
    int32_t  lr_base_x   = int32_t(floor(float(tile_x) * InvScale().x));
    int32_t  lr_base_y   = int32_t(floor(float(tile_y) * InvScale().y));
    int32_t  lr_tap_x    = int32_t(floor((float(tile_x + tap_offset_yx.y) + 0.5) * InvScale().x + 1e-3)) + tile_xy.x;
    int32_t  lr_tap_y    = int32_t(floor((float(tile_y + tap_offset_yx.x) + 0.5) * InvScale().y + 1e-3)) + tile_xy.y;
    int32_t  lr_offset_x = lr_tap_x - lr_base_x;
    int32_t  lr_offset_y = lr_tap_y - lr_base_y;
    uint32_t valid_bit   = valid ? uint32_t(1) : uint32_t(0);
    // Match the torch high-quality path: an invalid padded zero-offset tap
    // still contributes the center sample, while its filter weight remains
    // masked out by the valid bit. Sparse mode is generated from the pruned
    // 4x4 candidate set and only marks valid selected taps as center taps.
#if NSS_FILTER_MODE == 3
    uint32_t center_bit = (valid && all(equal(tap_offset_yx, int32_t2(0)))) ? uint32_t(1) : uint32_t(0);
#else
    uint32_t center_bit = all(equal(tap_offset_yx, int32_t2(0))) ? uint32_t(1) : uint32_t(0);
#endif  // NSS_FILTER_MODE == 3

    return EncodeI8(lr_offset_x) | (EncodeI8(lr_offset_y) << uint32_t(8)) | ((uint32_t(tap_channel) & uint32_t(0x3F)) << uint32_t(16)) |
           (valid_bit << uint32_t(22)) | (center_bit << uint32_t(23));
}

void StoreLutRow(int32_t tile_x, int32_t tile_y, uint32_t packed_taps[kPackedTapCount])
{
    int32_t lut_base_x = tile_x * kLutGroupCount;
#if NSS_FILTER_MODE == 3
    imageStore(rw_offset_lut, int32_t2(lut_base_x, tile_y), uint32_t4(packed_taps[0], packed_taps[1], packed_taps[2], packed_taps[3]));
#else
    imageStore(rw_offset_lut, int32_t2(lut_base_x + 0, tile_y), uint32_t4(packed_taps[0], packed_taps[1], packed_taps[2], packed_taps[3]));
    imageStore(rw_offset_lut, int32_t2(lut_base_x + 1, tile_y), uint32_t4(packed_taps[4], packed_taps[5], packed_taps[6], packed_taps[7]));
    imageStore(rw_offset_lut, int32_t2(lut_base_x + 2, tile_y), uint32_t4(packed_taps[8], 0, 0, 0));
#endif
}

#if NSS_FILTER_MODE == 2
// Dense (QUALITY) LUT in separable form. Whether a candidate tap is valid and its low-res offset depend on x and y
// independently (LoadBaseLutSample / PackTap are per axis), and the candidate order is fixed. So instead of
// T_W x T_H tiles of 9 packed taps (3 rgba32ui texels each), the LUT holds one texel per x tile (row 0) and one per
// y tile (row 1):
//   .x  int8 lr offsets for axis offsets d = -2..1 (byte d + 2)
//   .y  int8 lr offsets for d = 2..3 (bytes 0..1)
//   .zw 36-bit mask over the candidate ranks: bit r set when this axis' component of candidate r is valid
// The postprocess ANDs the x and y masks and takes the first 9 set ranks, which reproduces the packed taps of the
// per-tile LUT exactly (same arithmetic per axis, integer combination only). lut_idx < T_W: x tile, then y tiles.
void GenerateOffsetLutAxis(int32_t lut_idx)
{
    int32_t idx_mod_x = max(IndexModulo().x, int32_t(1));
    int32_t idx_mod_y = max(IndexModulo().y, int32_t(1));
    bool    y_axis    = lut_idx >= idx_mod_x;
    int32_t t         = y_axis ? lut_idx - idx_mod_x : lut_idx;
    if (y_axis && t >= idx_mod_y)
    {
        return;
    }

    int32_t modulo    = y_axis ? idx_mod_y : idx_mod_x;
    int32_t in_modulo = y_axis ? ReducedInputModulo().y : ReducedInputModulo().x;
    float   scale     = y_axis ? Scale().y : Scale().x;
    float   inv_scale = y_axis ? InvScale().y : InvScale().x;
    float   jitter    = y_axis ? JitterOffset().y : JitterOffset().x;

    uint32_t offsets_lo = uint32_t(0);
    uint32_t offsets_hi = uint32_t(0);
    uint32_t valid_bits = uint32_t(0);
    for (int32_t i = int32_t(0); i < int32_t(6); ++i)
    {
        int32_t d  = i - int32_t(2);
        int32_t hr = PositiveMod(t + d, modulo);
        int32_t lr = int32_t(0);
        if (!AxisSourceForHr(hr, in_modulo, scale, inv_scale, jitter, lr))
        {
            continue;
        }
        // LoadBaseLutSample + PackTap, one axis.
        int32_t back_projected = int32_t(floor((float(hr) + 0.5) * inv_scale));
        int32_t tile_offset    = lr - back_projected;
        int32_t lr_base        = int32_t(floor(float(t) * inv_scale));
        int32_t lr_tap         = int32_t(floor((float(t + d) + 0.5) * inv_scale + 1e-3)) + tile_offset;
        uint32_t code          = EncodeI8(lr_tap - lr_base);
        if (i < int32_t(4))
        {
            offsets_lo |= code << uint32_t(8 * i);
        }
        else
        {
            offsets_hi |= code << uint32_t(8 * (i - 4));
        }
        valid_bits |= uint32_t(1) << uint32_t(i);
    }

    uint32_t rank_lo = uint32_t(0);
    uint32_t rank_hi = uint32_t(0);
    for (int32_t r = int32_t(0); r < kCandidateCount; ++r)
    {
        int32_t d = y_axis ? kCandidateOffsetYx[r].x : kCandidateOffsetYx[r].y;
        if ((valid_bits & (uint32_t(1) << uint32_t(d + 2))) != uint32_t(0))
        {
            if (r < int32_t(32))
            {
                rank_lo |= uint32_t(1) << uint32_t(r);
            }
            else
            {
                rank_hi |= uint32_t(1) << uint32_t(r - 32);
            }
        }
    }
    imageStore(rw_offset_lut, int32_t2(t, y_axis ? 1 : 0), uint32_t4(offsets_lo, offsets_hi, rank_lo, rank_hi));
}
#endif  // NSS_FILTER_MODE == 2

void GenerateOffsetLut(int32_t lut_idx)
{
#if NSS_FILTER_MODE == 2
    GenerateOffsetLutAxis(lut_idx);
#else
    int32_t idx_mod_x  = max(IndexModulo().x, int32_t(1));
    int32_t idx_mod_y  = max(IndexModulo().y, int32_t(1));
    int32_t tile_count = idx_mod_x * idx_mod_y;
    if (lut_idx >= tile_count)
    {
        return;
    }

    int32_t  tile_y   = lut_idx / idx_mod_x;
    int32_t  tile_x   = lut_idx - tile_y * idx_mod_x;
    int32_t  selected = int32_t(0);
    uint32_t packed_taps[kPackedTapCount];

    for (int32_t candidate_idx = int32_t(0); candidate_idx < kCandidateCount; ++candidate_idx)
    {
        if (selected >= kPackedTapCount)
        {
            break;
        }

        int32_t2      tap_offset_yx = kCandidateOffsetYx[candidate_idx];
        int32_t2      tap_hr_yx     = int32_t2(PositiveMod(tile_y + tap_offset_yx.x, idx_mod_y), PositiveMod(tile_x + tap_offset_yx.y, idx_mod_x));
        BaseLutSample base_sample   = LoadBaseLutSample(tap_hr_yx);
        if (!base_sample.valid)
        {
            continue;
        }

        packed_taps[selected] = PackTap(base_sample.tile_offset, tap_offset_yx, kCandidateLinear[candidate_idx], true, tile_x, tile_y);
        ++selected;
    }

    while (selected < kPackedTapCount)
    {
        packed_taps[selected] = PackTap(int32_t2(0), int32_t2(0), int32_t(0), false, tile_x, tile_y);
        ++selected;
    }

    StoreLutRow(tile_x, tile_y, packed_taps);
#endif  // NSS_FILTER_MODE == 2
}

#else
// Dummy function. For static 2x case, we don't need a dynamic lut pass.
void GenerateOffsetLut(int32_t lut_idx)
{
}

#endif

#endif  // GPU_NSS_GENERATE_OFFSET_LUT_H
