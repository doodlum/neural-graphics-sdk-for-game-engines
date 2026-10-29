/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef GPU_NSS_POSTPROCESS_H
#define GPU_NSS_POSTPROCESS_H

#include "nss/ffx_nss_common_glsl.h"

int32_t2 InputDims()
{
    return cbNSS._InputDims;
}

float2 InvInputDims()
{
    return cbNSS._InvInputDims;
}

int32_t2 OutputDims()
{
    return cbNSS._OutputDims;
}

float2 InvOutputDims()
{
    return cbNSS._InvOutputDims;
}

int32_t2 PaddedDims()
{
    return cbNSS._InputTensorSize;
}

float2 Scale()
{
    return cbNSS._ScaleFactor.xy;
}

float2 InvScale()
{
    return cbNSS._ScaleFactor.zw;
}

// Should use unppaded process dims
int32_t2 PreprocessDims()
{
#if NSS_PREPROCESS_HALF_RES_INPUT
    return InputDims() / 2;
#else
    return InputDims();
#endif
}

// torch/slang use g.pc.PaddedUvScale() = float2(g.pc.PreprocessDims()) * inv_temporal_dims;
// where the inv_temporal_dims is input tensor size
float2 PaddedUvScale()
{
    return cbNSS._PaddedUvScale;  // float2(PreprocessDims()) * cbNSS._InputTensorSizeRcp, computed on the host
}

// Note: the reset variable in reference implementation is a bit confusing
// based on its usage and value, it should be renamed to NotHistoryReset.
half Reset()
{
    return half(NotHistoryReset());
}

#ifndef NSS_V1_SHARP_THETA
#define NSS_V1_SHARP_THETA 1
#endif  // !NSS_V1_SHARP_THETA

// ---------------------------------------------------------------------------
// Enable Catmull-Rom history when requested by quality flag
// ---------------------------------------------------------------------------
#if NSS_USE_HISTORY_CATMULL
#ifndef ENABLE_HISTORY_CATMULL
#define ENABLE_HISTORY_CATMULL
#endif
#endif

// ---------------------------------------------------------------------------
// Texture name aliases (prototype → SDK binding names)
// ---------------------------------------------------------------------------
// _NearestDepthOffsetTex → SDK's _NearestDepthCoordTex
#define _NearestDepthOffsetTex (_NearestDepthCoordTex)
// _MotionVectorTex → SDK's _MotionTex
#define _MotionVectorTex (_MotionTex)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const float kEps_Post          = 1e-7;
const float kMotionThreshold   = 0.1;
const float kMotionThresholdSq = kMotionThreshold * kMotionThreshold;

// Compute path, exact-2x sparse filter: one thread per 2x2 output quad (the host dispatches accordingly).
#ifndef NSS_POSTPROCESS_QUAD
#define NSS_POSTPROCESS_QUAD 0
#endif
// Compute path, exact-2x dense filter (QUALITY): one thread per 2x2 group of output pixels sharing their taps.
#ifndef NSS_POSTPROCESS_QUAD_DENSE2X
#define NSS_POSTPROCESS_QUAD_DENSE2X 0
#endif

#if NSS_USE_SPARSE_2X2_FILTER
const int32_t kKpnChannels = 16;
#else
const int32_t kKpnChannels = 36;
#endif
const int32_t kKpnPackedChannelsPerTexel = 4;
const int32_t kKpnPackedTexelsPerPixel   = kKpnChannels / kKpnPackedChannelsPerTexel;
const int32_t kKpnPackedChannelShift     = 2;
const int32_t kKpnPackedChannelMask      = kKpnPackedChannelsPerTexel - 1;

// QAT metadata for KPN coefficients (SINT) and temporal params (SNORM).
// QAT metadata (`_KpnCoefficients` / `_TemporalTensor` SINT).
const half2 kKpnQuant = half2(0.003937007859349251, -127.0);
// Temporal params are sampled from SNORM image alias, so use SNORM dequant.
const half2 kTemporalQuant = half2(0.49999999813735485, -1.0);

const int32_t kLutPatternCount = 4;

// ---------------------------------------------------------------------------
// KernelPattern LUT
// ---------------------------------------------------------------------------
struct KernelPattern
{
    int16_t2 base_offset;
    int16_t  base_channel;
    int16_t  _pad0;
};

#if !NSS_USE_SPARSE_2X2_FILTER
const int16_t4 kTapDx0 = int16_t4(0, 0, 0, 2);
const int16_t4 kTapDy0 = int16_t4(0, 2, 4, 0);
const int16_t4 kTapCh0 = int16_t4(0, 2, 4, 12);
const int16_t4 kTapDx1 = int16_t4(2, 2, 4, 4);
const int16_t4 kTapDy1 = int16_t4(2, 4, 0, 2);
const int16_t4 kTapCh1 = int16_t4(14, 16, 24, 26);
const int16_t2 kTapD2  = int16_t2(4, 4);
const int16_t  kTapCh2 = int16_t(28);

const KernelPattern kKernelLut[kLutPatternCount] = KernelPattern[kLutPatternCount](KernelPattern(int16_t2(-1, -1), int16_t(7), int16_t(0)),
                                                                                   KernelPattern(int16_t2(-2, -1), int16_t(1), int16_t(0)),
                                                                                   KernelPattern(int16_t2(-1, -2), int16_t(6), int16_t(0)),
                                                                                   KernelPattern(int16_t2(-2, -2), int16_t(0), int16_t(0)));
#else
const int16_t4 kTap2x2Dx[kLutPatternCount] =
    int16_t4[kLutPatternCount](int16_t4(-1, -1, +1, +1), int16_t4(+0, +0, +2, +2), int16_t4(-1, +1, -1, +1), int16_t4(+0, +0, +2, +2));
const int16_t4 kTap2x2Dy[kLutPatternCount] =
    int16_t4[kLutPatternCount](int16_t4(-1, +1, -1, +1), int16_t4(-1, +1, -1, +1), int16_t4(+0, +0, +2, +2), int16_t4(+0, +2, +0, +2));
const int16_t4 kTap2x2Ch[kLutPatternCount] =
    int16_t4[kLutPatternCount](int16_t4(0, 2, 8, 10), int16_t4(4, 6, 12, 14), int16_t4(1, 9, 3, 11), int16_t4(5, 7, 13, 15));
#endif

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------
half MaxHalf(half x)
{
    return min(x, half(65504.HF));
}

half3 MaxHalf(half3 x)
{
    return half3(min(x.r, half(65504.HF)), min(x.g, half(65504.HF)), min(x.b, half(65504.HF)));
}

half4 MaxHalf(half4 x)
{
    return half4(min(x.r, half(65504.HF)), min(x.g, half(65504.HF)), min(x.b, half(65504.HF)), min(x.a, half(65504.HF)));
}

// Alias Tonemap4 to the SDK's half4 Tonemap (functionally identical).
// Karis tonemapper for half4 (all 4 channels including alpha).
// Defined here because the existing Tonemap(half4) is only in the TONEMAP_REINHARD path.
half4 Tonemap4(half4 x)
{
    // Clamp internal working value, do not mutate input
    half4 xc = clamp(x, half4(0.HF), half4(MAX_FP16));

    // Compute max channel in FP32 to avoid FP16 Flush‑To‑Zero
    float m = max(max(float(xc.r), float(xc.g)), float(xc.b));

    return half4(float4(xc) * (1.0f / (1.0f + m)));
}

//=========================================================================
// Resource bindings exclusively used in the postprocess pass
//=========================================================================

//-------------------------------------------------------------------------
// Input: KPN coefficients tensor (36-channel, 1/4 resolution)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_KPN_TENSOR)

// ---- KPN coefficients tensor (36-channel) ----
// NSS v1: cannot use image alias (too many channels). Use native tensor ops or buffer alias (SSBO).
#if !NSS_SUPPORT_TENSOR
// Buffer alias path: tensor aliased as a std430 SSBO — read via linear NHWC indexing.
layout(set = 0, binding = NSS_BIND_KPN_TENSOR, std430) readonly buffer KpnCoeffBuffer
{
    // NHWC int8, four channels per word (lowest byte = lowest channel); read as words so a channel is one
    // bitfieldExtract instead of a per-lane select between the four bytes.
    uint32_t data[];
}
r_coefficients_kpn_buffer;
#else
// Native tensor path: read via tensorReadARM.
layout(set = 0, binding = NSS_BIND_KPN_TENSOR) uniform readonly tensorARM<tensor_t, 4> r_coefficients_kpn_tensor;
#endif

// ---------------------------------------------------------------------------
// KPN tensor read — NSS v1 uses a single 36-channel tensor at 1/4 of dataGraph resolution.
// Buffer alias (SSBO) path when native tensor ops are unavailable; tensorReadARM otherwise.
// ---------------------------------------------------------------------------
#if !NSS_SUPPORT_TENSOR
// Signed byte `channel & 3` of KPN word `word_idx`, as an int.
int32_t ReadKpnWordChannel(int32_t word_idx, int32_t channel)
{
    return bitfieldExtract(int32_t(r_coefficients_kpn_buffer.data[word_idx]), (channel & 3) << 3, 8);
}
#endif

int32_t ReadKpnParamsFromBase(int32_t kpn_texel_base_x, int32_t kpn_y, int32_t channel)
{
#if !NSS_SUPPORT_TENSOR
    // Buffer alias path: tensor stored as SSBO in NHWC layout, one word per group of 4 channels.
    // Linear index: (y * kpn_w + x) * (kKpnChannels / 4) + channel/4
    int32_t kpn_w      = KpnDims().x;
    int32_t ch_group   = channel >> 2;  // which word
    int32_t linear_idx = (kpn_y * kpn_w + kpn_texel_base_x) * (kKpnChannels / 4) + ch_group;
    return ReadKpnWordChannel(linear_idx, channel);
#else
    int8_t q = int8_t(0);
    tensorReadARM(r_coefficients_kpn_tensor, uint[](0, uint(kpn_y), uint(kpn_texel_base_x), uint(channel)), q);
    return int32_t(q);
#endif
}

int8_t ReadKpnParamsInt8FromBase(int32_t kpn_texel_base_x, int32_t kpn_y, int32_t channel)
{
#if !NSS_SUPPORT_TENSOR
    return int8_t(ReadKpnParamsFromBase(kpn_texel_base_x, kpn_y, channel));
#else
    // Native tensor path: read directly via tensorReadARM using [batch, y, x, channel] indexing.
    int8_t q = int8_t(0);
    tensorReadARM(r_coefficients_kpn_tensor, uint[](0, uint(kpn_y), uint(kpn_texel_base_x), uint(channel)), q);
    return q;
#endif
}

int8_t ReadKpnParamsInt8(int32_t2 kpn_tap, int32_t channel)
{
    return ReadKpnParamsInt8FromBase(kpn_tap.x, kpn_tap.y, channel);
}

// ---------------------------------------------------------------------------
// KPN coordinate mapping: input space → KPN tensor space
// ---------------------------------------------------------------------------
// floor(n * k / d) for n >= 0, k >= 0, d >= 1 without an integer divide (a generic 32-bit divide is ~20 VALU
// including quarter-rate multiplies on AMD). With x = n * k + 0.5 (exact in fp32), x / d lies at least 0.5 / d away
// from every integer, and x * (1 / d) has a relative error below 3 * 2^-23 (Vulkan allows 2.5 ulp for the division,
// the multiply adds 0.5 ulp), i.e. an absolute error below 0.5 / d while x < 2^20: the floor is exactly the integer
// quotient on any conformant implementation. Larger operands take the integer divide.
int32_t4 KpnScaledCoordDiv4(int32_t4 n, int32_t k, int32_t d)
{
    // d an exact power-of-two multiple of k (the 2x presets, e.g. 640 -> KPN 80): floor(n * k / d) = n >> log2(d / k).
    if (k > 0 && (d % k) == 0 && bitCount(uint32_t(d / k)) == 1)
    {
        return n >> findLSB(d / k);
    }
    if (d * k < (1 << 20))
    {
        return int32_t4(floor((float4(n) * float(k) + float4(0.5)) * (1.0 / float(d))));
    }
    return (n * int32_t4(k)) / int32_t4(d);
}

int32_t2 KpnScaledCoordDiv2(int32_t2 n, int32_t k, int32_t d)
{
    // d an exact power-of-two multiple of k (the 2x presets, e.g. 640 -> KPN 80): floor(n * k / d) = n >> log2(d / k).
    if (k > 0 && (d % k) == 0 && bitCount(uint32_t(d / k)) == 1)
    {
        return n >> findLSB(d / k);
    }
    if (d * k < (1 << 20))
    {
        return int32_t2(floor((float2(n) * float(k) + float2(0.5)) * (1.0 / float(d))));
    }
    return (n * int32_t2(k)) / int32_t2(d);
}

int32_t KpnCoordFromInputCoord(int32_t input_coord, int32_t input_dim, int32_t kpn_dim)
{
    // input_coord is a clamped (non-negative) tap coordinate.
    int32_t safe_dim = max(input_dim, int32_t(1));
    return min(KpnScaledCoordDiv4(int32_t4(input_coord), kpn_dim, safe_dim).x, kpn_dim - int32_t(1));
}

int16_t4 KpnCoordFromInputCoord4(int16_t4 input_coord, int32_t input_dim, int32_t kpn_dim)
{
    // input_coord holds clamped (non-negative) tap coordinates.
    int32_t  safe_dim = max(input_dim, int32_t(1));
    int32_t4 scaled   = KpnScaledCoordDiv4(int32_t4(input_coord), kpn_dim, safe_dim);
    return int16_t4(min(scaled, int32_t4(kpn_dim - int32_t(1))));
}

// KPN coordinate mapping using precomputed scale (kpnDims / paddedDims).
// Used in QUALITY mode where paddedDims > inputDims due to alignment padding.
// ---------------------------------------------------------------------------
//
// The KPN tensor is exactly 1/4 of the dataGraph size (ffx_nss.cpp: kpnDimension = dataGraphSize / 4, dataGraphSize a
// multiple of FFX_NSS_RESOURCE_ALIGNMENT = 8, asserted there), so `scale` is exactly 0.25. For any integer t no multiple
// of 4 lies in (t, t + 0.5 + 1e-3], so floor((t + 0.5 + 1e-3) * 0.25) == t >> 2 (arithmetic shift): the shift below is
// the same mapping without the float round trip (~24 VALU per pixel in the exact-2x filter). NSS_KPN_SCALE_QUARTER=0
// restores the float formula.
#ifndef NSS_KPN_SCALE_QUARTER
#define NSS_KPN_SCALE_QUARTER 1
#endif

int32_t KpnCoordFromScaledTap(int32_t tap_coord, float scale, int32_t kpn_dim)
{
#if NSS_KPN_SCALE_QUARTER
    int32_t scaled = tap_coord >> int32_t(2);
#else
    int32_t scaled = int32_t(floor((float(tap_coord) + 0.5 + 1e-3) * scale));
#endif
    return min(scaled, kpn_dim - int32_t(1));
}

int16_t4 KpnCoordFromScaledTap4(int16_t4 tap_coord, float scale, int32_t kpn_dim)
{
#if NSS_KPN_SCALE_QUARTER
    return min(tap_coord >> int16_t(2), int16_t4(kpn_dim - int32_t(1)));
#else
    float4   fcoord = float4(int32_t4(tap_coord)) + float4(0.5 + 1e-3);
    int32_t4 scaled = int32_t4(floor(fcoord * float4(scale)));
    return int16_t4(min(scaled, int32_t4(kpn_dim - int32_t(1))));
#endif
}

// ---------------------------------------------------------------------------
// KPN weight samplers
// ---------------------------------------------------------------------------
half SampleKpnWeight(int32_t tap_z, int32_t2 kpn_tap)
{
    int32_t q = ReadKpnParamsFromBase(kpn_tap.x, kpn_tap.y, tap_z);
    return half(max(float(Dequantize(half(q), kKpnQuant)), float(EPS)));
}

half4 SampleKpnWeight4(int16_t4 tap_z, int16_t4 kpn_x, int16_t4 kpn_y)
{
    int32_t4 q = int32_t4(ReadKpnParamsFromBase(kpn_x.x, kpn_y.x, tap_z.x),
                          ReadKpnParamsFromBase(kpn_x.y, kpn_y.y, tap_z.y),
                          ReadKpnParamsFromBase(kpn_x.z, kpn_y.z, tap_z.z),
                          ReadKpnParamsFromBase(kpn_x.w, kpn_y.w, tap_z.w));
    return half4(max(float4(Dequantize(half4(q), kKpnQuant)), float4(float(EPS))));
}

half4 NormalizeSparseKernelWeights4(half4 weights)
{
    half4 clamped_weights = max(weights, half4(MIN_SUMW_HALF));
    half  sum_w           = dot(clamped_weights, half4(1.HF));
    return half4(clamped_weights * rcp(sum_w));
}

half4 NormalizeSparseKernelWeights4(half4 weights, half4 valid_mask)
{
    half4 clamped_weights = max(weights, half4(MIN_SUMW_HALF)) * valid_mask;
    half  sum_w           = max(dot(clamped_weights, half4(1.HF)), half(MIN_SUMW_HALF));
    return half4(clamped_weights * rcp(sum_w));
}

#endif

//-------------------------------------------------------------------------
// Input: Temporal feedback tensor (4-channel) — use image alias
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_FEEDBACK_TENSOR)
layout(set = 0, binding = NSS_BIND_SRV_FEEDBACK_TENSOR) uniform lowp texture2D r_temporal_feedback_tensor;
#define _TemporalTensor sampler2D(r_temporal_feedback_tensor, s_LinearClamp)

// ---------------------------------------------------------------------------
// Temporal parameter sampling
// ---------------------------------------------------------------------------
void SampleTemporalParams(float2 uv, out half theta, out half alpha, out half gamma)
{
    float2 uv_temporal = uv * PaddedUvScale();
    half4  params      = Dequantize(half4(textureLod(_TemporalTensor, uv_temporal, 0.0)), kTemporalQuant);
#if NSS_V1_SHARP_THETA
    half theta_in  = clamp(params.x, half(0.HF), half(1.HF));
    half theta_inv = half(1.HF) - theta_in;
    half theta_a   = theta_in * theta_in;
    half theta_b   = theta_inv * theta_inv;
    theta          = theta_a * rcp(max(theta_a + theta_b, half(1e-6f)));
#else
    theta = params.x;
#endif  // NSS_V1_SHARP_THETA
    alpha = params.y * 0.35HF + 0.05HF;
    gamma = params.z * 2.0HF;
}

#endif

//-------------------------------------------------------------------------
// Input: Depth Offset (aka NearestDepthCoordTex)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_SRV_NEAREST_DEPTH_COORD)
layout(set = 0, binding = NSS_BIND_SRV_NEAREST_DEPTH_COORD) uniform highp texture2D r_input_nearest_depth_coord;
#define _NearestDepthCoordTex sampler2D(r_input_nearest_depth_coord, s_LinearClamp)
#endif

#if ((NSS_FILTER_MODE == 2) || (NSS_FILTER_MODE == 3)) && defined(NSS_BIND_SRV_OFFSET_LUT)
layout(set = 0, binding = NSS_BIND_SRV_OFFSET_LUT) uniform highp utexture2D r_offset_lut;
#define _OffsetLutUint4Tex usampler2D(r_offset_lut, s_PointClamp)
#endif

//-------------------------------------------------------------------------
// Output: Upscaled-output (RWTexture2D UAV)
//-------------------------------------------------------------------------
#if defined(NSS_BIND_UAV_UPSCALED_OUTPUT)
layout(set = 0, binding = NSS_BIND_UAV_UPSCALED_OUTPUT, OUTPUT_IMG_FORMAT) uniform mediump image2D rw_upscaled_output;
#elif defined(NSS_BIND_RENDER_TARGET_UPSCALED_OUTPUT)
layout(location = NSS_BIND_RENDER_TARGET_UPSCALED_OUTPUT) out FfxFloat32x4 rw_upscaled_output;
#endif

#if (defined(NSS_BIND_UAV_HISTORY_UPSCALED_COLOR) || defined(NSS_BIND_RENDER_TARGET_HISTORY_UPSCALED_COLOR)) && MANAGE_HISTORY
#if defined(NSS_BIND_UAV_HISTORY_UPSCALED_COLOR)
layout(set = 0, binding = NSS_BIND_UAV_HISTORY_UPSCALED_COLOR, OUTPUT_IMG_FORMAT) uniform writeonly mediump image2D rw_prev_upscaled_color;
#elif defined(NSS_BIND_RENDER_TARGET_HISTORY_UPSCALED_COLOR)
layout(location = NSS_BIND_RENDER_TARGET_HISTORY_UPSCALED_COLOR) out FfxFloat32x4 rw_prev_upscaled_color;
#endif
#endif

// ---------------------------------------------------------------------------
// Decode nearest-depth offset from the preprocess output texture
// ---------------------------------------------------------------------------
int32_t2 DecodeNearestOffset(int32_t2 pixel)
{
    float norm_code = 0.0;

#if NSS_PREPROCESS_HALF_RES_INPUT
    int32_t2 texel =
        clamp(int32_t2(floor((float2(pixel) + float2(0.5)) * float2(PreprocessDims()) * InvInputDims())), int32_t2(0), PreprocessDims() - int32_t2(1));
#if NSS_PACKED_NEAREST_OFFSET_QUAD
    float2  norm_code_pair = texelFetch(_NearestDepthOffsetTex, texel, 0).rg;
    int32_t lane_idx       = (pixel.y & int32_t(1)) * int32_t(2) + (pixel.x & int32_t(1));
    int32_t packed_byte    = lane_idx < int32_t(2) ? int32_t(norm_code_pair.x * 255.0 + 0.5) : int32_t(norm_code_pair.y * 255.0 + 0.5);
    int32_t packed_nibble  = (lane_idx & int32_t(1)) == int32_t(0) ? (packed_byte & 0xF) : ((packed_byte >> 4) & 0xF);
    // Decode nibble: x = (nibble & 3) - 1, y = ((nibble >> 2) & 3) - 1
    int32_t x = (packed_nibble & 3) - 1;
    int32_t y = ((packed_nibble >> 2) & 3) - 1;
    return int32_t2(x, y);
#else
    norm_code = texelFetch(_NearestDepthOffsetTex, texel, 0).r;
#endif
#else
    pixel = clamp(pixel, int32_t2(0), InputDims() - int32_t2(1));
    norm_code = texelFetch(_NearestDepthOffsetTex, pixel, 0).r;
#endif

    // Decode R8_UNORM: x = (code & 7) - 2, y = ((code >> 3) & 7) - 2
    int32_t code = int32_t(norm_code * 255.0 + 0.5);
    return int32_t2((code & 7) - 2, ((code >> 3) & 7) - 2);
}

// ---------------------------------------------------------------------------
// Motion loading for postprocess (uses nearest-depth dilation)
// ---------------------------------------------------------------------------
int32_t2 MotionPostInputCoord(int32_t2 output_coord)
{
    return int32_t2(float2(output_coord) * InvScale());
}

half2 LoadMotionPostAtInput(int32_t2 input_coord)
{
    // Reproject motion lookup using nearest-depth dilated coordinate.
    int32_t2 dilation_offset = DecodeNearestOffset(input_coord);
    int32_t2 dilated_coord   = clamp(input_coord + dilation_offset, int32_t2(0), InputDims() - int32_t2(1));

    half2 v = half2(texelFetch(_MotionVectorTex, dilated_coord, 0).xy * MotionVectorScale()) * half2(Scale());
    v *= half(dot(float2(v), float2(v)) > kMotionThresholdSq);
    return v;
}

half2 LoadMotionPost(int32_t2 output_coord)
{
    return LoadMotionPostAtInput(MotionPostInputCoord(output_coord));
}

// ---------------------------------------------------------------------------
// Colour tap loaders
// ---------------------------------------------------------------------------
half3 LoadColourTap(int32_t2 lr_tap)
{
    return half3(texelFetch(_ColourTex, lr_tap, 0).rgb);
}

half4 LoadColourTap4(int32_t2 lr_tap)
{
    return half4(LoadColourTap(lr_tap), 1.HF);
}

#if defined(NSS_BIND_SRV_HISTORY_UPSCALED_COLOR)
// ---------------------------------------------------------------------------
// History samplers (with optional Catmull-Rom)
// ---------------------------------------------------------------------------
half4 LoadWarpedHistoryPost(float2 uv)
{
    return half4(texture(_HistoryTex, uv).rgb, 1.HF);
}

half4 LoadWarpedHistoryCatmullPost(float2 uv)
{
    // Catmull-Rom 5-tap cross filter
    float2 scaledUV  = uv * cbNSS._OutputDimsF;  // float2(OutputDims()), converted on the host
    float2 baseFloor = floor(scaledUV - 0.5) + 0.5;

    half2 f  = half2(scaledUV - baseFloor);
    half2 f2 = f * f;
    half2 f3 = f2 * f;

    half2 w0  = f2 - 0.5HF * (f3 + f);
    half2 w3  = 0.5HF * (f3 - f2);
    half2 w2  = 0.5HF * f + f2 * (2.0HF - 1.5HF * f);
    half2 w12 = 1.0HF + 0.5HF * f - 0.5HF * f2;

    half2 wx02    = half2(w0.x, w3.x);
    half2 wy02    = half2(w0.y, w3.y);
    half2 w_ud    = half2(w12.x) * wy02;
    half2 w_lr    = half2(w12.y) * wx02;
    half  wCenter = w12.x * w12.y;

    half2  dxy       = w2 / half2(max(float2(w12), float2(float(EPS))));
    float2 base_uv   = baseFloor * InvOutputDims();
    float2 center_uv = base_uv + float2(dxy) * InvOutputDims();
    float  left_x    = base_uv.x - InvOutputDims().x;
    float  right_x   = base_uv.x + 2.0 * InvOutputDims().x;
    float  up_y      = base_uv.y - InvOutputDims().y;
    float  down_y    = base_uv.y + 2.0 * InvOutputDims().y;

    half4 left   = half4(texture(_HistoryTex, float2(left_x, center_uv.y)).rgb, 1.HF);
    half4 up     = half4(texture(_HistoryTex, float2(center_uv.x, up_y)).rgb, 1.HF);
    half4 center = half4(texture(_HistoryTex, center_uv).rgb, 1.HF);
    half4 right  = half4(texture(_HistoryTex, float2(right_x, center_uv.y)).rgb, 1.HF);
    half4 down   = half4(texture(_HistoryTex, float2(center_uv.x, down_y)).rgb, 1.HF);

    half4 accum = up * w_ud.x + left * w_lr.x + center * wCenter + right * w_lr.y + down * w_ud.y;
    half4 cmin4 = half4(min(float4(up), min(float4(left), min(float4(center), min(float4(right), float4(down))))));
    half4 cmax4 = half4(max(float4(up), max(float4(left), max(float4(center), max(float4(right), float4(down))))));

    half3 color = accum.rgb * rcp(accum.w);
    color       = any(lessThan(color, half3(0.HF))) ? clamp(color, cmin4.rgb, cmax4.rgb) : color;
    return half4(color, 1.HF);
}
#endif

// ---------------------------------------------------------------------------
// Spatial KPN filtering → first/second moments + centre sample
// ---------------------------------------------------------------------------
#define NSS_SQ_MAT(_M) half4x4(_M[0] * _M[0], _M[1] * _M[1], _M[2] * _M[2], _M[3] * _M[3])

#define NSS_LOAD_TAPS_4(TAP_X, TAP_Y, OUT_MAT)                   \
    {                                                            \
        OUT_MAT[0] = LoadColourTap4(int32_t2(TAP_X.x, TAP_Y.x)); \
        OUT_MAT[1] = LoadColourTap4(int32_t2(TAP_X.y, TAP_Y.y)); \
        OUT_MAT[2] = LoadColourTap4(int32_t2(TAP_X.z, TAP_Y.z)); \
        OUT_MAT[3] = LoadColourTap4(int32_t2(TAP_X.w, TAP_Y.w)); \
    }

void NormalizeFilterMoments(half4 accum_m1, half4 accum_m2, out half4 m1, out half4 m2)
{
    half denom = max(half(accum_m1.a), half(MIN_SUMW_HALF));
    m1         = half4(accum_m1.rgb * rcp(denom), 0.HF);
    m2         = half4(accum_m2.rgb * rcp(denom), 0.HF);
}

int32_t2 KpnCoordForDenseFilter(int32_t2 lr_tap, int32_t2 kpn_max)
{
#if NSS_PREPROCESS_HALF_RES_INPUT
    return clamp(int32_t2(KpnCoordFromInputCoord(lr_tap.x, InputDims().x, KpnDims().x), KpnCoordFromInputCoord(lr_tap.y, InputDims().y, KpnDims().y)),
                 int32_t2(0),
                 kpn_max);
#else
    return clamp(
        int32_t2(KpnCoordFromScaledTap(lr_tap.x, KpnScale().x, KpnDims().x), KpnCoordFromScaledTap(lr_tap.y, KpnScale().y, KpnDims().y)), int32_t2(0), kpn_max);
#endif
}

void KpnCoordsForDenseFilter4(int16_t4 tap_x, int16_t4 tap_y, int16_t4 zero4, int16_t4 kpn_max_x4, int16_t4 kpn_max_y4, out int16_t4 kpn_x, out int16_t4 kpn_y)
{
#if NSS_PREPROCESS_HALF_RES_INPUT
    kpn_x = clamp(KpnCoordFromInputCoord4(tap_x, InputDims().x, KpnDims().x), zero4, kpn_max_x4);
    kpn_y = clamp(KpnCoordFromInputCoord4(tap_y, InputDims().y, KpnDims().y), zero4, kpn_max_y4);
#else
    kpn_x = clamp(KpnCoordFromScaledTap4(tap_x, KpnScale().x, KpnDims().x), zero4, kpn_max_x4);
    kpn_y = clamp(KpnCoordFromScaledTap4(tap_y, KpnScale().y, KpnDims().y), zero4, kpn_max_y4);
#endif
}

#if (NSS_FILTER_MODE == 2) || (NSS_FILTER_MODE == 3)
struct OffsetLutTap
{
    int32_t2 lr_offset;
    int32_t  tap_channel;
    bool     valid;
    bool     center;
};

#if NSS_FILTER_MODE == 3
const int32_t kLutGroupCount = 1;
#else
const int32_t kLutGroupCount = 3;
#endif

int32_t DecodePackedI8(uint32_t packed, uint32_t shift)
{
    return bitfieldExtract(int32_t(packed), int32_t(shift), int32_t(8));
}

OffsetLutTap DecodePackedOffsetLutTap(uint32_t packed)
{
    OffsetLutTap tap;
    tap.lr_offset   = int32_t2(DecodePackedI8(packed, uint32_t(0)), DecodePackedI8(packed, uint32_t(8)));
    tap.tap_channel = int32_t((packed >> uint32_t(16)) & uint32_t(0x3F));
    tap.valid       = ((packed >> uint32_t(22)) & uint32_t(1)) != uint32_t(0);
    tap.center      = ((packed >> uint32_t(23)) & uint32_t(1)) != uint32_t(0);
    return tap;
}

uint32_t4 LoadOffsetLutPacked4(int32_t2 tile_idx, int32_t group_idx)
{
    return texelFetch(_OffsetLutUint4Tex, int32_t2(tile_idx.x * kLutGroupCount + group_idx, tile_idx.y), 0);
}

int32_t2 PackedOffsetLutModulo()
{
    return max(int32_t2(IndexModulo()), int32_t2(1));
}

// output_px % modulo for output_px >= 0 without integer divides: exact float quotient (see KpnScaledCoordDiv4),
// valid while output_px < 2^20; plain % otherwise.
int32_t2 PackedOffsetLutTile(int32_t2 output_px, int32_t2 modulo)
{
    if (all(lessThan(OutputDims(), int32_t2(1 << 20))))
    {
        int32_t2 q = int32_t2(floor((float2(output_px) + float2(0.5)) * (float2(1.0) / float2(modulo))));
        return output_px - q * modulo;
    }
    return output_px % modulo;
}

void AccumulatePackedLrOffsetTap(
    OffsetLutTap lut_tap, int32_t2 lr_base, int32_t2 colour_max, int32_t2 kpn_max, inout half4 accum_m1, inout half4 accum_m2, inout half4 center_sample)
{
    if (!lut_tap.valid)
    {
        return;
    }

    int32_t2 lr_tap  = clamp(lr_base + lut_tap.lr_offset, int32_t2(0), colour_max);
    int32_t2 kpn_tap = KpnCoordForDenseFilter(lr_tap, kpn_max);
    half4    tap_col = LoadColourTap4(lr_tap);
    if (lut_tap.center)
    {
        center_sample = tap_col;
    }
    half tap_weight = SampleKpnWeight(lut_tap.tap_channel, kpn_tap);
    accum_m1 += tap_col * tap_weight;
    accum_m2 += (tap_col * tap_col) * tap_weight;
}
#endif

#if NSS_FILTER_MODE == 2
// Candidate rank -> tap channel (bits 0..5) and the axis offsets + 2 (x: bits 8..10, y: bits 12..14). Same order as
// kCandidateLinear / kCandidateOffsetYx in ffx_nss_generate_offset_lut.h.
const uint32_t kDenseRankInfo[36] = uint32_t[36](
    0x220Eu, 0x2108u, 0x120Du, 0x320Fu, 0x2314u, 0x1107u, 0x3109u, 0x1313u, 0x3315u, 0x2002u, 0x020Cu, 0x4210u,
    0x241Au, 0x1001u, 0x3003u, 0x0106u, 0x410Au, 0x0312u, 0x4316u, 0x1419u, 0x341Bu, 0x0000u, 0x4004u, 0x0418u,
    0x441Cu, 0x5211u, 0x2520u, 0x510Bu, 0x5317u, 0x151Fu, 0x3521u, 0x5005u, 0x541Du, 0x051Eu, 0x4522u, 0x5523u);

#if NSS_POSTPROCESS_AXIS_LDS && !NSS_SUPPORT_TENSOR
// Compute path: every tap coordinate is (column part, row part). The 6 candidate offsets of each output column and row
// are mapped to clamped LR coordinates and KPN byte offsets once per frame by the LUT pass, with the same arithmetic
// the per-pixel path uses; a workgroup copies its 16 columns and 16 rows into LDS and the pixels read them back (one
// 64-bit LDS read per axis and tap instead of ~30 VALU of clamping and KPN mapping).
const int32_t kAxisLdsW = int32_t(FFX_NSS_THREAD_GROUP_WIDTH);
const int32_t kAxisLdsH = int32_t(FFX_NSS_THREAD_GROUP_HEIGHT);
// Columns first, then rows: x entries at [c * 6 + i], y entries at [kAxisLdsW * 6 + r * 6 + i]. Each entry is a pair
// (clamped LR coordinate, KPN byte offset), read with one 64-bit LDS load. The KPN tensor is [y][x][36] int8, so the byte
// address of channel 6 * ix + iy at KPN pixel (x, y) splits exactly into an x part and a y part:
//   36 * (y * kpn_w + x) + 6 * ix + iy = (36 * x + 6 * ix) + (36 * kpn_w * y + iy)
shared int32_t2 s_axis[(kAxisLdsW + kAxisLdsH) * 6];
shared uint32_t s_axis_valid[kAxisLdsW + kAxisLdsH];  // 6-bit validity of the column's / row's offsets
#define s_axis_x(I)       s_axis[(I)]
#define s_axis_y(I)       s_axis[kAxisLdsW * 6 + (I)]
#define s_axis_valid_x(I) s_axis_valid[(I)]
#define s_axis_valid_y(I) s_axis_valid[kAxisLdsW + (I)]

// The KPN tensor as bytes (same buffer, second binding): one byte load per tap, read unsigned (the two's complement
// byte u of the int8 value q) and used as an index into s_kpn_weight.
layout(set = 0, binding = NSS_BIND_KPN_TENSOR_BYTES, std430) readonly buffer KpnCoeffBytes
{
    uint8_t data[];
}
r_coefficients_kpn_bytes;
// Dequantised KPN weight for every byte value: s_kpn_weight[u] = SampleKpnWeight's max(Dequantize(q), EPS) for
// q = int8(u), filled by the workgroup (one entry per lane) with the same operations.
shared half s_kpn_weight[256];
#if FFX_NSS_THREAD_GROUP_WIDTH * FFX_NSS_THREAD_GROUP_HEIGHT < 256
#error "PrepareDenseFilterAxes fills s_kpn_weight with one lane per entry"
#endif

// The per-line tables come from the dynamic LUT pass (GenerateDenseAxisTable in ffx_nss_generate_offset_lut.h): for
// every output column and row, 3 texels of (LR coordinate, KPN byte offset) pairs for the 6 axis offsets and one texel
// with their 6-bit validity, computed once per frame with the arithmetic this pass used to run per workgroup (the
// first wave mapped its 16 columns and 16 rows while the others waited at the barrier). Here 128 lanes copy one texel
// each into LDS.
const int32_t kDenseAxisTableRow = 2 + 64;
void PrepareDenseFilterAxes()
{
    int32_t li = int32_t(gl_LocalInvocationIndex);
    if (li < int32_t(256))  // the 16x16 workgroup has exactly 256 lanes
    {
        int32_t q        = bitfieldExtract(li, int32_t(0), int32_t(8));
        s_kpn_weight[li] = half(max(float(Dequantize(half(q), kKpnQuant)), float(EPS)));
    }
    if (li < (kAxisLdsW + kAxisLdsH) * int32_t(4))
    {
        int32_t   line   = li >> int32_t(2);
        int32_t   j      = li & int32_t(3);
        bool      y_axis = line >= kAxisLdsW;
        int32_t   e      = y_axis ? line - kAxisLdsW : line;
        int32_t   p      = y_axis ? int32_t(gl_WorkGroupID.y) * kAxisLdsH + e : int32_t(gl_WorkGroupID.x) * kAxisLdsW + e;
        p                = min(p, (y_axis ? OutputDims().y : OutputDims().x) - int32_t(1));  // lines past the edge are unused
        uint32_t4 t      = texelFetch(_OffsetLutUint4Tex, int32_t2(p, kDenseAxisTableRow + (y_axis ? int32_t(4) : int32_t(0)) + j), 0);
        if (j < int32_t(3))
        {
            s_axis[line * 6 + j * 2]     = int32_t2(t.xy);
            s_axis[line * 6 + j * 2 + 1] = int32_t2(t.zw);
        }
        else
        {
            s_axis_valid[line] = t.x;
        }
    }
    barrier();
}

void FilterColour(int32_t2 output_px, out half4 m1, out half4 m2, out half4 center_sample)
{
    int32_t col = int32_t(gl_LocalInvocationID.x);
    int32_t row = int32_t(gl_LocalInvocationID.y);

    half4 accum_m1 = half4(0.HF);
    half4 accum_m2 = half4(0.HF);
    center_sample  = half4(0.HF);

    // Selected taps for this pixel's (x, y) validity patterns, in candidate order (see GenerateOffsetLutAxis).
    uint32_t2 sel    = texelFetch(_OffsetLutUint4Tex, int32_t2(int32_t(s_axis_valid_x(col)), int32_t(2 + s_axis_valid_y(row))), 0).xy;
    int32_t   count  = int32_t(sel.y >> uint32_t(24));
    int32_t   col6   = col * 6;
    int32_t   row6   = row * 6;

    int32_t2 lr_tap[9];
    int32_t  kpn_byte[9];
    bool     tap_valid[9];
    bool     tap_center[9];
    for (int32_t k = int32_t(0); k < int32_t(9); ++k)
    {
        uint32_t word = k < int32_t(5) ? sel.x : sel.y;
        int32_t  pos  = (k < int32_t(5) ? k : k - int32_t(5)) * int32_t(6);
        int32_t  ix   = int32_t(bitfieldExtract(word, pos, int32_t(3)));
        int32_t  iy   = int32_t(bitfieldExtract(word, pos + int32_t(3), int32_t(3)));
        int32_t2 ax   = s_axis_x(col6 + ix);
        int32_t2 ay   = s_axis_y(row6 + iy);
        tap_valid[k]  = k < count;
        lr_tap[k]     = int32_t2(ax.x, ay.x);
        // 36 * (kpn_y * kpn_w + kpn_x) + channel, channel = 6 * ix + iy (kCandidateLinear), as in ReadKpnParamsFromBase.
        kpn_byte[k]   = ax.y + ay.y;
        tap_center[k] = bitfieldExtract(word, pos, int32_t(6)) == uint32_t(2 | (2 << 3));  // offset (0, 0) is candidate rank 0
    }
    // Ranks are listed in increasing order and offset (0, 0) is rank 0, so only the first selected tap can be the centre
    // (the select below then gives the same center_sample as testing every tap, with 1 compare instead of 9).

    half4 tap_col[9];
    half  tap_weight[9];
    for (int32_t k = int32_t(0); k < int32_t(9); ++k)
    {
        tap_col[k]    = LoadColourTap4(lr_tap[k]);
        tap_weight[k] = s_kpn_weight[uint32_t(r_coefficients_kpn_bytes.data[kpn_byte[k]])];  // SampleKpnWeight
    }

    center_sample = (tap_valid[0] && tap_center[0]) ? tap_col[0] : half4(0.HF);
    // A missing tap (k >= count) adds +0 to every sum instead of being skipped: its colour is selected to 0 (alpha
    // included) before the multiplies, so its terms are 0 * w = +0 (w > 0) and acc + (+0) = acc for every acc here
    // (the sums start at +0 and can never become -0 under round-to-nearest, NaN and Inf stay as they are). Same bits as
    // the branch, but the taps stay in one basic block (the byte loads are not sunk into per-tap branches).
    for (int32_t k = int32_t(0); k < int32_t(9); ++k)
    {
        half4 c = tap_valid[k] ? tap_col[k] : half4(0.HF);
        accum_m1 += c * tap_weight[k];
        accum_m2 += (c * c) * tap_weight[k];
    }

    NormalizeFilterMoments(accum_m1, accum_m2, m1, m2);
}
#else
// Selected-tap reconstruction from the separable LUT (see GenerateOffsetLutAxis): ranks valid on both axes, lowest
// first, at most 9 - exactly the taps the per-tile LUT packed, in the same order.
void FilterColour(int32_t2 output_px, out half4 m1, out half4 m2, out half4 center_sample)
{
    int32_t2 colour_max = InputDims() - int32_t2(1);
    int32_t2 kpn_max    = KpnDims() - int32_t2(1);
    int32_t2 idx_mod_xy = PackedOffsetLutModulo();
    int32_t2 tile_idx   = PackedOffsetLutTile(output_px, idx_mod_xy);
    int32_t2 lr_base    = int32_t2(floor(float2(output_px) * InvScale()));

    half4 accum_m1 = half4(0.HF);
    half4 accum_m2 = half4(0.HF);
    center_sample  = half4(0.HF);

    uint32_t4 axis_x  = texelFetch(_OffsetLutUint4Tex, int32_t2(tile_idx.x, 0), 0);
    uint32_t4 axis_y  = texelFetch(_OffsetLutUint4Tex, int32_t2(tile_idx.y, 1), 0);
    uint32_t  rank_lo = axis_x.z & axis_y.z;
    uint32_t  rank_hi = axis_x.w & axis_y.w;

    // 1) The (up to) 9 selected taps: rank, then coordinates. 2) All colour and KPN fetches. 3) Accumulation in tap
    // order, skipping the missing taps exactly like AccumulatePackedLrOffsetTap does for invalid LUT entries. Same
    // values and the same sums as one tap at a time, without a memory round trip per tap.
    int32_t2 lr_tap[9];
    int32_t2 kpn_tap[9];
    int32_t  channel[9];
    bool     tap_valid[9];
    bool     tap_center[9];
    for (int32_t k = int32_t(0); k < int32_t(9); ++k)
    {
        tap_valid[k] = (rank_lo | rank_hi) != uint32_t(0);
        int32_t rank;
        if (rank_lo != uint32_t(0))
        {
            rank    = findLSB(rank_lo);
            rank_lo &= rank_lo - uint32_t(1);
        }
        else
        {
            rank    = int32_t(32) + findLSB(rank_hi);  // findLSB(0) = -1: rank 31 for a missing tap (not accumulated)
            rank_hi &= rank_hi - uint32_t(1);
        }
        uint32_t info = kDenseRankInfo[clamp(rank, int32_t(0), int32_t(35))];
        int32_t  ix   = int32_t((info >> uint32_t(8)) & uint32_t(7));
        int32_t  iy   = int32_t((info >> uint32_t(12)) & uint32_t(7));
        int32_t2 lr_offset = int32_t2(bitfieldExtract(int32_t(ix < int32_t(4) ? axis_x.x : axis_x.y), (ix & int32_t(3)) * int32_t(8), int32_t(8)),
                                      bitfieldExtract(int32_t(iy < int32_t(4) ? axis_y.x : axis_y.y), (iy & int32_t(3)) * int32_t(8), int32_t(8)));
        lr_tap[k]     = clamp(lr_base + lr_offset, int32_t2(0), colour_max);
        kpn_tap[k]    = KpnCoordForDenseFilter(lr_tap[k], kpn_max);
        channel[k]    = int32_t(info & uint32_t(0x3F));
        tap_center[k] = rank == int32_t(0);
    }

    half4 tap_col[9];
    half  tap_weight[9];
    for (int32_t k = int32_t(0); k < int32_t(9); ++k)
    {
        tap_col[k]    = LoadColourTap4(lr_tap[k]);
        tap_weight[k] = SampleKpnWeight(channel[k], kpn_tap[k]);
    }

    for (int32_t k = int32_t(0); k < int32_t(9); ++k)
    {
        if (tap_valid[k])
        {
            if (tap_center[k])
            {
                center_sample = tap_col[k];
            }
            accum_m1 += tap_col[k] * tap_weight[k];
            accum_m2 += (tap_col[k] * tap_col[k]) * tap_weight[k];
        }
    }

    NormalizeFilterMoments(accum_m1, accum_m2, m1, m2);
}
#endif  // NSS_POSTPROCESS_AXIS_LDS && !NSS_SUPPORT_TENSOR
#elif NSS_FILTER_MODE == 3
int16_t4 DecodePackedI8x4(uint32_t4 packed, uint32_t shift)
{
    return int16_t4(bitfieldExtract(int32_t4(packed), int32_t(shift), int32_t(8)));
}

int16_t4 DecodePackedTapChannel4(uint32_t4 packed)
{
    return int16_t4((packed >> uint32_t4(16)) & uint32_t4(0x3F));
}

half4 DecodePackedValidMask4(uint32_t4 packed)
{
    return half4((packed >> uint32_t4(22)) & uint32_t4(1));
}

void FilterColour(int32_t2 output_px, out half4 m1, out half4 m2, out half4 center_sample)
{
    int32_t2 colour_max = InputDims() - int32_t2(1);
    int32_t2 kpn_max = KpnDims() - int32_t2(1);
    int32_t2 idx_mod_xy = PackedOffsetLutModulo();
    int32_t2 tile_idx = PackedOffsetLutTile(output_px, idx_mod_xy);
    int32_t2 lr_base = int32_t2(floor(float2(output_px) * InvScale()));

    half4 accum_m1 = half4(0.HF);
    half4 accum_m2 = half4(0.HF);
    center_sample = half4(0.HF);

    uint32_t4 packed_taps = LoadOffsetLutPacked4(tile_idx, int32_t(0));
    int32_t4 lr_tap_x = clamp(int32_t4(lr_base.x) + int32_t4(DecodePackedI8x4(packed_taps, uint32_t(0))), int32_t4(0), int32_t4(colour_max.x));
    int32_t4 lr_tap_y = clamp(int32_t4(lr_base.y) + int32_t4(DecodePackedI8x4(packed_taps, uint32_t(8))), int32_t4(0), int32_t4(colour_max.y));
    int16_t4 tap_x = int16_t4(lr_tap_x);
    int16_t4 tap_y = int16_t4(lr_tap_y);
    int16_t4 kpn_x;
    int16_t4 kpn_y;
    KpnCoordsForDenseFilter4(tap_x, tap_y, int16_t4(0), int16_t4(kpn_max.x), int16_t4(kpn_max.y), kpn_x, kpn_y);

    f16mat4x4 taps;
    NSS_LOAD_TAPS_4(tap_x, tap_y, taps);
    half4 weights = NormalizeSparseKernelWeights4(SampleKpnWeight4(DecodePackedTapChannel4(packed_taps), kpn_x, kpn_y), DecodePackedValidMask4(packed_taps));

    uint32_t4 center_bits = (packed_taps >> uint32_t4(23)) & uint32_t4(1);
    if (center_bits.x != uint32_t(0))
    {
        center_sample = taps[0];
    }
    else if (center_bits.y != uint32_t(0))
    {
        center_sample = taps[1];
    }
    else if (center_bits.z != uint32_t(0))
    {
        center_sample = taps[2];
    }
    else if (center_bits.w != uint32_t(0))
    {
        center_sample = taps[3];
    }
    accum_m1 += taps * weights;
    accum_m2 += NSS_SQ_MAT(taps) * weights;

    NormalizeFilterMoments(accum_m1, accum_m2, m1, m2);
}
#elif (NSS_FILTER_MODE == 0) || (NSS_FILTER_MODE == 1)
#if NSS_FILTER_MODE == 0 && !NSS_SUPPORT_TENSOR && NSS_KPN_SCALE_QUARTER && !NSS_PREPROCESS_HALF_RES_INPUT
// Exact-2x dense filter (QUALITY), weight reads with the per-pixel channel arithmetic hoisted. A tap's channel is
// base + c (base = the pattern's base channel, c a tap constant with c & 3 in {0, 2}), so its KPN word is
// pixel * 9 + (c >> 2) + ((base + (c & 3)) >> 2) and its byte ((base + (c & 3)) & 3): two per-pixel variants, and the
// constant (c >> 2) folds into the load offset. Same words, bytes and dequantisation as SampleKpnWeight4/SampleKpnWeight.
#define NSS_DENSE2X_HOISTED_KPN 1
struct DenseKpnChannels
{
    int32_t hi0;  // (base + 0) >> 2
    int32_t hi2;  // (base + 2) >> 2
    int32_t sh0;  // ((base + 0) & 3) * 8
    int32_t sh2;  // ((base + 2) & 3) * 8
};

DenseKpnChannels MakeDenseKpnChannels(int32_t base)
{
    DenseKpnChannels c;
    c.hi0 = base >> 2;
    c.hi2 = (base + 2) >> 2;
    c.sh0 = (base & 3) << 3;
    c.sh2 = ((base + 2) & 3) << 3;
    return c;
}

// Tap constant C (a literal from kTapCh0/1/2), KPN pixel word base PIX = (y * kpn_w + x) * 9.
#define NSS_DENSE_KPN_Q(PIX, C, CH)                                                                                         \
    bitfieldExtract(int32_t(r_coefficients_kpn_buffer.data[(PIX) + ((C) >> 2) + ((((C) & 3) != 0) ? (CH).hi2 : (CH).hi0)]), \
                    ((((C) & 3) != 0) ? (CH).sh2 : (CH).sh0), 8)
#endif

#if NSS_FILTER_MODE == 1
// Integer tap coordinates of the exact-2x sparse filter: the 2x2 LR block {lo, hi} per axis and the KPN coordinates
// of lo and hi (.x = lo, .y = hi).
struct Sparse2xTaps
{
    int32_t2 lo;
    int32_t2 hi;
    int32_t2 kx;
    int32_t2 ky;
};

int32_t2 Sparse2xKpnCoords(int32_t2 lr, int32_t input_dim, int32_t kpn_dim)
{
    return clamp(min(KpnScaledCoordDiv2(lr, kpn_dim, max(input_dim, 1)), int32_t2(kpn_dim - 1)), int32_t2(0), int32_t2(kpn_dim - 1));
}

Sparse2xTaps Sparse2xTapsForPixel(int32_t2 output_px)
{
    // kTap2x2Dx/Dy hold {d, d + 2} with d = tile - 1 per axis, so the four taps of every pattern are the 2x2 LR block
    // {lo, lo + 1}^2 with lo = (px + tile - 1) >> 1: y-fastest, except pattern 2 which walks it x-fastest. Map the two
    // distinct coordinates per axis to KPN space and to KPN rows once instead of once per tap (same values as the
    // per-tap mapping, without the per-tap divides and multiplies).
    int32_t2     colour_max = InputDims() - int32_t2(1);
    int32_t2     tile_idx   = (output_px + LutOffset()) & int32_t2(1);
    int32_t2     raw_lo     = (output_px + tile_idx - int32_t2(1)) >> 1;
    Sparse2xTaps t;
    t.lo = clamp(raw_lo, int32_t2(0), colour_max);
    t.hi = clamp(raw_lo + int32_t2(1), int32_t2(0), colour_max);
    t.kx = Sparse2xKpnCoords(int32_t2(t.lo.x, t.hi.x), InputDims().x, KpnDims().x);
    t.ky = Sparse2xKpnCoords(int32_t2(t.lo.y, t.hi.y), InputDims().y, KpnDims().y);
    return t;
}

void FilterColourSparse2x(int32_t2 output_px, Sparse2xTaps sparse_taps, out half4 m1, out half4 m2, out half4 center_sample)
#else
void FilterColour(int32_t2 output_px, out half4 m1, out half4 m2, out half4 center_sample)
#endif
{
    // NSS v1: KPN tensor is at 1/4 of dataGraph resolution; map input-space tap coords to KPN space.
    //-------------------------------------------------------------------------
    // Spatial KPN filtering:
    // - pick tap pattern from LUT (tile-dependent)
    // - fetch tap colours from LR colour
    // - fetch per-tap weights from KPN tensor
    // - compute first/second moments (m1/m2) for history clamp
    //-------------------------------------------------------------------------
    int32_t2 colour_max    = InputDims() - int32_t2(1);
    int32_t2 kpn_max       = KpnDims() - int32_t2(1);
    int16_t4 colour_max_x4 = int16_t4(colour_max.x);
    int16_t4 colour_max_y4 = int16_t4(colour_max.y);
    int16_t4 kpn_max_x4    = int16_t4(kpn_max.x);
    int16_t4 kpn_max_y4    = int16_t4(kpn_max.y);
    int16_t4 zero4         = int16_t4(0);
    int16_t2 out_px16      = int16_t2(output_px);
    int32_t2 tile_idx      = (output_px + LutOffset()) & int32_t2(1);
    int32_t  lut_idx       = (tile_idx.y << int32_t(1)) + tile_idx.x;

    half4 accum_m1 = half4(0.HF);
    half4 accum_m2 = half4(0.HF);
    center_sample  = half4(0.HF);

#if NSS_FILTER_MODE == 1
    int32_t2 lo        = sparse_taps.lo;
    int32_t2 hi        = sparse_taps.hi;
    int32_t2 kx        = sparse_taps.kx;
    int32_t2 ky        = sparse_taps.ky;
    bool     x_fastest = lut_idx == int32_t(2);
    int16_t4 tap_x     = x_fastest ? int16_t4(lo.x, hi.x, lo.x, hi.x) : int16_t4(lo.x, lo.x, hi.x, hi.x);
    int16_t4 tap_y     = x_fastest ? int16_t4(lo.y, lo.y, hi.y, hi.y) : int16_t4(lo.y, hi.y, lo.y, hi.y);
    int16_t4 kpn_ch    = kTap2x2Ch[lut_idx];

    f16mat4x4 taps;
    NSS_LOAD_TAPS_4(tap_x, tap_y, taps);
#if !NSS_SUPPORT_TENSOR
    // Word index of (x, y, channel group): (y * kpn_w + x) * 4 + channel / 4, rows computed once per distinct y.
    int32_t2 row     = ky * int32_t2(KpnDims().x);
    int32_t4 pix_x   = x_fastest ? int32_t4(kx.x, kx.y, kx.x, kx.y) : int32_t4(kx.x, kx.x, kx.y, kx.y);
    int32_t4 pix_row = x_fastest ? int32_t4(row.x, row.x, row.y, row.y) : int32_t4(row.x, row.y, row.x, row.y);
    int32_t4 word    = (pix_row + pix_x) * int32_t4(kKpnChannels / 4) + (int32_t4(kpn_ch) >> 2);
    int32_t4 q       = int32_t4(ReadKpnWordChannel(word.x, int32_t(kpn_ch.x)),
                          ReadKpnWordChannel(word.y, int32_t(kpn_ch.y)),
                          ReadKpnWordChannel(word.z, int32_t(kpn_ch.z)),
                          ReadKpnWordChannel(word.w, int32_t(kpn_ch.w)));
    half4    weights = half4(max(float4(Dequantize(half4(q), kKpnQuant)), float4(float(EPS))));
#else
    int16_t4 kpn_x   = x_fastest ? int16_t4(kx.x, kx.y, kx.x, kx.y) : int16_t4(kx.x, kx.x, kx.y, kx.y);
    int16_t4 kpn_y   = x_fastest ? int16_t4(ky.x, ky.x, ky.y, ky.y) : int16_t4(ky.x, ky.y, ky.x, ky.y);
    half4    weights = SampleKpnWeight4(kpn_ch, kpn_x, kpn_y);
#endif
    weights       = NormalizeSparseKernelWeights4(weights);
    center_sample = (lut_idx == int32_t(3)) ? taps[0] : half4(0.HF);
    accum_m1 += taps * weights;
    accum_m2 += NSS_SQ_MAT(taps) * weights;
#else
    KernelPattern lut = kKernelLut[lut_idx];

    int16_t4 tap_x0 = clamp((int16_t4(out_px16.x + lut.base_offset.x) + kTapDx0) >> int16_t(1), zero4, colour_max_x4);
    int16_t4 tap_y0 = clamp((int16_t4(out_px16.y + lut.base_offset.y) + kTapDy0) >> int16_t(1), zero4, colour_max_y4);
#if NSS_PREPROCESS_HALF_RES_INPUT
    int16_t4 kpn_x0 = clamp(KpnCoordFromInputCoord4(tap_x0, InputDims().x, KpnDims().x), zero4, kpn_max_x4);
    int16_t4 kpn_y0 = clamp(KpnCoordFromInputCoord4(tap_y0, InputDims().y, KpnDims().y), zero4, kpn_max_y4);
#else
#if NSS_DENSE2X_HOISTED_KPN
    // tap >= 0, so clamp(min(tap >> 2, kpn_max), 0, kpn_max) == min(tap >> 2, kpn_max).
    int16_t4 kpn_x0 = min(tap_x0 >> int16_t(2), kpn_max_x4);
    int16_t4 kpn_y0 = min(tap_y0 >> int16_t(2), kpn_max_y4);
#else
    int16_t4 kpn_x0 = clamp(KpnCoordFromScaledTap4(tap_x0, KpnScale().x, KpnDims().x), zero4, kpn_max_x4);
    int16_t4 kpn_y0 = clamp(KpnCoordFromScaledTap4(tap_y0, KpnScale().y, KpnDims().y), zero4, kpn_max_y4);
#endif
#endif

    f16mat4x4 taps0;
    NSS_LOAD_TAPS_4(tap_x0, tap_y0, taps0);
#if NSS_DENSE2X_HOISTED_KPN
    const int32_t    kpn_words = kKpnChannels / 4;
    DenseKpnChannels kch       = MakeDenseKpnChannels(int32_t(lut.base_channel));
    // (y * kpn_w + x) * 9 as y * (kpn_w * 9) + x * 9: the row and column parts are shared by the 3x3 block.
    int32_t          kpn_row   = KpnDims().x * kpn_words;
    int32_t4         pix0      = int32_t4(kpn_y0) * int32_t4(kpn_row) + int32_t4(kpn_x0) * int32_t4(kpn_words);
    int32_t4         q0        = int32_t4(NSS_DENSE_KPN_Q(pix0.x, 0, kch), NSS_DENSE_KPN_Q(pix0.y, 2, kch), NSS_DENSE_KPN_Q(pix0.z, 4, kch), NSS_DENSE_KPN_Q(pix0.w, 12, kch));
    half4            w0        = half4(max(float4(Dequantize(half4(q0), kKpnQuant)), float4(float(EPS))));
#else
    half4 w0 = SampleKpnWeight4(kTapCh0 + int16_t4(lut.base_channel), kpn_x0, kpn_y0);
#endif
    accum_m1 += taps0 * w0;
    accum_m2 += NSS_SQ_MAT(taps0) * w0;

    int16_t4 tap_x1 = clamp((int16_t4(out_px16.x + lut.base_offset.x) + kTapDx1) >> int16_t(1), zero4, colour_max_x4);
    int16_t4 tap_y1 = clamp((int16_t4(out_px16.y + lut.base_offset.y) + kTapDy1) >> int16_t(1), zero4, colour_max_y4);
#if NSS_PREPROCESS_HALF_RES_INPUT
    int16_t4 kpn_x1 = clamp(KpnCoordFromInputCoord4(tap_x1, InputDims().x, KpnDims().x), zero4, kpn_max_x4);
    int16_t4 kpn_y1 = clamp(KpnCoordFromInputCoord4(tap_y1, InputDims().y, KpnDims().y), zero4, kpn_max_y4);
#else
#if NSS_DENSE2X_HOISTED_KPN
    int16_t4 kpn_x1 = min(tap_x1 >> int16_t(2), kpn_max_x4);
    int16_t4 kpn_y1 = min(tap_y1 >> int16_t(2), kpn_max_y4);
#else
    int16_t4 kpn_x1 = clamp(KpnCoordFromScaledTap4(tap_x1, KpnScale().x, KpnDims().x), zero4, kpn_max_x4);
    int16_t4 kpn_y1 = clamp(KpnCoordFromScaledTap4(tap_y1, KpnScale().y, KpnDims().y), zero4, kpn_max_y4);
#endif
#endif

    f16mat4x4 taps1;
    NSS_LOAD_TAPS_4(tap_x1, tap_y1, taps1);
#if NSS_DENSE2X_HOISTED_KPN
    int32_t4 pix1 = int32_t4(kpn_y1) * int32_t4(kpn_row) + int32_t4(kpn_x1) * int32_t4(kpn_words);
    int32_t4 q1   = int32_t4(NSS_DENSE_KPN_Q(pix1.x, 14, kch), NSS_DENSE_KPN_Q(pix1.y, 16, kch), NSS_DENSE_KPN_Q(pix1.z, 24, kch), NSS_DENSE_KPN_Q(pix1.w, 26, kch));
    half4    w1   = half4(max(float4(Dequantize(half4(q1), kKpnQuant)), float4(float(EPS))));
#else
    half4 w1      = SampleKpnWeight4(kTapCh1 + int16_t4(lut.base_channel), kpn_x1, kpn_y1);
#endif
    center_sample = (lut_idx == int32_t(3)) ? taps1[0] : half4(0.HF);
    accum_m1 += taps1 * w1;
    accum_m2 += NSS_SQ_MAT(taps1) * w1;

    int32_t2 tap2    = output_px + int32_t2(lut.base_offset) + int32_t2(kTapD2);
    int32_t2 lr_tap2 = clamp(tap2 >> int32_t(1), int32_t2(0), colour_max);
#if NSS_PREPROCESS_HALF_RES_INPUT
    int32_t2 kpn_tap2 =
        clamp(int32_t2(KpnCoordFromInputCoord(lr_tap2.x, InputDims().x, KpnDims().x), KpnCoordFromInputCoord(lr_tap2.y, InputDims().y, KpnDims().y)),
              int32_t2(0),
              kpn_max);
#else
#if NSS_DENSE2X_HOISTED_KPN
    int32_t2 kpn_tap2 = min(lr_tap2 >> int32_t(2), kpn_max);
#else
    int32_t2 kpn_tap2 =
        clamp(int32_t2(KpnCoordFromScaledTap(lr_tap2.x, KpnScale().x, KpnDims().x), KpnCoordFromScaledTap(lr_tap2.y, KpnScale().y, KpnDims().y)),
              int32_t2(0),
              kpn_max);
#endif
#endif
    half4 tap2_col = LoadColourTap4(lr_tap2);
#if NSS_DENSE2X_HOISTED_KPN
    int32_t pix2   = kpn_tap2.y * kpn_row + kpn_tap2.x * kpn_words;
    half    tap2_w = half(max(float(Dequantize(half(NSS_DENSE_KPN_Q(pix2, 28, kch)), kKpnQuant)), float(EPS)));
#else
    half  tap2_w   = SampleKpnWeight(int32_t(lut.base_channel + kTapCh2), kpn_tap2);
#endif
    accum_m1 += tap2_col * tap2_w;
    accum_m2 += (tap2_col * tap2_col) * tap2_w;
#endif

    NormalizeFilterMoments(accum_m1, accum_m2, m1, m2);
}
#else
#error "Unsupported NSS_FILTER_MODE"
#endif

#if NSS_FILTER_MODE == 1
void FilterColour(int32_t2 output_px, out half4 m1, out half4 m2, out half4 center_sample)
{
    FilterColourSparse2x(output_px, Sparse2xTapsForPixel(output_px), m1, m2, center_sample);
}
#endif

#undef NSS_LOAD_TAPS_4
#undef NSS_SQ_MAT

// ---------------------------------------------------------------------------
// History warp with motion + optional Catmull-Rom
// ---------------------------------------------------------------------------
void WarpHistory(half2 motion, float2 uv, out half4 warped_colour, out half onscreen)
{
    // Reproject history with motion; report if sample stayed on-screen.
    float2 reproj_uv = uv + (float2(motion) * InvOutputDims());
    onscreen         = half(all(greaterThanEqual(reproj_uv, float2(0.0))) && all(lessThanEqual(reproj_uv, float2(1.0))));
#ifdef ENABLE_HISTORY_CATMULL
    warped_colour = LoadWarpedHistoryCatmullPost(reproj_uv);
#else
    warped_colour = LoadWarpedHistoryPost(reproj_uv);
#endif
}

// ---------------------------------------------------------------------------
// Statistical history clamp (AABB)
// ---------------------------------------------------------------------------
void ClampHistoryToStats(half4 m1, half4 m2, half4 warped_history, half theta, half gamma, half reset, half onscreen, out half4 rectified)
{
    half4 sigma_4         = half4(sqrt(max(float4(abs(m2 - m1 * m1)), float4(float(EPS))))) * half4(gamma);
    sigma_4.a             = 0.HF;
    half4 aabb_min        = m1 - sigma_4;
    half4 aabb_max        = m1 + sigma_4;
    half4 history_clamped = mix(m1, clamp(warped_history, aabb_min, aabb_max), half4(reset));
    rectified             = mix(history_clamped, warped_history, half4(theta * onscreen * reset));
}

// ---------------------------------------------------------------------------
// Clamp to invertible Karis range and convert back to linear
// ---------------------------------------------------------------------------
half3 ClampToInvertibleRange(half3 accumulated, half inv_exposure)
{
    half3 clamped = clamp(accumulated, half3(0.HF), half3(1.HF - EPS));
    // SafeColour clamps to [0, MAX_FP16] itself; the MaxHalf (min with 65504) that used to precede it was a no-op (see
    // PostprocessPixelFiltered).
    return SafeColour(InverseTonemap(clamped) * half3(inv_exposure));
}

// ---------------------------------------------------------------------------
// Upscaled output write + history store (resource bindings declared in ffx_nss_postprocess.h)
// ---------------------------------------------------------------------------
half3 SimulateR11G11B10PrecisionPost(half3 rgb)
{
    const float    epsilon  = 1e-12;
    const int32_t3 m_bits   = int32_t3(6, 6, 5);
    const int32_t  exp_bits = 5;
    const float    bias     = float((1 << (exp_bits - 1)) - 1);

    float3 val           = max(float3(rgb), float3(epsilon));
    float3 exp_unclamped = floor(log2(val));
    float3 exp_clipped   = clamp(exp_unclamped, -bias, bias + 1.0);
    float3 mant          = val / exp2(exp_clipped) - 1.0;

    float3 scale    = float3(int32_t3(1) << m_bits);
    float3 exp_max  = float3((1 << exp_bits) - 1);
    float3 rgb_exp  = clamp(exp_clipped + bias, float3(0.0), exp_max) - bias;
    float3 rgb_mant = clamp(round(mant * scale), float3(0.0), scale - 1.0) / scale;
    return half3((1.0 + rgb_mant) * exp2(rgb_exp));
}

// Rounds a non-negative colour to the nearest R11G11B10_FLOAT value (ties to even), so that storing it to the R11G11B10
// history is exact whatever rounding the hardware's format conversion uses. AMD GPUs (RADV, e.g. the Steam Deck's Van
// Gogh) convert to the 11/10-bit floats by truncation; the history is re-read and re-stored every frame, so with
// integer-pixel motion (the Catmull-Rom history fetch then returns one texel unchanged) the truncation compounds frame
// after frame into dark pixels wherever no fresh centre sample is blended in, i.e. dark columns in flat bright areas.
// On hardware that already rounds to nearest this is a no-op (the result is the value that hardware would store).
half3 RoundToR11G11B10Nearest(half3 rgb)
{
    // Works on the fp32 value (the colour is fp32 in the non-FFX_HALF permutations), so the result is the single
    // correctly rounded R11G11B10 value. Normal range: round the fp32 mantissa to 6/6/5 bits, ties to even (a carry
    // moves to the next exponent). Below 2^-14 the formats are denormal with a fixed step of 2^-20 / 2^-20 / 2^-19.
    const uint32_t3 shift     = uint32_t3(17u, 17u, 18u);
    const float3    step      = float3(9.5367431640625e-07, 9.5367431640625e-07, 1.9073486328125e-06);
    const float3    max_value = float3(65024.0, 65024.0, 64512.0);  // largest finite values, (2 - 2^-m) * 2^15
    float3          x         = max(float3(rgb), float3(0.0));
    uint32_t3       bits      = floatBitsToUint(x);
    bits += ((uint32_t3(1u) << (shift - uint32_t3(1u))) - uint32_t3(1u)) + ((bits >> shift) & uint32_t3(1u));
    bits &= ~((uint32_t3(1u) << shift) - uint32_t3(1u));
    float3 r = mix(uintBitsToFloat(bits), roundEven(x * (1.0 / step)) * step, lessThan(x, float3(6.103515625e-05)));
    return half3(min(r, max_value));
}

void WriteUpsampledColour(int32_t2 pixel, half3 colour)
{
    half3 to_write = SafeColour(colour);
#ifdef SIMULATE_R11G11B10_BEFORE_WRITE
    to_write = SimulateR11G11B10PrecisionPost(to_write);
#endif
#if defined(NSS_BIND_UAV_UPSCALED_OUTPUT)
    imageStore(rw_upscaled_output, pixel, half4(to_write, 1.0));
#elif defined(NSS_BIND_RENDER_TARGET_UPSCALED_OUTPUT)
    rw_upscaled_output = half4(to_write, 1.0);
#else
#error "The output texture is not defined!"
#endif
}

#if (defined(NSS_BIND_UAV_HISTORY_UPSCALED_COLOR) || defined(NSS_BIND_RENDER_TARGET_HISTORY_UPSCALED_COLOR)) && MANAGE_HISTORY
void StorePrevUpscaledColor(int32_t2 iPxPos, half3 colour)
{
    half3 to_write = SafeColour(colour);
#ifdef SIMULATE_R11G11B10_BEFORE_WRITE
    to_write = SimulateR11G11B10PrecisionPost(to_write);
#else
    to_write = RoundToR11G11B10Nearest(to_write);
#endif
#if defined(NSS_BIND_UAV_HISTORY_UPSCALED_COLOR)
    imageStore(rw_prev_upscaled_color, iPxPos, half4(to_write, 1.0));
#else
    rw_prev_upscaled_color = half4(to_write, 1.0);
#endif
}
#endif

// ---------------------------------------------------------------------------
// Output write
// ---------------------------------------------------------------------------
void WriteColourOut(int32_t2 coord, half3 out_linear)
{
    WriteUpsampledColour(coord, out_linear);
}

#if NSS_POSTPROCESS
// ---------------------------------------------------------------------------
// Main postprocess entry point
// ---------------------------------------------------------------------------
// motion: LoadMotionPost(output_pixel), passed in so a thread handling several pixels can share it.
#if NSS_FILTER_MODE == 1
#define NSS_POST_FILTER_ARGS , Sparse2xTaps sparse_taps
#else
#define NSS_POST_FILTER_ARGS
#endif
// Steps 3-6 of the postprocess for one pixel, given its filtered moments and centre sample.
// no_center: the caller knows center_sample is zero (a pattern without the centre tap; a compile-time constant).
void PostprocessPixelFiltered(int32_t2 output_px, half2 motion, half4 m1, half4 m2, half4 center_sample, bool store, bool no_center)
{
    half   exposure = half(Exposure().x);
    half   reset    = half(Reset());
    float2 uv       = (float2(output_px) + float2(0.5)) * InvOutputDims();

    //-------------------------------------------------------------------------
    // 3) Read temporal controls and reproject history
    //-------------------------------------------------------------------------
    half theta;
    half alpha;
    half gamma;
    SampleTemporalParams(uv, theta, alpha, gamma);

    half4 warped_colour;
    half  onscreen;
    WarpHistory(motion, uv, warped_colour, onscreen);

    //-------------------------------------------------------------------------
    // 4) Clamp and rectify history
    //-------------------------------------------------------------------------
    half4 rectified;
    ClampHistoryToStats(m1, m2, warped_colour, theta, gamma, reset, onscreen, rectified);

    //-------------------------------------------------------------------------
    // 5) Tonemapped accumulation
    //-------------------------------------------------------------------------
    // Tonemap4 (and SafeColour) clamp to [0, MAX_FP16 = 65504] first, so a preceding MaxHalf (min with 65504) changes
    // nothing for any input: finite and +-Inf trivially; for NaN the clamp is one v_med3_f32, which returns
    // min3(...) = 0 when an input is NaN, and the extra v_min_f32 the MaxHalf compiled to came after it. Dropped.
    half4 rectified_tm4 = Tonemap4(rectified * half4(exposure));
    half4 accumulated4;
#if !FFX_HALF
    if (no_center)
    {
        // centre_sample == 0: the blend weight alpha * 0 * reset is +0 (alpha > 0, reset >= 0) and the blend target
        // Tonemap4(0) is +0, so mix(x, 0, +0) = x + (+0 * -x) = x + (+0 or -0): exactly x + 0.0 for the finite x here
        // (Tonemap4 output; -0 becomes +0 in both). `precise` keeps the add (and so the -0 -> +0) from being folded.
        precise half4 kept = rectified_tm4 + half4(0.HF);
        accumulated4       = kept;
    }
    else
#endif
    {
        half  learnt_masked_alpha = alpha * center_sample.a * reset;
        half4 colour_to_accum_tm4 = Tonemap4(center_sample * half4(exposure));
        accumulated4              = mix(rectified_tm4, colour_to_accum_tm4, half4(learnt_masked_alpha));
    }

    //-------------------------------------------------------------------------
    // 6) Convert back to linear and write outputs
    //-------------------------------------------------------------------------
    half3 out_linear = ClampToInvertibleRange(accumulated4.rgb, half(Exposure().y));
    if (store)
    {
        WriteColourOut(output_px, out_linear);
#if MANAGE_HISTORY && (defined(NSS_BIND_UAV_HISTORY_UPSCALED_COLOR) || defined(NSS_BIND_RENDER_TARGET_HISTORY_UPSCALED_COLOR))
        StorePrevUpscaledColor(output_px, out_linear);
#endif
    }
}

void PostprocessPixel(int32_t2 output_pixel, half2 motion NSS_POST_FILTER_ARGS)
{
    //-------------------------------------------------------------------------
    // 2) Spatial filtering (KPN) + local moments
    //-------------------------------------------------------------------------
    half4 m1;
    half4 m2;
    half4 center_sample;
#if NSS_FILTER_MODE == 1
    FilterColourSparse2x(output_pixel, sparse_taps, m1, m2, center_sample);
#else
    FilterColour(output_pixel, m1, m2, center_sample);
#endif
    PostprocessPixelFiltered(output_pixel, motion, m1, m2, center_sample, true, false);
}

void Postprocess(int32_t2 output_pixel)
{
#if NSS_FILTER_MODE == 2 && NSS_POSTPROCESS_AXIS_LDS && !NSS_SUPPORT_TENSOR
    PrepareDenseFilterAxes();  // whole workgroup, before the guard (barrier)
#endif
    //-------------------------------------------------------------------------
    // 1) Dispatch guard and setup
    //-------------------------------------------------------------------------
    if (any(greaterThanEqual(output_pixel, OutputDims())))
    {
        return;
    }
#if NSS_FILTER_MODE == 1
    PostprocessPixel(output_pixel, LoadMotionPost(output_pixel), Sparse2xTapsForPixel(output_pixel));
#else
    PostprocessPixel(output_pixel, LoadMotionPost(output_pixel));
#endif
}

#if NSS_POSTPROCESS_QUAD
// One thread per 2x2 output quad (exact-2x sparse filter, compute path). At 2x the four pixels of a quad map to the
// same input pixel, so the nearest-offset decode and dilated motion fetch are done once; the per-pixel work is
// unchanged. Falls back to per-pixel motion when the input coordinates differ (non-2x dispatch sizes).
void PostprocessQuad(int32_t2 quad)
{
    int32_t2 base = quad * int32_t2(2);
    if (any(greaterThanEqual(base, OutputDims())))
    {
        return;
    }

    int32_t2 ic00 = MotionPostInputCoord(base);
    int32_t2 ic10 = MotionPostInputCoord(base + int32_t2(1, 0));
    int32_t2 ic01 = MotionPostInputCoord(base + int32_t2(0, 1));
    int32_t2 ic11 = MotionPostInputCoord(base + int32_t2(1, 1));
    half2    mv00 = LoadMotionPostAtInput(ic00);
    half2    mv10 = mv00;
    half2    mv01 = mv00;
    half2    mv11 = mv00;
    if (any(notEqual(ic10, ic00)) || any(notEqual(ic01, ic00)) || any(notEqual(ic11, ic00)))
    {
        mv10 = LoadMotionPostAtInput(ic10);
        mv01 = LoadMotionPostAtInput(ic01);
        mv11 = LoadMotionPostAtInput(ic11);
    }

    // Tap coordinates: pixel base + s has tile (s + LutOffset) & 1 (uniform per slot) and
    // raw_lo = quad + ((s + tile - 1) >> 1), i.e. quad - 1 when s + tile == 0 and quad otherwise; so the LR/KPN
    // coordinates of all four pixels come from {quad - 1, quad, quad + 1} per axis, mapped once per quad.
    int32_t2 colour_max = InputDims() - int32_t2(1);
    int32_t2 u0         = clamp(quad - int32_t2(1), int32_t2(0), colour_max);
    int32_t2 u1         = clamp(quad, int32_t2(0), colour_max);
    int32_t2 u2         = clamp(quad + int32_t2(1), int32_t2(0), colour_max);
    int32_t2 kx01       = Sparse2xKpnCoords(int32_t2(u0.x, u1.x), InputDims().x, KpnDims().x);
    int32_t2 kx12       = int32_t2(kx01.y, Sparse2xKpnCoords(int32_t2(u2.x), InputDims().x, KpnDims().x).x);
    int32_t2 ky01       = Sparse2xKpnCoords(int32_t2(u0.y, u1.y), InputDims().y, KpnDims().y);
    int32_t2 ky12       = int32_t2(ky01.y, Sparse2xKpnCoords(int32_t2(u2.y), InputDims().y, KpnDims().y).x);
    bvec2    low0       = equal(LutOffset() & int32_t2(1), int32_t2(0));  // slot 0 uses (quad - 1, quad)
    // slot 1 always has s + tile >= 1, so it uses (quad, quad + 1)

    Sparse2xTaps t00, t10, t01, t11;
    int32_t2     lo_s0 = int32_t2(low0.x ? u0.x : u1.x, low0.y ? u0.y : u1.y);
    int32_t2     hi_s0 = int32_t2(low0.x ? u1.x : u2.x, low0.y ? u1.y : u2.y);
    int32_t2     kx_s0 = low0.x ? kx01 : kx12;
    int32_t2     ky_s0 = low0.y ? ky01 : ky12;
    t00.lo = lo_s0;                     t00.hi = hi_s0;                     t00.kx = kx_s0; t00.ky = ky_s0;
    t10.lo = int32_t2(u1.x, lo_s0.y);   t10.hi = int32_t2(u2.x, hi_s0.y);   t10.kx = kx12;  t10.ky = ky_s0;
    t01.lo = int32_t2(lo_s0.x, u1.y);   t01.hi = int32_t2(hi_s0.x, u2.y);   t01.kx = kx_s0; t01.ky = ky12;
    t11.lo = u1;                        t11.hi = u2;                        t11.kx = kx12;  t11.ky = ky12;

    bvec2 inside = lessThan(base + int32_t2(1), OutputDims());
    PostprocessPixel(base, mv00, t00);
    if (inside.x)
    {
        PostprocessPixel(base + int32_t2(1, 0), mv10, t10);
    }
    if (inside.y)
    {
        PostprocessPixel(base + int32_t2(0, 1), mv01, t01);
    }
    if (inside.x && inside.y)
    {
        PostprocessPixel(base + int32_t2(1, 1), mv11, t11);
    }
}
#endif  // NSS_POSTPROCESS_QUAD

#if NSS_POSTPROCESS_QUAD_DENSE2X
// Exact-2x dense filter (QUALITY, compute path), one thread per group of 2x2 output pixels that share their 3x3 LR taps.
// A pixel's taps are clamp(((px + off) >> 1) - 1 + {0, 1, 2}) per axis (off = LutOffset), so the pixels
// px = 2g - (off & 1) + s, s in {0, 1}, all use LR {g - 1, g, g + 1}, and s is their LUT tile, i.e. a compile-time
// pattern per slot. The 9 colours, their squares, the LR/KPN coordinates and KPN pixel words are computed once per
// group; per pixel the KPN reads take a constant word offset and byte. Every pixel then accumulates the same values in
// the same order as FilterColour (mode 0) and runs the unchanged steps 3-6. ffx_nss.cpp dispatches W / 2 + 1 groups
// per axis.
#if NSS_FILTER_MODE == 0 && NSS_DENSE2X_HOISTED_KPN
// KPN byte of tap k = (i, j) of slot (sx, sy): channel base + c, c = 12 i + 2 j, base = kKernelLut[sy * 2 + sx].base_channel
// in {7, 1, 6, 0}. Bases 0 and 1 read KPN word c >> 2, bases 6 and 7 word (c + 6) >> 2: the group loads these two words
// per tap once (kw[2k], kw[2k + 1]) and every slot takes its byte.
// Exact build: the words are pre-XORed with 0x80 per byte, so a byte is u = q + 128 (unsigned) and converts with one
// v_cvt_f32_ubyte; Dequantize(q) = (q + 127) * s = (u - 1) * s, and fma(u, s, -s) rounds that same exact product once,
// i.e. bit-identical to the (q + 127) then * s of SampleKpnWeight.
#if FFX_HALF
#define NSS_QD2X_W(K, C) half(max(float(Dequantize(half(bitfieldExtract(int32_t(kw[2 * (K) + (base >= 6 ? 1 : 0)]), (((C) + base) & 3) << 3, 8)), kKpnQuant)), float(EPS)))
#else
#define NSS_QD2X_W(K, C) max(fma(float((kw[2 * (K) + (base >= 6 ? 1 : 0)] >> uint32_t((((C) + base) & 3) << 3)) & 0xFFu), kpn_s, -kpn_s), float(EPS))
#endif

void Dense2xQuadPixel(int32_t2 output_px, int32_t sx, int32_t sy, half2 motion, half4 col[9], half4 sq[9], uint32_t kw[18], bool store)
{
    // Tap (i, j) is element k = i * 3 + j: i the x index, j the y index.
    const int32_t base  = int32_t(7) - int32_t(6) * sx - sy;
    const float   kpn_s = float(kKpnQuant.x);

    half4 accum_m1 = half4(0.HF);
    half4 accum_m2 = half4(0.HF);
    half4 center_sample;

    // Taps (dx, dy) = (0, 0), (0, 2), (0, 4), (2, 0): channels base + 0, 2, 4, 12.
    f16mat4x4 taps0 = f16mat4x4(col[0], col[1], col[2], col[3]);
    f16mat4x4 sq0   = f16mat4x4(sq[0], sq[1], sq[2], sq[3]);
    half4     w0    = half4(NSS_QD2X_W(0, 0), NSS_QD2X_W(1, 2), NSS_QD2X_W(2, 4), NSS_QD2X_W(3, 12));
    accum_m1 += taps0 * w0;
    accum_m2 += sq0 * w0;

    // (2, 2), (2, 4), (4, 0), (4, 2): channels base + 14, 16, 24, 26.
    f16mat4x4 taps1 = f16mat4x4(col[4], col[5], col[6], col[7]);
    f16mat4x4 sq1   = f16mat4x4(sq[4], sq[5], sq[6], sq[7]);
    half4     w1    = half4(NSS_QD2X_W(4, 14), NSS_QD2X_W(5, 16), NSS_QD2X_W(6, 24), NSS_QD2X_W(7, 26));
    // Only pattern 3 (slot (1, 1)) has the centre tap, taps1[0] = LR (g, g).
    center_sample = (sx == 1 && sy == 1) ? taps1[0] : half4(0.HF);
    accum_m1 += taps1 * w1;
    accum_m2 += sq1 * w1;

    // (4, 4): channel base + 28.
    half tap2_w = half(NSS_QD2X_W(8, 28));
    accum_m1 += col[8] * tap2_w;
    accum_m2 += sq[8] * tap2_w;

    half4 m1;
    half4 m2;
    NormalizeFilterMoments(accum_m1, accum_m2, m1, m2);
    PostprocessPixelFiltered(output_px, motion, m1, m2, center_sample, store, !(sx == 1 && sy == 1));
}

void PostprocessQuadDense2x(int32_t2 g)
{
    int32_t2 o    = LutOffset() & int32_t2(1);
    int32_t2 base = g * int32_t2(2) - o;  // slot (0, 0); slot s is base + s
    int32_t2 dims = OutputDims();
    if (any(greaterThanEqual(base, dims)))
    {
        return;
    }
    bvec2 in0 = greaterThanEqual(base, int32_t2(0));               // slot 0 inside (only base = -1 is outside)
    bvec2 in1 = lessThan(base + int32_t2(1), dims);                // slot 1 inside

    // Motion: per pixel input coordinate as LoadMotionPost; one fetch per distinct coordinate.
    int32_t2 ic0 = MotionPostInputCoord(base);
    int32_t2 ic1 = MotionPostInputCoord(base + int32_t2(1));
    half2    mv00 = LoadMotionPostAtInput(ic0);
    half2    mv10 = mv00;
    half2    mv01 = mv00;
    half2    mv11 = mv00;
    if (ic1.x != ic0.x)
    {
        mv10 = LoadMotionPostAtInput(int32_t2(ic1.x, ic0.y));
    }
    if (ic1.y != ic0.y)
    {
        mv01 = LoadMotionPostAtInput(int32_t2(ic0.x, ic1.y));
        mv11 = (ic1.x != ic0.x) ? LoadMotionPostAtInput(ic1) : mv01;
    }
    else
    {
        mv11 = mv10;
    }

    // Shared taps: LR {g - 1, g, g + 1} per axis, clamped; KPN coordinate lr >> 2 (non-negative), clamped.
    int32_t2 colour_max = InputDims() - int32_t2(1);
    int32_t2 kpn_max    = KpnDims() - int32_t2(1);
    int32_t  kpn_words  = kKpnChannels / 4;
    int32_t  kpn_row    = KpnDims().x * kpn_words;
    int32_t  lx[3];
    int32_t  ly[3];
    int32_t  px_word[3];
    int32_t  py_word[3];
    for (int32_t i = 0; i < 3; ++i)
    {
        lx[i]      = clamp(g.x - 1 + i, int32_t(0), colour_max.x);
        ly[i]      = clamp(g.y - 1 + i, int32_t(0), colour_max.y);
        px_word[i] = min(lx[i] >> int32_t(2), kpn_max.x) * kpn_words;
        py_word[i] = min(ly[i] >> int32_t(2), kpn_max.y) * kpn_row;
    }
    half4   col[9];
    half4   sq[9];
    uint32_t kw[18];
    for (int32_t i = 0; i < 3; ++i)
    {
        for (int32_t j = 0; j < 3; ++j)
        {
            col[i * 3 + j] = LoadColourTap4(int32_t2(lx[i], ly[j]));
            sq[i * 3 + j]  = col[i * 3 + j] * col[i * 3 + j];
            int32_t  pix   = py_word[j] + px_word[i];
            int32_t  c     = 12 * i + 2 * j;
#if FFX_HALF
            const uint32_t flip = 0u;
#else
            const uint32_t flip = 0x80808080u;
#endif
            kw[2 * (i * 3 + j)]     = r_coefficients_kpn_buffer.data[pix + (c >> 2)] ^ flip;
            kw[2 * (i * 3 + j) + 1] = r_coefficients_kpn_buffer.data[pix + ((c + 6) >> 2)] ^ flip;
        }
    }

    // All four pixels unconditionally (out-of-range pixels read clamped data), only the stores are guarded, so the
    // compiler can overlap one pixel's history fetches with another's arithmetic.
    Dense2xQuadPixel(base, 0, 0, mv00, col, sq, kw, in0.x && in0.y);
    Dense2xQuadPixel(base + int32_t2(1, 0), 1, 0, mv10, col, sq, kw, in1.x && in0.y);
    Dense2xQuadPixel(base + int32_t2(0, 1), 0, 1, mv01, col, sq, kw, in0.x && in1.y);
    Dense2xQuadPixel(base + int32_t2(1, 1), 1, 1, mv11, col, sq, kw, in1.x && in1.y);
}
#undef NSS_QD2X_W
#else
// Same pixel groups, per-pixel path (permutations without the hoisted KPN reads).
void PostprocessQuadDense2x(int32_t2 g)
{
    int32_t2 base = g * int32_t2(2) - (LutOffset() & int32_t2(1));
    for (int32_t s = 0; s < 4; ++s)
    {
        int32_t2 px = base + int32_t2(s & 1, s >> 1);
        if (all(greaterThanEqual(px, int32_t2(0))))
        {
            Postprocess(px);
        }
    }
}
#endif
#endif  // NSS_POSTPROCESS_QUAD_DENSE2X
#endif

#undef _NearestDepthOffsetTex
#undef _MotionVectorTex

#endif  // GPU_NSS_POSTPROCESS_H
