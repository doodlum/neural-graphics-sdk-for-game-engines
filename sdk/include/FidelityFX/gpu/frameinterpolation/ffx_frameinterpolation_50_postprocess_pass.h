// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#ifndef GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H
#define GPU_FFX_FRAMEINTERPOLATION_40_POSTPROCESS_PASS_H

#define HALF_MIN 0.00006103515625HF

#include "ffx_frameinterpolation_callbacks_glsl.h"

// methods

FFX_MIN16_F max4(FFX_MIN16_F4 x)
{
    return max(max(max(x.x, x.y), x.z), x.w);
}

FFX_MIN16_F4 Stable_Softmax(FFX_MIN16_F4 tensor)
{
    // Subtract the max value to help reduce overflow issues
    FFX_MIN16_F4 stable_ten     = tensor - max4(tensor);
    FFX_MIN16_F4 e_ten          = exp(stable_ten);
    FFX_MIN16_F  sum            = dot(e_ten, FFX_MIN16_F4(1.0HF));
    FFX_MIN16_F4 smax_activated = e_ten / (sum + HALF_MIN);
    return smax_activated;
}

// A warp offset under half a pixel samples the pixel itself: static and nearly static content (whose warped motion is
// zero or a quantisation step or two, about 0.06 px each) is passed through unfiltered, where bilinear filtering at
// those fractional offsets softened it in every generated frame.
float2 SnapSmallOffset(float2 uv, float2 offset, float2 size)
{
    const float2 px = offset * size;
    return dot(px, px) < 0.25f ? uv : uv + offset;
}

void postprocess(int32_t2 output_pixel)
{
    int32_t2 displaySize = DisplaySize();
    if (any(greaterThanEqual(output_pixel, displaySize)))
        return;

    float2 uv = (float2(output_pixel.xy) + 0.5f) * DisplaySizeRcp();

    float2   mvDepthLanef = float2(MvDepthLaneSize());
    float2   flowLanef    = float2(FlowLaneSize());
    float2   ofSizef      = float2(GetOpticalFlowResolution());
    int32_t2 iMvCoord     = int32_t2(floor(uv * mvDepthLanef));
    int32_t2 iFlowCoord   = int32_t2(floor(uv * flowLanef));

    // Read Output parameters - bilinear upsampling is done here
    float2       tensor_uv = uv * ofSizef * InputTensorSizeRcp();
    FFX_MIN16_F4 params    = Dequantize(FFX_MIN16_F4(SampleOutParamsTensor(tensor_uv)), QuantParamsSNORM().zw);

    // Read Vectors warped to "t" position - sampling nearest
    FFX_MIN16_F2 flow_t_f30_p1 = LoadWarpedFilledFlowTM1(iFlowCoord);
    FFX_MIN16_F2 mv_t_f30_m1   = LoadWarpedFilledMotionTP1(iMvCoord);

    // Calculate warped game mv positions
    float        timestep = Timestep();
    const float2 size     = float2(displaySize);
    float2       uv_m1_mv = SnapSmallOffset(uv, float2(mv_t_f30_m1 * timestep), size);
    float2       uv_p1_mv = SnapSmallOffset(uv, -float2(mv_t_f30_m1 * (1.0f - timestep)), size);

    // Calculate warped optical flow positions
    float2 uv_m1_flow = SnapSmallOffset(uv, -float2(flow_t_f30_p1 * timestep), size);
    float2 uv_p1_flow = SnapSmallOffset(uv, float2(flow_t_f30_p1 * (1.0f - timestep)), size);

    // Warp colour using gather approach with warped vectors
    FFX_MIN16_F3 rgb_m1_warp_t_mv   = SamplePreviousBackbuffer(uv_m1_mv);
    FFX_MIN16_F3 rgb_p1_warp_t_mv   = SampleCurrentBackbuffer(uv_p1_mv);
    FFX_MIN16_F3 rgb_m1_warp_t_flow = SamplePreviousBackbuffer(uv_m1_flow);
    FFX_MIN16_F3 rgb_p1_warp_t_flow = SampleCurrentBackbuffer(uv_p1_flow);

    // Softmax network output to get blending coeffs
    params = Stable_Softmax(params);

    // Composite Output
    FFX_MIN16_F3 rgb_t = FFX_MIN16_F3(0.HF);
    rgb_t += rgb_m1_warp_t_mv * params.x;
    rgb_t += rgb_p1_warp_t_mv * params.y;
    rgb_t += rgb_m1_warp_t_flow * params.z;
    rgb_t += rgb_p1_warp_t_flow * params.w;

#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_NO_WARP_MASK
    // Pixels the game marks not to warp (DLSS-G's no-warp mask): the current frame's, any UI composited over them.
    if (HasNoWarpMask() && LoadNoWarp(output_pixel))
        rgb_t = LoadCurrentBackbuffer(output_pixel);
#endif

    // Write alpha composited output
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_UI_SOURCE
    {
        const FFX_MIN16_F4 ui = LoadUiSource(output_pixel);
        if (UiKind() == 1u)
        {
            // The UI alone, premultiplied (DLSS-G's UI colour and alpha): blended over the interpolated scene as the
            // game blends it over the HUD-less frame, so soft edges, shadows and translucent panels keep the scene
            // behind them interpolated.
            rgb_t = ui.rgb + (FFX_MIN16_F(1.0f) - ui.a) * rgb_t;
        }
#ifdef FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_UI_ALPHA
        else if (UiKind() == 2u || UiKind() == 3u)
        {
            // The frame with UI and the UI's opacity (DLSS-G's UI alpha, or a UI layer's alpha): the UI colour is
            // frame - (1 - a) x HUD-less, so over the interpolated scene the output is frame - (1 - a) x (HUD-less -
            // interpolated): where nothing moves, the frame exactly. Where that UI colour cannot be right (not
            // finite, -1 or below, or below frame - 2 x HUD-less) the pixel is taken as opaque UI, as DLSS-G does.
            const FFX_MIN16_F  a   = LoadUiAlpha(output_pixel, UiKind() == 3u);
            const FFX_MIN16_F3 cur = LoadCurrentBackbuffer(output_pixel);
            const FFX_MIN16_F3 uiColor = ui.rgb - (FFX_MIN16_F(1.0f) - a) * cur;
            if (any(isnan(uiColor)) || any(isinf(uiColor)) || any(lessThanEqual(uiColor, FFX_MIN16_F3(-1.0f))) ||
                any(lessThan(uiColor, ui.rgb - FFX_MIN16_F(2.0f) * cur)))
                rgb_t = ui.rgb;
            else
                rgb_t = max(uiColor + (FFX_MIN16_F(1.0f) - a) * rgb_t, FFX_MIN16_F3(0.0f));
        }
#endif
        else
        {
            // The frame with UI: where it differs from the one interpolated (HUD-less), its pixel as it is, not warped.
            // DLSS-G's test: a channel off by more than 0.07 anywhere in the pixel's 2x2 block, so grain or sharpening
            // the game adds after its HUD-less copy is not taken for UI, and text's soft edges go with its strokes.
            const int32_t2 block = output_pixel & ~int32_t2(1);
            bool           isUi  = false;
            for (int32_t i = 0; i < 4; ++i)
            {
                const int32_t2     q   = min(block + int32_t2(i & 1, i >> 1), displaySize - 1);
                const FFX_MIN16_F3 d   = abs(LoadUiSource(q).rgb - LoadCurrentBackbuffer(q));
                isUi = isUi || max(d.r, max(d.g, d.b)) > FFX_MIN16_F(0.07f);
            }
            if (isUi)
                rgb_t = ui.rgb;
        }
    }
#endif

    StoreFrameinterpolationOutput(output_pixel, FFX_MIN16_F4(rgb_t, 1.0f));

    // When the SDK owns the previous-source ping-pong buffer, persist the current
    // backbuffer pixel so it can be sampled as colorTm1 on the next frame without
    // an extra copy job (see PREVIOUS_INTERPOLATION_SOURCE_BACKUP).
#if FFX_ARM_FRAMEINTERPOLATION_OPTION_MANAGE_PREVIOUS_COLOR
    StorePreviousInterpolationSource(output_pixel, FFX_MIN16_F4(LoadCurrentBackbuffer(output_pixel), 1.0f));
#endif
}

#endif