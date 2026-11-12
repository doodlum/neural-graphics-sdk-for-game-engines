// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_resources.h>
#include <FidelityFX/host/ffx_frameinterpolation_resolution.h>
#include <FidelityFX/host/ffx_types.h>

#include <cstddef>

/// An enumeration of all the permutations that can be passed to the FSR3 algorithm.
///
/// FSR3 features are organized through a set of pre-defined compile
/// permutation options that need to be specified. Which shader blob
/// is returned for pipeline creation will be determined by what combination
/// of shader permutations are enabled.
///
/// @ingroup ffxFrameGeneration
typedef enum FfxFrameInterpolationShaderPermutationOptions
{
    FRAMEINTERPOLATION_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS = (1 << 0),
    FRAMEINTERPOLATION_SHADER_PERMUTATION_JITTER_MOTION_VECTORS  = (1 << 1),
    FRAMEINTERPOLATION_SHADER_PERMUTATION_DEPTH_INVERTED         = (1 << 2),  ///< Indicates input resources were generated with inverted depth
    FRAMEINTERPOLATION_SHADER_PERMUTATION_FORCE_WAVE64           = (1 << 3),  ///< doesn't map to a define, selects different table
    FRAMEINTERPOLATION_SHADER_PERMUTATION_ALLOW_FP16             = (1 << 4),  ///< Enables fast math computations where possible
    FRAMEINTERPOLATION_SHADER_PERMUTATION_ENABLE_DATA_GRAPH_FI   = (1 << 5),  ///< Enable the data graph frame interpolation shader permutation
    FRAMEINTERPOLATION_SHADER_PERMUTATION_WARP_COMPUTE        = (1 << 6),   ///< Run warp stages (MV TP1 + flow TM1) as compute shaders; unset selects fragment.
    FRAMEINTERPOLATION_SHADER_PERMUTATION_FILL_HOLES_COMPUTE  = (1 << 7),   ///< Run fill holes stages (TP1 + TM1) as compute shaders; unset selects fragment.
    FRAMEINTERPOLATION_SHADER_PERMUTATION_PRE_PROCESS_COMPUTE = (1 << 8),   ///< Run preprocess as a compute shader; unset selects fragment.
    FRAMEINTERPOLATION_SHADER_PERMUTATION_POSTPROCESS_COMPUTE = (1 << 9),   ///< Run postprocess as a compute shader; unset selects fragment.
    FRAMEINTERPOLATION_SHADER_PERMUTATION_DEBUG_VIEW_COMPUTE  = (1 << 11),  ///< Run debug view as a compute shader; unset selects fragment.
    FRAMEINTERPOLATION_SHADER_PERMUTATION_COMPUTE_SUPPORT_TENSOR  = (1 << 12),  ///< Indicates that tensor ops are supported in compute shader
    FRAMEINTERPOLATION_SHADER_PERMUTATION_FRAGMENT_SUPPORT_TENSOR = (1 << 13),  ///< Indicates that tensor ops are supported in fragment shader
    FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_DEPTH   = (1 << 14),  ///< Indicates that the sdk manages depth history internally
    FRAMEINTERPOLATION_SHADER_PERMUTATION_MANAGE_PREVIOUS_COLOR   = (1 << 15),  ///< Indicates that the sdk manages color history internally
} FfxInterpolationShaderPermutationOptions;

struct FfxPipelineState;
struct FfxResource;

/// Number of inpaint-pyramid mip downsample passes (levels 1..4 of the five-level pyramid).
#define FFX_FRAMEINTERPOLATION_INPAINT_MIP_PIPELINE_COUNT (4)

typedef struct FrameInterpolationConstants
{
    FfxFloat32x4x4 _MotionTM1ToTP1;
    FfxFloat32x4x4 _MotionTP1ToTM1;

    FfxFloat32x4 _QuantParamsSINT;   ///< Quant parameters between float and sint. .xy for quantize, .zw for dequantize
    FfxFloat32x4 _QuantParamsSNORM;  ///< Quant parameters between float and snorm. .xy for quantize, .zw for dequantize

    float _DeviceToViewDepth[4];

    int32_t _RenderSize[2];
    int32_t _DisplaySize[2];

    float _DisplaySizeRcp[2];

    float _Jitter[2];
    float _MotionVectorScale[2];

    float _JitterCancellation[2];
    float    _MvSimilarityThresholds[3];  // x: similarity threshold (eps), y: noise threshold (tau), z: relative threshold (rel_eps)
    uint32_t _InputTensorChannels;        // 16: original graph input; 20: channel 16 carries the timestep (std140 packs it after the vec3)

    int32_t  _InputTensorSize[2];
    int      _Reset;
    float    _Timestep;
    uint32_t _RandomSeed;
    uint32_t _UseReducedMotionInput;

    int32_t _MvDepthLaneSize[2];
    float   _MvDepthLaneSizeRcp[2];
    int32_t _FlowLaneSize[2];
    float   _FlowLaneSizeRcp[2];
    float   _ParamsLaneSizeRcp[2];
    float   _QuantParamsGateSNORM[4];  ///< gate dequant params (.zw used). .xy reserved.
    float   _OfInputSizeRcp[2];
    float   _DepthClipScale;
    float   _DepthClipPower;

    // Multi-frame generation: 1 for the first generated frame of a dispatch. Storage-image history writes in the
    // warp and postprocess passes are skipped for later frames (the values are timestep independent).
    uint32_t _WriteHistory;
    uint32_t _WriteHistoryPadding[3];
} FrameInterpolationConstants;

static_assert(offsetof(FrameInterpolationConstants, _JitterCancellation) == 216, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _MvSimilarityThresholds) == 224, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _InputTensorChannels) == 236, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _InputTensorSize) == 240, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _Timestep) == 252, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _MvDepthLaneSize) == 264, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _QuantParamsGateSNORM) == 304, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _OfInputSizeRcp) == 320, "Frame interpolation constants must match the std140 shader layout.");
static_assert(offsetof(FrameInterpolationConstants, _WriteHistory) == 336, "Frame interpolation constants must match the std140 shader layout.");
static_assert(sizeof(FrameInterpolationConstants) == 352, "Frame interpolation constants must preserve std140 block size.");

typedef struct FrameInterpolationRenderDescription
{
    FfxDimensions2D renderSize;
    FfxDimensions2D upscaleSize;

    float cameraNear;
    float cameraFar;
    float cameraFovAngleVertical;
    float viewSpaceToMetersFactor;

    FfxFloatCoords2D motionVectorScale;  ///< The scale factor to apply to motion vectors.
} FrameInterpolationRenderDescription;

// FfxFsr3Context_Private
// The private implementation of the FSR3 context.
typedef struct FfxArmFrameInterpolationContext_Private
{
    FfxFrameInterpolationContextDescription contextDescription;
    FfxUInt32                               effectContextId;
    FrameInterpolationRenderDescription     renderDescription;
    FrameInterpolationConstants             constants;
    FfxDevice                               device;
    bool                                    backendContextCreated;

    // Canonical resolution lane model: cap internal resource dimensions
    FfxNfruResolutionPlan resolutionPlan;
    FfxDimensions2D       ofInputColour;  // Capped OF input colour resolution (≤1080p height)
    FfxDimensions2D       mvDepthLane;    // MV/depth processing lane (capped at 540p height)
    FfxDimensions2D       flowLane;       // Optical flow lane (capped at 270p height)
    FfxDimensions2D       tensorLane;     // Neural network tensor lane (preprocess lane, padded below)
    FfxDimensions2D       paramsLane;     // Graph output lane (padded tensor lane / 2): params + gate

    // FrameInterpolation Pipelines
    FfxPipelineState pipelineDebugView;

    // NFRU Pipeline States
    FfxPipelineState pipelineDownsampleOfColour;
    FfxPipelineState pipelineDownsampleAllInput;
    FfxPipelineState pipelineDownsampleColour;
    FfxPipelineState pipelineInitWarp;
    FfxPipelineState pipelineWarpFlow;
    FfxPipelineState pipelineWarpFlowTM1;
    FfxPipelineState pipelineFillHolesTM1;
    FfxPipelineState pipelinePreprocess;
    FfxPipelineState pipelineNFRUInterpolation;
    FfxPipelineState pipelineDownsampleInpaintSeed;  // 31 seed -> mip0
    // 31 mip i-1 -> i, one pipeline state per level (index level - 1). The backend rotates descriptor sets per
    // pipeline layout with a fixed per-frame budget, so separate states keep multi-frame generation (one pyramid
    // per generated frame) within that budget instead of using one layout FFX_FRAMEINTERPOLATION_INPAINT_MIP_PIPELINE_COUNT * N times.
    FfxPipelineState pipelineDownsampleInpaintMip[FFX_FRAMEINTERPOLATION_INPAINT_MIP_PIPELINE_COUNT];
    FfxPipelineState pipelineInpaint;                // 50_inpaint resolver
    FfxPipelineState pipelinePostprocess;

    FfxConstantBuffer constantBuffers[FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_COUNT];

    // 2 arrays of resources, as e.g. FFX_FSR3_RESOURCE_IDENTIFIER_LOCK_STATUS will use different resources when bound as SRV vs when bound as UAV
    FfxResourceInternal srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT];

    bool firstExecution;
    bool refreshPipelineStates;

    FfxDataGraphTensorInfo dataGraphTensorInfo[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT];
    uint32_t               dataGraphTensorInfoCount;

    uint32_t pipelineFlags;
    uint32_t dataGraphWidth;
    uint32_t dataGraphHeight;
    uint32_t inputTensorChannels;  // 16 or 20, read from the data graph's input tensor shape at creation

    uint64_t       previousFrameID;
    uint64_t       dispatchCount;
    FfxFloat32x4x4 lastFrameViewProjection;
    FfxFloat32x4x4 pendingViewProjection;

    bool enableDataGraphFI = true;
} FfxArmFrameInterpolationContext_Private;
