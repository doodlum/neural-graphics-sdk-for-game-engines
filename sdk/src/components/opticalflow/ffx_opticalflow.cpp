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

// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include <algorithm>  // for max used inside SPD CPU code.
#include <cfloat>     // for FLT_EPSILON
#include <cmath>      // for fabs, abs, sinf, sqrt, etc.
#include <string.h>   // for memset
#include <cstdlib>    // for getenv
#if defined(_WIN32)
#include <windows.h>  // for GetEnvironmentVariableA
#endif

#include "FidelityFX/host/ffx_opticalflow.h"
#define FFX_CPU

#include "FidelityFX/gpu/ffx_core.h"
#include "ffx_object_management.h"

#include "FidelityFX/host/ffx_util.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#include "ffx_opticalflow_private.h"

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    // sdk data
    uint32_t index;
    char     name[64];
} ResourceBinding;

constexpr int                srvTextureCount                         = 17;
static const ResourceBinding srvTextureBindingTable[srvTextureCount] = {
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_GAME_MOTION_VECTORS, "r_game_motion_vectors"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_DEPTH, "r_scatter_depth"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR, "r_color_tp1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR, "r_color_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH, "r_depth"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH, "r_depth_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS, "r_motion_vectors"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TP1, "r_pyramid_tp1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TM1, "r_pyramid_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_RAW, "r_flow_raw"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_LEVELS, "r_flow_levels"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_HINT, "r_flow_hint"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_WARP, "r_arm_warp"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_MATCH, "r_arm_match"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_SUB, "r_arm_sub"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_MED, "r_arm_med"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_FLOW, "r_arm_flow"},
};

constexpr int                uavTextureCount                         = 13;
static const ResourceBinding uavTextureBindingTable[uavTextureCount] = {
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_DEPTH, "rw_scatter_depth"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT, "rw_flow"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS, "rw_motion_vectors"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH, "rw_depth_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR, "rw_color_tm1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TP1, "rw_pyramid_tp1"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_RAW, "rw_flow_raw"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_LEVELS, "rw_flow_levels"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_WARP, "rw_arm_warp"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_MATCH, "rw_arm_match"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_SUB, "rw_arm_sub"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_MED, "rw_arm_med"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_FLOW, "rw_arm_flow"},
};

static const ResourceBinding uavBufferBindingTable[] = {
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_KEYS, "rw_scatter_keys"},
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_KEYS, "ScatterKeys_t"},  // the block's name, as reflection may give it
};

static const ResourceBinding rtResourceBindingTable[] = {
    {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS, "rw_motion_vectors"},
};

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvTextureCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(srvTextureBindingTable); ++mapIndex)
        {
            if (0 == strcmp(srvTextureBindingTable[mapIndex].name, inoutPipeline->srvTextureBindings[srvIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(srvTextureBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvTextureBindings[srvIndex].resourceIdentifier = srvTextureBindingTable[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavTextureCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(uavTextureBindingTable); ++mapIndex)
        {
            if (0 == strcmp(uavTextureBindingTable[mapIndex].name, inoutPipeline->uavTextureBindings[uavIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(uavTextureBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavTextureBindings[uavIndex].resourceIdentifier = uavTextureBindingTable[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavBufferCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(uavBufferBindingTable); ++mapIndex)
        {
            if (0 == strcmp(uavBufferBindingTable[mapIndex].name, inoutPipeline->uavBufferBindings[uavIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(uavBufferBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavBufferBindings[uavIndex].resourceIdentifier = uavBufferBindingTable[mapIndex].index;
    }

    for (uint32_t rtIndex = 0; rtIndex < inoutPipeline->rtCount; ++rtIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < FFX_COUNTOF(rtResourceBindingTable); ++mapIndex)
        {
            if (0 == strcmp(rtResourceBindingTable[mapIndex].name, inoutPipeline->rtBindings[rtIndex].name))
                break;
        }
        if (mapIndex == FFX_COUNTOF(rtResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->rtBindings[rtIndex].resourceIdentifier = rtResourceBindingTable[mapIndex].index;
    }

    return FFX_OK;
}

// Settings from the process environment. On Windows from the Win32 environment block, not the C runtime's copy, so a
// value the host sets after this DLL's runtime started (DeckyUpscaler's proxy passes its plugin settings so) counts.
static const char* SettingEnv(const char* name)
{
#if defined(_WIN32)
    static thread_local char value[64];
    const DWORD              n = GetEnvironmentVariableA(name, value, sizeof(value));
    return n && n < sizeof(value) ? value : nullptr;
#else
    return getenv(name);
#endif
}

static FfxOpticalFlowDescription createOpticalFlowDescription(FfxSurfaceFormat               colorFormat,
                                                              FfxDimensions2D                resolution,
                                                              FfxOpticalFlowGridSize         gridSize,
                                                              FfxOpticalFlowPerformanceLevel performanceLevel)
{
    // Mimic the shader blob memory style for now.
    // TODO: We might want to dynamically allocate this and clean it up after we are done with it instead of allocating static memory.
    static const char*                  g_srv_names[]            = {"r_color_tp1", "r_color_tm1", "r_motion_vectors"};
    static uint32_t                     g_srv_texture_bindings[] = {0, 1, 2};
    static uint32_t                     g_srv_textureCount[]     = {1, 1, 1};
    static uint32_t                     g_srv_textureSpaces[]    = {0, 0, 0};
    static FfxSurfaceFormat             g_srv_formats[]          = {colorFormat, colorFormat, FFX_SURFACE_FORMAT_R16G16_FLOAT};
    static FfxOpticalFlowConnectionType g_srv_connection_types[] = {
        // Connection type mapping between SDK resource names and DDK optical flow roles:
        // The optical flow pipeline uses forward optical flow convention, where the TP1 (current frame)
        // image is the "search" frame and TM1 (previous frame) is the "template" frame.
        // In DDK terms:
        //   FFX_OPTICAL_FLOW_CONNECTION_REFERENCE -> "search" frame
        //   FFX_OPTICAL_FLOW_CONNECTION_INPUT     -> "template" frame
        // In SDK terms:
        //   "r_color_tp1"     = TP1 (current frame) color  -> "search" frame -> maps to FFX_OPTICAL_FLOW_CONNECTION_REFERENCE
        //   "r_color_tm1" = TM1 (previous frame) color -> "template" frame -> maps to FFX_OPTICAL_FLOW_CONNECTION_INPUT
        FFX_OPTICAL_FLOW_CONNECTION_REFERENCE,
        FFX_OPTICAL_FLOW_CONNECTION_INPUT,
        FFX_OPTICAL_FLOW_CONNECTION_HINT};

    static const char*                  g_uav_names[]            = {"rw_flow"};
    static uint32_t                     g_uav_texture_bindings[] = {3};
    static uint32_t                     g_uav_textureCount[]     = {1};
    static uint32_t                     g_uav_textureSpaces[]    = {0};
    static FfxOpticalFlowConnectionType g_uav_connection_types[] = {FFX_OPTICAL_FLOW_CONNECTION_FLOW_VECTOR};
    static FfxSurfaceFormat             g_uav_formats[]          = {FFX_SURFACE_FORMAT_R16G16_FLOAT};

    FfxOpticalFlowDescription ofDesc{};

    ofDesc.srvTextureCount               = sizeof(g_srv_texture_bindings) / sizeof(g_srv_texture_bindings[0]);
    ofDesc.boundSRVTextureNames          = g_srv_names;
    ofDesc.boundSRVTextures              = g_srv_texture_bindings;
    ofDesc.boundSRVTextureCounts         = g_srv_textureCount;
    ofDesc.boundSRVTextureSpaces         = g_srv_textureSpaces;
    ofDesc.boundSRVTextureConnectionType = g_srv_connection_types;
    ofDesc.boundSRVTextureFormats        = g_srv_formats;
    // Input MVs are R16G16F, Color is in the passed in format

    ofDesc.uavTextureCount               = sizeof(g_uav_texture_bindings) / sizeof(g_uav_texture_bindings[0]);
    ofDesc.boundUAVTextureNames          = g_uav_names;
    ofDesc.boundUAVTextures              = g_uav_texture_bindings;
    ofDesc.boundUAVTextureCounts         = g_uav_textureCount;
    ofDesc.boundUAVTextureSpaces         = g_uav_textureSpaces;
    ofDesc.boundUAVTextureConnectionType = g_uav_connection_types;
    ofDesc.boundUAVTextureFormats        = g_uav_formats;

    // OF metadata
    ofDesc.dimensions       = resolution;
    ofDesc.gridSize         = gridSize;
    ofDesc.performanceLevel = performanceLevel;

    return ofDesc;
}

static uint32_t getPipelinePermutationOptions(uint32_t contextFlags)
{
    uint32_t permutationOptions = 0;

    // Check for depth inverted flag
    if (contextFlags & FFX_OPTICALFLOW_ENABLE_DEPTH_INVERTED)
    {
        permutationOptions |= OPTICALFLOW_SHADER_PERMUTATION_DEPTH_INVERTED;
    }

    // Check for MV hints fragment flag
    if (contextFlags & FFX_OPTICALFLOW_ENABLE_MV_HINTS_FRAGMENT)
    {
        permutationOptions |= OPTICALFLOW_SHADER_PERMUTATION_MV_HINTS_FRAGMENT;
    }

    return permutationOptions;
}

static FfxErrorCode createPipelineStates(OpticalFlowContext_Private* context)
{
    FFX_ASSERT(context);

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineOpticalflow, context->effectContextId);

    FfxOpticalFlowDescription ofDesc = createOpticalFlowDescription(context->contextDescription.backBufferFormat,
                                                                    context->contextDescription.resolution,
                                                                    context->contextDescription.gridSize,
                                                                    context->contextDescription.performanceLevel);

    // TODO: add more validation on the connection type with device caps like costSupported, hintSupported

    if (!context->hintsOnly)
    {
        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateOpticalFlowPipeline(
            &context->contextDescription.backendInterface, "ARM-OpticalFlow", ofDesc, context->effectContextId, &context->pipelineOpticalflow));

        patchResourceBindings(&context->pipelineOpticalflow);
    }

    if (context->hintsOnly)
    {
        const struct
        {
            FfxOpticalflowPass pass;
            const char*        name;
            FfxPipelineState*  pipeline;
            uint32_t           constantsSize;
        } scatter[] = {{FFX_OPTICALFLOW_PASS_SCATTER_DEPTH, "ARM-OpticalFlow-ScatterDepth", &context->pipelineScatterDepth, sizeof(OpticalFlowScatterConstants)},
                       {FFX_OPTICALFLOW_PASS_SCATTER_FLOW, "ARM-OpticalFlow-ScatterFlow", &context->pipelineScatterFlow, sizeof(OpticalFlowScatterConstants)},
                       {FFX_OPTICALFLOW_PASS_PYRAMID, "ARM-OpticalFlow-Pyramid", &context->pipelinePyramid, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_FLOW_SEARCH, "ARM-OpticalFlow-FlowSearch", &context->pipelineFlowSearch, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_FLOW_FILTER, "ARM-OpticalFlow-FlowFilter", &context->pipelineFlowFilter, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_FLOW_SEARCH_FINE, "ARM-OpticalFlow-FlowSearchFine", &context->pipelineFlowSearchFine, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_SCATTER, "ARM-OpticalFlow-Scatter", &context->pipelineScatter, sizeof(OpticalFlowScatterConstants)},
                       {FFX_OPTICALFLOW_PASS_SCATTER_RESOLVE, "ARM-OpticalFlow-ScatterResolve", &context->pipelineScatterResolve, sizeof(OpticalFlowScatterConstants)},
                       {FFX_OPTICALFLOW_PASS_FSR3_SEARCH, "ARM-OpticalFlow-Fsr3Search", &context->pipelineFsr3Search, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_FSR3_FILTER, "ARM-OpticalFlow-Fsr3Filter", &context->pipelineFsr3Filter, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_ARM_PYRAMID, "ARM-OpticalFlow-ArmPyramid", &context->pipelineArmPyramid, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_ARM_WARP, "ARM-OpticalFlow-ArmWarp", &context->pipelineArmWarp, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_ARM_MATCH, "ARM-OpticalFlow-ArmMatch", &context->pipelineArmMatch, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_ARM_MEDIAN, "ARM-OpticalFlow-ArmMedian", &context->pipelineArmMedian, sizeof(OpticalFlowImageConstants)},
                       {FFX_OPTICALFLOW_PASS_ARM_BILATERAL, "ARM-OpticalFlow-ArmBilateral", &context->pipelineArmBilateral, sizeof(OpticalFlowImageConstants)}};
        for (const auto& sp : scatter)
        {
            // Passes the device cannot run (their shaders need 64-bit atomics or packed dot products) are not built.
            // Of the image flows, only the one in use.
            const bool own   = sp.pass == FFX_OPTICALFLOW_PASS_FLOW_SEARCH || sp.pass == FFX_OPTICALFLOW_PASS_FLOW_FILTER ||
                               sp.pass == FFX_OPTICALFLOW_PASS_FLOW_SEARCH_FINE;
            const bool fsr3  = sp.pass == FFX_OPTICALFLOW_PASS_FSR3_SEARCH || sp.pass == FFX_OPTICALFLOW_PASS_FSR3_FILTER;
            const bool arm   = sp.pass >= FFX_OPTICALFLOW_PASS_ARM_PYRAMID && sp.pass <= FFX_OPTICALFLOW_PASS_ARM_BILATERAL;
            const bool image = sp.pass == FFX_OPTICALFLOW_PASS_PYRAMID || own || fsr3 || arm;
            if ((image && !context->imageFlow) || (own && (context->fsr3Flow || context->armFlow)) || (fsr3 && !context->fsr3Flow) ||
                (arm && !context->armFlow) || (sp.pass == FFX_OPTICALFLOW_PASS_PYRAMID && context->armFlow) ||
                (!context->scatter64 && (sp.pass == FFX_OPTICALFLOW_PASS_SCATTER || sp.pass == FFX_OPTICALFLOW_PASS_SCATTER_RESOLVE)))
                continue;
            ffxSafeReleasePipeline(&context->contextDescription.backendInterface, sp.pipeline, context->effectContextId);
            FfxPipelineDescription pipelineDescription = {};
            pipelineDescription.stage                  = FFX_BIND_COMPUTE_SHADER_STAGE;
            pipelineDescription.contextFlags           = context->contextDescription.flags;
            FfxRootConstantDescription rootConstantDescs;
            rootConstantDescs.size                      = sp.constantsSize;
            rootConstantDescs.stage                     = FFX_BIND_COMPUTE_SHADER_STAGE;
            pipelineDescription.rootConstantBufferCount = 1;
            pipelineDescription.rootConstants           = &rootConstantDescs;
            strncpy(pipelineDescription.name, sp.name, FFX_RESOURCE_NAME_SIZE - 1);
            pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
            FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateComputePipeline(
                &context->contextDescription.backendInterface, ARM_EFFECT_OPTICALFLOW, sp.pass, 0, &pipelineDescription, context->effectContextId, sp.pipeline));
            patchResourceBindings(sp.pipeline);
        }
    }

    context->useMVHintsFragment = !!(context->contextDescription.flags & FFX_OPTICALFLOW_ENABLE_MV_HINTS_FRAGMENT);
    if (context->useMVHintsFragment)
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFragmentMVHints, context->effectContextId);

        FfxPipelineDescription pipelineDescription = {};
        pipelineDescription.stage                  = FFX_BIND_PIXEL_SHADER_STAGE;
        pipelineDescription.contextFlags           = context->contextDescription.flags;

        FfxRootConstantDescription rootConstantDescs;
        rootConstantDescs.size                      = sizeof(context->computeMVHintsConstants);
        rootConstantDescs.stage                     = FFX_BIND_PIXEL_SHADER_STAGE;
        pipelineDescription.rootConstantBufferCount = 1;
        pipelineDescription.rootConstants           = &rootConstantDescs;

        strncpy(pipelineDescription.name, "ARM-OpticalFlow-ComputeMVHints", FFX_RESOURCE_NAME_SIZE - 1);
        pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        uint32_t permutationOptions = getPipelinePermutationOptions(context->contextDescription.flags);

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateGraphicsPipeline(&context->contextDescription.backendInterface,
                                                                                           ARM_EFFECT_OPTICALFLOW,
                                                                                           FFX_OPTICALFLOW_PASS_COMPUTE_MOTION_FROM_DEPTH,
                                                                                           permutationOptions,
                                                                                           &pipelineDescription,
                                                                                           context->effectContextId,
                                                                                           &context->pipelineFragmentMVHints));

        patchResourceBindings(&context->pipelineFragmentMVHints);
    }
    else
    {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineComputeMVHints, context->effectContextId);

        FfxPipelineDescription pipelineDescription = {};
        pipelineDescription.stage                  = FFX_BIND_COMPUTE_SHADER_STAGE;
        pipelineDescription.contextFlags           = context->contextDescription.flags;

        FfxRootConstantDescription rootConstantDescs;
        rootConstantDescs.size                      = sizeof(context->computeMVHintsConstants);
        rootConstantDescs.stage                     = FFX_BIND_COMPUTE_SHADER_STAGE;
        pipelineDescription.rootConstantBufferCount = 1;
        pipelineDescription.rootConstants           = &rootConstantDescs;

        strncpy(pipelineDescription.name, "ARM-OpticalFlow-ComputeMVHints", FFX_RESOURCE_NAME_SIZE - 1);
        pipelineDescription.name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';

        uint32_t permutationOptions = getPipelinePermutationOptions(context->contextDescription.flags);
        permutationOptions &= ~OPTICALFLOW_SHADER_PERMUTATION_MV_HINTS_FRAGMENT;

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateComputePipeline(&context->contextDescription.backendInterface,
                                                                                          ARM_EFFECT_OPTICALFLOW,
                                                                                          FFX_OPTICALFLOW_PASS_COMPUTE_MOTION_FROM_DEPTH,
                                                                                          permutationOptions,
                                                                                          &pipelineDescription,
                                                                                          context->effectContextId,
                                                                                          &context->pipelineComputeMVHints));

        patchResourceBindings(&context->pipelineComputeMVHints);
    }

    return FFX_OK;
}

static FfxErrorCode createResourceFromDescription(OpticalFlowContext_Private* context, const FfxInternalResourceDescription* resDesc)
{
    const FfxResourceType        resourceType        = resDesc->type;
    const FfxResourceDescription resourceDescription = {resourceType,
                                                        resDesc->format,
                                                        resDesc->width,
                                                        resDesc->height,
                                                        (resourceType == FFX_RESOURCE_TYPE_TENSOR) ? resDesc->channel : 1,
                                                        resDesc->mipCount,
                                                        resDesc->flags,
                                                        resDesc->usage,
                                                        resDesc->batchSize,
                                                        resDesc->shapeSize};
    const FfxResourceStates initialState = (resDesc->usage == FFX_RESOURCE_USAGE_READ_ONLY) ? FFX_RESOURCE_STATE_COMPUTE_READ : FFX_RESOURCE_STATE_GENERIC_UAV;
    const FfxCreateResourceDescription createResourceDescription = {
        FFX_HEAP_TYPE_DEFAULT, resourceDescription, initialState, resDesc->name, resDesc->id, resDesc->initData};
    return context->contextDescription.backendInterface.fpCreateResource(
        &context->contextDescription.backendInterface, &createResourceDescription, context->effectContextId, &context->srvResources[resDesc->id]);
}

static FfxErrorCode opticalFlowVkCreate(OpticalFlowContext_Private* context, const FfxOpticalFlowContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // Setup the data for implementation.
    memset(context, 0, sizeof(OpticalFlowContext_Private));
    context->device = contextDescription->backendInterface.device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxOpticalFlowContextDescription));

    // Create the context.
    FfxErrorCode errorCode = context->contextDescription.backendInterface.fpCreateBackendContext(
        &context->contextDescription.backendInterface, ARM_EFFECT_OPTICALFLOW, nullptr, &context->effectContextId);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // call out for device caps.
    FfxDeviceCapabilities capabilities;
    FFX_VALIDATE(context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities));
    // Without the data graph optical flow engine (it is not emulated on the native compute ML path, for example on
    // RADV), the flow is the exact motion of what is visible in TP1, scattered from the game's motion vectors to the TM1
    // grid (the ScatterDepth / ScatterFlow passes), over the camera-motion field the MV hints pass computes from depth
    // (which also serves alone without motion vectors). The engine's refinement of non-geometric motion is missing.
    context->hintsOnly = !capabilities.dataGraphOFSupported;
    // The motion vectors' scatter in one pass where 64-bit buffer atomics allow (FFX_OF_SCATTER64=0: two passes).
    {
        const char* v      = getenv("FFX_OF_SCATTER64");
        context->scatter64 = context->hintsOnly && capabilities.bufferInt64AtomicsSupported && !(v && !strcmp(v, "0"));
    }
    const bool outputGridSizeSupported = context->hintsOnly || (contextDescription->gridSize & capabilities.supportedOutputGridSizes) != 0;
    // TODO: add check on hint grid size when we start using hint.
    // const bool hintGridSizeSupported   = (contextDescription->gridSize & capabilities.supportedHintGridSizes) != 0;
    if (!outputGridSizeSupported)
    {
        return FFX_ERROR_NULL_DEVICE;
    }

    if (!context->hintsOnly && ((contextDescription->resolution.width < capabilities.dataGraphOFMinWidth) ||
                                (contextDescription->resolution.height < capabilities.dataGraphOFMinHeight) ||
                                (contextDescription->resolution.width > capabilities.dataGraphOFMaxWidth) ||
                                (contextDescription->resolution.height > capabilities.dataGraphOFMaxHeight)))
    {
        return FFX_ERROR_NULL_DEVICE;
    }

    // set defaults
    context->firstExecution     = true;
    context->resourceFrameIndex = 0;

    // set inital view projection matrix
    memcpy(context->lastFrameViewProjection, contextDescription->initialViewProjection, sizeof(context->lastFrameViewProjection));

    const FfxDimensions2D        displaySize = contextDescription->resolution;
    const FfxDimensions2D        renderSize  = contextDescription->maxRenderSize;
    const FfxOpticalFlowGridSize gridSize    = contextDescription->gridSize;
    context->opticalFlowSize                 = GetOpticalFlowTextureSizeFromBlockSize(displaySize, GetOpticalFlowBlockSize(gridSize));

    // Image optical flow (hintsOnly): a luma pyramid from the colour's resolution, halved per level (FFX_OF_LEVELS, 4
    // by default: motion of up to about 40 pixels the motion vectors do not have), the flow searched from the coarsest
    // over the hint flow (ffx_opticalflow_pyramid.glsl, _flow_search.glsl, _flow_filter.glsl). On by default where the
    // device has packed 8-bit dot products and level 0's cells are the flow's texels (the 4x4 grid); FFX_OF_IMAGE=0
    // turns it off. FFX_OF_LAMBDA_HINT and FFX_OF_LAMBDA_SEED set the search's cost per texel of distance from the
    // hint flow and from the seed, FFX_OF_SETTLED the cost below which agreeing seeds are not searched around, and
    // FFX_OF_FINEST=1 ends the search at level 1 (0.1 ms less at 1280x720, 1 dB less on the moving square).
    // FFX_OF_IMAGE=fsr3 (the plugin's optical flow setting) runs FidelityFX FSR 3's optical flow instead, on the same
    // pyramid with FSR 3's luma (ffx_opticalflow_fsr3_search.glsl, _fsr3_filter.glsl): a vector per 8x8 block,
    // searched -8..+7 around the coarser level's, without the hint flow; it needs no packed dot products.
    // FFX_OF_IMAGE=arm runs the optical flow engine NFRU is made for (VK_ARM_data_graph_optical_flow, which RADV does
    // not have) as Arm's ML Emulation Layer for Vulkan models it, configured as NFRU's data graph path configures it
    // (ffx_opticalflow_arm_*.glsl), with the hint flow as its hint (FFX_OF_ARM_HINT=camera: the depth and camera
    // motion alone, as the data graph path passes).
    int32_t pyramidRows = 0, flowRows = 0;
    {
        const char* v      = SettingEnv("FFX_OF_IMAGE");
        context->fsr3Flow  = v && !strcmp(v, "fsr3");
        context->armFlow   = v && !strcmp(v, "arm");
        context->imageFlow = context->hintsOnly && (capabilities.integerDotProductSupported || context->fsr3Flow || context->armFlow) &&
                             !(v && !strcmp(v, "0")) &&
                             context->opticalFlowSize.width == (displaySize.width + 3) / 4 &&
                             context->opticalFlowSize.height == (displaySize.height + 3) / 4;
        const char*     lv        = getenv("FFX_OF_LEVELS");
        const uint32_t  maxLevels = lv ? std::clamp(uint32_t(atoi(lv)), 1u, uint32_t(FFX_OPTICALFLOW_IMAGE_MAX_LEVELS)) : 4u;
        FfxDimensions2D d         = displaySize;
        context->levelCount       = 0;
        while (context->levelCount < maxLevels && (context->levelCount == 0 || d.height >= 40))
        {
            context->levelDims[context->levelCount] = d;
            context->levelRow[context->levelCount]  = pyramidRows;
            context->flowRow[context->levelCount]   = flowRows;
            pyramidRows += int32_t(d.height);
            flowRows += int32_t((d.height + 3) / 4);
            ++context->levelCount;
            d = {(d.width + 1) / 2, (d.height + 1) / 2};
        }
        const char* fl       = getenv("FFX_OF_FINEST");
        context->finestLevel = context->fsr3Flow ? 0u : std::min(fl ? uint32_t(atoi(fl)) : 0u, context->levelCount - 1);
        const char* lh      = getenv("FFX_OF_LAMBDA_HINT");
        const char* ls      = getenv("FFX_OF_LAMBDA_SEED");
        context->lambdaHint = lh ? uint32_t(atoi(lh)) : 256u;
        context->lambdaSeed = ls ? uint32_t(atoi(ls)) : 64u;
        const char* sc       = getenv("FFX_OF_SETTLED");
        context->settledCost = sc ? uint32_t(atoi(sc)) : 256u;
        // The engine's pyramid: each level half the one before, rounded up to even (setupPyramidLevelDimensions).
        context->armDims[0] = displaySize;
        context->armRow[0]  = 0;
        int32_t armRows     = 0;
        for (uint32_t l = 1; l < 6; ++l)
        {
            const FfxDimensions2D p = context->armDims[l - 1];
            context->armDims[l]     = {((p.width + 1) / 2 + 1) & ~1u, ((p.height + 1) / 2 + 1) & ~1u};
            context->armRow[l]      = armRows;
            armRows += int32_t(context->armDims[l].height);
        }
        if (context->armFlow)
            pyramidRows = armRows;
    }

    // FFX_OF_IMAGE=arm's one-level images, level 2's size (1 x 1 when not in use).
    const uint32_t armW = context->armFlow ? context->armDims[2].width : 1u, armH = context->armFlow ? context->armDims[2].height : 1u;
    const FfxResourceUsage otherProcessOutputImageUsage =
        context->contextDescription.flags & FFX_OPTICALFLOW_ENABLE_MV_HINTS_FRAGMENT ? FFX_RESOURCE_USAGE_RENDERTARGET : FFX_RESOURCE_USAGE_UAV;
    // Declare internal resources needed
    const FfxInternalResourceDescription internalSurfaceDesc[] = {
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS,
         "ArmOF_Computed_Motion_Vectors",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FfxResourceUsage(otherProcessOutputImageUsage),
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         context->opticalFlowSize.width,
         context->opticalFlowSize.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_DEPTH,
         "ArmOF_Scatter_Depth",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R32_UINT,
         context->opticalFlowSize.width,
         context->opticalFlowSize.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        // Two spare words a row for the search's unaligned reads.
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TP1,
         "ArmOF_Pyramid_0",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         context->armFlow ? FFX_SURFACE_FORMAT_R8_UNORM : FFX_SURFACE_FORMAT_R32_UINT,
         context->armFlow ? context->armDims[1].width : (displaySize.width + 3) / 4 + 2,
         uint32_t(pyramidRows),
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TM1,
         "ArmOF_Pyramid_1",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         context->armFlow ? FFX_SURFACE_FORMAT_R8_UNORM : FFX_SURFACE_FORMAT_R32_UINT,
         context->armFlow ? context->armDims[1].width : (displaySize.width + 3) / 4 + 2,
         uint32_t(pyramidRows),
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_RAW,
         "ArmOF_Flow_Raw",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         (displaySize.width + 3) / 4,
         uint32_t(flowRows),
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_LEVELS,
         "ArmOF_Flow_Levels",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         (displaySize.width + 3) / 4,
         uint32_t(flowRows),
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_KEYS,
         "ArmOF_Scatter_Keys",
         FFX_RESOURCE_TYPE_BUFFER,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_UNKNOWN,
         context->opticalFlowSize.width * context->opticalFlowSize.height * 8,
         8,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         FfxResourceInitData::FfxResourceInitValue(size_t(context->opticalFlowSize.width) * context->opticalFlowSize.height * 8, 0xff)},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_HINT,
         "ArmOF_Flow_Hint",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         context->opticalFlowSize.width,
         context->opticalFlowSize.height,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_WARP,
         "ArmOF_Arm_Warp",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R8_UNORM,
         armW,
         armH,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_MATCH,
         "ArmOF_Arm_Match",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R32_UINT,
         armW,
         armH,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_SUB,
         "ArmOF_Arm_Sub",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         armW,
         armH,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_MED,
         "ArmOF_Arm_Med",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         armW,
         armH,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_ARM_FLOW,
         "ArmOF_Arm_Flow",
         FFX_RESOURCE_TYPE_TEXTURE2D,
         FFX_RESOURCE_USAGE_UAV,
         FFX_SURFACE_FORMAT_R16G16_FLOAT,
         armW,
         armH,
         1,
         FFX_RESOURCE_FLAGS_NONE,
         {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
    };

    // clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < FFX_ARRAY_ELEMENTS(internalSurfaceDesc); ++currentSurfaceIndex)
    {
        FFX_VALIDATE(createResourceFromDescription(context, &internalSurfaceDesc[currentSurfaceIndex]));
    }

    // copy resources to uavResrouces list
    memcpy(context->uavResources, context->srvResources, sizeof(context->srvResources));
    context->pyramid[0] = context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TP1];
    context->pyramid[1] = context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TM1];

    {
        errorCode = createPipelineStates(context);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }
    return FFX_OK;
}

static FfxErrorCode opticalFlowVkRelease(OpticalFlowContext_Private* context)
{
    FFX_ASSERT(context);

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineOpticalflow, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineComputeMVHints, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineScatterDepth, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineScatterFlow, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelinePyramid, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFlowSearch, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFlowFilter, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFlowSearchFine, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineScatter, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineScatterResolve, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFsr3Search, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFsr3Filter, context->effectContextId);
    for (FfxPipelineState* p : {&context->pipelineArmPyramid, &context->pipelineArmWarp, &context->pipelineArmMatch, &context->pipelineArmMedian,
                                &context->pipelineArmBilateral})
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, p, context->effectContextId);
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TP1] = context->pyramid[0];
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TM1] = context->pyramid[1];
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFragmentMVHints, context->effectContextId);

    // unregister resources not created internally
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR]      = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH]      = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT]     = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR] = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH] = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_GAME_MOTION_VECTORS] = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL};  // registered

    // release internal resources
    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex)
    {
        ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->srvResources[currentResourceIndex], context->effectContextId);
    }

    // Destroy the context
    context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId);

    return FFX_OK;
}

static void scheduleDataGraph(OpticalFlowContext_Private* context, FfxPipelineState* pipeline, bool reset)
{
    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_DATA_GRAPH};

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->srvResources[currentResourceId];
        dispatchJob.dataGraphJobDescription.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.dataGraphJobDescription.srvTextures[currentShaderResourceViewIndex].name,
                pipeline->srvTextureBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.dataGraphJobDescription.srvTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].name,
                pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif

        {
            const FfxResourceInternal currentResource                                                 = context->uavResources[currentResourceId];
            dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
            dispatchJob.dataGraphJobDescription.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
        }
    }
    if (!context->firstExecution && !reset)
    {
        // dispatchJob.dataGraphJobDescription.opticalFlowExecuteFlags = FFX_DATA_GRAPH_OPTICAL_FLOW_EXECUTE_INPUT_IS_PREVIOUS_REFERENCE;
    }
    dispatchJob.dataGraphJobDescription.pipeline = *pipeline;

    context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static void scheduleMVHintsAndCopyResources(
    OpticalFlowContext_Private* context, const FfxPipelineState* pipeline, uint32_t dispatchX, uint32_t dispatchY, const char* debugName)
{
    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_COMPUTE};
    if (debugName != nullptr)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.jobLabel, debugName, FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource   = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name,
                pipeline->srvTextureBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.uavTextureNames[currentUnorderedAccessViewIndex],
                pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.uavTextureNames[currentUnorderedAccessViewIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif

        {
            const FfxResourceInternal currentResource                                              = context->uavResources[currentResourceId];
            dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
            dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
        }
    }

    for (uint32_t i = 0; i < pipeline->uavBufferCount; ++i)
        dispatchJob.computeJobDescriptor.uavBuffers[i].resource = context->uavResources[pipeline->uavBufferBindings[i].resourceIdentifier];

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->srvTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                              = pipeline->srvTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].name,
                pipeline->srvTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.srvTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentTensorIndex = 0; currentTensorIndex < pipeline->uavTensorCount; ++currentTensorIndex)
    {
        const uint32_t            currentResourceId                              = pipeline->uavTensorBindings[currentTensorIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                                = context->uavResources[currentResourceId];
        dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].name,
                pipeline->uavTensorBindings[currentTensorIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.uavTensors[currentTensorIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    dispatchJob.computeJobDescriptor.dimensions[0] = dispatchX;
    dispatchJob.computeJobDescriptor.dimensions[1] = dispatchY;
    dispatchJob.computeJobDescriptor.dimensions[2] = 1;
    dispatchJob.computeJobDescriptor.pipeline      = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.computeJobDescriptor.cbNames[currentRootConstantIndex],
                pipeline->constantBufferBindings[currentRootConstantIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.computeJobDescriptor.cbNames[currentRootConstantIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
        dispatchJob.computeJobDescriptor.cbs[currentRootConstantIndex] =
            context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    FFX_ASSERT(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob) == FFX_OK);
}

static void scheduleFragmentMVHints(
    OpticalFlowContext_Private* context, const FfxPipelineState* pipeline, uint32_t width, uint32_t height, const char* debugName)
{
    FfxFragmentJobDescription jobDescriptor = {};

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId                        = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource                          = context->srvResources[currentResourceId];
        jobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.srvTextures[currentShaderResourceViewIndex].name,
                pipeline->srvTextureBindings[currentShaderResourceViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.srvTextures[currentShaderResourceViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].name,
                pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name,
                FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif

        const FfxResourceInternal currentResource                           = context->uavResources[currentResourceId];
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
        jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip      = 0;
    }

    jobDescriptor.viewport[0] = width;
    jobDescriptor.viewport[1] = height;
    jobDescriptor.pipeline    = *pipeline;

    for (uint32_t currentRTIndex = 0; currentRTIndex < pipeline->rtCount; ++currentRTIndex)
    {
        const uint32_t            currentResourceId       = pipeline->rtBindings[currentRTIndex].resourceIdentifier;
        const FfxResourceInternal currentResource         = context->uavResources[currentResourceId];
        jobDescriptor.rtTextures[currentRTIndex].resource = currentResource;
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.rtTextures[currentRTIndex].name, pipeline->rtBindings[currentRTIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.rtTextures[currentRTIndex].name[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex)
    {
#ifdef FFX_DEBUG
        strncpy(jobDescriptor.cbNames[currentRootConstantIndex], pipeline->constantBufferBindings[currentRootConstantIndex].name, FFX_RESOURCE_NAME_SIZE - 1);
        jobDescriptor.cbNames[currentRootConstantIndex][FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
        jobDescriptor.cbs[currentRootConstantIndex] = context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_FRAGMENT};
    if (debugName != nullptr)
    {
#ifdef FFX_DEBUG
        strncpy(dispatchJob.jobLabel, debugName, FFX_RESOURCE_NAME_SIZE - 1);
        dispatchJob.jobLabel[FFX_RESOURCE_NAME_SIZE - 1] = '\0';
#endif
    }
    dispatchJob.fragmentJobDescriptor = jobDescriptor;

    FFX_ASSERT(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob) == FFX_OK);
}

// Image optical flow over the hint flow in the result: TP1's luma pyramid; then, when there is TM1's, the hint copied
// aside and the flow searched and filtered level by level from the coarsest, level 0's filter writing the result.
// FFX_OF_IMAGE=arm: the emulated optical flow engine (ffx_opticalflow_arm_*.glsl). TP1's pyramid (levels 1 to 5);
// then, with TM1's, per level from 5 to 2 its warp, block match and subpixel step, median and bilateral filters, and
// at level 2 the hint (the hint flow, or the depth and camera motion with FFX_OF_ARM_HINT=camera) where it matches
// better. The data graph path's flags (TM1's pyramid kept from the frame before) are the ping-pong here.
static FfxErrorCode scheduleArmFlow(OpticalFlowContext_Private* context, const FfxOpticalFlowDispatchDescription* params)
{
    const FfxSurfaceFormat f = params->color.description.format;
    OpticalFlowImageConstants c = {};
    c.src_dims[0] = int32_t(params->color.description.width), c.src_dims[1] = int32_t(params->color.description.height);
    c.hint_dims[0] = int32_t(context->opticalFlowSize.width), c.hint_dims[1] = int32_t(context->opticalFlowSize.height);
    c.unorm8 = f == FFX_SURFACE_FORMAT_R8G8B8A8_UNORM || f == FFX_SURFACE_FORMAT_B8G8R8A8_UNORM || f == FFX_SURFACE_FORMAT_R8G8B8A8_SRGB ||
               f == FFX_SURFACE_FORMAT_B8G8R8A8_SRGB || f == FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS || f == FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS ||
               f == FFX_SURFACE_FORMAT_R8_UNORM;
    auto stage = [&]() {
        return context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
            &context->contextDescription.backendInterface, &c, sizeof(c), &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]);
    };
    auto groups = [](FfxDimensions2D d) { return FfxDimensions2D{(d.width + 7) / 8, (d.height + 7) / 8}; };
    for (int32_t l = 1; l < 6; ++l)
    {
        const FfxDimensions2D d = context->armDims[l], s = context->armDims[l - 1];
        c.level = l, c.level_row = context->armRow[l], c.coarse_row = context->armRow[l - 1];
        c.level_dims[0] = int32_t(d.width), c.level_dims[1] = int32_t(d.height);
        c.coarse_dims[0] = int32_t(s.width), c.coarse_dims[1] = int32_t(s.height);
        FFX_VALIDATE(stage());
        const FfxDimensions2D g = groups(d);
        scheduleMVHintsAndCopyResources(context, &context->pipelineArmPyramid, g.width, g.height, "ArmOF_ArmPyramid");
    }
    if (!context->pyramidValid || params->reset)
        return FFX_OK;

    static const bool cameraHint = [] {
        const char* v = getenv("FFX_OF_ARM_HINT");
        return v && !strcmp(v, "camera");
    }();
    FfxGpuJobDescription copyJob       = {FFX_GPU_JOB_COPY};
    copyJob.copyJobDescriptor.src      = cameraHint ? context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS]
                                                    : context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT];
    copyJob.copyJobDescriptor.dst      = context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_HINT];
    copyJob.copyJobDescriptor.copyMode = FFX_GPU_COPY_SRC_EXTENT;
    FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &copyJob));

    for (int32_t l = 5; l >= 2; --l)
    {
        const FfxDimensions2D d  = context->armDims[l];
        const FfxDimensions2D cc = l == 5 ? FfxDimensions2D{0, 0} : context->armDims[l + 1];
        c.level = l, c.level_row = context->armRow[l], c.coarse_row = 0, c.final = l == 2;
        c.level_dims[0] = int32_t(d.width), c.level_dims[1] = int32_t(d.height);
        c.coarse_dims[0] = int32_t(cc.width), c.coarse_dims[1] = int32_t(cc.height);
        FFX_VALIDATE(stage());
        const FfxDimensions2D g = groups(d);
        scheduleMVHintsAndCopyResources(context, &context->pipelineArmWarp, g.width, g.height, "ArmOF_ArmWarp");
        scheduleMVHintsAndCopyResources(context, &context->pipelineArmMatch, g.width, g.height, "ArmOF_ArmMatch");
        scheduleMVHintsAndCopyResources(context, &context->pipelineArmMedian, g.width, g.height, "ArmOF_ArmMedian");
        scheduleMVHintsAndCopyResources(context, &context->pipelineArmBilateral, g.width, g.height, "ArmOF_ArmBilateral");
    }
    return FFX_OK;
}

static FfxErrorCode scheduleImageFlow(OpticalFlowContext_Private* context, const FfxOpticalFlowDispatchDescription* params)
{
    const uint32_t tp1 = context->resourceFrameIndex;
    for (FfxResourceInternal* r : {context->srvResources, context->uavResources})
    {
        r[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TP1] = context->pyramid[tp1];
        r[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TM1] = context->pyramid[tp1 ^ 1];
    }
    if (context->armFlow)
    {
        FFX_VALIDATE(scheduleArmFlow(context, params));
        context->pyramidValid = true;
        return FFX_OK;
    }
    auto cells = [](FfxDimensions2D d) { return FfxDimensions2D{(d.width + 3) / 4, (d.height + 3) / 4}; };

    OpticalFlowImageConstants c = {};
    c.src_dims[0] = int32_t(params->color.description.width), c.src_dims[1] = int32_t(params->color.description.height);
    c.level_count = int32_t(context->levelCount);
    c.fsr3        = context->fsr3Flow ? 1 : 0;
    for (uint32_t l = 0; l < context->levelCount; ++l)
    {
        c.levels[l][0] = int32_t(context->levelDims[l].width), c.levels[l][1] = int32_t(context->levelDims[l].height);
        c.levels[l][2] = context->levelRow[l], c.levels[l][3] = context->flowRow[l];
    }
    FFX_VALIDATE(context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
        &context->contextDescription.backendInterface, &c, sizeof(c), &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]));
    scheduleMVHintsAndCopyResources(context,
                                    &context->pipelinePyramid,
                                    (context->levelDims[0].width + 63) / 64,
                                    (context->levelDims[0].height + 63) / 64,
                                    "ArmOF_Pyramid");

    if (context->pyramidValid && !params->reset && context->fsr3Flow)
    {
        // FSR 3's: 8x8 blocks at each level (cells here), seeded from the coarser level's (step 2 runs its scale pass).
        auto blocks = [](FfxDimensions2D d) { return FfxDimensions2D{(d.width + 7) / 8, (d.height + 7) / 8}; };
        for (int32_t l = int32_t(context->levelCount) - 1; l >= 0; --l)
        {
            const FfxDimensions2D d   = context->levelDims[l];
            const FfxDimensions2D n   = blocks(d);
            const bool            top = l + 1 == int32_t(context->levelCount);
            const FfxDimensions2D cc  = top ? FfxDimensions2D{0, 0} : blocks(context->levelDims[l + 1]);
            c.level_dims[0] = int32_t(d.width), c.level_dims[1] = int32_t(d.height);
            c.cell_dims[0] = int32_t(n.width), c.cell_dims[1] = int32_t(n.height);
            c.coarse_dims[0] = int32_t(cc.width), c.coarse_dims[1] = int32_t(cc.height);
            c.hint_dims[0] = int32_t(context->opticalFlowSize.width), c.hint_dims[1] = int32_t(context->opticalFlowSize.height);
            c.out_scale[0] = float(params->color.description.width) / float(context->levelDims[0].width);
            c.out_scale[1] = float(params->color.description.height) / float(context->levelDims[0].height);
            c.level_row    = context->levelRow[l];
            c.flow_row     = context->flowRow[l];
            c.coarse_row   = top ? 0 : context->flowRow[l + 1];
            c.level        = l;
            c.final        = l == 0;
            FFX_VALIDATE(context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
                &context->contextDescription.backendInterface, &c, sizeof(c), &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]));
            scheduleMVHintsAndCopyResources(context, &context->pipelineFsr3Search, (n.width + 7) / 8, (n.height + 7) / 8, "ArmOF_Fsr3Search");
            const FfxDimensions2D o = c.final ? context->opticalFlowSize : n;
            scheduleMVHintsAndCopyResources(context, &context->pipelineFsr3Filter, (o.width + 7) / 8, (o.height + 7) / 8, "ArmOF_Fsr3Filter");
        }
    }
    else if (context->pyramidValid && !params->reset)
    {
        FfxGpuJobDescription copyJob       = {FFX_GPU_JOB_COPY};
        copyJob.copyJobDescriptor.src      = context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT];
        copyJob.copyJobDescriptor.dst      = context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_HINT];
        copyJob.copyJobDescriptor.copyMode = FFX_GPU_COPY_SRC_EXTENT;
        FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &copyJob));

        for (int32_t l = int32_t(context->levelCount) - 1; l >= int32_t(context->finestLevel); --l)
        {
            const FfxDimensions2D d  = context->levelDims[l];
            const FfxDimensions2D n  = cells(d);
            const bool            top = l + 1 == int32_t(context->levelCount);
            const FfxDimensions2D cc = top ? FfxDimensions2D{0, 0} : cells(context->levelDims[l + 1]);
            c.level_dims[0] = int32_t(d.width), c.level_dims[1] = int32_t(d.height);
            c.cell_dims[0] = int32_t(n.width), c.cell_dims[1] = int32_t(n.height);
            c.coarse_dims[0] = int32_t(cc.width), c.coarse_dims[1] = int32_t(cc.height);
            c.hint_dims[0] = int32_t(context->opticalFlowSize.width), c.hint_dims[1] = int32_t(context->opticalFlowSize.height);
            c.out_scale[0] = float(params->color.description.width) / float(context->levelDims[0].width);
            c.out_scale[1] = float(params->color.description.height) / float(context->levelDims[0].height);
            c.level_row    = context->levelRow[l];
            c.flow_row     = context->flowRow[l];
            c.coarse_row   = top ? 0 : context->flowRow[l + 1];
            c.level        = l;
            c.lambda_hint  = context->lambdaHint;
            c.lambda_seed  = context->lambdaSeed;
            c.settled_cost = context->settledCost;
            c.final        = l == int32_t(context->finestLevel);
            FFX_VALIDATE(context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
                &context->contextDescription.backendInterface, &c, sizeof(c), &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]));
            scheduleMVHintsAndCopyResources(context,
                                            l == 0 && !top ? &context->pipelineFlowSearchFine : &context->pipelineFlowSearch,
                                            (n.width + 7) / 8,
                                            (n.height + 7) / 8,
                                            "ArmOF_FlowSearch");
            const FfxDimensions2D o = c.final ? context->opticalFlowSize : n;
            scheduleMVHintsAndCopyResources(context, &context->pipelineFlowFilter, (o.width + 7) / 8, (o.height + 7) / 8, "ArmOF_FlowFilter");
        }

    }
    context->pyramidValid = true;
    return FFX_OK;
}

static FfxErrorCode ArmOpticalFlowDispatch(OpticalFlowContext_Private* context, const FfxOpticalFlowDispatchDescription* params)
{
    // take a short cut to the command list
    FfxCommandList commandList = params->commandList;

    // Input: register color resource
    context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                    &params->color,
                                                                    context->effectContextId,
                                                                    &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR]);

    // Input: register depth resource
    context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                    &params->depth,
                                                                    context->effectContextId,
                                                                    &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH]);

    // Input: register depthtm1 resource
    context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                    &params->depthTm1,
                                                                    context->effectContextId,
                                                                    &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH]);

    // Input: register colortm1 resource
    context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                    &params->colorTm1,
                                                                    context->effectContextId,
                                                                    &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR]);

    // Output: register output optical flow vector
    context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                    &params->opticalFlowVector,
                                                                    context->effectContextId,
                                                                    &context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT]);

    // On first execution or reset, clear previous depth buffers
    if (params->reset)
    {
        FfxGpuJobDescription clearJob = {FFX_GPU_JOB_CLEAR_FLOAT};
        const float          clearValuesToZeroFloat[]{0.f, 0.f, 0.f, 0.f};
        memcpy(clearJob.clearJobDescriptor.color, clearValuesToZeroFloat, 4 * sizeof(float));

        constexpr uint32_t resources_to_clear[] = {FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH, FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR};
        for (uint32_t i = 0; i < FFX_COUNTOF(resources_to_clear); ++i)
        {
            clearJob.clearJobDescriptor.target = context->srvResources[resources_to_clear[i]];
            FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob));
        }
    }

    // Compute MV hints from depth and copy resources for next frame use
    {
        // Set constant buffer for MV hints computation
        const FfxDimensions2D depthResolution{params->depthTm1.description.width, params->depthTm1.description.height};

        context->computeMVHintsConstants.output_dims[0]    = context->opticalFlowSize.width;
        context->computeMVHintsConstants.output_dims[1]    = context->opticalFlowSize.height;
        context->computeMVHintsConstants.depth_size[0]     = depthResolution.width;
        context->computeMVHintsConstants.depth_size[1]     = depthResolution.height;
        context->computeMVHintsConstants.depth_size_rcp[0] = 1.0f / (float)depthResolution.width;
        context->computeMVHintsConstants.depth_size_rcp[1] = 1.0f / (float)depthResolution.height;
        context->computeMVHintsConstants.color_size[0]     = params->color.description.width;
        context->computeMVHintsConstants.color_size[1]     = params->color.description.height;

        // Calculate reprojection matrix TM1 -> TP1 (previous to current)
        FfxFloat32x4x4 invViewProjTM1;
        MatrixInvert4x4(context->lastFrameViewProjection, invViewProjTM1);
        MatrixMul4x4(params->viewProjection, invViewProjTM1, context->computeMVHintsConstants.motion_matrix_m1p1);

        // Stage constant buffer
        FFX_VALIDATE(
            context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(&context->contextDescription.backendInterface,
                                                                                       &context->computeMVHintsConstants,
                                                                                       sizeof(context->computeMVHintsConstants),
                                                                                       &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]));

        if (context->useMVHintsFragment)
        {
            // Fragment path: dispatch at OF dims (the MV hints render target size)
            scheduleFragmentMVHints(
                context, &context->pipelineFragmentMVHints, context->opticalFlowSize.width, context->opticalFlowSize.height, "ArmOF_MVHints");
        }
        else
        {
            // Compute path: dispatch at OF dims
            uint32_t dispatchX = (context->opticalFlowSize.width + 15) / 16;
            uint32_t dispatchY = (context->opticalFlowSize.height + 15) / 16;
            scheduleMVHintsAndCopyResources(context, &context->pipelineComputeMVHints, dispatchX, dispatchY, "ArmOF_MVHints");
        }
    }

    // Schedule optical flow data graph with computed mv hints, or take the hints as the flow without the engine
    if (context->hintsOnly)
    {
        // The camera-motion hints, then (with the game's motion vectors) the exact motion of everything visible in
        // TP1 scattered over them (ffx_opticalflow_scatter_depth.glsl / _flow.glsl).
        FfxGpuJobDescription copyJob       = {FFX_GPU_JOB_COPY};
        copyJob.copyJobDescriptor.src      = context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS];
        copyJob.copyJobDescriptor.dst      = context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT];
        copyJob.copyJobDescriptor.copyMode = FFX_GPU_COPY_SRC_EXTENT;
        FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &copyJob));

        if (params->motionVectors.resource && !params->reset)
        {
            context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface,
                                                                            &params->motionVectors,
                                                                            context->effectContextId,
                                                                            &context->srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_GAME_MOTION_VECTORS]);
            if (!context->scatter64)  // its keys are reset by the resolve
            {
                FfxGpuJobDescription clearJob = {FFX_GPU_JOB_CLEAR_UINT};
                for (uint32_t& v : clearJob.clearUintJobDescriptor.color)
                    v = 0xffffffffu;
                clearJob.clearUintJobDescriptor.target = context->uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_DEPTH];
                FFX_VALIDATE(context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &clearJob));
            }

            OpticalFlowScatterConstants c = {};
            c.motion_vector_scale[0] = params->motionVectorScale[0];
            c.motion_vector_scale[1] = params->motionVectorScale[1];
            c.output_dims[0]         = int32_t(context->opticalFlowSize.width);
            c.output_dims[1]         = int32_t(context->opticalFlowSize.height);
            c.motion_vector_dims[0]  = int32_t(params->motionVectors.description.width);
            c.motion_vector_dims[1]  = int32_t(params->motionVectors.description.height);
            c.depth_dims[0]          = int32_t(params->depth.description.width);
            c.depth_dims[1]          = int32_t(params->depth.description.height);
            c.depth_inverted         = (context->contextDescription.flags & FFX_OPTICALFLOW_ENABLE_DEPTH_INVERTED) ? 1 : 0;
            c.display_dims[0]        = int32_t(context->contextDescription.resolution.width);
            c.display_dims[1]        = int32_t(context->contextDescription.resolution.height);
            FFX_VALIDATE(context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
                &context->contextDescription.backendInterface, &c, sizeof(c), &context->constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER]));
            // One invocation per 2x2 block of motion vectors.
            const uint32_t groupsX = (params->motionVectors.description.width + 31) / 32;
            const uint32_t groupsY = (params->motionVectors.description.height + 31) / 32;
            if (context->scatter64)
            {
                scheduleMVHintsAndCopyResources(context, &context->pipelineScatter, groupsX, groupsY, "ArmOF_Scatter");
                scheduleMVHintsAndCopyResources(context,
                                                &context->pipelineScatterResolve,
                                                (context->opticalFlowSize.width + 7) / 8,
                                                (context->opticalFlowSize.height + 7) / 8,
                                                "ArmOF_ScatterResolve");
            }
            else
            {
                scheduleMVHintsAndCopyResources(context, &context->pipelineScatterDepth, groupsX, groupsY, "ArmOF_ScatterDepth");
                scheduleMVHintsAndCopyResources(context, &context->pipelineScatterFlow, groupsX, groupsY, "ArmOF_ScatterFlow");
            }
        }

        if (context->imageFlow)
            FFX_VALIDATE(scheduleImageFlow(context, params));
    }
    else
        scheduleDataGraph(context, &context->pipelineOpticalflow, params->reset);

    // Save current view projection for next frame
    memcpy(context->lastFrameViewProjection, params->viewProjection, sizeof(context->lastFrameViewProjection));

    context->contextDescription.backendInterface.fpExecuteGpuJobs(&context->contextDescription.backendInterface, commandList, context->effectContextId);

    // Release dynamic resources
    context->contextDescription.backendInterface.fpUnregisterResources(&context->contextDescription.backendInterface, commandList, context->effectContextId);

    context->firstExecution = false;
    // Toggle between 0 and 1 to choose the correct internal resources
    context->resourceFrameIndex ^= 1;

    return FFX_OK;
}

FfxErrorCode ffxOpticalFlowContextCreate(FfxOpticalFlowContext* context, FfxOpticalFlowContextDescription* contextDescription)
{
    // zero context memory
    memset(context, 0, sizeof(FfxOpticalFlowContext));

    // check pointers are valid.
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(contextDescription, FFX_ERROR_INVALID_POINTER);

    // validate that all callbacks are set for the interface
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

    // if a scratch buffer is declared, then we must have a size
    if (contextDescription->backendInterface.scratchBuffer)
    {
        FFX_RETURN_ON_ERROR(contextDescription->backendInterface.scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
    }

    // ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxOpticalFlowContext) >= sizeof(OpticalFlowContext_Private));

    // create the context.
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    const FfxErrorCode          errorCode      = opticalFlowVkCreate(contextPrivate, contextDescription);

    return errorCode;
}

FfxErrorCode ffxOpticalFlowContextDestroy(FfxOpticalFlowContext* context)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);

    // destroy the context.
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    const FfxErrorCode          errorCode      = opticalFlowVkRelease(contextPrivate);
    return errorCode;
}

FfxOpticalFlowGridSize ffxGetDefaultDataGraphOpticalFlowGridSize(FfxInterface& backendInterface)
{
    FFX_ASSERT(backendInterface.devCapInitialized);

    FfxDeviceCapabilities capabilities;
    backendInterface.fpGetDeviceCapabilities(&backendInterface, &capabilities);
    const auto supportedSizes = capabilities.supportedOutputGridSizes;

    FfxOpticalFlowGridSize defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
    if (!capabilities.dataGraphOFSupported || (supportedSizes & FFX_OPTICAL_FLOW_GRID_SIZE_4X4))
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_4X4;
    }
    else if (supportedSizes & FFX_OPTICAL_FLOW_GRID_SIZE_8X8)
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_8X8;
    }
    else if (supportedSizes & FFX_OPTICAL_FLOW_GRID_SIZE_2X2)
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_2X2;
    }
    else
    {
        defaultGridSize = FFX_OPTICAL_FLOW_GRID_SIZE_1X1;
    }

    return defaultGridSize;
}

bool ffxOpticalFlowGridSizeSupported(FfxInterface& backendInterface, const FfxOpticalFlowGridSize gridSize)
{
    FFX_ASSERT(backendInterface.devCapInitialized);

    FfxDeviceCapabilities capabilities;
    backendInterface.fpGetDeviceCapabilities(&backendInterface, &capabilities);
    const auto supportedSizes = capabilities.supportedOutputGridSizes;

    return (gridSize & supportedSizes) != 0;
}

FfxOpticalFlowGridSize ffxOpticalFlowGetGridSize(FfxOpticalFlowContext* context)
{
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    const auto                  gridSize       = contextPrivate ? contextPrivate->contextDescription.gridSize : FFX_OPTICAL_FLOW_GRID_SIZE_UNKNOWN;
    return gridSize;
}

FFX_API FfxErrorCode ffxOpticalFlowGetSharedResourceDescriptions(FfxOpticalFlowContext* context, FfxOpticalFlowSharedResourceDescriptions* SharedResources)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(SharedResources, FFX_ERROR_INVALID_POINTER);

    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    FFX_ASSERT(contextPrivate);

    SharedResources->opticalFlowVector = {FFX_HEAP_TYPE_DEFAULT,
                                          {FFX_RESOURCE_TYPE_TEXTURE2D,
                                           FFX_SURFACE_FORMAT_R16G16_FLOAT,
                                           contextPrivate->opticalFlowSize.width,
                                           contextPrivate->opticalFlowSize.height,
                                           1,
                                           1,
                                           FFX_RESOURCE_FLAGS_NONE,
                                           FfxResourceUsage(FFX_RESOURCE_USAGE_UAV)},
                                          FFX_RESOURCE_STATE_GENERIC_UAV,
                                          "ARM_OpticalFlow_Result",
                                          0,
                                          {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};
    SharedResources->depthTm1          = {FFX_HEAP_TYPE_DEFAULT,
                                 {FFX_RESOURCE_TYPE_TEXTURE2D,
                                  FFX_SURFACE_FORMAT_R32_FLOAT,
                                  contextPrivate->contextDescription.maxRenderSize.width,
                                  contextPrivate->contextDescription.maxRenderSize.height,
                                  1,
                                  1,
                                  FFX_RESOURCE_FLAGS_NONE,
                                  FfxResourceUsage(FFX_RESOURCE_USAGE_RENDERTARGET)},
                                 FFX_RESOURCE_STATE_GENERIC_READ,
                                 "ARM_DepthTm1",
                                 0,
                                 {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};
    SharedResources->depthTm1Next      = SharedResources->depthTm1;
    SharedResources->depthTm1Next.name = "ARM_DepthTm1_Next";
    SharedResources->colorTm1          = {FFX_HEAP_TYPE_DEFAULT,
                                 {FFX_RESOURCE_TYPE_TEXTURE2D,
                                  contextPrivate->contextDescription.backBufferFormat,
                                  contextPrivate->contextDescription.resolution.width,
                                  contextPrivate->contextDescription.resolution.height,
                                  1,
                                  1,
                                  FFX_RESOURCE_FLAGS_NONE,
                                  FfxResourceUsage(FFX_RESOURCE_USAGE_UAV)},
                                 FFX_RESOURCE_STATE_GENERIC_UAV,
                                 "ARM_ColorTm1",
                                 0,
                                 {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};
    return FFX_OK;
}

FfxErrorCode ffxOpticalFlowContextDispatch(FfxOpticalFlowContext* context, FfxOpticalFlowDispatchDescription* dispatchParams)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->commandList, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->color.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->depth.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.type == FFX_RESOURCE_TYPE_TEXTURE2D, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->opticalFlowVector.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->depthTm1.resource != NULL, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams->colorTm1.resource != NULL, FFX_ERROR_INVALID_POINTER);

    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);

    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.width == contextPrivate->contextDescription.resolution.width, FFX_ERROR_INVALID_ARGUMENT);
    FFX_RETURN_ON_ERROR(dispatchParams->color.description.height == contextPrivate->contextDescription.resolution.height, FFX_ERROR_INVALID_ARGUMENT);

    const FfxErrorCode errorCode = ArmOpticalFlowDispatch(contextPrivate, dispatchParams);
    return errorCode;
}

FfxDimensions2D ffxOpticalFlowGetSize(FfxOpticalFlowContext* context)
{
    OpticalFlowContext_Private* contextPrivate = (OpticalFlowContext_Private*)(context);
    return contextPrivate ? contextPrivate->opticalFlowSize : FfxDimensions2D{0, 0};
}
