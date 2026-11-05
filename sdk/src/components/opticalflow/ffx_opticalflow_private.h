// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_NULL                    0
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR                   1
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_COLOR              2
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH                   3
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PREV_DEPTH              4
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COMPUTED_MOTION_VECTORS 5
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_RESULT                  6
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH_1                 7
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_DEPTH_2                 8
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR_1                 9
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COLOR_2                 10
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_GAME_MOTION_VECTORS     11
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_DEPTH           12  // nearest TP1 depth per TM1 flow texel
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FIXED_MOTION_VECTORS    13  // the game's, corrected by block matching
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_LUMA_TP1                14  // half-resolution packed luma, ping-ponged with
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_LUMA_TM1                15  // the previous frame's
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_BLOCK_FLOW              16  // the full search's flow per 32x32 pixel block
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_MOTION_FIX              17  // a correction per 8x8 pixels
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_LUMAQ_TP1               18  // the same at a quarter of the resolution
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_LUMAQ_TM1               19
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TP1             20  // image flow: the luma pyramid (all levels in one
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_PYRAMID_TM1             21  // image), ping-ponged with the previous frame's
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_RAW                22  // each level's searched vectors (likewise)
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_LEVELS             23  // ... filtered, the next finer level's seeds
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_FLOW_HINT               24  // the flow from the motion vectors and depth
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_SCATTER_KEYS            25  // depth key and flow per texel (64-bit atomics)
#define FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT                   26

#define FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER 0
#define FFX_OPTICALFLOW_CONSTANTBUFFER_COUNT      1

struct FfxPipelineState;

typedef enum OpticalFlowShaderPermutationOptions
{
    OPTICALFLOW_SHADER_PERMUTATION_DEPTH_INVERTED    = (1 << 0),
    OPTICALFLOW_SHADER_PERMUTATION_MV_HINTS_FRAGMENT = (1 << 1),
} OpticalFlowShaderPermutationOptions;

typedef struct OpticalFlowComputeMVHintsConstants
{
    float   motion_matrix_m1p1[16];
    int32_t output_dims[2];
    int32_t depth_size[2];
    float   depth_size_rcp[2];
    int32_t color_size[2];
} OpticalFlowComputeMVHintsConstants;

typedef struct OpticalFlowScatterConstants
{
    float   motion_vector_scale[2];
    int32_t output_dims[2];
    int32_t motion_vector_dims[2];
    int32_t depth_dims[2];
    int32_t depth_inverted;
    int32_t pad;
    int32_t display_dims[2];
} OpticalFlowScatterConstants;

typedef struct OpticalFlowRefineConstants
{
    int32_t  output_dims[2];
    int32_t  luma_dims[2];
    int32_t  color_dims[2];
    int32_t  packed_width;
    uint32_t penalty_candidate;
    uint32_t penalty_step;
    int32_t  match_ratio;
    int32_t  block_dims[2];
    int32_t  depth_inverted;
    int32_t  packed_width_q;
    int32_t  mv_dims[2];
    float    mv_scale[2];
} OpticalFlowRefineConstants;

// The image flow's, per pass and level (ffx_opticalflow_image_common.h).
#define FFX_OPTICALFLOW_IMAGE_MAX_LEVELS 5
typedef struct OpticalFlowImageConstants
{
    int32_t  level_dims[2];
    int32_t  src_dims[2];
    int32_t  cell_dims[2];
    int32_t  coarse_dims[2];
    int32_t  hint_dims[2];
    float    out_scale[2];
    int32_t  level_row;
    int32_t  flow_row;
    int32_t  coarse_row;
    int32_t  level;
    uint32_t lambda_hint;
    uint32_t lambda_seed;
    int32_t  level_count;
    uint32_t settled_cost;
    int32_t  final;
    int32_t  pad[3];
    int32_t  levels[FFX_OPTICALFLOW_IMAGE_MAX_LEVELS][4];
} OpticalFlowImageConstants;

// The private implementation of the arm data graph optical flow context.
typedef struct OpticalFlowContext_Private
{
    FfxOpticalFlowContextDescription   contextDescription;
    FfxUInt32                          effectContextId;
    FfxDevice                          device;
    FfxPipelineState                   pipelineOpticalflow;
    FfxPipelineState                   pipelineComputeMVHints;
    FfxPipelineState                   pipelineFragmentMVHints;
    FfxPipelineState                   pipelineScatterDepth;  // hintsOnly: the flow from the game's motion vectors
    FfxPipelineState                   pipelineScatterFlow;
    FfxPipelineState                   pipelineLuma;    // hintsOnly: block matching on the image over that flow
    FfxPipelineState                   pipelineRefine;
    FfxPipelineState                   pipelineSearch;
    FfxPipelineState                   pipelineApply;
    FfxPipelineState                   pipelinePyramid;  // hintsOnly: image optical flow over that flow
    FfxPipelineState                   pipelineFlowSearch;
    FfxPipelineState                   pipelineFlowFilter;
    FfxPipelineState                   pipelineFlowSearchFine;
    FfxPipelineState                   pipelineScatter;  // the scatter in one pass (64-bit atomics)
    FfxPipelineState                   pipelineScatterResolve;
    bool                               scatter64;
    bool                               imageFlow;
    bool                               pyramidValid;    // the TM1 pyramid holds the previous frame
    uint32_t                           levelCount;
    uint32_t                           finestLevel;  // searched; the flow is its vectors at level 0's cells
    FfxDimensions2D                    levelDims[FFX_OPTICALFLOW_IMAGE_MAX_LEVELS];  // luma texels
    int32_t                            levelRow[FFX_OPTICALFLOW_IMAGE_MAX_LEVELS];   // in the pyramid
    int32_t                            flowRow[FFX_OPTICALFLOW_IMAGE_MAX_LEVELS];    // in the flow images
    uint32_t                           lambdaHint, lambdaSeed, settledCost;
    FfxResourceInternal                pyramid[2];
    bool                               refine;
    bool                               lumaValid;       // the TM1 luma holds the previous frame
    bool                               fixedValid;      // this frame's corrected motion vectors are written
    FfxResourceInternal                luma[2];
    FfxResourceInternal                lumaQ[2];
    OpticalFlowRefineConstants         refineConstants;
    bool                               useMVHintsFragment;
    bool                               hintsOnly;  // no data graph optical flow engine: the MV hints are the flow
    FfxResourceInternal                srvResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal                uavResources[FFX_OPTICALFLOW_RESOURCE_IDENTIFIER_COUNT];
    FfxConstantBuffer                  constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_COUNT];
    FfxDimensions2D                    dimensions;
    OpticalFlowComputeMVHintsConstants computeMVHintsConstants;
    FfxFloat32x4x4                     lastFrameViewProjection;
    FfxDimensions2D                    opticalFlowSize;

    bool     firstExecution;
    uint32_t resourceFrameIndex;
} OpticalFlowContext_Private;
