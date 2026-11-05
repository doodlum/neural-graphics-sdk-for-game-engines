// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include <FidelityFX/host/ffx_util.h>
#include "ffx_opticalflow_shaderblobs.h"
#include "opticalflow/ffx_opticalflow_private.h"

#include <ffx_opticalflow_mv_hints_permutations.h>
#include <ffx_opticalflow_mv_hints_fs_permutations.h>
#include <ffx_opticalflow_mv_hints_vs_permutations.h>
#include <ffx_opticalflow_scatter_depth_permutations.h>
#include <ffx_opticalflow_scatter_flow_permutations.h>
#include <ffx_opticalflow_luma_permutations.h>
#include <ffx_opticalflow_refine_permutations.h>
#include <ffx_opticalflow_search_permutations.h>
#include <ffx_opticalflow_apply_permutations.h>
#include <ffx_opticalflow_pyramid_permutations.h>
#include <ffx_opticalflow_flow_search_permutations.h>
#include <ffx_opticalflow_flow_filter_permutations.h>
#include <ffx_opticalflow_flow_search_fine_permutations.h>
#include <ffx_opticalflow_scatter_permutations.h>
#include <ffx_opticalflow_scatter_resolve_permutations.h>
#include <string.h>  // for memset

#if defined(POPULATE_PERMUTATION_KEY)
#undef POPULATE_PERMUTATION_KEY
#endif  // #if defined(POPULATE_PERMUTATION_KEY)
#define POPULATE_PERMUTATION_KEY(options, key) \
    key.index          = 0;                    \
    key.DEPTH_INVERTED = FFX_CONTAINS_FLAG(options, OPTICALFLOW_SHADER_PERMUTATION_DEPTH_INVERTED);

static FfxShaderBlob opticalflowGetComputeMotionFromDepthPassPermutationBlobByIndex(uint32_t permutationOptions)
{
    ffx_opticalflow_mv_hints_PermutationKey key;

    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const bool useFragmentJob = FFX_CONTAINS_FLAG(permutationOptions, OPTICALFLOW_SHADER_PERMUTATION_MV_HINTS_FRAGMENT);
    if (useFragmentJob)
    {
        const int32_t tableIndex = g_ffx_opticalflow_mv_hints_fs_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_mv_hints_fs_PermutationInfo, tableIndex);
    }

    {
        const int32_t tableIndex = g_ffx_opticalflow_mv_hints_IndirectionTable[key.index];
        return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_mv_hints_PermutationInfo, tableIndex);
    }
}

static FfxShaderBlob opticalflowGetVertexPermutationBlobByIndex(uint32_t permutationOptions)
{
    ffx_opticalflow_mv_hints_vs_PermutationKey key;
    POPULATE_PERMUTATION_KEY(permutationOptions, key);

    const int32_t tableIndex = g_ffx_opticalflow_mv_hints_vs_IndirectionTable[key.index];
    return POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_mv_hints_vs_PermutationInfo, tableIndex);
}

FfxErrorCode opticalflowGetPermutationBlobByIndex(FfxOpticalflowPass passId, uint32_t permutationOptions, FfxShaderBlob* outBlob, FfxShaderBlob* outVertBlob)
{
    if (outVertBlob)
    {
        FfxShaderBlob blob = opticalflowGetVertexPermutationBlobByIndex(permutationOptions);
        memcpy(outVertBlob, &blob, sizeof(FfxShaderBlob));
    }

    switch (passId)
    {
    case FFX_OPTICALFLOW_PASS_COMPUTE_MOTION_FROM_DEPTH:
    {
        FfxShaderBlob blob = opticalflowGetComputeMotionFromDepthPassPermutationBlobByIndex(permutationOptions);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    case FFX_OPTICALFLOW_PASS_SCATTER_DEPTH:
    {
        ffx_opticalflow_scatter_depth_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_scatter_depth_PermutationInfo, g_ffx_opticalflow_scatter_depth_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_SCATTER_FLOW:
    {
        ffx_opticalflow_scatter_flow_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_scatter_flow_PermutationInfo, g_ffx_opticalflow_scatter_flow_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_LUMA:
    {
        ffx_opticalflow_luma_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_luma_PermutationInfo, g_ffx_opticalflow_luma_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_SEARCH:
    {
        ffx_opticalflow_search_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_search_PermutationInfo, g_ffx_opticalflow_search_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_APPLY:
    {
        ffx_opticalflow_apply_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_apply_PermutationInfo, g_ffx_opticalflow_apply_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_REFINE:
    {
        ffx_opticalflow_refine_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_refine_PermutationInfo, g_ffx_opticalflow_refine_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_PYRAMID:
    {
        ffx_opticalflow_pyramid_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_pyramid_PermutationInfo, g_ffx_opticalflow_pyramid_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_FLOW_SEARCH:
    {
        ffx_opticalflow_flow_search_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_flow_search_PermutationInfo, g_ffx_opticalflow_flow_search_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_FLOW_FILTER:
    {
        ffx_opticalflow_flow_filter_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_flow_filter_PermutationInfo, g_ffx_opticalflow_flow_filter_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_FLOW_SEARCH_FINE:
    {
        ffx_opticalflow_flow_search_fine_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_flow_search_fine_PermutationInfo, g_ffx_opticalflow_flow_search_fine_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_SCATTER:
    {
        ffx_opticalflow_scatter_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_scatter_PermutationInfo, g_ffx_opticalflow_scatter_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }
    case FFX_OPTICALFLOW_PASS_SCATTER_RESOLVE:
    {
        ffx_opticalflow_scatter_resolve_PermutationKey key;
        POPULATE_PERMUTATION_KEY(permutationOptions, key);
        FfxShaderBlob blob = POPULATE_SHADER_BLOB_FFX(g_ffx_opticalflow_scatter_resolve_PermutationInfo, g_ffx_opticalflow_scatter_resolve_IndirectionTable[key.index]);
        memcpy(outBlob, &blob, sizeof(FfxShaderBlob));
        return FFX_OK;
    }

    default:
        FFX_ASSERT_MESSAGE(false, "Not implemented");
        break;
    }

    // return an empty blob
    memset(outBlob, 0, sizeof(FfxShaderBlob));
    return FFX_OK;
}

FfxErrorCode opticalflowIsWave64(uint32_t permutationOptions, bool& isWave64)
{
    isWave64 = false;
    return FFX_OK;
}