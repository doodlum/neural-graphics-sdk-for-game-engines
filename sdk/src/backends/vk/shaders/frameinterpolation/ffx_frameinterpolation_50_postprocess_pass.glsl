// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_KHR_shader_subgroup_basic : require

//-----------------------------------------------------------------------------
// Input: SRV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_OUT_PARAMS_TENSOR               0
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TP1          1
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_FLOW_TM1            2
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_WARP_FILLED_MOTION_TM1          3
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE   4
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE    5
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_UI_SOURCE                       9
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_UI_ALPHA                        10
#define FFX_ARM_FRAMEINTERPOLATION_BIND_SRV_NO_WARP_MASK                    11

//-----------------------------------------------------------------------------
// Input: UAV bindings
//-----------------------------------------------------------------------------
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_OUTPUT                          6

#define FFX_ARM_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION               7
// UAV that the postprocess pass writes to so the next frame can read it as colorTm1.
#define FFX_ARM_FRAMEINTERPOLATION_BIND_UAV_PREVIOUS_INTERPOLATION_SOURCE   8


#define FFX_FRAMEINTERPOLATION_QUAD_UI 1
#include "frameinterpolation/ffx_frameinterpolation_50_postprocess_pass.h"

//------------------------------------------------------------------------------

// entry-point
#ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
#define FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS layout (local_size_x = FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, local_size_y = FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT, local_size_z = FFX_FRAMEINTERPOLATION_THREAD_GROUP_DEPTH) in;
#endif // #ifndef FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS

FFX_ARM_FRAMEINTERPOLATION_NUM_THREADS
void main()
{
    // Each 4 consecutive lanes a 2x2 block (FFX_FRAMEINTERPOLATION_QUAD_UI), 8x8 such blocks a group. From the lane
    // itself, not the local invocation index: with quad operations the driver may reorder local invocations.
    const uint32_t i = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    const int32_t2 q = int32_t2((i >> 2) % (FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH / 2), (i >> 2) / (FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH / 2));
    postprocess(int32_t2(gl_WorkGroupID.xy) * int32_t2(FFX_FRAMEINTERPOLATION_THREAD_GROUP_WIDTH, FFX_FRAMEINTERPOLATION_THREAD_GROUP_HEIGHT) +
                q * 2 + int32_t2(i & 1u, (i >> 1) & 1u));
}