// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Image optical flow without the data graph optical flow engine, step 2 of 3, one dispatch per pyramid level from
// the coarsest: block matching in the manner of FidelityFX FSR 3's optical flow (ffx_opticalflow_compute_optical_flow
// _v5.h: an 8x8 block searched around the coarser level's vector, level by level), for the interpolation's flow: TM1
// -> TP1 on its TM1 grid, a vector per 4x4 texel cell (the block centred on it). Each cell's seeds are the coarser
// level's vectors of the four nearest cells (doubled; FSR 3's ScaleOpticalFlowAdvanced chooses among the same), the
// hint flow (the game's motion vectors scattered over the camera's motion, ffx_opticalflow_scatter_flow.glsl) and
// zero; a full search of +-RADIUS texels follows around the best. The cost is the block's SSD plus _LambdaHint per
// texel of distance from the hint, so that where the image does not clearly contradict the motion vectors the flow
// stays theirs, and _LambdaSeed per texel from the seed, which settles flat regions on it.
// Without msad4 (FSR 3's D3D path), RDNA's packed 8-bit dot product (v_dot4_u32_u8, GL_EXT_integer_dot_product)
// gives the SSD four texels an instruction: SSD = sum a^2 + sum b^2 - 2 sum ab, and the search window's rows of b^2
// are shared by the offsets of a column. One invocation per cell, the window in registers.

#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : enable

#define RADIUS 3
#include "ffx_opticalflow_flow_search.h"
