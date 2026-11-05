// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// Image optical flow, step 2 at level 0 (the colour's resolution, 4x as many cells as level 1): the search of
// ffx_opticalflow_flow_search.glsl cut to +-1 texel around its seeds (level 1's vectors doubled, so that level 0
// adds their missing odd texel, and the hint), as FSR 3's ScaleOpticalFlowAdvanced chooses its finest level among
// the upscaled ones instead of searching it.

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

#ifndef RADIUS
#define RADIUS 1
#endif
#define FINE_LEVEL 1
#include "ffx_opticalflow_flow_search.h"
