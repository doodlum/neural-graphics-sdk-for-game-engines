// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
// The image optical flow's constants (OpticalFlowImageConstants), the same for its passes and per level.

layout(set = 0, binding = 0, std140) uniform Uniforms {
    FfxInt32x2   _LevelDims;    // this level's luma texels
    FfxInt32x2   _SrcDims;      // pyramid: the colour's pixels
    FfxInt32x2   _CellDims;     // flow cells (4x4 texels each) at this level
    FfxInt32x2   _CoarseDims;   // search: the coarser level's cells (none at the coarsest)
    FfxInt32x2   _HintDims;     // the hint flow's texels (level 0's cells)
    FfxFloat32x2 _OutScale;     // colour pixels per level 0 texel
    FfxInt32     _LevelRow;     // this level's first row in the pyramid
    FfxInt32     _FlowRow;      // this level's first row in the flow images
    FfxInt32     _CoarseRow;    // search: the coarser level's
    FfxInt32     _Level;        // 0: the colour's resolution
    FfxUInt32    _LambdaHint;   // search: added to a candidate's SSD per texel of distance from the hint flow
    FfxUInt32    _LambdaSeed;   // ... and from the seed the search is centred on
    FfxInt32     _LevelCount;   // pyramid: all levels in one dispatch
    FfxUInt32    _SettledCost;  // search: a cell whose seeds agree and cost no more is not searched
    FfxInt32     _Final;        // filter: the finest level searched, written to rw_flow per level 0 cell (pixels)
    FfxInt32     _Fsr3;         // pyramid: FSR 3's luma (FFX_OF_IMAGE=fsr3: Rec. 709, truncated, the levels the truncated mean)
    FfxInt32     _Unorm8;       // FFX_OF_IMAGE=arm, pyramid: the colour is 8-bit (the emulation layer's UNORM8_INPUT)
    FfxInt32     _Pad2;
    FfxInt32x4   _Levels[5];    // ... their texels (xy) and first rows (z)
}ubo;
#define _LevelDimsAll(l) (ubo._Levels[l].xy)
#define _LevelRows(l)    (ubo._Levels[l].z)
