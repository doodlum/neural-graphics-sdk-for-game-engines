// SPDX-FileCopyrightText: Copyright 2026 Neural Graphics SDK universal contributors
// SPDX-License-Identifier: MIT

// TOSA (SPIR-V graph) parsing and lowering to fused int8 conv layers. Internal to the ML runtime.

#pragma once

#include "ffx_vk_ml_runtime.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ffx_ml
{
    namespace tosa
    {
        enum Op : uint32_t
        {
            OP_CONV2D   = 2,
            OP_TABLE    = 30,
            OP_CONCAT   = 54,
            OP_RESIZE   = 63,
            OP_RESCALE  = 65,
        };

        // A TOSA value (always rank-4 NHWC tensors in the graphs we execute, batch 1).
        struct Value
        {
            enum Kind
            {
                Input,     // graph input, index = input index
                OpResult,  // result of node[index]
            };
            Kind     kind  = OpResult;
            uint32_t index = 0;
            int32_t  h = 0, w = 0, c = 0;
            uint32_t elementBits = 8;
        };

        struct Node
        {
            uint32_t              op = 0;
            uint32_t              result = 0;            // value index
            std::vector<uint32_t> inputs;                // value indices

            // CONV2D
            std::array<int32_t, 4> pad      = {0, 0, 0, 0};  // top, bottom, left, right
            std::array<int32_t, 2> stride   = {1, 1};
            std::array<int32_t, 2> dilation = {1, 1};
            int32_t                inputZp  = 0;
            int32_t                weightZp = 0;
            std::vector<int8_t>    weights;              // [OC][KH][KW][IC]
            std::vector<int32_t>   bias;                 // [OC]
            int32_t                kernelH = 0, kernelW = 0;

            // RESCALE
            bool                 scale32     = true;
            bool                 doubleRound = false;
            bool                 perChannel  = false;
            std::vector<int32_t> multiplier;
            std::vector<int32_t> shift;
            int32_t              rescaleInZp  = 0;
            int32_t              rescaleOutZp = 0;

            // TABLE
            std::vector<int8_t> table;                   // 256 entries

            // RESIZE
            uint32_t               resizeMode = 0;       // 1 = nearest, 2 = bilinear
            std::array<int32_t, 4> scale  = {1, 1, 1, 1};// y_n, y_d, x_n, x_d
            std::array<int32_t, 2> offset = {0, 0};
            std::array<int32_t, 2> border = {0, 0};

            // CONCAT
            uint32_t axis = 3;
        };

        struct Graph
        {
            std::vector<Value>    values;
            std::vector<Node>     nodes;                 // topological order
            std::vector<uint32_t> inputBindings;         // per graph input
            std::vector<uint32_t> outputBindings;        // per graph output
            std::vector<uint32_t> outputValues;          // per graph output
        };

        // Parses the TOSA graph module and infers all shapes from the input shapes in desc.
        bool Parse(const GraphDesc& desc, Graph& out, std::string* error);
    }  // namespace tosa

    // A stored (materialised) int8 NHWC tensor.
    struct StoredTensor
    {
        int32_t  h = 0, w = 0, c = 0;
        int32_t  channelStride = 0;     // stored channels per pixel
        bool     external = false;
        uint32_t binding  = 0;          // external tensors
        uint32_t internalIndex = 0;     // internal tensors
    };

    // One conv input source: a stored tensor, optionally nearest-resized.
    struct ConvSource
    {
        uint32_t tensor = 0;            // StoredTensor index
        bool     resize = false;
        std::array<int32_t, 4> scale  = {1, 1, 1, 1};
        std::array<int32_t, 2> offset = {0, 0};
    };

    // CONV2D + RESCALE (+ TABLE), the unit the GPU executes.
    struct FusedConv
    {
        std::vector<ConvSource> sources;   // concatenated along channels
        int32_t inH = 0, inW = 0, inC = 0; // virtual input
        int32_t outH = 0, outW = 0, outC = 0;
        uint32_t output = 0;               // StoredTensor index
        const tosa::Node* conv    = nullptr;
        const tosa::Node* rescale = nullptr;
        const tosa::Node* table   = nullptr;
    };

    struct LoweredGraph
    {
        std::vector<StoredTensor> tensors;
        std::vector<FusedConv>    convs;
        uint32_t                  internalTensorCount = 0;
    };

    bool Lower(const tosa::Graph& graph, LoweredGraph& out, std::string* error);
}  // namespace ffx_ml
