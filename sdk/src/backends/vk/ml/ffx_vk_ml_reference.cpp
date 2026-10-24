// SPDX-FileCopyrightText: Copyright 2026 Neural Graphics SDK universal contributors
// SPDX-License-Identifier: MIT

// Straightforward CPU evaluation of the parsed TOSA graph, operator by operator, following the TOSA 1.0 reference
// semantics. It deliberately does not reuse the GPU lowering so it can validate the fusion as well as the kernels.

#include "ffx_vk_ml_graph.h"

#include <algorithm>

namespace ffx_ml
{
    namespace
    {
        int32_t ApplyScale32(int32_t value, int32_t multiplier, int32_t shift, bool doubleRound)
        {
            int64_t round = int64_t(1) << (shift - 1);
            if (doubleRound && shift > 31)
                round += (value >= 0) ? (int64_t(1) << 30) : -(int64_t(1) << 30);
            int64_t result = int64_t(value) * int64_t(multiplier) + round;
            result >>= shift;
            return int32_t(result);
        }

        int32_t NearestCoord(int32_t v, int32_t n, int32_t d, int32_t offset, int32_t size)
        {
            const int32_t y  = v * d + offset;
            int32_t       iy = y / n;
            if (iy * n > y)
                iy--;
            const int32_t r  = y - iy * n;
            const int32_t i0 = std::max(iy, 0);
            const int32_t i1 = std::min(iy + 1, size - 1);
            return (2 * r >= n) ? i1 : i0;
        }
    }  // namespace

    bool RunReference(const GraphDesc& desc, const std::map<uint32_t, std::vector<int8_t>>& inputs, std::map<uint32_t, std::vector<int8_t>>& outputs, std::string* error)
    {
        using namespace tosa;
        tosa::Graph graph;
        if (!Parse(desc, graph, error))
            return false;

        std::vector<std::vector<int32_t>> data(graph.values.size());  // all values widened to int32, NHWC

        for (uint32_t vi = 0; vi < graph.values.size(); ++vi)
        {
            const Value& v = graph.values[vi];
            if (v.kind != Value::Input)
                continue;
            auto it = inputs.find(graph.inputBindings[v.index]);
            if (it == inputs.end() || it->second.size() != size_t(v.h) * v.w * v.c)
            {
                if (error)
                    *error = "reference: missing or mis-sized input";
                return false;
            }
            data[vi].assign(it->second.begin(), it->second.end());
        }

        for (const Node& node : graph.nodes)
        {
            const Value&          out = graph.values[node.result];
            std::vector<int32_t>& dst = data[node.result];
            dst.assign(size_t(out.h) * out.w * out.c, 0);

            switch (node.op)
            {
            case OP_CONV2D:
            {
                const Value&                in  = graph.values[node.inputs[0]];
                const std::vector<int32_t>& src = data[node.inputs[0]];
                for (int32_t oy = 0; oy < out.h; ++oy)
                    for (int32_t ox = 0; ox < out.w; ++ox)
                        for (int32_t oc = 0; oc < out.c; ++oc)
                        {
                            int32_t acc = node.bias[oc];
                            for (int32_t ky = 0; ky < node.kernelH; ++ky)
                                for (int32_t kx = 0; kx < node.kernelW; ++kx)
                                {
                                    const int32_t y = oy * node.stride[0] - node.pad[0] + ky * node.dilation[0];
                                    const int32_t x = ox * node.stride[1] - node.pad[2] + kx * node.dilation[1];
                                    if (y < 0 || x < 0 || y >= in.h || x >= in.w)
                                        continue;
                                    const int32_t* px = &src[(size_t(y) * in.w + x) * in.c];
                                    const int8_t*  wt = &node.weights[((size_t(oc) * node.kernelH + ky) * node.kernelW + kx) * in.c];
                                    for (int32_t ic = 0; ic < in.c; ++ic)
                                        acc += (px[ic] - node.inputZp) * (int32_t(wt[ic]) - node.weightZp);
                                }
                            dst[(size_t(oy) * out.w + ox) * out.c + oc] = acc;
                        }
                break;
            }
            case OP_RESCALE:
            {
                const std::vector<int32_t>& src = data[node.inputs[0]];
                for (size_t i = 0; i < dst.size(); ++i)
                {
                    const size_t  c  = node.perChannel ? i % size_t(out.c) : 0;
                    int32_t       r  = ApplyScale32(src[i] - node.rescaleInZp, node.multiplier[c], node.shift[c], node.doubleRound);
                    dst[i]           = std::clamp(r + node.rescaleOutZp, -128, 127);
                }
                break;
            }
            case OP_TABLE:
            {
                const std::vector<int32_t>& src = data[node.inputs[0]];
                for (size_t i = 0; i < dst.size(); ++i)
                    dst[i] = node.table[size_t(src[i] + 128)];
                break;
            }
            case OP_CONCAT:
            {
                int32_t channelBase = 0;
                for (uint32_t in : node.inputs)
                {
                    const Value&                v   = graph.values[in];
                    const std::vector<int32_t>& src = data[in];
                    for (int32_t p = 0; p < out.h * out.w; ++p)
                        for (int32_t c = 0; c < v.c; ++c)
                            dst[size_t(p) * out.c + channelBase + c] = src[size_t(p) * v.c + c];
                    channelBase += v.c;
                }
                break;
            }
            case OP_RESIZE:
            {
                const Value&                in  = graph.values[node.inputs[0]];
                const std::vector<int32_t>& src = data[node.inputs[0]];
                for (int32_t oy = 0; oy < out.h; ++oy)
                    for (int32_t ox = 0; ox < out.w; ++ox)
                    {
                        const int32_t iy = NearestCoord(oy, node.scale[0], node.scale[1], node.offset[0], in.h);
                        const int32_t ix = NearestCoord(ox, node.scale[2], node.scale[3], node.offset[1], in.w);
                        for (int32_t c = 0; c < out.c; ++c)
                            dst[(size_t(oy) * out.w + ox) * out.c + c] = src[(size_t(iy) * in.w + ix) * in.c + c];
                    }
                break;
            }
            default:
                if (error)
                    *error = "reference: unsupported operator";
                return false;
            }
        }

        outputs.clear();
        for (size_t i = 0; i < graph.outputValues.size(); ++i)
        {
            const std::vector<int32_t>& src = data[graph.outputValues[i]];
            outputs[graph.outputBindings[i]].assign(src.begin(), src.end());
        }
        return true;
    }
}  // namespace ffx_ml
