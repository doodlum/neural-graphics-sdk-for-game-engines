// SPDX-FileCopyrightText: Copyright 2026 Neural Graphics SDK universal contributors
// SPDX-License-Identifier: MIT

#include "ffx_vk_ml_graph.h"

#include <algorithm>
#include <functional>
#include <cstring>
#include <map>
#include <unordered_map>

namespace ffx_ml
{
    namespace tosa
    {
        namespace
        {
            // SPIR-V opcodes used by TOSA graph modules (SPV_ARM_graph, SPV_ARM_tensors, SPV_EXT_replicated_composites).
            enum SpvOp : uint32_t
            {
                SpvOpName                         = 5,
                SpvOpExtInstImport                = 11,
                SpvOpExtInst                      = 12,
                SpvOpTypeBool                     = 20,
                SpvOpTypeInt                      = 21,
                SpvOpTypeArray                    = 28,
                SpvOpConstantTrue                 = 41,
                SpvOpConstantFalse                = 42,
                SpvOpConstant                     = 43,
                SpvOpConstantComposite            = 44,
                SpvOpConstantNull                 = 46,
                SpvOpVariable                     = 59,
                SpvOpDecorate                     = 71,
                SpvOpTypeTensorARM                = 4163,
                SpvOpGraphConstantARM             = 4181,
                SpvOpGraphEntryPointARM           = 4182,
                SpvOpGraphARM                     = 4183,
                SpvOpGraphInputARM                = 4184,
                SpvOpGraphSetOutputARM            = 4185,
                SpvOpGraphEndARM                  = 4186,
                SpvOpConstantCompositeReplicateEXT = 4461,
            };

            constexpr uint32_t SpvDecorationBinding = 33;

            struct SpvType
            {
                uint32_t              opcode = 0;
                uint32_t              width  = 0;  // int
                uint32_t              elementType = 0;
                uint32_t              lengthId = 0;  // array
                uint32_t              rankId = 0;    // tensor
                uint32_t              shapeId = 0;   // tensor
            };

            struct SpvConstant
            {
                uint32_t             typeId = 0;
                std::vector<int64_t> values;  // flattened, sign extended according to the element width
                bool                 isGraphConstant = false;
                uint32_t             graphConstantId = 0;
            };

            struct Parser
            {
                const GraphDesc&                           desc;
                std::string*                               error;
                std::unordered_map<uint32_t, SpvType>      types;
                std::unordered_map<uint32_t, SpvConstant>  constants;
                std::unordered_map<uint32_t, uint32_t>     bindings;     // variable id -> binding
                std::vector<uint32_t>                      interfaceIds; // entry point interface
                std::unordered_map<uint32_t, uint32_t>     idToValue;    // SSA id -> value index
                uint32_t                                   tosaSet = 0;
                Graph&                                     graph;

                bool Fail(const std::string& message)
                {
                    if (error)
                        *error = message;
                    return false;
                }

                static int64_t SignExtend(uint64_t v, uint32_t bits)
                {
                    if (bits >= 64)
                        return int64_t(v);
                    const uint64_t m = 1ull << (bits - 1);
                    v &= (1ull << bits) - 1;
                    return int64_t((v ^ m) - m);
                }

                uint32_t ElementWidth(uint32_t typeId) const
                {
                    auto it = types.find(typeId);
                    if (it == types.end())
                        return 32;
                    const SpvType& t = it->second;
                    if (t.opcode == SpvOpTypeInt)
                        return t.width;
                    if (t.opcode == SpvOpTypeBool)
                        return 1;
                    if (t.opcode == SpvOpTypeArray || t.opcode == SpvOpTypeTensorARM)
                        return ElementWidth(t.elementType);
                    return 32;
                }

                int64_t ScalarValue(uint32_t id) const
                {
                    auto it = constants.find(id);
                    return (it != constants.end() && !it->second.values.empty()) ? it->second.values[0] : 0;
                }

                // Number of elements described by a composite type.
                size_t ElementCount(uint32_t typeId) const
                {
                    auto it = types.find(typeId);
                    if (it == types.end())
                        return 1;
                    const SpvType& t = it->second;
                    if (t.opcode == SpvOpTypeArray)
                        return size_t(ScalarValue(t.lengthId)) * ElementCount(t.elementType);
                    if (t.opcode == SpvOpTypeTensorARM)
                    {
                        if (t.shapeId == 0)
                            return 0;
                        size_t n = 1;
                        for (int64_t d : constants.at(t.shapeId).values)
                            n *= size_t(d);
                        return n;
                    }
                    return 1;
                }

                std::vector<int64_t> TensorShape(uint32_t typeId) const
                {
                    auto it = types.find(typeId);
                    if (it == types.end() || it->second.opcode != SpvOpTypeTensorARM || it->second.shapeId == 0)
                        return {};
                    return constants.at(it->second.shapeId).values;
                }

                // Integer values of a constant, whether inline or a graph constant from the VGF.
                bool ConstantValues(uint32_t id, std::vector<int64_t>& out)
                {
                    auto it = constants.find(id);
                    if (it == constants.end())
                        return Fail("TOSA operand is not a constant");
                    const SpvConstant& c = it->second;
                    if (!c.isGraphConstant)
                    {
                        out = c.values;
                        return true;
                    }
                    const GraphConstant* gc = nullptr;
                    for (const GraphConstant& candidate : desc.constants)
                    {
                        if (candidate.id == c.graphConstantId)
                            gc = &candidate;
                    }
                    if (!gc)
                        return Fail("missing graph constant " + std::to_string(c.graphConstantId));
                    const uint32_t bits  = ElementWidth(c.typeId);
                    const uint32_t bytes = std::max(1u, bits / 8);
                    const size_t   count = gc->dataSize / bytes;
                    out.resize(count);
                    for (size_t i = 0; i < count; ++i)
                    {
                        uint64_t raw = 0;
                        memcpy(&raw, gc->data + i * bytes, bytes);
                        out[i] = SignExtend(raw, bits);
                    }
                    return true;
                }

                uint32_t AddValue(const std::vector<int64_t>& shape, uint32_t bits, Value::Kind kind, uint32_t index)
                {
                    Value v;
                    v.kind        = kind;
                    v.index       = index;
                    v.elementBits = bits;
                    if (shape.size() == 4)
                    {
                        v.h = int32_t(shape[1]);
                        v.w = int32_t(shape[2]);
                        v.c = int32_t(shape[3]);
                    }
                    graph.values.push_back(v);
                    return uint32_t(graph.values.size() - 1);
                }

                bool Run()
                {
                    const uint32_t* words = desc.spirv;
                    const size_t    count = desc.spirvWords;
                    if (count < 5 || words[0] != 0x07230203u)
                        return Fail("not a SPIR-V module");

                    for (size_t pos = 5; pos < count;)
                    {
                        const uint32_t wordCount = words[pos] >> 16;
                        const uint32_t opcode    = words[pos] & 0xFFFFu;
                        if (wordCount == 0 || pos + wordCount > count)
                            return Fail("malformed SPIR-V");
                        const uint32_t* w = words + pos;
                        if (!Instruction(opcode, w, wordCount))
                            return false;
                        pos += wordCount;
                    }
                    return true;
                }

                bool Instruction(uint32_t opcode, const uint32_t* w, uint32_t n)
                {
                    switch (opcode)
                    {
                    case SpvOpExtInstImport:
                        if (strcmp(reinterpret_cast<const char*>(w + 2), "TOSA.001000.1") != 0)
                            return Fail(std::string("unsupported extended instruction set ") + reinterpret_cast<const char*>(w + 2));
                        tosaSet = w[1];
                        return true;
                    case SpvOpTypeInt:
                        types[w[1]] = {opcode, w[2]};
                        return true;
                    case SpvOpTypeBool:
                        types[w[1]] = {opcode, 1};
                        return true;
                    case SpvOpTypeArray:
                    {
                        SpvType t;
                        t.opcode      = opcode;
                        t.elementType = w[2];
                        t.lengthId    = w[3];
                        types[w[1]]   = t;
                        return true;
                    }
                    case SpvOpTypeTensorARM:
                    {
                        SpvType t;
                        t.opcode      = opcode;
                        t.elementType = w[2];
                        t.rankId      = n > 3 ? w[3] : 0;
                        t.shapeId     = n > 4 ? w[4] : 0;
                        types[w[1]]   = t;
                        return true;
                    }
                    case SpvOpConstantTrue:
                    case SpvOpConstantFalse:
                        constants[w[2]] = {w[1], {opcode == SpvOpConstantTrue ? 1 : 0}};
                        return true;
                    case SpvOpConstant:
                    {
                        const uint32_t bits = ElementWidth(w[1]);
                        uint64_t       raw  = w[3];
                        if (bits > 32 && n > 4)
                            raw |= uint64_t(w[4]) << 32;
                        constants[w[2]] = {w[1], {SignExtend(raw, bits)}};
                        return true;
                    }
                    case SpvOpConstantComposite:
                    {
                        SpvConstant c;
                        c.typeId = w[1];
                        for (uint32_t i = 3; i < n; ++i)
                        {
                            const auto& part = constants[w[i]].values;
                            c.values.insert(c.values.end(), part.begin(), part.end());
                        }
                        constants[w[2]] = std::move(c);
                        return true;
                    }
                    case SpvOpConstantCompositeReplicateEXT:
                    {
                        SpvConstant c;
                        c.typeId                     = w[1];
                        const std::vector<int64_t> v = constants[w[3]].values;
                        const size_t total           = ElementCount(w[1]);
                        while (c.values.size() < total)
                            c.values.insert(c.values.end(), v.begin(), v.end());
                        c.values.resize(total);
                        constants[w[2]] = std::move(c);
                        return true;
                    }
                    case SpvOpConstantNull:
                    {
                        SpvConstant c;
                        c.typeId = w[1];
                        c.values.assign(std::max<size_t>(1, ElementCount(w[1])), 0);
                        constants[w[2]] = std::move(c);
                        return true;
                    }
                    case SpvOpGraphConstantARM:
                    {
                        SpvConstant c;
                        c.typeId          = w[1];
                        c.isGraphConstant = true;
                        c.graphConstantId = w[3];
                        constants[w[2]]   = std::move(c);
                        return true;
                    }
                    case SpvOpDecorate:
                        if (n >= 4 && w[2] == SpvDecorationBinding)
                            bindings[w[1]] = w[3];
                        return true;
                    case SpvOpGraphEntryPointARM:
                    {
                        // graph id, name (literal string), interface ids
                        uint32_t i = 2;
                        while (i < n)
                        {
                            const uint32_t word = w[i++];
                            if ((word >> 24) == 0 || ((word >> 16) & 0xFF) == 0 || ((word >> 8) & 0xFF) == 0 || (word & 0xFF) == 0)
                                break;
                        }
                        interfaceIds.assign(w + i, w + n);
                        return true;
                    }
                    case SpvOpGraphInputARM:
                    {
                        const uint32_t inputIndex = uint32_t(ScalarValue(w[3]));
                        if (inputIndex >= interfaceIds.size())
                            return Fail("graph input index out of range");
                        const uint32_t binding = bindings[interfaceIds[inputIndex]];
                        auto           shapeIt = desc.inputShapes.find(binding);
                        if (shapeIt == desc.inputShapes.end() || shapeIt->second.size() != 4)
                            return Fail("missing NHWC shape for graph input binding " + std::to_string(binding));
                        if (graph.inputBindings.size() <= inputIndex)
                            graph.inputBindings.resize(inputIndex + 1);
                        graph.inputBindings[inputIndex] = binding;
                        idToValue[w[2]] = AddValue(shapeIt->second, ElementWidth(w[1]), Value::Input, inputIndex);
                        return true;
                    }
                    case SpvOpGraphSetOutputARM:
                    {
                        const uint32_t outputIndex = uint32_t(ScalarValue(w[2]));
                        auto           it          = idToValue.find(w[1]);
                        if (it == idToValue.end())
                            return Fail("graph output is not a computed value");
                        const uint32_t interfaceIndex = uint32_t(graph.inputBindings.size()) + outputIndex;
                        if (interfaceIndex >= interfaceIds.size())
                            return Fail("graph output index out of range");
                        if (graph.outputBindings.size() <= outputIndex)
                        {
                            graph.outputBindings.resize(outputIndex + 1);
                            graph.outputValues.resize(outputIndex + 1);
                        }
                        graph.outputBindings[outputIndex] = bindings[interfaceIds[interfaceIndex]];
                        graph.outputValues[outputIndex]   = it->second;
                        return true;
                    }
                    case SpvOpExtInst:
                        if (w[3] != tosaSet)
                            return Fail("extended instruction from an unknown set");
                        return TosaOp(w[1], w[2], w[4], w + 5, n - 5);
                    default:
                        return true;
                    }
                }

                bool Operand(uint32_t id, uint32_t& valueIndex)
                {
                    auto it = idToValue.find(id);
                    if (it == idToValue.end())
                        return Fail("TOSA operand is not a tensor value");
                    valueIndex = it->second;
                    return true;
                }

                bool TosaOp(uint32_t resultType, uint32_t resultId, uint32_t op, const uint32_t* a, uint32_t n)
                {
                    Node node;
                    node.op = op;
                    std::vector<int64_t> v;
                    int32_t              outH = 0, outW = 0, outC = 0;
                    uint32_t             outBits = ElementWidth(resultType);

                    switch (op)
                    {
                    case OP_CONV2D:
                    {
                        if (n < 10)
                            return Fail("CONV2D: operand count");
                        if (!ConstantValues(a[0], v) || v.size() != 4)
                            return Fail("CONV2D: pad");
                        node.pad = {int32_t(v[0]), int32_t(v[1]), int32_t(v[2]), int32_t(v[3])};
                        if (!ConstantValues(a[1], v) || v.size() != 2)
                            return Fail("CONV2D: stride");
                        node.stride = {int32_t(v[0]), int32_t(v[1])};
                        if (!ConstantValues(a[2], v) || v.size() != 2)
                            return Fail("CONV2D: dilation");
                        node.dilation = {int32_t(v[0]), int32_t(v[1])};
                        uint32_t input;
                        if (!Operand(a[5], input))
                            return false;
                        node.inputs.push_back(input);

                        const std::vector<int64_t> wShape = TensorShape(constants[a[6]].typeId);
                        if (wShape.size() != 4)
                            return Fail("CONV2D: weight shape");
                        if (!ConstantValues(a[6], v))
                            return false;
                        node.weights.assign(v.begin(), v.end());
                        node.kernelH = int32_t(wShape[1]);
                        node.kernelW = int32_t(wShape[2]);
                        if (!ConstantValues(a[7], v))
                            return false;
                        node.bias.assign(v.begin(), v.end());
                        if (!ConstantValues(a[8], v))
                            return false;
                        node.inputZp = int32_t(SignExtend(uint64_t(v[0]), ElementWidth(constants[a[8]].typeId)));
                        if (!ConstantValues(a[9], v))
                            return false;
                        node.weightZp = int32_t(v[0]);

                        const Value& in = graph.values[input];
                        if (wShape[3] != in.c)
                            return Fail("CONV2D: weight IC does not match input channels");
                        outC = int32_t(wShape[0]);
                        outH = (in.h - 1 + node.pad[0] + node.pad[1] - (node.kernelH - 1) * node.dilation[0]) / node.stride[0] + 1;
                        outW = (in.w - 1 + node.pad[2] + node.pad[3] - (node.kernelW - 1) * node.dilation[1]) / node.stride[1] + 1;
                        if (node.bias.size() == 1 && outC > 1)
                            node.bias.assign(size_t(outC), node.bias[0]);
                        if (int32_t(node.bias.size()) != outC || int32_t(node.weights.size()) != outC * node.kernelH * node.kernelW * in.c)
                            return Fail("CONV2D: weight/bias size");
                        outBits = 32;
                        break;
                    }
                    case OP_RESCALE:
                    {
                        if (n < 10)
                            return Fail("RESCALE: operand count");
                        node.scale32     = ScalarValue(a[0]) != 0;
                        node.doubleRound = ScalarValue(a[1]) == 3;  // SINGLE_ROUND = 1, INEXACT_ROUND = 2, DOUBLE_ROUND = 3
                        node.perChannel  = ScalarValue(a[2]) != 0;
                        if (ScalarValue(a[3]) != 0 || ScalarValue(a[4]) != 0)
                            return Fail("RESCALE: unsigned inputs/outputs are not supported");
                        uint32_t input;
                        if (!Operand(a[5], input))
                            return false;
                        node.inputs.push_back(input);
                        if (!ConstantValues(a[6], v))
                            return false;
                        node.multiplier.assign(v.begin(), v.end());
                        if (!ConstantValues(a[7], v))
                            return false;
                        node.shift.assign(v.begin(), v.end());
                        if (!ConstantValues(a[8], v))
                            return false;
                        node.rescaleInZp = int32_t(v[0]);
                        if (!ConstantValues(a[9], v))
                            return false;
                        node.rescaleOutZp = int32_t(SignExtend(uint64_t(v[0]), ElementWidth(constants[a[9]].typeId)));
                        const Value& in = graph.values[input];
                        outH = in.h, outW = in.w, outC = in.c;
                        for (int32_t& s : node.shift)
                            s &= 0xFF;
                        break;
                    }
                    case OP_TABLE:
                    {
                        uint32_t input;
                        if (n < 2 || !Operand(a[0], input))
                            return Fail("TABLE: input");
                        node.inputs.push_back(input);
                        if (!ConstantValues(a[1], v) || v.size() != 256)
                            return Fail("TABLE: only 256 entry int8 tables are supported");
                        node.table.assign(v.begin(), v.end());
                        const Value& in = graph.values[input];
                        if (in.elementBits != 8)
                            return Fail("TABLE: only int8 inputs are supported");
                        outH = in.h, outW = in.w, outC = in.c;
                        break;
                    }
                    case OP_CONCAT:
                    {
                        node.axis = uint32_t(ScalarValue(a[0]));
                        if (node.axis != 3)
                            return Fail("CONCAT: only channel concatenation is supported");
                        for (uint32_t i = 1; i < n; ++i)
                        {
                            uint32_t input;
                            if (!Operand(a[i], input))
                                return false;
                            node.inputs.push_back(input);
                            const Value& in = graph.values[input];
                            outH            = in.h;
                            outW            = in.w;
                            outC += in.c;
                        }
                        break;
                    }
                    case OP_RESIZE:
                    {
                        if (n < 5)
                            return Fail("RESIZE: operand count");
                        node.resizeMode = uint32_t(ScalarValue(a[0]));
                        uint32_t input;
                        if (!Operand(a[1], input))
                            return false;
                        node.inputs.push_back(input);
                        if (!ConstantValues(a[2], v) || v.size() != 4)
                            return Fail("RESIZE: scale");
                        node.scale = {int32_t(v[0]), int32_t(v[1]), int32_t(v[2]), int32_t(v[3])};
                        if (!ConstantValues(a[3], v) || v.size() != 2)
                            return Fail("RESIZE: offset");
                        node.offset = {int32_t(v[0]), int32_t(v[1])};
                        if (!ConstantValues(a[4], v) || v.size() != 2)
                            return Fail("RESIZE: border");
                        node.border = {int32_t(v[0]), int32_t(v[1])};
                        if (node.resizeMode != 1)
                            return Fail("RESIZE: only NEAREST_NEIGHBOR is supported for int8 graphs");
                        const Value& in = graph.values[input];
                        outH = ((in.h - 1) * node.scale[0] - node.offset[0] + node.border[0]) / node.scale[1] + 1;
                        outW = ((in.w - 1) * node.scale[2] - node.offset[1] + node.border[1]) / node.scale[3] + 1;
                        outC = in.c;
                        break;
                    }
                    default:
                        return Fail("unsupported TOSA operator " + std::to_string(op));
                    }

                    node.result = AddValue({1, outH, outW, outC}, outBits, Value::OpResult, uint32_t(graph.nodes.size()));
                    idToValue[resultId] = node.result;
                    graph.nodes.push_back(std::move(node));
                    return true;
                }
            };
        }  // namespace

        bool Parse(const GraphDesc& desc, Graph& out, std::string* error)
        {
            out = Graph();
            Parser parser{desc, error, {}, {}, {}, {}, {}, 0, out};
            if (!parser.Run())
                return false;
            if (out.outputValues.empty())
            {
                if (error)
                    *error = "graph has no outputs";
                return false;
            }
            return true;
        }
    }  // namespace tosa

    namespace
    {
        bool LowerFail(std::string* error, const std::string& message)
        {
            if (error)
                *error = message;
            return false;
        }
    }  // namespace

    bool Lower(const tosa::Graph& graph, LoweredGraph& out, std::string* error)
    {
        using namespace tosa;
        out = LoweredGraph();

        // Count uses so fusion only happens when an intermediate has a single consumer.
        std::vector<uint32_t> uses(graph.values.size(), 0);
        for (const Node& node : graph.nodes)
            for (uint32_t in : node.inputs)
                uses[in]++;
        for (uint32_t v : graph.outputValues)
            uses[v]++;

        std::map<uint32_t, uint32_t> outputBindingOf;  // value -> binding
        for (size_t i = 0; i < graph.outputValues.size(); ++i)
            outputBindingOf[graph.outputValues[i]] = graph.outputBindings[i];

        std::vector<int32_t> storedOf(graph.values.size(), -1);  // value -> StoredTensor

        // Graph inputs are external stored tensors.
        for (uint32_t vi = 0; vi < graph.values.size(); ++vi)
        {
            const Value& v = graph.values[vi];
            if (v.kind != Value::Input)
                continue;
            if (v.c % 4 != 0)
                return LowerFail(error, "graph input channels must be a multiple of 4");
            StoredTensor t;
            t.h = v.h, t.w = v.w, t.c = v.c, t.channelStride = v.c;
            t.external = true;
            t.binding  = graph.inputBindings[v.index];
            storedOf[vi] = int32_t(out.tensors.size());
            out.tensors.push_back(t);
        }

        auto producer = [&](uint32_t value) -> const Node* {
            const Value& v = graph.values[value];
            return v.kind == Value::OpResult ? &graph.nodes[v.index] : nullptr;
        };

        // Resolve a (possibly virtual) value into conv sources.
        std::function<bool(uint32_t, std::vector<ConvSource>&)> resolve = [&](uint32_t value, std::vector<ConvSource>& sources) -> bool {
            if (storedOf[value] >= 0)
            {
                ConvSource s;
                s.tensor = uint32_t(storedOf[value]);
                sources.push_back(s);
                return true;
            }
            const Node* node = producer(value);
            if (!node)
                return LowerFail(error, "unresolved tensor");
            if (node->op == OP_CONCAT)
            {
                for (uint32_t in : node->inputs)
                    if (!resolve(in, sources))
                        return false;
                return true;
            }
            if (node->op == OP_RESIZE)
            {
                std::vector<ConvSource> inner;
                if (!resolve(node->inputs[0], inner))
                    return false;
                if (inner.size() != 1 || inner[0].resize)
                    return LowerFail(error, "RESIZE of a concatenation or of a resized tensor is not supported");
                inner[0].resize = true;
                inner[0].scale  = node->scale;
                inner[0].offset = node->offset;
                sources.push_back(inner[0]);
                return true;
            }
            return LowerFail(error, "tensor produced by an unfusable operator");
        };

        for (const Node& node : graph.nodes)
        {
            if (node.op != OP_CONV2D)
                continue;

            FusedConv fc;
            fc.conv = &node;
            if (!resolve(node.inputs[0], fc.sources))
                return false;
            if (fc.sources.size() > 2)
                return LowerFail(error, "convolutions over more than two concatenated tensors are not supported");
            for (const ConvSource& s : fc.sources)
            {
                if (out.tensors[s.tensor].c % 4 != 0)
                    return LowerFail(error, "conv source channels must be a multiple of 4");
            }
            if (node.weightZp != 0)
                return LowerFail(error, "non-zero weight zero point is not supported");

            const Value& in = graph.values[node.inputs[0]];
            fc.inH = in.h, fc.inW = in.w, fc.inC = in.c;

            // The conv result must feed exactly one RESCALE, optionally followed by a single TABLE.
            const Node* rescale = nullptr;
            for (const Node& candidate : graph.nodes)
            {
                if (candidate.op == OP_RESCALE && candidate.inputs[0] == node.result)
                    rescale = &candidate;
            }
            if (!rescale || uses[node.result] != 1)
                return LowerFail(error, "CONV2D must be followed by a single RESCALE");
            fc.rescale = rescale;

            uint32_t finalValue = rescale->result;
            for (const Node& candidate : graph.nodes)
            {
                if (candidate.op == OP_TABLE && candidate.inputs[0] == rescale->result)
                {
                    if (uses[rescale->result] != 1)
                        return LowerFail(error, "RESCALE result consumed by TABLE and other operators");
                    fc.table   = &candidate;
                    finalValue = candidate.result;
                }
            }

            const Value& result = graph.values[finalValue];
            fc.outH = result.h, fc.outW = result.w, fc.outC = result.c;

            StoredTensor t;
            t.h = result.h, t.w = result.w, t.c = result.c;
            auto ext = outputBindingOf.find(finalValue);
            if (ext != outputBindingOf.end())
            {
                if (result.c % 4 != 0)
                    return LowerFail(error, "graph output channels must be a multiple of 4");
                t.external      = true;
                t.binding       = ext->second;
                t.channelStride = result.c;
            }
            else
            {
                t.channelStride = (result.c + 3) & ~3;
                t.internalIndex = out.internalTensorCount++;
            }
            fc.output             = uint32_t(out.tensors.size());
            storedOf[finalValue]  = int32_t(out.tensors.size());
            out.tensors.push_back(t);
            out.convs.push_back(fc);
        }

        for (uint32_t v : graph.outputValues)
        {
            if (storedOf[v] < 0)
                return LowerFail(error, "graph output is not produced by a fused convolution");
        }
        return true;
    }
}  // namespace ffx_ml
