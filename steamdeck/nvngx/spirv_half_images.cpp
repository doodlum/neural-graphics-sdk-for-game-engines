// SPIR-V rewrite for AMD's FSR 4 shaders on Mesa (RADV).
//
// The shaders amdvkffx64.dll builds declare images whose sampled type is a 16-bit float (texel reads return half
// vectors, writes take them). Only AMD's own driver accepts that without SPV_AMD_gpu_shader_half_float_fetch; Vulkan
// requires 32-bit sampled types, and RADV crashes compiling the pipeline. The rewrite gives those images a 32-bit
// float sampled type, makes the texel instructions produce / consume 32-bit values, and converts with OpFConvert
// right after a read and right before a write, so the rest of the shader still computes in half precision.
//
// Written from the SPIR-V specification (unified1); no third-party code.
// SPDX-License-Identifier: MIT
#include "spirv_half_images.h"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace
{
enum : uint32_t
{
    kMagic             = 0x07230203u,
    kOpSource          = 3,
    kOpSourceExtension = 4,
    kOpName            = 5,
    kOpMemberName      = 6,
    kOpExtension       = 10,
    kOpMemoryModel     = 14,
    kOpCapability      = 17,
    kOpTypeFloat       = 22,
    kOpTypeVector      = 23,
    kOpTypeImage       = 25,
    kOpTypeSampledImage = 27,
    kOpTypeArray       = 28,
    kOpTypeRuntimeArray = 29,
    kOpTypeStruct      = 30,
    kOpTypePointer     = 32,
    kOpTypeFunction    = 33,
    kOpFunction        = 54,
    kOpImageSampleImplicitLod = 87,  // ... through OpImageRead (98): texel-producing image instructions
    kOpImageRead       = 98,
    kOpImageWrite      = 99,
    kOpFConvert        = 115,
    kOpImageSparseFirst = 305,
    kOpImageSparseLast  = 315,
};

struct Instruction
{
    uint32_t op;
    std::vector<uint32_t> words;  // the whole instruction, word 0 included
};
}  // namespace

std::vector<uint32_t> PromoteHalfImages(const uint32_t* code, size_t wordCount, HalfImageStats* stats)
{
    if (stats)
        *stats = {};
    if (!code || wordCount < 5 || code[0] != kMagic)
        return {};

    std::vector<Instruction> module;
    for (size_t at = 5; at < wordCount;)
    {
        const uint32_t len = code[at] >> 16;
        if (!len || at + len > wordCount)
            return {};
        module.push_back({code[at] & 0xFFFFu, std::vector<uint32_t>(code + at, code + at + len)});
        at += len;
    }

    // Float types and vectors of them.
    std::set<uint32_t>      halfScalars;
    uint32_t                f32 = 0;
    std::map<uint32_t, uint32_t> halfVectorCount;          // half vector id -> component count
    std::map<uint32_t, uint32_t> floatVectorByCount;       // component count -> float vector id
    for (const Instruction& in : module)
    {
        if (in.op == kOpTypeFloat && in.words.size() >= 3)
        {
            if (in.words[2] == 16)
                halfScalars.insert(in.words[1]);
            else if (in.words[2] == 32 && in.words.size() == 3 && !f32)
                f32 = in.words[1];
        }
    }
    if (halfScalars.empty())
        return {};
    for (const Instruction& in : module)
        if (in.op == kOpTypeVector && in.words.size() == 4)
        {
            if (halfScalars.count(in.words[2]))
                halfVectorCount[in.words[1]] = in.words[3];
            else if (in.words[2] == f32 && f32 && !floatVectorByCount.count(in.words[3]))
                floatVectorByCount[in.words[3]] = in.words[1];
        }

    bool anyHalfImage = false;
    for (const Instruction& in : module)
        if (in.op == kOpTypeImage && in.words.size() >= 9 && halfScalars.count(in.words[2]))
            anyHalfImage = true;
    if (!anyHalfImage)
        return {};

    uint32_t bound = code[3];
    std::vector<Instruction> newTypes;  // declarations to insert (after the float types they use)
    if (!f32)
    {
        f32 = bound++;
        newTypes.push_back({kOpTypeFloat, {(3u << 16) | kOpTypeFloat, f32, 32}});
    }
    std::vector<Instruction> newVectors;
    // The 32-bit type a half scalar / vector type becomes.
    std::unordered_map<uint32_t, uint32_t> promoted;
    for (uint32_t h : halfScalars)
        promoted[h] = f32;
    for (const auto& [vec, count] : halfVectorCount)
    {
        auto found = floatVectorByCount.find(count);
        if (found == floatVectorByCount.end())
        {
            const uint32_t id = bound++;
            newVectors.push_back({kOpTypeVector, {(4u << 16) | kOpTypeVector, id, f32, count}});
            found = floatVectorByCount.emplace(count, id).first;
        }
        promoted[vec] = found->second;
    }

    // Value id -> type, for the texels handed to OpImageWrite (any instruction whose second word is the value id and
    // whose first word is a half type defines it: result ids are unique).
    std::unordered_map<uint32_t, uint32_t> halfValueType;
    for (const Instruction& in : module)
        if (in.words.size() >= 3 && promoted.count(in.words[1]) && in.op != kOpTypeVector && in.op != kOpName &&
            in.op != kOpMemberName)
            halfValueType[in.words[2]] = in.words[1];

    // Give half images a 32-bit sampled type; an image (or sampled image) type that then duplicates an earlier one is
    // dropped and its id mapped to the earlier one, as SPIR-V forbids duplicate non-aggregate types.
    std::unordered_map<uint32_t, uint32_t> alias;
    auto resolve = [&](uint32_t& id) {
        auto a = alias.find(id);
        if (a != alias.end())
            id = a->second;
    };
    std::vector<bool>              drop(module.size(), false);
    std::vector<const Instruction*> uniqueTypes;
    for (size_t i = 0; i < module.size(); ++i)
    {
        Instruction& in = module[i];
        if (in.op == kOpTypeImage && in.words.size() >= 9)
        {
            if (halfScalars.count(in.words[2]))
            {
                in.words[2] = f32;
                if (stats)
                    ++stats->images;
            }
        }
        else if (in.op == kOpTypeSampledImage && in.words.size() == 3)
            resolve(in.words[2]);
        else
            continue;
        const Instruction* same = nullptr;
        for (const Instruction* u : uniqueTypes)
            if (u->words.size() == in.words.size() && u->words[0] == in.words[0] &&
                std::equal(u->words.begin() + 2, u->words.end(), in.words.begin() + 2))
                same = u;
        if (same)
        {
            alias[in.words[1]] = same->words[1];
            drop[i]            = true;
        }
        else
            uniqueTypes.push_back(&in);
    }
    // Uses of dropped type ids: result types, type operands of other types, names and decorations.
    if (!alias.empty())
        for (size_t i = 0; i < module.size(); ++i)
        {
            Instruction& in = module[i];
            if (drop[i] || in.words.size() < 2)
                continue;
            switch (in.op)
            {
            case kOpSource:
            case kOpSourceExtension:
            case kOpExtension:
            case kOpMemoryModel:
            case kOpCapability:
                break;  // literal operands
            case kOpTypePointer:
                if (in.words.size() == 4)
                    resolve(in.words[3]);
                break;
            case kOpTypeArray:
            case kOpTypeRuntimeArray:
                resolve(in.words[2]);
                break;
            case kOpTypeStruct:
            case kOpTypeFunction:
                for (size_t w = 2; w < in.words.size(); ++w)
                    resolve(in.words[w]);
                break;
            default:
                resolve(in.words[1]);  // result type, or the target of OpName / OpDecorate
                break;
            }
        }

    // Texel instructions on the promoted images: produce 32-bit, convert back to the half type the shader uses; writes
    // convert their half texel to 32-bit first.
    std::vector<Instruction> out;
    out.reserve(module.size() + 64);
    bool inserted = false;
    for (size_t i = 0; i < module.size(); ++i)
    {
        if (drop[i])
            continue;
        Instruction& in = module[i];
        if (!inserted && in.op == kOpFunction)
        {
            // New types go before the first function; they are only used inside functions (the 32-bit float type,
            // when it had to be added, is placed before the image types instead, below).
            for (const Instruction& v : newVectors)
                out.push_back(v);
            inserted = true;
        }
        if (in.op >= kOpImageSampleImplicitLod && in.op <= kOpImageRead && in.words.size() >= 4 && promoted.count(in.words[1]))
        {
            const uint32_t halfType = in.words[1], result = in.words[2], temp = bound++;
            in.words[1] = promoted[halfType];
            in.words[2] = temp;
            out.push_back(in);
            out.push_back({kOpFConvert, {(4u << 16) | kOpFConvert, halfType, result, temp}});
            if (stats)
                ++stats->reads;
            continue;
        }
        if (in.op == kOpImageWrite && in.words.size() >= 4)
        {
            auto t = halfValueType.find(in.words[3]);
            if (t != halfValueType.end())
            {
                const uint32_t temp = bound++;
                out.push_back({kOpFConvert, {(4u << 16) | kOpFConvert, promoted[t->second], temp, in.words[3]}});
                in.words[3] = temp;
                if (stats)
                    ++stats->writes;
            }
        }
        if (in.op >= kOpImageSparseFirst && in.op <= kOpImageSparseLast && stats)
            ++stats->unhandled;
        out.push_back(in);
        if (in.op == kOpTypeFloat && !newTypes.empty() && in.words.size() >= 3 && halfScalars.count(in.words[1]))
        {
            for (const Instruction& t : newTypes)
                out.push_back(t);
            newTypes.clear();
        }
    }
    if (!inserted || !newTypes.empty())
        return {};  // no function, or nowhere to put a needed type: leave the module alone

    std::vector<uint32_t> result(code, code + 5);
    result[3] = bound;
    for (const Instruction& in : out)
        result.insert(result.end(), in.words.begin(), in.words.end());
    return result;
}
