// SPIR-V rewrite: images with a 16-bit float sampled type -> 32-bit (see spirv_half_images.cpp).
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct HalfImageStats
{
    uint32_t images = 0, reads = 0, writes = 0, unhandled = 0;
};

// The rewritten module, or an empty vector when it has no such images (or cannot be parsed).
std::vector<uint32_t> PromoteHalfImages(const uint32_t* code, size_t wordCount, HalfImageStats* stats = nullptr);
