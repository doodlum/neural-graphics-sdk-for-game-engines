// Input passes recorded ahead of FSR: the proxy's own compute dispatch on the game's VkDevice, writing a proxy-owned
// image that FSR then reads in place of a game input.
//  - DilateMv (mv_dilate.comp): motion vector dilation into RG32F, read as FSR's motion vectors.
//  - SanitizeColor (color_sanitize.comp): colour with Inf / NaN made finite into RGBA16F, read as FSR's colour.
// SPDX-License-Identifier: MIT
#pragma once

#include "nss_upscaler.h"

#include <memory>

class MvDilator
{
public:
    enum class Pass
    {
        DilateMv,
        SanitizeColor,
    };
    // Null when the pipeline or the image cannot be created (the caller then passes the game's input).
    // SanitizeColor writes `colorFormat` (RGBA16F, RGBA32F or B10G11R11, see SanitizeFormat).
    static std::unique_ptr<MvDilator> Create(const VulkanDevice& dev, uint32_t maxWidth, uint32_t maxHeight,
                                             Pass pass = Pass::DilateMv, VkFormat colorFormat = VK_FORMAT_UNDEFINED);
    // The format SanitizeColor writes for game colour in `f`, or VK_FORMAT_UNDEFINED when `f` cannot hold Inf or
    // NaN (UNORM / sRGB: no clean-up needed).
    static VkFormat SanitizeFormat(VkFormat f);
    ~MvDilator();

    // Records the pass into the owned image and returns that image, in GENERAL layout and visible to compute reads, as
    // FSR's input. DilateMv: `motion` dilated by `depth`; SanitizeColor: `motion` is the colour, `depth` is ignored
    // (pass the colour for both). Inputs at the render size, in the layouts given.
    UpscalerImage Record(VkCommandBuffer cmd, const UpscalerImage& depth, const UpscalerImage& motion, uint32_t width,
                         uint32_t height, bool depthInverted);

private:
    MvDilator() = default;
    struct Fn;
    std::unique_ptr<Fn> m_fn;
    VulkanDevice        m_dev;
    VkSampler             m_sampler    = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout  = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipeLayout = VK_NULL_HANDLE;
    VkPipeline            m_pipeline   = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool       = VK_NULL_HANDLE;
    VkImage               m_image      = VK_NULL_HANDLE;
    VkDeviceMemory        m_memory     = VK_NULL_HANDLE;
    VkImageView           m_view       = VK_NULL_HANDLE;
    VkFormat              m_format = VK_FORMAT_R32G32_SFLOAT;
    uint32_t              m_width = 0, m_height = 0;
    bool                  m_initialised = false;  // m_image has left VK_IMAGE_LAYOUT_UNDEFINED
    // Per-dispatch views of the game's depth and motion vectors, recycled once the ring wraps (frames in flight are far
    // fewer than kRing).
    static constexpr uint32_t kRing = 8;
    VkDescriptorSet m_sets[kRing]       = {};
    VkImageView     m_views[kRing * 2]  = {};
    uint64_t        m_count             = 0;
};
