// Vulkan-level NSS upscaler shared by the D3D12 (vkd3d-proton interop) and Vulkan NGX entry points.
// SPDX-License-Identifier: MIT
#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

struct VulkanDevice
{
    VkInstance                instance       = VK_NULL_HANDLE;
    VkPhysicalDevice          physicalDevice = VK_NULL_HANDLE;
    VkDevice                  device         = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr gipa           = nullptr;
    PFN_vkGetDeviceProcAddr   gdpa           = nullptr;
    uint32_t                  mlFeatures     = 0;  // FFX_API_VK_ML_FEATURE_* enabled on this device
};

// One image handed in by the game, with the layout it is in when the upscaler runs.
struct UpscalerImage
{
    VkImage       image  = VK_NULL_HANDLE;
    VkFormat      format = VK_FORMAT_UNDEFINED;
    uint32_t      width  = 0;
    uint32_t      height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
};

struct UpscalerCreateInfo
{
    uint32_t renderWidth   = 0;  // maximum render size
    uint32_t renderHeight  = 0;
    uint32_t displayWidth  = 0;
    uint32_t displayHeight = 0;
    bool     hdr           = false;
    bool     depthInverted = false;
    int      nssQuality    = 2;  // FfxApiNssShaderQualityMode
};

struct UpscalerDispatch
{
    UpscalerImage color, depth, motionVectors, output;
    uint32_t      renderWidth = 0, renderHeight = 0;
    int32_t       depthOffsetX = 0, depthOffsetY = 0;
    float         jitterX = 0, jitterY = 0;      // DLSS convention, render pixels
    float         mvScaleX = 1, mvScaleY = 1;
    float         frameTimeMs = 16.6f;
    float         exposure = 1.0f;
    bool          reset = false;
};

class NssUpscaler
{
public:
    static std::unique_ptr<NssUpscaler> Create(const VulkanDevice& device, const UpscalerCreateInfo& info);
    ~NssUpscaler();

    bool Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d);

    const UpscalerCreateInfo& Info() const { return m_info; }

private:
    NssUpscaler() = default;
    bool Init(const VulkanDevice& device, const UpscalerCreateInfo& info);
    bool CreateDepthCopy();
    VkImageView CreateView(VkImage image, VkFormat format, VkImageAspectFlags aspect);
    uint32_t    FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags);
    void        RecordDump(VkCommandBuffer cmd, const UpscalerImage& output);
    void        WriteDump();

    struct Fn;
    std::unique_ptr<Fn> m_fn;

    VulkanDevice       m_dev;
    UpscalerCreateInfo m_info;
    void*              m_context = nullptr;  // ffxContext

    // R32F copy of the game's depth: NSS reads depth through a colour view.
    VkImage               m_depthImage  = VK_NULL_HANDLE;
    VkDeviceMemory        m_depthMemory = VK_NULL_HANDLE;
    VkImageView           m_depthView   = VK_NULL_HANDLE;
    VkSampler             m_sampler     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout   = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipeLayout  = VK_NULL_HANDLE;
    VkPipeline            m_pipeline    = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool        = VK_NULL_HANDLE;

    // Per-dispatch views of game images, recycled once the ring wraps (frames in flight are far fewer).
    static constexpr uint32_t kRing = 8;
    VkDescriptorSet m_sets[kRing]       = {};
    VkImageView     m_gameViews[kRing]  = {};
    uint64_t        m_dispatchCount     = 0;

    // AISS_DUMP_FRAME readback of one output frame.
    VkBuffer       m_dumpBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_dumpMemory = VK_NULL_HANDLE;
    UpscalerImage  m_dumpImage;
    uint64_t       m_dumpAt     = 0;
};

// NGX handles map to upscalers; released ones are destroyed once the GPU can no longer use them.
class UpscalerRegistry
{
public:
    unsigned int Add(std::unique_ptr<NssUpscaler> upscaler);
    NssUpscaler* Get(unsigned int id);
    void         Release(unsigned int id);
    void         CollectGarbage(bool force = false);

private:
    struct Entry
    {
        unsigned int                          id;
        std::unique_ptr<NssUpscaler>          upscaler;
        bool                                  released = false;
        std::chrono::steady_clock::time_point releasedAt;
    };
    std::vector<Entry> m_entries;
    unsigned int       m_nextId = 1;
};

UpscalerRegistry& Registry();

// Maps the game's DLSS quality mode to an NSS shader quality, honouring AISS_NSS_QUALITY.
int NssQualityForDlss(int perfQualityValue);
