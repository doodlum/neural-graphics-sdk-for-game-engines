// Input passes ahead of FSR: motion vector dilation and colour clean-up (see mv_dilate.h).
// SPDX-License-Identifier: MIT
#include "mv_dilate.h"

#include "common.h"

#include <algorithm>

namespace
{
const uint32_t kDilateSpirv[] = {
#include "mv_dilate.spv.inc"
};
const uint32_t kSanitizeSpirv[] = {
#include "color_sanitize.spv.inc"
};

bool IsDepthFormat(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}
}  // namespace

struct MvDilator::Fn
{
#define FN(name) PFN_##name name = nullptr;
    FN(vkCreateSampler) FN(vkDestroySampler) FN(vkCreateDescriptorSetLayout) FN(vkDestroyDescriptorSetLayout)
    FN(vkCreatePipelineLayout) FN(vkDestroyPipelineLayout) FN(vkCreateShaderModule) FN(vkDestroyShaderModule)
    FN(vkCreateComputePipelines) FN(vkDestroyPipeline) FN(vkCreateDescriptorPool) FN(vkDestroyDescriptorPool)
    FN(vkAllocateDescriptorSets) FN(vkUpdateDescriptorSets) FN(vkCreateImage) FN(vkDestroyImage)
    FN(vkGetImageMemoryRequirements) FN(vkAllocateMemory) FN(vkFreeMemory) FN(vkBindImageMemory)
    FN(vkCreateImageView) FN(vkDestroyImageView) FN(vkCmdBindPipeline) FN(vkCmdBindDescriptorSets)
    FN(vkCmdPushConstants) FN(vkCmdDispatch) FN(vkCmdPipelineBarrier) FN(vkGetPhysicalDeviceMemoryProperties)
#undef FN
};

std::unique_ptr<MvDilator> MvDilator::Create(const VulkanDevice& dev, uint32_t maxWidth, uint32_t maxHeight, Pass pass)
{
    std::unique_ptr<MvDilator> d(new MvDilator());
    d->m_dev    = dev;
    d->m_format = pass == Pass::SanitizeColor ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32_SFLOAT;
    d->m_fn.reset(new Fn());
    Fn& fn = *d->m_fn;
#define LOAD(name) fn.name = reinterpret_cast<PFN_##name>(dev.gdpa(dev.device, #name));
    LOAD(vkCreateSampler) LOAD(vkDestroySampler) LOAD(vkCreateDescriptorSetLayout) LOAD(vkDestroyDescriptorSetLayout)
    LOAD(vkCreatePipelineLayout) LOAD(vkDestroyPipelineLayout) LOAD(vkCreateShaderModule) LOAD(vkDestroyShaderModule)
    LOAD(vkCreateComputePipelines) LOAD(vkDestroyPipeline) LOAD(vkCreateDescriptorPool) LOAD(vkDestroyDescriptorPool)
    LOAD(vkAllocateDescriptorSets) LOAD(vkUpdateDescriptorSets) LOAD(vkCreateImage) LOAD(vkDestroyImage)
    LOAD(vkGetImageMemoryRequirements) LOAD(vkAllocateMemory) LOAD(vkFreeMemory) LOAD(vkBindImageMemory)
    LOAD(vkCreateImageView) LOAD(vkDestroyImageView) LOAD(vkCmdBindPipeline) LOAD(vkCmdBindDescriptorSets)
    LOAD(vkCmdPushConstants) LOAD(vkCmdDispatch) LOAD(vkCmdPipelineBarrier)
#undef LOAD
    fn.vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        dev.gipa(dev.instance, "vkGetPhysicalDeviceMemoryProperties"));
    const VkDevice device = dev.device;

    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (fn.vkCreateSampler(device, &sci, nullptr, &d->m_sampler) != VK_SUCCESS)
        return nullptr;

    VkDescriptorSetLayoutBinding b[3] = {};
    for (uint32_t i = 0; i < 3; ++i)
    {
        b[i].binding         = i;
        b[i].descriptorType  = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].pImmutableSamplers = i < 2 ? &d->m_sampler : nullptr;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount                    = 3;
    dslci.pBindings                       = b;
    if (fn.vkCreateDescriptorSetLayout(device, &dslci, nullptr, &d->m_setLayout) != VK_SUCCESS)
        return nullptr;
    VkPushConstantRange        pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pl  = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount              = 1;
    pl.pSetLayouts                 = &d->m_setLayout;
    pl.pushConstantRangeCount      = 1;
    pl.pPushConstantRanges         = &pcr;
    if (fn.vkCreatePipelineLayout(device, &pl, nullptr, &d->m_pipeLayout) != VK_SUCCESS)
        return nullptr;
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize                 = pass == Pass::SanitizeColor ? sizeof(kSanitizeSpirv) : sizeof(kDilateSpirv);
    smci.pCode                    = pass == Pass::SanitizeColor ? kSanitizeSpirv : kDilateSpirv;
    VkShaderModule module         = VK_NULL_HANDLE;
    if (fn.vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS)
        return nullptr;
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module                = module;
    cpci.stage.pName                 = "main";
    cpci.layout                      = d->m_pipeLayout;
    const VkResult pr = fn.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &d->m_pipeline);
    fn.vkDestroyShaderModule(device, module, nullptr);
    if (pr != VK_SUCCESS)
        return nullptr;

    VkDescriptorPoolSize       sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kRing}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kRing}};
    VkDescriptorPoolCreateInfo dpci     = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets                        = kRing;
    dpci.poolSizeCount                  = 2;
    dpci.pPoolSizes                     = sizes;
    if (fn.vkCreateDescriptorPool(device, &dpci, nullptr, &d->m_pool) != VK_SUCCESS)
        return nullptr;
    VkDescriptorSetLayout layouts[kRing];
    std::fill(layouts, layouts + kRing, d->m_setLayout);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool              = d->m_pool;
    dsai.descriptorSetCount          = kRing;
    dsai.pSetLayouts                 = layouts;
    if (fn.vkAllocateDescriptorSets(device, &dsai, d->m_sets) != VK_SUCCESS)
        return nullptr;

    // The output (RG32F / RGBA16F) at the maximum render size, device local, sampled by FSR and written as storage here.
    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType         = VK_IMAGE_TYPE_2D;
    ici.format            = d->m_format;
    ici.extent            = {maxWidth, maxHeight, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples                     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling                      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage                       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout               = VK_IMAGE_LAYOUT_UNDEFINED;
    if (fn.vkCreateImage(device, &ici, nullptr, &d->m_image) != VK_SUCCESS)
        return nullptr;
    VkMemoryRequirements req;
    fn.vkGetImageMemoryRequirements(device, d->m_image, &req);
    VkPhysicalDeviceMemoryProperties mp = {};
    fn.vkGetPhysicalDeviceMemoryProperties(dev.physicalDevice, &mp);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize       = req.size;
    mai.memoryTypeIndex      = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && mai.memoryTypeIndex == UINT32_MAX; ++i)
        if ((req.memoryTypeBits & (1u << i)) && mp.memoryTypes[i].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            mai.memoryTypeIndex = i;
    for (uint32_t i = 0; i < mp.memoryTypeCount && mai.memoryTypeIndex == UINT32_MAX; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            mai.memoryTypeIndex = i;
    if (mai.memoryTypeIndex == UINT32_MAX || fn.vkAllocateMemory(device, &mai, nullptr, &d->m_memory) != VK_SUCCESS ||
        fn.vkBindImageMemory(device, d->m_image, d->m_memory, 0) != VK_SUCCESS)
        return nullptr;
    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image                 = d->m_image;
    vci.viewType              = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                = d->m_format;
    vci.subresourceRange      = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (fn.vkCreateImageView(device, &vci, nullptr, &d->m_view) != VK_SUCCESS)
        return nullptr;
    d->m_width  = maxWidth;
    d->m_height = maxHeight;
    return d;
}

MvDilator::~MvDilator()
{
    if (!m_fn)
        return;
    Fn&            fn = *m_fn;
    const VkDevice dv = m_dev.device;
    for (VkImageView v : m_views)
        if (v)
            fn.vkDestroyImageView(dv, v, nullptr);
    if (m_view)
        fn.vkDestroyImageView(dv, m_view, nullptr);
    if (m_image)
        fn.vkDestroyImage(dv, m_image, nullptr);
    if (m_memory)
        fn.vkFreeMemory(dv, m_memory, nullptr);
    if (m_pool)
        fn.vkDestroyDescriptorPool(dv, m_pool, nullptr);
    if (m_pipeline)
        fn.vkDestroyPipeline(dv, m_pipeline, nullptr);
    if (m_pipeLayout)
        fn.vkDestroyPipelineLayout(dv, m_pipeLayout, nullptr);
    if (m_setLayout)
        fn.vkDestroyDescriptorSetLayout(dv, m_setLayout, nullptr);
    if (m_sampler)
        fn.vkDestroySampler(dv, m_sampler, nullptr);
}

UpscalerImage MvDilator::Record(VkCommandBuffer cmd, const UpscalerImage& depth, const UpscalerImage& motion,
                                uint32_t width, uint32_t height, bool depthInverted)
{
    Fn&            fn   = *m_fn;
    const VkDevice dv   = m_dev.device;
    const uint32_t slot = uint32_t(m_count++ % kRing);
    width               = std::min(width, m_width);
    height              = std::min(height, m_height);

    // Views of this dispatch's game images (the slot's previous views are kRing dispatches old: retired).
    VkImageView* views = &m_views[slot * 2];
    for (int i = 0; i < 2; ++i)
        if (views[i])
            fn.vkDestroyImageView(dv, views[i], nullptr), views[i] = VK_NULL_HANDLE;
    const UpscalerImage* src[2] = {&depth, &motion};
    for (int i = 0; i < 2; ++i)
    {
        VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image                 = src[i]->image;
        vci.viewType              = VK_IMAGE_VIEW_TYPE_2D;
        vci.format                = src[i]->format;
        vci.subresourceRange      = {IsDepthFormat(src[i]->format) ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT)
                                                                   : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT),
                                     0, 1, 0, 1};
        if (fn.vkCreateImageView(dv, &vci, nullptr, &views[i]) != VK_SUCCESS)
            return motion;  // pass the game's input through
    }
    VkDescriptorImageInfo infos[3] = {{VK_NULL_HANDLE, views[0], depth.layout},
                                      {VK_NULL_HANDLE, views[1], motion.layout},
                                      {VK_NULL_HANDLE, m_view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet  writes[3] = {};
    for (uint32_t i = 0; i < 3; ++i)
    {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = m_sets[slot];
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo      = &infos[i];
    }
    fn.vkUpdateDescriptorSets(dv, 3, writes, 0, nullptr);

    // The owned image: previous FSR reads done before this write; first use leaves UNDEFINED.
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask        = VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask        = VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout            = m_initialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image                = m_image;
    b.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                            0, nullptr, 1, &b);
    m_initialised = true;

    const int32_t push[4] = {int32_t(width), int32_t(height), depthInverted ? 1 : 0, 0};
    fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    fn.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeLayout, 0, 1, &m_sets[slot], 0, nullptr);
    fn.vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    fn.vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                            0, nullptr, 1, &b);

    UpscalerImage out;
    out.image  = m_image;
    out.format = m_format;
    out.width  = m_width;
    out.height = m_height;
    out.layout = VK_IMAGE_LAYOUT_GENERAL;
    return out;
}
