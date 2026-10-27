// SPDX-License-Identifier: MIT
#include "nss_upscaler.h"

#include "common.h"

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_nss.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

static const uint32_t kDepthCopySpirv[] = {
#include "depth_copy.spv.inc"
};

struct NssUpscaler::Fn
{
#define VK_FN(name) PFN_##name name = nullptr;
    VK_FN(vkGetPhysicalDeviceMemoryProperties)
    VK_FN(vkCreateImage)
    VK_FN(vkDestroyImage)
    VK_FN(vkGetImageMemoryRequirements)
    VK_FN(vkAllocateMemory)
    VK_FN(vkFreeMemory)
    VK_FN(vkBindImageMemory)
    VK_FN(vkCreateImageView)
    VK_FN(vkDestroyImageView)
    VK_FN(vkCreateSampler)
    VK_FN(vkDestroySampler)
    VK_FN(vkCreateDescriptorSetLayout)
    VK_FN(vkDestroyDescriptorSetLayout)
    VK_FN(vkCreatePipelineLayout)
    VK_FN(vkDestroyPipelineLayout)
    VK_FN(vkCreateShaderModule)
    VK_FN(vkDestroyShaderModule)
    VK_FN(vkCreateComputePipelines)
    VK_FN(vkDestroyPipeline)
    VK_FN(vkCreateDescriptorPool)
    VK_FN(vkDestroyDescriptorPool)
    VK_FN(vkAllocateDescriptorSets)
    VK_FN(vkUpdateDescriptorSets)
    VK_FN(vkCmdPipelineBarrier)
    VK_FN(vkCmdBindPipeline)
    VK_FN(vkCmdBindDescriptorSets)
    VK_FN(vkCmdPushConstants)
    VK_FN(vkCmdDispatch)
    VK_FN(vkCreateBuffer)
    VK_FN(vkDestroyBuffer)
    VK_FN(vkGetBufferMemoryRequirements)
    VK_FN(vkBindBufferMemory)
    VK_FN(vkMapMemory)
    VK_FN(vkUnmapMemory)
    VK_FN(vkCmdCopyImageToBuffer)
#undef VK_FN
};

static bool IsDepthFormat(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

// Layouts in which an image can be bound for sampling without a transition.
static bool IsSampledLayout(VkImageLayout l)
{
    switch (l)
    {
    case VK_IMAGE_LAYOUT_GENERAL:
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
        return true;
    default:
        return false;
    }
}

int NssQualityForDlss(int perfQualityValue)
{
    // The full (QUALITY) network by default for every DLSS mode: with the RDNA2 kernels it costs about 1.3 ms more
    // than the small one at 640x400 -> 1280x800 on a Steam Deck. The control file (plugin, at runtime) and
    // AISS_NSS_QUALITY (launch option) override it.
    (void)perfQualityValue;
    const int live = LiveNssQuality();
    if (live >= 0)
        return live;
    if (Config().nssQuality >= 0 && Config().nssQuality <= 2)
        return Config().nssQuality;
    return FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY;
}

bool FastMathEnabled()
{
    const int live = LiveFastMath();
    return live >= 0 ? live != 0 : Config().fastMath;
}

std::unique_ptr<NssUpscaler> NssUpscaler::Create(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    std::unique_ptr<NssUpscaler> up(new NssUpscaler());
    if (!up->Init(device, info))
        return nullptr;
    return up;
}

bool NssUpscaler::Init(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    m_dev  = device;
    m_info = info;
    m_fn   = std::make_unique<Fn>();

#define VK_FN(name) m_fn->name = reinterpret_cast<PFN_##name>(device.gdpa(device.device, #name));
    VK_FN(vkCreateImage)
    VK_FN(vkDestroyImage)
    VK_FN(vkGetImageMemoryRequirements)
    VK_FN(vkAllocateMemory)
    VK_FN(vkFreeMemory)
    VK_FN(vkBindImageMemory)
    VK_FN(vkCreateImageView)
    VK_FN(vkDestroyImageView)
    VK_FN(vkCreateSampler)
    VK_FN(vkDestroySampler)
    VK_FN(vkCreateDescriptorSetLayout)
    VK_FN(vkDestroyDescriptorSetLayout)
    VK_FN(vkCreatePipelineLayout)
    VK_FN(vkDestroyPipelineLayout)
    VK_FN(vkCreateShaderModule)
    VK_FN(vkDestroyShaderModule)
    VK_FN(vkCreateComputePipelines)
    VK_FN(vkDestroyPipeline)
    VK_FN(vkCreateDescriptorPool)
    VK_FN(vkDestroyDescriptorPool)
    VK_FN(vkAllocateDescriptorSets)
    VK_FN(vkUpdateDescriptorSets)
    VK_FN(vkCmdPipelineBarrier)
    VK_FN(vkCmdBindPipeline)
    VK_FN(vkCmdBindDescriptorSets)
    VK_FN(vkCmdPushConstants)
    VK_FN(vkCmdDispatch)
    VK_FN(vkCreateBuffer)
    VK_FN(vkDestroyBuffer)
    VK_FN(vkGetBufferMemoryRequirements)
    VK_FN(vkBindBufferMemory)
    VK_FN(vkMapMemory)
    VK_FN(vkUnmapMemory)
    VK_FN(vkCmdCopyImageToBuffer)
#undef VK_FN
    m_fn->vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        device.gipa(device.instance, "vkGetPhysicalDeviceMemoryProperties"));

    if (!CreateDepthCopy())
        return false;

    ffxCreateBackendVKDesc backend = {};
    backend.header.type            = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice               = device.device;
    backend.vkPhysicalDevice       = device.physicalDevice;
    backend.vkInstance             = device.instance;
    backend.vkDeviceProcAddr       = device.gdpa;
    backend.vkGetInstanceProcAddr  = device.gipa;

    ffxCreateBackendVKMlFeaturesDesc ml = {};
    ml.header.type                      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_ML_FEATURES;
    ml.enabledFeatures                  = device.mlFeatures;

    ffxApiCreateContextDescNss nss = {};
    nss.header.type                = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
    nss.flags                      = FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY |
                FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE | (info.hdr ? FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE : 0u) |
                (info.depthInverted ? FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED : 0u);
    nss.maxRenderSize  = {info.renderWidth, info.renderHeight};
    nss.maxUpscaleSize = {info.displayWidth, info.displayHeight};
    nss.qualityMode    = static_cast<FfxApiNssShaderQualityMode>(info.nssQuality);
    nss.fpMessage      = [](uint32_t type, const char* message) { LOGI("nss: [%u] %s", type, message ? message : ""); };

    nss.header.pNext     = &backend.header;
    backend.header.pNext = &ml.header;

    // The ML runtime reads FFX_ML_RESCALE_FLOAT while it builds the graph's pipelines. Config() has already captured
    // the launch value, so setting it here only selects the mode for this context.
    (void)Config();
    _putenv_s("FFX_ML_RESCALE_FLOAT", info.fastMath ? "1" : "");

    ffxContext ctx = nullptr;
    const ffxReturnCode_t rc = ffxCreateContext(&ctx, &nss.header, nullptr);
    if (rc != FFX_API_RETURN_OK)
    {
        LOGI("ffxCreateContext(NSS) failed: %u", rc);
        return false;
    }
    m_context = ctx;
    LOGI("NSS context %ux%u -> %ux%u, quality %d, hdr %d, inverted depth %d, fast maths %d, ML features 0x%x",
         info.renderWidth, info.renderHeight, info.displayWidth, info.displayHeight, info.nssQuality, info.hdr,
         info.depthInverted, info.fastMath, device.mlFeatures);
    return true;
}

bool NssUpscaler::CreateDepthCopy()
{
    VkDevice dev = m_dev.device;
    auto&    fn  = *m_fn;

    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType         = VK_IMAGE_TYPE_2D;
    ici.format            = VK_FORMAT_R32_SFLOAT;
    ici.extent            = {m_info.renderWidth, m_info.renderHeight, 1};
    ici.mipLevels         = 1;
    ici.arrayLayers       = 1;
    ici.samples           = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling            = VK_IMAGE_TILING_OPTIMAL;
    ici.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    if (fn.vkCreateImage(dev, &ici, nullptr, &m_depthImage) != VK_SUCCESS)
        return false;

    VkMemoryRequirements req;
    fn.vkGetImageMemoryRequirements(dev, m_depthImage, &req);
    const uint32_t type = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX)
        return false;
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize       = req.size;
    mai.memoryTypeIndex      = type;
    if (fn.vkAllocateMemory(dev, &mai, nullptr, &m_depthMemory) != VK_SUCCESS ||
        fn.vkBindImageMemory(dev, m_depthImage, m_depthMemory, 0) != VK_SUCCESS)
        return false;
    m_depthView = CreateView(m_depthImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    fn.vkCreateSampler(dev, &sci, nullptr, &m_sampler);

    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding            = 0;
    bindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount    = 1;
    bindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = &m_sampler;
    bindings[1].binding            = 1;
    bindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount    = 1;
    bindings[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount                    = 2;
    dslci.pBindings                       = bindings;
    fn.vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &m_setLayout);

    VkPushConstantRange pcr       = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount             = 1;
    pl.pSetLayouts                = &m_setLayout;
    pl.pushConstantRangeCount     = 1;
    pl.pPushConstantRanges        = &pcr;
    fn.vkCreatePipelineLayout(dev, &pl, nullptr, &m_pipeLayout);

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize                 = sizeof(kDepthCopySpirv);
    smci.pCode                    = kDepthCopySpirv;
    VkShaderModule module         = VK_NULL_HANDLE;
    fn.vkCreateShaderModule(dev, &smci, nullptr, &module);
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module                = module;
    cpci.stage.pName                 = "main";
    cpci.layout                      = m_pipeLayout;
    const VkResult pr                = fn.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline);
    fn.vkDestroyShaderModule(dev, module, nullptr);
    if (pr != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kRing}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kRing}};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets                    = kRing;
    dpci.poolSizeCount              = 2;
    dpci.pPoolSizes                 = sizes;
    fn.vkCreateDescriptorPool(dev, &dpci, nullptr, &m_pool);
    VkDescriptorSetLayout layouts[kRing];
    std::fill(layouts, layouts + kRing, m_setLayout);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool              = m_pool;
    dsai.descriptorSetCount          = kRing;
    dsai.pSetLayouts                 = layouts;
    return fn.vkAllocateDescriptorSets(dev, &dsai, m_sets) == VK_SUCCESS && m_depthView != VK_NULL_HANDLE;
}

uint32_t NssUpscaler::FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags)
{
    VkPhysicalDeviceMemoryProperties props;
    m_fn->vkGetPhysicalDeviceMemoryProperties(m_dev.physicalDevice, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    return UINT32_MAX;
}

VkImageView NssUpscaler::CreateView(VkImage image, VkFormat format, VkImageAspectFlags aspect)
{
    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image                 = image;
    vci.viewType              = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                = format;
    vci.subresourceRange      = {aspect, 0, 1, 0, 1};
    VkImageView view          = VK_NULL_HANDLE;
    m_fn->vkCreateImageView(m_dev.device, &vci, nullptr, &view);
    return view;
}

static FfxApiResource WrapImage(const UpscalerImage& img, uint32_t usage, uint32_t state)
{
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType         = VK_IMAGE_TYPE_2D;
    info.format            = img.format;
    info.extent            = {img.width, img.height, 1};
    info.mipLevels         = 1;
    info.arrayLayers       = 1;
    FfxApiResourceDescription desc = ffxApiGetImageResourceDescriptionVK(img.image, info, usage);
    return ffxApiGetResourceVK(reinterpret_cast<void*>(img.image), desc, state);
}

// The FFX backend transitions external images from the state it is given and back again, but only knows GENERAL and
// SHADER_READ_ONLY_OPTIMAL for read inputs. Anything else is moved to GENERAL around the dispatch.
struct LayoutFix
{
    VkImageMemoryBarrier before[4];
    VkImageMemoryBarrier after[4];
    uint32_t             count = 0;

    uint32_t Adopt(const UpscalerImage& img, bool write)
    {
        if (img.layout == VK_IMAGE_LAYOUT_GENERAL)
            return write ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_COMMON;
        if (!write && img.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            return FFX_API_RESOURCE_STATE_COMPUTE_READ;

        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask        = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask        = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.oldLayout            = img.layout;
        b.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        b.image                = img.image;
        b.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        before[count]          = b;
        std::swap(b.oldLayout, b.newLayout);
        after[count++] = b;
        return write ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_COMMON;
    }
};

bool NssUpscaler::Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d)
{
    auto& fn = *m_fn;
    const uint32_t slot = static_cast<uint32_t>(m_dispatchCount++ % kRing);

    const uint32_t renderW = std::min(d.renderWidth, m_info.renderWidth);
    const uint32_t renderH = std::min(d.renderHeight, m_info.renderHeight);

    // Make the game's writes to the inputs visible to compute.
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask   = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    // ---- depth -> R32F ----
    const bool          depthIsDepth = IsDepthFormat(d.depth.format);
    const VkImageAspectFlags aspect  = depthIsDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    const VkFormat      viewFormat   = d.depth.format;  // a combined depth-stencil format is viewed as itself, depth aspect only
    VkImageLayout depthLayout = d.depth.layout;
    VkImageMemoryBarrier depthToRead = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    const bool depthNeedsTransition  = !IsSampledLayout(depthLayout);
    if (depthNeedsTransition)
    {
        depthToRead.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT;
        depthToRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        depthToRead.oldLayout           = depthLayout;
        depthToRead.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        depthToRead.srcQueueFamilyIndex = depthToRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthToRead.image               = d.depth.image;
        depthToRead.subresourceRange    = {aspect, 0, 1, 0, 1};
        if (d.depth.format == VK_FORMAT_D24_UNORM_S8_UINT || d.depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
            d.depth.format == VK_FORMAT_D16_UNORM_S8_UINT)
            depthToRead.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        depthLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkImageMemoryBarrier toStorage = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toStorage.srcAccessMask        = VK_ACCESS_SHADER_READ_BIT;
    toStorage.dstAccessMask        = VK_ACCESS_SHADER_WRITE_BIT;
    toStorage.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
    toStorage.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
    toStorage.srcQueueFamilyIndex = toStorage.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toStorage.image                = m_depthImage;
    toStorage.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier pre[2] = {toStorage, depthToRead};
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                            nullptr, depthNeedsTransition ? 2 : 1, pre);

    if (m_gameViews[slot])
        fn.vkDestroyImageView(m_dev.device, m_gameViews[slot], nullptr);
    m_gameViews[slot] = CreateView(d.depth.image, viewFormat, aspect);

    VkDescriptorImageInfo depthInfo = {VK_NULL_HANDLE, m_gameViews[slot], depthLayout};
    VkDescriptorImageInfo outInfo   = {VK_NULL_HANDLE, m_depthView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet  writes[2] = {};
    writes[0].sType                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet                = m_sets[slot];
    writes[0].dstBinding            = 0;
    writes[0].descriptorCount       = 1;
    writes[0].descriptorType        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo            = &depthInfo;
    writes[1]                       = writes[0];
    writes[1].dstBinding            = 1;
    writes[1].descriptorType        = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo            = &outInfo;
    fn.vkUpdateDescriptorSets(m_dev.device, 2, writes, 0, nullptr);

    const int32_t push[4] = {int32_t(renderW), int32_t(renderH), d.depthOffsetX, d.depthOffsetY};
    fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    fn.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeLayout, 0, 1, &m_sets[slot], 0, nullptr);
    fn.vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    fn.vkCmdDispatch(cmd, (renderW + 7) / 8, (renderH + 7) / 8, 1);

    VkImageMemoryBarrier toRead = toStorage;
    toRead.srcAccessMask        = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask        = VK_ACCESS_SHADER_READ_BIT;
    toRead.oldLayout            = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout            = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageMemoryBarrier post[2] = {toRead, depthToRead};
    if (depthNeedsTransition)
    {
        std::swap(post[1].oldLayout, post[1].newLayout);
        post[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        post[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    }

    // ---- NSS ----
    LayoutFix fix;
    UpscalerImage depthR32 = {m_depthImage, VK_FORMAT_R32_SFLOAT, m_info.renderWidth, m_info.renderHeight,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    ffxApiDispatchDescNss nss = {};
    nss.header.type           = FFX_API_DISPATCH_DESC_TYPE_NSS;
    nss.commandList           = cmd;
    nss.color                 = WrapImage(d.color, 0, fix.Adopt(d.color, false));
    nss.motionVectors         = WrapImage(d.motionVectors, 0, fix.Adopt(d.motionVectors, false));
    nss.output                = WrapImage(d.output, FFX_API_RESOURCE_USAGE_UAV, fix.Adopt(d.output, true));
    nss.depth                 = WrapImage(depthR32, 0, FFX_API_RESOURCE_STATE_COMPUTE_READ);

    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, depthNeedsTransition ? 2 : 1, post);
    if (fix.count)
        fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                0, nullptr, fix.count, fix.before);

    // NSS's jitter is the negative of the DLSS pixel offset on both axes.
    nss.jitterOffset           = {-d.jitterX * Config().jitterSign[0], -d.jitterY * Config().jitterSign[1]};
    nss.motionVectorScale      = {d.mvScaleX * Config().mvSign[0], d.mvScaleY * Config().mvSign[1]};
    nss.renderSize             = {renderW, renderH};
    nss.upscaleSize            = {m_info.displayWidth, m_info.displayHeight};
    nss.cameraNear             = Config().cameraNear;
    nss.cameraFar              = Config().cameraFar;
    nss.cameraFovAngleVertical = Config().cameraFovY;
    nss.exposure               = d.exposure;
    nss.frameTimeDelta         = d.frameTimeMs;
    nss.reset                  = d.reset || m_dispatchCount == 1;  // a fresh context has no history yet
    if (Config().debugView >= 0)
    {
        nss.flags         = FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
        nss.debugViews    = nss.output;
        nss.debugViewMode = static_cast<uint32_t>(Config().debugView);
    }

    ffxContext ctx = m_context;
    const ffxReturnCode_t rc = ffxDispatch(&ctx, &nss.header);

    if (fix.count)
        fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                                0, nullptr, fix.count, fix.after);

    // Make the output visible to whatever the game does next.
    VkMemoryBarrier done = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    done.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    done.dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &done, 0,
                            nullptr, 0, nullptr);

    if (Config().dumpFrame >= 0 && m_dispatchCount == uint64_t(Config().dumpFrame) + 1)
        RecordDump(cmd, d.output);
    else if (m_dumpBuffer && m_dispatchCount == m_dumpAt)
        WriteDump();

    if (rc != FFX_API_RETURN_OK)
    {
        LOGI("ffxDispatch(NSS) failed: %u", rc);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// AISS_DUMP_FRAME: copy one output frame to host memory and write it as a PPM next to the log (debugging aid).
// ---------------------------------------------------------------------------------------------------------------------
void NssUpscaler::RecordDump(VkCommandBuffer cmd, const UpscalerImage& output)
{
    auto& fn = *m_fn;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size               = VkDeviceSize(output.width) * output.height * 8;
    bci.usage              = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (fn.vkCreateBuffer(m_dev.device, &bci, nullptr, &m_dumpBuffer) != VK_SUCCESS)
        return;
    VkMemoryRequirements req;
    fn.vkGetBufferMemoryRequirements(m_dev.device, m_dumpBuffer, &req);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize       = req.size;
    mai.memoryTypeIndex      = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || fn.vkAllocateMemory(m_dev.device, &mai, nullptr, &m_dumpMemory) != VK_SUCCESS)
        return;
    fn.vkBindBufferMemory(m_dev.device, m_dumpBuffer, m_dumpMemory, 0);

    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask        = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout            = output.layout;
    b.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image                = output.image;
    b.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy region = {};
    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent       = {output.width, output.height, 1};
    fn.vkCmdCopyImageToBuffer(cmd, output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_dumpBuffer, 1, &region);
    std::swap(b.oldLayout, b.newLayout);
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    m_dumpImage = output;
    m_dumpAt    = m_dispatchCount + 16;  // read back once the frame has certainly retired
}

static float HalfToFloat(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31, man = h & 1023;
    float v = exp == 0 ? std::ldexp(float(man), -24) : exp == 31 ? 65504.0f : std::ldexp(float(man | 1024), int(exp) - 25);
    return sign ? -v : v;
}

static float Uf11ToFloat(uint32_t v, int manBits)
{
    const uint32_t exp = v >> manBits, man = v & ((1u << manBits) - 1);
    if (exp == 0)
        return std::ldexp(float(man), -14 - manBits);
    return std::ldexp(float(man | (1u << manBits)), int(exp) - 15 - manBits);
}

void NssUpscaler::WriteDump()
{
    auto&    fn  = *m_fn;
    void*    ptr = nullptr;
    if (fn.vkMapMemory(m_dev.device, m_dumpMemory, 0, VK_WHOLE_SIZE, 0, &ptr) == VK_SUCCESS)
    {
        char name[64];
        snprintf(name, sizeof(name), "aiss_frame_%d.ppm", Config().dumpFrame);
        if (FILE* f = fopen(name, "wb"))
        {
            const uint32_t w = m_dumpImage.width, h = m_dumpImage.height;
            fprintf(f, "P6\n%u %u\n255\n", w, h);
            auto toByte = [](float x) {
                x = x < 0 ? 0 : x / (1.0f + x);                                      // Reinhard, HDR -> [0,1)
                x = x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1 / 2.4f) - 0.055f;  // sRGB
                return uint8_t(x * 255.0f + 0.5f);
            };
            for (uint32_t i = 0; i < w * h; ++i)
            {
                float rgb[3] = {};
                switch (m_dumpImage.format)
                {
                case VK_FORMAT_R16G16B16A16_SFLOAT: {
                    const uint16_t* p = static_cast<const uint16_t*>(ptr) + i * 4;
                    for (int c = 0; c < 3; ++c) rgb[c] = HalfToFloat(p[c]);
                    break;
                }
                case VK_FORMAT_B10G11R11_UFLOAT_PACK32: {
                    const uint32_t p = static_cast<const uint32_t*>(ptr)[i];
                    rgb[0] = Uf11ToFloat(p & 0x7FF, 6); rgb[1] = Uf11ToFloat((p >> 11) & 0x7FF, 6); rgb[2] = Uf11ToFloat(p >> 22, 5);
                    break;
                }
                default: {  // 8-bit UNORM RGBA/BGRA: already display-referred, undo nothing
                    const uint8_t* p = static_cast<const uint8_t*>(ptr) + i * 4;
                    const bool bgra  = m_dumpImage.format == VK_FORMAT_B8G8R8A8_UNORM || m_dumpImage.format == VK_FORMAT_B8G8R8A8_SRGB;
                    const uint8_t out[3] = {p[bgra ? 2 : 0], p[1], p[bgra ? 0 : 2]};
                    fwrite(out, 1, 3, f);
                    continue;
                }
                }
                const uint8_t out[3] = {toByte(rgb[0]), toByte(rgb[1]), toByte(rgb[2])};
                fwrite(out, 1, 3, f);
            }
            fclose(f);
            LOGI("wrote %s (%ux%u, format %d)", name, w, h, m_dumpImage.format);
        }
        fn.vkUnmapMemory(m_dev.device, m_dumpMemory);
    }
    fn.vkDestroyBuffer(m_dev.device, m_dumpBuffer, nullptr);
    fn.vkFreeMemory(m_dev.device, m_dumpMemory, nullptr);
    m_dumpBuffer = VK_NULL_HANDLE;
    m_dumpMemory = VK_NULL_HANDLE;
}

NssUpscaler::~NssUpscaler()
{
    if (!m_fn)
        return;
    auto&    fn  = *m_fn;
    VkDevice dev = m_dev.device;
    if (m_context)
    {
        ffxContext ctx = m_context;
        ffxDestroyContext(&ctx, nullptr);
    }
    for (VkImageView v : m_gameViews)
        if (v)
            fn.vkDestroyImageView(dev, v, nullptr);
    if (m_pool)
        fn.vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_pipeline)
        fn.vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipeLayout)
        fn.vkDestroyPipelineLayout(dev, m_pipeLayout, nullptr);
    if (m_setLayout)
        fn.vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_sampler)
        fn.vkDestroySampler(dev, m_sampler, nullptr);
    if (m_depthView)
        fn.vkDestroyImageView(dev, m_depthView, nullptr);
    if (m_depthImage)
        fn.vkDestroyImage(dev, m_depthImage, nullptr);
    if (m_depthMemory)
        fn.vkFreeMemory(dev, m_depthMemory, nullptr);
}

// ---------------------------------------------------------------------------------------------------------------------

static std::mutex g_registryMutex;

UpscalerRegistry& Registry()
{
    // Never destructed: at process exit the game's VkDevice is already gone, so nothing may touch it.
    static UpscalerRegistry* registry = new UpscalerRegistry();
    return *registry;
}

unsigned int UpscalerRegistry::Add(std::unique_ptr<NssUpscaler> upscaler)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    const unsigned int id = m_nextId++;
    m_entries.push_back({id, std::move(upscaler)});
    return id;
}

NssUpscaler* UpscalerRegistry::Get(unsigned int id)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    for (auto& e : m_entries)
        if (e.id == id && !e.released)
            return e.upscaler.get();
    return nullptr;
}

NssUpscaler* UpscalerRegistry::Replace(unsigned int id, std::unique_ptr<NssUpscaler> fresh)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    for (auto& e : m_entries)
        if (e.id == id && !e.released)
        {
            std::unique_ptr<NssUpscaler> old = std::move(e.upscaler);
            e.upscaler                       = std::move(fresh);
            NssUpscaler* result              = e.upscaler.get();
            // The old context may still be in flight: retire it like a released one. (`e` is invalid after this.)
            m_entries.push_back({0, std::move(old), true, std::chrono::steady_clock::now()});
            return result;
        }
    return nullptr;
}

void UpscalerRegistry::Release(unsigned int id)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    for (auto& e : m_entries)
        if (e.id == id && !e.released)
        {
            e.released   = true;
            e.releasedAt = std::chrono::steady_clock::now();
        }
}

// The proxy cannot wait on the game's queues (vkd3d-proton owns them), so a released upscaler is destroyed a second
// later, long after any frame that used it has retired.
void UpscalerRegistry::CollectGarbage(bool force)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    const auto now = std::chrono::steady_clock::now();
    m_entries.erase(std::remove_if(m_entries.begin(), m_entries.end(),
                                   [&](const Entry& e) {
                                       return e.released && (force || now - e.releasedAt > std::chrono::seconds(1));
                                   }),
                    m_entries.end());
}
