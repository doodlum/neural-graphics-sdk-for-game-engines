// SPDX-License-Identifier: MIT
#include "nss_upscaler.h"

#include "common.h"

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_nss.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
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
    if (Config().nssQuality >= 0 && Config().nssQuality <= 2)
        return Config().nssQuality;
    // The QUALITY shader mode switches to NSS's large network, roughly 5x the cost on a Steam Deck. Every DLSS mode
    // therefore maps to the small network; AISS_NSS_QUALITY=0 opts in to the large one.
    switch (perfQualityValue)
    {
    case 2:  // MaxQuality
    case 4:  // UltraQuality
    case 5:  // DLAA
        return FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED;
    default:
        return FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE;
    }
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

    ffxContext ctx = nullptr;
    const ffxReturnCode_t rc = ffxCreateContext(&ctx, &nss.header, nullptr);
    if (rc != FFX_API_RETURN_OK)
    {
        LOGI("ffxCreateContext(NSS) failed: %u", rc);
        return false;
    }
    m_context = ctx;
    LOGI("NSS context %ux%u -> %ux%u, quality %d, hdr %d, inverted depth %d, ML features 0x%x", info.renderWidth,
         info.renderHeight, info.displayWidth, info.displayHeight, info.nssQuality, info.hdr, info.depthInverted,
         device.mlFeatures);
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
    VkPhysicalDeviceMemoryProperties props;
    fn.vkGetPhysicalDeviceMemoryProperties(m_dev.physicalDevice, &props);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount && type == UINT32_MAX; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            type = i;
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
    nss.jitterOffset           = {-d.jitterX, -d.jitterY};
    nss.motionVectorScale      = {d.mvScaleX, d.mvScaleY};
    nss.renderSize             = {renderW, renderH};
    nss.upscaleSize            = {m_info.displayWidth, m_info.displayHeight};
    nss.cameraNear             = Config().cameraNear;
    nss.cameraFar              = Config().cameraFar;
    nss.cameraFovAngleVertical = Config().cameraFovY;
    nss.exposure               = d.exposure;
    nss.frameTimeDelta         = d.frameTimeMs;
    nss.reset                  = d.reset;

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

    if (rc != FFX_API_RETURN_OK)
    {
        LOGI("ffxDispatch(NSS) failed: %u", rc);
        return false;
    }
    return true;
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
    static UpscalerRegistry registry;
    return registry;
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
