// SPDX-License-Identifier: MIT
#include "nss_upscaler.h"

#include "common.h"

#include <windows.h>

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_nss.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

static const uint32_t kDepthCopySpirv[] = {
#include "depth_copy.spv.inc"
};
static const uint32_t kExposureSpirv[] = {
#include "exposure.spv.inc"
};
static const uint32_t kRegionCopy_rgba16f[] = {
#include "region_copy_rgba16f.spv.inc"
};
static const uint32_t kRegionCopy_rgba32f[] = {
#include "region_copy_rgba32f.spv.inc"
};
static const uint32_t kRegionCopy_r11f_g11f_b10f[] = {
#include "region_copy_r11f_g11f_b10f.spv.inc"
};
static const uint32_t kRegionCopy_rgba8[] = {
#include "region_copy_rgba8.spv.inc"
};
static const uint32_t kRegionCopy_rgb10_a2[] = {
#include "region_copy_rgb10_a2.spv.inc"
};
static const uint32_t kRegionCopy_rg16f[] = {
#include "region_copy_rg16f.spv.inc"
};
static const uint32_t kRegionCopy_rg32f[] = {
#include "region_copy_rg32f.spv.inc"
};
// Owned colour / motion vector copies keep the game's format when it is one of these (so NSS filters exactly the same
// texels as with an exact-size game texture), otherwise RGBA16F / RG16F.
static const struct
{
    VkFormat        format;
    const uint32_t* code;
    size_t          size;
} kRegionCopies[] = {
    {VK_FORMAT_R16G16B16A16_SFLOAT, kRegionCopy_rgba16f, sizeof(kRegionCopy_rgba16f)},
    {VK_FORMAT_R32G32B32A32_SFLOAT, kRegionCopy_rgba32f, sizeof(kRegionCopy_rgba32f)},
    {VK_FORMAT_B10G11R11_UFLOAT_PACK32, kRegionCopy_r11f_g11f_b10f, sizeof(kRegionCopy_r11f_g11f_b10f)},
    {VK_FORMAT_R8G8B8A8_UNORM, kRegionCopy_rgba8, sizeof(kRegionCopy_rgba8)},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, kRegionCopy_rgb10_a2, sizeof(kRegionCopy_rgb10_a2)},
    {VK_FORMAT_R16G16_SFLOAT, kRegionCopy_rg16f, sizeof(kRegionCopy_rg16f)},
    {VK_FORMAT_R32G32_SFLOAT, kRegionCopy_rg32f, sizeof(kRegionCopy_rg32f)},
};

struct NssUpscaler::Fn
{
#define VK_FN(name) PFN_##name name = nullptr;
    VK_FN(vkGetPhysicalDeviceMemoryProperties)
    VK_FN(vkGetPhysicalDeviceFormatProperties)
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

// "Capture frames" (plugin button): each new request counter in the control file saves the next kCaptureFrames
// dispatches' inputs and output to <control dir>\captures\<exe>_<time>\. A request that is already in the file when
// the game starts is ignored.
static constexpr int kCaptureFrames = 8;

static bool NextCaptureFrame(std::string& dir, int& index)
{
    static std::mutex  mutex;
    static bool        primed = false;
    static int         seen   = -1;
    static int         left   = 0;
    static int         next   = 0;
    static std::string current;
    std::lock_guard<std::mutex> lock(mutex);
    const int request = LiveCaptureRequest();
    if (!primed)
    {
        primed = true;
        seen   = request;
    }
    else if (request >= 0 && request != seen)
    {
        seen = request;
        std::string base = ControlFilePath();
        base             = base.substr(0, base.find_last_of('\\')) + "\\captures";
        CreateDirectoryA(base.c_str(), nullptr);
        char exe[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exe, MAX_PATH);
        std::string stem = exe;
        stem             = stem.substr(stem.find_last_of('\\') + 1);
        stem             = stem.substr(0, stem.find_last_of('.'));
        SYSTEMTIME t;
        GetLocalTime(&t);
        char stamp[32];
        snprintf(stamp, sizeof(stamp), "_%04u%02u%02u_%02u%02u%02u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
        current = base + "\\" + stem + stamp;
        if (!CreateDirectoryA(current.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            LOGI("capture: cannot create %s", current.c_str());
            return false;
        }
        left = kCaptureFrames;
        next = 0;
        LOGI("capture %d: saving %d frames to %s", request, kCaptureFrames, current.c_str());
    }
    if (left <= 0)
        return false;
    --left;
    dir   = current;
    index = next++;
    return true;
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
    m_fn->vkGetPhysicalDeviceFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
        device.gipa(device.instance, "vkGetPhysicalDeviceFormatProperties"));

    if (!CreateDepthCopy())
        return false;
    if (info.autoExposure && Config().exposure <= 0.0f && !CreateAutoExposure())
        LOGI("auto exposure unavailable; NSS runs with exposure 1");

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
    LOGI("NSS context %ux%u -> %ux%u, quality %d, hdr %d, inverted depth %d, auto exposure %d, fast maths %d, ML features 0x%x",
         info.renderWidth, info.renderHeight, info.displayWidth, info.displayHeight, info.nssQuality, info.hdr,
         info.depthInverted, m_aePipeline != VK_NULL_HANDLE, info.fastMath, device.mlFeatures);
    return true;
}

bool NssUpscaler::CreateDepthCopy()
{
    VkDevice dev = m_dev.device;
    auto&    fn  = *m_fn;

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

    auto makePipeline = [&](const uint32_t* code, size_t size, VkPipeline* out) {
        VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize                 = size;
        smci.pCode                    = code;
        VkShaderModule module         = VK_NULL_HANDLE;
        if (fn.vkCreateShaderModule(dev, &smci, nullptr, &module) != VK_SUCCESS)
            return false;
        VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module                = module;
        cpci.stage.pName                 = "main";
        cpci.layout                      = m_pipeLayout;
        const VkResult pr                = fn.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, out);
        fn.vkDestroyShaderModule(dev, module, nullptr);
        return pr == VK_SUCCESS;
    };
    if (!makePipeline(kDepthCopySpirv, sizeof(kDepthCopySpirv), &m_pipeline))
        return false;
    static_assert(sizeof(kRegionCopies) / sizeof(kRegionCopies[0]) == kCopyFormats, "kCopyFormats");
    for (uint32_t i = 0; i < kCopyFormats; ++i)
    {
        VkFormatProperties props = {};
        fn.vkGetPhysicalDeviceFormatProperties(m_dev.physicalDevice, kRegionCopies[i].format, &props);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        m_copyStorage[i] = (props.optimalTilingFeatures & need) == need;
        // The fp16 fallbacks (0: RGBA16F, 5: RG16F) are always storage formats in Vulkan.
        if ((m_copyStorage[i] || i == 0 || i == 5) && !makePipeline(kRegionCopies[i].code, kRegionCopies[i].size, &m_copyPipelines[i]))
            return false;
    }

    constexpr uint32_t kSets      = kRing * kOwnedKinds;
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSets}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSets}};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets                    = kSets;
    dpci.poolSizeCount              = 2;
    dpci.pPoolSizes                 = sizes;
    fn.vkCreateDescriptorPool(dev, &dpci, nullptr, &m_pool);
    VkDescriptorSetLayout layouts[kSets];
    std::fill(layouts, layouts + kSets, m_setLayout);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool              = m_pool;
    dsai.descriptorSetCount          = kSets;
    dsai.pSetLayouts                 = layouts;
    return fn.vkAllocateDescriptorSets(dev, &dsai, m_sets) == VK_SUCCESS;
}

const NssUpscaler::OwnedImage* NssUpscaler::AcquireOwned(uint32_t kind, VkFormat format, uint32_t width, uint32_t height)
{
    auto& fn = *m_fn;
    // m_dispatchCount was already advanced for this dispatch.
    const uint64_t now = m_dispatchCount;

    // Retire: drop entries of this kind unused for kRetireAfter dispatches, oldest first, while more than
    // kOwnedPerKind remain. Never touches anything a recent dispatch may still be reading.
    for (;;)
    {
        size_t count = 0, oldest = SIZE_MAX;
        for (size_t i = 0; i < m_owned.size(); ++i)
            if (m_owned[i].kind == kind)
            {
                ++count;
                if (m_owned[i].lastUse + kRetireAfter < now && (oldest == SIZE_MAX || m_owned[i].lastUse < m_owned[oldest].lastUse))
                    oldest = i;
            }
        if (count <= kOwnedPerKind || oldest == SIZE_MAX)
            break;
        DestroyOwned(m_owned[oldest]);
        m_owned.erase(m_owned.begin() + oldest);
    }

    for (OwnedImage& o : m_owned)
        if (o.kind == kind && o.format == format && o.width == width && o.height == height)
        {
            o.lastUse = now;
            return &o;
        }

    OwnedImage        o;
    o.kind    = kind;
    o.format  = format;
    o.width   = width;
    o.height  = height;
    o.lastUse = now;
    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType         = VK_IMAGE_TYPE_2D;
    ici.format            = format;
    ici.extent            = {width, height, 1};
    ici.mipLevels         = 1;
    ici.arrayLayers       = 1;
    ici.samples           = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling            = VK_IMAGE_TILING_OPTIMAL;
    ici.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    bool ok = fn.vkCreateImage(m_dev.device, &ici, nullptr, &o.image) == VK_SUCCESS;
    if (ok)
    {
        VkMemoryRequirements req;
        fn.vkGetImageMemoryRequirements(m_dev.device, o.image, &req);
        const uint32_t type = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize       = req.size;
        mai.memoryTypeIndex      = type;
        ok = type != UINT32_MAX && fn.vkAllocateMemory(m_dev.device, &mai, nullptr, &o.memory) == VK_SUCCESS &&
             fn.vkBindImageMemory(m_dev.device, o.image, o.memory, 0) == VK_SUCCESS;
    }
    if (ok)
        o.view = CreateView(o.image, format, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!ok || !o.view)
    {
        LOGI("failed to create a %ux%u image (format %d) for NSS input kind %u", width, height, format, kind);
        DestroyOwned(o);
        return nullptr;
    }
    LOGI("NSS input copy: kind %u, %ux%u, format %d (%u cached)", kind, width, height, format, unsigned(m_owned.size() + 1));
    m_owned.push_back(o);
    return &m_owned.back();
}

void NssUpscaler::DestroyOwned(const OwnedImage& img)
{
    auto& fn = *m_fn;
    if (img.view)
        fn.vkDestroyImageView(m_dev.device, img.view, nullptr);
    if (img.image)
        fn.vkDestroyImage(m_dev.device, img.image, nullptr);
    if (img.memory)
        fn.vkFreeMemory(m_dev.device, img.memory, nullptr);
}

void NssUpscaler::RecordCopy(VkCommandBuffer cmd, uint32_t setIndex, VkPipeline pipeline, VkImage src, VkFormat srcFormat,
                             VkImageAspectFlags aspect, VkImageLayout srcLayout, VkImageView dst, uint32_t width,
                             uint32_t height, int32_t offsetX, int32_t offsetY)
{
    auto& fn = *m_fn;
    // The view from kRing dispatches ago is no longer in flight.
    if (m_gameViews[setIndex])
        fn.vkDestroyImageView(m_dev.device, m_gameViews[setIndex], nullptr);
    m_gameViews[setIndex] = CreateView(src, srcFormat, aspect);

    VkDescriptorImageInfo srcInfo   = {VK_NULL_HANDLE, m_gameViews[setIndex], srcLayout};
    VkDescriptorImageInfo dstInfo   = {VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet  writes[2] = {};
    writes[0].sType                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet                = m_sets[setIndex];
    writes[0].dstBinding            = 0;
    writes[0].descriptorCount       = 1;
    writes[0].descriptorType        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo            = &srcInfo;
    writes[1]                       = writes[0];
    writes[1].dstBinding            = 1;
    writes[1].descriptorType        = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo            = &dstInfo;
    fn.vkUpdateDescriptorSets(m_dev.device, 2, writes, 0, nullptr);

    const int32_t push[4] = {int32_t(width), int32_t(height), offsetX, offsetY};
    fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    fn.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeLayout, 0, 1, &m_sets[setIndex], 0, nullptr);
    fn.vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    fn.vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
}

bool NssUpscaler::CreateAutoExposure()
{
    VkDevice dev = m_dev.device;
    auto&    fn  = *m_fn;

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size               = sizeof(float) * 4 * kRing;
    bci.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (fn.vkCreateBuffer(dev, &bci, nullptr, &m_aeBuffer) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req;
    fn.vkGetBufferMemoryRequirements(dev, m_aeBuffer, &req);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize       = req.size;
    mai.memoryTypeIndex      = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped             = nullptr;
    if (mai.memoryTypeIndex == UINT32_MAX || fn.vkAllocateMemory(dev, &mai, nullptr, &m_aeMemory) != VK_SUCCESS ||
        fn.vkBindBufferMemory(dev, m_aeBuffer, m_aeMemory, 0) != VK_SUCCESS || fn.vkMapMemory(dev, m_aeMemory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
        return false;
    m_aeResults = static_cast<float*>(mapped);
    memset(m_aeResults, 0, sizeof(float) * 4 * kRing);

    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0]                    = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, &m_sampler};
    bindings[1]                    = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount                    = 2;
    dslci.pBindings                       = bindings;
    if (fn.vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &m_aeSetLayout) != VK_SUCCESS)
        return false;
    VkPushConstantRange        pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pl  = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount              = 1;
    pl.pSetLayouts                 = &m_aeSetLayout;
    pl.pushConstantRangeCount      = 1;
    pl.pPushConstantRanges         = &pcr;
    if (fn.vkCreatePipelineLayout(dev, &pl, nullptr, &m_aePipeLayout) != VK_SUCCESS)
        return false;
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize                 = sizeof(kExposureSpirv);
    smci.pCode                    = kExposureSpirv;
    VkShaderModule module         = VK_NULL_HANDLE;
    if (fn.vkCreateShaderModule(dev, &smci, nullptr, &module) != VK_SUCCESS)
        return false;
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module                = module;
    cpci.stage.pName                 = "main";
    cpci.layout                      = m_aePipeLayout;
    const VkResult pr                = fn.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &m_aePipeline);
    fn.vkDestroyShaderModule(dev, module, nullptr);
    if (pr != VK_SUCCESS)
    {
        m_aePipeline = VK_NULL_HANDLE;
        return false;
    }
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kRing}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kRing}};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets                    = kRing;
    dpci.poolSizeCount              = 2;
    dpci.pPoolSizes                 = sizes;
    if (fn.vkCreateDescriptorPool(dev, &dpci, nullptr, &m_aePool) != VK_SUCCESS)
        return false;
    VkDescriptorSetLayout layouts[kRing];
    std::fill(layouts, layouts + kRing, m_aeSetLayout);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool              = m_aePool;
    dsai.descriptorSetCount          = kRing;
    dsai.pSetLayouts                 = layouts;
    if (fn.vkAllocateDescriptorSets(dev, &dsai, m_aeSets) != VK_SUCCESS)
    {
        fn.vkDestroyPipeline(dev, m_aePipeline, nullptr);
        m_aePipeline = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

// Records this dispatch's luminance measurement and returns the exposure to use now: 0.18 / geometric mean luminance
// of a frame kExposureLatency dispatches ago, adapted smoothly (the value only feeds NSS's internal tonemap and its
// inverse, so a slightly late exposure is harmless; it must just put the scene in the network's input range).
float NssUpscaler::AutoExposure(VkCommandBuffer cmd, const UpscalerImage& color, uint32_t width, uint32_t height)
{
    auto&          fn   = *m_fn;
    const uint64_t n    = m_dispatchCount;  // already incremented for this dispatch
    const uint32_t slot = uint32_t(n % kRing);

    // Read the measurement recorded kExposureLatency dispatches ago, if it carries the expected tag.
    if (n > kExposureLatency)
    {
        const uint64_t want = n - kExposureLatency;
        const float*   r    = m_aeResults + 4 * (want % kRing);
        uint32_t       tag  = 0;
        memcpy(&tag, &r[2], 4);
        if (tag == uint32_t(want & 0xFFFFFFu) && r[1] > 0.0f)
        {
            const float geoMean = std::exp2(r[0] / r[1]);
            const float target  = std::clamp(0.18f / std::max(geoMean, 1e-6f), 1.0f / 64.0f, 4096.0f);
            // Adapt in log space, ~1/4 of the way per frame; jump on the first measurement.
            m_exposure = m_exposure <= 0.0f ? target : std::exp2(std::log2(m_exposure) + 0.25f * (std::log2(target) - std::log2(m_exposure)));
        }
    }

    // Measure this frame's colour.
    VkImageLayout layout = color.layout;
    VkImageMemoryBarrier toRead = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    const bool needsTransition  = !IsSampledLayout(layout);
    if (needsTransition)
    {
        toRead.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT;
        toRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        toRead.oldLayout           = layout;
        toRead.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        toRead.srcQueueFamilyIndex = toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image               = color.image;
        toRead.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toRead);
        layout = VK_IMAGE_LAYOUT_GENERAL;
    }
    if (m_aeViews[slot])
        fn.vkDestroyImageView(m_dev.device, m_aeViews[slot], nullptr);
    m_aeViews[slot] = CreateView(color.image, color.format, VK_IMAGE_ASPECT_COLOR_BIT);
    VkDescriptorImageInfo  imageInfo  = {VK_NULL_HANDLE, m_aeViews[slot], layout};
    VkDescriptorBufferInfo bufferInfo = {m_aeBuffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet   writes[2]  = {};
    writes[0].sType                   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet                  = m_aeSets[slot];
    writes[0].dstBinding              = 0;
    writes[0].descriptorCount         = 1;
    writes[0].descriptorType          = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo              = &imageInfo;
    writes[1]                         = writes[0];
    writes[1].dstBinding              = 1;
    writes[1].descriptorType          = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pImageInfo              = nullptr;
    writes[1].pBufferInfo             = &bufferInfo;
    fn.vkUpdateDescriptorSets(m_dev.device, 2, writes, 0, nullptr);

    const uint32_t push[4] = {width, height, 4u, slot | uint32_t((n & 0xFFFFFFu) << 8)};
    fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_aePipeline);
    fn.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_aePipeLayout, 0, 1, &m_aeSets[slot], 0, nullptr);
    fn.vkCmdPushConstants(cmd, m_aePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    fn.vkCmdDispatch(cmd, 1, 1, 1);

    VkMemoryBarrier toHost = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toHost.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    toHost.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &toHost, 0, nullptr, 0, nullptr);
    if (needsTransition)
    {
        std::swap(toRead.oldLayout, toRead.newLayout);
        toRead.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toRead.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toRead);
    }
    return m_exposure > 0.0f ? m_exposure : 1.0f;
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

VkFormat NssUpscaler::CopyFormat(VkFormat game, VkFormat fallback) const
{
    for (uint32_t i = 0; i < kCopyFormats; ++i)
        if (kRegionCopies[i].format == game && m_copyStorage[i])
            return game;
    return fallback;
}

static uint32_t CopyPipeline(VkFormat f)
{
    for (uint32_t i = 0; i < sizeof(kRegionCopies) / sizeof(kRegionCopies[0]); ++i)
        if (kRegionCopies[i].format == f)
            return i;
    return 0;
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

    // ---- owned copies, exactly renderW x renderH (see OwnedImage) ----
    const bool copyColor  = d.color.width != renderW || d.color.height != renderH;
    const bool copyMotion = d.motionVectors.width != renderW || d.motionVectors.height != renderH;
    OwnedImage depthImg, colorImg, motionImg;
    if (const OwnedImage* o = AcquireOwned(kOwnedDepth, VK_FORMAT_R32_SFLOAT, renderW, renderH))
        depthImg = *o;
    else
        return false;
    if (copyColor)
    {
        if (const OwnedImage* o = AcquireOwned(kOwnedColor, CopyFormat(d.color.format, VK_FORMAT_R16G16B16A16_SFLOAT), renderW, renderH))
            colorImg = *o;
        else
            return false;
    }
    if (copyMotion)
    {
        if (const OwnedImage* o = AcquireOwned(kOwnedMotion, CopyFormat(d.motionVectors.format, VK_FORMAT_R16G16_SFLOAT), renderW, renderH))
            motionImg = *o;
        else
            return false;
    }
    if (copyColor != m_loggedCopy[0] || copyMotion != m_loggedCopy[1])
    {
        LOGI("NSS inputs at render %ux%u: colour %ux%u%s, motion vectors %ux%u%s", renderW, renderH, d.color.width,
             d.color.height, copyColor ? " (copied to the render size)" : "", d.motionVectors.width, d.motionVectors.height,
             copyMotion ? " (copied to the render size)" : "");
        m_loggedCopy[0] = copyColor;
        m_loggedCopy[1] = copyMotion;
    }

    VkImageMemoryBarrier toStorage = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toStorage.srcAccessMask        = VK_ACCESS_SHADER_READ_BIT;
    toStorage.dstAccessMask        = VK_ACCESS_SHADER_WRITE_BIT;
    toStorage.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
    toStorage.newLayout            = VK_IMAGE_LAYOUT_GENERAL;
    toStorage.srcQueueFamilyIndex = toStorage.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toStorage.image                = depthImg.image;
    toStorage.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier pre[4] = {toStorage};
    uint32_t             preCount = 1;
    if (depthNeedsTransition)
        pre[preCount++] = depthToRead;
    if (copyColor)
    {
        pre[preCount]         = toStorage;
        pre[preCount++].image = colorImg.image;
    }
    if (copyMotion)
    {
        pre[preCount]         = toStorage;
        pre[preCount++].image = motionImg.image;
    }
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                            nullptr, preCount, pre);

    const uint32_t setBase = slot * kOwnedKinds;
    RecordCopy(cmd, setBase + kOwnedDepth, m_pipeline, d.depth.image, viewFormat, aspect, depthLayout, depthImg.view, renderW,
               renderH, d.depthOffsetX, d.depthOffsetY);

    VkImageMemoryBarrier toRead = toStorage;
    toRead.srcAccessMask        = VK_ACCESS_SHADER_WRITE_BIT;
    toRead.dstAccessMask        = VK_ACCESS_SHADER_READ_BIT;
    toRead.oldLayout            = VK_IMAGE_LAYOUT_GENERAL;
    toRead.newLayout            = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    std::string captureDir;
    int         captureIndex = 0;
    const bool  captured     = NextCaptureFrame(captureDir, captureIndex);
    const bool  dumpInputs   = captured || (Config().dumpInputsFrom >= 0 && m_dispatchCount > uint64_t(Config().dumpInputsFrom) &&
                                         m_dispatchCount <= uint64_t(Config().dumpInputsFrom + Config().dumpInputsCount));
    // File prefix: "<capture dir>\NNN_" for a capture, "aiss_in_NNN_" next to the game for AISS_DUMP_INPUTS.
    char dumpPrefix[MAX_PATH];
    if (captured)
        snprintf(dumpPrefix, sizeof(dumpPrefix), "%s\\%03d_", captureDir.c_str(), captureIndex);
    else
        snprintf(dumpPrefix, sizeof(dumpPrefix), "aiss_in_%03llu_", (unsigned long long)(m_dispatchCount - 1));
    VkImageMemoryBarrier post[2] = {toRead, depthToRead};
    if (depthNeedsTransition)
    {
        std::swap(post[1].oldLayout, post[1].newLayout);
        post[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        post[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    }

    if (dumpInputs)
    {
        // The R32F copy is in GENERAL until the barrier below.
        fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        char name[MAX_PATH + 16];
        const uint64_t f = m_dispatchCount - 1;
        snprintf(name, sizeof(name), "%scolor.raw", dumpPrefix);
        RecordRaw(cmd, d.color, VK_IMAGE_ASPECT_COLOR_BIT, name);
        snprintf(name, sizeof(name), "%smv.raw", dumpPrefix);
        RecordRaw(cmd, d.motionVectors, VK_IMAGE_ASPECT_COLOR_BIT, name);
        snprintf(name, sizeof(name), "%sdepth.raw", dumpPrefix);
        RecordRaw(cmd, {depthImg.image, VK_FORMAT_R32_SFLOAT, renderW, renderH, VK_IMAGE_LAYOUT_GENERAL}, VK_IMAGE_ASPECT_COLOR_BIT, name);
        if (captured)
        {
            snprintf(name, sizeof(name), "%s\\frames.txt", captureDir.c_str());
            if (FILE* meta = fopen(name, "a"))
            {
                fprintf(meta, "%03d dispatch %llu render %ux%u display %ux%u jitter %.4f,%.4f mv scale %.3f,%.3f reset %d "
                              "quality %d hdr %d inverted %d game exposure %.3f formats colour %d mv %d depth %d\n",
                        captureIndex, (unsigned long long)f, renderW, renderH, m_info.displayWidth, m_info.displayHeight,
                        d.jitterX, d.jitterY, d.mvScaleX, d.mvScaleY, d.reset, m_info.nssQuality, m_info.hdr,
                        m_info.depthInverted, d.exposure, d.color.format, d.motionVectors.format, d.depth.format);
                fclose(meta);
            }
        }
        LOGI("dump %llu: render %ux%u jitter %.4f,%.4f mv scale %.3f,%.3f reset %d dt %.2f, formats colour %d mv %d depth %d (%ux%u, %ux%u, %ux%u)",
             (unsigned long long)f, renderW, renderH, d.jitterX, d.jitterY, d.mvScaleX, d.mvScaleY, d.reset, d.frameTimeMs,
             d.color.format, d.motionVectors.format, d.depth.format, d.color.width, d.color.height, d.motionVectors.width,
             d.motionVectors.height, d.depth.width, d.depth.height);
    }

    // ---- auto exposure (before the colour is moved to GENERAL for NSS) ----
    float exposure = d.exposure;
    if (m_aePipeline && Config().exposure <= 0.0f)
        exposure = AutoExposure(cmd, d.color, renderW, renderH);

    // ---- NSS ----
    LayoutFix fix;
    const uint32_t colorState  = fix.Adopt(d.color, false);
    const uint32_t motionState = fix.Adopt(d.motionVectors, false);
    ffxApiDispatchDescNss nss = {};
    nss.header.type           = FFX_API_DISPATCH_DESC_TYPE_NSS;
    nss.commandList           = cmd;
    nss.output                = WrapImage(d.output, FFX_API_RESOURCE_USAGE_UAV, fix.Adopt(d.output, true));
    nss.depth                 = WrapImage({depthImg.image, VK_FORMAT_R32_SFLOAT, renderW, renderH, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                          0, FFX_API_RESOURCE_STATE_COMPUTE_READ);

    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, depthNeedsTransition ? 2 : 1, post);
    if (fix.count)
        fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                0, nullptr, fix.count, fix.before);

    // Colour / motion vectors in a texture larger than the render size: copy the render-size region (the game images are
    // now in GENERAL or SHADER_READ_ONLY_OPTIMAL). A compute copy, as game images need not allow transfers.
    if (copyColor || copyMotion)
    {
        auto readLayout = [](const UpscalerImage& img) {
            return img.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? img.layout : VK_IMAGE_LAYOUT_GENERAL;
        };
        VkImageMemoryBarrier copied[2];
        uint32_t             copiedCount = 0;
        auto copy = [&](const UpscalerImage& src, const OwnedImage& dst, uint32_t kind) {
            RecordCopy(cmd, setBase + kind, m_copyPipelines[CopyPipeline(dst.format)], src.image, src.format,
                       VK_IMAGE_ASPECT_COLOR_BIT, readLayout(src), dst.view, renderW, renderH, 0, 0);
            VkImageMemoryBarrier b = toRead;
            b.image                = dst.image;
            copied[copiedCount++]  = b;
        };
        if (copyColor)
            copy(d.color, colorImg, kOwnedColor);
        if (copyMotion)
            copy(d.motionVectors, motionImg, kOwnedMotion);
        fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                0, nullptr, copiedCount, copied);
    }
    auto ownedInput = [&](const OwnedImage& o) {
        return WrapImage({o.image, o.format, renderW, renderH, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}, 0,
                         FFX_API_RESOURCE_STATE_COMPUTE_READ);
    };
    nss.color         = copyColor ? ownedInput(colorImg) : WrapImage(d.color, 0, colorState);
    nss.motionVectors = copyMotion ? ownedInput(motionImg) : WrapImage(d.motionVectors, 0, motionState);

    // NSS's jitter is the negative of the DLSS pixel offset on both axes.
    nss.jitterOffset           = {-d.jitterX * Config().jitterSign[0], -d.jitterY * Config().jitterSign[1]};
    nss.motionVectorScale      = {d.mvScaleX * Config().mvSign[0], d.mvScaleY * Config().mvSign[1]};
    nss.renderSize             = {renderW, renderH};
    nss.upscaleSize            = {m_info.displayWidth, m_info.displayHeight};
    nss.cameraNear             = Config().cameraNear;
    nss.cameraFar              = Config().cameraFar;
    nss.cameraFovAngleVertical = Config().cameraFovY;
    nss.exposure               = exposure;
    nss.frameTimeDelta         = d.frameTimeMs;
    nss.reset                  = d.reset || m_dispatchCount == 1;  // a fresh context has no history yet
    if (LogLevel() >= 2 || (m_dispatchCount % 600) == 1)
        LOGI("evaluate %llu: exposure %.3f%s", (unsigned long long)m_dispatchCount, exposure, m_aePipeline ? " (auto)" : "");
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

    if (dumpInputs)
    {
        char name[MAX_PATH + 16];
        snprintf(name, sizeof(name), "%soutput.raw", dumpPrefix);
        RecordRaw(cmd, d.output, VK_IMAGE_ASPECT_COLOR_BIT, name);
    }
    WriteRawDumps(false);

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

static uint32_t BytesPerPixel(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return 4;
    case VK_FORMAT_R16_SFLOAT:
        return 2;
    default:
        return 0;
    }
}

void NssUpscaler::RecordRaw(VkCommandBuffer cmd, const UpscalerImage& img, VkImageAspectFlags aspect, const char* name)
{
    auto&          fn  = *m_fn;
    const uint32_t bpp = BytesPerPixel(img.format);
    if (!bpp)
    {
        LOGI("dump %s: format %d not supported", name, img.format);
        return;
    }
    RawDump r;
    r.width = img.width, r.height = img.height, r.format = img.format, r.bpp = bpp;
    snprintf(r.name, sizeof(r.name), "%s", name);
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size               = VkDeviceSize(img.width) * img.height * bpp;
    bci.usage              = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (fn.vkCreateBuffer(m_dev.device, &bci, nullptr, &r.buffer) != VK_SUCCESS)
        return;
    VkMemoryRequirements req;
    fn.vkGetBufferMemoryRequirements(m_dev.device, r.buffer, &req);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize       = req.size;
    mai.memoryTypeIndex      = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || fn.vkAllocateMemory(m_dev.device, &mai, nullptr, &r.memory) != VK_SUCCESS)
        return;
    fn.vkBindBufferMemory(m_dev.device, r.buffer, r.memory, 0);

    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask        = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout            = img.layout;
    b.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image                = img.image;
    b.subresourceRange     = {aspect, 0, 1, 0, 1};
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy region = {};
    region.imageSubresource  = {aspect, 0, 0, 1};
    region.imageExtent       = {img.width, img.height, 1};
    fn.vkCmdCopyImageToBuffer(cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.buffer, 1, &region);
    std::swap(b.oldLayout, b.newLayout);
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    r.readAt = m_dispatchCount + 16;  // retired by then
    m_rawDumps.push_back(r);
}

void NssUpscaler::WriteRawDumps(bool all)
{
    auto& fn = *m_fn;
    for (auto it = m_rawDumps.begin(); it != m_rawDumps.end();)
    {
        if (!all && m_dispatchCount < it->readAt)
        {
            ++it;
            continue;
        }
        void* ptr = nullptr;
        if (fn.vkMapMemory(m_dev.device, it->memory, 0, VK_WHOLE_SIZE, 0, &ptr) == VK_SUCCESS)
        {
            if (FILE* f = fopen(it->name, "wb"))
            {
                const uint32_t header[5] = {0x52534941u /* 'AISR' */, it->width, it->height, it->format, it->bpp};
                fwrite(header, 4, 5, f);
                fwrite(ptr, 1, size_t(it->width) * it->height * it->bpp, f);
                fclose(f);
            }
            fn.vkUnmapMemory(m_dev.device, it->memory);
        }
        fn.vkDestroyBuffer(m_dev.device, it->buffer, nullptr);
        fn.vkFreeMemory(m_dev.device, it->memory, nullptr);
        it = m_rawDumps.erase(it);
    }
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
    for (VkImageView v : m_aeViews)
        if (v)
            fn.vkDestroyImageView(dev, v, nullptr);
    if (m_aePool)
        fn.vkDestroyDescriptorPool(dev, m_aePool, nullptr);
    if (m_aePipeline)
        fn.vkDestroyPipeline(dev, m_aePipeline, nullptr);
    if (m_aePipeLayout)
        fn.vkDestroyPipelineLayout(dev, m_aePipeLayout, nullptr);
    if (m_aeSetLayout)
        fn.vkDestroyDescriptorSetLayout(dev, m_aeSetLayout, nullptr);
    if (m_aeMemory)
        fn.vkFreeMemory(dev, m_aeMemory, nullptr);
    if (m_aeBuffer)
        fn.vkDestroyBuffer(dev, m_aeBuffer, nullptr);
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
    for (VkPipeline p : m_copyPipelines)
        if (p)
            fn.vkDestroyPipeline(dev, p, nullptr);
    for (const OwnedImage& o : m_owned)
        DestroyOwned(o);
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
