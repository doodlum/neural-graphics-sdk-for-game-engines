// Intel XeSS Super Resolution (Vulkan) as an alternative to NSS. libxess.dll (Intel's binary, redistributed
// unmodified) is loaded at runtime from next to the proxy and used only through its public API.
// SPDX-License-Identifier: MIT
#include "nss_upscaler.h"

#include "common.h"

#include <windows.h>

#include <xess/xess_vk.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>

namespace
{
struct XessApi
{
#define XESS_FN(name) decltype(&::name) name = nullptr;
    XESS_FN(xessGetVersion)
    XESS_FN(xessVKCreateContext)
    XESS_FN(xessVKBuildPipelines)
    XESS_FN(xessVKInit)
    XESS_FN(xessVKExecute)
    XESS_FN(xessDestroyContext)
    XESS_FN(xessSetVelocityScale)
    XESS_FN(xessSetJitterScale)
    XESS_FN(xessSetLoggingCallback)
    XESS_FN(xessGetOptimalInputResolution)
    XESS_FN(xessGetProperties)
    XESS_FN(xessIsOptimalDriver)
#undef XESS_FN
};

std::mutex g_xessMutex;
bool       g_xessFailed = false;

// libxess.dll from the proxy's own directory (the prefix's system32 next to _nvngx.dll), loaded once.
const XessApi* LoadXess()
{
    std::lock_guard<std::mutex> lock(g_xessMutex);
    static XessApi api;
    static bool    tried = false;
    if (tried)
        return g_xessFailed ? nullptr : &api;
    tried = true;

    char    path[MAX_PATH] = {};
    HMODULE self           = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(&LoadXess), &self);
    GetModuleFileNameA(self, path, MAX_PATH);
    if (char* slash = strrchr(path, '\\'))
        strcpy(slash + 1, "libxess.dll");
    else
        strcpy(path, "libxess.dll");
    HMODULE lib = LoadLibraryA(path);
    if (!lib)
    {
        LOGI("XeSS: cannot load %s (error %lu); using NSS", path, GetLastError());
        g_xessFailed = true;
        return nullptr;
    }
#define XESS_FN(name)                                                                                                  \
    api.name = reinterpret_cast<decltype(api.name)>(GetProcAddress(lib, #name));                                       \
    if (!api.name)                                                                                                     \
    {                                                                                                                  \
        LOGI("XeSS: %s has no %s; using NSS", path, #name);                                                            \
        g_xessFailed = true;                                                                                           \
        return nullptr;                                                                                                \
    }
    XESS_FN(xessGetVersion)
    XESS_FN(xessVKCreateContext)
    XESS_FN(xessVKBuildPipelines)
    XESS_FN(xessVKInit)
    XESS_FN(xessVKExecute)
    XESS_FN(xessDestroyContext)
    XESS_FN(xessSetVelocityScale)
    XESS_FN(xessSetJitterScale)
    XESS_FN(xessSetLoggingCallback)
    XESS_FN(xessGetOptimalInputResolution)
    XESS_FN(xessGetProperties)
    XESS_FN(xessIsOptimalDriver)
#undef XESS_FN
    xess_version_t v = {};
    api.xessGetVersion(&v);
    LOGI("XeSS: loaded %s, version %u.%u.%u", path, v.major, v.minor, v.patch);
    return &api;
}

void XessLog(const char* message, xess_logging_level_t level)
{
    LOGI("XeSS[%d]: %s", int(level), message);
}

bool IsDepth(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_X8_D24_UNORM_PACK32 ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

bool HasStencil(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

class XessUpscaler : public Upscaler
{
public:
    static std::unique_ptr<XessUpscaler> Create(const VulkanDevice& device, const UpscalerCreateInfo& info);
    ~XessUpscaler() override;
    bool Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d) override;

private:
    bool        Init(const VulkanDevice& device, const UpscalerCreateInfo& info);
    VkImageView View(uint32_t slot, const UpscalerImage& img, VkImageAspectFlags aspect);

    const XessApi*        m_api = nullptr;
    xess_context_handle_t m_ctx = nullptr;
    xess_2d_t             m_minInput = {}, m_maxInput = {};
    float                 m_velocityScale[2] = {0, 0};
    bool                  m_warnedMin        = false;

    PFN_vkCreateImageView    m_createView  = nullptr;
    PFN_vkDestroyImageView   m_destroyView = nullptr;
    PFN_vkCmdPipelineBarrier m_barrier     = nullptr;

    // Per-dispatch views of the game's images, recycled once the ring wraps (frames in flight are far fewer).
    static constexpr uint32_t kRing = 8;
    VkImageView m_views[kRing][4] = {};
    uint64_t    m_dispatchCount   = 0;
};

std::unique_ptr<XessUpscaler> XessUpscaler::Create(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    std::unique_ptr<XessUpscaler> up(new XessUpscaler());
    if (!up->Init(device, info))
        return nullptr;
    return up;
}

bool XessUpscaler::Init(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    m_dev  = device;
    m_info = info;
    m_api  = LoadXess();
    if (!m_api)
        return false;
    m_createView  = reinterpret_cast<PFN_vkCreateImageView>(device.gdpa(device.device, "vkCreateImageView"));
    m_destroyView = reinterpret_cast<PFN_vkDestroyImageView>(device.gdpa(device.device, "vkDestroyImageView"));
    m_barrier     = reinterpret_cast<PFN_vkCmdPipelineBarrier>(device.gdpa(device.device, "vkCmdPipelineBarrier"));

    xess_result_t r = m_api->xessVKCreateContext(device.instance, device.physicalDevice, device.device, &m_ctx);
    if (r != XESS_RESULT_SUCCESS)
    {
        LOGI("XeSS: xessVKCreateContext failed (%d)", r);
        return false;
    }
    m_api->xessSetLoggingCallback(m_ctx, LogLevel() >= 2 ? XESS_LOGGING_LEVEL_INFO : XESS_LOGGING_LEVEL_WARNING, XessLog);
    if (m_api->xessIsOptimalDriver(m_ctx) != XESS_RESULT_SUCCESS)
        LOGI("XeSS: xessIsOptimalDriver says the driver is not optimal");

    uint32_t flags = 0;
    if (info.highResMv)
        flags |= XESS_INIT_FLAG_HIGH_RES_MV;
    if (info.depthInverted)
        flags |= XESS_INIT_FLAG_INVERTED_DEPTH;
    if (!info.hdr)
        flags |= XESS_INIT_FLAG_LDR_INPUT_COLOR;
    // Like NSS: XeSS's own auto exposure when the game asked DLSS for it, unless DU_EXPOSURE forces a value.
    if (info.autoExposure && Config().exposure <= 0.0f)
        flags |= XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;

    // A preset only sets XeSS's input range (the output for a given input size is bit-identical across presets, checked
    // with ngx_d3d11_test), so Ultra Performance: the widest range, 1/3 of the output up to the output, which covers
    // the dynamic-resolution range the proxy reports to games.
    xess_2d_t               output  = {info.displayWidth, info.displayHeight};
    xess_quality_settings_t quality = XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
    const int liveXq = LiveXessQuality();
    const int xq     = liveXq >= 0 ? liveXq : Config().xessQuality;
    if (xq >= XESS_QUALITY_SETTING_ULTRA_PERFORMANCE && xq <= XESS_QUALITY_SETTING_AA)
        quality = xess_quality_settings_t(xq);

    // Build the pipelines now (blocking): it is the slow part of initialisation (~0.3 s on a Steam Deck).
    const auto t0 = GetTickCount64();
    r = m_api->xessVKBuildPipelines(m_ctx, VK_NULL_HANDLE, true, flags);
    if (r != XESS_RESULT_SUCCESS)
        LOGI("XeSS: xessVKBuildPipelines failed (%d)", r);
    const auto t1 = GetTickCount64();

    xess_vk_init_params_t ip = {};
    ip.outputResolution      = output;
    ip.qualitySetting        = quality;
    ip.initFlags             = flags;
    r                        = m_api->xessVKInit(m_ctx, &ip);
    if (r != XESS_RESULT_SUCCESS)
    {
        LOGI("XeSS: xessVKInit failed (%d)", r);
        return false;
    }
    xess_2d_t optimal = {};
    m_api->xessGetOptimalInputResolution(m_ctx, &ip.outputResolution, quality, &optimal, &m_minInput, &m_maxInput);
    xess_properties_t props = {};
    m_api->xessGetProperties(m_ctx, &ip.outputResolution, &props);
    LOGI("XeSS context %ux%u -> %ux%u, preset %d (input %ux%u..%ux%u), flags 0x%x, pipelines %llu ms, init %llu ms, "
         "temporary memory %.1f MB",
         info.renderWidth, info.renderHeight, info.displayWidth, info.displayHeight, int(quality), m_minInput.x,
         m_minInput.y, m_maxInput.x, m_maxInput.y, flags, (unsigned long long)(t1 - t0),
         (unsigned long long)(GetTickCount64() - t1), (props.tempBufferHeapSize + props.tempTextureHeapSize) / 1048576.0);
    if (info.renderWidth < m_minInput.x || info.renderHeight < m_minInput.y)
        LOGI("XeSS: warning: render size %ux%u is below the preset's minimum input", info.renderWidth, info.renderHeight);
    return true;
}

XessUpscaler::~XessUpscaler()
{
    if (m_ctx)
        m_api->xessDestroyContext(m_ctx);
    for (auto& slot : m_views)
        for (VkImageView v : slot)
            if (v)
                m_destroyView(m_dev.device, v, nullptr);
}

VkImageView XessUpscaler::View(uint32_t slot, const UpscalerImage& img, VkImageAspectFlags aspect)
{
    const uint32_t ring = uint32_t(m_dispatchCount % kRing);
    if (m_views[ring][slot])
        m_destroyView(m_dev.device, m_views[ring][slot], nullptr);
    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image                 = img.image;
    vci.viewType              = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                = img.format;
    vci.subresourceRange      = {aspect, 0, 1, 0, 1};
    VkImageView view          = VK_NULL_HANDLE;
    m_createView(m_dev.device, &vci, nullptr, &view);
    m_views[ring][slot] = view;
    return view;
}

bool XessUpscaler::Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d)
{
    // XeSS wants its inputs in SHADER_READ_ONLY_OPTIMAL and the output in GENERAL, and does no synchronisation itself.
    const UpscalerImage* inputs[3] = {&d.color, &d.motionVectors, &d.depth};
    VkImageMemoryBarrier before[4], after[4];
    uint32_t             count = 0;
    auto transition = [&](const UpscalerImage& img, VkImageLayout want, VkAccessFlags access) {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask        = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask        = access;
        b.oldLayout            = img.layout;
        b.newLayout            = want;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image                = img.image;
        const VkImageAspectFlags aspect =
            IsDepth(img.format) ? VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(img.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
                                : VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange = {aspect, 0, 1, 0, 1};
        before[count]      = b;
        std::swap(b.oldLayout, b.newLayout);
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        after[count++]  = b;
    };
    for (const UpscalerImage* img : inputs)
        transition(*img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    transition(d.output, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    m_barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, count,
              before);

    auto info = [&](uint32_t slot, const UpscalerImage& img, VkImageAspectFlags aspect) {
        xess_vk_image_view_info v = {};
        v.imageView               = View(slot, img, aspect);
        v.image                   = img.image;
        v.subresourceRange        = {aspect, 0, 1, 0, 1};
        v.format                  = img.format;
        v.width                   = img.width;
        v.height                  = img.height;
        return v;
    };
    xess_vk_execute_params_t ep = {};
    ep.colorTexture             = info(0, d.color, VK_IMAGE_ASPECT_COLOR_BIT);
    ep.velocityTexture          = info(1, d.motionVectors, VK_IMAGE_ASPECT_COLOR_BIT);
    ep.depthTexture             = info(2, d.depth, IsDepth(d.depth.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
    ep.outputTexture            = info(3, d.output, VK_IMAGE_ASPECT_COLOR_BIT);
    // DLSS and XeSS share the jitter convention (the sample offset in render pixels); DU_JITTER_SIGN still applies.
    ep.jitterOffsetX  = d.jitterX * Config().jitterSign[0];
    ep.jitterOffsetY  = d.jitterY * Config().jitterSign[1];
    ep.exposureScale  = d.exposure;
    ep.resetHistory   = d.reset || m_dispatchCount == 0;
    // Dynamic resolution: the per-frame render size (DLSS render subrect), up to the textures' and the preset's maximum.
    ep.inputWidth     = std::min({d.renderWidth, m_maxInput.x, d.color.width});
    ep.inputHeight    = std::min({d.renderHeight, m_maxInput.y, d.color.height});
    if ((ep.inputWidth < m_minInput.x || ep.inputHeight < m_minInput.y) && !m_warnedMin)
    {
        m_warnedMin = true;
        LOGI("XeSS: warning: render size %ux%u is below the preset's minimum input %ux%u", ep.inputWidth, ep.inputHeight,
             m_minInput.x, m_minInput.y);
    }
    ep.inputDepthBase = {uint32_t(std::max(d.depthOffsetX, 0)), uint32_t(std::max(d.depthOffsetY, 0))};

    // Motion vectors in render pixels, current -> previous, as DLSS's (times the game's MV scale).
    const float vx = d.mvScaleX * Config().mvSign[0], vy = d.mvScaleY * Config().mvSign[1];
    if (vx != m_velocityScale[0] || vy != m_velocityScale[1])
    {
        m_api->xessSetVelocityScale(m_ctx, vx, vy);
        m_velocityScale[0] = vx;
        m_velocityScale[1] = vy;
    }
    const xess_result_t r = m_api->xessVKExecute(m_ctx, cmd, &ep);
    ++m_dispatchCount;

    m_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, count,
              after);
    if ((m_dispatchCount % 600) == 1 || LogLevel() >= 2)
        LOGI("XeSS evaluate %llu: input %ux%u (render %ux%u), jitter %.3f,%.3f, velocity scale %.2f,%.2f, exposure %.3f%s",
             (unsigned long long)m_dispatchCount, ep.inputWidth, ep.inputHeight, d.renderWidth, d.renderHeight,
             ep.jitterOffsetX, ep.jitterOffsetY, vx, vy, ep.exposureScale,
             (m_info.autoExposure && Config().exposure <= 0.0f) ? " (auto exposure)" : "");
    if (r != XESS_RESULT_SUCCESS)
    {
        LOGI("XeSS: xessVKExecute failed (%d)", r);
        return false;
    }
    return true;
}
}  // namespace

bool XessUnavailable()
{
    std::lock_guard<std::mutex> lock(g_xessMutex);
    return g_xessFailed;
}

std::unique_ptr<Upscaler> CreateUpscaler(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    if (info.backend == kBackendFsr3Vk)
    {
        if (auto fsr3 = CreateFsr4VkUpscaler(device, info))
            return fsr3;
        LOGI("FSR 3.1 unavailable; falling back to NSS");
    }
    if (info.backend == kBackendFsr4Vk)
    {
        if (auto native = CreateFsr4VkUpscaler(device, info))
            return native;
        // The interop way instead (layerSplit, set by the Vulkan front end's remap, keeps the Vulkan layer's split).
        UpscalerCreateInfo interop = info;
        interop.backend            = kBackendFsr4Interop;
        return CreateUpscaler(device, interop);
    }
    if (info.backend == kBackendFsr4Interop || info.backend == kBackendFsr4)
    {
        if (auto interop = CreateFsr4InteropUpscaler(device, info))
            return interop;
        LOGI("FSR 4 (%s) through D3D12 interop unavailable; falling back to NSS", Fsr4ModelName(info.fsr4Model));
    }
    if (info.backend == kBackendXess)
    {
        if (auto xess = XessUpscaler::Create(device, info))
            return xess;
        {
            std::lock_guard<std::mutex> lock(g_xessMutex);
            g_xessFailed = true;
        }
        LOGI("XeSS unavailable; falling back to NSS");
    }
    UpscalerCreateInfo nss = info;
    nss.backend            = kBackendNss;
    nss.fsr4Model          = kFsr4Auto;
    return NssUpscaler::Create(device, nss);
}
