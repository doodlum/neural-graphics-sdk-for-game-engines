// SPDX-FileCopyrightText: Copyright 2026 Neural Graphics SDK universal contributors
// SPDX-License-Identifier: MIT

// Deterministic replay harness for the NSS passes (native Vulkan, no window).
//
// Creates an NSS context through ffx-api exactly as an application would, feeds a fixed sequence of synthetic but
// realistic frames (gradients + noise, a depth field with discontinuities, a panning camera plus an independently
// moving object, the NSS jitter sequence, a history reset on frame 0 and in the middle) and reads back the upscaled
// output after every frame. The outputs are either saved as a baseline or compared bit for bit against one, which is
// the gate for every pre/post-process shader or scheduling change.
//
//   ffx_nss_replay_test --suite --save DIR     run the standard permutation matrix, write DIR/<config>.bin
//   ffx_nss_replay_test --suite --check DIR    run it again and compare every frame/pixel/channel with DIR
//   ffx_nss_replay_test --suite --compare DIR  error vs DIR (max/mean abs diff, PSNR) for non-exact variants (e.g. --fp16)
//   ffx_nss_replay_test --bench N [config]     run N frames back to back and print per-pass GPU medians
//
// Single config options: --quality 0|1|2 (QUALITY/BALANCED/PERFORMANCE), --render WxH, --display WxH, --hdr 0|1,
// --frames N, --out rgba16f|r11g11b10, --gpu N.

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_nss.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_VK(x)                                                                 \
    do                                                                              \
    {                                                                               \
        VkResult r_ = (x);                                                          \
        if (r_ != VK_SUCCESS)                                                       \
        {                                                                           \
            fprintf(stderr, "%s failed: %d (%s:%d)\n", #x, r_, __FILE__, __LINE__); \
            exit(2);                                                                \
        }                                                                           \
    } while (0)

namespace
{
    // ---------------------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------------------
    uint16_t FloatToHalf(float f)
    {
        uint32_t x;
        memcpy(&x, &f, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        int32_t        exp  = int32_t((x >> 23) & 0xFF) - 127 + 15;
        uint32_t       man  = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFF) == 0xFF)
            return uint16_t(sign | 0x7C00u | (man ? 0x200u : 0u));
        if (exp >= 31)
            return uint16_t(sign | 0x7C00u);
        if (exp <= 0)
        {
            if (exp < -10)
                return uint16_t(sign);
            man |= 0x800000u;
            const uint32_t shift = uint32_t(14 - exp);
            uint32_t       h     = man >> shift;
            const uint32_t rem   = man & ((1u << shift) - 1u);
            const uint32_t half  = 1u << (shift - 1);
            if (rem > half || (rem == half && (h & 1u)))
                ++h;
            return uint16_t(sign | h);
        }
        uint32_t       h   = (uint32_t(exp) << 10) | (man >> 13);
        const uint32_t rem = man & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
            ++h;
        return uint16_t(sign | h);
    }

    float HalfToFloat(uint16_t h)
    {
        const uint32_t sign = uint32_t(h & 0x8000u) << 16;
        uint32_t       exp  = (h >> 10) & 0x1Fu;
        uint32_t       man  = h & 0x3FFu;
        uint32_t       x;
        if (exp == 0x1F)
            x = sign | 0x7F800000u | (man << 13);
        else if (exp == 0)
        {
            if (man == 0)
                x = sign;
            else
            {
                exp = 113;
                while ((man & 0x400u) == 0)
                {
                    man <<= 1;
                    --exp;
                }
                x = sign | (exp << 23) | ((man & 0x3FFu) << 13);
            }
        }
        else
            x = sign | ((exp + 112) << 23) | (man << 13);
        float f;
        memcpy(&f, &x, 4);
        return f;
    }

    uint32_t Hash32(uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    float Noise(int x, int y, int seed)
    {
        return float(Hash32(uint32_t(x) * 1973u + uint32_t(y) * 9277u + uint32_t(seed) * 26699u) & 0xFFFFu) / 65535.0f;
    }

    uint64_t Fnv1a(const uint8_t* p, size_t n)
    {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Vulkan context
    // ---------------------------------------------------------------------------------------------------------------
    struct Image
    {
        VkImage        image  = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkFormat       format = VK_FORMAT_UNDEFINED;
        VkImageUsageFlags usage = 0;
        uint32_t       width = 0, height = 0;
    };

    struct Buffer
    {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void*          mapped = nullptr;
        VkDeviceSize   size   = 0;
    };

    struct Context
    {
        VkInstance                       instance = VK_NULL_HANDLE;
        VkPhysicalDevice                 physical = VK_NULL_HANDLE;
        VkDevice                         device   = VK_NULL_HANDLE;
        VkQueue                          queue    = VK_NULL_HANDLE;
        uint32_t                         queueFamily = 0;
        VkCommandPool                    pool = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties memory = {};
        uint32_t                         mlFeatures = 0;

        uint32_t MemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const
        {
            for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
                if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags)
                    return i;
            fprintf(stderr, "no memory type\n");
            exit(2);
        }

        Image CreateImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage)
        {
            Image img;
            img.width = w;
            img.height = h;
            img.format = format;
            img.usage  = usage;
            VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType         = VK_IMAGE_TYPE_2D;
            info.format            = format;
            info.extent            = {w, h, 1};
            info.mipLevels         = 1;
            info.arrayLayers       = 1;
            info.samples           = VK_SAMPLE_COUNT_1_BIT;
            info.tiling            = VK_IMAGE_TILING_OPTIMAL;
            info.usage             = usage;
            info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            CHECK_VK(vkCreateImage(device, &info, nullptr, &img.image));
            VkMemoryRequirements req;
            vkGetImageMemoryRequirements(device, img.image, &req);
            VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            alloc.allocationSize       = req.size;
            alloc.memoryTypeIndex      = MemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            CHECK_VK(vkAllocateMemory(device, &alloc, nullptr, &img.memory));
            CHECK_VK(vkBindImageMemory(device, img.image, img.memory, 0));
            return img;
        }

        void Destroy(Image& img)
        {
            vkDestroyImage(device, img.image, nullptr);
            vkFreeMemory(device, img.memory, nullptr);
            img = {};
        }

        Buffer CreateHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage)
        {
            Buffer b;
            b.size                  = size;
            VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size               = size;
            info.usage              = usage;
            CHECK_VK(vkCreateBuffer(device, &info, nullptr, &b.buffer));
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, b.buffer, &req);
            VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            alloc.allocationSize       = req.size;
            alloc.memoryTypeIndex =
                MemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            CHECK_VK(vkAllocateMemory(device, &alloc, nullptr, &b.memory));
            CHECK_VK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
            CHECK_VK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
            return b;
        }

        void Destroy(Buffer& b)
        {
            vkUnmapMemory(device, b.memory);
            vkDestroyBuffer(device, b.buffer, nullptr);
            vkFreeMemory(device, b.memory, nullptr);
            b = {};
        }

        VkCommandBuffer AllocCmd()
        {
            VkCommandBufferAllocateInfo info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            info.commandPool                 = pool;
            info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount          = 1;
            VkCommandBuffer cmd;
            CHECK_VK(vkAllocateCommandBuffers(device, &info, &cmd));
            return cmd;
        }

        void Begin(VkCommandBuffer cmd)
        {
            CHECK_VK(vkResetCommandBuffer(cmd, 0));
            VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK_VK(vkBeginCommandBuffer(cmd, &begin));
        }

        void Submit(VkCommandBuffer cmd, VkFence fence)
        {
            CHECK_VK(vkEndCommandBuffer(cmd));
            VkSubmitInfo submit       = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers    = &cmd;
            CHECK_VK(vkQueueSubmit(queue, 1, &submit, fence));
        }
    };

    void ImageBarrier(VkCommandBuffer       cmd,
                      const Image&          img,
                      VkImageLayout         oldLayout,
                      VkImageLayout         newLayout,
                      VkAccessFlags         srcAccess,
                      VkAccessFlags         dstAccess,
                      VkPipelineStageFlags  srcStage,
                      VkPipelineStageFlags  dstStage)
    {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask        = srcAccess;
        b.dstAccessMask        = dstAccess;
        b.oldLayout            = oldLayout;
        b.newLayout            = newLayout;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image                                       = img.image;
        b.subresourceRange                            = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    void CreateContext(Context& ctx, int gpuIndex)
    {
        VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName  = "ffx_nss_replay_test";
        app.apiVersion        = VK_API_VERSION_1_3;
        VkInstanceCreateInfo instanceInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo     = &app;
        CHECK_VK(vkCreateInstance(&instanceInfo, nullptr, &ctx.instance));

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr);
        std::vector<VkPhysicalDevice> gpus(count);
        vkEnumeratePhysicalDevices(ctx.instance, &count, gpus.data());
        if (gpuIndex >= int(count))
        {
            fprintf(stderr, "GPU %d not found (%u available)\n", gpuIndex, count);
            exit(2);
        }
        ctx.physical = gpus[gpuIndex];
        vkGetPhysicalDeviceMemoryProperties(ctx.physical, &ctx.memory);

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &familyCount, families.data());
        for (uint32_t i = 0; i < familyCount; ++i)
        {
            if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            {
                ctx.queueFamily = i;
                break;
            }
        }

        // Like a game device: every core feature the GPU supports, plus the ML extensions the SDK asks for.
        uint32_t extensionCount = 8;
        const char* extensions[8] = {};
        ctx.mlFeatures = ffxQueryMlDeviceFeaturesVK(ctx.instance, ctx.physical, vkGetInstanceProcAddr, extensions, &extensionCount);
        extensionCount = std::min(extensionCount, 8u);

        VkPhysicalDeviceVulkan13Features v13       = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features v12       = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features v11       = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2        features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext                            = &v11;
        v11.pNext                                  = &v12;
        v12.pNext                                  = &v13;
        vkGetPhysicalDeviceFeatures2(ctx.physical, &features2);
        v11.protectedMemory = VK_FALSE;

        const float             priority  = 1.0f;
        VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, ctx.queueFamily, 1, &priority};
        VkDeviceCreateInfo      deviceInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.pNext                   = &features2;
        deviceInfo.queueCreateInfoCount    = 1;
        deviceInfo.pQueueCreateInfos       = &queueInfo;
        deviceInfo.enabledExtensionCount   = extensionCount;
        deviceInfo.ppEnabledExtensionNames = extensions;
        CHECK_VK(vkCreateDevice(ctx.physical, &deviceInfo, nullptr, &ctx.device));
        vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex        = ctx.queueFamily;
        CHECK_VK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.pool));

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx.physical, &props);
        printf("GPU %d: %s, ML features 0x%03x\n", gpuIndex, props.deviceName, ctx.mlFeatures);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Configuration and synthetic scene
    // ---------------------------------------------------------------------------------------------------------------
    bool g_allow16bit = false;  // --fp16: FFX_API_NSS_CONTEXT_FLAG_ALLOW_16BIT (fp16 shader permutations, not bit exact)

    struct Config
    {
        int      quality       = FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE;
        uint32_t renderW       = 640, renderH = 400;
        uint32_t displayW      = 1280, displayH = 800;
        bool     hdr           = false;
        int      frames        = 16;
        bool     outR11G11B10  = false;

        std::string Name() const
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "q%d_%ux%u_%ux%u_hdr%d_%s", quality, renderW, renderH, displayW, displayH, hdr ? 1 : 0, outR11G11B10 ? "r11g11b10" : "rgba16f");
            return buf;
        }
    };

    constexpr int kInputSets = 16;  // distinct input frames; longer runs cycle through them

    struct FrameInputs
    {
        Image color, depth, motion;
        float jitterX = 0, jitterY = 0;
    };

    // Scene in render-pixel units: a textured background panning with the camera, a few depth "boxes" fixed to the
    // background, and a round object moving on its own. Reversed infinite depth (device depth = near / viewZ).
    void GenerateFrame(const Config& cfg, int frame, float jx, float jy, std::vector<uint16_t>& color, std::vector<float>& depth, std::vector<uint16_t>& motion)
    {
        const int   W = int(cfg.renderW), H = int(cfg.renderH);
        const float sx = W / 640.0f, sy = H / 400.0f;  // keep the scene the same size in screen space
        const float panX = 0.75f * sx, panY = -0.35f * sy;
        const float objVX = 3.2f * sx, objVY = 1.4f * sy;
        const float camX = panX * frame, camY = panY * frame;
        const float objCX = (180.0f + objVX * frame) * sx, objCY = (150.0f + objVY * frame) * sy, objR = 55.0f * sx;
        const float range = cfg.hdr ? 6.0f : 1.0f;
        const float nearZ = 0.1f;

        color.resize(size_t(W) * H * 4);
        depth.resize(size_t(W) * H);
        motion.resize(size_t(W) * H * 2);
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                const float px = x + 0.5f - jx, py = y + 0.5f - jy;  // jittered sample position
                const size_t i = size_t(y) * W + x;
                float        r, g, b, z, mvx, mvy;

                const float dxo = px - objCX, dyo = py - objCY;
                if (dxo * dxo + dyo * dyo < objR * objR)
                {
                    // Moving object: object-space texture, shading, closer depth.
                    const float ox = dxo / objR, oy = dyo / objR;
                    const float shade = 0.55f + 0.45f * (1.0f - (ox * ox + oy * oy));
                    const bool  check = ((int(std::floor(dxo / 6.0f)) + int(std::floor(dyo / 6.0f))) & 1) != 0;
                    r = shade * (check ? 0.9f : 0.25f);
                    g = shade * 0.35f;
                    b = shade * (check ? 0.2f : 0.8f);
                    if (cfg.hdr && ox * ox + oy * oy < 0.08f)
                        r = g = b = range;  // specular highlight
                    z   = 3.0f + 0.5f * (ox * ox + oy * oy);
                    mvx = -objVX;
                    mvy = -objVY;
                }
                else
                {
                    // Background in world space: smooth gradients, stripes, noise; boxes nearer than the floor.
                    const float wx = px + camX, wy = py + camY;
                    const int   bx = int(std::floor(wx / (90.0f * sx))), by = int(std::floor(wy / (70.0f * sy)));
                    const bool  box = ((bx * 7 + by * 3) % 5) == 0;
                    const float n = Noise(int(std::floor(wx)), int(std::floor(wy)), 1);
                    const float stripes = 0.5f + 0.5f * std::sin(wx * 0.35f + wy * 0.12f);
                    r = 0.15f + 0.6f * (wx / (W * 1.5f)) + 0.1f * stripes + 0.08f * n;
                    g = 0.2f + 0.5f * (wy / (H * 1.5f)) + 0.05f * n;
                    b = 0.35f + 0.25f * std::sin(wx * 0.01f) + 0.12f * stripes * n;
                    z = 8.0f + 40.0f * (1.0f - py / H);
                    if (box)
                    {
                        r = 0.7f * r + 0.25f;
                        g *= 0.6f;
                        b = 0.9f - 0.5f * b;
                        z = 5.0f + 0.3f * float((bx + by) & 3);
                    }
                    if (cfg.hdr && Noise(int(wx / 3), int(wy / 3), 7) > 0.995f)
                        r = g = b = range * 2.0f;  // sparse HDR sparkles
                    mvx = -panX;
                    mvy = -panY;
                }
                color[i * 4 + 0] = FloatToHalf(std::max(r, 0.0f) * (cfg.hdr ? range * 0.5f : 1.0f));
                color[i * 4 + 1] = FloatToHalf(std::max(g, 0.0f) * (cfg.hdr ? range * 0.5f : 1.0f));
                color[i * 4 + 2] = FloatToHalf(std::max(b, 0.0f) * (cfg.hdr ? range * 0.5f : 1.0f));
                color[i * 4 + 3] = FloatToHalf(1.0f);
                depth[i]         = nearZ / z;
                motion[i * 2 + 0] = FloatToHalf(mvx);
                motion[i * 2 + 1] = FloatToHalf(mvy);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // One NSS run
    // ---------------------------------------------------------------------------------------------------------------
    struct Runner
    {
        Context&                ctx;
        Config                  cfg;
        ffxContext              nss = nullptr;
        FrameInputs             inputs[kInputSets];
        Image                   output;
        Buffer                  readback;
        uint32_t                outBpp = 8;
        static constexpr int    kRing = 3;
        VkCommandBuffer         cmds[kRing];
        VkFence                 fences[kRing];

        Runner(Context& c, const Config& config) : ctx(c), cfg(config) {}

        static void Message(uint32_t type, const char* message)
        {
            if (type == FFX_API_MESSAGE_TYPE_ERROR || getenv("NSS_REPLAY_VERBOSE"))
                fprintf(stderr, "nss: [%u] %s\n", type, message ? message : "");
        }

        void Init()
        {
            ffxCreateBackendVKDesc backend = {};
            backend.header.type            = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
            backend.vkDevice               = ctx.device;
            backend.vkPhysicalDevice       = ctx.physical;
            backend.vkInstance             = ctx.instance;
            backend.vkDeviceProcAddr       = vkGetDeviceProcAddr;
            backend.vkGetInstanceProcAddr  = vkGetInstanceProcAddr;

            ffxCreateBackendVKMlFeaturesDesc ml = {};
            ml.header.type                      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_ML_FEATURES;
            ml.enabledFeatures                  = ctx.mlFeatures;

            // The flags the Steam Deck DLSS proxy uses (steamdeck/nvngx/nss_upscaler.cpp).
            ffxApiCreateContextDescNss desc = {};
            desc.header.type                = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
            desc.flags = FFX_API_NSS_CONTEXT_FLAG_QUANTIZED | FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY | FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE |
                         FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED | (cfg.hdr ? FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE : 0u) |
                         (g_allow16bit ? FFX_API_NSS_CONTEXT_FLAG_ALLOW_16BIT : 0u);
            desc.maxRenderSize  = {cfg.renderW, cfg.renderH};
            desc.maxUpscaleSize = {cfg.displayW, cfg.displayH};
            desc.qualityMode    = FfxApiNssShaderQualityMode(cfg.quality);
            desc.fpMessage      = Message;
            desc.header.pNext   = &backend.header;
            backend.header.pNext = &ml.header;
            if (ffxCreateContext(&nss, &desc.header, nullptr) != FFX_API_RETURN_OK)
            {
                fprintf(stderr, "ffxCreateContext(NSS) failed\n");
                exit(2);
            }

            int32_t                               phaseCount = 0;
            ffxApiQueryDescNssGetJitterPhaseCount phaseQuery = {};
            phaseQuery.header.type                           = FFX_API_QUERY_DESC_TYPE_NSS_GETJITTERPHASECOUNT;
            phaseQuery.renderWidth                           = cfg.renderW;
            phaseQuery.displayWidth                          = cfg.displayW;
            phaseQuery.pOutPhaseCount                        = &phaseCount;
            ffxQuery(&nss, &phaseQuery.header);

            const VkImageUsageFlags inUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            const size_t            pixels  = size_t(cfg.renderW) * cfg.renderH;
            const size_t            setSize = pixels * (8 + 4 + 4);
            Buffer                  staging = ctx.CreateHostBuffer(setSize * kInputSets, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            VkCommandBuffer         cmd     = ctx.AllocCmd();
            ctx.Begin(cmd);
            std::vector<uint16_t> color, motion;
            std::vector<float>    depth;
            for (int f = 0; f < kInputSets; ++f)
            {
                FrameInputs& in = inputs[f];
                ffxApiQueryDescNssGetJitterOffset jq = {};
                jq.header.type                      = FFX_API_QUERY_DESC_TYPE_NSS_GETJITTEROFFSET;
                jq.index                            = f;
                jq.phaseCount                       = std::max(phaseCount, 1);
                jq.pOutX                            = &in.jitterX;
                jq.pOutY                            = &in.jitterY;
                ffxQuery(&nss, &jq.header);

                GenerateFrame(cfg, f, in.jitterX, in.jitterY, color, depth, motion);
                in.color  = ctx.CreateImage(cfg.renderW, cfg.renderH, VK_FORMAT_R16G16B16A16_SFLOAT, inUsage);
                in.depth  = ctx.CreateImage(cfg.renderW, cfg.renderH, VK_FORMAT_R32_SFLOAT, inUsage);
                in.motion = ctx.CreateImage(cfg.renderW, cfg.renderH, VK_FORMAT_R16G16_SFLOAT, inUsage);

                uint8_t*     base   = static_cast<uint8_t*>(staging.mapped) + setSize * f;
                VkDeviceSize offset = setSize * f;
                memcpy(base, color.data(), pixels * 8);
                memcpy(base + pixels * 8, depth.data(), pixels * 4);
                memcpy(base + pixels * 12, motion.data(), pixels * 4);
                const VkDeviceSize offsets[3] = {offset, offset + pixels * 8, offset + pixels * 12};
                const Image*       images[3]  = {&in.color, &in.depth, &in.motion};
                for (int k = 0; k < 3; ++k)
                {
                    ImageBarrier(cmd, *images[k], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                    VkBufferImageCopy region = {};
                    region.bufferOffset      = offsets[k];
                    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.imageExtent       = {cfg.renderW, cfg.renderH, 1};
                    vkCmdCopyBufferToImage(cmd, staging.buffer, images[k]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                    ImageBarrier(cmd, *images[k], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                }
            }

            outBpp = cfg.outR11G11B10 ? 4 : 8;
            output = ctx.CreateImage(cfg.displayW,
                                     cfg.displayH,
                                     cfg.outR11G11B10 ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : VK_FORMAT_R16G16B16A16_SFLOAT,
                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            ImageBarrier(cmd, output, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            readback = ctx.CreateHostBuffer(VkDeviceSize(cfg.displayW) * cfg.displayH * outBpp, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

            ctx.Submit(cmd, VK_NULL_HANDLE);
            CHECK_VK(vkQueueWaitIdle(ctx.queue));
            vkFreeCommandBuffers(ctx.device, ctx.pool, 1, &cmd);
            ctx.Destroy(staging);

            for (int i = 0; i < kRing; ++i)
            {
                cmds[i]                     = ctx.AllocCmd();
                VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                fenceInfo.flags             = VK_FENCE_CREATE_SIGNALED_BIT;
                CHECK_VK(vkCreateFence(ctx.device, &fenceInfo, nullptr, &fences[i]));
            }
        }

        static FfxApiResource Wrap(const Image& img, uint32_t usage, uint32_t state)
        {
            VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType         = VK_IMAGE_TYPE_2D;
            info.format            = img.format;
            info.extent            = {img.width, img.height, 1};
            info.mipLevels         = 1;
            info.arrayLayers       = 1;
            info.usage             = img.usage;
            return ffxApiGetResourceVK(reinterpret_cast<void*>(img.image), ffxApiGetImageResourceDescriptionVK(img.image, info, usage), state);
        }

        void RecordFrame(VkCommandBuffer cmd, int frame, bool reset, bool readOut)
        {
            const FrameInputs& in = inputs[frame % kInputSets];

            ffxApiDispatchDescNss d = {};
            d.header.type           = FFX_API_DISPATCH_DESC_TYPE_NSS;
            d.commandList           = cmd;
            d.color                 = Wrap(in.color, 0, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            d.depth                 = Wrap(in.depth, 0, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            d.motionVectors         = Wrap(in.motion, 0, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            d.output                = Wrap(output, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            d.jitterOffset          = {in.jitterX, in.jitterY};
            d.motionVectorScale     = {1.0f, 1.0f};
            d.renderSize            = {cfg.renderW, cfg.renderH};
            d.upscaleSize           = {cfg.displayW, cfg.displayH};
            d.cameraNear            = 0.1f;
            d.cameraFar             = 100000.0f;
            d.cameraFovAngleVertical = 60.0f * 3.14159265f / 180.0f;
            d.exposure              = cfg.hdr ? 0.5f : 1.0f;
            d.frameTimeDelta        = 16.6f;
            d.reset                 = reset;
            if (ffxDispatch(&nss, &d.header) != FFX_API_RETURN_OK)
            {
                fprintf(stderr, "ffxDispatch(NSS) failed\n");
                exit(2);
            }

            if (readOut)
            {
                ImageBarrier(cmd, output, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                VkBufferImageCopy region = {};
                region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent       = {cfg.displayW, cfg.displayH, 1};
                vkCmdCopyImageToBuffer(cmd, output.image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &region);
                ImageBarrier(cmd, output, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }
        }

        // Runs cfg.frames frames one by one and returns all outputs concatenated.
        std::vector<uint8_t> RunReplay()
        {
            const size_t         frameBytes = size_t(cfg.displayW) * cfg.displayH * outBpp;
            std::vector<uint8_t> all(frameBytes * cfg.frames);
            for (int f = 0; f < cfg.frames; ++f)
            {
                CHECK_VK(vkWaitForFences(ctx.device, 1, &fences[0], VK_TRUE, UINT64_MAX));
                CHECK_VK(vkResetFences(ctx.device, 1, &fences[0]));
                ctx.Begin(cmds[0]);
                RecordFrame(cmds[0], f, f == 0 || f == cfg.frames / 2 + 1, true);
                ctx.Submit(cmds[0], fences[0]);
                CHECK_VK(vkWaitForFences(ctx.device, 1, &fences[0], VK_TRUE, UINT64_MAX));
                memcpy(all.data() + frameBytes * f, readback.mapped, frameBytes);
            }
            return all;
        }

        void RunBench(int frames)
        {
            for (int f = 0; f < frames; ++f)
            {
                const int slot = f % kRing;
                CHECK_VK(vkWaitForFences(ctx.device, 1, &fences[slot], VK_TRUE, UINT64_MAX));
                CHECK_VK(vkResetFences(ctx.device, 1, &fences[slot]));
                ctx.Begin(cmds[slot]);
                RecordFrame(cmds[slot], f, f == 0, false);
                ctx.Submit(cmds[slot], fences[slot]);
            }
            CHECK_VK(vkQueueWaitIdle(ctx.queue));
        }

        void Destroy()
        {
            CHECK_VK(vkQueueWaitIdle(ctx.queue));
            ffxDestroyContext(&nss, nullptr);
            for (int i = 0; i < kRing; ++i)
            {
                vkDestroyFence(ctx.device, fences[i], nullptr);
                vkFreeCommandBuffers(ctx.device, ctx.pool, 1, &cmds[i]);
            }
            for (FrameInputs& in : inputs)
            {
                ctx.Destroy(in.color);
                ctx.Destroy(in.depth);
                ctx.Destroy(in.motion);
            }
            ctx.Destroy(output);
            ctx.Destroy(readback);
        }
    };

    // ---------------------------------------------------------------------------------------------------------------
    // Baseline files
    // ---------------------------------------------------------------------------------------------------------------
    struct DumpHeader
    {
        char     magic[8] = {'N', 'S', 'S', 'R', 'P', 'L', 'Y', '1'};
        uint32_t width = 0, height = 0, bpp = 0, frames = 0;
    };

    bool SaveDump(const std::string& path, const Config& cfg, uint32_t bpp, const std::vector<uint8_t>& data)
    {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f)
            return false;
        DumpHeader h;
        h.width  = cfg.displayW;
        h.height = cfg.displayH;
        h.bpp    = bpp;
        h.frames = uint32_t(cfg.frames);
        fwrite(&h, sizeof(h), 1, f);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        return true;
    }

    bool LoadDump(const std::string& path, DumpHeader& h, std::vector<uint8_t>& data)
    {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f)
            return false;
        if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, "NSSRPLY1", 8) != 0)
        {
            fclose(f);
            return false;
        }
        data.resize(size_t(h.width) * h.height * h.bpp * h.frames);
        const bool ok = fread(data.data(), 1, data.size(), f) == data.size();
        fclose(f);
        return ok;
    }

    // Returns true when identical.
    bool Compare(const Config& cfg, uint32_t bpp, const std::vector<uint8_t>& got, const std::vector<uint8_t>& ref)
    {
        const size_t frameBytes = size_t(cfg.displayW) * cfg.displayH * bpp;
        bool         identical  = true;
        for (int f = 0; f < cfg.frames; ++f)
        {
            const uint8_t* a = got.data() + frameBytes * f;
            const uint8_t* b = ref.data() + frameBytes * f;
            if (memcmp(a, b, frameBytes) == 0)
                continue;
            identical       = false;
            size_t diffs    = 0;
            size_t firstPx  = SIZE_MAX;
            for (size_t p = 0; p < size_t(cfg.displayW) * cfg.displayH; ++p)
            {
                if (memcmp(a + p * bpp, b + p * bpp, bpp) != 0)
                {
                    ++diffs;
                    if (firstPx == SIZE_MAX)
                        firstPx = p;
                }
            }
            printf("    frame %2d: %zu pixels differ (first at %zu,%zu)\n", f, diffs, firstPx % cfg.displayW, firstPx / cfg.displayW);
        }
        return identical;
    }

    // --compare: error of an RGBA16F run against an (exact) baseline, RGB channels. PSNR uses peak 1 for SDR and the
    // reference maximum of the compared frames for HDR (the HDR scene goes well above 1).
    struct ErrorStats
    {
        double maxDiff = 0, sumDiff = 0, sumSq = 0, peak = 0;
        size_t count = 0, nonFinite = 0;
        void   Add(float a, float b)
        {
            if (!std::isfinite(a) || !std::isfinite(b))
            {
                ++nonFinite;
                return;
            }
            const double d = std::fabs(double(a) - double(b));
            maxDiff        = std::max(maxDiff, d);
            sumDiff += d;
            sumSq += d * d;
            peak = std::max(peak, double(std::fabs(b)));
            ++count;
        }
        double Psnr(bool hdr) const
        {
            const double mse  = count ? sumSq / double(count) : 0.0;
            const double pk   = hdr ? std::max(peak, 1.0) : 1.0;
            return mse > 0 ? 10.0 * std::log10(pk * pk / mse) : INFINITY;
        }
        void Print(const char* label, bool hdr) const
        {
            printf("    %-10s max %.5f  mean %.6f  PSNR %6.2f dB%s", label, maxDiff, count ? sumDiff / double(count) : 0.0, Psnr(hdr),
                   nonFinite ? "  (non-finite values!)" : "");
            if (hdr)
                printf("  (peak %.2f)", std::max(peak, 1.0));
            printf("\n");
        }
    };

    void CompareError(const Config& cfg, uint32_t bpp, const std::vector<uint8_t>& got, const std::vector<uint8_t>& ref)
    {
        if (bpp != 8)
        {
            printf("    (compare needs the RGBA16F output)\n");
            return;
        }
        const size_t pixels = size_t(cfg.displayW) * cfg.displayH;
        ErrorStats   all, frame16, last;
        for (int f = 0; f < cfg.frames; ++f)
        {
            const uint16_t* a = reinterpret_cast<const uint16_t*>(got.data()) + pixels * 4 * f;
            const uint16_t* b = reinterpret_cast<const uint16_t*>(ref.data()) + pixels * 4 * f;
            for (size_t p = 0; p < pixels; ++p)
                for (int c = 0; c < 3; ++c)
                {
                    const float x = HalfToFloat(a[p * 4 + c]), y = HalfToFloat(b[p * 4 + c]);
                    all.Add(x, y);
                    if (f == 15)
                        frame16.Add(x, y);
                    if (f == cfg.frames - 1)
                        last.Add(x, y);
                }
        }
        all.Print("all frames", cfg.hdr);
        if (cfg.frames > 16)
            frame16.Print("frame 16", cfg.hdr);
        char label[32];
        snprintf(label, sizeof(label), "frame %d", cfg.frames);
        last.Print(label, cfg.hdr);
    }

    std::vector<Config> SuiteConfigs(int frames)
    {
        // Every permutation the Steam Deck uses: BALANCED/PERFORMANCE (mid_low network) and QUALITY (high network, below), exact 2x and a non-2x ratio,
        // HDR flag on and off. MANAGE_HISTORY, reversed + infinite depth and the compute path are always on.
        std::vector<Config> configs;
        const uint32_t      sizes[2][2] = {{640, 400}, {742, 464}};
        for (int q : {FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED, FFX_API_NSS_SHADER_QUALITY_MODE_PERFORMANCE})
            for (auto& s : sizes)
                for (bool hdr : {false, true})
                {
                    Config c;
                    c.quality = q;
                    c.renderW = s[0];
                    c.renderH = s[1];
                    c.hdr     = hdr;
                    c.frames  = frames;
                    configs.push_back(c);
                }
        // The output written through the r11g11b10 store path as well (what the history uses).
        Config c       = configs[0];
        c.outR11G11B10 = true;
        configs.push_back(c);
        // QUALITY (full network, the Steam Deck proxy default): Performance and Quality render sizes -> 1280x800.
        const uint32_t fullSizes[2][2] = {{640, 400}, {853, 533}};
        for (auto& s : fullSizes)
            for (bool hdr : {false, true})
            {
                Config q;
                q.quality = FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY;
                q.renderW = s[0];
                q.renderH = s[1];
                q.hdr     = hdr;
                q.frames  = frames;
                configs.push_back(q);
            }
        return configs;
    }

    bool ParseSize(const char* s, uint32_t& w, uint32_t& h)
    {
        return sscanf(s, "%ux%u", &w, &h) == 2;
    }
}  // namespace

int main(int argc, char** argv)
{
    Config      cfg;
    int         gpu = 0, bench = 0;
    bool        suite = false;
    std::string saveDir, checkDir, compareDir;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a    = argv[i];
        const char*       next = i + 1 < argc ? argv[i + 1] : nullptr;
        if (a == "--gpu" && next)
            gpu = atoi(argv[++i]);
        else if (a == "--quality" && next)
            cfg.quality = atoi(argv[++i]);
        else if (a == "--render" && next)
            ParseSize(argv[++i], cfg.renderW, cfg.renderH);
        else if (a == "--display" && next)
            ParseSize(argv[++i], cfg.displayW, cfg.displayH);
        else if (a == "--hdr" && next)
            cfg.hdr = atoi(argv[++i]) != 0;
        else if (a == "--frames" && next)
            cfg.frames = atoi(argv[++i]);
        else if (a == "--out" && next)
            cfg.outR11G11B10 = std::string(argv[++i]) == "r11g11b10";
        else if (a == "--bench" && next)
            bench = atoi(argv[++i]);
        else if (a == "--suite")
            suite = true;
        else if (a == "--save" && next)
            saveDir = argv[++i];
        else if (a == "--check" && next)
            checkDir = argv[++i];
        else if (a == "--compare" && next)
            compareDir = argv[++i];
        else if (a == "--fp16")
            g_allow16bit = true;
        else
        {
            fprintf(stderr,
                    "usage: %s [--suite] [--save DIR | --check DIR | --compare DIR] [--bench N] [--quality 0|1|2] [--render WxH] [--display WxH]\n"
                    "          [--hdr 0|1] [--frames N] [--out rgba16f|r11g11b10] [--fp16] [--gpu N]\n",
                    argv[0]);
            return 2;
        }
    }

    // Per-pass GPU timings come from the backend's FFX_PROFILE_PASSES profiler (medians every 250 executions).
    std::string profilePath;
    if (bench > 0)
    {
        if (const char* p = getenv("FFX_PROFILE_PASSES"))
            profilePath = p;
        else
        {
            profilePath = "/tmp/ffx_nss_replay_profile.txt";
            setenv("FFX_PROFILE_PASSES", profilePath.c_str(), 1);
        }
        remove(profilePath.c_str());
    }

    Context ctx;
    CreateContext(ctx, gpu);

    if (bench > 0)
    {
        Runner r(ctx, cfg);
        r.Init();
        printf("bench %s, %d frames\n", cfg.Name().c_str(), bench);
        r.RunBench(bench);
        r.Destroy();
        FILE* f = fopen(profilePath.c_str(), "r");
        if (!f)
        {
            printf("no profile written (need >= 250 frames)\n");
            return 1;
        }
        char line[256];
        while (fgets(line, sizeof(line), f))
            fputs(line, stdout);
        fclose(f);
        return 0;
    }

    const std::vector<Config> configs = suite ? SuiteConfigs(cfg.frames) : std::vector<Config>{cfg};
    int                       failures = 0;
    for (const Config& c : configs)
    {
        Runner r(ctx, c);
        r.Init();
        const std::vector<uint8_t> out = r.RunReplay();
        const uint32_t             bpp = r.outBpp;
        r.Destroy();

        const size_t frameBytes = size_t(c.displayW) * c.displayH * bpp;
        uint64_t     total      = Fnv1a(out.data(), out.size());
        printf("%-40s hash %016" PRIx64 " (last frame %016" PRIx64 ")", c.Name().c_str(), total, Fnv1a(out.data() + frameBytes * (c.frames - 1), frameBytes));

        if (!saveDir.empty())
        {
            const std::string path = saveDir + "/" + c.Name() + ".bin";
            printf(SaveDump(path, c, bpp, out) ? "  saved\n" : "  SAVE FAILED\n");
        }
        else if (!checkDir.empty())
        {
            DumpHeader           h;
            std::vector<uint8_t> ref;
            const std::string    path = checkDir + "/" + c.Name() + ".bin";
            if (!LoadDump(path, h, ref) || h.frames != uint32_t(c.frames) || h.bpp != bpp)
            {
                printf("  NO BASELINE (%s)\n", path.c_str());
                ++failures;
                continue;
            }
            if (Compare(c, bpp, out, ref))
                printf("  IDENTICAL\n");
            else
            {
                printf("  DIFFERENT\n");
                ++failures;
            }
        }
        else if (!compareDir.empty())
        {
            DumpHeader           h;
            std::vector<uint8_t> ref;
            const std::string    path = compareDir + "/" + c.Name() + ".bin";
            if (!LoadDump(path, h, ref) || h.frames != uint32_t(c.frames) || h.bpp != bpp)
            {
                printf("  NO BASELINE (%s)\n", path.c_str());
                continue;
            }
            printf(memcmp(out.data(), ref.data(), out.size()) == 0 ? "  IDENTICAL\n" : "\n");
            CompareError(c, bpp, out, ref);
        }
        else
            printf("\n");
    }

    vkDestroyCommandPool(ctx.device, ctx.pool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
    if (!checkDir.empty())
        printf(failures ? "FAILED: %d config(s) differ\n" : "ALL IDENTICAL\n", failures);
    return failures ? 1 : 0;
}
