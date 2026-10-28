// AISS_DUMP_FRAME=N for the upscalers other than NSS (NSS writes its own): copies the output of evaluation N to a
// host-visible buffer, reads it back 16 evaluations later (the proxy never waits on the game's queue) and writes
// aiss_frame_<N>.ppm next to the game, tone mapped like NSS's dump. AISS_DUMP_FRAME_COUNT=K writes evaluations N..N+K-1
// (consecutive frames, e.g. to measure temporal stability).
// SPDX-License-Identifier: MIT
#include "ngx_common.h"

#include "common.h"

#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace
{
struct Dump
{
    VulkanDevice   dev;
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    UpscalerImage  image;
    uint64_t       readAt = 0;
    uint64_t       index  = 0;  // evaluation index the dump holds
};

std::mutex g_mutex;
uint64_t   g_count = 0;  // evaluations of non-NSS upscalers
std::vector<Dump> g_dumps;

int DumpCount()
{
    static const int n = [] {
        const char* v = getenv("AISS_DUMP_FRAME_COUNT");
        return v && atoi(v) > 0 ? atoi(v) : 1;
    }();
    return n;
}

float HalfBits(uint16_t h)
{
    const uint32_t e = (h >> 10) & 31, m = h & 1023;
    const float    v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? 65504.0f : std::ldexp(float(m | 1024), int(e) - 25);
    return (h & 0x8000) ? -v : v;
}

float SmallFloat(uint32_t bits, int mantissa)
{
    const uint32_t e = bits >> mantissa, m = bits & ((1u << mantissa) - 1);
    return e == 0 ? std::ldexp(float(m), -14 - mantissa) : std::ldexp(float(m | (1u << mantissa)), int(e) - 15 - mantissa);
}

uint8_t ToByte(float x)
{
    x = x < 0 ? 0 : x / (1.0f + x);
    x = x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1 / 2.4f) - 0.055f;
    return uint8_t(x * 255.0f + 0.5f);
}

void Write(Dump& d)
{
    auto fn = [&](const char* name) { return d.dev.gdpa(d.dev.device, name); };
    void* ptr = nullptr;
    if (reinterpret_cast<PFN_vkMapMemory>(fn("vkMapMemory"))(d.dev.device, d.memory, 0, VK_WHOLE_SIZE, 0, &ptr) == VK_SUCCESS)
    {
        char name[64];
        snprintf(name, sizeof(name), "aiss_frame_%llu.ppm", (unsigned long long)d.index);
        if (FILE* f = fopen(name, "wb"))
        {
            const uint32_t w = d.image.width, h = d.image.height;
            fprintf(f, "P6\n%u %u\n255\n", w, h);
            for (uint32_t i = 0; i < w * h; ++i)
            {
                uint8_t out[3];
                if (d.image.format == VK_FORMAT_R16G16B16A16_SFLOAT)
                {
                    const uint16_t* p = static_cast<const uint16_t*>(ptr) + i * 4;
                    for (int c = 0; c < 3; ++c)
                        out[c] = ToByte(HalfBits(p[c]));
                }
                else if (d.image.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
                {
                    const uint32_t p = static_cast<const uint32_t*>(ptr)[i];
                    out[0] = ToByte(SmallFloat(p & 0x7FF, 6));
                    out[1] = ToByte(SmallFloat((p >> 11) & 0x7FF, 6));
                    out[2] = ToByte(SmallFloat(p >> 22, 5));
                }
                else
                {
                    const uint8_t* p    = static_cast<const uint8_t*>(ptr) + i * 4;
                    const bool     bgra = d.image.format == VK_FORMAT_B8G8R8A8_UNORM || d.image.format == VK_FORMAT_B8G8R8A8_SRGB;
                    out[0] = p[bgra ? 2 : 0], out[1] = p[1], out[2] = p[bgra ? 0 : 2];
                }
                fwrite(out, 1, 3, f);
            }
            fclose(f);
            LOGI("wrote %s (%ux%u, format %d)", name, w, h, d.image.format);
        }
        reinterpret_cast<PFN_vkUnmapMemory>(fn("vkUnmapMemory"))(d.dev.device, d.memory);
    }
    reinterpret_cast<PFN_vkDestroyBuffer>(fn("vkDestroyBuffer"))(d.dev.device, d.buffer, nullptr);
    reinterpret_cast<PFN_vkFreeMemory>(fn("vkFreeMemory"))(d.dev.device, d.memory, nullptr);
}

void Record(const VulkanDevice& dev, VkCommandBuffer cmd, const UpscalerImage& output)
{
    auto fn = [&](const char* name) { return dev.gdpa(dev.device, name); };
    Dump d;
    d.dev                  = dev;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size               = VkDeviceSize(output.width) * output.height * 8;
    bci.usage              = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (reinterpret_cast<PFN_vkCreateBuffer>(fn("vkCreateBuffer"))(dev.device, &bci, nullptr, &d.buffer) != VK_SUCCESS)
        return;
    VkMemoryRequirements req;
    reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(fn("vkGetBufferMemoryRequirements"))(dev.device, d.buffer, &req);
    VkPhysicalDeviceMemoryProperties mp = {};
    reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(dev.gipa(dev.instance, "vkGetPhysicalDeviceMemoryProperties"))(
        dev.physicalDevice, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryAllocateInfo        mai  = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize               = req.size;
    mai.memoryTypeIndex              = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && mai.memoryTypeIndex == UINT32_MAX; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            mai.memoryTypeIndex = i;
    if (mai.memoryTypeIndex == UINT32_MAX ||
        reinterpret_cast<PFN_vkAllocateMemory>(fn("vkAllocateMemory"))(dev.device, &mai, nullptr, &d.memory) != VK_SUCCESS)
        return;
    reinterpret_cast<PFN_vkBindBufferMemory>(fn("vkBindBufferMemory"))(dev.device, d.buffer, d.memory, 0);

    auto barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(fn("vkCmdPipelineBarrier"));
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask        = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout            = output.layout;
    b.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image                = output.image;
    b.subresourceRange     = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy region = {};
    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent       = {output.width, output.height, 1};
    reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(fn("vkCmdCopyImageToBuffer"))(
        cmd, output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.buffer, 1, &region);
    std::swap(b.oldLayout, b.newLayout);
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    d.image  = output;
    d.readAt = g_count + 16;
    d.index  = g_count - 1;
    g_dumps.push_back(d);
}
}  // namespace

void NgxDumpFrame(const Upscaler* upscaler, VkCommandBuffer cmd, const UpscalerImage& output)
{
    if (Config().dumpFrame < 0 || upscaler->Info().backend == kBackendNss)
        return;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_count;
    const uint64_t first = uint64_t(Config().dumpFrame) + 1;
    if (g_count >= first && g_count < first + uint64_t(DumpCount()))
        Record(upscaler->Device(), cmd, output);
    for (size_t i = 0; i < g_dumps.size();)
        if (g_count == g_dumps[i].readAt)
        {
            Write(g_dumps[i]);
            g_dumps.erase(g_dumps.begin() + i);
        }
        else
            ++i;
}
