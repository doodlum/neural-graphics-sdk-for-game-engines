// GPU timestamps around every upscale, read back without waiting (a few frames later). They feed the live status the
// plugin shows (<control file>.status, written about once a second: the active upscaler, sizes and GPU time), and with
// DU_GPU_TIMING=N (or the control file's gpu_timing) an average over N evaluations is also logged.
// SPDX-License-Identifier: MIT
#include "ngx_common.h"

#include "common.h"
#include "cost_model.h"

#include <windows.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace
{
struct Timer
{
    VkDevice                    device = VK_NULL_HANDLE;
    VkQueryPool                 pool   = VK_NULL_HANDLE;
    float                       period = 1.0f;  // ns per tick
    PFN_vkCmdResetQueryPool     reset  = nullptr;
    PFN_vkCmdWriteTimestamp     write  = nullptr;
    PFN_vkGetQueryPoolResults   results = nullptr;
    static constexpr uint32_t   kSlots = 32;  // pairs of queries; a slot is read back kSlots evaluations later
    uint64_t                    written = 0;
    std::vector<uint8_t>        valid = std::vector<uint8_t>(kSlots, 0);
    std::vector<int>            backend = std::vector<int>(kSlots, 0);
    double                      sum = 0;
    double                      minUs = 1e30, maxUs = 0;
    uint32_t                    samples = 0;
    std::vector<std::string>    name = std::vector<std::string>(kSlots);
    int                         lastBackend = -1;
    std::string                 lastName;
    uint64_t                    reportedAt  = 0;  // GetTickCount64 of the last report
    double                      smoothUs    = 0;  // exponential average for the live status
};

// The live status (one game at a time; the plugin ignores it once it is a few seconds old).
struct Status
{
    std::string upscaler;
    unsigned    renderW = 0, renderH = 0, outW = 0, outH = 0;
    uint64_t    evaluations = 0, evalsAtWrite = 0, writtenAt = 0;
};
Status            g_status;
std::atomic<int>  g_frontEnd{0};  // 0 DLSS (NGX), 1 FSR (FidelityFX API)

void WriteStatus(double gpuUs)
{
    const std::string& control = ControlFilePath();
    const uint64_t     now     = GetTickCount64();
    if (control.empty() || (g_status.writtenAt && now - g_status.writtenAt < 1000))
        return;
    const double fps = g_status.writtenAt ? (g_status.evaluations - g_status.evalsAtWrite) * 1000.0 /
                                                double(now - g_status.writtenAt)
                                          : 0.0;
    g_status.writtenAt    = now;
    g_status.evalsAtWrite = g_status.evaluations;
    char exe[MAX_PATH] = {}, appid[32] = {};
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    const char* slash = strrchr(exe, '\\');
    GetEnvironmentVariableA("SteamAppId", appid, sizeof(appid));
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    const double unixTime = (double((uint64_t(ft.dwHighDateTime) << 32) | ft.dwLowDateTime) - 116444736000000000.0) / 1e7;
    static const char* const kApi[] = {"", "D3D11", "D3D12", "Vulkan"};
    const int api = int(CostGetApi());
    char json[640];
    snprintf(json, sizeof(json),
             "{\"time\": %.1f, \"appid\": \"%s\", \"exe\": \"%s\", \"api\": \"%s\", \"input\": \"%s\", "
             "\"upscaler\": \"%s\", \"render\": [%u, %u], \"output\": [%u, %u], \"gpu_us\": %.1f, \"fps\": %.1f}\n",
             unixTime, appid, slash ? slash + 1 : exe, api >= 0 && api < 4 ? kApi[api] : "",
             g_frontEnd ? "FSR" : "DLSS", g_status.upscaler.c_str(), g_status.renderW, g_status.renderH, g_status.outW,
             g_status.outH, gpuUs, fps);
    const std::string tmp = control + ".status.tmp", dst = control + ".status";
    if (FILE* f = fopen(tmp.c_str(), "w"))
    {
        fputs(json, f);
        fclose(f);
        MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING);
    }
}

std::mutex g_timerMutex;
Timer      g_timer;

bool Setup(const VulkanDevice& dev)
{
    if (g_timer.device == dev.device)
        return g_timer.pool != VK_NULL_HANDLE;
    g_timer        = Timer();
    g_timer.device = dev.device;
    auto createPool = reinterpret_cast<PFN_vkCreateQueryPool>(dev.gdpa(dev.device, "vkCreateQueryPool"));
    g_timer.reset   = reinterpret_cast<PFN_vkCmdResetQueryPool>(dev.gdpa(dev.device, "vkCmdResetQueryPool"));
    g_timer.write   = reinterpret_cast<PFN_vkCmdWriteTimestamp>(dev.gdpa(dev.device, "vkCmdWriteTimestamp"));
    g_timer.results = reinterpret_cast<PFN_vkGetQueryPoolResults>(dev.gdpa(dev.device, "vkGetQueryPoolResults"));
    auto props      = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(dev.gipa(dev.instance, "vkGetPhysicalDeviceProperties"));
    VkPhysicalDeviceProperties pp = {};
    props(dev.physicalDevice, &pp);
    g_timer.period = pp.limits.timestampPeriod;
    VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType             = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount            = Timer::kSlots * 2;
    if (createPool(dev.device, &qci, nullptr, &g_timer.pool) != VK_SUCCESS)
        g_timer.pool = VK_NULL_HANDLE;
    if (GpuTimingFrames() > 0)
        LOGI("GPU timing on: upscale time averaged over %d evaluations", GpuTimingFrames());
    return g_timer.pool != VK_NULL_HANDLE;
}

void Collect(uint32_t slot)
{
    Timer& t = g_timer;
    if (!t.valid[slot])
        return;
    uint64_t       r[4] = {};
    const VkResult res  = t.results(t.device, t.pool, slot * 2, 2, sizeof(r), r, 2 * sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    t.valid[slot] = 0;
    if (res != VK_SUCCESS || !r[1] || !r[3])
        return;
    if (t.backend[slot] != t.lastBackend)
    {
        t.sum = 0, t.samples = 0, t.minUs = 1e30, t.maxUs = 0;
        t.lastBackend = t.backend[slot];
        t.lastName    = t.name[slot];
        t.reportedAt  = 0;
    }
    const double us = double(r[2] - r[0]) * t.period / 1000.0;
    t.smoothUs      = t.smoothUs > 0 ? t.smoothUs * 0.95 + us * 0.05 : us;
    if (GpuTimingFrames() <= 0)
        return;
    t.sum += us;
    t.minUs = us < t.minUs ? us : t.minUs;
    t.maxUs = us > t.maxUs ? us : t.maxUs;
    if (++t.samples >= uint32_t(GpuTimingFrames()))
    {
        // Evaluations per second over the same window: the game's frame rate when it upscales once per frame.
        const uint64_t now = GetTickCount64();
        const double   fps = t.reportedAt ? t.samples * 1000.0 / double(now - t.reportedAt) : 0.0;
        t.reportedAt       = now;
        LOGI("GPU time per upscale (%s): mean %.1f us, min %.1f, max %.1f over %u evaluations, %.1f evaluations/s",
             t.lastName.c_str(), t.sum / t.samples, t.minUs, t.maxUs, t.samples, fps);
        t.sum = 0, t.samples = 0, t.minUs = 1e30, t.maxUs = 0;
    }
}
}  // namespace

void NgxSetFrontEnd(bool fsr)
{
    g_frontEnd = fsr ? 1 : 0;
}

bool NgxDispatch(Upscaler* upscaler, VkCommandBuffer cmd, const UpscalerDispatch& d)
{
    std::lock_guard<std::mutex> lock(g_timerMutex);
    g_status.upscaler = UpscalerName(upscaler->Info());
    g_status.renderW = d.renderWidth, g_status.renderH = d.renderHeight;
    g_status.outW = upscaler->Info().displayWidth, g_status.outH = upscaler->Info().displayHeight;
    ++g_status.evaluations;
    if (!Setup(upscaler->Device()))
    {
        WriteStatus(0.0);
        const bool ok = upscaler->Dispatch(cmd, d);
        NgxDumpFrame(upscaler, cmd, d.output);
        return ok;
    }
    WriteStatus(g_timer.smoothUs);
    const uint32_t slot = uint32_t(g_timer.written++ % Timer::kSlots);
    Collect(slot);  // recorded kSlots evaluations ago: long retired
    g_timer.reset(cmd, g_timer.pool, slot * 2, 2);
    g_timer.write(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_timer.pool, slot * 2);
    const bool ok = upscaler->Dispatch(cmd, d);
    g_timer.write(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_timer.pool, slot * 2 + 1);
    g_timer.valid[slot]   = 1;
    g_timer.backend[slot] = upscaler->Info().backend * 16 + upscaler->Info().fsr4Model;
    g_timer.name[slot]    = UpscalerName(upscaler->Info());
    NgxDumpFrame(upscaler, cmd, d.output);
    return ok;
}
