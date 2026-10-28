// Equal-cost render scale (see cost_model.h).
//
// Frame cost model, in GPU milliseconds at the full 1600 MHz clock:
//   frame(U, P) = F + G * P + u_U(P)    P = render megapixels, F = the game's fixed cost, G = its cost per extra
//                                        render megapixel, u_U = upscaler U's measured cost (table below)
// The reference is NSS Quality at the DLSS mode's standard ratio (P0); every other upscaler gets the P that solves
// frame(U, P) = frame(NSS Quality, P0), where F cancels. G is measured while the game runs: the GPU's busy time per
// frame (amdgpu's gpu_busy_percent sampled by a thread, scaled by the shader clock to full-clock time, times the
// evaluate interval) minus the modelled upscale cost, kept per render size; G is the slope across the sizes seen (see
// SlopeLocked). The sizes are kept per game (next to the control file) so the next launch starts from them.
// SPDX-License-Identifier: MIT
#include "cost_model.h"

#include "common.h"
#include "ngx_common.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace
{
constexpr double kFullClockMhz = 1600.0;  // Van Gogh's top shader clock: the table's reference
constexpr double kDefaultG     = 60.0;    // ms per extra render megapixel before anything is measured

// Upscale cost per upscaler on the Steam Deck, ms at 1280x800 output (scaled by output pixels for other sizes),
// measured at render ratios 3, 2.5, 2 and 1.5 (Streamline sample, AISS_RATIO, mean of 2x 120 evaluations; sweep in
// steamdeck/README.md) and interpolated linearly in render pixels between them. NSS takes its exact-2x path (static
// LUT) only at exactly 2x, so for NSS the 2x column is used at exactly 2x and the other ratios interpolate around it.
// FSR 4 Lite in [0] is the SDK Vulkan DLL with the replacement shaders (about 22% under AMD's D3D12 DLL, which the
// other [0] FSR 4 rows measured), used on every API when that DLL works. The FSR 4 interop rows add 2.8 ms to the
// upscale's own GPU time: the frame-rate drop measured against NSS at the same render size is that much larger, the GPU
// idling while the Vulkan and D3D12 queues wait on each other.
struct UpscaleCost
{
    double ms[4];  // at ratio 3, 2.5, 2, 1.5
    bool   exact2x;
};
constexpr double kGridF[4]    = {1.0 / 9.0, 1.0 / 6.25, 1.0 / 4.0, 1.0 / 2.25};  // render / output pixels
constexpr double kGridOutMpx  = 1280.0 * 800.0 / 1e6;

enum CostRow
{
    kNssQuality,
    kNssBalanced,
    kNssPerformance,
    kXess,
    kRowFsr4Lite,  // not kFsr4Lite: that name is the FSR 4 model enum
    kRowFsr4Int8,
    kRowFsr4Rdna2,
    kCostRows
};

// [0]: D3D12 games (FSR 4 native). [1]: Vulkan games (FSR 4 through the D3D12 interop); D3D11 games use it too.
// clang-format off
const UpscaleCost kCost[2][kCostRows] = {
    {
        /* NSS Quality     */ {{1.558, 1.879, 2.224, 3.716}, true},
        /* NSS Balanced    */ {{1.285, 1.184, 1.217, 2.411}, true},
        /* NSS Performance */ {{1.155, 1.048, 1.089, 2.282}, true},
        /* XeSS            */ {{3.098, 3.128, 3.192, 3.355}, false},
        /* FSR 4 Lite      */ {{2.336, 2.338, 2.346, 2.373}, false},
        /* FSR 4 INT8      */ {{5.035, 5.006, 5.019, 5.050}, false},
        /* FSR 4 INT8 RDNA2*/ {{4.954, 4.964, 4.976, 5.021}, false},
    },
    {
        /* NSS Quality     */ {{1.577, 1.899, 2.456, 3.676}, true},
        /* NSS Balanced    */ {{1.491, 1.210, 1.432, 2.428}, true},
        /* NSS Performance */ {{1.360, 1.072, 1.293, 2.304}, true},
        /* XeSS            */ {{3.158, 3.188, 3.253, 3.426}, false},
        /* FSR 4 Lite      */ {{6.366, 6.366, 6.400, 6.470}, false},
        /* FSR 4 INT8      */ {{8.309, 8.333, 8.344, 8.445}, false},
        /* FSR 4 INT8 RDNA2*/ {{8.196, 8.270, 8.258, 8.371}, false},
    },
};
// clang-format on

std::atomic<int> g_api{static_cast<int>(CostApi::Unknown)};

struct CostKey
{
    int group, row;
};

CostKey KeyFor(int backend, int fsr4Model, int nssQuality, CostApi api)
{
    const int group = api == CostApi::D3D12 ? 0 : 1;
    if (backend == kBackendXess)
        return {group, kXess};
    if (backend == kBackendFsr4 || backend == kBackendFsr4Dx12 || backend == kBackendFsr4Interop || backend == kBackendFsr4Vk)
    {
        // Every API runs FSR 4 through AMD's SDK Vulkan DLL when it works (fsr4_vk.cpp; about the cost of native D3D12),
        // else D3D12 natively and D3D11 / Vulkan through the interop.
        const bool interop = backend == kBackendFsr4Interop ||
                             (backend == kBackendFsr4 && api != CostApi::D3D12 && Fsr4VkUnavailable());
        const int  m       = fsr4Model == kFsr4Int8 ? 1 : fsr4Model == kFsr4Rdna2 ? 2 : 0;  // Auto is Lite on the Deck
        return {interop ? 1 : 0, kRowFsr4Lite + m};
    }
    return {group, nssQuality == 1 ? kNssBalanced : nssQuality == 2 ? kNssPerformance : kNssQuality};
}

double UpscaleMs(CostKey key, double renderMpx, double outMpx, bool exact2x)
{
    const UpscaleCost& c     = kCost[key.group][key.row];
    const double       scale = outMpx / kGridOutMpx;
    if (exact2x && c.exact2x)
        return c.ms[2] * scale;
    // The points to interpolate between (NSS's exact-2x point is not on its general-ratio curve).
    double f[4], ms[4];
    int    n = 0;
    for (int i = 0; i < 4; ++i)
        if (!(c.exact2x && i == 2))
            f[n] = kGridF[i], ms[n++] = c.ms[i];
    const double x = renderMpx / outMpx;
    if (x <= f[0])
        return ms[0] * scale;  // below ratio 3: hold (fixed costs dominate)
    int i = 0;
    while (i < n - 2 && x > f[i + 1])
        ++i;  // past the last point: extrapolate the last segment
    return (ms[i] + (ms[i + 1] - ms[i]) * (x - f[i]) / (f[i + 1] - f[i])) * scale;
}

bool IsExact2x(unsigned rw, unsigned rh, unsigned ow, unsigned oh)
{
    return rw * 2 == ow && rh * 2 == oh;
}

// ---- GPU busy sampler ---------------------------------------------------------------------------------------------

struct Sampler
{
    std::mutex mutex;
    double     busyFullClock = 0;  // sum of busy fraction * clock / full clock
    uint32_t   samples       = 0;
    bool       ok            = true;
};

Sampler     g_sampler;
std::string g_busyPath, g_sclkPath;

bool ReadSmallFile(const std::string& path, char* buf, size_t size)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    const size_t n = fread(buf, 1, size - 1, f);
    fclose(f);
    buf[n] = 0;
    return n > 0;
}

bool FindCard()
{
    char buf[512];
    for (int i = 0; i < 4; ++i)
    {
        const std::string dir = "Z:\\sys\\class\\drm\\card" + std::to_string(i) + "\\device\\";
        if (ReadSmallFile(dir + "gpu_busy_percent", buf, sizeof(buf)))
        {
            g_busyPath = dir + "gpu_busy_percent";
            g_sclkPath = dir + "pp_dpm_sclk";
            return true;
        }
    }
    return false;
}

double CurrentClockMhz()
{
    // pp_dpm_sclk: "0: 200Mhz *\n1: 1100Mhz \n2: 1600Mhz" - the starred line is the current clock (on Van Gogh the
    // middle line shows the current clock when it is between the two limits).
    char buf[512];
    if (!ReadSmallFile(g_sclkPath, buf, sizeof(buf)))
        return kFullClockMhz;
    for (char* line = buf; line && *line;)
    {
        char* next = strchr(line, '\n');
        if (next)
            *next++ = 0;
        if (strchr(line, '*'))
        {
            const char* colon = strchr(line, ':');
            const double mhz  = colon ? atof(colon + 1) : 0.0;
            return mhz > 0 ? mhz : kFullClockMhz;
        }
        line = next;
    }
    return kFullClockMhz;
}

void SamplerThread()
{
    char buf[64];
    for (;;)
    {
        Sleep(20);
        if (!ReadSmallFile(g_busyPath, buf, sizeof(buf)))
        {
            std::lock_guard<std::mutex> lock(g_sampler.mutex);
            g_sampler.ok = false;
            return;
        }
        const double busy  = atof(buf) / 100.0;
        const double clock = CurrentClockMhz();
        std::lock_guard<std::mutex> lock(g_sampler.mutex);
        g_sampler.busyFullClock += busy * clock / kFullClockMhz;
        ++g_sampler.samples;
    }
}

// ---- Game cost (G) ------------------------------------------------------------------------------------------------

// The game's GPU time per frame (upscale excluded) at the render sizes it has run at. Only the slope matters for equal
// cost (the fixed part, shadows / ray-traced passes / UI that do not scale with the render size, is the same on both
// sides), so with two or more render sizes G is the least-squares slope through them; with one, G is that size's
// average cost per megapixel times kMarginalShare.
constexpr int    kBuckets       = 6;
constexpr double kMarginalShare = 0.55;  // Streamline sample: slope 83 ms/MPx vs 150 ms/MPx average at 640x361

struct Bucket
{
    double mpx = 0, gameMs = 0;  // render megapixels, smoothed game GPU ms per frame at full clock
    uint64_t used = 0;
};

struct GameCost
{
    std::mutex mutex;
    Bucket     buckets[kBuckets];
    bool       loaded    = false;
    bool       samplerUp = false;
    bool       dirty     = false;
    // current window
    LARGE_INTEGER windowStart = {};
    uint32_t      evals       = 0;
    double        upscaleMs   = 0;  // modelled upscale cost, summed over the window's evaluates
    double        renderMpx   = 0;  // summed
    uint64_t      savedAt     = 0;
    uint64_t      useCounter  = 0;
};

GameCost g_game;

std::string CostFilePath()
{
    // <control file's folder>\cost\<game exe>.txt
    const std::string& control = ControlFilePath();
    if (control.empty())
        return {};
    const size_t slash = control.find_last_of('\\');
    if (slash == std::string::npos)
        return {};
    char exe[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    const char* name = strrchr(exe, '\\');
    name             = name ? name + 1 : exe;
    return control.substr(0, slash) + "\\cost\\" + name + ".txt";
}

// The game's cost per extra render megapixel, 0 when nothing is measured yet.
double SlopeLocked()
{
    int    n = 0;
    double sx = 0, sy = 0, lo = 1e9, hi = 0;
    for (const Bucket& b : g_game.buckets)
        if (b.mpx > 0)
            ++n, sx += b.mpx, sy += b.gameMs, lo = std::min(lo, b.mpx), hi = std::max(hi, b.mpx);
    if (!n)
        return 0;
    const double avg = sy / sx;  // average ms per MPx over the measured sizes
    if (n < 2 || hi < lo * 1.2)
        return avg * kMarginalShare;
    const double mx = sx / n, my = sy / n;
    double       sxy = 0, sxx = 0;
    for (const Bucket& b : g_game.buckets)
        if (b.mpx > 0)
            sxy += (b.mpx - mx) * (b.gameMs - my), sxx += (b.mpx - mx) * (b.mpx - mx);
    const double slope = sxy / sxx;
    return std::min(std::max(slope, avg * 0.2), avg);
}

void LoadLocked()
{
    if (g_game.loaded)
        return;
    g_game.loaded          = true;
    const std::string path = CostFilePath();
    FILE*             f    = path.empty() ? nullptr : fopen(path.c_str(), "r");
    if (!f)
        return;
    int    n = 0;
    double mpx, ms;
    while (n < kBuckets && fscanf(f, "%lf %lf", &mpx, &ms) == 2)  // "render-MPx game-ms" per line
        if (mpx > 0.01 && mpx < 100 && ms > 0 && ms < 1000)
            g_game.buckets[n].mpx = mpx, g_game.buckets[n].gameMs = ms, g_game.buckets[n++].used = ++g_game.useCounter;
    fclose(f);
    if (n)
        LOGI("equal cost: %d render size(s) measured in earlier runs: game cost %.1f ms per extra render megapixel", n,
             SlopeLocked());
}

void SaveLocked()
{
    const std::string path = CostFilePath();
    if (path.empty())
        return;
    const size_t slash = path.find_last_of('\\');
    CreateDirectoryA(path.substr(0, slash).c_str(), nullptr);
    if (FILE* f = fopen(path.c_str(), "w"))
    {
        for (const Bucket& b : g_game.buckets)
            if (b.mpx > 0)
                fprintf(f, "%.4f %.3f\n", b.mpx, b.gameMs);
        fclose(f);
    }
    g_game.dirty   = false;
    g_game.savedAt = GetTickCount64();
}

// One measured window: smooth into the bucket of this render size (within 5%), else replace the least recently used.
void AddSampleLocked(double mpx, double gameMs)
{
    Bucket* hit = nullptr;
    for (Bucket& b : g_game.buckets)
        if (b.mpx > 0 && std::fabs(b.mpx - mpx) < 0.05 * mpx)
            hit = &b;
    if (hit)
    {
        hit->mpx    = hit->mpx * 0.7 + mpx * 0.3;
        hit->gameMs = hit->gameMs * 0.7 + gameMs * 0.3;
    }
    else
    {
        hit = &g_game.buckets[0];
        for (Bucket& b : g_game.buckets)
            if (b.used < hit->used)
                hit = &b;
        hit->mpx    = mpx;
        hit->gameMs = gameMs;
    }
    hit->used    = ++g_game.useCounter;
    g_game.dirty = true;
}

double GameCostLocked()
{
    LoadLocked();
    const double g = SlopeLocked();
    return g > 0 ? g : kDefaultG;
}
}  // namespace

void CostSetApi(CostApi api)
{
    g_api = static_cast<int>(api);
}

CostApi CostGetApi()
{
    return static_cast<CostApi>(g_api.load());
}

bool EqualCostEnabled()
{
    const int live = LiveEqualCost();
    return live >= 0 ? live != 0 : Config().equalCost;
}

void CostNoteEvaluate(const UpscalerCreateInfo& info, unsigned renderW, unsigned renderH, unsigned outW, unsigned outH)
{
    if (!renderW || !renderH || !outW || !outH)
        return;
    std::lock_guard<std::mutex> lock(g_game.mutex);
    LoadLocked();
    if (!g_game.samplerUp)
    {
        g_game.samplerUp = true;
        if (FindCard())
            std::thread(SamplerThread).detach();
        else
            LOGI("equal cost: amdgpu gpu_busy_percent not readable; the game cost stays at %.1f ms/MPx",
                 GameCostLocked());
    }
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    if (!g_game.evals)
    {
        g_game.windowStart = now;
        std::lock_guard<std::mutex> s(g_sampler.mutex);
        g_sampler.busyFullClock = 0;
        g_sampler.samples       = 0;
    }
    const int    model     = info.fsr4Active >= 0 ? info.fsr4Active : info.fsr4Model;
    const CostKey key      = KeyFor(info.backend, model, info.nssQuality, static_cast<CostApi>(g_api.load()));
    const double renderMpx = renderW * double(renderH) / 1e6;
    const double outMpx    = outW * double(outH) / 1e6;
    g_game.upscaleMs += UpscaleMs(key, renderMpx, outMpx, IsExact2x(renderW, renderH, outW, outH));
    g_game.renderMpx += renderMpx;
    ++g_game.evals;

    const double ms = double(now.QuadPart - g_game.windowStart.QuadPart) * 1000.0 / double(freq.QuadPart);
    if (ms < 2000.0)
        return;
    double   busy    = 0;
    uint32_t samples = 0;
    bool     ok      = true;
    {
        std::lock_guard<std::mutex> s(g_sampler.mutex);
        busy    = g_sampler.busyFullClock;
        samples = g_sampler.samples;
        ok      = g_sampler.ok;
    }
    // evals - 1 intervals fit in the window (the first evaluate opened it).
    const uint32_t frames = g_game.evals - 1;
    if (ok && samples >= 20 && frames >= 10)
    {
        const double gpuMsPerFrame = busy / samples * ms / frames;
        const double gameMs        = gpuMsPerFrame - g_game.upscaleMs / g_game.evals;
        const double mpx           = g_game.renderMpx / g_game.evals;
        if (gameMs > 0 && mpx > 0)
        {
            AddSampleLocked(mpx, gameMs);
            LOGV("equal cost: %.2f ms GPU per frame (full clock), upscale %.2f ms, %.3f MPx -> game %.2f ms (cost per "
                 "extra render megapixel %.1f ms)",
                 gpuMsPerFrame, g_game.upscaleMs / g_game.evals, mpx, gameMs, SlopeLocked());
            if (GetTickCount64() - g_game.savedAt > 10000)
            {
                SaveLocked();
                LOGI("equal cost: game cost %.1f ms per extra render megapixel", SlopeLocked());
            }
        }
    }
    g_game.evals     = 0;
    g_game.upscaleMs = 0;
    g_game.renderMpx = 0;
}

float EqualCostRatio(int perfQuality, float standardRatio, unsigned outW, unsigned outH)
{
    if (perfQuality == NVSDK_NGX_PerfQuality_Value_DLAA || standardRatio <= 1.0f || !EqualCostEnabled() || !outW || !outH)
        return standardRatio;
    int       model   = kFsr4Auto;
    const int backend = SelectedBackend(&model);
    const int nssQ    = backend == kBackendNss ? NssQualityForDlss(perfQuality) : 0;
    const CostApi api = static_cast<CostApi>(g_api.load());
    const bool interopDown = backend == kBackendFsr4 && api != CostApi::D3D12 && Fsr4VkUnavailable() &&
                             Fsr4InteropUnavailable(model);
    if (interopDown)
        return standardRatio;  // runs NSS Quality
    const CostKey key = KeyFor(backend, model, nssQ, api);
    if (key.row == kNssQuality)
        return standardRatio;

    double g;
    {
        std::lock_guard<std::mutex> lock(g_game.mutex);
        g = GameCostLocked();
    }
    const double outMpx = outW * double(outH) / 1e6;
    const unsigned refW = static_cast<unsigned>(outW / standardRatio), refH = static_cast<unsigned>(outH / standardRatio);
    const double refMpx = refW * double(refH) / 1e6;
    const double target = g * refMpx + UpscaleMs({key.group, kNssQuality}, refMpx, outMpx, IsExact2x(refW, refH, outW, outH));
    // Solve g * P + u(P) = target by bisection (u is measured, not linear). No larger than native and no smaller than a
    // quarter per axis (ratio 4: below that the image falls apart, so an expensive upscaler at Ultra Performance may stay
    // costlier than the reference).
    auto         frame = [&](double mpx) { return g * mpx + UpscaleMs(key, mpx, outMpx, false); };
    double       lo = outMpx / 16.0, hi = outMpx, mpx;
    const bool   clamped = frame(lo) > target || frame(hi) < target;
    if (frame(lo) > target)
        mpx = lo;
    else if (frame(hi) < target)
        mpx = hi;
    else
    {
        for (int i = 0; i < 40; ++i)
        {
            const double mid = 0.5 * (lo + hi);
            (frame(mid) < target ? lo : hi) = mid;
        }
        mpx = 0.5 * (lo + hi);
    }
    const float  ratio = static_cast<float>(std::sqrt(outMpx / mpx));
    const double cost  = frame(mpx);
    LOGI("equal cost: mode %d, %ux%u output: NSS Quality at %ux%u costs %.2f ms (game %.1f ms/MPx); the selected "
         "upscaler renders at ratio %.3f (standard %.3f) for %.2f ms%s",
         perfQuality, outW, outH, refW, refH, target, g, ratio, standardRatio, cost,
         clamped ? " (clamped: equal cost not reachable)" : "");
    return ratio;
}
