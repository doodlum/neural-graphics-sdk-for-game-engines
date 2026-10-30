// D3D12 counterpart of ngx_d3d11_test's replay mode: drives nvngx.dll's D3D12 entry points on vkd3d-proton the way a
// game's NGX code does, with dumped inputs, and saves the output.
// Usage (under Proton): ngx_d3d12_test.exe <path to nvngx.dll> --replay DIR [options]
//   --replay DIR  replay dumped inputs (NNN_*.raw or deckyupscaler_in_NNN_*.raw, with jitter / MV scale from
//                 DIR/frames.txt): the textures keep the dumped extents and formats, the render size is the depth
//                 dump's, the display size the output dump's
//   --first N --count K   frames of the replay (default 0, 8)
//   --loops L     play the replayed frames L times (reset only on the first)
//   --crop        colour and motion vectors cropped to the render size (the exact-size reference)
//   --color16     B10G11R11 colour as RGBA16F (the same values)
//   --anysize     accept any MaxPerf render size from the capability query (e.g. with DU_RATIO=1.5)
//   --flags N     DLSS create flags (default 11: HDR, low-res MVs, inverted depth; 64 = auto exposure)
//   --perf N      PerfQualityValue at create (default 0 = MaxPerf; FH6 uses 2)
//   --save FILE   write the output (RGBA16F) to FILE
//   --log FILE    print to FILE instead of stdout
//   --sync        wait for the GPU after every frame
// D3D12-specific variations (defaults are what FH6 is believed to do):
//   --depth32f    depth as R32_FLOAT (default R32_TYPELESS with ALLOW_DEPTH_STENCIL, D32_FLOAT DSV, like the game)
//   --depthpad W,H  depth texture W x H (junk outside the render size), e.g. 1280,800
//   --typeless    colour / MV in their TYPELESS formats where one exists (RGBA16, RG16)
//   --colorflags F  colour resource flags: none, rt (ALLOW_RENDER_TARGET, default), uav, rtuav
//   --state S     state the inputs are left in: npsr (default), allsr (PSR|NPSR), common, depthread (depth only:
//                 DEPTH_READ|NPSR)
//   --placed OFF  inputs and output as placed resources at OFF (multiple of 64 KiB) in a fresh heap
//   --subrect     set DLSS.Render.Subrect.Dimensions (as Streamline does)
//   --depthbase X,Y  depth subrect base: the depth data is written at (X,Y) inside a larger depth texture
//   --gamework    record unrelated compute-free work (a clear of a scratch UAV) in the same list before evaluating
//   --outfmt F    output format: rgba16f (default), r11g11b10 (read back converted to RGBA16F)
//   --outsize W,H output texture W x H (larger than the display size; the top-left display-size region is saved)
//   --ctlseq S    runtime upscaler switches through the control file (DU_CONTROL): S = "code@frame,..."; before
//                 that frame the harness writes "code 0 0" to the file and waits for the proxy to re-read it (codes as
//                 the plugin's: 0-2 NSS quality/balanced/performance, 9 FSR 3.1, ...). E.g. "9@0,1@40,0@80" is FH6's
//                 session (FSR 3 -> NSS Balanced -> NSS Quality)
// Measurement:
//   --gputime     GPU time of each upscale (a timestamp pair around it on the same command list), reported as mean /
//                 median / min / max over the frames after the first quarter (at most 20 skipped)
//   --ffx         the DLL is an AMD FidelityFX API upscaler for D3D12 (ffxCreateContext / ffxDispatch, e.g. an
//                 amd_fidelityfx_upscaler_dx12.dll) instead of an NGX DLL; --flags maps onto its create flags, with
//                 the proxy's camera defaults (near 0.1, far 100000, 60 degrees)
// SPDX-License-Identifier: MIT
#include <d3d12.h>
#include <windows.h>

#include "amd_ffx/ffx_api.h"
#include "amd_ffx/ffx_api_dx12.h"
#include "amd_ffx/ffx_upscale.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Result = int;
constexpr Result kSuccess = 1;

// NVSDK_NGX_Parameter vtable as MSVC lays it out (overloads in reverse declaration order).
struct Params
{
    void** vtbl;
    void Set(const char* n, void* v) { reinterpret_cast<void (*)(Params*, const char*, void*)>(vtbl[0])(this, n, v); }
    void Set(const char* n, int v) { reinterpret_cast<void (*)(Params*, const char*, int)>(vtbl[3])(this, n, v); }
    void Set(const char* n, unsigned v) { reinterpret_cast<void (*)(Params*, const char*, unsigned)>(vtbl[4])(this, n, v); }
    void Set(const char* n, float v) { reinterpret_cast<void (*)(Params*, const char*, float)>(vtbl[6])(this, n, v); }
    Result Get(const char* n, unsigned* v) { return reinterpret_cast<Result (*)(Params*, const char*, unsigned*)>(vtbl[12])(this, n, v); }
    Result Get(const char* n, int* v) { return reinterpret_cast<Result (*)(Params*, const char*, int*)>(vtbl[11])(this, n, v); }
    Result Get(const char* n, void** v) { return reinterpret_cast<Result (*)(Params*, const char*, void**)>(vtbl[8])(this, n, v); }
};
struct Handle { unsigned Id; };

#define NGX_FN(name, ...) using PFN_##name = Result (*)(__VA_ARGS__); PFN_##name name = nullptr;
NGX_FN(NVSDK_NGX_D3D12_Init, unsigned long long, const wchar_t*, ID3D12Device*, const void*, int)
NGX_FN(NVSDK_NGX_D3D12_GetCapabilityParameters, Params**)
NGX_FN(NVSDK_NGX_D3D12_AllocateParameters, Params**)
NGX_FN(NVSDK_NGX_D3D12_CreateFeature, ID3D12GraphicsCommandList*, int, Params*, Handle**)
NGX_FN(NVSDK_NGX_D3D12_EvaluateFeature, ID3D12GraphicsCommandList*, const Handle*, Params*, void*)
NGX_FN(NVSDK_NGX_D3D12_ReleaseFeature, Handle*)
NGX_FN(NVSDK_NGX_D3D12_DestroyParameters, Params*)
NGX_FN(NVSDK_NGX_D3D12_Shutdown1, ID3D12Device*)

static float HalfToFloat(uint16_t h)
{
    const uint32_t exp = (h >> 10) & 31, man = h & 1023;
    const float    v   = exp == 0 ? std::ldexp(float(man), -24) : std::ldexp(float(man | 1024), int(exp) - 25);
    return (h & 0x8000) ? -v : v;
}

// Raw input dump as the proxy writes them: uint32 {'AISR', w, h, VkFormat, bpp}, pixels.
struct Raw
{
    uint32_t             w = 0, h = 0, format = 0, bpp = 0;
    std::vector<uint8_t> data;
    bool Load(const std::string& path)
    {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t hdr[5] = {};
        bool ok = fread(hdr, 4, 5, f) == 5 && hdr[0] == 0x52534941u;
        if (ok)
        {
            w = hdr[1], h = hdr[2], format = hdr[3], bpp = hdr[4];
            data.resize(size_t(w) * h * bpp);
            ok = fread(data.data(), 1, data.size(), f) == data.size();
        }
        fclose(f);
        return ok;
    }
    void ToHalf()
    {
        if (format != 122) return;
        std::vector<uint8_t> c(size_t(w) * h * 8);
        for (size_t i = 0; i < size_t(w) * h; ++i)
        {
            uint32_t v;
            memcpy(&v, &data[i * 4], 4);
            const uint16_t px[4] = {uint16_t((v & 0x7ff) << 4), uint16_t(((v >> 11) & 0x7ff) << 4), uint16_t(((v >> 22) & 0x3ff) << 5), 0x3c00};
            memcpy(&c[i * 8], px, 8);
        }
        data.swap(c), format = 97, bpp = 8;
    }
    void Crop(uint32_t cw, uint32_t ch)
    {
        std::vector<uint8_t> c(size_t(cw) * ch * bpp);
        for (uint32_t y = 0; y < ch; ++y)
            memcpy(&c[size_t(y) * cw * bpp], &data[size_t(y) * w * bpp], size_t(cw) * bpp);
        data.swap(c), w = cw, h = ch;
    }
    // Places the data at (x, y) inside a W x H image filled with junk (pseudo-random bytes for colour; for 32-bit
    // depth a value far from the scene's).
    void Embed(uint32_t W, uint32_t H, uint32_t x, uint32_t y)
    {
        std::vector<uint8_t> c(size_t(W) * H * bpp);
        uint32_t seed = 12345;
        for (size_t i = 0; i < c.size() / 4; ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            const float junk = 0.03f + (seed >> 8) * (0.5f / 16777216.0f);
            memcpy(&c[i * 4], &junk, 4);
        }
        for (uint32_t r = 0; r < h; ++r)
            memcpy(&c[(size_t(r + y) * W + x) * bpp], &data[size_t(r) * w * bpp], size_t(w) * bpp);
        data.swap(c), w = W, h = H;
    }
};

static bool g_typeless = false;
static DXGI_FORMAT DxgiFromVk(uint32_t vk)
{
    switch (vk)
    {
    case 97: return g_typeless ? DXGI_FORMAT_R16G16B16A16_TYPELESS : DXGI_FORMAT_R16G16B16A16_FLOAT;
    case 109: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case 122: return DXGI_FORMAT_R11G11B10_FLOAT;
    case 64: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case 37: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case 44: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case 83: return g_typeless ? DXGI_FORMAT_R16G16_TYPELESS : DXGI_FORMAT_R16G16_FLOAT;
    case 103: return DXGI_FORMAT_R32G32_FLOAT;
    case 100: return DXGI_FORMAT_R32_TYPELESS;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { printf("FAIL: %s = 0x%08lx (line %d)\n", #x, (unsigned long)hr_, __LINE__); return 1; } } while (0)

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dll = argc > 1 ? argv[1] : "nvngx.dll";
    const char* saveFile = nullptr;
    const char* replayDir = nullptr;
    int         first = 0, count = 8, loops = 1, createFlags = 11, perf = 0;
    bool        crop = false, color16 = false, anySize = false, sync = false, depth32f = false, subrect = false,
                gamework = false, outR11 = false, gpuTime = false, ffx = false;
    unsigned    outTexW = 0, outTexH = 0;
    unsigned    depthPadW = 0, depthPadH = 0, depthBaseX = 0, depthBaseY = 0;
    UINT64      placed = UINT64(-1);
    std::vector<std::pair<int, int>> ctlSeq;  // (frame, code)
    D3D12_RESOURCE_FLAGS colorFlags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_RESOURCE_STATES inState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, depthState = inState;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--save" && more) saveFile = argv[++i];
        else if (a == "--log" && more) freopen(argv[++i], "w", stdout);
        else if (a == "--replay" && more) replayDir = argv[++i];
        else if (a == "--first" && more) first = atoi(argv[++i]);
        else if (a == "--count" && more) count = atoi(argv[++i]);
        else if (a == "--loops" && more) loops = atoi(argv[++i]);
        else if (a == "--flags" && more) createFlags = atoi(argv[++i]);
        else if (a == "--perf" && more) perf = atoi(argv[++i]);
        else if (a == "--crop") crop = true;
        else if (a == "--color16") color16 = true;
        else if (a == "--anysize") anySize = true;
        else if (a == "--sync") sync = true;
        else if (a == "--depth32f") depth32f = true;
        else if (a == "--typeless") g_typeless = true;
        else if (a == "--subrect") subrect = true;
        else if (a == "--gamework") gamework = true;
        else if (a == "--gputime") gpuTime = true;
        else if (a == "--ffx") ffx = true;
        else if (a == "--depthpad" && more) sscanf(argv[++i], "%u,%u", &depthPadW, &depthPadH);
        else if (a == "--outsize" && more) sscanf(argv[++i], "%u,%u", &outTexW, &outTexH);
        else if (a == "--depthbase" && more) sscanf(argv[++i], "%u,%u", &depthBaseX, &depthBaseY);
        else if (a == "--placed" && more) placed = strtoull(argv[++i], nullptr, 0);
        else if (a == "--ctlseq" && more)
        {
            for (const char* c = argv[++i]; *c;)
            {
                int code = 0, frame = 0;
                if (sscanf(c, "%d@%d", &code, &frame) == 2) ctlSeq.push_back({frame, code});
                const char* comma = strchr(c, ',');
                c = comma ? comma + 1 : c + strlen(c);
            }
        }
        else if (a == "--outfmt" && more) outR11 = std::string(argv[++i]) == "r11g11b10";
        else if (a == "--colorflags" && more)
        {
            const std::string f = argv[++i];
            colorFlags = f == "rt" ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
                       : f == "uav" ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                       : f == "rtuav" ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                      : D3D12_RESOURCE_FLAG_NONE;
        }
        else if (a == "--state" && more)
        {
            const std::string s = argv[++i];
            if (s == "allsr") inState = depthState = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
            else if (s == "common") inState = depthState = D3D12_RESOURCE_STATE_COMMON;
            else if (s == "depthread") depthState = D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        else { printf("FAIL: unknown option %s\n", argv[i]); return 1; }
    }
    if (!replayDir) { printf("FAIL: --replay DIR is required\n"); return 1; }
    std::string ctlPath;  // DU_CONTROL as the proxy opens it
    if (const char* c = getenv("DU_CONTROL"))
    {
        ctlPath = c[0] == '/' ? std::string("Z:") + c : c;
        for (char& ch : ctlPath) if (ch == '/') ch = '\\';
    }
    auto writeCtl = [&](int code) {
        if (FILE* f = fopen(ctlPath.c_str(), "w")) { fprintf(f, "%d 0 0\n", code); fclose(f); }
        else printf("note: cannot write control file %s\n", ctlPath.c_str());
        Sleep(700);  // the proxy re-reads the control file at most every 500 ms
    };
    if (!ctlSeq.empty() && ctlPath.empty()) { printf("FAIL: --ctlseq needs DU_CONTROL\n"); return 1; }
    for (auto& [frame, code] : ctlSeq) if (frame == 0) { writeCtl(code); printf("control %d before create\n", code); }

    HMODULE ngx = LoadLibraryA(dll);
    if (!ngx) { printf("FAIL: cannot load %s\n", dll); return 1; }
    PfnFfxCreateContext ffxCreate = nullptr;
    PfnFfxDispatch      ffxRun = nullptr;
    PfnFfxDestroyContext ffxDestroy = nullptr;
    if (ffx)
    {
        ffxCreate  = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(ngx, "ffxCreateContext"));
        ffxRun     = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(ngx, "ffxDispatch"));
        ffxDestroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(ngx, "ffxDestroyContext"));
        if (!ffxCreate || !ffxRun || !ffxDestroy) { printf("FAIL: %s is not an FFX API DLL\n", dll); return 1; }
    }
    else
    {
#define LOAD(name) name = reinterpret_cast<PFN_##name>(GetProcAddress(ngx, #name)); if (!name) { printf("FAIL: missing %s\n", #name); return 1; }
    LOAD(NVSDK_NGX_D3D12_Init) LOAD(NVSDK_NGX_D3D12_GetCapabilityParameters) LOAD(NVSDK_NGX_D3D12_AllocateParameters)
    LOAD(NVSDK_NGX_D3D12_CreateFeature) LOAD(NVSDK_NGX_D3D12_EvaluateFeature) LOAD(NVSDK_NGX_D3D12_ReleaseFeature)
    LOAD(NVSDK_NGX_D3D12_DestroyParameters) LOAD(NVSDK_NGX_D3D12_Shutdown1)
    }

    ID3D12Device* dev = nullptr;
    CHECK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT};
    ID3D12CommandQueue* queue = nullptr;
    CHECK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    constexpr int kAlloc = 3;
    ID3D12CommandAllocator* alloc[kAlloc] = {};
    UINT64 allocFence[kAlloc] = {};
    for (auto& a : alloc) CHECK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a)));
    ID3D12GraphicsCommandList* list = nullptr;
    CHECK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[0], nullptr, IID_PPV_ARGS(&list)));
    ID3D12Fence* fence = nullptr;
    CHECK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    UINT64 fenceValue = 0;
    HANDLE event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    auto waitFor = [&](UINT64 v) {
        if (fence->GetCompletedValue() < v) { fence->SetEventOnCompletion(v, event); WaitForSingleObject(event, INFINITE); }
    };
    auto submit = [&]() {
        list->Close();
        ID3D12CommandList* l = list;
        queue->ExecuteCommandLists(1, &l);
        queue->Signal(fence, ++fenceValue);
        return fenceValue;
    };

    if (!ffx && NVSDK_NGX_D3D12_Init(0x1234, L".", dev, nullptr, 0x14) != kSuccess) { printf("FAIL: Init\n"); return 1; }

    // ---- dumped inputs ----
    struct Frame { Raw color, mv, depth; float jx = 0, jy = 0, sx = 1, sy = 1; };
    std::vector<Frame> frames(count);
    for (int i = 0; i < count; ++i)
    {
        const std::string dir = std::string(replayDir) + "\\";
        std::string pre;
        for (const char* fmt : {"%03d_", "deckyupscaler_in_%03d_", "aiss_in_%03d_"})  // the last: older builds' dumps
        {
            char a[64];
            snprintf(a, sizeof(a), fmt, first + i);
            Raw probe;
            pre = dir + a;
            if (probe.Load(pre + "depth.raw")) break;
        }
        Frame& f = frames[i];
        if (!f.color.Load(pre + "color.raw") || !f.mv.Load(pre + "mv.raw") || !f.depth.Load(pre + "depth.raw"))
        { printf("FAIL: cannot load %s*.raw\n", pre.c_str()); return 1; }
    }
    if (FILE* meta = fopen((std::string(replayDir) + "\\frames.txt").c_str(), "r"))
    {
        char line[1024];
        while (fgets(line, sizeof(line), meta))
        {
            const int n = atoi(line) - first;
            if (n < 0 || n >= count) continue;
            if (const char* j = strstr(line, "jitter ")) sscanf(j, "jitter %f,%f", &frames[n].jx, &frames[n].jy);
            if (const char* m = strstr(line, "mv scale ")) sscanf(m, "mv scale %f,%f", &frames[n].sx, &frames[n].sy);
        }
        fclose(meta);
    }
    else
        printf("note: no frames.txt, jitter 0\n");
    Raw out0;
    {
        char a[64];
        const std::string dir = std::string(replayDir) + "\\";
        bool ok = false;
        for (const char* fmt : {"%03d_output.raw", "deckyupscaler_in_%03d_output.raw", "aiss_in_%03d_output.raw"})
        {
            snprintf(a, sizeof(a), fmt, first);
            if ((ok = out0.Load(dir + a))) break;
        }
        if (!ok) { printf("FAIL: no output dump for the display size\n"); return 1; }
    }
    const unsigned rw = frames[0].depth.w, rh = frames[0].depth.h, outW = out0.w, outH = out0.h;
    if (crop)
        for (Frame& f : frames) f.color.Crop(rw, rh), f.mv.Crop(rw, rh);
    if (color16)
        for (Frame& f : frames) f.color.ToHalf();
    if (depthPadW || depthBaseX || depthBaseY)
    {
        const unsigned W = std::max(depthPadW, rw + depthBaseX), H = std::max(depthPadH, rh + depthBaseY);
        for (Frame& f : frames) f.depth.Embed(W, H, depthBaseX, depthBaseY);
    }
    const Raw& c0 = frames[0].color;
    const Raw& m0 = frames[0].mv;
    const Raw& d0 = frames[0].depth;
    printf("replay %s frames %d..%d: render %ux%u, colour %ux%u fmt %u, mv %ux%u fmt %u, depth %ux%u %s -> %ux%u\n", replayDir,
           first, first + count - 1, rw, rh, c0.w, c0.h, c0.format, m0.w, m0.h, m0.format, d0.w, d0.h,
           depth32f ? "R32_FLOAT" : "R32_TYPELESS+DSV", outW, outH);

    // ---- capability query, as games do before creating the feature ----
    if (!ffx)
    {
        Params* caps = nullptr;
        int available = 0;
        NVSDK_NGX_D3D12_GetCapabilityParameters(&caps);
        caps->Get("SuperSampling.Available", &available);
        void* optimal = nullptr;
        caps->Get("DLSSOptimalSettingsCallback", &optimal);
        caps->Set("Width", outW);
        caps->Set("Height", outH);
        caps->Set("PerfQualityValue", perf);
        if (optimal) reinterpret_cast<Result (*)(Params*)>(optimal)(caps);
        unsigned ow = 0, oh = 0;
        caps->Get("OutWidth", &ow);
        caps->Get("OutHeight", &oh);
        printf("SuperSampling.Available=%d, optimal render size for %ux%u (perf %d): %ux%u\n", available, outW, outH, perf, ow, oh);
        NVSDK_NGX_D3D12_DestroyParameters(caps);
        if (!available || (!anySize && (ow != rw || oh != rh))) { printf("FAIL: capability parameters\n"); return 1; }
    }

    // ---- resources ----
    ID3D12Heap* heap = nullptr;
    UINT64 heapCursor = 0;
    if (placed != UINT64(-1))
    {
        D3D12_HEAP_DESC hd = {};
        hd.SizeInBytes = 512ull << 20;
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;  // tier 1 safe for the RT/DS inputs; others go committed
        CHECK(dev->CreateHeap(&hd, IID_PPV_ARGS(&heap)));
        heapCursor = placed;
        printf("placed RT/DS resources from heap offset %llu\n", (unsigned long long)placed);
    }
    auto makeTex = [&](unsigned w, unsigned h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initial) {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w, d.Height = h, d.DepthOrArraySize = 1, d.MipLevels = 1, d.Format = fmt, d.SampleDesc = {1, 0};
        d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        d.Flags = flags;
        ID3D12Resource* r = nullptr;
        const bool rtds = flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        if (heap && rtds)
        {
            const D3D12_RESOURCE_ALLOCATION_INFO ai = dev->GetResourceAllocationInfo(0, 1, &d);
            heapCursor = (heapCursor + ai.Alignment - 1) / ai.Alignment * ai.Alignment;
            if (FAILED(dev->CreatePlacedResource(heap, heapCursor, &d, initial, nullptr, IID_PPV_ARGS(&r)))) return (ID3D12Resource*)nullptr;
            heapCursor += ai.SizeInBytes + placed;  // keep the next one off the natural packing too
        }
        else
        {
            D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
            if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, initial, nullptr, IID_PPV_ARGS(&r)))) return (ID3D12Resource*)nullptr;
        }
        return r;
    };
    auto makeBuffer = [&](UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = size, d.Height = 1, d.DepthOrArraySize = 1, d.MipLevels = 1, d.SampleDesc = {1, 0};
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES hp = {type};
        ID3D12Resource* r = nullptr;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    };
    auto barrier = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        if (from == to) return;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = from, b.Transition.StateAfter = to;
        list->ResourceBarrier(1, &b);
    };
    std::vector<ID3D12Resource*> uploads;
    // Uploads raw pixels (copy footprint in the copyable format) into a texture created in COPY_DEST, then moves it
    // to its final state.
    auto upload = [&](ID3D12Resource* tex, const Raw& r, DXGI_FORMAT copyFormat, D3D12_RESOURCE_STATES finalState) {
        D3D12_RESOURCE_DESC d = tex->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
        UINT64 total = 0;
        dev->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
        fp.Footprint.Format = copyFormat;
        ID3D12Resource* buf = makeBuffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        uint8_t* p = nullptr;
        buf->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < r.h; ++y)
            memcpy(p + fp.Offset + size_t(y) * fp.Footprint.RowPitch, &r.data[size_t(y) * r.w * r.bpp], size_t(r.w) * r.bpp);
        buf->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst = {tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src = {buf, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        src.PlacedFootprint = fp;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(tex, D3D12_RESOURCE_STATE_COPY_DEST, finalState);
        uploads.push_back(buf);
    };

    const DXGI_FORMAT outFormat = outR11 ? DXGI_FORMAT_R11G11B10_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (!outTexW) outTexW = outW, outTexH = outH;
    ID3D12Resource* output = makeTex(outTexW, outTexH, outFormat, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* scratch = gamework ? makeTex(1024, 1024, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) : nullptr;
    if (!output) { printf("FAIL: output\n"); return 1; }
    std::vector<ID3D12Resource*> textures(size_t(count) * 3);
    for (int i = 0; i < count; ++i)
    {
        const Frame& f = frames[i];
        const DXGI_FORMAT cf = DxgiFromVk(f.color.format), mf = DxgiFromVk(f.mv.format);
        if (cf == DXGI_FORMAT_UNKNOWN || mf == DXGI_FORMAT_UNKNOWN) { printf("FAIL: formats %u %u\n", f.color.format, f.mv.format); return 1; }
        ID3D12Resource* color = makeTex(f.color.w, f.color.h, cf, colorFlags, D3D12_RESOURCE_STATE_COPY_DEST);
        ID3D12Resource* mv    = makeTex(f.mv.w, f.mv.h, mf, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
        ID3D12Resource* depth = depth32f ? makeTex(f.depth.w, f.depth.h, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)
                                         : makeTex(f.depth.w, f.depth.h, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!color || !mv || !depth) { printf("FAIL: textures (formats %u %u %u)\n", f.color.format, f.mv.format, f.depth.format); return 1; }
        auto copyFmt = [](DXGI_FORMAT f) {
            return f == DXGI_FORMAT_R16G16B16A16_TYPELESS ? DXGI_FORMAT_R16G16B16A16_FLOAT : f == DXGI_FORMAT_R16G16_TYPELESS ? DXGI_FORMAT_R16G16_FLOAT : f;
        };
        upload(color, f.color, copyFmt(cf), inState);
        upload(mv, f.mv, copyFmt(mf), inState);
        upload(depth, f.depth, depth32f ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R32_TYPELESS, depthState);
        textures[i * 3 + 0] = color, textures[i * 3 + 1] = mv, textures[i * 3 + 2] = depth;
    }
    allocFence[0] = submit();
    waitFor(allocFence[0]);
    for (ID3D12Resource* b : uploads) b->Release();
    uploads.clear();

    // ---- feature (NGX) or context (FFX) ----
    Params* p = nullptr;
    Handle* feature = nullptr;
    ffxContext fx = nullptr;
    // DLSS create flags as FFX upscale flags: HDR (1), low-res MVs (2, else display-res), inverted depth (8), auto
    // exposure (64)
    const bool ffxDepthInv = createFlags & 8;
    if (ffx)
    {
        struct { ffxCreateContextDescHeader header; ID3D12Device* device; } backend = {};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.device      = dev;
        ffxCreateContextDescUpscale create = {};
        create.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        create.header.pNext = &backend.header;
        create.flags = FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION | ((createFlags & 1) ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0) |
                       ((createFlags & 2) ? 0 : FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) |
                       (ffxDepthInv ? FFX_UPSCALE_ENABLE_DEPTH_INVERTED : 0) |
                       ((createFlags & 64) ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0);
        create.maxRenderSize  = {rw, rh};
        create.maxUpscaleSize = {outW, outH};
        const ffxReturnCode_t rc = ffxCreate(&fx, &create.header, nullptr);
        if (rc != FFX_API_RETURN_OK) { printf("FAIL: ffxCreateContext = %u\n", rc); return 1; }
        printf("FFX context %ux%u -> %ux%u, flags 0x%x\n", rw, rh, outW, outH, create.flags);
    }
    else
    {
    NVSDK_NGX_D3D12_AllocateParameters(&p);
    p->Set("Width", rw);
    p->Set("Height", rh);
    p->Set("OutWidth", outW);
    p->Set("OutHeight", outH);
    p->Set("PerfQualityValue", perf);
    p->Set("DLSS.Feature.Create.Flags", createFlags);
    p->Set("DLSS.Enable.Output.Subrects", 0);
    alloc[0]->Reset();
    list->Reset(alloc[0], nullptr);
    if (NVSDK_NGX_D3D12_CreateFeature(list, 1, p, &feature) != kSuccess) { printf("FAIL: CreateFeature\n"); return 1; }
    allocFence[0] = submit();
    waitFor(allocFence[0]);
    }

    // GPU time per upscale: a timestamp pair around each, resolved into a readback buffer read at the end
    const int evals = count * loops;
    ID3D12QueryHeap* tsHeap = nullptr;
    ID3D12Resource* tsBuf = nullptr;
    if (gpuTime)
    {
        D3D12_QUERY_HEAP_DESC qh = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, UINT(evals * 2), 0};
        CHECK(dev->CreateQueryHeap(&qh, IID_PPV_ARGS(&tsHeap)));
        tsBuf = makeBuffer(UINT64(evals) * 2 * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!tsBuf) { printf("FAIL: timestamp buffer\n"); return 1; }
    }

    ID3D12DescriptorHeap* uavHeap = nullptr;
    if (gamework)
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        CHECK(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&uavHeap)));
        dev->CreateUnorderedAccessView(scratch, nullptr, nullptr, uavHeap->GetCPUDescriptorHandleForHeapStart());
    }
    ID3D12DescriptorHeap* cpuHeap = nullptr;
    if (gamework)
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
        CHECK(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cpuHeap)));
        dev->CreateUnorderedAccessView(scratch, nullptr, nullptr, cpuHeap->GetCPUDescriptorHandleForHeapStart());
    }

    const DWORD t0 = GetTickCount();
    for (int i = 0; i < count * loops; ++i)
    {
        const int k = i % kAlloc;
        for (auto& [frame, code] : ctlSeq)
            if (frame == i && i > 0) { writeCtl(code); printf("control %d before frame %d\n", code, i); }
        waitFor(allocFence[k]);
        alloc[k]->Reset();
        list->Reset(alloc[k], nullptr);
        if (gamework)
        {
            list->SetDescriptorHeaps(1, &uavHeap);
            const float v[4] = {float(i), 0, 0, 0};
            list->ClearUnorderedAccessViewFloat(uavHeap->GetGPUDescriptorHandleForHeapStart(), cpuHeap->GetCPUDescriptorHandleForHeapStart(), scratch, v, 0, nullptr);
        }
        const Frame& f = frames[i % count];
        if (!ffx)
        {
        p->Set("Color", static_cast<void*>(textures[(i % count) * 3 + 0]));
        p->Set("MotionVectors", static_cast<void*>(textures[(i % count) * 3 + 1]));
        p->Set("Depth", static_cast<void*>(textures[(i % count) * 3 + 2]));
        p->Set("Output", static_cast<void*>(output));
        p->Set("MV.Scale.X", f.sx);
        p->Set("MV.Scale.Y", f.sy);
        p->Set("Jitter.Offset.X", f.jx);
        p->Set("Jitter.Offset.Y", f.jy);
        p->Set("Reset", i == 0 ? 1 : 0);
        p->Set("FrameTimeDeltaInMsec", 16.6f);
        if (subrect)
        {
            p->Set("DLSS.Render.Subrect.Dimensions.Width", rw);
            p->Set("DLSS.Render.Subrect.Dimensions.Height", rh);
        }
        p->Set("DLSS.Input.Depth.Subrect.Base.X", depthBaseX);
        p->Set("DLSS.Input.Depth.Subrect.Base.Y", depthBaseY);
        }
        if (gpuTime) list->EndQuery(tsHeap, D3D12_QUERY_TYPE_TIMESTAMP, UINT(i * 2));
        if (ffx)
        {
            ffxDispatchDescUpscale up = {};
            up.header.type   = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
            up.commandList   = list;
            up.color         = ffxApiGetResourceDX12(textures[(i % count) * 3 + 0], FFX_API_RESOURCE_STATE_COMPUTE_READ);
            up.motionVectors = ffxApiGetResourceDX12(textures[(i % count) * 3 + 1], FFX_API_RESOURCE_STATE_COMPUTE_READ);
            up.depth         = ffxApiGetResourceDX12(textures[(i % count) * 3 + 2], FFX_API_RESOURCE_STATE_COMPUTE_READ);
            up.output        = ffxApiGetResourceDX12(output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            up.jitterOffset      = {f.jx, f.jy};
            up.motionVectorScale = {f.sx, f.sy};
            up.renderSize  = {rw, rh};
            up.upscaleSize = {outW, outH};
            up.frameTimeDelta = 16.6f;
            up.preExposure    = 1.0f;
            up.reset          = i == 0;
            up.cameraNear = ffxDepthInv ? 100000.0f : 0.1f;
            up.cameraFar  = ffxDepthInv ? 0.1f : 100000.0f;
            up.cameraFovAngleVertical  = 60.0f * 3.14159265f / 180.0f;
            up.viewSpaceToMetersFactor = 1.0f;
            const ffxReturnCode_t rc = ffxRun(&fx, &up.header);
            if (rc != FFX_API_RETURN_OK) { printf("FAIL: ffxDispatch %d = %u\n", i, rc); return 1; }
        }
        else if (NVSDK_NGX_D3D12_EvaluateFeature(list, feature, p, nullptr) != kSuccess) { printf("FAIL: EvaluateFeature %d\n", i); return 1; }
        if (gpuTime)
        {
            list->EndQuery(tsHeap, D3D12_QUERY_TYPE_TIMESTAMP, UINT(i * 2 + 1));
            list->ResolveQueryData(tsHeap, D3D12_QUERY_TYPE_TIMESTAMP, UINT(i * 2), 2, tsBuf, UINT64(i) * 2 * sizeof(uint64_t));
        }
        allocFence[k] = submit();
        if (sync) waitFor(allocFence[k]);
    }
    waitFor(fenceValue);
    printf("%d evaluations in %lu ms\n", count * loops, GetTickCount() - t0);
    if (gpuTime)
    {
        UINT64 freq = 1;
        queue->GetTimestampFrequency(&freq);
        const uint64_t* ts = nullptr;
        CHECK(tsBuf->Map(0, nullptr, (void**)&ts));
        const int skip = std::min(20, evals / 4);
        std::vector<double> us;
        for (int i = skip; i < evals; ++i)
            us.push_back(double(ts[i * 2 + 1] - ts[i * 2]) * 1e6 / double(freq));
        tsBuf->Unmap(0, nullptr);
        if (!us.empty())
        {
            double sum = 0;
            for (double v : us) sum += v;
            std::vector<double> sorted = us;
            std::sort(sorted.begin(), sorted.end());
            printf("harness GPU time per upscale: mean %.1f us, median %.1f, min %.1f, max %.1f over %zu evaluations\n",
                   sum / us.size(), sorted[sorted.size() / 2], sorted.front(), sorted.back(), us.size());
        }
    }

    // ---- read back ----
    const unsigned obpp = outR11 ? 4 : 8;
    const D3D12_RESOURCE_DESC od = output->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT64 total = 0;
    dev->GetCopyableFootprints(&od, 0, 1, 0, &fp, nullptr, nullptr, &total);
    ID3D12Resource* rb = makeBuffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const int k = 0;
    waitFor(allocFence[k]);
    alloc[k]->Reset();
    list->Reset(alloc[k], nullptr);
    barrier(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst = {rb, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src = {output, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    src.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    barrier(output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    waitFor(submit());
    const uint8_t* map = nullptr;
    CHECK(rb->Map(0, nullptr, (void**)&map));
    std::vector<uint16_t> out(size_t(outW) * outH * 4);
    for (unsigned y = 0; y < outH; ++y)
    {
        const uint8_t* row = map + fp.Offset + size_t(y) * fp.Footprint.RowPitch;
        if (!outR11)
            memcpy(&out[size_t(y) * outW * 4], row, size_t(outW) * obpp);
        else
            for (unsigned x = 0; x < outW; ++x)
            {
                uint32_t v;
                memcpy(&v, row + x * 4, 4);
                uint16_t* px = &out[(size_t(y) * outW + x) * 4];
                px[0] = uint16_t((v & 0x7ff) << 4), px[1] = uint16_t(((v >> 11) & 0x7ff) << 4), px[2] = uint16_t(((v >> 22) & 0x3ff) << 5), px[3] = 0x3c00;
            }
    }
    rb->Unmap(0, nullptr);
    size_t nonFinite = 0;
    double sum = 0;
    for (size_t i = 0; i < out.size(); ++i)
    {
        if ((i & 3) == 3) continue;
        const float v = HalfToFloat(out[i]);
        if (!std::isfinite(v)) ++nonFinite; else sum += v;
    }
    printf("output: mean %.4f, %zu non-finite channel values\n", sum / (out.size() * 3 / 4), nonFinite);
    if (saveFile)
        if (FILE* f = fopen(saveFile, "wb")) { fwrite(out.data(), 2, out.size(), f); fclose(f); printf("saved %s\n", saveFile); }

    if (ffx)
        ffxDestroy(&fx, nullptr);
    else
    {
        NVSDK_NGX_D3D12_ReleaseFeature(feature);
        NVSDK_NGX_D3D12_DestroyParameters(p);
        NVSDK_NGX_D3D12_Shutdown1(dev);
    }
    printf("PASS\n");
    return 0;
}
