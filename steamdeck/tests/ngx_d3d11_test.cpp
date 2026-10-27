// End-to-end check of the proxy's D3D11 path on DXVK: drives nvngx.dll the way a game's NGX code does and verifies
// the upscaled output against the input.
// Usage (under Proton/Wine): ngx_d3d11_test.exe <path to nvngx.dll> [options]
//   --pad N       colour and motion vector textures N pixels wider and taller than the render size (as some games
//                 allocate them), with junk outside the render region; the output must match the unpadded run
//   --save FILE   write the output (RGBA16F) to FILE
//   --cmp FILE    compare the output with one written by --save
//   --log FILE    print to FILE instead of stdout (Proton does not always pass a console through)
//   --replay DIR  instead of the synthetic scene, replay dumped inputs (NNN_*.raw or aiss_in_NNN_*.raw, with
//                 jitter / MV scale from DIR/frames.txt): the textures keep the dumped extents, the render size is the
//                 depth dump's (the proxy dumps its render-size depth copy), the display size the output dump's
//   --first N --count K   frames of the replay (default 0, 8)
//   --crop        replay with colour and motion vectors cropped to the render size (the exact-size reference)
//   --color16     replay B10G11R11 colour as RGBA16F (the same values)
//   --loops L     play the replayed frames L times (reset only on the first)
//   --sync        wait for the GPU after every replayed frame
// SPDX-License-Identifier: MIT
#include <d3d11.h>
#include <windows.h>

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
NGX_FN(NVSDK_NGX_D3D11_Init, unsigned long long, const wchar_t*, ID3D11Device*, const void*, int)
NGX_FN(NVSDK_NGX_D3D11_GetCapabilityParameters, Params**)
NGX_FN(NVSDK_NGX_D3D11_AllocateParameters, Params**)
NGX_FN(NVSDK_NGX_D3D11_CreateFeature, ID3D11DeviceContext*, int, Params*, Handle**)
NGX_FN(NVSDK_NGX_D3D11_EvaluateFeature, ID3D11DeviceContext*, const Handle*, Params*, void*)
NGX_FN(NVSDK_NGX_D3D11_ReleaseFeature, Handle*)
NGX_FN(NVSDK_NGX_D3D11_DestroyParameters, Params*)
NGX_FN(NVSDK_NGX_D3D11_Shutdown1, ID3D11Device*)

static uint16_t FloatToHalf(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int            exp  = int((x >> 23) & 0xff) - 127 + 15;
    uint32_t       man  = (x >> 13) & 0x3ff;
    if (exp <= 0) return uint16_t(sign);
    if (exp >= 31) return uint16_t(sign | 0x7c00);
    return uint16_t(sign | (exp << 10) | man);
}

static float HalfToFloat(uint16_t h)
{
    const uint32_t exp = (h >> 10) & 31, man = h & 1023;
    const float    v   = exp == 0 ? std::ldexp(float(man), -24) : std::ldexp(float(man | 1024), int(exp) - 25);
    return (h & 0x8000) ? -v : v;
}

// Smooth, textured test scene in linear HDR.
static void Scene(float u, float v, float rgb[3])
{
    rgb[0] = 0.25f + 0.5f * u;
    rgb[1] = 0.25f + 0.5f * v;
    rgb[2] = 0.5f + 0.25f * std::sin(20.0f * u) * std::sin(20.0f * v);
}

// Waits for the GPU. Inputs must be fully uploaded before the proxy reads them: DXVK uploads initial data separately,
// and without this the replayed output depended on timing.
static void WaitIdle(ID3D11Device* dev, ID3D11DeviceContext* ctx)
{
    D3D11_QUERY_DESC qd = {D3D11_QUERY_EVENT, 0};
    ID3D11Query* q = nullptr;
    dev->CreateQuery(&qd, &q);
    ctx->End(q);
    BOOL done = FALSE;
    while (ctx->GetData(q, &done, sizeof(done), 0) != S_OK || !done) Sleep(1);
    q->Release();
}

static void Save(const std::vector<uint16_t>& out, const char* file)
{
    if (FILE* f = fopen(file, "wb")) { fwrite(out.data(), 2, out.size(), f); fclose(f); printf("saved %s\n", file); }
}

// Compares RGBA16F output with a file written by Save; passes when no colour channel is off by 0.01 or more.
static bool Compare(const std::vector<uint16_t>& out, const char* file)
{
    std::vector<uint16_t> ref(out.size());
    FILE* f = fopen(file, "rb");
    if (!f || fread(ref.data(), 2, ref.size(), f) != ref.size()) { printf("FAIL: cannot read %s\n", file); if (f) fclose(f); return false; }
    fclose(f);
    double sum = 0, sq = 0, mx = 0;
    size_t differ = 0, n = 0;
    for (size_t i = 0; i < out.size(); ++i)
    {
        if ((i & 3) == 3) continue;
        const double e = std::fabs(double(HalfToFloat(out[i])) - HalfToFloat(ref[i]));
        differ += out[i] != ref[i];
        sum += e, sq += e * e, mx = e > mx ? e : mx, ++n;
    }
    const double psnr = sq > 0 ? 10 * std::log10(1.0 / (sq / n)) : INFINITY;
    printf("output vs %s: %zu of %zu channel values differ, mean |diff| %.6f, max %.4f, PSNR %.1f dB\n", file, differ, n,
           sum / n, mx, psnr);
    return mx < 0.01;
}

// Raw input dump as the proxy writes them (AISS_DUMP_INPUTS / capture): uint32 {'AISR', w, h, VkFormat, bpp}, pixels.
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
    // B10G11R11 -> RGBA16F, exact (same exponent bias, shorter mantissas).
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
};

static DXGI_FORMAT DxgiFromVk(uint32_t vk)
{
    switch (vk)
    {
    case 97: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case 109: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case 122: return DXGI_FORMAT_R11G11B10_FLOAT;
    case 64: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case 37: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case 44: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case 83: return DXGI_FORMAT_R16G16_FLOAT;
    case 103: return DXGI_FORMAT_R32G32_FLOAT;
    case 100: return DXGI_FORMAT_R32_TYPELESS;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dll = argc > 1 ? argv[1] : "nvngx.dll";
    unsigned    pad = 0;
    const char* saveFile = nullptr;
    const char* cmpFile  = nullptr;
    const char* replayDir = nullptr;
    int         first = 0, replayCount = 8;
    bool        crop = false, color16 = false;
    int         loops = 1;
    bool        sync = false;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--pad" && i + 1 < argc) pad = unsigned(atoi(argv[++i]));
        else if (a == "--save" && i + 1 < argc) saveFile = argv[++i];
        else if (a == "--cmp" && i + 1 < argc) cmpFile = argv[++i];
        else if (a == "--log" && i + 1 < argc) freopen(argv[++i], "w", stdout);
        else if (a == "--replay" && i + 1 < argc) replayDir = argv[++i];
        else if (a == "--first" && i + 1 < argc) first = atoi(argv[++i]);
        else if (a == "--count" && i + 1 < argc) replayCount = atoi(argv[++i]);
        else if (a == "--crop") crop = true;
        else if (a == "--color16") color16 = true;
        else if (a == "--loops" && i + 1 < argc) loops = atoi(argv[++i]);
        else if (a == "--sync") sync = true;
        else { printf("FAIL: unknown option %s\n", argv[i]); return 1; }
    }
    HMODULE ngx = LoadLibraryA(dll);
    if (!ngx) { printf("FAIL: cannot load %s\n", dll); return 1; }
#define LOAD(name) name = reinterpret_cast<PFN_##name>(GetProcAddress(ngx, #name)); if (!name) { printf("FAIL: missing %s\n", #name); return 1; }
    LOAD(NVSDK_NGX_D3D11_Init) LOAD(NVSDK_NGX_D3D11_GetCapabilityParameters) LOAD(NVSDK_NGX_D3D11_AllocateParameters)
    LOAD(NVSDK_NGX_D3D11_CreateFeature) LOAD(NVSDK_NGX_D3D11_EvaluateFeature) LOAD(NVSDK_NGX_D3D11_ReleaseFeature)
    LOAD(NVSDK_NGX_D3D11_DestroyParameters) LOAD(NVSDK_NGX_D3D11_Shutdown1)

    ID3D11Device*        dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    D3D_FEATURE_LEVEL    fl  = D3D_FEATURE_LEVEL_11_1;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &fl, 1, D3D11_SDK_VERSION, &dev, nullptr, &ctx)))
    { printf("FAIL: D3D11CreateDevice\n"); return 1; }

    if (NVSDK_NGX_D3D11_Init(0x1234, L".", dev, nullptr, 0x14) != kSuccess) { printf("FAIL: Init\n"); return 1; }

    if (replayDir)
    {
        // ---- replay of dumped inputs ----
        const int count = replayCount;
        struct Frame { Raw color, mv, depth; float jx = 0, jy = 0, sx = 1, sy = 1; };
        std::vector<Frame> frames(count);
        for (int i = 0; i < count; ++i)
        {
            const int n = first + i;
            char a[64], b[64];
            snprintf(a, sizeof(a), "%03d_", n);
            snprintf(b, sizeof(b), "aiss_in_%03d_", n);
            const std::string dir = std::string(replayDir) + "\\";
            std::string pre = dir + a;
            Raw probe;
            if (!probe.Load(pre + "depth.raw")) pre = dir + b;
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
            snprintf(a, sizeof(a), "%03d_output.raw", first);
            std::string dir = std::string(replayDir) + "\\";
            if (!out0.Load(dir + a))
            {
                snprintf(a, sizeof(a), "aiss_in_%03d_output.raw", first);
                if (!out0.Load(dir + a)) { printf("FAIL: no output dump for the display size\n"); return 1; }
            }
        }
        const unsigned rw = frames[0].depth.w, rh = frames[0].depth.h, outW = out0.w, outH = out0.h;
        const Raw& c0 = frames[0].color;
        const Raw& m0 = frames[0].mv;
        printf("replay %s frames %d..%d: render %ux%u, colour %ux%u fmt %u, mv %ux%u fmt %u -> %ux%u%s\n", replayDir, first,
               first + count - 1, rw, rh, c0.w, c0.h, c0.format, m0.w, m0.h, m0.format, outW, outH,
               crop ? " (colour and mv cropped to the render size)" : "");
        if (crop)
            for (Frame& f : frames) f.color.Crop(rw, rh), f.mv.Crop(rw, rh);
        if (color16)
            for (Frame& f : frames) f.color.ToHalf();
        auto tex = [&](const Raw& r, UINT bind) {
            D3D11_TEXTURE2D_DESC d = {r.w, r.h, 1, 1, DxgiFromVk(r.format), {1, 0}, D3D11_USAGE_DEFAULT, bind, 0, 0};
            D3D11_SUBRESOURCE_DATA sd = {r.data.data(), r.w * r.bpp, 0};
            ID3D11Texture2D* t = nullptr;
            if (d.Format != DXGI_FORMAT_UNKNOWN) dev->CreateTexture2D(&d, &sd, &t);
            return t;
        };
        Params* p = nullptr;
        NVSDK_NGX_D3D11_AllocateParameters(&p);
        p->Set("Width", rw);
        p->Set("Height", rh);
        p->Set("OutWidth", outW);
        p->Set("OutHeight", outH);
        p->Set("PerfQualityValue", 0);
        p->Set("DLSS.Feature.Create.Flags", (1 << 0) | (1 << 1) | (1 << 3));  // HDR, low-res MVs, inverted depth
        Handle* feature = nullptr;
        if (NVSDK_NGX_D3D11_CreateFeature(ctx, 1, p, &feature) != kSuccess) { printf("FAIL: CreateFeature\n"); return 1; }
        D3D11_TEXTURE2D_DESC od = {outW, outH, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, {1, 0}, D3D11_USAGE_DEFAULT,
                                   D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0, 0};
        ID3D11Texture2D* output = nullptr;
        dev->CreateTexture2D(&od, nullptr, &output);
        // All inputs are uploaded before the first evaluation.
        std::vector<ID3D11Texture2D*> textures(size_t(count) * 3);
        for (int i = 0; i < count; ++i)
        {
            const Frame& f = frames[i];
            textures[i * 3 + 0] = tex(f.color, D3D11_BIND_SHADER_RESOURCE);
            textures[i * 3 + 1] = tex(f.mv, D3D11_BIND_SHADER_RESOURCE);
            textures[i * 3 + 2] = tex(f.depth, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL);
            if (!textures[i * 3] || !textures[i * 3 + 1] || !textures[i * 3 + 2] || !output)
            { printf("FAIL: textures (formats %u %u %u)\n", f.color.format, f.mv.format, f.depth.format); return 1; }
        }
        ctx->Flush();
        WaitIdle(dev, ctx);
        for (int i = 0; i < count * loops; ++i)
        {
            const Frame& f = frames[i % count];
            ID3D11Texture2D* color = textures[(i % count) * 3 + 0];
            ID3D11Texture2D* mv    = textures[(i % count) * 3 + 1];
            ID3D11Texture2D* depth = textures[(i % count) * 3 + 2];
            p->Set("Color", static_cast<void*>(color));
            p->Set("Depth", static_cast<void*>(depth));
            p->Set("MotionVectors", static_cast<void*>(mv));
            p->Set("Output", static_cast<void*>(output));
            p->Set("MV.Scale.X", f.sx);
            p->Set("MV.Scale.Y", f.sy);
            p->Set("Jitter.Offset.X", f.jx);
            p->Set("Jitter.Offset.Y", f.jy);
            p->Set("Reset", i == 0 ? 1 : 0);
            if (NVSDK_NGX_D3D11_EvaluateFeature(ctx, feature, p, nullptr) != kSuccess) { printf("FAIL: EvaluateFeature %d\n", i); return 1; }
            ctx->Flush();
            if (sync)
                WaitIdle(dev, ctx);  // rules out timing-dependent results
        }
        for (ID3D11Texture2D* t : textures) t->Release();
        D3D11_TEXTURE2D_DESC sd = od;
        sd.Usage = D3D11_USAGE_STAGING, sd.BindFlags = 0, sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging = nullptr;
        dev->CreateTexture2D(&sd, nullptr, &staging);
        ctx->CopyResource(staging, output);
        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map))) { printf("FAIL: Map\n"); return 1; }
        std::vector<uint16_t> out(size_t(outW) * outH * 4);
        for (unsigned y = 0; y < outH; ++y)
            memcpy(&out[size_t(y) * outW * 4], static_cast<const uint8_t*>(map.pData) + y * map.RowPitch, outW * 8);
        ctx->Unmap(staging, 0);
        bool pass = true;
        if (saveFile) Save(out, saveFile);
        if (cmpFile) pass = Compare(out, cmpFile);
        NVSDK_NGX_D3D11_ReleaseFeature(feature);
        NVSDK_NGX_D3D11_DestroyParameters(p);
        NVSDK_NGX_D3D11_Shutdown1(dev);
        printf("%s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    Params* caps = nullptr;
    int available = 0;
    NVSDK_NGX_D3D11_GetCapabilityParameters(&caps);
    caps->Get("SuperSampling.Available", &available);
    void* optimal = nullptr;
    caps->Get("DLSSOptimalSettingsCallback", &optimal);
    const unsigned outW = 1280, outH = 800;
    caps->Set("Width", outW);
    caps->Set("Height", outH);
    caps->Set("PerfQualityValue", 0);  // MaxPerf
    reinterpret_cast<Result (*)(Params*)>(optimal)(caps);
    unsigned rw = 0, rh = 0;
    caps->Get("OutWidth", &rw);
    caps->Get("OutHeight", &rh);
    printf("SuperSampling.Available=%d, MaxPerf render size for %ux%u: %ux%u\n", available, outW, outH, rw, rh);
    NVSDK_NGX_D3D11_DestroyParameters(caps);
    if (!available || rw != 640 || rh != 400) { printf("FAIL: capability parameters\n"); return 1; }

    auto tex = [&](UINT w, UINT h, DXGI_FORMAT fmt, UINT bind, const void* data, UINT pitch) {
        D3D11_TEXTURE2D_DESC d = {w, h, 1, 1, fmt, {1, 0}, D3D11_USAGE_DEFAULT, bind, 0, 0};
        D3D11_SUBRESOURCE_DATA sd = {data, pitch, 0};
        ID3D11Texture2D* t = nullptr;
        dev->CreateTexture2D(&d, data ? &sd : nullptr, &t);
        return t;
    };
    // Inputs are allocated (rw + pad) x (rh + pad); the region outside the current render size holds junk that must
    // never reach the output: bright magenta colour and large motion vectors.
    const unsigned tw = rw + pad, th = rh + pad;
    std::vector<uint16_t> colorData(tw * th * 4), mvData(tw * th * 2);
    auto fillInputs = [&](unsigned w, unsigned h) {
        for (unsigned y = 0; y < th; ++y)
            for (unsigned x = 0; x < tw; ++x)
            {
                uint16_t* c = &colorData[(y * tw + x) * 4];
                uint16_t* m = &mvData[(y * tw + x) * 2];
                if (x < w && y < h)
                {
                    float rgb[3];
                    Scene((x + 0.5f) / w, (y + 0.5f) / h, rgb);
                    for (int i = 0; i < 3; ++i) c[i] = FloatToHalf(rgb[i]);
                    m[0] = m[1] = 0;
                }
                else
                {
                    c[0] = c[2] = FloatToHalf(4.0f), c[1] = 0;
                    m[0] = FloatToHalf(37.0f), m[1] = FloatToHalf(-23.0f);
                }
                c[3] = FloatToHalf(1.0f);
            }
    };
    fillInputs(rw, rh);
    ID3D11Texture2D* color  = tex(tw, th, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_SHADER_RESOURCE, colorData.data(), tw * 8);
    ID3D11Texture2D* mv     = tex(tw, th, DXGI_FORMAT_R16G16_FLOAT, D3D11_BIND_SHADER_RESOURCE, mvData.data(), tw * 4);
    ID3D11Texture2D* depth  = tex(rw, rh, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL, nullptr, 0);
    ID3D11Texture2D* output = tex(outW, outH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, nullptr, 0);
    if (!color || !mv || !depth || !output) { printf("FAIL: textures\n"); return 1; }
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = {DXGI_FORMAT_D32_FLOAT, D3D11_DSV_DIMENSION_TEXTURE2D};
    ID3D11DepthStencilView* dsv = nullptr;
    dev->CreateDepthStencilView(depth, &dsvd, &dsv);
    ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 0.25f, 0);
    ctx->Flush();
    WaitIdle(dev, ctx);

    Params* p = nullptr;
    NVSDK_NGX_D3D11_AllocateParameters(&p);
    p->Set("Width", rw);
    p->Set("Height", rh);
    p->Set("OutWidth", outW);
    p->Set("OutHeight", outH);
    p->Set("PerfQualityValue", 0);
    p->Set("DLSS.Feature.Create.Flags", (1 << 0) | (1 << 1) | (1 << 3));  // HDR, low-res MVs, inverted depth
    Handle* feature = nullptr;
    if (NVSDK_NGX_D3D11_CreateFeature(ctx, 1, p, &feature) != kSuccess) { printf("FAIL: CreateFeature\n"); return 1; }

    p->Set("Color", static_cast<void*>(color));
    p->Set("Depth", static_cast<void*>(depth));
    p->Set("MotionVectors", static_cast<void*>(mv));
    p->Set("Output", static_cast<void*>(output));
    p->Set("MV.Scale.X", 1.0f);
    p->Set("MV.Scale.Y", 1.0f);
    const DWORD t0 = GetTickCount();
    const int frames = 64;
    for (int i = 0; i < frames; ++i)
    {
        // Halton(2,3) jitter, as games use.
        auto halton = [](int i, int b) { float f = 1, r = 0; for (; i > 0; i /= b) { f /= b; r += f * (i % b); } return r; };
        p->Set("Jitter.Offset.X", halton(i + 1, 2) - 0.5f);
        p->Set("Jitter.Offset.Y", halton(i + 1, 3) - 0.5f);
        p->Set("Reset", i == 0 ? 1 : 0);
        if (NVSDK_NGX_D3D11_EvaluateFeature(ctx, feature, p, nullptr) != kSuccess) { printf("FAIL: EvaluateFeature %d\n", i); return 1; }
        if (i == 0 || i == frames - 1)
            printf("evaluated frame %d\n", i);
        ctx->Flush();
    }

    D3D11_TEXTURE2D_DESC sd = {outW, outH, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, {1, 0}, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0};
    ID3D11Texture2D* staging = nullptr;
    dev->CreateTexture2D(&sd, nullptr, &staging);
    ctx->CopyResource(staging, output);
    D3D11_MAPPED_SUBRESOURCE map = {};
    if (FAILED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map))) { printf("FAIL: Map\n"); return 1; }
    printf("%d evaluations in %lu ms (CPU wall time incl. sync)\n", frames, GetTickCount() - t0);

    // Compare against the scene evaluated at output resolution (it is smooth enough to upscale near-perfectly).
    double err = 0, maxErr = 0;
    unsigned bad = 0, count = 0;
    for (unsigned y = 8; y < outH - 8; ++y)
        for (unsigned x = 8; x < outW - 8; ++x)
        {
            const uint16_t* px = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(map.pData) + y * map.RowPitch) + x * 4;
            float ref[3];
            Scene((x + 0.5f) / outW, (y + 0.5f) / outH, ref);
            for (int c = 0; c < 3; ++c)
            {
                const float v = HalfToFloat(px[c]);
                const double e = std::fabs(v - ref[c]);
                if (!std::isfinite(v) || e > 0.1) ++bad;
                err += e;
                maxErr = e > maxErr ? e : maxErr;
                ++count;
            }
        }
    const double mean = err / count;
    printf("output vs reference: mean |err| %.4f, max %.4f, %u of %u channel values off by >0.1\n", mean, maxErr, bad, count);

    // Whole output as RGBA16F, for --save / --cmp.
    std::vector<uint16_t> out(size_t(outW) * outH * 4);
    for (unsigned y = 0; y < outH; ++y)
        memcpy(&out[size_t(y) * outW * 4], static_cast<const uint8_t*>(map.pData) + y * map.RowPitch, outW * 8);
    ctx->Unmap(staging, 0);
    bool cmpPass = true;
    if (saveFile)
        Save(out, saveFile);
    if (cmpFile)
        cmpPass = Compare(out, cmpFile);

    NVSDK_NGX_D3D11_ReleaseFeature(feature);
    NVSDK_NGX_D3D11_DestroyParameters(p);
    NVSDK_NGX_D3D11_Shutdown1(dev);
    const bool pass = mean < 0.02 && bad < count / 100 && cmpPass;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
