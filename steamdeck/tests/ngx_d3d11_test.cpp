// End-to-end check of the proxy's D3D11 path on DXVK: drives nvngx.dll the way a game's NGX code does and verifies
// the upscaled output against the input. Usage (under Proton/Wine): ngx_d3d11_test.exe <path to nvngx.dll>
// SPDX-License-Identifier: MIT
#include <d3d11.h>
#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dll = argc > 1 ? argv[1] : "nvngx.dll";
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
    std::vector<uint16_t> colorData(rw * rh * 4), mvData(rw * rh * 2, 0);
    for (unsigned y = 0; y < rh; ++y)
        for (unsigned x = 0; x < rw; ++x)
        {
            float rgb[3];
            Scene((x + 0.5f) / rw, (y + 0.5f) / rh, rgb);
            for (int c = 0; c < 3; ++c) colorData[(y * rw + x) * 4 + c] = FloatToHalf(rgb[c]);
            colorData[(y * rw + x) * 4 + 3] = FloatToHalf(1.0f);
        }
    ID3D11Texture2D* color  = tex(rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_SHADER_RESOURCE, colorData.data(), rw * 8);
    ID3D11Texture2D* mv     = tex(rw, rh, DXGI_FORMAT_R16G16_FLOAT, D3D11_BIND_SHADER_RESOURCE, mvData.data(), rw * 4);
    ID3D11Texture2D* depth  = tex(rw, rh, DXGI_FORMAT_R32_TYPELESS, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL, nullptr, 0);
    ID3D11Texture2D* output = tex(outW, outH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, nullptr, 0);
    if (!color || !mv || !depth || !output) { printf("FAIL: textures\n"); return 1; }
    D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = {DXGI_FORMAT_D32_FLOAT, D3D11_DSV_DIMENSION_TEXTURE2D};
    ID3D11DepthStencilView* dsv = nullptr;
    dev->CreateDepthStencilView(depth, &dsvd, &dsv);
    ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 0.25f, 0);

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
    ctx->Unmap(staging, 0);
    const double mean = err / count;
    printf("output vs reference: mean |err| %.4f, max %.4f, %u of %u channel values off by >0.1\n", mean, maxErr, bad, count);

    NVSDK_NGX_D3D11_ReleaseFeature(feature);
    NVSDK_NGX_D3D11_DestroyParameters(p);
    NVSDK_NGX_D3D11_Shutdown1(dev);
    const bool pass = mean < 0.02 && bad < count / 100;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
