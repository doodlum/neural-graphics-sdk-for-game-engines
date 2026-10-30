// SPDX-License-Identifier: MIT
#include "params.h"

#include "common.h"
#include "cost_model.h"

#include <algorithm>
#include <type_traits>

// ---------------------------------------------------------------------------------------------------------------------
// MSVC vtable for NVSDK_NGX_Parameter.
//
// The class declares Set(ull, float, double, uint, int, ID3D11Resource*, ID3D12Resource*, void*), then the matching
// Get overloads, then Reset(). MSVC emits each group of overloaded virtuals in reverse declaration order, so the
// slots below run void* .. ull for both groups. All entries use the Windows x64 calling convention with `this` first.
// ---------------------------------------------------------------------------------------------------------------------
namespace
{
using Self = NgxParams;

void SetPtr(Self* s, const char* n, void* v) { Log(4, "Set(ptr) %s", n); s->SetPtr(n, v); }
void SetI32(Self* s, const char* n, int v) { Log(4, "Set(i32) %s=%d", n, v); s->SetI32(n, v); }
void SetU32(Self* s, const char* n, unsigned int v) { Log(4, "Set(u32) %s=%u", n, v); s->SetU32(n, v); }
void SetF64(Self* s, const char* n, double v) { Log(4, "Set(f64) %s=%f", n, v); s->SetF64(n, v); }
void SetF32(Self* s, const char* n, float v) { Log(4, "Set(f32) %s=%f", n, (double)v); s->SetF32(n, v); }
void SetU64(Self* s, const char* n, unsigned long long v) { Log(4, "Set(u64) %s=%llu", n, v); s->SetU64(n, v); }

NVSDK_NGX_Result Ok(bool found) { return found ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail; }

NVSDK_NGX_Result GetPtr(const Self* s, const char* n, void** v) { Log(4, "Get(ptr) %s", n); return Ok(s->GetPtr(n, v)); }
NVSDK_NGX_Result GetI32(const Self* s, const char* n, int* v) { Log(4, "Get(i32) %s", n); return Ok(s->GetI32(n, v)); }
NVSDK_NGX_Result GetU32(const Self* s, const char* n, unsigned int* v) { Log(4, "Get(u32) %s", n); return Ok(s->GetU32(n, v)); }
NVSDK_NGX_Result GetF64(const Self* s, const char* n, double* v) { Log(4, "Get(f64) %s", n); return Ok(s->GetF64(n, v)); }
NVSDK_NGX_Result GetF32(const Self* s, const char* n, float* v) { Log(4, "Get(f32) %s", n); return Ok(s->GetF32(n, v)); }
NVSDK_NGX_Result GetU64(const Self* s, const char* n, unsigned long long* v) { Log(4, "Get(u64) %s", n); return Ok(s->GetU64(n, v)); }

void ResetAll(Self* s) { s->Reset(); }

const void* const kVtable[] = {
    // Set: void*, ID3D12Resource*, ID3D11Resource*, int, unsigned int, double, float, unsigned long long
    reinterpret_cast<const void*>(&SetPtr),
    reinterpret_cast<const void*>(&SetPtr),
    reinterpret_cast<const void*>(&SetPtr),
    reinterpret_cast<const void*>(&SetI32),
    reinterpret_cast<const void*>(&SetU32),
    reinterpret_cast<const void*>(&SetF64),
    reinterpret_cast<const void*>(&SetF32),
    reinterpret_cast<const void*>(&SetU64),
    // Get: same order
    reinterpret_cast<const void*>(&GetPtr),
    reinterpret_cast<const void*>(&GetPtr),
    reinterpret_cast<const void*>(&GetPtr),
    reinterpret_cast<const void*>(&GetI32),
    reinterpret_cast<const void*>(&GetU32),
    reinterpret_cast<const void*>(&GetF64),
    reinterpret_cast<const void*>(&GetF32),
    reinterpret_cast<const void*>(&GetU64),
    reinterpret_cast<const void*>(&ResetAll),
};
} // namespace

NgxParams::NgxParams() : m_vtable(kVtable) {}

// NGX also addresses parameters by short keys ("#" followed by one byte, NVSDK_NGX_EParameter_*). They name the same
// values as the long keys, so both spellings are stored under the long one.
static const char* Canonical(const char* name)
{
    if (!name || name[0] != '#' || name[1] == 0 || name[2] != 0)
        return name;
    switch (static_cast<unsigned char>(name[1]))
    {
    case 0x01: return "SuperSampling.Available";
    case 0x0f: return "Scale";
    case 0x10: return "Width";
    case 0x11: return "Height";
    case 0x12: return "OutWidth";
    case 0x13: return "OutHeight";
    case 0x14: return "Sharpness";
    case 0x15: return "Scratch";
    case 0x16: return "Scratch.SizeInBytes";
    case 0x1e: return "Color";
    case 0x22: return "Output";
    case 0x25: return "Reset";
    case 0x27: return "MotionVectors";
    case 0x2c: return "MV.Scale.X";
    case 0x2d: return "MV.Scale.Y";
    case 0x30: return "SizeInBytes";
    case 0x35: return "CreationNodeMask";
    case 0x36: return "VisibilityNodeMask";
    case 0x37: return "PreviousOutput";
    case 0x38: return "MV.Offset.X";
    case 0x39: return "MV.Offset.Y";
    case 0x3b: return "ResourceWidth";
    case 0x3c: return "ResourceHeight";
    case 0x3d: return "Depth";
    case 0x3e: return "DLSSOptimalSettingsCallback";
    case 0x3f: return "PerfQualityValue";
    case 0x40: return "RTXValue";
    case 0x41: return "DLSSMode";
    case 0x44: return "Snippet.OptLevel";
    case 0x45: return "Snippet.IsDevBranch";
    default:   return name;
    }
}

// Printable form of a key for the log (short keys carry a raw byte).
static void LogMiss(const char* name)
{
    if (name[0] == '#' && name[1] && !name[2])
        Log(3, "param miss: #\\x%02x", static_cast<unsigned char>(name[1]));
    else
        Log(3, "param miss: %s", name);
}

#define NGX_SET(Fn, T, field, tag)                         \
    void NgxParams::Fn(const char* name, T v)              \
    {                                                      \
        if (!name)                                         \
            return;                                        \
        std::lock_guard<std::mutex> lock(m_mutex);         \
        Value& value = m_values[Canonical(name)];          \
        value.type   = Type::tag;                          \
        value.field  = v;                                  \
    }

NGX_SET(SetU64, unsigned long long, u64, U64)
NGX_SET(SetF32, float, f32, F32)
NGX_SET(SetF64, double, f64, F64)
NGX_SET(SetU32, unsigned int, u32, U32)
NGX_SET(SetI32, int, i32, I32)
NGX_SET(SetPtr, void*, ptr, Ptr)
#undef NGX_SET

// Numbers convert between types like the NGX runtime does: games often Set as int and Get as unsigned, or the reverse.
template <typename T> bool NgxParams::GetNumber(const char* name, T* out) const
{
    if (!name || !out)
        return false;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_values.find(Canonical(name));
    if (it == m_values.end())
    {
        LogMiss(name);
        return false;
    }
    const Value& v = it->second;
    switch (v.type)
    {
    case Type::U64: *out = static_cast<T>(v.u64); break;
    case Type::F32: *out = static_cast<T>(v.f32); break;
    case Type::F64: *out = static_cast<T>(v.f64); break;
    case Type::U32: *out = static_cast<T>(v.u32); break;
    case Type::I32: *out = static_cast<T>(v.i32); break;
    case Type::Ptr:
        if constexpr (std::is_same_v<T, unsigned long long>)
        {
            *out = reinterpret_cast<unsigned long long>(v.ptr);
            break;
        }
        return false;
    }
    return true;
}

bool NgxParams::GetU64(const char* n, unsigned long long* v) const { return GetNumber(n, v); }
bool NgxParams::GetF32(const char* n, float* v) const { return GetNumber(n, v); }
bool NgxParams::GetF64(const char* n, double* v) const { return GetNumber(n, v); }
bool NgxParams::GetU32(const char* n, unsigned int* v) const { return GetNumber(n, v); }
bool NgxParams::GetI32(const char* n, int* v) const { return GetNumber(n, v); }

bool NgxParams::GetPtr(const char* name, void** out) const
{
    if (!name || !out)
        return false;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_values.find(Canonical(name));
    if (it == m_values.end())
    {
        LogMiss(name);
        return false;
    }
    if (it->second.type == Type::Ptr)
        *out = it->second.ptr;
    else if (it->second.type == Type::U64)
        *out = reinterpret_cast<void*>(it->second.u64);
    else
        return false;
    return true;
}

unsigned int NgxParams::U32(const char* name, unsigned int def) const
{
    unsigned int v = def;
    return GetU32(name, &v) ? v : def;
}

int NgxParams::I32(const char* name, int def) const
{
    int v = def;
    return GetI32(name, &v) ? v : def;
}

float NgxParams::F32(const char* name, float def) const
{
    float v = def;
    return GetF32(name, &v) ? v : def;
}

void* NgxParams::Ptr(const char* name) const
{
    void* v = nullptr;
    return GetPtr(name, &v) ? v : nullptr;
}

void NgxParams::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values.clear();
}

static NVSDK_NGX_Result NgxStats(NVSDK_NGX_Parameter* abi)
{
    NgxParams* p = NgxParams::From(abi);
    unsigned int w = p->U32(ngxp::Width), h = p->U32(ngxp::Height);
    // Rough VRAM estimate for the NSS history, intermediate tensors and output.
    p->SetU64("SizeInBytes", static_cast<unsigned long long>(w) * h * 31);
    return NVSDK_NGX_Result_Success;
}

void NgxParams::FillCapabilities()
{
    SetI32("SuperSampling.Available", 1);
    SetI32("SuperSampling.NeedsUpdatedDriver", 0);
    SetI32("SuperSampling.MinDriverVersionMajor", 0);
    SetI32("SuperSampling.MinDriverVersionMinor", 0);
    SetI32("SuperSampling.FeatureInitResult", NVSDK_NGX_Result_Success);
    SetI32("Snippet.OptLevel", 0);
    SetI32("Snippet.IsDevBranch", 0);
    // Ray reconstruction and frame generation are not provided by this proxy.
    SetI32("SuperSamplingDenoising.Available", 0);
    SetI32("FrameGeneration.Available", 0);
    SetPtr("DLSSOptimalSettingsCallback", reinterpret_cast<void*>(&NgxOptimalSettings));
    SetPtr("DLSSGetStatsCallback", reinterpret_cast<void*>(&NgxStats));
    SetI32("DLSS.Enable.Output.Subrects", 1);
    SetI32("CreationNodeMask", 1);
    SetI32("VisibilityNodeMask", 1);
    SetF32("Sharpness", 0.0f);
    SetF32(ngxp::MVScaleX, 1.0f);
    SetF32(ngxp::MVScaleY, 1.0f);
}

NVSDK_NGX_Result NgxOptimalSettings(NVSDK_NGX_Parameter* abi)
{
    NgxParams* p = NgxParams::From(abi);
    unsigned int width = 0, height = 0;
    int          quality = 0;
    if (!p->GetU32(ngxp::Width, &width) || !p->GetU32(ngxp::Height, &height) || !p->GetI32(ngxp::PerfQuality, &quality))
        return NVSDK_NGX_Result_FAIL_InvalidParameter;

    const int    mode   = (quality >= 0 && quality < 6) ? quality : 0;
    // DU_RATIO fixes the ratio; otherwise each upscaler gets the render size at which the frame costs the same as
    // with NSS Quality at the DLSS mode's standard ratio (cost_model.cpp).
    const float  ratio  = Config().ratioForced ? Config().qualityRatio[mode]
                                               : EqualCostRatio(mode, Config().qualityRatio[mode], width, height);
    unsigned int outW   = static_cast<unsigned int>(width / ratio);
    unsigned int outH   = static_cast<unsigned int>(height / ratio);

    p->SetU32(ngxp::OutWidth, outW);
    p->SetU32(ngxp::OutHeight, outH);
    p->SetF32("Sharpness", 0.0f);
    p->SetF32("Scale", 1.0f / ratio);
    p->SetF32("SuperSampling.ScaleFactor", 1.0f / ratio);
    // Dynamic resolution range: NSS accepts any render size up to the one the context was created with.
    p->SetU32("DLSS.Get.Dynamic.Max.Render.Width", outW);
    p->SetU32("DLSS.Get.Dynamic.Max.Render.Height", outH);
    p->SetU32("DLSS.Get.Dynamic.Min.Render.Width", std::min(outW, width / 3));
    p->SetU32("DLSS.Get.Dynamic.Min.Render.Height", std::min(outH, height / 3));
    p->SetU64("SizeInBytes", static_cast<unsigned long long>(width) * height * 31);

    LOGV("optimal settings: %ux%u quality %d -> %ux%u", width, height, quality, outW, outH);
    return NVSDK_NGX_Result_Success;
}
