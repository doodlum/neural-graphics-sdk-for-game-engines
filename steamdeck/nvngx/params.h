// NGX parameter table with an MSVC-compatible vtable.
// SPDX-License-Identifier: MIT
#pragma once

#include "ngx_abi.h"

#include <mutex>
#include <string>
#include <unordered_map>

class NgxParams
{
    const void* const* m_vtable;  // must stay the first member: callers dispatch through it

public:
    NgxParams();

    static NgxParams* From(NVSDK_NGX_Parameter* p) { return reinterpret_cast<NgxParams*>(p); }
    static const NgxParams* From(const NVSDK_NGX_Parameter* p) { return reinterpret_cast<const NgxParams*>(p); }
    NVSDK_NGX_Parameter* Abi() { return reinterpret_cast<NVSDK_NGX_Parameter*>(this); }

    void SetU64(const char* name, unsigned long long v);
    void SetF32(const char* name, float v);
    void SetF64(const char* name, double v);
    void SetU32(const char* name, unsigned int v);
    void SetI32(const char* name, int v);
    void SetPtr(const char* name, void* v);

    bool GetU64(const char* name, unsigned long long* v) const;
    bool GetF32(const char* name, float* v) const;
    bool GetF64(const char* name, double* v) const;
    bool GetU32(const char* name, unsigned int* v) const;
    bool GetI32(const char* name, int* v) const;
    bool GetPtr(const char* name, void** v) const;

    // Convenience readers with defaults.
    unsigned int U32(const char* name, unsigned int def = 0) const;
    int          I32(const char* name, int def = 0) const;
    float        F32(const char* name, float def = 0.0f) const;
    void*        Ptr(const char* name) const;

    void Reset();

    // Fills in the capability values DLSS callers query (SuperSampling.Available, optimal settings callback, ...).
    void FillCapabilities();

    bool persistent = false;  // owned by the proxy (GetParameters/GetCapabilityParameters), never freed by callers

private:
    enum class Type : uint8_t { U64, F32, F64, U32, I32, Ptr };
    struct Value
    {
        Type type;
        union
        {
            unsigned long long u64;
            float              f32;
            double             f64;
            unsigned int       u32;
            int                i32;
            void*              ptr;
        };
    };

    template <typename T> bool GetNumber(const char* name, T* out) const;

    mutable std::mutex                     m_mutex;
    std::unordered_map<std::string, Value> m_values;
};

// NGX optimal-settings callback: render size per quality mode for a given display size.
NVSDK_NGX_Result NgxOptimalSettings(NVSDK_NGX_Parameter* params);
