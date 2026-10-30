// Shared FSR 4 driver plumbing: see fsr4_common.h. The ABI facts (provider data block layout and sizes, the per-network
// provider classes and that their device check is the third virtual function, the D3DKMT private-data offsets of family
// and revision, the model families, MLSR-WATERMARK) were learnt from OptiScaler's FSR 4 notes, FSR4Injector's README,
// and disassembling AMD's FSR 4 driver DLLs; none of their code is used.
// SPDX-License-Identifier: MIT
#include "fsr4_common.h"

#include "common.h"

#include <atomic>
#include <cctype>
#include <string>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace
{
// Provider data block of UpdateFfxApiProvider: caller sets blockVersion + descType, the DLL fills the provider.
// Block version 3 is 0x60 bytes (adds a device check); 1 and 2 end at 0x50.
struct ProviderBlock
{
    uint32_t     blockVersion;
    uint64_t     descType;
    Fsr4Provider provider;
    bool (*deviceCheck)(const void* backend);
    void* unused;
};
static_assert(offsetof(ProviderBlock, provider) == 0x10, "provider block layout");
static_assert(offsetof(ProviderBlock, deviceCheck) == 0x50, "provider block layout");
static_assert(sizeof(ProviderBlock) == 0x60, "provider block layout");

typedef HRESULT (*PFN_UpdateFfxApiProvider)(void* block, uint32_t blockSize);
constexpr uint64_t kUpscaleDescType = 0x00010000u;  // FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE

// Per-network provider class, in the order FidelityFX asks them.
struct ModelRtti
{
    int         model;
    const char* rttiName;
};
const ModelRtti kModelRtti[] = {
    {kFsr4Fp8, ".?AVffxProvider_FSR4@@"},
    {kFsr4Int8, ".?AVffxProvider_FSR4_Int8@@"},
    {kFsr4Rdna2, ".?AVffxProvider_FSR4_Int8_Navi2@@"},
    {kFsr4Lite, ".?AVffxProvider_FSR4Lite_Int8@@"},
};
constexpr size_t kDeviceCheckSlot = 2;  // third virtual function

typedef bool (*PFN_DeviceCheck)(void* self, const void* backend);

// A network's wrapped check: which driver and model a provider vtable belongs to, and AMD's original check.
struct WrappedCheck
{
    void**          vtable;
    const Fsr4Driver* driver;
    int             model;
    PFN_DeviceCheck original;
};
std::mutex                g_mutex;
std::vector<WrappedCheck> g_wrapped;

std::atomic<int> g_wantedModel{kFsr4Auto};
std::atomic<int> g_acceptedModel{-1};
std::atomic<int> g_nativeMask{0};

bool WrappedDeviceCheck(void* self, const void* backend)
{
    void** vtable = *static_cast<void***>(self);
    const WrappedCheck* w = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (const WrappedCheck& c : g_wrapped)
            if (c.vtable == vtable)
                w = &c;
    }
    if (!w)
        return false;
    const bool native = w->original(self, backend);
    const int  wanted = g_wantedModel.load();
    const bool answer = wanted == kFsr4Auto ? native : w->model == wanted;
    if (native)
        g_nativeMask.fetch_or(1 << w->model);
    int none = -1;
    if (answer && backend)
        g_acceptedModel.compare_exchange_strong(none, w->model);
    LOGI("FSR 4 (%s): device check %s: AMD %d, answered %d", w->driver->name, Fsr4ModelName(w->model), native, answer);
    return answer;
}

// ---- D3DKMT GPU query the DLLs make (amdgpu private data) -----------------------------------------------------------
Fsr4Gpu g_gpu;

struct KmtAdapterInfo
{
    uint32_t handle;
    LUID     luid;
    uint32_t sources;
    BOOL     presentMoveRegionsPreferred;
};
struct KmtEnumAdapters
{
    uint32_t       count;
    KmtAdapterInfo adapters[16];
};
static_assert(sizeof(KmtEnumAdapters) == 0x144, "D3DKMT_ENUMADAPTERS");
struct KmtQueryAdapterInfo
{
    uint32_t adapter;
    uint32_t type;  // 0 = KMTQAITYPE_UMDRIVERPRIVATE
    void*    data;
    uint32_t size;
};
constexpr uint32_t kFakeAdapter = 0x40000040u;
constexpr LONG     kStatusOk = 0, kStatusInvalid = (LONG)0xC000000Du, kStatusNotSupported = (LONG)0xC00000BBu;
// Offsets in the AMD private adapter data where the DLLs read family and revision (first short query, >= 0x38 bytes).
constexpr uint32_t kPrivateFamily = 0x30, kPrivateRevision = 0x34, kPrivateMinSize = 0x38;

LONG WINAPI KmtEnumAdapters_(KmtEnumAdapters* e)
{
    if (!e)
        return kStatusInvalid;
    memset(e, 0, sizeof(*e));
    if (!g_gpu.family)
        return kStatusOk;
    e->count               = 1;
    e->adapters[0].handle  = kFakeAdapter;
    e->adapters[0].sources = 1;
    return kStatusOk;
}

LONG WINAPI KmtQueryAdapterInfo_(KmtQueryAdapterInfo* q)
{
    if (!q || q->adapter != kFakeAdapter)
        return kStatusInvalid;
    if (q->type != 0 || !q->data || q->size < kPrivateMinSize)
        return kStatusNotSupported;
    memset(q->data, 0, q->size);
    memcpy(static_cast<uint8_t*>(q->data) + kPrivateFamily, &g_gpu.family, 4);
    memcpy(static_cast<uint8_t*>(q->data) + kPrivateRevision, &g_gpu.revision, 4);
    return kStatusOk;
}

// ---- import redirection and RTTI ------------------------------------------------------------------------------------
struct Redirects
{
    decltype(&GetModuleHandleW) realGetModuleHandleW = nullptr;
    decltype(&GetProcAddress)   realGetProcAddress   = nullptr;
    HMODULE                     module               = nullptr;
    const wchar_t*              sdkInfix             = nullptr;  // "_vk" or "_dx12": the SDK modules this DLL asks for
};
std::vector<Redirects> g_redirects;


// A FidelityFX SDK module the DLL looks up stands in as the DLL itself (it is signed with the same certificate; Wine's
// WinVerifyTrust accepts it, and its file version counts as the SDK version). The DLL only ever looks up its own kind
// of SDK module, so answering with the matching caller is correct.
HMODULE WINAPI SdkModuleLookup(LPCWSTR name)
{
    if (name && !_wcsnicmp(name, L"amd_fidelityfx_", 15))
    {
        // amdxcffx64 asks for amd_fidelityfx_*_dx12.dll (the Vulkan driver DLL would ask for *_vk.dll): answer with the
        // loaded DLL itself (same signing certificate; its file version then counts as the SDK version).
        for (const Redirects& r : g_redirects)
            if (r.sdkInfix && wcsstr(name, r.sdkInfix))
                return r.module;
        return g_redirects.empty() ? nullptr : g_redirects.back().module;
    }
    return GetModuleHandleW(name);  // only the DLL calls this
}

FARPROC WINAPI KmtProcLookup(HMODULE module, LPCSTR name)
{
    if (reinterpret_cast<uintptr_t>(name) > 0xFFFF)
    {
        if (!strcmp(name, "D3DKMTEnumAdapters"))
            return reinterpret_cast<FARPROC>(&KmtEnumAdapters_);
        if (!strcmp(name, "D3DKMTQueryAdapterInfo"))
            return reinterpret_cast<FARPROC>(&KmtQueryAdapterInfo_);
    }
    for (const Redirects& r : g_redirects)
        if (r.realGetProcAddress)
            return r.realGetProcAddress(module, name);  // any driver's real GetProcAddress is the kernel32 one
    return GetProcAddress(module, name);
}

bool WritePointer(void** slot, void* value)
{
    DWORD old = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old))
        return false;
    *slot = value;
    VirtualProtect(slot, sizeof(void*), old, &old);
    return true;
}

void* RedirectImport(HMODULE module, const char* dll, const char* function, void* replacement)
{
    auto* base = reinterpret_cast<uint8_t*>(module);
    auto* nt   = reinterpret_cast<IMAGE_NT_HEADERS64*>(base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress)
        return nullptr;
    for (auto* imp = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); imp->Name; ++imp)
    {
        if (_stricmp(reinterpret_cast<char*>(base + imp->Name), dll))
            continue;
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA64*>(
            base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots)
        {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal))
                continue;
            auto* byName = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (strcmp(reinterpret_cast<char*>(byName->Name), function))
                continue;
            void* previous = reinterpret_cast<void*>(slots->u1.Function);
            return WritePointer(reinterpret_cast<void**>(&slots->u1.Function), replacement) ? previous : nullptr;
        }
    }
    return nullptr;
}

void** FindVtable(HMODULE module, const char* rttiName)
{
    auto* base = reinterpret_cast<uint8_t*>(module);
    auto* nt   = reinterpret_cast<IMAGE_NT_HEADERS64*>(base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const IMAGE_SECTION_HEADER* sections = IMAGE_FIRST_SECTION(nt);
    const size_t                nameLen  = strlen(rttiName) + 1;

    uint32_t typeRva = 0;
    for (uint32_t s = 0; s < nt->FileHeader.NumberOfSections && !typeRva; ++s)
    {
        if (sections[s].Characteristics & IMAGE_SCN_MEM_EXECUTE)
            continue;
        const uint8_t* b = base + sections[s].VirtualAddress;
        const uint8_t* e = b + sections[s].Misc.VirtualSize;
        for (const uint8_t* p = b; p + nameLen <= e; p += 8)
            if (*p == '.' && !memcmp(p, rttiName, nameLen))
            {
                typeRva = uint32_t(p - base) - 16;
                break;
            }
    }
    if (!typeRva)
        return nullptr;
    for (uint32_t s = 0; s < nt->FileHeader.NumberOfSections; ++s)
    {
        if (sections[s].Characteristics & (IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_WRITE))
            continue;
        uint8_t* b = base + sections[s].VirtualAddress;
        uint8_t* e = b + sections[s].Misc.VirtualSize;
        for (uint8_t* p = b; p + 24 <= e; p += 4)
        {
            const uint32_t* col = reinterpret_cast<const uint32_t*>(p);
            if (col[0] != 1 || col[1] != 0 || col[3] != typeRva || col[5] != uint32_t(p - base))
                continue;
            const uint64_t target = reinterpret_cast<uint64_t>(p);
            for (uint8_t* q = b; q + 16 <= e; q += 8)
                if (*reinterpret_cast<uint64_t*>(q) == target)
                    return reinterpret_cast<void**>(q + 8);
        }
    }
    return nullptr;
}
}  // namespace

Fsr4Gpu Fsr4KnownGpu(uint32_t vendorId, uint32_t deviceId, const char* deviceName)
{
    Fsr4Gpu ids;
    if (const char* v = getenv("DU_FSR4_GPU"))
    {
        sscanf(v, "%u,%u", &ids.family, &ids.revision);
        return ids;
    }
    // Steam Deck, Van Gogh (amdgpu family 144 = AMDGPU_FAMILY_VGH, external revision 1): LCD 0x163F, OLED 0x1435.
    // Proton's NVIDIA spoof changes the reported IDs, but RADV's device name keeps "VANGOGH".
    if ((vendorId == 0x1002 && (deviceId == 0x163F || deviceId == 0x1435)) || (deviceName && strstr(deviceName, "VANGOGH")))
        ids = {144, 1};
    return ids;
}

int Fsr4ModelForGpu(const Fsr4Gpu& g)
{
    if (g.family == 144 && g.revision >= 1 && g.revision <= 0x7F)
        return kFsr4Lite;  // Van Gogh
    if (g.family == 150 && g.revision >= 1 && g.revision <= 0x3F)
        return kFsr4Lite;  // Strix Point
    if (g.family == 145 || (g.family == 150 && g.revision >= 0xC0 && g.revision <= 0xCF))
        return kFsr4Int8;  // RDNA 3, Strix Halo
    if (g.family == 143 && g.revision >= 0x28 && g.revision <= 0x4F)
        return kFsr4Rdna2;
    if (g.family == 152)
        return kFsr4Fp8;  // RDNA 4
    return -1;
}

void Fsr4EnableWatermark()
{
    // Desired state: the live control-file toggle wins, else DU_FSR4_WATERMARK / FSR_WATERMARK. Set or clear
    // MLSR-WATERMARK (the variable both DLLs read) in both the Win32 and the CRT environment, before each context build.
    const int   live = LiveWatermark();
    const char* a    = getenv("DU_FSR4_WATERMARK");
    const char* b    = getenv("FSR_WATERMARK");
    const bool  want = live >= 0 ? live != 0 : ((a && *a && strcmp(a, "0")) || (b && *b && strcmp(b, "0")));
    static int  last = -1;
    SetEnvironmentVariableA("MLSR-WATERMARK", want ? "1" : nullptr);
    if (HMODULE ucrt = GetModuleHandleA("ucrtbase.dll"))
        if (auto put = reinterpret_cast<int (*)(const char*, const char*)>(GetProcAddress(ucrt, "_putenv_s")))
            put("MLSR-WATERMARK", want ? "1" : "");
    if (int(want) != last)
        LOGI("FSR 4: watermark %s", want ? "on (MLSR-WATERMARK=1)" : "off");
    last = int(want);
}

std::string Fsr4ModelList(const Fsr4Driver& drv)
{
    std::string s;
    for (int m = 1; m < kFsr4ModelCount; ++m)
        if (drv.HasModel(m))
            s += std::string(s.empty() ? "" : ", ") + Fsr4ModelName(m);
    return s.empty() ? "none" : s;
}

bool Fsr4LoadDriver(const wchar_t* dllName, const Fsr4Gpu& gpu, Fsr4Driver& out)
{
    const bool isVk = wcsstr(dllName, L"vk") != nullptr;
    out.name        = isVk ? "amdvkffx64" : "amdxcffx64";
    g_gpu = gpu;  // the GPU answered to whichever DLL runs its query next (loads are serialised by the callers)

    wchar_t path[MAX_PATH] = {};
    HMODULE self           = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&Fsr4LoadDriver), &self);
    GetModuleFileNameW(self, path, MAX_PATH);
    // Load from a private "deckyupscaler\" subfolder next to _nvngx.dll, by full path. Proton reinstalls its own
    // (older) amdxcffx64.dll into the prefix's system32 on launch, which would shadow a copy placed there; the subfolder
    // is ours alone. DU_FSR4_DIR overrides it (a Windows path, e.g. for the test harness).
    if (wchar_t* slash = wcsrchr(path, L'\\'))
        wcscpy(slash + 1, L"deckyupscaler\\");
    else
        path[0] = 0;
    if (const char* dir = getenv("DU_FSR4_DIR"))
    {
        MultiByteToWideChar(CP_UTF8, 0, dir, -1, path, MAX_PATH);
        size_t n = wcslen(path);
        if (n && path[n - 1] != L'\\' && n + 1 < MAX_PATH)
            wcscat(path, L"\\");
    }
    wcscat(path, dllName);
    if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES)
    {
        LOGI("FSR 4: %ls not found; using NSS", path);
        return false;
    }
    // Log the DLL's file version (proof of which build loaded); the authoritative 4.1.2 check is the provider name below.
    if (DWORD vsz = GetFileVersionInfoSizeW(path, nullptr))
    {
        std::vector<uint8_t> buf(vsz);
        VS_FIXEDFILEINFO*    fi  = nullptr;
        UINT                 len = 0;
        if (GetFileVersionInfoW(path, 0, vsz, buf.data()) &&
            VerQueryValueW(buf.data(), L"\\", reinterpret_cast<void**>(&fi), &len) && fi)
            LOGI("FSR 4: %ls file version %u.%u.%u.%u", path, HIWORD(fi->dwFileVersionMS), LOWORD(fi->dwFileVersionMS),
                 HIWORD(fi->dwFileVersionLS), LOWORD(fi->dwFileVersionLS));
    }
    const auto t0 = GetTickCount64();
    HMODULE    lib = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!lib)
    {
        LOGI("FSR 4: cannot load %ls (error %lu); using NSS", path, GetLastError());
        return false;
    }
    auto update = reinterpret_cast<PFN_UpdateFfxApiProvider>(GetProcAddress(lib, "UpdateFfxApiProvider"));
    if (!update)
    {
        LOGI("FSR 4: %ls has no UpdateFfxApiProvider; using NSS", path);
        return false;
    }
    out.module = lib;
    Redirects redir;
    redir.module               = lib;
    redir.sdkInfix             = isVk ? L"_vk" : L"_dx12";
    redir.realGetModuleHandleW  = reinterpret_cast<decltype(redir.realGetModuleHandleW)>(
        RedirectImport(lib, "KERNEL32.dll", "GetModuleHandleW", reinterpret_cast<void*>(&SdkModuleLookup)));
    redir.realGetProcAddress = reinterpret_cast<decltype(redir.realGetProcAddress)>(
        RedirectImport(lib, "KERNEL32.dll", "GetProcAddress", reinterpret_cast<void*>(&KmtProcLookup)));
    if (!redir.realGetModuleHandleW || !redir.realGetProcAddress)
    {
        LOGI("FSR 4: cannot redirect %ls's GetModuleHandleW / GetProcAddress; using NSS", dllName);
        return false;
    }
    g_redirects.push_back(redir);

    for (const ModelRtti& m : kModelRtti)
    {
        void** vtable = FindVtable(lib, m.rttiName);
        if (!vtable)
            continue;
        auto original = reinterpret_cast<PFN_DeviceCheck>(vtable[kDeviceCheckSlot]);
        if (!WritePointer(&vtable[kDeviceCheckSlot], reinterpret_cast<void*>(&WrappedDeviceCheck)))
            continue;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_wrapped.push_back({vtable, &out, m.model, original});
        }
        out.modelsFound |= 1 << m.model;
    }

    Fsr4EnableWatermark();

    ProviderBlock block = {};
    HRESULT       hr    = E_FAIL;
    for (uint32_t version : {3u, 2u})
    {
        block              = {};
        block.blockVersion = version;
        block.descType     = kUpscaleDescType;
        hr = update(&block, version >= 3 ? uint32_t(sizeof(ProviderBlock)) : uint32_t(offsetof(ProviderBlock, deviceCheck)));
        if (SUCCEEDED(hr) && block.provider.createContext)
            break;
        LOGI("FSR 4: %ls UpdateFfxApiProvider (block %u) returned 0x%lx, provider %s", dllName, version, (unsigned long)hr,
             block.provider.createContext ? "given" : "not given");
    }
    if (FAILED(hr) || !block.provider.createContext || !block.provider.dispatch)
    {
        LOGI("FSR 4: %ls handed out no upscaler provider; using NSS", dllName);
        return false;
    }
    out.provider    = block.provider;
    out.versionName = block.provider.versionName ? block.provider.versionName : "";
    out.deviceCheck = block.blockVersion >= 3 ? block.deviceCheck : nullptr;
    LOGI("FSR 4: loaded %ls in %llu ms: provider \"%s\" (id 0x%llx), networks: %s", path,
         (unsigned long long)(GetTickCount64() - t0), out.versionName.c_str(),
         (unsigned long long)block.provider.versionId, Fsr4ModelList(out).c_str());
    // Require FSR 4.1.2 exactly: refuse an older provider (e.g. a 4.1.1 / 4.0.x driver, or an FSR 3.1 fallback) loudly
    // rather than silently upscale with it. DU_FSR4_ALLOW_ANY_VERSION bypasses the check (diagnostics only).
    if (out.versionName.find("4.1.2") == std::string::npos && !getenv("DU_FSR4_ALLOW_ANY_VERSION"))
    {
        LOGI("FSR 4: %ls reports provider version \"%s\", not 4.1.2 - refusing it and using NSS (set "
             "DU_FSR4_ALLOW_ANY_VERSION=1 to override)", dllName, out.versionName.c_str());
        return false;
    }
    return true;
}

uint32_t Fsr4CreateContext(const Fsr4Driver& drv, int wantedModel, void* context, void* createDesc, int* accepted,
                           int* nativeMask)
{
    g_wantedModel   = wantedModel;
    g_acceptedModel = -1;
    g_nativeMask    = 0;
    const uint32_t rc = drv.provider.createContext(context, createDesc, nullptr);
    if (accepted)
        *accepted = g_acceptedModel.load();
    if (nativeMask)
        *nativeMask = g_nativeMask.load();
    return rc;
}

int Fsr4CreateWithModel(const Fsr4Driver& drv, int wanted, const Fsr4Gpu& gpu, void* context, void* createDesc)
{
    Fsr4EnableWatermark();  // apply the live watermark toggle before (re)building the context
    int        accepted = -1, nativeMask = 0, effectiveWanted = wanted;
    uint32_t   rc = Fsr4CreateContext(drv, wanted, context, createDesc, &accepted, &nativeMask);
    void**     ctx = static_cast<void**>(context);
    const int  gpuModel = Fsr4ModelForGpu(gpu);
    if (rc == 0 && *ctx && wanted == kFsr4Auto && accepted < 0 && gpuModel > 0 && gpuModel != kFsr4Fp8 &&
        drv.HasModel(gpuModel))
    {
        LOGI("FSR 4 (%s): AMD's checks accepted no network (the DLL sees a non-AMD vendor); auto -> %s for amdgpu "
             "family %u revision %u", drv.name, Fsr4ModelName(gpuModel), gpu.family, gpu.revision);
        drv.provider.destroyContext(ctx, nullptr);
        *ctx            = nullptr;
        effectiveWanted = gpuModel;
        rc              = Fsr4CreateContext(drv, gpuModel, context, createDesc, &accepted, &nativeMask);
    }
    std::string native;
    for (int m = 1; m < kFsr4ModelCount; ++m)
        if (nativeMask & (1 << m))
            native += std::string(native.empty() ? "" : ", ") + Fsr4ModelName(m);
    if (rc != 0 || !*ctx)
    {
        LOGI("FSR 4 (%s): createContext failed (%u), asked for %s", drv.name, rc, Fsr4ModelName(wanted));
        if (*ctx)
        {
            drv.provider.destroyContext(ctx, nullptr);
            *ctx = nullptr;
        }
        return -1;
    }
    if (accepted < 0 || (effectiveWanted != kFsr4Auto && accepted != effectiveWanted))
    {
        LOGI("FSR 4 (%s): no network accepted the device (asked for %s; AMD accepts: %s); not using it", drv.name,
             Fsr4ModelName(wanted), native.empty() ? "none" : native.c_str());
        drv.provider.destroyContext(ctx, nullptr);
        *ctx = nullptr;
        return -1;
    }
    LOGI("FSR 4 (%s): network %s runs (asked for %s; AMD's own choice: %s)", drv.name, Fsr4ModelName(accepted),
         Fsr4ModelName(wanted), native.empty() ? "none" : native.c_str());
    return accepted;
}

// ---- model names (also used by the logs and the control-file / env parsing) ----------------------------------------
const char* Fsr4ModelName(int model)
{
    switch (model)
    {
    case kFsr4Auto: return "auto";
    case kFsr4Lite: return "Lite";
    case kFsr4Int8: return "INT8";
    case kFsr4Rdna2: return "INT8 RDNA 2";
    case kFsr4Fp8: return "FP8";
    default: return "?";
    }
}

int Fsr4ModelFromName(const char* name)
{
    if (!name)
        return -1;
    std::string s(name);
    for (char& c : s)
        c = char(tolower(c));
    if (s == "auto" || s.empty())
        return kFsr4Auto;
    if (s == "lite" || s == "int8lite" || s == "int8-lite")
        return kFsr4Lite;
    if (s == "int8")
        return kFsr4Int8;
    if (s == "rdna2" || s == "navi2" || s == "int8-rdna2" || s == "int8rdna2")
        return kFsr4Rdna2;
    if (s == "fp8")
        return kFsr4Fp8;
    return -1;
}

void* Fsr4RedirectImport(HMODULE module, const char* dll, const char* function, void* replacement)
{
    return RedirectImport(module, dll, function, replacement);
}

int Fsr4PrepareModule(HMODULE lib, const Fsr4Gpu& gpu, Fsr4Driver& out)
{
    // Same treatment as a driver DLL (Fsr4LoadDriver) for a module that is not one: AMD's FidelityFX SDK DLL carries
    // the same FSR 4 networks. Its GPU query (if it makes one) gets `gpu`; each network's device check is wrapped.
    g_gpu      = gpu;
    out.module = lib;
    Redirects redir;
    redir.module              = lib;
    redir.sdkInfix            = L"_vk";
    redir.realGetModuleHandleW = reinterpret_cast<decltype(redir.realGetModuleHandleW)>(
        RedirectImport(lib, "KERNEL32.dll", "GetModuleHandleW", reinterpret_cast<void*>(&SdkModuleLookup)));
    redir.realGetProcAddress = reinterpret_cast<decltype(redir.realGetProcAddress)>(
        RedirectImport(lib, "KERNEL32.dll", "GetProcAddress", reinterpret_cast<void*>(&KmtProcLookup)));
    if (redir.realGetModuleHandleW || redir.realGetProcAddress)
        g_redirects.push_back(redir);
    for (const ModelRtti& m : kModelRtti)
    {
        void** vtable = FindVtable(lib, m.rttiName);
        if (!vtable)
            continue;
        auto original = reinterpret_cast<PFN_DeviceCheck>(vtable[kDeviceCheckSlot]);
        if (!WritePointer(&vtable[kDeviceCheckSlot], reinterpret_cast<void*>(&WrappedDeviceCheck)))
            continue;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_wrapped.push_back({vtable, &out, m.model, original});
        }
        out.modelsFound |= 1 << m.model;
    }
    Fsr4EnableWatermark();
    return out.modelsFound;
}

void Fsr4BeginCreate(int wantedModel)
{
    g_wantedModel   = wantedModel;
    g_acceptedModel = -1;
    g_nativeMask    = 0;
}

int Fsr4EndCreate(int* nativeMask)
{
    if (nativeMask)
        *nativeMask = g_nativeMask.load();
    return g_acceptedModel.load();
}
