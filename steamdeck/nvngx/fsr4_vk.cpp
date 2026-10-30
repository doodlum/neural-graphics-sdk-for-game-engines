// AMD FSR 4 natively on a Vulkan game's device: amd_fidelityfx_upscaler_vk.dll from AMD's FSR SDK 2.4 (FSR Upscaling
// 4.1.2 with a Vulkan backend, signed SDK binary) records straight into the game's command buffer, so there is no
// D3D12 device, shared texture or command buffer split (fsr4_interop.cpp stays as the fallback when the DLL is missing
// or refuses the device). The DLL picks its network itself (Lite INT8 on a Steam Deck); the model the plugin asks for
// is only logged against what runs.
// SPDX-License-Identifier: MIT
#include "common.h"
#include "cost_model.h"
#include "fsr4_common.h"
#include "mv_dilate.h"
#include "nss_upscaler.h"

#include <windows.h>

#include "amd_ffx/ffx_api.h"
#include "amd_ffx/ffx_api_vk.h"
#include "amd_ffx/ffx_upscale.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace
{
constexpr wchar_t kDllName[] = L"amd_fidelityfx_upscaler_vk.dll";

struct Sdk
{
    HMODULE              lib = nullptr;
    PfnFfxCreateContext  create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxQuery          query = nullptr;
    PfnFfxConfigure      configure = nullptr;
    PfnFfxDispatch       dispatch = nullptr;
};

std::mutex g_mutex;
bool       g_loadTried = false, g_loadOk = false, g_deviceFailed = false, g_fsr3Failed = false;
Sdk        g_sdk;
Fsr4Driver g_drv;  // the SDK DLL's FSR 4 networks (device checks wrapped, see fsr4_common)

// <folder of _nvngx.dll>\deckyupscaler\<name>, or DU_FSR4_DIR\<name> (same place as amdxcffx64.dll, see
// fsr4_common.cpp).
std::wstring DllPath()
{
    wchar_t path[MAX_PATH] = {};
    HMODULE self           = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&DllPath), &self);
    GetModuleFileNameW(self, path, MAX_PATH);
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
    return std::wstring(path) + kDllName;
}

bool LoadSdk()
{
    if (g_loadTried)
        return g_loadOk;
    g_loadTried            = true;
    const std::wstring path = DllPath();
    if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES)
    {
        LOGI("FSR 4 Vulkan: %ls not found; FSR 4 goes through the D3D12 interop", path.c_str());
        return false;
    }
    // Only FSR Upscaling 4.1.2 (file version 4.1.2.x) unless DU_FSR4_ALLOW_ANY_VERSION=1.
    unsigned v[4] = {};
    if (DWORD vsz = GetFileVersionInfoSizeW(path.c_str(), nullptr))
    {
        std::vector<uint8_t> buf(vsz);
        VS_FIXEDFILEINFO*    fi  = nullptr;
        UINT                 len = 0;
        if (GetFileVersionInfoW(path.c_str(), 0, vsz, buf.data()) &&
            VerQueryValueW(buf.data(), L"\\", reinterpret_cast<void**>(&fi), &len) && fi)
            v[0] = HIWORD(fi->dwFileVersionMS), v[1] = LOWORD(fi->dwFileVersionMS), v[2] = HIWORD(fi->dwFileVersionLS),
            v[3] = LOWORD(fi->dwFileVersionLS);
    }
    LOGI("FSR 4 Vulkan: %ls file version %u.%u.%u.%u", path.c_str(), v[0], v[1], v[2], v[3]);
    if (!(v[0] == 4 && v[1] == 1 && v[2] == 2) && !(getenv("DU_FSR4_ALLOW_ANY_VERSION") && atoi(getenv("DU_FSR4_ALLOW_ANY_VERSION"))))
    {
        LOGI("FSR 4 Vulkan: not FSR Upscaling 4.1.2; refused (DU_FSR4_ALLOW_ANY_VERSION=1 accepts it)");
        return false;
    }
    const auto t0 = GetTickCount64();
    g_sdk.lib     = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_sdk.lib)
    {
        LOGI("FSR 4 Vulkan: cannot load %ls (error %lu)", path.c_str(), GetLastError());
        return false;
    }
    g_sdk.create   = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(g_sdk.lib, "ffxCreateContext"));
    g_sdk.destroy  = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(g_sdk.lib, "ffxDestroyContext"));
    g_sdk.query    = reinterpret_cast<PfnFfxQuery>(GetProcAddress(g_sdk.lib, "ffxQuery"));
    g_sdk.configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(g_sdk.lib, "ffxConfigure"));
    g_sdk.dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(g_sdk.lib, "ffxDispatch"));
    if (!g_sdk.create || !g_sdk.destroy || !g_sdk.query || !g_sdk.dispatch)
    {
        LOGI("FSR 4 Vulkan: %ls lacks the ffx* entry points", path.c_str());
        return false;
    }
    g_drv.name = "SDK Vulkan";
    const int found = Fsr4PrepareModule(g_sdk.lib, Fsr4KnownGpu(0x1002, 0x163F, "VANGOGH"), g_drv);
    LOGI("FSR 4 Vulkan: loaded in %llu ms; FSR 4 networks in the DLL: %s (mask 0x%x)",
         (unsigned long long)(GetTickCount64() - t0), Fsr4ModelList(g_drv).c_str(), found);
    g_loadOk = true;
    return true;
}

void FfxMessage(uint32_t type, const wchar_t* message)
{
    LOGI("FSR 4 Vulkan runtime %s: %ls", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", message);
}

// The upscaler versions the DLL offers on this device; returns the id of the newest <major>.x one (0 if none), logging
// all.
uint64_t PickVersion(VkDeviceContext* dc, std::string* name, char major)
{
    ffxQueryDescGetVersions q = {};
    q.header.type             = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType          = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.device                  = dc;
    uint64_t count            = 0;
    q.outputCount             = &count;
    if (g_sdk.query(nullptr, &q.header) != FFX_API_RETURN_OK || !count)
        return 0;
    std::vector<uint64_t>    ids(count);
    std::vector<const char*> names(count);
    q.versionIds   = ids.data();
    q.versionNames = names.data();
    if (g_sdk.query(nullptr, &q.header) != FFX_API_RETURN_OK)
        return 0;
    uint64_t    best = 0;
    std::string list;
    for (uint64_t i = 0; i < count; ++i)
    {
        const char* n = names[i] ? names[i] : "?";
        list += std::string(list.empty() ? "" : ", ") + n;
        if (n[0] == major && n[1] == '.' && (best == 0 || strcmp(n, name->c_str()) > 0))
            best = ids[i], *name = n;
    }
    LOGI("FSR 4 Vulkan: upscaler versions offered on this device: %s", list.c_str());
    return best;
}

// AMD's Windows Vulkan driver exposes vkUpdateFfxApiProviderAMD (it hands the SDK a driver-side FSR 4 provider); RADV
// does not, and the SDK's own FSR 4 networks then serve (their device checks are answered in fsr4_common).
// DU_FSR4_VK_PROBE=1/2 answers the lookup with a logger that fails / succeeds (diagnostics).
PFN_vkGetDeviceProcAddr g_gameGdpa = nullptr;
int                     g_probeMode = 0;  // DU_FSR4_VK_PROBE: 0 = NULL (as RADV), 1 = log and fail, 2 = log and succeed

VkResult VKAPI_CALL ProbeUpdateProvider(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
    LOGI("FSR 4 Vulkan probe: vkUpdateFfxApiProviderAMD(0x%llx, 0x%llx, 0x%llx, 0x%llx)", (unsigned long long)a,
         (unsigned long long)b, (unsigned long long)c, (unsigned long long)d);
    for (uint64_t ptr : {a, b, c})
        if (ptr > 0x10000 && !IsBadReadPtr(reinterpret_cast<void*>(ptr), 0x60))
        {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(ptr);
            LOGI("  *0x%llx: %08x %08x %08x %08x %08x %08x %08x %08x  %08x %08x %08x %08x %08x %08x %08x %08x",
                 (unsigned long long)ptr, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11],
                 w[12], w[13], w[14], w[15]);
        }
    return g_probeMode == 2 ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

void ProfRememberModule(VkShaderModule mod, const VkShaderModuleCreateInfo* ci, uint64_t originalHash);

// Shader modules the DLL creates, keyed by a 64-bit FNV-1a hash of the original SPIR-V: a module is replaced with
// its replacement from Shaders() when there is one, so a replacement only ever applies to the exact module it was made
// for. DU_FSR4_VK_DUMP=dir writes each original to dir\<hash>_<entry point>.spv (diagnostics).
uint64_t SpirvHash(const void* code, size_t size)
{
    uint64_t       h = 1469598103934665603ull;
    const uint8_t* b = static_cast<const uint8_t*>(code);
    for (size_t i = 0; i < size; ++i)
        h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

const char* SpirvEntryPoint(const uint32_t* w, size_t words)
{
    for (size_t i = 5; i < words;)
    {
        const uint32_t op = w[i] & 0xFFFF, len = w[i] >> 16;
        if (!len)
            break;
        if (op == 15 && len > 3)  // OpEntryPoint: model, id, literal name
            return reinterpret_cast<const char*>(&w[i + 3]);
        i += len;
    }
    return "?";
}

std::vector<uint32_t> ReadSpirv(const char* path)
{
    std::vector<uint32_t> code;
    if (FILE* f = fopen(path, "rb"))
    {
        fseek(f, 0, SEEK_END);
        const long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (n > 20 && n % 4 == 0)
        {
            code.resize(size_t(n) / 4);
            if (fread(code.data(), 1, size_t(n), f) != size_t(n))
                code.clear();
        }
        fclose(f);
    }
    return code;
}

// Where replacement modules come from (none of them are part of this repository):
//  - DU_FSR4_VK_SHADER_DIR=dir: dir\<hash>.spv files (shader development);
//  - else packs beside the DLL, installed by the plugin when it has them: fsr4_shaders.bin, and fsr4_shaders_dot2.bin
//    whose modules use v_dot2_f32_f16 (SPV_VALVE_mixed_float_dot_product) and take precedence when the device has
//    VK_VALVE_shader_mixed_float_dot_product enabled (see Dot2Enabled). Format: "DUSPACK1" (older packs: "AISSSPV1"),
//    u32 count, then per module u64 hash, u32 word count and the SPIR-V words.
// DU_FSR4_VK_SHADERS=0 turns replacement off; DU_FSR4_VK_DOT2=0 keeps to fsr4_shaders.bin.
//
// A set can also take over work the proxy or the DLL otherwise does, declared in an options entry under the reserved
// hash 0 (dir\0000000000000000.spv in a shader dir). It has a SPIR-V header so older proxies load it as a module no
// hash matches; after the 5 header words: u32 flags, u32 count, then count u64 module hashes whose dispatches are not
// recorded (the set's shaders do their work). Flag 1: the set's colour readers clean Inf / NaN colour themselves, so
// the proxy's clean-up pass for auto exposure is skipped.
using ShaderPack = std::vector<std::pair<uint64_t, std::vector<uint32_t>>>;

struct PackOptions
{
    static constexpr uint32_t kCleansColour = 1;
    uint32_t              flags = 0;
    std::vector<uint64_t> skipDispatch;
    bool operator==(const PackOptions& o) const { return flags == o.flags && skipDispatch == o.skipDispatch; }
};

PackOptions ParseOptions(const ShaderPack& pack)
{
    PackOptions o;
    for (const auto& m : pack)
    {
        const std::vector<uint32_t>& w = m.second;
        if (m.first != 0 || w.size() < 7 || w.size() < 7 + size_t(w[6]) * 2)
            continue;
        o.flags = w[5];
        for (uint32_t i = 0; i < w[6]; ++i)
            o.skipDispatch.push_back(w[7 + i * 2] | uint64_t(w[8 + i * 2]) << 32);
    }
    return o;
}

struct ShaderSource
{
    std::string dir;
    ShaderPack  pack, dot2;
    PackOptions options, dot2Options;
};

ShaderPack ReadPack(const wchar_t* name)
{
    ShaderPack   pack;
    std::wstring path = DllPath();
    path              = path.substr(0, path.find_last_of(L'\\') + 1) + name;
    FILE* f           = _wfopen(path.c_str(), L"rb");
    if (!f)
        return pack;
    char     magic[8] = {};
    uint32_t count    = 0;
    bool ok = fread(magic, 1, 8, f) == 8 && (!memcmp(magic, "DUSPACK1", 8) || !memcmp(magic, "AISSSPV1", 8)) &&
              fread(&count, 4, 1, f) == 1 && count < 4096;
    for (uint32_t i = 0; ok && i < count; ++i)
    {
        uint64_t hash  = 0;
        uint32_t words = 0;
        ok = fread(&hash, 8, 1, f) == 1 && fread(&words, 4, 1, f) == 1 && words > 5 && words < (16u << 20);
        if (!ok)
            break;
        std::vector<uint32_t> code(words);
        ok = fread(code.data(), 4, words, f) == words && code[0] == 0x07230203u;  // SPIR-V magic
        if (ok)
            pack.push_back({hash, std::move(code)});
    }
    fclose(f);
    if (!ok)
        pack.clear();
    LOGI("FSR 4 Vulkan: %ls: %s", path.c_str(),
         ok ? (std::to_string(pack.size()) + " replacement shader modules").c_str() : "unreadable, not used");
    return pack;
}

const ShaderSource& Shaders()
{
    static const ShaderSource src = [] {
        ShaderSource s;
        if (const char* off = getenv("DU_FSR4_VK_SHADERS"); off && !strcmp(off, "0"))
            return s;
        if (const char* env = getenv("DU_FSR4_VK_SHADER_DIR"); env && *env)
        {
            s.dir = env;
            std::vector<uint32_t> opts = ReadSpirv((s.dir + "\\0000000000000000.spv").c_str());
            if (!opts.empty())
                s.options = ParseOptions({{0, std::move(opts)}});
            return s;
        }
        s.pack    = ReadPack(L"fsr4_shaders.bin");
        s.options = ParseOptions(s.pack);
        if (const char* off = getenv("DU_FSR4_VK_DOT2"); !(off && !strcmp(off, "0")))
        {
            s.dot2        = ReadPack(L"fsr4_shaders_dot2.bin");
            s.dot2Options = ParseOptions(s.dot2);
        }
        return s;
    }();
    return src;
}

// Whether the dot2 modules are valid on the FSR 4 device: the GPU must have VK_VALVE_shader_mixed_float_dot_product, and
// the device must have enabled it, which vkd3d-proton does itself (D3D12 games) and the DeckyUpscaler layer does for
// every other device (DU_LAYER=1, which the plugin sets). Set by Fsr4VkUpscaler::Create before the DLL builds its
// pipelines.
std::atomic<bool> g_dot2{false};

bool Dot2Enabled(const VulkanDevice& dev, CostApi api)
{
    if (Shaders().dot2.empty() || !(api == CostApi::D3D12 || Config().mlFeaturesForced))
        return false;
    auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        dev.gipa(dev.instance, "vkEnumerateDeviceExtensionProperties"));
    uint32_t n = 0;
    if (!enumerate || enumerate(dev.physicalDevice, nullptr, &n, nullptr) != VK_SUCCESS)
        return false;
    std::vector<VkExtensionProperties> exts(n);
    enumerate(dev.physicalDevice, nullptr, &n, exts.data());
    for (const auto& e : exts)
        if (!strcmp(e.extensionName, "VK_VALVE_shader_mixed_float_dot_product"))
            return true;
    return false;
}

// Modules replaced so far: a set's options only hold when its modules are in use (not for another DLL build).
std::atomic<uint32_t> g_replacedModules{0};

// The options of the set in use: the dot2 set's must match, as its modules stand in for some of the first set's.
const PackOptions& ActiveOptions()
{
    static const PackOptions none;
    const ShaderSource& src = Shaders();
    if (!g_replacedModules || (g_dot2 && !src.dot2.empty() && !(src.dot2Options == src.options)))
        return none;
    return src.options;
}

PFN_vkCreateShaderModule g_realCreateShaderModule = nullptr;
VkResult VKAPI_CALL HookedCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* ci,
                                             const VkAllocationCallbacks* alloc, VkShaderModule* out)
{
    if (!ci || !ci->pCode)
        return g_realCreateShaderModule(device, ci, alloc, out);
    const uint64_t hash  = SpirvHash(ci->pCode, ci->codeSize);
    const char*    entry = SpirvEntryPoint(ci->pCode, ci->codeSize / 4);
    char           path[MAX_PATH];
    if (const char* dir = getenv("DU_FSR4_VK_DUMP"); dir && *dir)
    {
        snprintf(path, sizeof(path), "%s\\%016llx_%s.spv", dir, (unsigned long long)hash, entry);
        if (FILE* f = fopen(path, "wb"))
        {
            fwrite(ci->pCode, 1, ci->codeSize, f);
            fclose(f);
        }
    }
    VkShaderModuleCreateInfo replaced = *ci;
    std::vector<uint32_t>    code;
    const ShaderSource& src = Shaders();
    if (!src.dir.empty())
    {
        snprintf(path, sizeof(path), "%s\\%016llx.spv", src.dir.c_str(), (unsigned long long)hash);
        code = ReadSpirv(path);
    }
    else
    {
        snprintf(path, sizeof(path), "fsr4_shaders.bin");
        for (const auto& m : src.pack)
            if (m.first == hash)
                code = m.second;
        if (g_dot2)
            for (const auto& m : src.dot2)
                if (m.first == hash)
                    code = m.second, snprintf(path, sizeof(path), "fsr4_shaders_dot2.bin");
    }
    if (!code.empty())
    {
        replaced.pCode    = code.data();
        replaced.codeSize = code.size() * 4;
        ++g_replacedModules;
        LOGI("FSR 4 Vulkan: %s (%016llx) replaced from %s", entry, (unsigned long long)hash, path);
    }
    const VkResult r = g_realCreateShaderModule(device, &replaced, alloc, out);
    if (r == VK_SUCCESS)
        ProfRememberModule(*out, &replaced, hash);
    else
        LOGI("FSR 4 Vulkan: vkCreateShaderModule for %s failed (%d)", entry, r);
    return r;
}

// The DLL's compute pipelines and dispatches. Dispatches of pipelines made from a module in the set's skip list are not
// recorded (ActiveOptions). DU_FSR4_VK_PROFILE=N: GPU time of each compute dispatch the DLL records, by shader entry
// point, averaged over N evaluations (diagnostics). Pipelines are named from the entry point of the module they were
// created from.
struct PassProfiler
{
    static constexpr uint32_t kMaxPerEval = 48, kRing = 6;
    VkDevice                   device = VK_NULL_HANDLE;
    VkQueryPool                pool   = VK_NULL_HANDLE;
    float                      period = 1.0f;
    PFN_vkCmdWriteTimestamp    write  = nullptr;
    PFN_vkCmdResetQueryPool    reset  = nullptr;
    PFN_vkGetQueryPoolResults  results = nullptr;
    PFN_vkCmdDispatch          realDispatch = nullptr;
    PFN_vkCmdBindPipeline      realBind = nullptr;
    PFN_vkCreateComputePipelines realCreatePipes = nullptr;
    std::mutex                 m;
    std::vector<std::pair<VkShaderModule, std::string>> moduleNames;
    std::vector<std::pair<VkPipeline, std::string>>     pipeNames;
    std::vector<VkShaderModule> skipModules;
    std::vector<VkPipeline>     skipPipes;
    std::string                current;
    bool                       currentSkipped = false;
    uint64_t                   eval = 0;
    uint32_t                   used[kRing] = {};
    std::vector<std::string>   names[kRing];
    std::vector<std::pair<std::string, double>> sums;
    uint32_t                   evalsSummed = 0;
    bool                       active = false;
};
PassProfiler g_prof;

void ProfRememberModule(VkShaderModule mod, const VkShaderModuleCreateInfo* ci, uint64_t originalHash)
{
    const std::vector<uint64_t>& skip = Shaders().options.skipDispatch;  // whether it holds is decided per dispatch
    if (std::find(skip.begin(), skip.end(), originalHash) != skip.end())
    {
        std::lock_guard<std::mutex> l(g_prof.m);
        g_prof.skipModules.push_back(mod);
    }
    // Entry point name from the SPIR-V OpEntryPoint (opcode 15): word 3 onwards is the literal string.
    std::string name = "?";
    const uint32_t* w = ci->pCode;
    const size_t    n = ci->codeSize / 4;
    for (size_t i = 5; i < n;)
    {
        const uint32_t op = w[i] & 0xFFFF, len = w[i] >> 16;
        if (!len)
            break;
        if (op == 15 && len > 3)
        {
            name = reinterpret_cast<const char*>(&w[i + 3]);
            break;
        }
        i += len;
    }
    std::lock_guard<std::mutex> l(g_prof.m);
    g_prof.moduleNames.push_back({mod, name});
}

VkResult VKAPI_CALL ProfCreateComputePipelines(VkDevice d, VkPipelineCache c, uint32_t n, const VkComputePipelineCreateInfo* ci,
                                               const VkAllocationCallbacks* a, VkPipeline* out)
{
    const VkResult r = g_prof.realCreatePipes(d, c, n, ci, a, out);
    std::lock_guard<std::mutex> l(g_prof.m);
    for (uint32_t i = 0; i < n && r == VK_SUCCESS; ++i)
    {
        std::string name = "?";
        for (auto& mn : g_prof.moduleNames)
            if (mn.first == ci[i].stage.module)
                name = mn.second;
        g_prof.pipeNames.push_back({out[i], name});
        if (std::find(g_prof.skipModules.begin(), g_prof.skipModules.end(), ci[i].stage.module) != g_prof.skipModules.end())
            g_prof.skipPipes.push_back(out[i]);
    }
    return r;
}

void VKAPI_CALL ProfBindPipeline(VkCommandBuffer cmd, VkPipelineBindPoint bp, VkPipeline p)
{
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE)
    {
        std::lock_guard<std::mutex> l(g_prof.m);
        g_prof.current = "?";
        for (auto& pn : g_prof.pipeNames)
            if (pn.first == p)
                g_prof.current = pn.second;
        g_prof.currentSkipped = std::find(g_prof.skipPipes.begin(), g_prof.skipPipes.end(), p) != g_prof.skipPipes.end();
    }
    g_prof.realBind(cmd, bp, p);
}

void VKAPI_CALL ProfDispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z)
{
    uint32_t q = UINT32_MAX;
    {
        std::lock_guard<std::mutex> l(g_prof.m);
        if (g_prof.currentSkipped && !ActiveOptions().skipDispatch.empty())
            return;
        const uint32_t ring = uint32_t(g_prof.eval % PassProfiler::kRing);
        if (g_prof.active && g_prof.pool && g_prof.used[ring] < PassProfiler::kMaxPerEval)
        {
            q = ring * PassProfiler::kMaxPerEval * 2 + g_prof.used[ring] * 2;
            g_prof.names[ring].push_back(g_prof.current + " " + std::to_string(x) + "x" + std::to_string(y));
            ++g_prof.used[ring];
        }
    }
    if (q != UINT32_MAX)
        g_prof.write(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_prof.pool, q);
    g_prof.realDispatch(cmd, x, y, z);
    if (q != UINT32_MAX)
        g_prof.write(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_prof.pool, q + 1);
}

// Around each evaluate: collect the ring slot recorded kRing evaluations ago, reset it, and mark it current.
void ProfBeginEval(const VulkanDevice& dev, VkCommandBuffer cmd)
{
    static const int every = getenv("DU_FSR4_VK_PROFILE") ? atoi(getenv("DU_FSR4_VK_PROFILE")) : 0;
    if (every <= 0)
        return;
    std::lock_guard<std::mutex> l(g_prof.m);
    if (!g_prof.pool)
    {
        auto create = reinterpret_cast<PFN_vkCreateQueryPool>(dev.gdpa(dev.device, "vkCreateQueryPool"));
        g_prof.write   = reinterpret_cast<PFN_vkCmdWriteTimestamp>(dev.gdpa(dev.device, "vkCmdWriteTimestamp"));
        g_prof.reset   = reinterpret_cast<PFN_vkCmdResetQueryPool>(dev.gdpa(dev.device, "vkCmdResetQueryPool"));
        g_prof.results = reinterpret_cast<PFN_vkGetQueryPoolResults>(dev.gdpa(dev.device, "vkGetQueryPoolResults"));
        auto props     = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(dev.gipa(dev.instance, "vkGetPhysicalDeviceProperties"));
        VkPhysicalDeviceProperties pp = {};
        props(dev.physicalDevice, &pp);
        g_prof.period             = pp.limits.timestampPeriod;
        VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType             = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount            = PassProfiler::kMaxPerEval * 2 * PassProfiler::kRing;
        create(dev.device, &qci, nullptr, &g_prof.pool);
        g_prof.device = dev.device;
    }
    ++g_prof.eval;
    const uint32_t ring = uint32_t(g_prof.eval % PassProfiler::kRing);
    // Collect what this slot held (kRing evaluations ago).
    if (g_prof.used[ring])
    {
        std::vector<uint64_t> r(g_prof.used[ring] * 4);
        if (g_prof.results(g_prof.device, g_prof.pool, ring * PassProfiler::kMaxPerEval * 2, g_prof.used[ring] * 2,
                           r.size() * 8, r.data(), 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ==
            VK_SUCCESS)
        {
            if (g_prof.sums.size() != g_prof.used[ring])
                g_prof.sums.assign(g_prof.used[ring], {"", 0.0}), g_prof.evalsSummed = 0;
            for (uint32_t i = 0; i < g_prof.used[ring]; ++i)
            {
                g_prof.sums[i].first = g_prof.names[ring][i];
                g_prof.sums[i].second += double(r[i * 4 + 2] - r[i * 4]) * g_prof.period / 1000.0;
            }
            if (++g_prof.evalsSummed >= uint32_t(every))
            {
                double total = 0;
                for (auto& sp : g_prof.sums)
                    total += sp.second;
                LOGI("FSR 4 Vulkan profile: %zu dispatches, %.1f us per evaluate", g_prof.sums.size(), total / g_prof.evalsSummed);
                for (auto& sp : g_prof.sums)
                    LOGI("  %8.1f us  %s", sp.second / g_prof.evalsSummed, sp.first.c_str());
                g_prof.sums.clear();
                g_prof.evalsSummed = 0;
            }
        }
    }
    g_prof.used[ring] = 0;
    g_prof.names[ring].clear();
    g_prof.reset(cmd, g_prof.pool, ring * PassProfiler::kMaxPerEval * 2, PassProfiler::kMaxPerEval * 2);
    g_prof.active = true;
}

// The DLL allocates its weights and tensors from the first memory type its filter allows, which on RADV is a
// DEVICE_COHERENT_AMD | DEVICE_UNCACHED_AMD type: every access then bypasses the GPU's L2 and FSR 4 runs ~2.4x slower
// (7.3 vs 3.0 ms on the Deck). Such allocations move to the same heap's plain type with the same other flags.
// DU_FSR4_VK_KEEP_UNCACHED=1 keeps AMD's choice; DU_LOG=2 logs each allocation.
PFN_vkAllocateMemory             g_realAllocateMemory = nullptr;
VkPhysicalDeviceMemoryProperties g_memProps           = {};
VkResult VKAPI_CALL LogAllocateMemory(VkDevice d, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks* a,
                                      VkDeviceMemory* out)
{
    VkMemoryAllocateInfo fixed = *ai;
    const VkMemoryPropertyFlags kAmdUncached = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    const VkMemoryType& t = g_memProps.memoryTypes[ai->memoryTypeIndex];
    if ((t.propertyFlags & kAmdUncached) && !getenv("DU_FSR4_VK_KEEP_UNCACHED"))
        for (uint32_t i = 0; i < g_memProps.memoryTypeCount; ++i)
            if (g_memProps.memoryTypes[i].propertyFlags == (t.propertyFlags & ~kAmdUncached) &&
                g_memProps.memoryTypes[i].heapIndex == t.heapIndex)
            {
                fixed.memoryTypeIndex = i;
                break;
            }
    const VkResult r = g_realAllocateMemory(d, &fixed, a, out);
    if (ai)
        Log(2, "FSR 4 Vulkan memory: %llu bytes, type %u (flags 0x%x) -> type %u: %d", (unsigned long long)ai->allocationSize,
             ai->memoryTypeIndex, g_memProps.memoryTypes[ai->memoryTypeIndex].propertyFlags, fixed.memoryTypeIndex, r);
    return r;
}

PFN_vkVoidFunction VKAPI_CALL HookedGdpa(VkDevice device, const char* name)
{
    if (name && !strcmp(name, "vkUpdateFfxApiProviderAMD"))
    {
        Log(2, "FSR 4 Vulkan: the DLL looks up %s (AMD driver hook; RADV has none)", name);
        return g_probeMode ? reinterpret_cast<PFN_vkVoidFunction>(&ProbeUpdateProvider) : nullptr;
    }
    if (name && !strcmp(name, "vkAllocateMemory"))
        return g_realAllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(g_gameGdpa(device, name)),
               reinterpret_cast<PFN_vkVoidFunction>(&LogAllocateMemory);
    if (name && (getenv("DU_FSR4_VK_PROFILE") || !Shaders().options.skipDispatch.empty()))
    {
        if (!strcmp(name, "vkCmdDispatch"))
            return g_prof.realDispatch = reinterpret_cast<PFN_vkCmdDispatch>(g_gameGdpa(device, name)),
                   reinterpret_cast<PFN_vkVoidFunction>(&ProfDispatch);
        if (!strcmp(name, "vkCmdBindPipeline"))
            return g_prof.realBind = reinterpret_cast<PFN_vkCmdBindPipeline>(g_gameGdpa(device, name)),
                   reinterpret_cast<PFN_vkVoidFunction>(&ProfBindPipeline);
        if (!strcmp(name, "vkCreateComputePipelines"))
            return g_prof.realCreatePipes = reinterpret_cast<PFN_vkCreateComputePipelines>(g_gameGdpa(device, name)),
                   reinterpret_cast<PFN_vkVoidFunction>(&ProfCreateComputePipelines);
    }
    if (name && !strcmp(name, "vkCreateShaderModule"))
    {
        g_realCreateShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(g_gameGdpa(device, name));
        return reinterpret_cast<PFN_vkVoidFunction>(&HookedCreateShaderModule);
    }
    return g_gameGdpa(device, name);
}

FfxApiResource Resource(const UpscalerImage& img, bool output)
{
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType         = VK_IMAGE_TYPE_2D;
    ci.format            = img.format;
    ci.extent            = {img.width, img.height, 1};
    ci.mipLevels         = 1;
    ci.arrayLayers       = 1;
    ci.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | (output ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    const uint32_t state = output || img.layout == VK_IMAGE_LAYOUT_GENERAL ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS
                                                                          : FFX_API_RESOURCE_STATE_COMPUTE_READ;
    return ffxApiGetResourceVK(reinterpret_cast<void*>(img.image), ffxApiGetImageResourceDescriptionVK(img.image, ci, 0),
                               state);
}

// FSR 3.1 settings (the SDK DLL's FSR 3.1.6, DU_UPSCALER=fsr3 / control code 9): AMD's defaults. The values that bring
// its output closest to FSR 4 INT8's (accumulation 0.05, shading change 32: 53.9 -> 56.1 dB against INT8) are in the
// README ("FSR 3.1"). Each can be overridden:
// DU_FSR3_SHARPNESS (< 0 = sharpening off), DU_FSR3_VELOCITY, DU_FSR3_REACTIVENESS, DU_FSR3_SHADING,
// DU_FSR3_ACCUMULATION, DU_FSR3_MIN_DISOCCLUSION (the SDK's configure keys of the same names).
struct Fsr3Tuning
{
    float sharpness        = -1.0f;
    float velocity         = 1.0f;
    float reactiveness     = 1.0f;
    float shading          = 1.0f;
    float accumulation     = 0.333f;
    float minDisocclusion  = -0.333f;
};

const Fsr3Tuning& Fsr3Settings()
{
    static const Fsr3Tuning t = [] {
        Fsr3Tuning r;
        auto env = [](const char* name, float& v) {
            if (const char* s = getenv(name); s && *s)
                v = float(atof(s));
        };
        env("DU_FSR3_SHARPNESS", r.sharpness);
        env("DU_FSR3_VELOCITY", r.velocity);
        env("DU_FSR3_REACTIVENESS", r.reactiveness);
        env("DU_FSR3_SHADING", r.shading);
        env("DU_FSR3_ACCUMULATION", r.accumulation);
        env("DU_FSR3_MIN_DISOCCLUSION", r.minDisocclusion);
        return r;
    }();
    return t;
}

class Fsr4VkUpscaler : public Upscaler
{
public:
    static std::unique_ptr<Upscaler> Create(const VulkanDevice& device, const UpscalerCreateInfo& info)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const bool fsr3 = info.backend == kBackendFsr3Vk;
        if ((fsr3 ? g_fsr3Failed : g_deviceFailed) || !LoadSdk())
            return nullptr;
        std::unique_ptr<Fsr4VkUpscaler> up(new Fsr4VkUpscaler());
        up->m_fsr3 = fsr3;
        up->m_dev  = device;
        up->m_info = info;
        up->m_dc   = {device.device, device.physicalDevice, device.gdpa};
        // Every Vulkan entry point the DLL looks up goes through HookedGdpa (it uses both its vulkan-1.dll import and
        // the device context's pointer): memory type fix-up, the absent driver hook, and the optional diagnostics.
        if (!g_gameGdpa)
        {
            g_probeMode = getenv("DU_FSR4_VK_PROBE") ? atoi(getenv("DU_FSR4_VK_PROBE")) : 0;
            g_gameGdpa  = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
                Fsr4RedirectImport(g_sdk.lib, "vulkan-1.dll", "vkGetDeviceProcAddr", reinterpret_cast<void*>(&HookedGdpa)));
            if (!g_gameGdpa)
                LOGI("FSR 4 Vulkan: the DLL imports no vulkan-1.dll!vkGetDeviceProcAddr; only its device context is hooked");
            if (!g_gameGdpa)
                g_gameGdpa = device.gdpa;
        }
        if (g_gameGdpa)
        {
            up->m_dc.vkDeviceProcAddr = &HookedGdpa;
            auto memProps = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
                device.gipa(device.instance, "vkGetPhysicalDeviceMemoryProperties"));
            memProps(device.physicalDevice, &g_memProps);
        }

        // AMD's own device checks accept no network on RADV (they expect AMD's driver), so the network is chosen here:
        // auto is the GPU's own network (Lite on a Steam Deck). The checks run while the DLL enumerates versions.
        const Fsr4Gpu gpu    = Fsr4KnownGpu(0x1002, 0x163F, "VANGOGH");
        const int     wanted = info.fsr4Model == kFsr4Auto && Fsr4ModelForGpu(gpu) > 0 ? Fsr4ModelForGpu(gpu) : info.fsr4Model;
        std::string   versionName;
        Fsr4BeginCreate(wanted);
        const uint64_t versionId = PickVersion(&up->m_dc, &versionName, fsr3 ? '3' : '4');
        g_dot2 = Dot2Enabled(device, CostGetApi());
        if (!fsr3 && !Shaders().dot2.empty())
            LOGI("FSR 4 Vulkan: dot2 shader modules %s", g_dot2 ? "in use" : "not used (device lacks VK_VALVE_shader_mixed_float_dot_product or it is not enabled)");
        int            native    = 0;
        const int      accepted  = Fsr4EndCreate(&native);
        if (!versionId)
        {
            if (fsr3)
                LOGI("FSR 3 Vulkan: no FSR 3 upscaler offered on this device; using NSS");
            else
                LOGI("FSR 4 Vulkan: no FSR 4 upscaler offered on this device; FSR 4 goes through the D3D12 interop");
            (fsr3 ? g_fsr3Failed : g_deviceFailed) = true;
            return nullptr;
        }

        ffxCreateBackendVKDesc backend = {};
        backend.header.type            = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.deviceContext          = up->m_dc;
        ffxOverrideVersion override    = {};
        override.header.type           = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        override.versionId             = versionId;

        ffxCreateContextDescUpscale create = {};
        create.header.type                 = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        create.flags                       = FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
        if (info.hdr)
            create.flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
        if (info.depthInverted)
            create.flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
        if (info.highResMv)
            create.flags |= FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
        up->m_autoExposure = info.autoExposure && Config().exposure <= 0.0f;
        if (up->m_autoExposure)
            create.flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        if (LogLevel() >= 2)
            create.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
        create.maxRenderSize  = {std::min(info.renderWidth, info.displayWidth), std::min(info.renderHeight, info.displayHeight)};
        create.maxUpscaleSize = {info.displayWidth, info.displayHeight};
        create.fpMessage      = &FfxMessage;
        create.header.pNext   = &backend.header;
        backend.header.pNext  = &override.header;

        const auto           t0 = GetTickCount64();
        const ffxReturnCode_t rc = g_sdk.create(&up->m_ctx, &create.header, nullptr);
        if (rc != FFX_API_RETURN_OK || !up->m_ctx)
        {
            LOGI("FSR %d Vulkan: createContext failed (%u); %s", fsr3 ? 3 : 4, rc,
                 fsr3 ? "using NSS" : "FSR 4 goes through the D3D12 interop");
            (fsr3 ? g_fsr3Failed : g_deviceFailed) = true;
            up->m_ctx      = nullptr;
            return nullptr;
        }
        if (fsr3)
        {
            // FSR 3.1's own knobs (no network): the tuned defaults, or the DU_FSR3_* overrides.
            const Fsr3Tuning& t = Fsr3Settings();
            const std::pair<uint64_t, float> keys[] = {
                {FFX_API_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR, t.velocity},
                {FFX_API_CONFIGURE_UPSCALE_KEY_FREACTIVENESSSCALE, t.reactiveness},
                {FFX_API_CONFIGURE_UPSCALE_KEY_FSHADINGCHANGESCALE, t.shading},
                {FFX_API_CONFIGURE_UPSCALE_KEY_FACCUMULATIONADDEDPERFRAME, t.accumulation},
                {FFX_API_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION, t.minDisocclusion},
            };
            for (const auto& kv : keys)
            {
                float                           v = kv.second;
                ffxConfigureDescUpscaleKeyValue c = {};
                c.header.type                     = FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE;
                c.key                             = kv.first;
                c.ptr                             = &v;
                if (g_sdk.configure)
                    g_sdk.configure(&up->m_ctx, &c.header);
            }
            up->m_info.fsr4Active = -1;
            LOGI("FSR 3 Vulkan: upscaler %s, context %ux%u -> %ux%u, flags 0x%x; sharpness %.2f, velocity %.2f, "
                 "reactiveness %.2f, shading %.2f, accumulation %.3f, min disocclusion %.3f",
                 versionName.c_str(), create.maxRenderSize.width, create.maxRenderSize.height, info.displayWidth,
                 info.displayHeight, create.flags, t.sharpness, t.velocity, t.reactiveness, t.shading, t.accumulation,
                 t.minDisocclusion);
            return up;
        }
        up->m_info.fsr4Active = accepted;
        LOGI("FSR 4 Vulkan: upscaler %s, network %s (asked for %s; AMD's own checks accept: 0x%x), context %ux%u -> "
             "%ux%u, flags 0x%x, created in %llu ms on the game's device",
             versionName.c_str(), Fsr4ModelName(accepted), Fsr4ModelName(info.fsr4Model), native,
             create.maxRenderSize.width, create.maxRenderSize.height, info.displayWidth, info.displayHeight, create.flags,
             (unsigned long long)(GetTickCount64() - t0));
        return up;
    }

    ~Fsr4VkUpscaler() override
    {
        // NGX releases features after the game has idled its device (see NVSDK_NGX_VULKAN_ReleaseFeature).
        if (m_ctx)
            g_sdk.destroy(&m_ctx, nullptr);
    }

    bool Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& in) override
    {
        // The FFX Vulkan backend reads inputs in SHADER_READ_ONLY_OPTIMAL or GENERAL and writes the output in GENERAL.
        // Images in any other layout (DXVK can hand over depth as DEPTH_STENCIL_READ_ONLY_OPTIMAL) are moved there for
        // the dispatch and back afterwards.
        UpscalerDispatch d = in;
        UpscalerImage* imgs[4] = {&d.color, &d.depth, &d.motionVectors, &d.output};
        VkImageMemoryBarrier pre[4], post[4];
        uint32_t             nb = 0;
        for (int i = 0; i < 4; ++i)
        {
            UpscalerImage&      img  = *imgs[i];
            const VkImageLayout want = i == 3 ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (img.layout == want || img.layout == VK_IMAGE_LAYOUT_GENERAL)
                continue;
            VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | (i == 3 ? VK_ACCESS_SHADER_WRITE_BIT : 0);
            b.oldLayout = img.layout;
            b.newLayout = want;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img.image;
            b.subresourceRange = {ffxApiIsDepthFormat(img.format)
                                      ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT |
                                                           (ffxApiIsStencilFormat(img.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0))
                                      : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT),
                                  0, 1, 0, 1};
            pre[nb]  = b;
            std::swap(b.oldLayout, b.newLayout);
            b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            post[nb++] = b;
            img.layout = want;
        }
        if (nb && !m_barrier)
            m_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(m_dev.gdpa(m_dev.device, "vkCmdPipelineBarrier"));
        if (nb)
            m_barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                      nullptr, nb, pre);
        const bool ok = Record(cmd, d);
        if (nb)
            m_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                      nullptr, nb, post);
        return ok;
    }

private:
    bool Record(VkCommandBuffer cmd, const UpscalerDispatch& in)
    {
        // Optional motion vector dilation (render-resolution MVs only), switchable live.
        UpscalerDispatch d = in;
        if (!m_info.highResMv && DilateMvEnabled() && !m_dilatorFailed)
        {
            if (!m_dilator)
            {
                m_dilator = MvDilator::Create(m_dev, m_info.renderWidth, m_info.renderHeight);
                m_dilatorFailed = !m_dilator;
                LOGI("FSR %d Vulkan: motion vector dilation %s", m_fsr3 ? 3 : 4, m_dilator ? "on" : "unavailable");
            }
            if (m_dilator)
                d.motionVectors = m_dilator->Record(cmd, in.depth, in.motionVectors, in.renderWidth, in.renderHeight,
                                                    m_info.depthInverted);
        }
        // FSR 4's auto exposure turns one Inf / NaN colour pixel into a black frame (and black history): clean the
        // colour first, in its own format. UNORM colour cannot hold either; FSR 3.1's auto exposure is unaffected.
        // A shader set that cleans the colour itself makes the pass unnecessary (known once the DLL has created its
        // modules, during the first evaluate).
        const bool setCleans = (ActiveOptions().flags & PackOptions::kCleansColour) != 0;
        if (setCleans && m_autoExposure && !m_setCleansLogged)
        {
            m_setCleansLogged = true;
            LOGI("FSR 4 Vulkan: colour clean-up for auto exposure done by the replacement shaders");
        }
        const VkFormat cleanFormat = m_autoExposure && !m_fsr3 && !setCleans ? MvDilator::SanitizeFormat(in.color.format)
                                                                              : VK_FORMAT_UNDEFINED;
        if (cleanFormat != VK_FORMAT_UNDEFINED && cleanFormat != m_sanitizerFormat)
        {
            m_sanitizer       = MvDilator::Create(m_dev, m_info.renderWidth, m_info.renderHeight,
                                                  MvDilator::Pass::SanitizeColor, cleanFormat);
            m_sanitizerFormat = cleanFormat;
            LOGI("FSR 4 Vulkan: colour clean-up for auto exposure %s (format %d)", m_sanitizer ? "on" : "unavailable",
                 int(cleanFormat));
        }
        if (cleanFormat != VK_FORMAT_UNDEFINED && m_sanitizer)
            d.color = m_sanitizer->Record(cmd, in.color, in.color, in.renderWidth, in.renderHeight, false);
        ffxDispatchDescUpscale up  = {};
        up.header.type             = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        up.commandList             = cmd;
        up.color                   = Resource(d.color, false);
        up.depth                   = Resource(d.depth, false);
        up.motionVectors           = Resource(d.motionVectors, false);
        up.output                  = Resource(d.output, true);
        up.jitterOffset            = {d.jitterX * Config().jitterSign[0], d.jitterY * Config().jitterSign[1]};
        up.motionVectorScale       = {d.mvScaleX * Config().mvSign[0], d.mvScaleY * Config().mvSign[1]};
        up.renderSize              = {d.renderWidth, d.renderHeight};
        up.upscaleSize             = {d.output.width, d.output.height};
        up.enableSharpening        = m_fsr3 && Fsr3Settings().sharpness >= 0.0f;
        up.sharpness               = m_fsr3 ? std::min(std::max(Fsr3Settings().sharpness, 0.0f), 1.0f) : 0.0f;
        up.frameTimeDelta          = d.frameTimeMs;
        up.preExposure             = 1.0f;
        up.reset                   = d.reset || m_dispatchCount == 0;
        up.cameraNear              = Config().cameraNear;
        up.cameraFar               = Config().cameraFar;
        up.cameraFovAngleVertical  = Config().cameraFovY;
        up.viewSpaceToMetersFactor = 1.0f;
        if (m_info.depthInverted)
            std::swap(up.cameraNear, up.cameraFar);
        ProfBeginEval(m_dev, cmd);
        const ffxReturnCode_t rc = g_sdk.dispatch(&m_ctx, &up.header);
        ++m_dispatchCount;
        if ((m_dispatchCount % 600) == 1 || LogLevel() >= 2)
            LOGI("FSR %d Vulkan evaluate %llu: render %ux%u -> %ux%u, jitter %.3f,%.3f, mv scale %.2f,%.2f%s",
                 m_fsr3 ? 3 : 4, (unsigned long long)m_dispatchCount, up.renderSize.width, up.renderSize.height,
                 up.upscaleSize.width, up.upscaleSize.height, up.jitterOffset.x, up.jitterOffset.y,
                 up.motionVectorScale.x, up.motionVectorScale.y, m_autoExposure ? " (auto exposure)" : "");
        if (rc != FFX_API_RETURN_OK)
        {
            LOGI("FSR %d Vulkan: dispatch failed (%u)", m_fsr3 ? 3 : 4, rc);
            return false;
        }
        return true;
    }

    Fsr4VkUpscaler() = default;
    PFN_vkCmdPipelineBarrier m_barrier = nullptr;
    bool                     m_fsr3    = false;  // FSR 3.1 from the same DLL (no network, no shader replacement)
    std::unique_ptr<MvDilator> m_dilator;
    bool                       m_dilatorFailed = false;
    std::unique_ptr<MvDilator> m_sanitizer;  // colour clean-up (MvDilator::Pass::SanitizeColor)
    VkFormat                   m_sanitizerFormat = VK_FORMAT_UNDEFINED;  // m_sanitizer's format (tried, even if null)
    bool                       m_setCleansLogged = false;
    VkDeviceContext m_dc            = {};
    ffxContext      m_ctx           = nullptr;
    uint64_t        m_dispatchCount = 0;
    bool            m_autoExposure  = false;
};
}  // namespace

std::unique_ptr<Upscaler> CreateFsr4VkUpscaler(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    return Fsr4VkUpscaler::Create(device, info);
}

bool Fsr3VkUnavailable()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_fsr3Failed || (g_loadTried && !g_loadOk);
}

bool Fsr4VkUnavailable()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_deviceFailed || (g_loadTried && !g_loadOk);
}
