// API-independent NGX core exports.
// SPDX-License-Identifier: MIT
#include "ngx_common.h"

#include "common.h"

#include <windows.h>

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_UpdateFeature(const NVSDK_NGX_Application_Identifier*, NVSDK_NGX_Feature)
{
    Log(3, "%s", "NVSDK_NGX_UpdateFeature");
    return NVSDK_NGX_Result_Success;
}

NGX_EXPORT unsigned int NVSDK_NGX_GetSnippetVersion()
{
    Log(3, "%s", "NVSDK_NGX_GetSnippetVersion");
    return 0x30500;  // reports DLSS 3.5-era snippet semantics
}

// D3D11 has no Vulkan interop path worth taking on the Deck; report the platform as unsupported so games fall back.
NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init(unsigned long long, const wchar_t*, void*, const void*, int)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init");
    LOGI("D3D11 is not supported");
    return NVSDK_NGX_Result_FAIL_PlatformError;
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init_Ext(unsigned long long, const wchar_t*, void*, int, const void*)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init_Ext");
    return NVSDK_NGX_D3D11_Init(0, nullptr, nullptr, nullptr, 0);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init_ProjectID(const char*, int, const char*, const wchar_t*, void*, int, const void*)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init_ProjectID");
    return NVSDK_NGX_D3D11_Init(0, nullptr, nullptr, nullptr, 0);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_Init_with_ProjectID(const char*, int, const char*, const wchar_t*, void*, const void*, int)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_Init_with_ProjectID");
    return NVSDK_NGX_D3D11_Init(0, nullptr, nullptr, nullptr, 0);
}

NGX_EXPORT NVSDK_NGX_Result NVSDK_NGX_D3D11_GetFeatureRequirements(void*, const NVSDK_NGX_FeatureDiscoveryInfo*,
                                                                   NVSDK_NGX_FeatureRequirement* out)
{
    Log(3, "%s", "NVSDK_NGX_D3D11_GetFeatureRequirements");
    if (out)
        out->FeatureSupported = NVSDK_NGX_FeatureSupportResult_AdapterUnsupported;
    return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
        LOGI("aiss nvngx proxy loaded (DLSS -> Arm NSS, Steam Deck / Proton)");
    return TRUE;
}
