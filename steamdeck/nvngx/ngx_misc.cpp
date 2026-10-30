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

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
        LOGI("DeckyUpscaler nvngx proxy loaded (DLSS -> Arm NSS, Steam Deck / Proton)");
    return TRUE;
}
