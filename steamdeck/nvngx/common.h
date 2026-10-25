// Logging and environment configuration for the NGX proxy.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdarg>
#include <cstdint>

// AISS_LOG=0 silences the log, 1 (default) logs lifecycle events, 2 also logs every evaluation, 3 traces every NGX
// call and every parameter the caller asked for but was not set, 4 also every parameter Set/Get.
int  LogLevel();
void Log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

#define LOGI(...) Log(1, __VA_ARGS__)
#define LOGV(...) Log(2, __VA_ARGS__)

// Settings read once from the environment (Steam launch options).
struct ProxyConfig
{
    float qualityRatio[6];  // render scale divisor per NVSDK_NGX_PerfQuality_Value
    int   nssQuality;       // -1 = follow the game's DLSS mode, otherwise FfxApiNssShaderQualityMode
    float cameraNear;
    float cameraFar;
    float cameraFovY;       // radians
    float exposure;         // 0 = use the game's pre-exposure when given, else 1
    bool  mlFeaturesForced; // AISS_LAYER=1: the Vulkan layer enabled int8/dot-product device features
};

const ProxyConfig& Config();
