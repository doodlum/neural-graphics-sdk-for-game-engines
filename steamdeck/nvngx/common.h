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
    int   debugView;        // AISS_DEBUG_VIEW: -1 off, 0 = NSS's 4x4 grid of internal views, 1-16 = one view
    int   dumpFrame;        // AISS_DUMP_FRAME: evaluation index whose output is written to aiss_frame_<n>.ppm, -1 off
    float jitterSign[2];    // AISS_JITTER_SIGN=x,y: multipliers on the jitter handed to NSS (convention debugging)
    float mvSign[2];        // AISS_MV_SIGN=x,y: multipliers on the motion vector scale (convention debugging)
    bool  fastMath;         // FFX_ML_RESCALE_FLOAT set at launch: fp32 requantisation in the ML graph (not bit exact)
};

const ProxyConfig& Config();

// NSS shader quality requested at runtime through the control file named by AISS_CONTROL (a Linux path, written by
// the Decky plugin): 0, 1 or 2, or -1 when there is no file or no valid value. Re-read at most twice a second.
int LiveNssQuality();
// Optional second value of the control file: 1 = fast maths (fp32 requantisation, ~2% faster, not bit exact), 0 = exact,
// -1 when absent.
int LiveFastMath();
