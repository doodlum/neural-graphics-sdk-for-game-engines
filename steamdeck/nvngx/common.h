// Logging and environment configuration for the NGX proxy.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdarg>
#include <cstdint>
#include <string>

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
    float exposure;         // AISS_EXPOSURE: > 0 forces NSS's exposure; 0 = auto exposure when the game asks DLSS for it, else 1
    bool  mlFeaturesForced; // AISS_LAYER=1: the Vulkan layer enabled int8/dot-product device features
    int   debugView;        // AISS_DEBUG_VIEW: -1 off, 0 = NSS's 4x4 grid of internal views, 1-16 = one view
    int   dumpFrame;        // AISS_DUMP_FRAME: evaluation index whose output is written to aiss_frame_<n>.ppm, -1 off
    int   dumpInputsFrom;   // AISS_DUMP_INPUTS=first,count: raw colour/depth/MV/output per dispatch (aiss_in_*.raw), -1 off
    int   dumpInputsCount;
    float jitterSign[2];    // AISS_JITTER_SIGN=x,y: multipliers on the jitter handed to NSS (convention debugging)
    float mvSign[2];        // AISS_MV_SIGN=x,y: multipliers on the motion vector scale (convention debugging)
    bool  fastMath;         // FFX_ML_RESCALE_FLOAT set at launch: fp32 requantisation in the ML graph (not bit exact)
    bool  xess;             // AISS_UPSCALER=xess: Intel XeSS (libxess.dll next to the proxy) instead of NSS
    int   xessQuality;      // AISS_XESS_QUALITY: xess_quality_settings_t (100-105), 0 = from the render scale
    int   fsr4Model;        // AISS_UPSCALER=fsr4: AMD FSR 4 (amdxcffx64.dll in the aiss folder next to the proxy), Fsr4Model from
                            // AISS_FSR4_MODEL (auto, lite, int8, rdna2, fp8); -1 when FSR 4 is not asked for
    int   gpuTiming;        // AISS_GPU_TIMING=N: log the GPU time of the upscale, averaged over N evaluations
};

const ProxyConfig& Config();

// Upscaler requested at runtime through the control file named by AISS_CONTROL (a Linux path, written by the Decky
// plugin). Its first value is 0, 1 or 2 for NSS at that shader quality, 3 for XeSS, or 4 + n for FSR 4 model n - 1
// (4 = Lite, 5 = INT8, 6 = INT8 RDNA 2, 7 = FP8, 8 = AMD's automatic choice). Re-read at most twice a second.
// LiveNssQuality: the NSS quality, -1 when there is no file, no valid value or another upscaler is selected.
int LiveNssQuality();
// LiveUpscaler: the control file's first value (0-8), -1 when it selects nothing.
int LiveUpscaler();
// Control file values for the FSR 4 models.
int LiveCodeForFsr4(int model);
int Fsr4ModelForLiveCode(int code);  // -1 when the code is not an FSR 4 one
// Optional second value of the control file: 1 = fast maths (fp32 requantisation, ~2% faster, not bit exact), 0 = exact,
// -1 when absent.
int LiveFastMath();
// Optional third value: a capture request counter. The plugin's "Capture frames" button increments it; each new value
// asks the proxy to save the next few frames' inputs. -1 when absent.
int LiveCaptureRequest();
// Further control-file keys (key=value lines the plugin writes), each -1 when absent so the launch option / env wins.
int LiveWatermark();    // 0 off, 1 on (AMD's FSR 4 banner)
int LiveGpuTiming();    // 0 off, 1 on (log the GPU time of each upscale)
int LiveXessQuality();  // xess_quality_settings_t, 0 = from the render scale
// The number of evaluations to average GPU timing over: the live gpu_timing toggle (120 when on), else AISS_GPU_TIMING.
int GpuTimingFrames();
// The control file as a Windows path (Z:\...), empty when AISS_CONTROL is not set.
const std::string& ControlFilePath();
