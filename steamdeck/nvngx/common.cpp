// SPDX-License-Identifier: MIT
#include "common.h"

#include "nss_upscaler.h"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

static float EnvFloat(const char* name, float def)
{
    const char* v = getenv(name);
    return (v && *v) ? static_cast<float>(atof(v)) : def;
}

static int EnvInt(const char* name, int def)
{
    const char* v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}

static int LiveLogLevelRaw();  // the control file's log value (-1 when absent), defined with the other live getters

int LogLevel()
{
    static const int envLevel = EnvInt("AISS_LOG", 1);
    const int        live     = LiveLogLevelRaw();
    return live >= 0 ? live : envLevel;
}

void Log(int level, const char* fmt, ...)
{
    if (level > LogLevel())
        return;

    static std::mutex mutex;
    static FILE*      file = nullptr;
    std::lock_guard<std::mutex> lock(mutex);

    if (!file)
    {
        // Next to the game executable, so it ends up in the game folder rather than the prefix.
        char path[MAX_PATH] = {};
        DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
        char* slash = strrchr(path, '\\');
        if (len && slash)
            strcpy(slash + 1, "aiss_nvngx.log");
        else
            strcpy(path, "aiss_nvngx.log");
        file = fopen(path, "w");
        if (!file)
            return;
    }

    char line[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);

    fprintf(file, "[aiss] %s\n", line);
    fflush(file);
}

const ProxyConfig& Config()
{
    static const ProxyConfig config = [] {
        ProxyConfig c = {};
        // DLSS's standard ratios; AISS_RATIO overrides every mode with one value (e.g. 2.0 for 1280x800 -> 640x400).
        const float standard[6] = {2.0f, 1.724f, 1.5f, 3.0f, 1.3f, 1.0f};
        const float forced      = EnvFloat("AISS_RATIO", 0.0f);
        for (int i = 0; i < 6; ++i)
            c.qualityRatio[i] = forced >= 1.0f ? forced : standard[i];
        c.ratioForced      = forced >= 1.0f;
        c.equalCost        = EnvInt("AISS_EQUAL_COST", 0) != 0;
        c.dilateMv         = EnvInt("AISS_FSR_DILATE_MV", 0) != 0;
        c.nssQuality       = EnvInt("AISS_NSS_QUALITY", -1);
        c.xess             = false;
        if (const char* v = getenv("AISS_UPSCALER"))
            c.xess = !strcmp(v, "xess") || !strcmp(v, "XeSS") || !strcmp(v, "XESS") || !strcmp(v, "3");
        c.fsr3             = false;
        if (const char* v = getenv("AISS_UPSCALER"))
            c.fsr3 = !_stricmp(v, "fsr3") || !_stricmp(v, "fsr3.1") || !strcmp(v, "9");
        c.xessQuality      = EnvInt("AISS_XESS_QUALITY", 0);
        c.fsr4Model        = -1;
        if (const char* v = getenv("AISS_UPSCALER"))
        {
            // fsr4 (model from AISS_FSR4_MODEL), or fsr4-<model> / the control file's codes 4-8
            if (!_strnicmp(v, "fsr4", 4))
            {
                const char* m = v[4] == '-' || v[4] == '_' || v[4] == ':' ? v + 5 : getenv("AISS_FSR4_MODEL");
                c.fsr4Model   = Fsr4ModelFromName(m ? m : "auto");
                if (c.fsr4Model < 0)
                    c.fsr4Model = kFsr4Auto;
            }
            else if (v[0] >= '4' && v[0] <= '8' && !v[1])
                c.fsr4Model = Fsr4ModelForLiveCode(v[0] - '0');
        }
        c.gpuTiming        = EnvInt("AISS_GPU_TIMING", 0);
        c.cameraNear       = EnvFloat("AISS_CAMERA_NEAR", 0.1f);
        c.cameraFar        = EnvFloat("AISS_CAMERA_FAR", 100000.0f);
        c.cameraFovY       = EnvFloat("AISS_CAMERA_FOV", 60.0f) * 3.14159265f / 180.0f;
        c.exposure         = EnvFloat("AISS_EXPOSURE", 0.0f);
        c.mlFeaturesForced = EnvInt("AISS_LAYER", 0) != 0;
        c.debugView        = EnvInt("AISS_DEBUG_VIEW", -1);
        c.dumpFrame        = EnvInt("AISS_DUMP_FRAME", -1);
        c.dumpInputsFrom   = -1;
        c.dumpInputsCount  = 0;
        if (const char* v = getenv("AISS_DUMP_INPUTS"))
            sscanf(v, "%d,%d", &c.dumpInputsFrom, &c.dumpInputsCount);
        c.fastMath         = getenv("FFX_ML_RESCALE_FLOAT") != nullptr;
        auto pair = [](const char* name, float* out) {
            out[0] = out[1] = 1.0f;
            if (const char* v = getenv(name))
                sscanf(v, "%f,%f", &out[0], &out[1]);
        };
        pair("AISS_JITTER_SIGN", c.jitterSign);
        pair("AISS_MV_SIGN", c.mvSign);
        return c;
    }();
    return config;
}

struct LiveControl
{
    int nssQuality  = -1;  // control-file upscaler value 0-9 (0-2 NSS, 3 XeSS, 4-8 FSR 4, 9 FSR 3.1)
    int fastMath    = -1;
    int capture     = -1;
    int watermark   = -1;  // 0 off, 1 on
    int logLevel    = -1;  // 0-4
    int gpuTiming   = -1;  // 0 off, 1 on
    int xessQuality = -1;  // xess_quality_settings_t, 0 = from render scale
    int equalCost   = -1;  // 0 off, 1 on
    int dilateMv    = -1;  // 0 off, 1 on
};

const std::string& ControlFilePath()
{
    static const std::string path = [] {
        const char* unix = getenv("AISS_CONTROL");
        if (!unix || unix[0] != '/')
            return std::string();
        std::string dos = std::string("Z:") + unix;  // Proton maps Z: to the Linux root
        for (char& c : dos)
            if (c == '/')
                c = '\\';
        return dos;
    }();
    return path;
}

static LiveControl ReadLiveControl()
{
    static std::mutex  mutex;
    static ULONGLONG   lastCheck = 0;
    static LiveControl value;
    const std::string& path = ControlFilePath();
    if (path.empty())
        return value;
    std::lock_guard<std::mutex> lock(mutex);
    const ULONGLONG now = GetTickCount64();
    if (lastCheck && now - lastCheck < 500)
        return value;
    lastCheck = now;
    LiveControl read;
    FILE*       f = fopen(path.c_str(), "r");
    if (f)
    {
        // Two formats, both accepted: the legacy first line "<upscaler> [fast_math] [capture]" (three ints), and
        // key=value lines (one per line). The plugin writes key=value; older plugins and manual files still work.
        char line[256];
        bool firstLine = true;
        auto setInt = [](const char* s, int lo, int hi, int& out) {
            int v;
            if (sscanf(s, "%d", &v) == 1 && v >= lo && v <= hi)
                out = v;
        };
        while (fgets(line, sizeof(line), f))
        {
            char* eq = strchr(line, '=');
            if (eq)
            {
                *eq           = 0;
                const char* k = line;
                const char* v = eq + 1;
                if (!strcmp(k, "upscaler")) setInt(v, 0, 9, read.nssQuality);
                else if (!strcmp(k, "fast_math")) setInt(v, 0, 1, read.fastMath);
                else if (!strcmp(k, "capture")) setInt(v, 0, 1 << 30, read.capture);
                else if (!strcmp(k, "fsr4_path")) {}  // older plugins wrote it; there is only one FSR 4 path now
                else if (!strcmp(k, "watermark")) setInt(v, 0, 1, read.watermark);
                else if (!strcmp(k, "log")) setInt(v, 0, 4, read.logLevel);
                else if (!strcmp(k, "gpu_timing")) setInt(v, 0, 1, read.gpuTiming);
                else if (!strcmp(k, "xess_quality")) setInt(v, 0, 1000, read.xessQuality);
                else if (!strcmp(k, "equal_cost")) setInt(v, 0, 1, read.equalCost);
                else if (!strcmp(k, "dilate_mv")) setInt(v, 0, 1, read.dilateMv);
                firstLine = false;
                continue;
            }
            if (firstLine)  // legacy three-int line
            {
                int q = -1, fast = -1, capture = -1;
                const int n = sscanf(line, "%d %d %d", &q, &fast, &capture);
                if (n >= 1 && q >= 0 && q <= 9) read.nssQuality = q;
                if (n >= 2 && (fast == 0 || fast == 1)) read.fastMath = fast;
                if (n >= 3 && capture >= 0) read.capture = capture;
            }
            firstLine = false;
        }
        fclose(f);
    }
    value = read;
    return value;
}

int LiveNssQuality()
{
    const int q = ReadLiveControl().nssQuality;
    return q <= 2 ? q : -1;
}

int LiveUpscaler()
{
    return ReadLiveControl().nssQuality;
}

int LiveCodeForFsr4(int model)
{
    return model == kFsr4Auto ? 8 : 3 + model;
}

int Fsr4ModelForLiveCode(int code)
{
    return code == 8 ? kFsr4Auto : code >= 4 && code <= 7 ? code - 3 : -1;
}

int LiveFastMath()
{
    return ReadLiveControl().fastMath;
}

int LiveCaptureRequest()
{
    return ReadLiveControl().capture;
}

int LiveWatermark()
{
    return ReadLiveControl().watermark;
}

int LiveGpuTiming()
{
    return ReadLiveControl().gpuTiming;
}

int LiveXessQuality()
{
    return ReadLiveControl().xessQuality;
}

int LiveEqualCost()
{
    return ReadLiveControl().equalCost;
}

int LiveDilateMv()
{
    return ReadLiveControl().dilateMv;
}

bool DilateMvEnabled()
{
    const int live = LiveDilateMv();
    return live >= 0 ? live != 0 : Config().dilateMv;
}

static int LiveLogLevelRaw()
{
    return ReadLiveControl().logLevel;
}

int GpuTimingFrames()
{
    const int live = LiveGpuTiming();
    if (live >= 0)
        return live ? (Config().gpuTiming > 0 ? Config().gpuTiming : 120) : 0;
    return Config().gpuTiming;
}
