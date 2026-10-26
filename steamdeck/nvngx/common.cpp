// SPDX-License-Identifier: MIT
#include "common.h"

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

int LogLevel()
{
    static const int level = EnvInt("AISS_LOG", 1);
    return level;
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
        c.nssQuality       = EnvInt("AISS_NSS_QUALITY", -1);
        c.cameraNear       = EnvFloat("AISS_CAMERA_NEAR", 0.1f);
        c.cameraFar        = EnvFloat("AISS_CAMERA_FAR", 100000.0f);
        c.cameraFovY       = EnvFloat("AISS_CAMERA_FOV", 60.0f) * 3.14159265f / 180.0f;
        c.exposure         = EnvFloat("AISS_EXPOSURE", 0.0f);
        c.mlFeaturesForced = EnvInt("AISS_LAYER", 0) != 0;
        c.debugView        = EnvInt("AISS_DEBUG_VIEW", -1);
        c.dumpFrame        = EnvInt("AISS_DUMP_FRAME", -1);
        return c;
    }();
    return config;
}

int LiveNssQuality()
{
    static std::mutex mutex;
    static ULONGLONG  lastCheck = 0;
    static int        value     = -1;
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
    if (path.empty())
        return -1;
    std::lock_guard<std::mutex> lock(mutex);
    const ULONGLONG now = GetTickCount64();
    if (lastCheck && now - lastCheck < 500)
        return value;
    lastCheck = now;
    int   read = -1;
    FILE* f    = fopen(path.c_str(), "r");
    if (f)
    {
        if (fscanf(f, "%d", &read) != 1 || read < 0 || read > 2)
            read = -1;
        fclose(f);
    }
    value = read;
    return value;
}
