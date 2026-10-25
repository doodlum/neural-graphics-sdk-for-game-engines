// Minimal declarations of the NGX core ABI that games and Streamline call into.
// Only what the DLSS Super Resolution path needs; names and values follow the public NGX interface.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cstddef>

#define NGX_EXPORT extern "C" __declspec(dllexport)

typedef enum NVSDK_NGX_Result
{
    NVSDK_NGX_Result_Success                  = 0x1,
    NVSDK_NGX_Result_Fail                     = (int)0xBAD00000,
    NVSDK_NGX_Result_FAIL_FeatureNotSupported = NVSDK_NGX_Result_Fail | 1,
    NVSDK_NGX_Result_FAIL_PlatformError       = NVSDK_NGX_Result_Fail | 2,
    NVSDK_NGX_Result_FAIL_FeatureNotFound     = NVSDK_NGX_Result_Fail | 4,
    NVSDK_NGX_Result_FAIL_InvalidParameter    = NVSDK_NGX_Result_Fail | 5,
    NVSDK_NGX_Result_FAIL_NotInitialized      = NVSDK_NGX_Result_Fail | 7,
    NVSDK_NGX_Result_FAIL_UnsupportedFormat   = NVSDK_NGX_Result_Fail | 14,
    NVSDK_NGX_Result_FAIL_UnsupportedParameter = NVSDK_NGX_Result_Fail | 16,
} NVSDK_NGX_Result;

typedef enum NVSDK_NGX_Feature
{
    NVSDK_NGX_Feature_SuperSampling     = 1,
    NVSDK_NGX_Feature_FrameGeneration   = 11,
    NVSDK_NGX_Feature_RayReconstruction = 13,
} NVSDK_NGX_Feature;

typedef enum NVSDK_NGX_PerfQuality_Value
{
    NVSDK_NGX_PerfQuality_Value_MaxPerf,
    NVSDK_NGX_PerfQuality_Value_Balanced,
    NVSDK_NGX_PerfQuality_Value_MaxQuality,
    NVSDK_NGX_PerfQuality_Value_UltraPerformance,
    NVSDK_NGX_PerfQuality_Value_UltraQuality,
    NVSDK_NGX_PerfQuality_Value_DLAA,
} NVSDK_NGX_PerfQuality_Value;

enum
{
    NVSDK_NGX_DLSS_Feature_Flags_IsHDR         = 1 << 0,
    NVSDK_NGX_DLSS_Feature_Flags_MVLowRes      = 1 << 1,
    NVSDK_NGX_DLSS_Feature_Flags_MVJittered    = 1 << 2,
    NVSDK_NGX_DLSS_Feature_Flags_DepthInverted = 1 << 3,
    NVSDK_NGX_DLSS_Feature_Flags_DoSharpening  = 1 << 5,
    NVSDK_NGX_DLSS_Feature_Flags_AutoExposure  = 1 << 6,
};

typedef enum NVSDK_NGX_Feature_Support_Result
{
    NVSDK_NGX_FeatureSupportResult_Supported          = 0,
    NVSDK_NGX_FeatureSupportResult_AdapterUnsupported = 4,
} NVSDK_NGX_Feature_Support_Result;

struct NVSDK_NGX_Handle
{
    unsigned int Id;
};

struct NVSDK_NGX_PathListInfo
{
    wchar_t const* const* Path;
    unsigned int          Length;
};

struct NVSDK_NGX_FeatureCommonInfo
{
    NVSDK_NGX_PathListInfo PathListInfo;
    void*                  InternalData;
    // Logging info follows in newer SDKs; never read here.
};

struct NVSDK_NGX_ProjectIdDescription
{
    const char* ProjectId;
    int         EngineType;
    const char* EngineVersion;
};

struct NVSDK_NGX_Application_Identifier
{
    int IdentifierType;
    union
    {
        NVSDK_NGX_ProjectIdDescription ProjectDesc;
        unsigned long long             ApplicationId;
    } v;
};

struct NVSDK_NGX_FeatureDiscoveryInfo
{
    int                                SDKVersion;
    NVSDK_NGX_Feature                  FeatureID;
    NVSDK_NGX_Application_Identifier   Identifier;
    const wchar_t*                     ApplicationDataPath;
    const NVSDK_NGX_FeatureCommonInfo* FeatureInfo;
};

struct NVSDK_NGX_FeatureRequirement
{
    NVSDK_NGX_Feature_Support_Result FeatureSupported;
    unsigned int                     MinHWArchitecture;
    char                             MinOSVersion[255];
};

// NVSDK_NGX_Parameter is an MSVC C++ abstract class. Its vtable layout is reproduced by hand in params.cpp so the
// proxy works with any compiler; callers only ever hold a pointer to the object.
struct NVSDK_NGX_Parameter;

typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback)(NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_DLSS_GetStatsCallback)(NVSDK_NGX_Parameter*);
typedef void (*PFN_NVSDK_NGX_ProgressCallback)(float, bool&);

// Vulkan resource wrapper passed through NGX parameters on the Vulkan path.
#include <vulkan/vulkan.h>

struct NVSDK_NGX_ImageViewInfo_VK
{
    VkImageView             ImageView;
    VkImage                 Image;
    VkImageSubresourceRange SubresourceRange;
    VkFormat                Format;
    unsigned int            Width;
    unsigned int            Height;
};

struct NVSDK_NGX_BufferInfo_VK
{
    VkBuffer     Buffer;
    unsigned int SizeInBytes;
};

enum NVSDK_NGX_Resource_VK_Type
{
    NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
    NVSDK_NGX_RESOURCE_VK_TYPE_VK_BUFFER,
};

struct NVSDK_NGX_Resource_VK
{
    union
    {
        NVSDK_NGX_ImageViewInfo_VK ImageViewInfo;
        NVSDK_NGX_BufferInfo_VK    BufferInfo;
    } Resource;
    NVSDK_NGX_Resource_VK_Type Type;
    bool                       ReadWrite;
};

// Parameter names used by the DLSS-SR path.
namespace ngxp
{
constexpr const char* Width          = "Width";
constexpr const char* Height         = "Height";
constexpr const char* OutWidth       = "OutWidth";
constexpr const char* OutHeight      = "OutHeight";
constexpr const char* PerfQuality    = "PerfQualityValue";
constexpr const char* CreateFlags    = "DLSS.Feature.Create.Flags";
constexpr const char* Color          = "Color";
constexpr const char* Output         = "Output";
constexpr const char* Depth          = "Depth";
constexpr const char* MotionVectors  = "MotionVectors";
constexpr const char* JitterX        = "Jitter.Offset.X";
constexpr const char* JitterY        = "Jitter.Offset.Y";
constexpr const char* MVScaleX       = "MV.Scale.X";
constexpr const char* MVScaleY       = "MV.Scale.Y";
constexpr const char* Reset          = "Reset";
constexpr const char* FrameTime      = "FrameTimeDeltaInMsec";
constexpr const char* PreExposure    = "DLSS.Pre.Exposure";
constexpr const char* RenderSubrectW = "DLSS.Render.Subrect.Dimensions.Width";
constexpr const char* RenderSubrectH = "DLSS.Render.Subrect.Dimensions.Height";
constexpr const char* ColorBaseX     = "DLSS.Input.Color.Subrect.Base.X";
constexpr const char* ColorBaseY     = "DLSS.Input.Color.Subrect.Base.Y";
constexpr const char* DepthBaseX     = "DLSS.Input.Depth.Subrect.Base.X";
constexpr const char* DepthBaseY     = "DLSS.Input.Depth.Subrect.Base.Y";
constexpr const char* MVBaseX        = "DLSS.Input.MV.Subrect.Base.X";
constexpr const char* MVBaseY        = "DLSS.Input.MV.Subrect.Base.Y";
constexpr const char* OutputBaseX    = "DLSS.Output.Subrect.Base.X";
constexpr const char* OutputBaseY    = "DLSS.Output.Subrect.Base.Y";
} // namespace ngxp
