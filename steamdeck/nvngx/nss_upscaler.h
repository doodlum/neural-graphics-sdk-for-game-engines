// Vulkan-level NSS upscaler shared by the D3D12 (vkd3d-proton interop) and Vulkan NGX entry points.
// SPDX-License-Identifier: MIT
#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

struct VulkanDevice
{
    VkInstance                instance       = VK_NULL_HANDLE;
    VkPhysicalDevice          physicalDevice = VK_NULL_HANDLE;
    VkDevice                  device         = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr gipa           = nullptr;
    PFN_vkGetDeviceProcAddr   gdpa           = nullptr;
    uint32_t                  mlFeatures     = 0;  // FFX_API_VK_ML_FEATURE_* enabled on this device
};

// One image handed in by the game, with the layout it is in when the upscaler runs.
struct UpscalerImage
{
    VkImage       image  = VK_NULL_HANDLE;
    VkFormat      format = VK_FORMAT_UNDEFINED;
    uint32_t      width  = 0;
    uint32_t      height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
};

struct UpscalerCreateInfo
{
    uint32_t renderWidth   = 0;  // maximum render size
    uint32_t renderHeight  = 0;
    uint32_t displayWidth  = 0;
    uint32_t displayHeight = 0;
    bool     hdr           = false;
    bool     depthInverted = false;
    int      nssQuality    = 2;  // FfxApiNssShaderQualityMode
    bool     fastMath      = false;  // fp32 requantisation in the ML graph: ~2% faster, not bit exact
    bool     autoExposure  = false;  // the game asked DLSS for auto exposure: its colour is not pre-exposed
    bool     highResMv     = false;  // motion vectors at display resolution (DLSS MVLowRes not set)
    int      backend       = 0;      // UpscalerBackend
    int      fsr4Model     = 0;      // Fsr4Model, for kBackendFsr4: the one asked for
    int      fsr4Active    = -1;     // the FSR 4 network that runs (set by the upscaler)
    bool     layerSplit    = false;  // FSR 4 interop on a Vulkan game: the AISS layer splits the game's command buffer
};

enum UpscalerBackend : int
{
    kBackendNss  = 0,
    kBackendXess = 1,
    kBackendFsr4 = 2,       // AMD FSR 4 asked for (amdxcffx64.dll); each front end maps it to one of the two below
    kBackendFsr4Dx12 = 3,  // AMD FSR 4 natively on the game's D3D12 device (amdxcffx64.dll); only the D3D12 front end
    kBackendFsr4Interop = 4,  // AMD FSR 4 via amdxcffx64.dll on an in-process D3D12 device (Vulkan / D3D11 games)
};

// Lets an upscaler split its work over several submissions on the front end's queue: Begin() starts a command buffer,
// Submit() submits it, optionally waiting for and/or signalling a timeline semaphore value. The D3D11 front end submits
// its own command buffers; for Vulkan games the interop upscaler implements it with the AISS layer's command buffer
// split (one game command buffer, cut where Submit() is called).
struct SplitSubmitter
{
    virtual VkCommandBuffer Begin() = 0;
    virtual bool            Submit(VkSemaphore wait, uint64_t waitValue, VkSemaphore signal, uint64_t signalValue) = 0;

protected:
    ~SplitSubmitter() = default;
};

// AMD FSR 4 networks in amdvkffx64.dll. Auto lets the driver pick by GPU (Lite on Van Gogh).
enum Fsr4Model : int
{
    kFsr4Auto  = 0,
    kFsr4Lite  = 1,  // INT8 Lite: AMD's handheld model (Van Gogh, Strix Point)
    kFsr4Int8  = 2,  // INT8: the desktop RDNA 3 model
    kFsr4Rdna2 = 3,  // INT8 for RDNA 2 desktop GPUs (Navi 2x)
    kFsr4Fp8   = 4,  // FP8: RDNA 4
    kFsr4ModelCount
};
const char* Fsr4ModelName(int model);
// -1 when the name is not a model; accepts auto, lite, int8, rdna2 (or navi2), fp8.
int Fsr4ModelFromName(const char* name);

struct UpscalerDispatch
{
    UpscalerImage color, depth, motionVectors, output;
    uint32_t      renderWidth = 0, renderHeight = 0;
    int32_t       depthOffsetX = 0, depthOffsetY = 0;
    float         jitterX = 0, jitterY = 0;      // DLSS convention, render pixels
    float         mvScaleX = 1, mvScaleY = 1;
    float         frameTimeMs = 16.6f;
    float         exposure = 1.0f;
    bool          reset = false;
};

// One upscaling context (NSS, XeSS or FSR 4) behind an NGX handle.
class Fsr4Dx12;  // fsr4_dx12.h (D3D12 front end only)

class Upscaler
{
public:
    virtual ~Upscaler() = default;
    virtual bool Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d) = 0;
    // Non-null only for the D3D12-native FSR 4 carrier: it runs on the game's D3D12 device, not the Vulkan one.
    virtual Fsr4Dx12* AsDx12() { return nullptr; }
    // Split dispatch over several submissions (see SplitSubmitter); the D3D11 front end uses it when supported.
    virtual bool SupportsSplit() const { return false; }
    virtual bool DispatchSplit(SplitSubmitter&, const UpscalerDispatch&) { return false; }

    const UpscalerCreateInfo& Info() const { return m_info; }
    const VulkanDevice&       Device() const { return m_dev; }

protected:
    VulkanDevice       m_dev;
    UpscalerCreateInfo m_info;
};

// Creates the upscaler info.backend asks for; XeSS and FSR 4 fall back to NSS when their DLL is missing or fails.
// FSR 4 here is the D3D12 interop path (Vulkan and D3D11 front ends); the D3D12 front end runs FSR 4 natively itself.
std::unique_ptr<Upscaler> CreateUpscaler(const VulkanDevice& device, const UpscalerCreateInfo& info);
// True once XeSS has failed to load or initialise in this process (it is not retried).
bool XessUnavailable();
// FSR 4 through amdxcffx64.dll on an in-process D3D12 device sharing memory with the game's VkDevice (fsr4_interop.cpp).
std::unique_ptr<Upscaler> CreateFsr4InteropUpscaler(const VulkanDevice& device, const UpscalerCreateInfo& info);
bool Fsr4InteropUnavailable(int model);

class NssUpscaler : public Upscaler
{
public:
    static std::unique_ptr<NssUpscaler> Create(const VulkanDevice& device, const UpscalerCreateInfo& info);
    ~NssUpscaler() override;

    bool Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d) override;

private:
    NssUpscaler() = default;
    bool Init(const VulkanDevice& device, const UpscalerCreateInfo& info);
    bool CreateDepthCopy();
    bool CreateAutoExposure();
    float AutoExposure(VkCommandBuffer cmd, const UpscalerImage& color, uint32_t width, uint32_t height);
    VkImageView CreateView(VkImage image, VkFormat format, VkImageAspectFlags aspect);
    uint32_t    FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags);
    void        RecordDump(VkCommandBuffer cmd, const UpscalerImage& output);
    void        WriteDump();

    struct Fn;
    std::unique_ptr<Fn> m_fn;

    void*              m_context = nullptr;  // ffxContext

    // Images the proxy owns and hands to NSS instead of a game input, each exactly the dispatch's render size: NSS
    // samples its inputs with UVs from 1 / renderSize, so a texture larger than the render size would be read
    // stretched. Depth always goes through one (R32F: NSS reads depth through a colour view); colour and motion
    // vectors only when the game's texture extent differs from the render size. Cached by kind, format and size so
    // dynamic resolution does not reallocate every frame; an entry is destroyed only once it has gone unused for
    // kRetireAfter dispatches (the proxy cannot wait on the game's queue, so that is when no frame in flight can
    // still read it).
    enum OwnedKind : uint32_t { kOwnedDepth, kOwnedColor, kOwnedMotion, kOwnedKinds };
    struct OwnedImage
    {
        VkImage        image   = VK_NULL_HANDLE;
        VkDeviceMemory memory  = VK_NULL_HANDLE;
        VkImageView    view    = VK_NULL_HANDLE;
        uint32_t       kind    = 0, width = 0, height = 0;
        VkFormat       format  = VK_FORMAT_UNDEFINED;
        uint64_t       lastUse = 0;  // m_dispatchCount of the last dispatch that used it
    };
    static constexpr uint64_t kRetireAfter   = 16;
    static constexpr uint32_t kOwnedPerKind  = 4;  // retired entries beyond this many per kind are destroyed
    std::vector<OwnedImage> m_owned;
    bool                    m_loggedCopy[2] = {};  // colour / motion copy state last logged
    VkFormat          CopyFormat(VkFormat game, VkFormat fallback) const;
    const OwnedImage* AcquireOwned(uint32_t kind, VkFormat format, uint32_t width, uint32_t height);
    void              DestroyOwned(const OwnedImage& img);
    void              RecordCopy(VkCommandBuffer cmd, uint32_t setIndex, VkPipeline pipeline, VkImage src, VkFormat srcFormat,
                                 VkImageAspectFlags aspect, VkImageLayout srcLayout, VkImageView dst, uint32_t width,
                                 uint32_t height, int32_t offsetX, int32_t offsetY);
    VkSampler             m_sampler     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout   = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipeLayout  = VK_NULL_HANDLE;
    VkPipeline            m_pipeline    = VK_NULL_HANDLE;  // depth -> R32F
    static constexpr uint32_t kCopyFormats = 7;  // region_copy.comp variants, see kRegionCopies
    VkPipeline            m_copyPipelines[kCopyFormats] = {};
    bool                  m_copyStorage[kCopyFormats]   = {};  // the device supports storage writes to that format
    VkDescriptorPool      m_pool        = VK_NULL_HANDLE;

    // Per-dispatch views of game images, recycled once the ring wraps (frames in flight are far fewer).
    static constexpr uint32_t kRing = 8;
    // kOwnedKinds sets and views per slot: one per copy (depth, colour, motion vectors).
    VkDescriptorSet m_sets[kRing * kOwnedKinds]      = {};
    VkImageView     m_gameViews[kRing * kOwnedKinds] = {};
    uint64_t        m_dispatchCount     = 0;

    // Auto exposure (DLSS AutoExposure flag): log-average luminance per dispatch into a host-visible ring, read back
    // kExposureLatency dispatches later (the proxy cannot wait on the game's queue), smoothed over frames.
    static constexpr uint32_t kExposureLatency = 6;
    VkDescriptorSetLayout m_aeSetLayout   = VK_NULL_HANDLE;
    VkPipelineLayout      m_aePipeLayout  = VK_NULL_HANDLE;
    VkPipeline            m_aePipeline    = VK_NULL_HANDLE;
    VkDescriptorPool      m_aePool        = VK_NULL_HANDLE;
    VkDescriptorSet       m_aeSets[kRing] = {};
    VkImageView           m_aeViews[kRing] = {};
    VkBuffer              m_aeBuffer      = VK_NULL_HANDLE;
    VkDeviceMemory        m_aeMemory      = VK_NULL_HANDLE;
    float*                m_aeResults     = nullptr;  // kRing x {sum log2 lum, count, tag, 0}
    float                 m_exposure      = 0.0f;     // 0 until the first readback

    // AISS_DUMP_INPUTS: raw copies of the dispatch inputs and output, read back a few frames later.
    struct RawDump
    {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        uint32_t       width = 0, height = 0, format = 0, bpp = 0;
        uint64_t       readAt = 0;
        char           name[300] = {};
    };
    std::vector<RawDump> m_rawDumps;
    void RecordRaw(VkCommandBuffer cmd, const UpscalerImage& img, VkImageAspectFlags aspect, const char* name);
    void WriteRawDumps(bool all);

    // AISS_DUMP_FRAME readback of one output frame.
    VkBuffer       m_dumpBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_dumpMemory = VK_NULL_HANDLE;
    UpscalerImage  m_dumpImage;
    uint64_t       m_dumpAt     = 0;
};

// NGX handles map to upscalers; released ones are destroyed once the GPU can no longer use them.
class UpscalerRegistry
{
public:
    unsigned int Add(std::unique_ptr<Upscaler> upscaler);
    Upscaler*    Get(unsigned int id);
    Upscaler*    Replace(unsigned int id, std::unique_ptr<Upscaler> fresh);  // swap in a new context for a handle
    void         Release(unsigned int id);
    void         CollectGarbage(bool force = false);

private:
    struct Entry
    {
        unsigned int                          id;
        std::unique_ptr<Upscaler>             upscaler;
        bool                                  released = false;
        std::chrono::steady_clock::time_point releasedAt;
    };
    std::vector<Entry> m_entries;
    unsigned int       m_nextId = 1;
};

UpscalerRegistry& Registry();

// Maps the game's DLSS quality mode to an NSS shader quality, honouring AISS_NSS_QUALITY.
int NssQualityForDlss(int perfQualityValue);

// Fast maths (FFX_ML_RESCALE_FLOAT) for new upscalers: the control file's value, else the launch environment.
bool FastMathEnabled();
