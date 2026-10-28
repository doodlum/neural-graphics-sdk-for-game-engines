// FSR 4 through AMD's D3D12 DLL (amdxcffx64) for D3D11 games ("D3D12 interop"): the faster DX12 FSR 4 for a game whose
// resources live on a Vulkan device (DXVK).
//
// The proxy creates a D3D12 device in-process (vkd3d-proton under Proton) on the game's adapter (LUID from
// VkPhysicalDeviceIDProperties) with a COMPUTE queue. It shares memory through D3D12 committed 2D textures created with
// D3D12_HEAP_FLAG_SHARED (ID3D12Device::CreateSharedHandle), imported into the game's VkDevice as VkImages with
// VK_KHR_external_memory_win32 (handle type D3D12_RESOURCE, dedicated allocation, the usage vkd3d-proton gives the
// texture so both drivers lay it out alike). vkd3d-proton gives no handle for shared buffers and cannot share heaps,
// so textures are what is shared. Per frame:
//   game queue:   copy colour / depth (via a buffer, as R32F) / motion vectors (render subrect) -> shared textures,
//                 signal timeline value N
//   D3D12 queue:  wait N, FSR 4 (Fsr4Dx12, amdxcffx64) reading the shared inputs and writing the shared output, signal N+1
//   game queue:   wait N+1, copy the shared output -> the game's output image
// The fence is one shared ID3D12Fence imported as a Vulkan timeline semaphore (VK_KHR_external_semaphore_win32,
// D3D12_FENCE handle). All waits are GPU/kernel-side; the CPU never waits per frame. This needs the front end to own
// the submissions (SplitSubmitter): the D3D11 front end submits its own command buffers on DXVK's queue.
//
// Vulkan games: NGX hands the proxy only the game's command buffer, which the game submits later. The proxy records
// copy-in, a split marker (vkCmdFillBuffer on a marker buffer, aiss_split_abi.h) and copy-out into it, and queues the
// D3D12 work (wait N, FSR 4, signal N+1) right away; VK_LAYER_AISS_ngx (native) cuts the game's command buffer at the
// marker. A "capture" vkQueueSubmit2 (signal N, wait N+1) right after the marker goes through winevulkan, which emulates
// D3D12 fences with per-operation host semaphores; the layer keeps those instead of submitting it, and when the game
// submits its command buffer makes the part before signal N and the part after wait N+1 - the same
// submissions as the D3D11 path, all in the same frame. Without the layer (or with AISS_LAYER_SPLIT=0) the
// registration below fails and the game gets NSS.
//
// GPU-side waits inside one command buffer do not work (measured on a Steam Deck, both removed): vkCmdWaitEvents on a
// host-set event with D3D12 textures shared between the devices ran into amdgpu's 5 s gfx lockup timeout ("ring
// gfx_0.0.0 timeout ... ring reset", three times), and with host-memory staging instead (no shared BO) and a 750 ms
// release, the D3D12 work never completed while the game's job waited.
// SPDX-License-Identifier: MIT
#include "nss_upscaler.h"

#include "common.h"
#include "fsr4_dx12.h"
#include "ngx_common.h"
#include "../layer/aiss_split_abi.h"

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
// ---- formats --------------------------------------------------------------------------------------------------------
struct FormatInfo
{
    DXGI_FORMAT dxgi = DXGI_FORMAT_UNKNOWN;
    uint32_t    bpp  = 0;  // bytes per texel of the buffer copy
};

FormatInfo MapFormat(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT: return {DXGI_FORMAT_R16G16B16A16_FLOAT, 8};
    case VK_FORMAT_R32G32B32A32_SFLOAT: return {DXGI_FORMAT_R32G32B32A32_FLOAT, 16};
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return {DXGI_FORMAT_R11G11B10_FLOAT, 4};
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return {DXGI_FORMAT_R10G10B10A2_UNORM, 4};
    case VK_FORMAT_R8G8B8A8_UNORM: return {DXGI_FORMAT_R8G8B8A8_UNORM, 4};
    case VK_FORMAT_R8G8B8A8_SRGB: return {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4};
    case VK_FORMAT_B8G8R8A8_UNORM: return {DXGI_FORMAT_B8G8R8A8_UNORM, 4};
    case VK_FORMAT_B8G8R8A8_SRGB: return {DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 4};
    case VK_FORMAT_R16G16_SFLOAT: return {DXGI_FORMAT_R16G16_FLOAT, 4};
    case VK_FORMAT_R32G32_SFLOAT: return {DXGI_FORMAT_R32G32_FLOAT, 8};
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return {DXGI_FORMAT_R32_FLOAT, 4};  // the depth aspect copies as 32-bit floats
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_D16_UNORM: return {DXGI_FORMAT_R16_UNORM, 2};
    case VK_FORMAT_R16_SFLOAT: return {DXGI_FORMAT_R16_FLOAT, 2};
    default: return {};
    }
}

bool IsDepth(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_X8_D24_UNORM_PACK32 ||
           f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

bool HasStencil(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

constexpr uint32_t kPitchAlign = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;  // 256

uint32_t AlignedPitch(uint32_t width, uint32_t bpp)
{
    return (width * bpp + kPitchAlign - 1) / kPitchAlign * kPitchAlign;
}

// ---- the process-wide D3D12 device ----------------------------------------------------------------------------------
typedef HRESULT(WINAPI* PFN_D3D12CreateDevice_)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
typedef HRESULT(WINAPI* PFN_CreateDXGIFactory1_)(REFIID, void**);

struct D3D12Side
{
    ID3D12Device*       device = nullptr;
    ID3D12CommandQueue* queue  = nullptr;  // COMPUTE
    bool                tried = false, ok = false;
    std::string         why;
};
std::mutex g_d3dMutex;
D3D12Side  g_d3d;

const D3D12Side* GetD3D12(const VulkanDevice& vk)
{
    std::lock_guard<std::mutex> lock(g_d3dMutex);
    if (g_d3d.tried)
        return g_d3d.ok ? &g_d3d : nullptr;
    g_d3d.tried = true;

    // The game's adapter LUID (winevulkan fills deviceLUID), to pick the same GPU for D3D12.
    VkPhysicalDeviceIDProperties idp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2  p2  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
    auto props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vk.gipa(vk.instance, "vkGetPhysicalDeviceProperties2"));
    if (props2)
        props2(vk.physicalDevice, &p2);

    HMODULE d3d12 = LoadLibraryA("d3d12.dll");
    auto    create = d3d12 ? reinterpret_cast<PFN_D3D12CreateDevice_>(GetProcAddress(d3d12, "D3D12CreateDevice")) : nullptr;
    if (!create)
    {
        g_d3d.why = "d3d12.dll / D3D12CreateDevice not available";
        return nullptr;
    }
    IDXGIAdapter1* adapter = nullptr;
    if (idp.deviceLUIDValid)
    {
        HMODULE dxgi = LoadLibraryA("dxgi.dll");
        auto    cf   = dxgi ? reinterpret_cast<PFN_CreateDXGIFactory1_>(GetProcAddress(dxgi, "CreateDXGIFactory1")) : nullptr;
        IDXGIFactory4* factory = nullptr;
        if (cf && SUCCEEDED(cf(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))) && factory)
        {
            LUID luid;
            memcpy(&luid, idp.deviceLUID, sizeof(luid));
            if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&adapter))))
                adapter = nullptr;
            factory->Release();
        }
    }
    const HRESULT hr = create(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&g_d3d.device));
    if (adapter)
        adapter->Release();
    if (FAILED(hr) || !g_d3d.device)
    {
        g_d3d.why = "D3D12CreateDevice failed, hr 0x" + std::to_string(unsigned(hr));
        return nullptr;
    }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type                     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(g_d3d.device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&g_d3d.queue))))
    {
        g_d3d.why = "CreateCommandQueue (compute) failed";
        return nullptr;
    }
    LOGI("FSR 4 interop: D3D12 device created on the game's adapter (LUID %s), compute queue",
         idp.deviceLUIDValid ? "matched" : "unknown: default adapter");
    g_d3d.ok = true;
    return &g_d3d;
}

std::mutex g_failMutex;
bool       g_modelFailed[kFsr4ModelCount] = {};
bool       g_allFailed                    = false;

// ---- one shared texture ---------------------------------------------------------------------------------------------
struct SharedImage
{
    ID3D12Resource* d3d    = nullptr;
    VkImage         image  = VK_NULL_HANDLE;
    VkDeviceMemory  memory = VK_NULL_HANDLE;
    VkFormat        format = VK_FORMAT_UNDEFINED;
    uint32_t        w = 0, h = 0;
};

class Fsr4InteropUpscaler : public Upscaler
{
public:
    ~Fsr4InteropUpscaler() override;
    bool Init(const VulkanDevice& device, const UpscalerCreateInfo& info, std::string& why);
    bool Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d) override;  // Vulkan front end: layer split
    bool SupportsSplit() const override { return m_semaphore != VK_NULL_HANDLE && !m_info.layerSplit; }
    bool DispatchSplit(SplitSubmitter& s, const UpscalerDispatch& d) override;  // D3D11 front end: timeline semaphores

private:
    // Vulkan functions on the game's device.
#define IFN(name) PFN_##name name = nullptr;
    IFN(vkCreateBuffer) IFN(vkDestroyBuffer) IFN(vkGetBufferMemoryRequirements) IFN(vkAllocateMemory) IFN(vkFreeMemory)
    IFN(vkBindBufferMemory) IFN(vkGetMemoryWin32HandlePropertiesKHR) IFN(vkCmdCopyImageToBuffer)
    IFN(vkCmdCopyBufferToImage) IFN(vkCmdPipelineBarrier) IFN(vkCreateEvent) IFN(vkDestroyEvent) IFN(vkCmdSetEvent)
    IFN(vkCmdWaitEvents) IFN(vkGetEventStatus) IFN(vkSetEvent) IFN(vkResetEvent) IFN(vkCreateSemaphore)
    IFN(vkDestroySemaphore) IFN(vkImportSemaphoreWin32HandleKHR) IFN(vkCreateQueryPool) IFN(vkDestroyQueryPool)
    IFN(vkCmdResetQueryPool) IFN(vkCmdWriteTimestamp) IFN(vkGetQueryPoolResults)
    IFN(vkGetPhysicalDeviceProperties) IFN(vkGetPhysicalDeviceMemoryProperties) IFN(vkCreateImage) IFN(vkDestroyImage)
    IFN(vkGetImageMemoryRequirements) IFN(vkBindImageMemory) IFN(vkCmdCopyImage) IFN(vkCmdFillBuffer)
    IFN(vkWaitSemaphores) IFN(vkQueueSubmit2) IFN(vkGetDeviceQueue)
#undef IFN

    bool  RegisterWithLayer(std::string& why);
    bool  CreateShared(SharedImage& si, VkFormat vkFormat, DXGI_FORMAT dxgi, uint32_t w, uint32_t h, bool uav, std::string& why);
    void  FreeShared(SharedImage& si);
    bool  EnsureResources(const UpscalerDispatch& d);
    void  RecordCopyIn(VkCommandBuffer cmd, const UpscalerDispatch& d);
    void  RecordCopyOut(VkCommandBuffer cmd, const UpscalerDispatch& d);
    bool  RecordD3D12(const UpscalerDispatch& d, uint32_t slot);
    void  Timestamp(VkCommandBuffer cmd, uint32_t index);
    void  ReportTiming();

    const D3D12Side* m_d3d = nullptr;
    Fsr4Dx12*        m_fsr4 = nullptr;
    VkExternalMemoryHandleTypeFlagBits m_memHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;

    // Shared textures FSR 4 reads (colour, depth as R32F, motion vectors) and writes (output), at the game textures'
    // sizes; a local buffer for the depth-aspect copy.
    enum { kColor, kDepth, kMotion, kOutput, kCount };
    SharedImage     m_tex[kCount];
    VkFormat        m_fmt[kCount] = {};  // the game images' formats
    uint32_t        m_w[kCount] = {}, m_h[kCount] = {};
    VkBuffer        m_depthBuf = VK_NULL_HANDLE;
    VkDeviceMemory  m_depthMem = VK_NULL_HANDLE;
    bool            m_ready = false, m_firstUse = true;

    // Synchronisation.
    ID3D12Fence*   m_fence     = nullptr;  // shared (D3D11 path imports it as a timeline semaphore)
    VkSemaphore    m_semaphore = VK_NULL_HANDLE;
    uint64_t       m_value     = 0;  // last fence value used
    static constexpr uint32_t kSlots = 4;
    ID3D12CommandAllocator*    m_alloc[kSlots] = {};
    ID3D12GraphicsCommandList* m_list[kSlots]  = {};
    uint64_t                   m_slotDone[kSlots] = {};  // fence value that marks the slot's last use as finished
    uint64_t m_frame = 0;

    // Vulkan front end (m_info.layerSplit): the layer's split marker buffer and our token.
    VkBuffer m_marker = VK_NULL_HANDLE;
    uint32_t m_token  = UINT32_MAX;
    VkQueue  m_captureQueue = VK_NULL_HANDLE;  // any queue of the game's device: the capture is never submitted

    HANDLE                  m_fenceEvent = nullptr;
    void                    WaitFence(uint64_t value)  // CPU wait for our D3D12 work (resizes, slot reuse, teardown)
    {
        const ULONGLONG t0 = GetTickCount64();
        while (value && m_fence->GetCompletedValue() < value && GetTickCount64() - t0 < 5000)
        {
            ResetEvent(m_fenceEvent);
            m_fence->SetEventOnCompletion(value, m_fenceEvent);
            WaitForSingleObject(m_fenceEvent, 500);
        }
    }

    // Timing (AISS_GPU_TIMING): 4 timestamps per frame on the game's queue.
    VkQueryPool m_qp = VK_NULL_HANDLE;
    float       m_nsPerTick = 1.0f;
    uint64_t    m_timed = 0;
    double      m_sumIn = 0, m_sumMid = 0, m_sumOut = 0, m_sumTotal = 0;
    uint32_t    m_samples = 0;
    uint64_t    m_reportedAt = 0;
};

bool Fsr4InteropUpscaler::Init(const VulkanDevice& device, const UpscalerCreateInfo& info, std::string& why)
{
    m_dev  = device;
    m_info = info;
#define IFN(name) name = reinterpret_cast<PFN_##name>(device.gdpa(device.device, #name));
    IFN(vkCreateBuffer) IFN(vkDestroyBuffer) IFN(vkGetBufferMemoryRequirements) IFN(vkAllocateMemory) IFN(vkFreeMemory)
    IFN(vkBindBufferMemory) IFN(vkGetMemoryWin32HandlePropertiesKHR) IFN(vkCmdCopyImageToBuffer)
    IFN(vkCmdCopyBufferToImage) IFN(vkCmdPipelineBarrier) IFN(vkCreateEvent) IFN(vkDestroyEvent) IFN(vkCmdSetEvent)
    IFN(vkCmdWaitEvents) IFN(vkGetEventStatus) IFN(vkSetEvent) IFN(vkResetEvent) IFN(vkCreateSemaphore)
    IFN(vkDestroySemaphore) IFN(vkImportSemaphoreWin32HandleKHR) IFN(vkCreateQueryPool) IFN(vkDestroyQueryPool)
    IFN(vkCmdResetQueryPool) IFN(vkCmdWriteTimestamp) IFN(vkGetQueryPoolResults) IFN(vkCreateImage) IFN(vkDestroyImage)
    IFN(vkGetImageMemoryRequirements) IFN(vkBindImageMemory) IFN(vkCmdCopyImage) IFN(vkCmdFillBuffer)
    IFN(vkWaitSemaphores) IFN(vkQueueSubmit2) IFN(vkGetDeviceQueue)
#undef IFN
    if (!vkQueueSubmit2)
        vkQueueSubmit2 = reinterpret_cast<PFN_vkQueueSubmit2>(device.gdpa(device.device, "vkQueueSubmit2KHR"));
    if (!vkWaitSemaphores)
        vkWaitSemaphores = reinterpret_cast<PFN_vkWaitSemaphores>(device.gdpa(device.device, "vkWaitSemaphoresKHR"));
    vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        device.gipa(device.instance, "vkGetPhysicalDeviceMemoryProperties"));
    vkGetPhysicalDeviceProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(device.gipa(device.instance, "vkGetPhysicalDeviceProperties"));
    if (!vkGetMemoryWin32HandlePropertiesKHR)
    {
        why = "the game's VkDevice lacks VK_KHR_external_memory_win32";
        return false;
    }
    m_d3d = GetD3D12(device);
    if (!m_d3d)
    {
        why = g_d3d.why;
        return false;
    }
    m_fsr4 = Fsr4Dx12::Create(m_d3d->device, info.renderWidth, info.renderHeight, info.displayWidth, info.displayHeight,
                              info.hdr, info.depthInverted, info.highResMv, info.autoExposure, info.fsr4Model);
    if (!m_fsr4)
    {
        why = "amdxcffx64 FSR 4 context could not be created on the interop D3D12 device";
        return false;
    }
    m_info.fsr4Active = m_fsr4->Model();

    if (FAILED(m_d3d->device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence),
                                          reinterpret_cast<void**>(&m_fence))))
    {
        why = "CreateFence (shared) failed";
        return false;
    }
    for (uint32_t i = 0; i < kSlots; ++i)
    {
        if (FAILED(m_d3d->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator),
                                                         reinterpret_cast<void**>(&m_alloc[i]))) ||
            FAILED(m_d3d->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_alloc[i], nullptr,
                                                    __uuidof(ID3D12GraphicsCommandList),
                                                    reinterpret_cast<void**>(&m_list[i]))))
        {
            why = "CreateCommandAllocator/List (compute) failed";
            return false;
        }
        m_list[i]->Close();
    }

    // The shared fence as a Vulkan timeline semaphore (used by the D3D11 front end, which submits its own work).
    HANDLE fenceHandle = nullptr;
    if (vkImportSemaphoreWin32HandleKHR &&
        SUCCEEDED(m_d3d->device->CreateSharedHandle(m_fence, nullptr, GENERIC_ALL, nullptr, &fenceHandle)) && fenceHandle)
    {
        VkSemaphoreTypeCreateInfo tci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        tci.semaphoreType             = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci     = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &tci};
        VkSemaphore           sem     = VK_NULL_HANDLE;
        if (vkCreateSemaphore(device.device, &sci, nullptr, &sem) == VK_SUCCESS)
        {
            VkImportSemaphoreWin32HandleInfoKHR imp = {VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
            imp.semaphore                           = sem;
            imp.handleType                          = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
            imp.handle                              = fenceHandle;
            const VkResult vr                       = vkImportSemaphoreWin32HandleKHR(device.device, &imp);
            if (vr == VK_SUCCESS)
                m_semaphore = sem;
            else
            {
                LOGI("FSR 4 interop: vkImportSemaphoreWin32HandleKHR(D3D12_FENCE) failed (%d)", vr);
                vkDestroySemaphore(device.device, sem, nullptr);
            }
        }
        CloseHandle(fenceHandle);
    }
    if (!m_semaphore)
    {
        why = "the shared D3D12 fence cannot be imported as a Vulkan timeline semaphore";
        return false;
    }
    m_fenceEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);  // manual reset; reset before each wait
    if (m_info.layerSplit && !RegisterWithLayer(why))
        return false;

    if (GpuTimingFrames() > 0)
    {
        VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType             = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount            = kSlots * 4;
        if (vkCreateQueryPool(device.device, &qci, nullptr, &m_qp) != VK_SUCCESS)
            m_qp = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties pp = {};
        vkGetPhysicalDeviceProperties(device.physicalDevice, &pp);
        m_nsPerTick = pp.limits.timestampPeriod;
    }
    LOGI("FSR 4 interop: network %s on the D3D12 device, sync: shared ID3D12Fence = Vulkan timeline semaphore%s",
         Fsr4ModelName(m_info.fsr4Active), m_info.layerSplit ? ", game command buffer split by the AISS layer" : "");
    return true;
}

// Vulkan front end: tell VK_LAYER_AISS_ngx our semaphore and create its marker buffer (aiss_split_abi.h). winevulkan
// forwards no private entry points, so this is a vkWaitSemaphores the layer recognises by its timeout; it answers
// VK_TIMEOUT, which a driver never does for value 0.
bool Fsr4InteropUpscaler::RegisterWithLayer(std::string& why)
{
    static std::atomic<uint32_t> nextToken{0};
    if (!vkWaitSemaphores || !vkCmdFillBuffer || !vkQueueSubmit2 || !vkGetDeviceQueue)
    {
        why = "the game's VkDevice lacks vkWaitSemaphores / vkQueueSubmit2 (timeline semaphores, synchronization2)";
        return false;
    }
    for (uint32_t family = 0; family < 16 && !m_captureQueue; ++family)
        vkGetDeviceQueue(m_dev.device, family, 0, &m_captureQueue);
    if (!m_captureQueue)
    {
        why = "no queue found on the game's VkDevice";
        return false;
    }
    m_token = nextToken++ % AISS_SPLIT_TOKEN_COUNT;
    const uint64_t      zero = 0;
    VkSemaphoreWaitInfo wi   = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount        = 1;
    wi.pSemaphores           = &m_semaphore;
    wi.pValues               = &zero;
    const VkResult r         = vkWaitSemaphores(m_dev.device, &wi, AISS_SPLIT_REGISTER | m_token);
    if (r != VK_TIMEOUT)
    {
        m_token = UINT32_MAX;
        why     = "the AISS Vulkan layer (VK_LAYER_AISS_ngx with command buffer split; AISS_LAYER=1) is not active on the "
                  "game's device (registration returned " + std::to_string(r) + ")";
        return false;
    }
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size               = AISS_SPLIT_BUFFER_SIZE;
    bci.usage              = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(m_dev.device, &bci, nullptr, &m_marker) != VK_SUCCESS)
    {
        m_marker = VK_NULL_HANDLE;
        why      = "cannot create the command buffer split marker buffer";
        return false;
    }
    LOGI("FSR 4 interop: AISS layer command buffer split registered (token %u)", m_token);
    return true;
}

Fsr4InteropUpscaler::~Fsr4InteropUpscaler()
{
    // D3D12 work queued behind a fence wait the relay will now never signal would block the shared D3D12 queue for
    // every later upscaler: release all our pending waits first (that work then runs on whatever the inputs hold).
    // Vulkan front end: the game's segments signal our values; give them a moment first (a CPU signal ahead of a
    // pending GPU signal would make that one go backwards).
    if (m_info.layerSplit && m_fence && m_fenceEvent && m_value && m_fence->GetCompletedValue() < m_value)
    {
        const ULONGLONG t0 = GetTickCount64();
        while (m_fence->GetCompletedValue() < m_value && GetTickCount64() - t0 < 1000)
            Sleep(5);
    }
    if (m_fence && m_value && m_fence->GetCompletedValue() < m_value)
        m_fence->Signal(m_value);
    // The registry destroys upscalers once the GPU is done with them; wait for our D3D12 work too.
    if (m_fence && m_fenceEvent)
        WaitFence(m_value);
    delete m_fsr4;
    for (uint32_t i = 0; i < kCount; ++i)
        FreeShared(m_tex[i]);
    if (m_depthBuf) vkDestroyBuffer(m_dev.device, m_depthBuf, nullptr);
    if (m_depthMem) vkFreeMemory(m_dev.device, m_depthMem, nullptr);
    for (uint32_t i = 0; i < kSlots; ++i)
    {
        if (m_list[i]) m_list[i]->Release();
        if (m_alloc[i]) m_alloc[i]->Release();
    }
    if (m_token != UINT32_MAX && vkWaitSemaphores)
    {
        const uint64_t      zero = 0;
        VkSemaphoreWaitInfo wi   = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount        = 1;
        wi.pSemaphores           = &m_semaphore;
        wi.pValues               = &zero;
        vkWaitSemaphores(m_dev.device, &wi, AISS_SPLIT_UNREGISTER | m_token);
    }
    if (m_marker) vkDestroyBuffer(m_dev.device, m_marker, nullptr);
    if (m_semaphore) vkDestroySemaphore(m_dev.device, m_semaphore, nullptr);
    if (m_qp) vkDestroyQueryPool(m_dev.device, m_qp, nullptr);
    if (m_fence) m_fence->Release();
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
}

// A 2D texture created on the D3D12 device with D3D12_HEAP_FLAG_SHARED and imported into the game's VkDevice as a
// VkImage on the same memory. vkd3d-proton shares textures (it gives no handle for shared buffers and does not share
// heaps). The Vulkan image is created with the usage vkd3d-proton gives such a texture (transfer src/dst + sampled,
// + storage for a UAV) so both drivers compute the same layout; RADV does not compress shareable images.
bool Fsr4InteropUpscaler::CreateShared(SharedImage& si, VkFormat vkFormat, DXGI_FORMAT dxgi, uint32_t w, uint32_t h, bool uav,
                                       std::string& why)
{
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type                  = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td   = {};
    td.Dimension             = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width                 = w;
    td.Height                = h;
    td.DepthOrArraySize = td.MipLevels = 1;
    td.Format                          = dxgi;
    td.SampleDesc.Count                = 1;
    td.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    HRESULT hr = m_d3d->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &td, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        __uuidof(ID3D12Resource), reinterpret_cast<void**>(&si.d3d));
    if (FAILED(hr))
    {
        why = "CreateCommittedResource(shared 2D texture, DXGI " + std::to_string(dxgi) + ") failed, hr 0x" +
              std::to_string(unsigned(hr));
        return false;
    }
    HANDLE sh = nullptr;
    hr        = m_d3d->device->CreateSharedHandle(si.d3d, nullptr, GENERIC_ALL, nullptr, &sh);
    if (FAILED(hr) || !sh)
    {
        why = "CreateSharedHandle(texture) failed, hr 0x" + std::to_string(unsigned(hr));
        return false;
    }
    VkResult vr = VK_ERROR_UNKNOWN;
    for (VkExternalMemoryHandleTypeFlagBits type :
         {VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT})
    {
        VkMemoryWin32HandlePropertiesKHR mp = {VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
        vr = vkGetMemoryWin32HandlePropertiesKHR(m_dev.device, type, sh, &mp);
        if (vr != VK_SUCCESS || !mp.memoryTypeBits)
            continue;
        VkExternalMemoryImageCreateInfo ext = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        ext.handleTypes                     = type;
        VkImageCreateInfo ci                = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext};
        ci.imageType                        = VK_IMAGE_TYPE_2D;
        ci.format                           = vkFormat;
        ci.extent                           = {w, h, 1};
        ci.mipLevels = ci.arrayLayers = 1;
        ci.samples                    = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling                     = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                   (uav ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(m_dev.device, &ci, nullptr, &si.image) != VK_SUCCESS)
            continue;
        VkMemoryRequirements req = {};
        vkGetImageMemoryRequirements(m_dev.device, si.image, &req);
        const uint32_t bits = req.memoryTypeBits & mp.memoryTypeBits;
        uint32_t       mt   = UINT32_MAX;
        for (uint32_t i = 0; i < 32 && mt == UINT32_MAX; ++i)
            if (bits & (1u << i))
                mt = i;
        VkMemoryDedicatedAllocateInfo    ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        ded.image                            = si.image;
        VkImportMemoryWin32HandleInfoKHR imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &ded};
        imp.handleType                       = type;
        imp.handle                           = sh;
        VkMemoryAllocateInfo mai             = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp};
        mai.allocationSize                   = req.size;
        mai.memoryTypeIndex                  = mt;
        vr = mt == UINT32_MAX ? VK_ERROR_FEATURE_NOT_PRESENT : vkAllocateMemory(m_dev.device, &mai, nullptr, &si.memory);
        if (vr == VK_SUCCESS && vkBindImageMemory(m_dev.device, si.image, si.memory, 0) == VK_SUCCESS)
        {
            static bool logged = false;
            if (!logged)
                LOGI("FSR 4 interop: shared textures import into the game's VkDevice as %s",
                     type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT ? "OPAQUE_WIN32" : "D3D12_RESOURCE");
            logged    = true;
            si.format = vkFormat;
            si.w = w, si.h = h;
            CloseHandle(sh);
            return true;
        }
        if (si.memory) vkFreeMemory(m_dev.device, si.memory, nullptr);
        vkDestroyImage(m_dev.device, si.image, nullptr);
        si.memory = VK_NULL_HANDLE;
        si.image  = VK_NULL_HANDLE;
    }
    CloseHandle(sh);
    why = "importing the shared D3D12 texture into the game's VkDevice failed (vk result " + std::to_string(vr) + ")";
    return false;
}

void Fsr4InteropUpscaler::FreeShared(SharedImage& si)
{
    if (si.image) vkDestroyImage(m_dev.device, si.image, nullptr);
    if (si.memory) vkFreeMemory(m_dev.device, si.memory, nullptr);
    if (si.d3d) si.d3d->Release();
    si = SharedImage();
}

bool Fsr4InteropUpscaler::EnsureResources(const UpscalerDispatch& d)
{
    const UpscalerImage* img[kCount] = {&d.color, &d.depth, &d.motionVectors, &d.output};
    bool same = m_ready;
    for (uint32_t i = 0; i < kCount && same; ++i)
        same = m_fmt[i] == img[i]->format && m_w[i] == img[i]->width && m_h[i] == img[i]->height;
    if (same)
        return true;
    if (m_ready)
    {
        // A resize or format change (rare): drain our D3D12 work, then rebuild.
        WaitFence(m_value);
        for (uint32_t i = 0; i < kCount; ++i)
            FreeShared(m_tex[i]);
        if (m_depthBuf) vkDestroyBuffer(m_dev.device, m_depthBuf, nullptr);
        if (m_depthMem) vkFreeMemory(m_dev.device, m_depthMem, nullptr);
        m_depthBuf = VK_NULL_HANDLE;
        m_depthMem = VK_NULL_HANDLE;
        m_ready    = false;
    }
    for (uint32_t i = 0; i < kCount; ++i)
    {
        const VkFormat   f  = img[i]->format;
        const FormatInfo fi = MapFormat(f);
        const bool depthOk  = f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
        if (!fi.bpp || (i == kDepth && IsDepth(f) && !depthOk))
        {
            LOGI("FSR 4 interop: %s format %d not supported", i == kDepth ? "depth" : "image", f);
            return false;
        }
        m_fmt[i] = f;
        m_w[i]   = img[i]->width;
        m_h[i]   = img[i]->height;
        // Depth is shared as R32_FLOAT (copied through a buffer: a depth aspect cannot be copied to a colour image).
        const VkFormat shared = i == kDepth && IsDepth(f) ? VK_FORMAT_R32_SFLOAT : f;
        std::string    why;
        if (!CreateShared(m_tex[i], shared, fi.dxgi, m_w[i], m_h[i], i == kOutput, why))
        {
            LOGI("FSR 4 interop: %s", why.c_str());
            return false;
        }
    }
    if (IsDepth(m_fmt[kDepth]))
    {
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size               = VkDeviceSize(m_w[kDepth]) * m_h[kDepth] * 4;
        bci.usage              = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(m_dev.device, &bci, nullptr, &m_depthBuf) != VK_SUCCESS)
            return false;
        VkMemoryRequirements req = {};
        vkGetBufferMemoryRequirements(m_dev.device, m_depthBuf, &req);
        VkPhysicalDeviceMemoryProperties mp = {};
        vkGetPhysicalDeviceMemoryProperties(m_dev.physicalDevice, &mp);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount && mt == UINT32_MAX; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                mt = i;
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize       = req.size;
        mai.memoryTypeIndex      = mt;
        if (mt == UINT32_MAX || vkAllocateMemory(m_dev.device, &mai, nullptr, &m_depthMem) != VK_SUCCESS ||
            vkBindBufferMemory(m_dev.device, m_depthBuf, m_depthMem, 0) != VK_SUCCESS)
            return false;
    }
    m_firstUse = true;
    // Warm-up: FSR 4 builds its pipelines on the first dispatch, which takes seconds under vkd3d-proton. Run one
    // dispatch now and wait for it on the CPU (once, like XeSS's pipeline build), so no game frame's GPU waits on it.
    {
        const ULONGLONG t0  = GetTickCount64();
        UpscalerDispatch warm = d;
        warm.reset            = true;
        RecordD3D12(warm, 0);
        ID3D12CommandList* lists[] = {m_list[0]};
        m_d3d->queue->ExecuteCommandLists(1, lists);
        m_d3d->queue->Signal(m_fence, ++m_value);
        m_slotDone[0] = m_value;
        WaitFence(m_value);
        LOGI("FSR 4 interop: warm-up dispatch (pipeline builds) took %llu ms", (unsigned long long)(GetTickCount64() - t0));
    }
    LOGI("FSR 4 interop: shared textures colour %ux%u (fmt %d), depth %ux%u, motion %ux%u, output %ux%u (fmt %d)",
         m_w[kColor], m_h[kColor], m_fmt[kColor], m_w[kDepth], m_h[kDepth], m_w[kMotion], m_h[kMotion], m_w[kOutput],
         m_h[kOutput], m_fmt[kOutput]);
    m_ready = true;
    return true;
}

void Fsr4InteropUpscaler::Timestamp(VkCommandBuffer cmd, uint32_t index)
{
    if (m_qp)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_qp, uint32_t(m_frame % kSlots) * 4 + index);
}

static VkImageMemoryBarrier ImageBarrier(VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                                         VkAccessFlags src, VkAccessFlags dst)
{
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask        = src;
    b.dstAccessMask        = dst;
    b.oldLayout            = from;
    b.newLayout            = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image                = image;
    b.subresourceRange     = {aspect, 0, 1, 0, 1};
    return b;
}

static VkImageAspectFlags AspectOf(VkFormat f)
{
    return IsDepth(f) ? VK_IMAGE_ASPECT_DEPTH_BIT | (HasStencil(f) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0) : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Game images -> shared textures (render subrect). The shared textures stay in GENERAL on the Vulkan side.
void Fsr4InteropUpscaler::RecordCopyIn(VkCommandBuffer cmd, const UpscalerDispatch& d)
{
    const UpscalerImage* img[3] = {&d.color, &d.depth, &d.motionVectors};
    if (m_qp)
        vkCmdResetQueryPool(cmd, m_qp, uint32_t(m_frame % kSlots) * 4, 4);
    Timestamp(cmd, 0);
    VkImageMemoryBarrier pre[6];
    uint32_t             n = 0;
    for (uint32_t i = 0; i < 3; ++i)
        pre[n++] = ImageBarrier(img[i]->image, AspectOf(img[i]->format), img[i]->layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    const VkImageLayout sharedFrom = m_firstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    for (uint32_t i = 0; i < 3; ++i)
        pre[n++] = ImageBarrier(m_tex[i].image, VK_IMAGE_ASPECT_COLOR_BIT, sharedFrom, VK_IMAGE_LAYOUT_GENERAL,
                                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, n,
                         pre);
    const uint32_t rw = std::min(d.renderWidth, d.color.width), rh = std::min(d.renderHeight, d.color.height);
    for (uint32_t i = 0; i < 3; ++i)
    {
        const bool     full = i == kMotion && m_info.highResMv;
        const VkExtent3D e  = {full ? m_w[i] : std::min(rw, m_w[i]), full ? m_h[i] : std::min(rh, m_h[i]), 1};
        if (i == kDepth && IsDepth(m_fmt[i]))
        {
            // depth aspect -> buffer -> R32F shared texture
            VkBufferImageCopy r = {};
            r.imageSubresource  = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
            r.imageExtent       = e;
            vkCmdCopyImageToBuffer(cmd, img[i]->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_depthBuf, 1, &r);
            VkBufferMemoryBarrier bb = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            bb.srcAccessMask         = VK_ACCESS_TRANSFER_WRITE_BIT;
            bb.dstAccessMask         = VK_ACCESS_TRANSFER_READ_BIT;
            bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.buffer                = m_depthBuf;
            bb.size                  = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bb, 0,
                                 nullptr);
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            vkCmdCopyBufferToImage(cmd, m_depthBuf, m_tex[i].image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            continue;
        }
        VkImageCopy c    = {};
        c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent         = e;
        vkCmdCopyImage(cmd, img[i]->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_tex[i].image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    VkImageMemoryBarrier post[6];
    n = 0;
    for (uint32_t i = 0; i < 3; ++i)
        post[n++] = ImageBarrier(img[i]->image, AspectOf(img[i]->format), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img[i]->layout,
                                 VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    for (uint32_t i = 0; i < 3; ++i)  // make the copies available to the other device
        post[n++] = ImageBarrier(m_tex[i].image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, n,
                         post);
    Timestamp(cmd, 1);
}

// Shared output texture -> the game's output image.
void Fsr4InteropUpscaler::RecordCopyOut(VkCommandBuffer cmd, const UpscalerDispatch& d)
{
    Timestamp(cmd, 2);
    VkImageMemoryBarrier pre[2] = {
        ImageBarrier(d.output.image, VK_IMAGE_ASPECT_COLOR_BIT, d.output.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT),
        ImageBarrier(m_tex[kOutput].image, VK_IMAGE_ASPECT_COLOR_BIT, m_firstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
    };
    // On first use the output is not yet defined on the Vulkan side; GENERAL keeps the D3D12 side's contents.
    if (m_firstUse)
        pre[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, pre);
    VkImageCopy c    = {};
    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.extent         = {m_w[kOutput], m_h[kOutput], 1};
    vkCmdCopyImage(cmd, m_tex[kOutput].image, VK_IMAGE_LAYOUT_GENERAL, d.output.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    VkImageMemoryBarrier post = ImageBarrier(d.output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                             d.output.layout, VK_ACCESS_TRANSFER_WRITE_BIT,
                                             VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &post);
    Timestamp(cmd, 3);
    m_firstUse = false;
}

// This frame's D3D12 work on the shared textures: COMMON -> FSR 4 states, FSR 4 (amdxcffx64), back to COMMON.
bool Fsr4InteropUpscaler::RecordD3D12(const UpscalerDispatch& d, uint32_t slot)
{
    WaitFence(m_slotDone[slot]);  // rare: the slot's previous use still runs
    ID3D12GraphicsCommandList* list = m_list[slot];
    m_alloc[slot]->Reset();
    list->Reset(m_alloc[slot], nullptr);
    auto barriers = [&](bool toFsr) {
        D3D12_RESOURCE_BARRIER b[kCount] = {};
        for (uint32_t i = 0; i < kCount; ++i)
        {
            const D3D12_RESOURCE_STATES fsr = i == kOutput ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                           : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            b[i].Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b[i].Transition.pResource   = m_tex[i].d3d;
            b[i].Transition.StateBefore = toFsr ? D3D12_RESOURCE_STATE_COMMON : fsr;
            b[i].Transition.StateAfter  = toFsr ? fsr : D3D12_RESOURCE_STATE_COMMON;
            b[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        list->ResourceBarrier(kCount, b);
    };
    barriers(true);
    const uint32_t   rw = std::min(d.renderWidth, d.color.width), rh = std::min(d.renderHeight, d.color.height);
    Fsr4Dx12Dispatch fd;
    fd.color = m_tex[kColor].d3d, fd.depth = m_tex[kDepth].d3d, fd.motion = m_tex[kMotion].d3d, fd.output = m_tex[kOutput].d3d;
    fd.renderWidth = rw, fd.renderHeight = rh;
    fd.outputWidth = m_w[kOutput], fd.outputHeight = m_h[kOutput];
    fd.jitterX = d.jitterX, fd.jitterY = d.jitterY;
    fd.mvScaleX = d.mvScaleX, fd.mvScaleY = d.mvScaleY;
    fd.frameTimeMs = d.frameTimeMs;
    fd.exposure    = d.exposure;
    fd.reset       = d.reset;
    const bool ok  = m_fsr4->Dispatch(list, fd);
    barriers(false);
    list->Close();
    return ok;
}

bool Fsr4InteropUpscaler::DispatchSplit(SplitSubmitter& s, const UpscalerDispatch& d)
{
    if (!EnsureResources(d))
    {
        // e.g. a depth format that cannot be shared: give up on interop for the process; the next evaluate's live
        // settings rebuild the handle on NSS.
        std::lock_guard<std::mutex> lock(g_failMutex);
        if (!g_allFailed)
            LOGI("FSR 4 interop: cannot share this game's images; falling back to NSS");
        g_allFailed = true;
        return false;
    }
    const uint32_t slot = uint32_t(m_frame % kSlots);
    const uint64_t in = m_value + 1, out = m_value + 2;
    m_value           = out;

    VkCommandBuffer a = s.Begin();
    RecordCopyIn(a, d);
    if (!s.Submit(VK_NULL_HANDLE, 0, m_semaphore, in))
        return false;

    const bool ok = RecordD3D12(d, slot);
    m_d3d->queue->Wait(m_fence, in);
    ID3D12CommandList* lists[] = {m_list[slot]};
    m_d3d->queue->ExecuteCommandLists(1, lists);
    m_d3d->queue->Signal(m_fence, out);
    m_slotDone[slot] = out;

    VkCommandBuffer b = s.Begin();
    RecordCopyOut(b, d);
    if (!m_info.layerSplit)  // the Vulkan front end dumps after Dispatch itself
        NgxDumpFrame(this, b, d.output);
    if (!s.Submit(m_semaphore, out, VK_NULL_HANDLE, 0))
        return false;
    ++m_frame;
    ReportTiming();
    return ok;
}

// Vulkan front end: everything goes into the game's command buffer; the "submissions" are split markers the AISS layer
// turns into real ones when the game submits it.
bool Fsr4InteropUpscaler::Dispatch(VkCommandBuffer cmd, const UpscalerDispatch& d)
{
    if (!m_info.layerSplit || !m_marker)
        return false;
    struct LayerSplit : SplitSubmitter
    {
        Fsr4InteropUpscaler* self;
        VkCommandBuffer      cmd;
        uint64_t             signalled = 0;
        VkCommandBuffer      Begin() override { return cmd; }
        bool                 Submit(VkSemaphore wait, uint64_t wv, VkSemaphore signal, uint64_t sv) override
        {
            if (signal)
            {
                // The layer cuts here: the part before signals sv, the part after waits for sv + 1 (D3D12's signal).
                signalled = sv;
                self->vkCmdFillBuffer(cmd, self->m_marker, sv, sv + 1, AISS_SPLIT_FILL | self->m_token);
                // Capture: winevulkan turns these D3D12-fence operations into host semaphore operations, which the
                // layer puts on the cut instead of submitting this.
                const uint64_t        stage = AISS_SPLIT_STAGE | (uint64_t(self->m_token) << 32);
                VkSemaphoreSubmitInfo w = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                w.semaphore             = signal;
                w.value                 = sv + 1;
                w.stageMask             = stage;
                VkSemaphoreSubmitInfo g = w;
                g.value                 = sv;
                VkSubmitInfo2 si        = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
                si.waitSemaphoreInfoCount   = 1;
                si.pWaitSemaphoreInfos      = &w;
                si.signalSemaphoreInfoCount = 1;
                si.pSignalSemaphoreInfos    = &g;
                const VkResult r            = self->vkQueueSubmit2(self->m_captureQueue, 1, &si, VK_NULL_HANDLE);
                if (r != VK_SUCCESS)
                    LOGI("FSR 4 interop: split capture failed (%d)", r);
                return r == VK_SUCCESS;
            }
            return !wait || wv == signalled + 1;  // already part of the marker
        }
    } split;
    split.self = this;
    split.cmd  = cmd;
    // Evaluate marker: the layer ends a render pass the game left open (copies and the cut need to be outside one).
    vkCmdFillBuffer(cmd, m_marker, 0, 0, AISS_SPLIT_FILL | m_token);
    return DispatchSplit(split, d);
}

void Fsr4InteropUpscaler::ReportTiming()
{
    if (!m_qp || m_frame < kSlots)
        return;
    const uint32_t slot = uint32_t(m_frame % kSlots);  // written kSlots frames ago: retired
    uint64_t       t[8] = {};
    if (vkGetQueryPoolResults(m_dev.device, m_qp, slot * 4, 4, sizeof(t), t, 2 * sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS ||
        !t[1] || !t[3] || !t[5] || !t[7])
        return;
    const double us = m_nsPerTick / 1000.0;
    m_sumIn += double(t[2] - t[0]) * us;
    m_sumMid += double(t[4] - t[2]) * us;
    m_sumOut += double(t[6] - t[4]) * us;
    m_sumTotal += double(t[6] - t[0]) * us;
    if (++m_samples >= uint32_t(GpuTimingFrames() > 0 ? GpuTimingFrames() : 120))
    {
        const uint64_t now = GetTickCount64();
        const double   fps = m_reportedAt ? m_samples * 1000.0 / double(now - m_reportedAt) : 0.0;
        m_reportedAt       = now;
        LOGI("FSR 4 interop GPU time (%s, game queue): copy-in %.1f us, wait for D3D12 (FSR 4 + sync) %.1f us, "
             "copy-out %.1f us, total %.1f us over %u evaluations, %.1f evaluations/s",
             Fsr4ModelName(m_info.fsr4Active), m_sumIn / m_samples, m_sumMid / m_samples, m_sumOut / m_samples,
             m_sumTotal / m_samples, m_samples, fps);
        m_sumIn = m_sumMid = m_sumOut = m_sumTotal = 0;
        m_samples                                  = 0;
    }
}
}  // namespace

bool Fsr4InteropUnavailable(int model)
{
    std::lock_guard<std::mutex> lock(g_failMutex);
    return g_allFailed || (model >= 0 && model < kFsr4ModelCount && g_modelFailed[model]);
}

std::unique_ptr<Upscaler> CreateFsr4InteropUpscaler(const VulkanDevice& device, const UpscalerCreateInfo& info)
{
    const int model = info.fsr4Model >= 0 && info.fsr4Model < kFsr4ModelCount ? info.fsr4Model : kFsr4Auto;
    if (Fsr4InteropUnavailable(model))
        return nullptr;
    std::unique_ptr<Fsr4InteropUpscaler> up(new Fsr4InteropUpscaler());
    std::string                          why;
    UpscalerCreateInfo                   fixed = info;
    fixed.fsr4Model                            = model;
    if (up->Init(device, fixed, why))
        return up;
    LOGI("FSR 4 interop (%s) unavailable: %s", Fsr4ModelName(model), why.c_str());
    std::lock_guard<std::mutex> lock(g_failMutex);
    if (why.find("model") == std::string::npos && why.find("context") == std::string::npos)
        g_allFailed = true;  // device-level failure: no model will work
    else
        g_modelFailed[model] = true;
    return nullptr;
}
