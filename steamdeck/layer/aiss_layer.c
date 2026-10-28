// VK_LAYER_AISS_ngx: native (Linux) implicit Vulkan layer used by the NSS NGX proxy under Proton.
//
// 1. Advertises VK_NVX_binary_import, VK_NVX_image_view_handle and VK_NV_low_latency(2) with stub entry points when the
//    driver lacks them. Vulkan DLSS integrations (Streamline, NGX) require them at device creation even though the NSS
//    proxy never uses them; the layer strips them again before calling the driver. Reflex sleep becomes a no-op that
//    signals the requested semaphore value, so frame pacing never blocks.
// 2. Enables the device features NSS's ML kernels run fastest with (int8/fp16 arithmetic, 8/16-bit storage, integer dot
//    products, subgroup size control, the Vulkan memory model) on every device, as far as the GPU supports them.
//    A Vulkan device's enabled features cannot be queried later, so this is the only way to guarantee them.
// 3. Enables what Intel XeSS (the proxy's alternative upscaler) needs on the same device: VK_EXT_mutable_descriptor_type
//    with mutableDescriptorType, shaderStorageImageWriteWithoutFormat, scalarBlockLayout, plus shaderInt8 and
//    shaderIntegerDotProduct from (2). This is what xessVKGetRequiredDeviceExtensions/Features (XeSS 2.0.2) return on
//    a Steam Deck under Proton (no instance extensions, Vulkan 1.3); the layer is native and cannot call the Windows
//    library, so the list is fixed here.
//
// Enabled with AISS_LAYER=1 (see VkLayer_aiss_ngx.json); applies to the game's devices and to DXVK/vkd3d-proton's.
// SPDX-License-Identifier: MIT
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <pthread.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))

// ---------------------------------------------------------------------------------------------------------------------
// Dispatch bookkeeping, keyed by the loader's dispatch table pointer.
// ---------------------------------------------------------------------------------------------------------------------
typedef struct InstanceData
{
    void*                                    key;
    VkInstance                               instance;
    uint32_t                                 apiVersion;
    bool                                     emulate;  // advertise the NVIDIA-only extensions to this instance
    PFN_vkGetInstanceProcAddr                gipa;
    PFN_GetPhysicalDeviceProcAddr            gpdpa;  // next layer's physical-device proc lookup (may be NULL)
    PFN_vkDestroyInstance                    destroyInstance;
    PFN_vkEnumerateDeviceExtensionProperties enumerateDeviceExtensions;
    PFN_vkGetPhysicalDeviceFeatures2         getFeatures2;
    PFN_vkGetPhysicalDeviceProperties        getProperties;
    PFN_vkEnumeratePhysicalDevices           enumeratePhysicalDevices;
    PFN_vkEnumeratePhysicalDeviceGroups      enumeratePhysicalDeviceGroups;
    struct InstanceData*                     next;
} InstanceData;

// Physical device handle -> owning instance. Dispatch keys cannot tell instances apart reliably here: a process under
// Proton has several (game, DXVK, vkd3d-proton, helpers) and the per-instance settings must not leak between them.
typedef struct PhysicalDeviceData
{
    VkPhysicalDevice           handle;
    InstanceData*              instance;
    struct PhysicalDeviceData* next;
} PhysicalDeviceData;

typedef struct DeviceData
{
    void*                   key;
    PFN_vkGetDeviceProcAddr gdpa;
    PFN_vkDestroyDevice     destroyDevice;
    PFN_vkSignalSemaphore   signalSemaphore;
    bool                    fakeNvx;         // NVX entry points are served by the layer
    bool                    fakeLowLatency;  // VK_NV_low_latency2 entry points are served by the layer
    struct DeviceData*      next;
} DeviceData;

static pthread_mutex_t g_lock      = PTHREAD_MUTEX_INITIALIZER;
static InstanceData*   g_instances = NULL;
static DeviceData*     g_devices   = NULL;
static PhysicalDeviceData* g_physicalDevices = NULL;

static void* DispatchKey(const void* handle) { return *(void* const*)handle; }

static InstanceData* FindInstance(const void* handle)
{
    void* key = DispatchKey(handle);
    pthread_mutex_lock(&g_lock);
    InstanceData* it = g_instances;
    while (it && it->key != key)
        it = it->next;
    pthread_mutex_unlock(&g_lock);
    return it;
}

static DeviceData* FindDevice(const void* handle)
{
    void* key = DispatchKey(handle);
    pthread_mutex_lock(&g_lock);
    DeviceData* it = g_devices;
    while (it && it->key != key)
        it = it->next;
    pthread_mutex_unlock(&g_lock);
    return it;
}

static void RememberPhysicalDevice(VkPhysicalDevice pd, InstanceData* inst)
{
    pthread_mutex_lock(&g_lock);
    PhysicalDeviceData* it = g_physicalDevices;
    while (it && it->handle != pd)
        it = it->next;
    if (!it)
    {
        it                = calloc(1, sizeof(*it));
        it->handle        = pd;
        it->next          = g_physicalDevices;
        g_physicalDevices = it;
    }
    it->instance = inst;
    pthread_mutex_unlock(&g_lock);
}

static InstanceData* FindInstanceForPhysicalDevice(VkPhysicalDevice pd)
{
    pthread_mutex_lock(&g_lock);
    PhysicalDeviceData* it = g_physicalDevices;
    while (it && it->handle != pd)
        it = it->next;
    InstanceData* inst = it ? it->instance : NULL;
    pthread_mutex_unlock(&g_lock);
    return inst ? inst : FindInstance(pd);
}

// ---------------------------------------------------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------------------------------------------------
static VkResult VKAPI_CALL Layer_CreateInstance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* alloc, VkInstance* out)
{
    VkLayerInstanceCreateInfo* chain = (VkLayerInstanceCreateInfo*)info->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo*)chain->pNext;
    if (!chain)
        return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr     gipa  = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_GetPhysicalDeviceProcAddr gpdpa = chain->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr;
    chain->u.pLayerInfo                 = chain->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create         = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult             res       = create(info, alloc, out);
    if (res != VK_SUCCESS)
        return res;

    InstanceData* d              = calloc(1, sizeof(*d));
    d->key                       = DispatchKey(*out);
    d->instance                  = *out;
    d->apiVersion                = info->pApplicationInfo && info->pApplicationInfo->apiVersion ? info->pApplicationInfo->apiVersion
                                                                                                : VK_API_VERSION_1_0;
    d->gipa                      = gipa;
    d->gpdpa                     = gpdpa;
    // DXVK and vkd3d-proton switch on NVIDIA-only interop paths (CUDA, Reflex) when they see the NVX/NV extensions;
    // the emulation is only for games that use Vulkan directly. ML features are still enabled for them.
    const char* engine = info->pApplicationInfo ? info->pApplicationInfo->pEngineName : NULL;
    d->emulate         = !(engine && (!strncmp(engine, "vkd3d", 5) || !strncmp(engine, "DXVK", 4)));
    const char* emulateEnv = getenv("AISS_LAYER_EMULATE");
    if (emulateEnv && !strcmp(emulateEnv, "0"))
        d->emulate = false;
    if (getenv("AISS_LAYER_LOG"))
        fprintf(stderr, "aiss-layer: instance engine '%s' api 0x%x emulate %d\n", engine ? engine : "(null)", d->apiVersion, d->emulate);
    d->destroyInstance           = (PFN_vkDestroyInstance)gipa(*out, "vkDestroyInstance");
    d->enumerateDeviceExtensions = (PFN_vkEnumerateDeviceExtensionProperties)gipa(*out, "vkEnumerateDeviceExtensionProperties");
    d->getFeatures2              = (PFN_vkGetPhysicalDeviceFeatures2)gipa(*out, "vkGetPhysicalDeviceFeatures2");
    if (!d->getFeatures2)
        d->getFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(*out, "vkGetPhysicalDeviceFeatures2KHR");
    d->getProperties = (PFN_vkGetPhysicalDeviceProperties)gipa(*out, "vkGetPhysicalDeviceProperties");
    d->enumeratePhysicalDevices      = (PFN_vkEnumeratePhysicalDevices)gipa(*out, "vkEnumeratePhysicalDevices");
    d->enumeratePhysicalDeviceGroups = (PFN_vkEnumeratePhysicalDeviceGroups)gipa(*out, "vkEnumeratePhysicalDeviceGroups");
    pthread_mutex_lock(&g_lock);
    d->next     = g_instances;
    g_instances = d;
    pthread_mutex_unlock(&g_lock);
    return VK_SUCCESS;
}

static void VKAPI_CALL Layer_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* alloc)
{
    InstanceData* d = FindInstance(instance);
    if (!d)
        return;
    d->destroyInstance(instance, alloc);
    pthread_mutex_lock(&g_lock);
    for (InstanceData** it = &g_instances; *it; it = &(*it)->next)
        if (*it == d)
        {
            *it = d->next;
            break;
        }
    pthread_mutex_unlock(&g_lock);
    free(d);
}

static VkResult VKAPI_CALL Layer_EnumeratePhysicalDevices(VkInstance instance, uint32_t* count, VkPhysicalDevice* out)
{
    InstanceData* inst = FindInstance(instance);
    VkResult      res  = inst->enumeratePhysicalDevices(instance, count, out);
    if (out && (res == VK_SUCCESS || res == VK_INCOMPLETE))
        for (uint32_t i = 0; i < *count; ++i)
            RememberPhysicalDevice(out[i], inst);
    return res;
}

static VkResult VKAPI_CALL Layer_EnumeratePhysicalDeviceGroups(VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* out)
{
    InstanceData* inst = FindInstance(instance);
    VkResult      res  = inst->enumeratePhysicalDeviceGroups(instance, count, out);
    if (out && (res == VK_SUCCESS || res == VK_INCOMPLETE))
        for (uint32_t i = 0; i < *count; ++i)
            for (uint32_t j = 0; j < out[i].physicalDeviceCount; ++j)
                RememberPhysicalDevice(out[i].physicalDevices[j], inst);
    return res;
}

// ---------------------------------------------------------------------------------------------------------------------
// NVX extension emulation
// ---------------------------------------------------------------------------------------------------------------------
static const VkExtensionProperties kFakeExtensions[] = {
    {VK_NVX_BINARY_IMPORT_EXTENSION_NAME, 1},
    {VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME, 2},
    {VK_NV_LOW_LATENCY_2_EXTENSION_NAME, 2},
    {VK_NV_LOW_LATENCY_EXTENSION_NAME, 1},  // struct-only (VkQueryLowLatencySupportNV), nothing to stub
};
#define FAKE_EXTENSION_COUNT (sizeof(kFakeExtensions) / sizeof(kFakeExtensions[0]))

static bool IsFakeExtension(const char* name)
{
    for (uint32_t f = 0; f < FAKE_EXTENSION_COUNT; ++f)
        if (!strcmp(name, kFakeExtensions[f].extensionName))
            return true;
    return false;
}

// The loader reports the extensions this layer's manifest declares as if the driver had them, so the emulated ones
// are never taken from here: RADV has none of them (this layer targets the Steam Deck).
static bool DriverHasExtension(InstanceData* inst, VkPhysicalDevice pd, const char* name)
{
    if (IsFakeExtension(name))
        return false;
    uint32_t count = 0;
    inst->enumerateDeviceExtensions(pd, NULL, &count, NULL);
    VkExtensionProperties* props = calloc(count ? count : 1, sizeof(*props));
    inst->enumerateDeviceExtensions(pd, NULL, &count, props);
    bool found = false;
    for (uint32_t i = 0; i < count && !found; ++i)
        found = !strcmp(props[i].extensionName, name);
    free(props);
    return found;
}

static VkResult VKAPI_CALL Layer_EnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char* layer, uint32_t* count,
                                                                    VkExtensionProperties* out)
{
    InstanceData* inst = FindInstanceForPhysicalDevice(pd);
    if (layer && !strcmp(layer, "VK_LAYER_AISS_ngx"))
    {
        *count = 0;
        return VK_SUCCESS;
    }
    if (layer)
        return inst->enumerateDeviceExtensions(pd, layer, count, out);

    uint32_t driverCount = 0;
    VkResult res         = inst->enumerateDeviceExtensions(pd, NULL, &driverCount, NULL);
    if (res != VK_SUCCESS)
        return res;
    VkExtensionProperties* all = calloc(driverCount + FAKE_EXTENSION_COUNT, sizeof(*all));
    inst->enumerateDeviceExtensions(pd, NULL, &driverCount, all);
    // Drop the emulated extensions from the loader's list (it adds them for every instance because the manifest
    // declares them), then add them back only for instances that should see them.
    uint32_t    total = 0;
    const char* hide  = getenv("AISS_LAYER_HIDE");  // comma-separated extension names hidden from the app (debugging)
    for (uint32_t i = 0; i < driverCount; ++i)
        if (!IsFakeExtension(all[i].extensionName) && !(hide && strstr(hide, all[i].extensionName)))
            all[total++] = all[i];
    for (uint32_t f = 0; f < FAKE_EXTENSION_COUNT && inst->emulate; ++f)
        all[total++] = kFakeExtensions[f];

    if (!out)
    {
        *count = total;
        free(all);
        return VK_SUCCESS;
    }
    const uint32_t n = *count < total ? *count : total;
    memcpy(out, all, n * sizeof(*out));
    *count = n;
    free(all);
    return n < total ? VK_INCOMPLETE : VK_SUCCESS;
}

static uint32_t VKAPI_CALL Fake_GetImageViewHandleNVX(VkDevice d, const VkImageViewHandleInfoNVX* i) { return 0; }
static uint64_t VKAPI_CALL Fake_GetImageViewHandle64NVX(VkDevice d, const VkImageViewHandleInfoNVX* i) { return 0; }
static VkResult VKAPI_CALL Fake_GetImageViewAddressNVX(VkDevice d, VkImageView v, VkImageViewAddressPropertiesNVX* p)
{
    if (p)
    {
        p->deviceAddress = 0;
        p->size          = 0;
    }
    return VK_ERROR_UNKNOWN;
}
static VkResult VKAPI_CALL Fake_CreateCuModuleNVX(VkDevice d, const VkCuModuleCreateInfoNVX* i, const VkAllocationCallbacks* a, VkCuModuleNVX* m)
{
    return VK_ERROR_INITIALIZATION_FAILED;
}
static VkResult VKAPI_CALL Fake_CreateCuFunctionNVX(VkDevice d, const VkCuFunctionCreateInfoNVX* i, const VkAllocationCallbacks* a, VkCuFunctionNVX* f)
{
    return VK_ERROR_INITIALIZATION_FAILED;
}
static void VKAPI_CALL Fake_DestroyCuModuleNVX(VkDevice d, VkCuModuleNVX m, const VkAllocationCallbacks* a) {}
static void VKAPI_CALL Fake_DestroyCuFunctionNVX(VkDevice d, VkCuFunctionNVX f, const VkAllocationCallbacks* a) {}
static void VKAPI_CALL Fake_CmdCuLaunchKernelNVX(VkCommandBuffer c, const VkCuLaunchInfoNVX* i) {}

static VkResult VKAPI_CALL Fake_SetLatencySleepModeNV(VkDevice d, VkSwapchainKHR s, const VkLatencySleepModeInfoNV* i) { return VK_SUCCESS; }
static VkResult VKAPI_CALL Fake_LatencySleepNV(VkDevice device, VkSwapchainKHR s, const VkLatencySleepInfoNV* info)
{
    DeviceData* d = FindDevice(device);
    if (!d || !d->signalSemaphore || !info)
        return VK_SUCCESS;
    VkSemaphoreSignalInfo signal = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, NULL, info->signalSemaphore, info->value};
    return d->signalSemaphore(device, &signal);
}
static void VKAPI_CALL Fake_SetLatencyMarkerNV(VkDevice d, VkSwapchainKHR s, const VkSetLatencyMarkerInfoNV* i) {}
static void VKAPI_CALL Fake_GetLatencyTimingsNV(VkDevice d, VkSwapchainKHR s, VkGetLatencyMarkerInfoNV* i)
{
    if (i)
        i->timingCount = 0;
}
static void VKAPI_CALL Fake_QueueNotifyOutOfBandNV(VkQueue q, const VkOutOfBandQueueTypeInfoNV* i) {}

static PFN_vkVoidFunction FakeLowLatencyProc(const char* name)
{
#define FAKE(fn) if (!strcmp(name, "vk" #fn)) return (PFN_vkVoidFunction)Fake_##fn;
    FAKE(SetLatencySleepModeNV)
    FAKE(LatencySleepNV)
    FAKE(SetLatencyMarkerNV)
    FAKE(GetLatencyTimingsNV)
    FAKE(QueueNotifyOutOfBandNV)
#undef FAKE
    return NULL;
}

static PFN_vkVoidFunction FakeNvxProc(const char* name)
{
#define FAKE(fn) if (!strcmp(name, "vk" #fn)) return (PFN_vkVoidFunction)Fake_##fn;
    FAKE(GetImageViewHandleNVX)
    FAKE(GetImageViewHandle64NVX)
    FAKE(GetImageViewAddressNVX)
    FAKE(CreateCuModuleNVX)
    FAKE(CreateCuFunctionNVX)
    FAKE(DestroyCuModuleNVX)
    FAKE(DestroyCuFunctionNVX)
    FAKE(CmdCuLaunchKernelNVX)
#undef FAKE
    return NULL;
}

// ---------------------------------------------------------------------------------------------------------------------
// Device creation: drop emulated extensions, enable ML features
// ---------------------------------------------------------------------------------------------------------------------
typedef struct FeatureStorage
{
    VkPhysicalDeviceShaderFloat16Int8Features       f16i8;
    VkPhysicalDevice8BitStorageFeatures             storage8;
    VkPhysicalDevice16BitStorageFeatures            storage16;
    VkPhysicalDeviceVulkanMemoryModelFeatures       memoryModel;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot;
    VkPhysicalDeviceSubgroupSizeControlFeatures     subgroup;
    VkPhysicalDeviceScalarBlockLayoutFeatures       scalar;
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutableDesc;
    VkPhysicalDeviceFeatures2                       features2;
    VkPhysicalDeviceFeatures                        features;
} FeatureStorage;

static const void* FindInChain(const void* pNext, VkStructureType type)
{
    for (const VkBaseInStructure* s = pNext; s; s = s->pNext)
        if (s->sType == type)
            return s;
    return NULL;
}

static void Append(VkDeviceCreateInfo* info, void* s)
{
    ((VkBaseOutStructure*)s)->pNext = (VkBaseOutStructure*)info->pNext;
    info->pNext                     = s;
}

// Adds `name` to the enabled extension list unless already there. `list` has room for the extra entries.
static void AddExtension(const char** list, uint32_t* count, const char* name)
{
    for (uint32_t i = 0; i < *count; ++i)
        if (!strcmp(list[i], name))
            return;
    list[(*count)++] = name;
}

static void EnableMlFeatures(InstanceData* inst, VkPhysicalDevice pd, VkDeviceCreateInfo* info, FeatureStorage* st,
                             const char** exts, uint32_t* extCount)
{
    if (!inst->getFeatures2)
        return;

    // What the GPU supports.
    VkPhysicalDeviceShaderFloat16Int8Features       f16i8  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice8BitStorageFeatures             s8     = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &f16i8};
    VkPhysicalDevice16BitStorageFeatures            s16    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &s8};
    VkPhysicalDeviceVulkanMemoryModelFeatures       mm     = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, &s16};
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, &mm};
    VkPhysicalDeviceSubgroupSizeControlFeatures     sg     = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, &dot};
    VkPhysicalDeviceScalarBlockLayoutFeatures       sbl    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, &sg};
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mdt   = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT};
    if (DriverHasExtension(inst, pd, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME))
        mdt.pNext = &sbl;
    VkPhysicalDeviceFeatures2                       all    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                                              mdt.pNext ? (void*)&mdt : (void*)&sbl};
    inst->getFeatures2(pd, &all);

    VkPhysicalDeviceProperties props;
    inst->getProperties(pd, &props);
    const uint32_t api = inst->apiVersion < props.apiVersion ? inst->apiVersion : props.apiVersion;

    VkPhysicalDeviceVulkan11Features* v11 = (VkPhysicalDeviceVulkan11Features*)FindInChain(info->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    VkPhysicalDeviceVulkan12Features* v12 = (VkPhysicalDeviceVulkan12Features*)FindInChain(info->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    VkPhysicalDeviceVulkan13Features* v13 = (VkPhysicalDeviceVulkan13Features*)FindInChain(info->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);

    // Each group: write into the app's VulkanXX struct, else into its individual struct, else chain our own (and
    // enable the extension when the API version does not include it).
#define ENABLE_GROUP(V, vstruct, Type, STYPE, storage, ext, coreVersion, ...)                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (V)                                                                                                         \
        {                                                                                                              \
            vstruct* t = V;                                                                                            \
            __VA_ARGS__                                                                                                \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            Type* t = (Type*)FindInChain(info->pNext, STYPE);                                                         \
            if (!t)                                                                                                    \
            {                                                                                                          \
                t = &st->storage;                                                                                      \
                memset(t, 0, sizeof(*t));                                                                              \
                t->sType = STYPE;                                                                                      \
                Append(info, t);                                                                                       \
                if (api < coreVersion && DriverHasExtension(inst, pd, ext))                                            \
                    AddExtension(exts, extCount, ext);                                                                 \
            }                                                                                                          \
            __VA_ARGS__                                                                                                \
        }                                                                                                              \
    } while (0)

    if (f16i8.shaderInt8 || f16i8.shaderFloat16)
        ENABLE_GROUP(v12, VkPhysicalDeviceVulkan12Features, VkPhysicalDeviceShaderFloat16Int8Features,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, f16i8,
                     VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME, VK_API_VERSION_1_2,
                     t->shaderInt8 |= f16i8.shaderInt8; t->shaderFloat16 |= f16i8.shaderFloat16;);
    if (s8.storageBuffer8BitAccess)
        ENABLE_GROUP(v12, VkPhysicalDeviceVulkan12Features, VkPhysicalDevice8BitStorageFeatures,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, storage8,
                     VK_KHR_8BIT_STORAGE_EXTENSION_NAME, VK_API_VERSION_1_2,
                     t->storageBuffer8BitAccess = VK_TRUE;);
    if (mm.vulkanMemoryModel)
        ENABLE_GROUP(v12, VkPhysicalDeviceVulkan12Features, VkPhysicalDeviceVulkanMemoryModelFeatures,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, memoryModel,
                     VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME, VK_API_VERSION_1_2,
                     t->vulkanMemoryModel = VK_TRUE;);
    if (s16.storageBuffer16BitAccess)
        ENABLE_GROUP(v11, VkPhysicalDeviceVulkan11Features, VkPhysicalDevice16BitStorageFeatures,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, storage16,
                     VK_KHR_16BIT_STORAGE_EXTENSION_NAME, VK_API_VERSION_1_1,
                     t->storageBuffer16BitAccess = VK_TRUE;);
    if (dot.shaderIntegerDotProduct)
        ENABLE_GROUP(v13, VkPhysicalDeviceVulkan13Features, VkPhysicalDeviceShaderIntegerDotProductFeatures,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, dot,
                     VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, VK_API_VERSION_1_3,
                     t->shaderIntegerDotProduct = VK_TRUE;);
    if (sg.subgroupSizeControl && sg.computeFullSubgroups)
        ENABLE_GROUP(v13, VkPhysicalDeviceVulkan13Features, VkPhysicalDeviceSubgroupSizeControlFeatures,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, subgroup,
                     VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, VK_API_VERSION_1_3,
                     t->subgroupSizeControl = VK_TRUE; t->computeFullSubgroups = VK_TRUE;);

    // XeSS (see the top of the file).
    const char* xessEnv = getenv("AISS_LAYER_XESS");
    if (xessEnv && !strcmp(xessEnv, "0"))
        return;
    if (sbl.scalarBlockLayout)
        ENABLE_GROUP(v12, VkPhysicalDeviceVulkan12Features, VkPhysicalDeviceScalarBlockLayoutFeatures,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, scalar,
                     VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_API_VERSION_1_2,
                     t->scalarBlockLayout = VK_TRUE;);
    if (mdt.mutableDescriptorType)
    {
        // Never core: always the extension (VALVE's alias shares the feature structure).
        VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT* t = (VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT*)FindInChain(
            info->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT);
        if (!t)
        {
            t = &st->mutableDesc;
            memset(t, 0, sizeof(*t));
            t->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
            Append(info, t);
        }
        t->mutableDescriptorType = VK_TRUE;
        AddExtension(exts, extCount, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);
    }
    if (all.features.shaderStorageImageWriteWithoutFormat)
    {
        VkPhysicalDeviceFeatures2* f2 = (VkPhysicalDeviceFeatures2*)FindInChain(info->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        if (f2)
            f2->features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        else if (info->pEnabledFeatures)
        {
            st->features                                      = *info->pEnabledFeatures;
            st->features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
            info->pEnabledFeatures                            = &st->features;
        }
        else
        {
            memset(&st->features2, 0, sizeof(st->features2));
            st->features2.sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            st->features2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
            Append(info, &st->features2);
        }
    }
#undef ENABLE_GROUP
}

static VkResult VKAPI_CALL Layer_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo* info, const VkAllocationCallbacks* alloc,
                                              VkDevice* out)
{
    VkLayerDeviceCreateInfo* chain = (VkLayerDeviceCreateInfo*)info->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo*)chain->pNext;
    if (!chain)
        return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo            = chain->u.pLayerInfo->pNext;
    InstanceData*      inst        = FindInstanceForPhysicalDevice(pd);
    PFN_vkCreateDevice create      = (PFN_vkCreateDevice)gipa(inst->instance, "vkCreateDevice");

    // Enabled extensions minus the emulated ones, plus room for the feature extensions we may add.
    const char** exts     = calloc(info->enabledExtensionCount + 12, sizeof(char*));
    uint32_t     extCount = 0;
    bool         fakeNvx  = false, fakeLowLatency = false;
    for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
    {
        const char* name     = info->ppEnabledExtensionNames[i];
        bool        emulated = false;
        for (uint32_t f = 0; f < FAKE_EXTENSION_COUNT; ++f)
            if (!strcmp(name, kFakeExtensions[f].extensionName) && !DriverHasExtension(inst, pd, name))
                emulated = true;
        if (!emulated)
            exts[extCount++] = name;
        else if (!strcmp(name, VK_NV_LOW_LATENCY_2_EXTENSION_NAME))
            fakeLowLatency = true;
        else if (!strcmp(name, VK_NV_LOW_LATENCY_EXTENSION_NAME))
            ;
        else
            fakeNvx = true;
    }

    if (getenv("AISS_LAYER_LOG"))
        for (uint32_t f = 0; f < FAKE_EXTENSION_COUNT; ++f)
            fprintf(stderr, "aiss-layer: createDevice inst %p emulate %d driver has %s: %d\n", (void*)inst, inst->emulate,
                    kFakeExtensions[f].extensionName, DriverHasExtension(inst, pd, kFakeExtensions[f].extensionName));
    VkDeviceCreateInfo modified = *info;
    FeatureStorage     storage;
    const char*        featuresEnv = getenv("AISS_LAYER_FEATURES");
    if (!featuresEnv || strcmp(featuresEnv, "0"))
        EnableMlFeatures(inst, pd, &modified, &storage, exts, &extCount);
    modified.enabledExtensionCount   = extCount;
    modified.ppEnabledExtensionNames = exts;

    VkResult res = create(pd, &modified, alloc, out);
    if (getenv("AISS_LAYER_LOG"))
    {
        for (uint32_t i = 0; i < modified.enabledExtensionCount; ++i)
            fprintf(stderr, "aiss-layer: device extension %s\n", modified.ppEnabledExtensionNames[i]);
        fprintf(stderr, "aiss-layer: vkCreateDevice -> %d\n", res);
    }
    free(exts);
    if (res != VK_SUCCESS)
        return res;

    DeviceData* d    = calloc(1, sizeof(*d));
    d->key           = DispatchKey(*out);
    d->gdpa          = gdpa;
    d->destroyDevice = (PFN_vkDestroyDevice)gdpa(*out, "vkDestroyDevice");
    // Never serve stubs to an instance the extensions were not advertised to (DXVK, vkd3d-proton).
    d->fakeNvx        = fakeNvx && inst->emulate;
    d->fakeLowLatency = fakeLowLatency && inst->emulate;
    d->signalSemaphore = (PFN_vkSignalSemaphore)gdpa(*out, "vkSignalSemaphore");
    if (!d->signalSemaphore)
        d->signalSemaphore = (PFN_vkSignalSemaphore)gdpa(*out, "vkSignalSemaphoreKHR");
    pthread_mutex_lock(&g_lock);
    d->next   = g_devices;
    g_devices = d;
    pthread_mutex_unlock(&g_lock);
    return VK_SUCCESS;
}

static void VKAPI_CALL Layer_DestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc)
{
    DeviceData* d = FindDevice(device);
    if (!d)
        return;
    d->destroyDevice(device, alloc);
    pthread_mutex_lock(&g_lock);
    for (DeviceData** it = &g_devices; *it; it = &(*it)->next)
        if (*it == d)
        {
            *it = d->next;
            break;
        }
    pthread_mutex_unlock(&g_lock);
    free(d);
}

// ---------------------------------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------------------------------
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL aiss_GetDeviceProcAddr(VkDevice device, const char* name)
{
    if (!strcmp(name, "vkGetDeviceProcAddr"))
        return (PFN_vkVoidFunction)aiss_GetDeviceProcAddr;
    if (!strcmp(name, "vkDestroyDevice"))
        return (PFN_vkVoidFunction)Layer_DestroyDevice;
    DeviceData* d = FindDevice(device);
    if (d && d->fakeNvx)
    {
        PFN_vkVoidFunction fake = FakeNvxProc(name);
        if (fake)
            return fake;
    }
    if (d && d->fakeLowLatency)
    {
        PFN_vkVoidFunction fake = FakeLowLatencyProc(name);
        if (fake)
            return fake;
    }
    PFN_vkVoidFunction fn = d ? d->gdpa(device, name) : NULL;
    if (!fn && getenv("AISS_LAYER_LOG"))
        fprintf(stderr, "aiss-layer: vkGetDeviceProcAddr(%s) -> NULL (device %s)\n", name, d ? "known" : "UNKNOWN");
    return fn;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL aiss_GetInstanceProcAddr(VkInstance instance, const char* name)
{
    if (!strcmp(name, "vkGetInstanceProcAddr"))
        return (PFN_vkVoidFunction)aiss_GetInstanceProcAddr;
    if (!strcmp(name, "vkCreateInstance"))
        return (PFN_vkVoidFunction)Layer_CreateInstance;
    if (!strcmp(name, "vkDestroyInstance"))
        return (PFN_vkVoidFunction)Layer_DestroyInstance;
    if (!strcmp(name, "vkCreateDevice"))
        return (PFN_vkVoidFunction)Layer_CreateDevice;
    if (!strcmp(name, "vkEnumerateDeviceExtensionProperties"))
        return (PFN_vkVoidFunction)Layer_EnumerateDeviceExtensionProperties;
    if (!strcmp(name, "vkEnumeratePhysicalDevices"))
        return (PFN_vkVoidFunction)Layer_EnumeratePhysicalDevices;
    if (!strcmp(name, "vkEnumeratePhysicalDeviceGroups") || !strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR"))
        return (PFN_vkVoidFunction)Layer_EnumeratePhysicalDeviceGroups;
    if (!strcmp(name, "vkGetDeviceProcAddr"))
        return (PFN_vkVoidFunction)aiss_GetDeviceProcAddr;
    if (!strcmp(name, "vkDestroyDevice"))
        return (PFN_vkVoidFunction)Layer_DestroyDevice;
    if (!instance)
        return NULL;
    InstanceData* d = FindInstance(instance);
    return d ? d->gipa(instance, name) : NULL;
}

// Physical-device functions the loader has no trampoline for (newer extensions) are resolved through this chain;
// without it they would come back NULL to DXVK/vkd3d-proton.
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL aiss_GetPhysicalDeviceProcAddr(VkInstance instance, const char* name)
{
    InstanceData* d = FindInstance(instance);
    return d && d->gpdpa ? d->gpdpa(instance, name) : NULL;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL aiss_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v)
{
    if (v->loaderLayerInterfaceVersion > 2)
        v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr       = aiss_GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr         = aiss_GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = aiss_GetPhysicalDeviceProcAddr;
    return VK_SUCCESS;
}
