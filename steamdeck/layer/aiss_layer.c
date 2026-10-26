// VK_LAYER_AISS_ngx: native (Linux) implicit Vulkan layer used by the NSS NGX proxy under Proton.
//
// 1. Advertises VK_NVX_binary_import, VK_NVX_image_view_handle and VK_NV_low_latency(2) with stub entry points when the
//    driver lacks them. Vulkan DLSS integrations (Streamline, NGX) require them at device creation even though the NSS
//    proxy never uses them; the layer strips them again before calling the driver. Reflex sleep becomes a no-op that
//    signals the requested semaphore value, so frame pacing never blocks.
// 2. Enables the device features NSS's ML kernels run fastest with (int8/fp16 arithmetic, 8/16-bit storage, integer dot
//    products, subgroup size control, the Vulkan memory model) on every device, as far as the GPU supports them.
//    A Vulkan device's enabled features cannot be queried later, so this is the only way to guarantee them.
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
    PFN_vkGetInstanceProcAddr                gipa;
    PFN_vkDestroyInstance                    destroyInstance;
    PFN_vkEnumerateDeviceExtensionProperties enumerateDeviceExtensions;
    PFN_vkGetPhysicalDeviceFeatures2         getFeatures2;
    PFN_vkGetPhysicalDeviceProperties        getProperties;
    struct InstanceData*                     next;
} InstanceData;

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

// Physical devices share their instance's dispatch table, so instance lookup works for them too.
static InstanceData* FindInstanceForPhysicalDevice(VkPhysicalDevice pd) { return FindInstance(pd); }

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

    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo            = chain->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create    = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult             res       = create(info, alloc, out);
    if (res != VK_SUCCESS)
        return res;

    InstanceData* d              = calloc(1, sizeof(*d));
    d->key                       = DispatchKey(*out);
    d->instance                  = *out;
    d->apiVersion                = info->pApplicationInfo && info->pApplicationInfo->apiVersion ? info->pApplicationInfo->apiVersion
                                                                                                : VK_API_VERSION_1_0;
    d->gipa                      = gipa;
    d->destroyInstance           = (PFN_vkDestroyInstance)gipa(*out, "vkDestroyInstance");
    d->enumerateDeviceExtensions = (PFN_vkEnumerateDeviceExtensionProperties)gipa(*out, "vkEnumerateDeviceExtensionProperties");
    d->getFeatures2              = (PFN_vkGetPhysicalDeviceFeatures2)gipa(*out, "vkGetPhysicalDeviceFeatures2");
    if (!d->getFeatures2)
        d->getFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(*out, "vkGetPhysicalDeviceFeatures2KHR");
    d->getProperties = (PFN_vkGetPhysicalDeviceProperties)gipa(*out, "vkGetPhysicalDeviceProperties");
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

static bool DriverHasExtension(InstanceData* inst, VkPhysicalDevice pd, const char* name)
{
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
    uint32_t total = driverCount;
    for (uint32_t f = 0; f < FAKE_EXTENSION_COUNT; ++f)
    {
        bool present = false;
        for (uint32_t i = 0; i < driverCount && !present; ++i)
            present = !strcmp(all[i].extensionName, kFakeExtensions[f].extensionName);
        if (!present)
            all[total++] = kFakeExtensions[f];
    }

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
    VkPhysicalDeviceFeatures2                       all    = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &sg};
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
    const char** exts     = calloc(info->enabledExtensionCount + 8, sizeof(char*));
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

    VkDeviceCreateInfo modified = *info;
    FeatureStorage     storage;
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
    d->fakeNvx        = fakeNvx;
    d->fakeLowLatency = fakeLowLatency;
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
    return d ? d->gdpa(device, name) : NULL;
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
    if (!strcmp(name, "vkGetDeviceProcAddr"))
        return (PFN_vkVoidFunction)aiss_GetDeviceProcAddr;
    if (!strcmp(name, "vkDestroyDevice"))
        return (PFN_vkVoidFunction)Layer_DestroyDevice;
    if (!instance)
        return NULL;
    InstanceData* d = FindInstance(instance);
    return d ? d->gipa(instance, name) : NULL;
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL aiss_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v)
{
    if (v->loaderLayerInterfaceVersion > 2)
        v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr       = aiss_GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr         = aiss_GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
