// SPDX-FileCopyrightText: Copyright 2026 Neural Graphics SDK universal contributors
// SPDX-License-Identifier: MIT

// Conformance and performance test for the native ML compute path.
//
// Runs the NSS networks through every kernel path the selected GPU supports (coop-matrix int8/fp16/bf16, dot4,
// scalar), compares each output bit-for-bit against the CPU TOSA reference and reports the GPU time per inference.
//
//   ffx_ml_runtime_test [--gpu N] [--width W] [--height H] [--iterations N]

#include "../ffx_vk_ml_runtime.h"

#include <nss_v1_0_1_high_int8.h>
#include <nss_v1_0_1_mid_low_int8.h>

#include <algorithm>
#include <cinttypes>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
static void SetEnv(const char* name, const char* value) { _putenv_s(name, value ? value : ""); }
#else
static void SetEnv(const char* name, const char* value)
{
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
}
#endif

#define CHECK_VK(x)                                                                 \
    do                                                                              \
    {                                                                               \
        VkResult r_ = (x);                                                          \
        if (r_ != VK_SUCCESS)                                                       \
        {                                                                           \
            fprintf(stderr, "%s failed: %d (%s:%d)\n", #x, r_, __FILE__, __LINE__); \
            exit(2);                                                                \
        }                                                                           \
    } while (0)

namespace
{
    struct Model
    {
        const char*       name;
        ffx_ml::GraphDesc desc;
        std::vector<uint32_t> words;
    };

    template <typename Info>
    Model MakeModel(const char* name, const Info& info, int height, int width)
    {
        Model m;
        m.name = name;
        m.words.resize(info.graphDataSize / 4);
        memcpy(m.words.data(), info.graphData, info.graphDataSize);
        m.desc.spirv      = m.words.data();
        m.desc.spirvWords = m.words.size();
        for (uint32_t i = 0; i < info.constantNums; ++i)
        {
            ffx_ml::GraphConstant c;
            c.id       = info.constantIds[i];
            c.vkFormat = info.constantFormats[i];
            for (uint32_t d = 0; d < info.constantShapeSize[i]; ++d)
                c.shape.push_back(int64_t(info.constantShapes[i][d]));
            c.data     = info.constantDatas[i];
            c.dataSize = info.constantDataSize[i];
            m.desc.constants.push_back(c);
        }
        for (uint32_t i = 0; i < info.inputTensorNums; ++i)
        {
            std::vector<int64_t> shape{1};
            bool                 isWidth = false;
            for (uint32_t d = 1; d < info.inputTensorDimSize[i]; ++d)
            {
                int64_t v = info.inputTensorDims[i][d];
                if (v <= 0)
                {
                    v       = isWidth ? width : height;
                    isWidth = true;
                }
                shape.push_back(v);
            }
            m.desc.inputShapes[info.inputTensorBindings[i]] = shape;
        }
        return m;
    }

    struct Buffer
    {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void*          mapped = nullptr;
        size_t         size   = 0;
    };

    struct Context
    {
        VkInstance                       instance = VK_NULL_HANDLE;
        VkPhysicalDevice                 physical = VK_NULL_HANDLE;
        VkDevice                         device   = VK_NULL_HANDLE;
        VkQueue                          queue    = VK_NULL_HANDLE;
        uint32_t                         queueFamily = 0;
        VkCommandPool                    pool     = VK_NULL_HANDLE;
        VkQueryPool                      queries  = VK_NULL_HANDLE;
        uint32_t                         features = 0;
        float                            timestampPeriod = 1.0f;
        VkPhysicalDeviceMemoryProperties memory   = {};

        // Host-visible buffers are for staging; the graph reads and writes device-local ones, as it does in NSS.
        Buffer CreateBuffer(size_t size, bool deviceLocal = false)
        {
            Buffer             b;
            VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size               = size;
            info.usage              = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            CHECK_VK(vkCreateBuffer(device, &info, nullptr, &b.buffer));
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, b.buffer, &req);
            uint32_t type = UINT32_MAX;
            for (uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX; ++i)
            {
                const VkMemoryPropertyFlags want =
                    deviceLocal ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if ((req.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want)
                    type = i;
            }
            VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type};
            CHECK_VK(vkAllocateMemory(device, &alloc, nullptr, &b.memory));
            CHECK_VK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
            if (!deviceLocal)
                CHECK_VK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
            b.size = size;
            return b;
        }

        void Destroy(Buffer& b)
        {
            if (b.mapped)
                vkUnmapMemory(device, b.memory);
            vkDestroyBuffer(device, b.buffer, nullptr);
            vkFreeMemory(device, b.memory, nullptr);
            b = {};
        }
    };

    void CreateContext(Context& ctx, int gpuIndex)
    {
        VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName  = "ffx_ml_runtime_test";
        app.apiVersion        = VK_API_VERSION_1_3;
        VkInstanceCreateInfo instanceInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo     = &app;
        CHECK_VK(vkCreateInstance(&instanceInfo, nullptr, &ctx.instance));

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr);
        std::vector<VkPhysicalDevice> gpus(count);
        vkEnumeratePhysicalDevices(ctx.instance, &count, gpus.data());
        if (gpuIndex >= int(count))
        {
            fprintf(stderr, "GPU %d not found (%u available)\n", gpuIndex, count);
            exit(2);
        }
        ctx.physical = gpus[gpuIndex];

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx.physical, &props);
        ctx.timestampPeriod = props.limits.timestampPeriod;
        vkGetPhysicalDeviceMemoryProperties(ctx.physical, &ctx.memory);

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical, &familyCount, families.data());
        for (uint32_t i = 0; i < familyCount; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                ctx.queueFamily = i;
                break;
            }
        }

        // Enable everything the ML path can use that the device supports.
        std::vector<const char*> extensions;
        const uint32_t supported = ffx_ml::QueryDeviceFeatures(ctx.instance, ctx.physical, vkGetInstanceProcAddr, &extensions);

        VkPhysicalDeviceShaderBfloat16FeaturesKHR    bfloat16   = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrix = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceVulkan13Features             v13        = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features             v12        = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features             v11        = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2                    features2  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &v11;
        v11.pNext       = &v12;
        v12.pNext       = &v13;
        void** tail     = &v13.pNext;

        using namespace ffx_ml;
        v12.shaderInt8               = (supported & FEATURE_SHADER_INT8) ? VK_TRUE : VK_FALSE;
        v12.storageBuffer8BitAccess  = (supported & FEATURE_STORAGE_BUFFER_8BIT) ? VK_TRUE : VK_FALSE;
        v12.shaderFloat16            = (supported & FEATURE_SHADER_FLOAT16) ? VK_TRUE : VK_FALSE;
        v12.vulkanMemoryModel        = (supported & FEATURE_VULKAN_MEMORY_MODEL) ? VK_TRUE : VK_FALSE;
        v12.vulkanMemoryModelDeviceScope = v12.vulkanMemoryModel;
        v11.storageBuffer16BitAccess = (supported & FEATURE_STORAGE_BUFFER_16BIT) ? VK_TRUE : VK_FALSE;
        v13.shaderIntegerDotProduct  = (supported & FEATURE_INTEGER_DOT_PRODUCT) ? VK_TRUE : VK_FALSE;
        v13.subgroupSizeControl      = (supported & FEATURE_SUBGROUP_SIZE_CONTROL) ? VK_TRUE : VK_FALSE;
        v13.computeFullSubgroups     = v13.subgroupSizeControl;
        v13.synchronization2         = VK_TRUE;
        if (supported & FEATURE_COOPERATIVE_MATRIX)
        {
            coopMatrix.cooperativeMatrix = VK_TRUE;
            *tail                        = &coopMatrix;
            tail                         = &coopMatrix.pNext;
        }
        if (supported & FEATURE_SHADER_BFLOAT16)
        {
            bfloat16.shaderBFloat16Type              = VK_TRUE;
            bfloat16.shaderBFloat16CooperativeMatrix = VK_TRUE;
            *tail                                    = &bfloat16;
            tail                                     = &bfloat16.pNext;
        }

        const float             priority  = 1.0f;
        VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, ctx.queueFamily, 1, &priority};
        VkDeviceCreateInfo      deviceInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.pNext                   = &features2;
        deviceInfo.queueCreateInfoCount    = 1;
        deviceInfo.pQueueCreateInfos       = &queueInfo;
        deviceInfo.enabledExtensionCount   = uint32_t(extensions.size());
        deviceInfo.ppEnabledExtensionNames = extensions.data();
        CHECK_VK(vkCreateDevice(ctx.physical, &deviceInfo, nullptr, &ctx.device));
        ctx.features = supported;
        vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex        = ctx.queueFamily;
        CHECK_VK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.pool));
        VkQueryPoolCreateInfo queryInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryInfo.queryType             = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount            = 2;
        CHECK_VK(vkCreateQueryPool(ctx.device, &queryInfo, nullptr, &ctx.queries));

        printf("GPU %d: %s, ML features 0x%03x\n", gpuIndex, props.deviceName, supported);
    }

    struct Config
    {
        ffx_ml::Path    path;
        ffx_ml::MatType type;
        const char*     env = nullptr;  // "NAME=value" set for this run only (kernel variants)
    };

    // Returns true when the GPU result matches the reference exactly.
    bool RunConfig(Context& ctx, const Model& model, const Config& config, const std::map<uint32_t, std::vector<int8_t>>& inputs,
                   const std::map<uint32_t, std::vector<int8_t>>& reference, int iterations, bool& skipped)
    {
        const ffx_ml::DeviceContext dc  = {ctx.instance, ctx.physical, ctx.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, ctx.features};
        const ffx_ml::Selection     sel = ffx_ml::SelectPath(dc, config.path, config.type);
        skipped                         = sel.path != config.path || (config.type != ffx_ml::MatType::None && sel.matType != config.type);
        if (skipped)
            return true;

        std::string error;
        auto        graph = ffx_ml::Graph::Create(dc, sel, model.desc, &error);
        if (!graph)
        {
            printf("  %-48s FAILED to create: %s\n", sel.Describe().c_str(), error.c_str());
            return false;
        }

        std::map<uint32_t, Buffer>          buffers;
        std::vector<ffx_ml::ExternalTensor> externals;
        for (const auto& [binding, data] : inputs)
        {
            buffers[binding] = ctx.CreateBuffer(data.size());
            memcpy(buffers[binding].mapped, data.data(), data.size());
        }
        for (const auto& [binding, data] : reference)
        {
            buffers[binding] = ctx.CreateBuffer(data.size());
            memset(buffers[binding].mapped, 0x55, data.size());
        }
        std::map<uint32_t, Buffer> deviceBuffers;
        for (auto& [binding, b] : buffers)
        {
            deviceBuffers[binding] = ctx.CreateBuffer(b.size, true);
            externals.push_back({binding, deviceBuffers[binding].buffer, 0});
        }

        VkCommandBuffer             cmd;
        VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, ctx.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        CHECK_VK(vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd));

        // Copies between the staging and device-local buffers, outside the timed region.
        auto copyAll = [&](bool upload) {
            VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK_VK(vkBeginCommandBuffer(cmd, &begin));
            for (auto& [binding, b] : buffers)
            {
                VkBufferCopy region = {0, 0, b.size};
                if (upload)
                    vkCmdCopyBuffer(cmd, b.buffer, deviceBuffers[binding].buffer, 1, &region);
                else
                    vkCmdCopyBuffer(cmd, deviceBuffers[binding].buffer, b.buffer, 1, &region);
            }
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            CHECK_VK(vkEndCommandBuffer(cmd));
            VkSubmitInfo submit       = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers    = &cmd;
            CHECK_VK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
            CHECK_VK(vkQueueWaitIdle(ctx.queue));
        };
        copyAll(true);

        // Per-layer profile (separate query pool): first query before the graph, one after each dispatch. Each submit runs
        // the graph kProfileRuns times back to back (own query range each) so the GPU stays clocked up.
        const uint32_t layerCount = graph->GetDispatchCount();
        const uint32_t kProfileRuns = 16;
        VkQueryPool    profilePool = VK_NULL_HANDLE;
        if (getenv("ML_PROFILE"))
        {
            VkQueryPoolCreateInfo queryInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            queryInfo.queryType             = VK_QUERY_TYPE_TIMESTAMP;
            queryInfo.queryCount            = (layerCount + 1) * kProfileRuns;
            CHECK_VK(vkCreateQueryPool(ctx.device, &queryInfo, nullptr, &profilePool));
        }
        std::vector<double> layerMs(layerCount, 1e30);
        double              profileTotalMs = 1e30;

        double bestMs = 1e30, totalMs = 0;
        for (int it = 0; it < iterations + 1; ++it)
        {
            VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK_VK(vkBeginCommandBuffer(cmd, &begin));
            vkCmdResetQueryPool(cmd, ctx.queries, 0, 2);
            if (profilePool)
                vkCmdResetQueryPool(cmd, profilePool, 0, (layerCount + 1) * kProfileRuns);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.queries, 0);
            const uint32_t runs = (profilePool && it > 0) ? kProfileRuns : 1;
            for (uint32_t run = 0; run < runs; ++run)
            {
                if (run > 0)
                {
                    VkMemoryBarrier between = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &between, 0, nullptr, 0, nullptr);
                }
                if (profilePool)
                    graph->SetProfiling(it > 0 ? profilePool : VK_NULL_HANDLE, run * (layerCount + 1));
                graph->Record(cmd, externals.data(), uint32_t(externals.size()));
            }
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.queries, 1);
            VkMemoryBarrier toHost = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &toHost, 0, nullptr, 0, nullptr);
            CHECK_VK(vkEndCommandBuffer(cmd));
            VkSubmitInfo submit       = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers    = &cmd;
            CHECK_VK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
            CHECK_VK(vkQueueWaitIdle(ctx.queue));
            uint64_t ts[2];
            CHECK_VK(vkGetQueryPoolResults(ctx.device, ctx.queries, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const double ms = double(ts[1] - ts[0]) * ctx.timestampPeriod * 1e-6;
            if (profilePool && it > 0)
            {
                std::vector<uint64_t> lts((layerCount + 1) * kProfileRuns);
                CHECK_VK(vkGetQueryPoolResults(ctx.device, profilePool, 0, uint32_t(lts.size()), lts.size() * 8, lts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                const bool warm = it > iterations / 3;  // the first submits ramp the GPU clocks up
                for (uint32_t run = 1; warm && run < kProfileRuns; ++run)  // run 0 starts from an idle GPU
                {
                    const uint64_t* r = &lts[run * (layerCount + 1)];
                    double          t = 0;
                    for (uint32_t li = 0; li < layerCount; ++li)
                    {
                        layerMs[li] = std::min(layerMs[li], double(r[li + 1] - r[li]) * ctx.timestampPeriod * 1e-6);
                        t += double(r[li + 1] - r[li]) * ctx.timestampPeriod * 1e-6;
                    }
                    profileTotalMs = std::min(profileTotalMs, t);
                }
                continue;  // multi-run submits are not single-graph timings
            }
            if (it > 0)  // first iteration includes the one-time weight upload
            {
                bestMs = std::min(bestMs, ms);
                totalMs += ms;
            }
        }

        // Steady-state cost: many back-to-back executions in one submit, so the GPU stays clocked up as in a game
        // frame (single submits leave it idle in between and read high). Best of a few batches.
        double batchMs = 1e30;
        if (!profilePool)
        {
            const int batch = 64;
            for (int rep = 0; rep < 16; ++rep)  // the first batches warm the clocks up; only the last 4 count
            {
                VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                CHECK_VK(vkBeginCommandBuffer(cmd, &begin));
                vkCmdResetQueryPool(cmd, ctx.queries, 0, 2);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.queries, 0);
                for (int b = 0; b < batch; ++b)
                {
                    VkMemoryBarrier between = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &between, 0, nullptr, 0, nullptr);
                    graph->Record(cmd, externals.data(), uint32_t(externals.size()));
                }
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.queries, 1);
                CHECK_VK(vkEndCommandBuffer(cmd));
                VkSubmitInfo submit       = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers    = &cmd;
                CHECK_VK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
                CHECK_VK(vkQueueWaitIdle(ctx.queue));
                uint64_t ts[2];
                CHECK_VK(vkGetQueryPoolResults(ctx.device, ctx.queries, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                if (rep >= 12)
                    batchMs = std::min(batchMs, double(ts[1] - ts[0]) * ctx.timestampPeriod * 1e-6 / batch);
            }
        }

        copyAll(false);

        size_t mismatches = 0, total = 0;
        int    maxDiff    = 0;
        for (const auto& [binding, expected] : reference)
        {
            const int8_t* got = static_cast<const int8_t*>(buffers[binding].mapped);
            for (size_t i = 0; i < expected.size(); ++i)
            {
                const int d = std::abs(int(got[i]) - int(expected[i]));
                mismatches += d != 0;
                maxDiff = std::max(maxDiff, d);
            }
            total += expected.size();
        }

        const bool pass = mismatches == 0;
        printf("  %-56s %s  mismatches %zu/%zu (max |diff| %d)  %.3f ms best, %.3f ms avg, %.3f ms batched\n",
               sel.Describe().c_str(),
               pass ? "EXACT " : "FAIL  ",
               mismatches,
               total,
               maxDiff,
               bestMs,
               totalMs / iterations,
               batchMs);

        if (profilePool)
        {
            for (uint32_t li = 0; li < layerCount; ++li)
                printf("      layer %2u %-52s %.4f ms\n", li, graph->DescribeLayer(li).c_str(), layerMs[li]);
            printf("      sum of per-layer minima %.4f ms, fastest whole graph %.4f ms\n", std::accumulate(layerMs.begin(), layerMs.end(), 0.0), profileTotalMs);
            vkDestroyQueryPool(ctx.device, profilePool, nullptr);
        }
        vkFreeCommandBuffers(ctx.device, ctx.pool, 1, &cmd);
        graph.reset();
        for (auto& [binding, b] : buffers)
            ctx.Destroy(b);
        for (auto& [binding, b] : deviceBuffers)
            ctx.Destroy(b);
        return pass;
    }
}  // namespace

int main(int argc, char** argv)
{
    int gpu = 0, width = 960, height = 544, iterations = 20;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        if (!strcmp(argv[i], "--gpu"))
            gpu = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--width"))
            width = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--height"))
            height = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--iterations"))
            iterations = atoi(argv[i + 1]);
    }

    Context ctx;
    CreateContext(ctx, gpu);

    std::vector<Model> models;
    models.push_back(MakeModel("nss_v1_0_1_high_int8", g_nss_v1_0_1_high_int8_Info, height, width));
    models.push_back(MakeModel("nss_v1_0_1_mid_low_int8", g_nss_v1_0_1_mid_low_int8_Info, height / 2, width / 2));
    for (Model& m : models)
        m.desc.spirv = m.words.data();  // re-point after the copies into the vector
    // FFX_ML_TEST_MODELS=high|mid_low runs one model only (tuning loops).
    if (const char* only = getenv("FFX_ML_TEST_MODELS"))
        models.erase(std::remove_if(models.begin(), models.end(), [&](const Model& m) { return !strstr(m.name, (std::string("v1_0_1_") + only + "_int8").c_str()); }),
                     models.end());

    const Config configs[] = {
        {ffx_ml::Path::CoopMat, ffx_ml::MatType::Int8},
        {ffx_ml::Path::CoopMat, ffx_ml::MatType::Fp16},
        {ffx_ml::Path::CoopMat, ffx_ml::MatType::Bf16},
        {ffx_ml::Path::Dot4, ffx_ml::MatType::None},
        {ffx_ml::Path::Dot4, ffx_ml::MatType::None, "FFX_ML_DOT4_TILED=0"},  // untiled dot4 kernel (same as above where tiling is off)
        {ffx_ml::Path::Scalar, ffx_ml::MatType::None},
    };

    const ffx_ml::DeviceContext dc = {ctx.instance, ctx.physical, ctx.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, ctx.features};
    printf("Auto selection: %s\n", ffx_ml::SelectPath(dc).Describe().c_str());

    bool allPass = true;
    for (const Model& model : models)
    {
        std::mt19937                            rng(1234);
        std::uniform_int_distribution<int>      dist(-128, 127);
        std::map<uint32_t, std::vector<int8_t>> inputs, reference;
        for (const auto& [binding, shape] : model.desc.inputShapes)
        {
            size_t n = 1;
            for (int64_t d : shape)
                n *= size_t(d);
            auto& v = inputs[binding];
            v.resize(n);
            for (auto& x : v)
                x = int8_t(dist(rng));
        }
        std::string error;
        if (!ffx_ml::RunReference(model.desc, inputs, reference, &error))
        {
            printf("%s: reference failed: %s\n", model.name, error.c_str());
            return 1;
        }
        const auto& inShape = model.desc.inputShapes.begin()->second;
        printf("%s (input %" PRId64 "x%" PRId64 "x%" PRId64 ")\n", model.name, inShape[1], inShape[2], inShape[3]);
        for (const Config& config : configs)
        {
            // FFX_ML_TEST_PATHS=dot4,scalar,... limits the paths run (faster tuning loops).
            if (const char* only = getenv("FFX_ML_TEST_PATHS"))
                if (!strstr(only, ffx_ml::ToString(config.path)))
                    continue;
            std::string name;
            if (config.env)
            {
                const char* eq = strchr(config.env, '=');
                name           = std::string(config.env, eq);
                if (getenv(name.c_str()))
                    continue;  // the caller forced this variable for every run, so the variant run would be a duplicate
                SetEnv(name.c_str(), eq + 1);
            }
            bool skipped = false;
            allPass &= RunConfig(ctx, model, config, inputs, reference, iterations, skipped);
            if (config.env)
                SetEnv(name.c_str(), nullptr);
            if (skipped)
                printf("  %-56s skipped (not supported)\n", (std::string(ffx_ml::ToString(config.path)) + " " + ffx_ml::ToString(config.type)).c_str());
        }
    }

    vkDestroyQueryPool(ctx.device, ctx.queries, nullptr);
    vkDestroyCommandPool(ctx.device, ctx.pool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
    printf("%s\n", allPass ? "ALL PASSED" : "FAILURES");
    return allPass ? 0 : 1;
}
