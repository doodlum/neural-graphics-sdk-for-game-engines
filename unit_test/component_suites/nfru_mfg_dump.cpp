/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

// Opt-in dataset replay / dump harness for NFRU multi-frame generation parity work.
//
// Environment:
//   NFRU_MFG_DATASET   directory written by tools/nfru_mfg_dump/extract_safetensors_frames.py (meta.txt + raw frames)
//   NFRU_MFG_DUMP_DIR  output root; <root>/<path>/N<n>/ receives the dumps (see tools/nfru_mfg_dump/README.md)
//   NFRU_MFG_DUMP_N    generated-frame counts to run, e.g. "1,2,3,4,5" (default "1"; > 1 needs an MFG build)
//   NFRU_MFG_DUMP_PATH "fragment" (default) or "compute" (ALL_STAGES_COMPUTE)
//   NFRU_MFG_DUMP_CAPTURE 1 to dump per-timestep intermediates and histories (MFG build; compute path recommended,
//                      and FFX_NFRU_DEBUG_CAPTURE_TENSORS=1 to include the input tensor)
//   NFRU_MFG_DUMP_FRAMES  number of rendered frames to replay (default: all in the dataset)
// The test is skipped when NFRU_MFG_DATASET is not set.

#include "../framework/test_framework.h"
#include "../framework/vk_test_utils.h"

#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_framegeneration.h>
#include "../../sdk/src/components/frameinterpolation/ffx_frameinterpolation_private.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES)
#define NFRU_MFG_DUMP_MFG_API 1
#else
#define NFRU_MFG_DUMP_MFG_API 0
#endif

namespace
{
    struct DumpFrameMeta
    {
        float near     = 0.1f;
        float far      = 1000.0f;
        float fovY     = 1.0f;
        bool  infinite = false;
        float vp[16]   = {};
    };

    struct DumpDataset
    {
        uint32_t                   displayW = 0, displayH = 0, mvW = 0, mvH = 0;
        std::vector<DumpFrameMeta> frames;
    };

    std::vector<std::string> g_dumpMessages;

    void DumpMessageSink(uint32_t type, const char* message)
    {
        if (message)
        {
            g_dumpMessages.emplace_back(std::to_string(type) + " " + message);
        }
    }

    bool ReadFile(const std::filesystem::path& path, std::vector<uint8_t>& bytes)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            return false;
        bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        return true;
    }

    bool WriteFile(const std::filesystem::path& path, const void* data, size_t size)
    {
        std::ofstream file(path, std::ios::binary);
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        return static_cast<bool>(file);
    }

    bool LoadDataset(const std::filesystem::path& dir, DumpDataset& dataset)
    {
        std::ifstream meta(dir / "meta.txt");
        if (!meta)
            return false;
        std::string line;
        while (std::getline(meta, line))
        {
            std::istringstream in(line);
            std::string        key;
            in >> key;
            if (key == "display")
                in >> dataset.displayW >> dataset.displayH;
            else if (key == "mvdepth")
                in >> dataset.mvW >> dataset.mvH;
            else if (key == "frame")
            {
                DumpFrameMeta frame;
                int           index = 0, infinite = 0;
                in >> index >> frame.near >> frame.far >> frame.fovY >> infinite;
                frame.infinite = infinite != 0;
                for (float& v : frame.vp)
                    in >> v;
                dataset.frames.push_back(frame);
            }
        }
        return dataset.displayW && dataset.mvW && !dataset.frames.empty();
    }

    std::vector<uint32_t> ParseCounts(const char* text)
    {
        std::vector<uint32_t> counts;
        std::stringstream     in(text ? text : "1");
        std::string           item;
        while (std::getline(in, item, ','))
        {
            if (!item.empty())
                counts.push_back(static_cast<uint32_t>(std::stoul(item)));
        }
        return counts;
    }

    struct HostBuffer
    {
        unit_test::vk::Buffer buffer;
        VkBufferCreateInfo    info{};
        void*                 mapped = nullptr;
    };

    bool CreateHostBuffer(unit_test::vk::VulkanContext& vk, VkDeviceSize size, HostBuffer& out, std::string* error)
    {
        out.info       = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        out.info.size  = size;
        out.info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (!unit_test::vk::CreateBuffer(
                vk, size, out.info.usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, out.buffer, error))
            return false;
        return vkMapMemory(vk.device, out.buffer.memory, 0, size, 0, &out.mapped) == VK_SUCCESS;
    }

    FfxApiResource HostBufferResource(const HostBuffer& buffer)
    {
        return ffxApiGetResourceVK(reinterpret_cast<void*>(buffer.buffer.buffer),
                                   ffxApiGetBufferResourceDescriptionVK(buffer.buffer.buffer, buffer.info, 0),
                                   FFX_API_RESOURCE_STATE_COPY_DEST);
    }

    bool RunDumpVariant(const std::filesystem::path& datasetDir,
                        const DumpDataset&           dataset,
                        uint32_t                     generated,
                        bool                         compute,
                        bool                         capture,
                        uint32_t                     frameCount,
                        const std::filesystem::path& outDir,
                        std::string&                 failure)
    {
        std::filesystem::create_directories(outDir);
        g_dumpMessages.clear();

        unit_test::vk::VulkanContext vk{};
        std::string                  error;
        if (!unit_test::vk::InitializeVulkanContext(vk, &error))
        {
            failure = "Vulkan unavailable: " + error;
            return false;
        }
        VkDeviceContext deviceContext{vk.device, vk.physicalDevice, vk.getDeviceProcAddr, vk.instance, vk.getInstanceProcAddr};
        InitVulkanWrapper(deviceContext);

        uint32_t flags =
            FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
        if (dataset.frames[0].infinite)
            flags |= FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE;
        if (compute)
            flags |= FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE;

        auto                                   backendDesc = unit_test::vk::BuildBackendDesc(vk);
        ffxApiCreateContextDescFrameGeneration createDesc{};
        createDesc.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        createDesc.header.pNext     = reinterpret_cast<ffxApiHeader*>(&backendDesc);
        createDesc.flags            = flags;
        createDesc.displaySize      = {dataset.displayW, dataset.displayH};
        createDesc.renderSize       = {dataset.displayW, dataset.displayH};
        createDesc.mvDepthSize      = {dataset.mvW, dataset.mvH};
        createDesc.backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
        createDesc.fpMessage        = DumpMessageSink;
        std::memcpy(createDesc.initialViewProjection, dataset.frames[0].vp, sizeof(createDesc.initialViewProjection));

        ffxContext context = nullptr;
        bool       ok      = ffxCreateContext(&context, &createDesc.header, nullptr) == FFX_API_RETURN_OK;
        if (!ok)
            failure = "ffxCreateContext failed";

        const VkImageUsageFlags colourUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkImageUsageFlags           dataUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        unit_test::vk::Image              present{}, depth{}, motion{};
        std::vector<unit_test::vk::Image> outputs(generated);
        ok = ok && unit_test::vk::CreateImage2D(vk, dataset.displayW, dataset.displayH, VK_FORMAT_R8G8B8A8_UNORM, colourUsage, present, &error);
        ok = ok && unit_test::vk::CreateImage2D(vk, dataset.mvW, dataset.mvH, VK_FORMAT_R32_SFLOAT, dataUsage, depth, &error);
        ok = ok && unit_test::vk::CreateImage2D(vk, dataset.mvW, dataset.mvH, VK_FORMAT_R32G32_SFLOAT, dataUsage, motion, &error);
        for (auto& output : outputs)
            ok = ok && unit_test::vk::CreateImage2D(vk, dataset.displayW, dataset.displayH, VK_FORMAT_R8G8B8A8_UNORM, colourUsage, output, &error);
        const VkImageLayout outputLayout = compute ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        const uint32_t      outputState  = compute ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_RENDER_TARGET;
        const uint32_t      outputUsage  = compute ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_RENDERTARGET;

        // Debug capture buffers (one slice per generated frame).
        const FfxNfruResolutionPlan plan =
            ffxNfruDeriveResolutionPlan({dataset.displayW, dataset.displayH}, {dataset.mvW, dataset.mvH}, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
        const VkDeviceSize laneTexels   = VkDeviceSize(plan.mvDepthLane.width) * plan.mvDepthLane.height;
        const VkDeviceSize flowTexels   = VkDeviceSize(plan.flowLane.width) * plan.flowLane.height;
        const VkDeviceSize tensorTexels = VkDeviceSize(plan.paddedTensorLane.width) * plan.paddedTensorLane.height;
        const VkDeviceSize paramTexels  = VkDeviceSize(plan.paramsLane.width) * plan.paramsLane.height;
        HostBuffer         qmvd{}, flow{}, tensor{}, params{}, gate{}, inpaint{}, dynmask{}, depthHist{}, colourHist{};
#if NFRU_MFG_DUMP_MFG_API
        ffxApiFrameGenerationDebugCapture captureDesc{};
        if (ok && capture)
        {
            ok = CreateHostBuffer(vk, laneTexels * 4 * generated, qmvd, &error) && CreateHostBuffer(vk, flowTexels * 4 * generated, flow, &error) &&
                 CreateHostBuffer(vk, tensorTexels * 20 * generated, tensor, &error) && CreateHostBuffer(vk, paramTexels * 4 * generated, params, &error) &&
                 CreateHostBuffer(vk, paramTexels * generated, gate, &error) && CreateHostBuffer(vk, flowTexels * 4 * generated, inpaint, &error) &&
                 CreateHostBuffer(vk, laneTexels, dynmask, &error) && CreateHostBuffer(vk, laneTexels * 4, depthHist, &error) &&
                 CreateHostBuffer(vk, VkDeviceSize(dataset.displayW) * dataset.displayH * 4, colourHist, &error);
            captureDesc.warpedMotionQData = HostBufferResource(qmvd);
            captureDesc.warpedFilledFlow  = HostBufferResource(flow);
            captureDesc.inputTensor       = HostBufferResource(tensor);
            captureDesc.paramsTensor      = HostBufferResource(params);
            captureDesc.gateTensor        = HostBufferResource(gate);
            captureDesc.inpaintResult     = HostBufferResource(inpaint);
            captureDesc.dynamicMaskTp1    = HostBufferResource(dynmask);
            captureDesc.depthHistory      = HostBufferResource(depthHist);
            captureDesc.colourHistory     = HostBufferResource(colourHist);
            ffxApiConfigureDescFrameGenerationKeyValue keyValue{};
            keyValue.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE;
            keyValue.key         = FFX_API_CONFIGURE_FG_KEY_DEBUG_CAPTURE;
            keyValue.ptr         = &captureDesc;
            ok                   = ok && ffxConfigure(&context, &keyValue.header) == FFX_API_RETURN_OK;
        }
#else
        (void)capture;
#endif
        if (!ok && failure.empty())
            failure = "setup failed: " + error;

        std::ofstream index(outDir / "index.txt");
        index << "generated " << generated << "\npath " << (compute ? "compute" : "fragment") << "\ndisplay " << dataset.displayW << " " << dataset.displayH
              << "\nmvdepth_lane " << plan.mvDepthLane.width << " " << plan.mvDepthLane.height << "\nflow_lane " << plan.flowLane.width << " "
              << plan.flowLane.height << "\ntensor_lane " << plan.paddedTensorLane.width << " " << plan.paddedTensorLane.height << "\nparams_lane "
              << plan.paramsLane.width << " " << plan.paramsLane.height << "\n";

        std::vector<uint8_t> bytes;
        for (uint32_t frame = 0; ok && frame < frameCount; ++frame)
        {
            char name[64];
            std::snprintf(name, sizeof(name), "color_%04u.bin", frame);
            ok = ReadFile(datasetDir / name, bytes) &&
                 unit_test::vk::UploadImage(vk, present, bytes.data(), bytes.size(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, &error);
            std::snprintf(name, sizeof(name), "depth_%04u.bin", frame);
            ok = ok && ReadFile(datasetDir / name, bytes) &&
                 unit_test::vk::UploadImage(vk, depth, bytes.data(), bytes.size(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, &error);
            std::snprintf(name, sizeof(name), "mv_%04u.bin", frame);
            ok = ok && ReadFile(datasetDir / name, bytes) &&
                 unit_test::vk::UploadImage(vk, motion, bytes.data(), bytes.size(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, &error);
            for (auto& output : outputs)
                ok = ok && unit_test::vk::TransitionImageLayout(vk, output, outputLayout, &error);
            if (!ok)
            {
                failure = "frame upload failed: " + error;
                break;
            }

            ffxApiConfigureDescFrameGeneration configureDesc{};
            configureDesc.header.type            = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
            configureDesc.frameGenerationEnabled = true;
            configureDesc.flags                  = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY | FFX_API_FG_DISPATCH_FLAG_ENABLE_DEBUG_CHECKING;
            configureDesc.frameID                = frame;
            ffxConfigure(&context, &configureDesc.header);

            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            unit_test::vk::BeginCommandBuffer(vk, commandBuffer, &error);
            const DumpFrameMeta&                     meta = dataset.frames[frame];
            ffxApiDispatchDescFrameGenerationPrepare prepare{};
            prepare.header.type             = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            prepare.frameID                 = frame;
            prepare.commandList             = commandBuffer;
            prepare.motionVectorScale       = {1.0f, 1.0f};
            prepare.frameTimeDelta          = 33.3f;
            prepare.cameraNear              = meta.near;
            prepare.cameraFar               = meta.far;
            prepare.cameraFovAngleVertical  = meta.fovY;
            prepare.viewSpaceToMetersFactor = 1.0f;
            prepare.depth         = unit_test::vk::MakeApiImageResource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            prepare.motionVectors = unit_test::vk::MakeApiImageResource(motion, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            std::memcpy(prepare.viewProjection, meta.vp, sizeof(prepare.viewProjection));

            ffxApiDispatchDescFrameGeneration dispatch{};
            dispatch.header.type  = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
            dispatch.commandList  = commandBuffer;
            dispatch.presentColor = unit_test::vk::MakeApiImageResource(present, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
#if NFRU_MFG_DUMP_MFG_API
            ffxApiDispatchDescFrameGenerationExtraOutputs extra{};
            extra.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_EXTRA_OUTPUTS;
            for (uint32_t k = 0; k < generated; ++k)
            {
                const FfxApiResource resource = unit_test::vk::MakeApiImageResource(outputs[k], outputUsage, outputState);
                if (k < FFX_API_FRAMEGENERATION_INLINE_OUTPUTS)
                    dispatch.outputs[k] = resource;
                else
                    extra.outputs[k - FFX_API_FRAMEGENERATION_INLINE_OUTPUTS] = resource;
            }
            if (generated > FFX_API_FRAMEGENERATION_INLINE_OUTPUTS)
                dispatch.header.pNext = &extra.header;
#else
            dispatch.outputs[0] = unit_test::vk::MakeApiImageResource(outputs[0], outputUsage, outputState);
#endif
            dispatch.numGeneratedFrames = generated;
            dispatch.reset              = frame == 0;
            dispatch.frameID            = frame;

            const ffxReturnCode_t prepareResult  = ffxDispatch(&context, &prepare.header);
            const ffxReturnCode_t dispatchResult = ffxDispatch(&context, &dispatch.header);
            const bool            submitted      = unit_test::vk::SubmitCommandBuffer(vk, commandBuffer, &error);
            if (prepareResult != FFX_API_RETURN_OK || dispatchResult != FFX_API_RETURN_OK || !submitted)
            {
                failure = "dispatch failed at frame " + std::to_string(frame);
                ok      = false;
                break;
            }

            for (uint32_t k = 0; ok && k < generated; ++k)
            {
                ok = unit_test::vk::ReadbackRgba8Image(vk, outputs[k], bytes, &error);
                std::snprintf(name, sizeof(name), "frame_%04u_k%u.rgba8", frame, k + 1);
                ok = ok && WriteFile(outDir / name, bytes.data(), bytes.size());
            }
            if (capture && qmvd.mapped)
            {
                const auto dumpSlices = [&](const HostBuffer& buffer, VkDeviceSize sliceBytes, const char* suffix) {
                    for (uint32_t k = 0; k < generated && frame > 0; ++k)
                    {
                        std::snprintf(name, sizeof(name), "frame_%04u_k%u_%s", frame, k + 1, suffix);
                        WriteFile(outDir / name, static_cast<const uint8_t*>(buffer.mapped) + k * sliceBytes, static_cast<size_t>(sliceBytes));
                    }
                };
                uint32_t channels = 16;
                for (const std::string& message : g_dumpMessages)
                {
                    const size_t at = message.find("input_tensor_channels=");
                    if (at != std::string::npos)
                        channels = static_cast<uint32_t>(std::stoul(message.substr(at + 22)));
                }
                dumpSlices(qmvd, laneTexels * 4, "qmvd.u32");
                dumpSlices(flow, flowTexels * 4, "flow.f16x2");
                if (std::getenv("FFX_NFRU_DEBUG_CAPTURE_TENSORS"))
                    dumpSlices(tensor, tensorTexels * channels, (std::string("tensor.i8x") + std::to_string(channels)).c_str());
                dumpSlices(params, paramTexels * 4, "params.i8x4");
                dumpSlices(gate, paramTexels, "gate.i8");
                dumpSlices(inpaint, flowTexels * 4, "inpaint.rgba8");
                std::snprintf(name, sizeof(name), "frame_%04u_hist_dynmask.u8", frame);
                WriteFile(outDir / name, dynmask.mapped, static_cast<size_t>(laneTexels));
                std::snprintf(name, sizeof(name), "frame_%04u_hist_depth.f32", frame);
                WriteFile(outDir / name, depthHist.mapped, static_cast<size_t>(laneTexels * 4));
                std::snprintf(name, sizeof(name), "frame_%04u_hist_colour.rgba8", frame);
                WriteFile(outDir / name, colourHist.mapped, static_cast<size_t>(dataset.displayW) * dataset.displayH * 4);
            }
            if (!ok)
                failure = "readback failed: " + error;
        }

        std::ofstream log(outDir / "sdk_log.txt");
        for (const std::string& message : g_dumpMessages)
            log << message << "\n";

#if NFRU_MFG_DUMP_MFG_API
        if (context && capture)
        {
            ffxApiConfigureDescFrameGenerationKeyValue keyValue{};
            keyValue.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE;
            keyValue.key         = FFX_API_CONFIGURE_FG_KEY_DEBUG_CAPTURE;
            keyValue.ptr         = nullptr;
            ffxConfigure(&context, &keyValue.header);
        }
#endif
        if (context)
            ffxDestroyContext(&context, nullptr);
        for (HostBuffer* buffer : {&qmvd, &flow, &tensor, &params, &gate, &inpaint, &dynmask, &depthHist, &colourHist})
        {
            if (buffer->mapped)
                vkUnmapMemory(vk.device, buffer->buffer.memory);
            unit_test::vk::DestroyBuffer(vk, buffer->buffer);
        }
        for (auto& output : outputs)
            unit_test::vk::DestroyImage(vk, output);
        unit_test::vk::DestroyImage(vk, motion);
        unit_test::vk::DestroyImage(vk, depth);
        unit_test::vk::DestroyImage(vk, present);
        unit_test::vk::DestroyVulkanContext(vk);
        return ok;
    }
}  // namespace

UNIT_TEST_CASE(nfru_mfg, gpu_dataset_dump)
{
    const char* datasetEnv = std::getenv("NFRU_MFG_DATASET");
    const char* outEnv     = std::getenv("NFRU_MFG_DUMP_DIR");
    if (!datasetEnv || !outEnv)
    {
        UNIT_TEST_SKIP("set NFRU_MFG_DATASET and NFRU_MFG_DUMP_DIR to run");
    }
    DumpDataset dataset;
    if (!LoadDataset(datasetEnv, dataset))
    {
        UNIT_TEST_FAIL(std::string("cannot read ") + datasetEnv + "/meta.txt");
    }
    const char*    pathEnv    = std::getenv("NFRU_MFG_DUMP_PATH");
    const bool     compute    = pathEnv && std::string(pathEnv) == "compute";
    const char*    captureEnv = std::getenv("NFRU_MFG_DUMP_CAPTURE");
    const bool     capture    = captureEnv && captureEnv[0] == '1';
    const char*    framesEnv  = std::getenv("NFRU_MFG_DUMP_FRAMES");
    const uint32_t frames     = std::min<uint32_t>(framesEnv ? static_cast<uint32_t>(std::stoul(framesEnv)) : static_cast<uint32_t>(dataset.frames.size()),
                                               static_cast<uint32_t>(dataset.frames.size()));
    for (const uint32_t generated : ParseCounts(std::getenv("NFRU_MFG_DUMP_N")))
    {
        std::string failure;
        const auto  outDir = std::filesystem::path(outEnv) / ("N" + std::to_string(generated));
        if (!RunDumpVariant(datasetEnv, dataset, generated, compute, capture, frames, outDir, failure))
        {
            UNIT_TEST_FAIL("N=" + std::to_string(generated) + ": " + failure);
        }
        std::printf("NFRU_MFG_DUMP N=%u frames=%u path=%s capture=%d -> %s\n",
                    generated,
                    frames,
                    compute ? "compute" : "fragment",
                    capture ? 1 : 0,
                    outDir.string().c_str());
    }
    return true;
}
