/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */

// NFRU multi-frame generation (MFG): 1 to 5 generated frames per rendered frame.
//
// Host tests drive the real component prepare/dispatch logic against a stub backend and assert the per-timestep
// constants, output binding and reset copies. GPU tests run the full ffx-api path on the selected Vulkan device (use
// the ML emulation layers on devices without VK_ARM_tensors / VK_ARM_data_graph) and read the generated images back.

#include "../framework/test_framework.h"
#include "../framework/vk_test_utils.h"

#include <FidelityFX/host/backends/vk/vk_wrapper.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_framegeneration.h>
#include "../../sdk/src/components/frameinterpolation/ffx_frameinterpolation_private.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES)
#define NFRU_MFG_API_AVAILABLE 1
#else
// Building against an SDK without multi-frame generation (used to record reference hashes): N = 1 only.
#define NFRU_MFG_API_AVAILABLE 0
#endif

namespace
{
    uint64_t Fnv1a(const std::vector<uint8_t>& bytes)
    {
        uint64_t hash = 1469598103934665603ull;
        for (const uint8_t byte : bytes)
        {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    double MeanAbsDifference(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
    {
        if (a.size() != b.size() || a.empty())
        {
            return -1.0;
        }
        uint64_t sum = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            sum += static_cast<uint64_t>(std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
        }
        return static_cast<double>(sum) / static_cast<double>(a.size());
    }

    void IgnoreApiMessage(uint32_t, const char*)
    {
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // GPU scene: the same triangle in every frame, depth 0.5 (not sky) and uniform motion so that every timestep
    // samples the warped colour at a different position. Frame 1 is a reset; frames 2..frameCount interpolate.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct MfgSceneResult
    {
        // outputs[frame - 1][k - 1]: RGBA8 readback of generated frame k after rendered frame `frame`.
        std::vector<std::vector<std::vector<uint8_t>>> outputs;
        std::vector<uint8_t>                           presentColor;
        bool                                           skipped = false;
        std::string                                    message;
        std::vector<double>                            frameMilliseconds;  // prepare + dispatch + submit + wait, per frame
    };

    struct MfgSceneOptions
    {
        uint32_t displayWidth  = 192;
        uint32_t displayHeight = 192;
        uint32_t renderWidth   = 96;  // render / MV / depth
        uint32_t renderHeight  = 96;
        bool     readback    = true;
    };

    bool RunMfgScene(uint32_t generatedFrames, uint32_t extraFlags, uint32_t frameCount, MfgSceneResult& result, MfgSceneOptions options = {})
    {
        const uint32_t kDisplayW = options.displayWidth, kDisplayH = options.displayHeight;
        const uint32_t kRenderW = options.renderWidth, kRenderH = options.renderHeight;

        unit_test::vk::VulkanContext vkContext{};
        std::string                  errorMessage;
        if (!unit_test::vk::InitializeVulkanContext(vkContext, &errorMessage))
        {
            result.message = "Vulkan unavailable: " + errorMessage;
            return false;
        }

        VkDeviceContext deviceContext{};
        deviceContext.vkDevice              = vkContext.device;
        deviceContext.vkPhysicalDevice      = vkContext.physicalDevice;
        deviceContext.vkInstance            = vkContext.instance;
        deviceContext.vkDeviceProcAddr      = vkContext.getDeviceProcAddr;
        deviceContext.vkGetInstanceProcAddr = vkContext.getInstanceProcAddr;
        if (!InitVulkanWrapper(deviceContext))
        {
            unit_test::vk::DestroyVulkanContext(vkContext);
            result.message = "InitVulkanWrapper failed";
            return false;
        }

        uint32_t                 flags = FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH | FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR | extraFlags;
        VkPhysicalDeviceFeatures physicalFeatures{};
        vkGetPhysicalDeviceFeatures(vkContext.physicalDevice, &physicalFeatures);
        if (!physicalFeatures.fragmentStoresAndAtomics)
        {
            flags |= FFX_API_FG_CONTEXT_FLAG_WARP_COMPUTE | FFX_API_FG_CONTEXT_FLAG_PRE_PROCESS_COMPUTE;
        }
        const bool postprocessCompute = (flags & FFX_API_FG_CONTEXT_FLAG_POSTPROCESS_COMPUTE) != 0;

        auto                                   backendDesc = unit_test::vk::BuildBackendDesc(vkContext);
        ffxApiCreateContextDescFrameGeneration createDesc{};
        createDesc.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        createDesc.header.pNext     = reinterpret_cast<ffxApiHeader*>(&backendDesc);
        createDesc.flags            = flags;
        createDesc.displaySize      = {kDisplayW, kDisplayH};
        createDesc.renderSize       = {kRenderW, kRenderH};
        createDesc.backBufferFormat = FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
        createDesc.fpMessage        = IgnoreApiMessage;
        unit_test::vk::SetIdentityMatrix(createDesc.initialViewProjection);

        ffxContext context = nullptr;
        bool       ok      = ffxCreateContext(&context, &createDesc.header, nullptr) == FFX_API_RETURN_OK;
        if (!ok)
        {
            result.message = "ffxCreateContext failed";
        }

        unit_test::vk::Image              presentColor{}, depth{}, motionVectors{}, depthTm1{}, colorTm1{};
        std::vector<unit_test::vk::Image> outputs(generatedFrames);
        const VkImageUsageFlags           colourUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkImageUsageFlags dataUsage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        ok = ok && unit_test::vk::CreateImage2D(vkContext, kDisplayW, kDisplayH, VK_FORMAT_R8G8B8A8_UNORM, colourUsage, presentColor, &errorMessage);
        ok = ok && unit_test::vk::CreateImage2D(vkContext, kRenderW, kRenderH, VK_FORMAT_R32_SFLOAT, dataUsage, depth, &errorMessage);
        ok = ok && unit_test::vk::CreateImage2D(vkContext, kRenderW, kRenderH, VK_FORMAT_R16G16_SFLOAT, dataUsage, motionVectors, &errorMessage);
        ok = ok && unit_test::vk::CreateImage2D(vkContext, kRenderW, kRenderH, VK_FORMAT_R32_SFLOAT, dataUsage, depthTm1, &errorMessage);
        ok = ok && unit_test::vk::CreateImage2D(vkContext, kDisplayW, kDisplayH, VK_FORMAT_R8G8B8A8_UNORM, colourUsage, colorTm1, &errorMessage);
        for (auto& output : outputs)
        {
            ok = ok && unit_test::vk::CreateImage2D(vkContext, kDisplayW, kDisplayH, VK_FORMAT_R8G8B8A8_UNORM, colourUsage, output, &errorMessage);
        }
        ok = ok && unit_test::vk::RenderTriangle(vkContext, presentColor, &errorMessage);
        ok = ok && unit_test::vk::RenderTriangle(vkContext, colorTm1, &errorMessage);
        ok = ok && unit_test::vk::ClearImage(vkContext, depth, {0.5f, 0.0f, 0.0f, 0.0f}, &errorMessage);
        ok = ok && unit_test::vk::ClearImage(vkContext, depthTm1, {0.5f, 0.0f, 0.0f, 0.0f}, &errorMessage);
        // Backward motion in UV units (motionVectorScale = size): content moves 3% of the width to the left per frame.
        ok = ok && unit_test::vk::ClearImage(vkContext, motionVectors, {0.03f, 0.01f, 0.0f, 0.0f}, &errorMessage);
        ok = ok && unit_test::vk::ReadbackRgba8Image(vkContext, presentColor, result.presentColor, &errorMessage);
        if (!ok && result.message.empty())
        {
            result.message = "resource setup failed: " + errorMessage;
        }

        for (uint32_t frame = 1; ok && frame <= frameCount; ++frame)
        {
            for (auto& output : outputs)
            {
                ok = ok && unit_test::vk::ClearImage(vkContext, output, {0.0f, 0.0f, 0.0f, 1.0f}, &errorMessage);
            }

            ffxApiConfigureDescFrameGeneration configureDesc{};
            configureDesc.header.type            = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
            configureDesc.frameGenerationEnabled = true;
            configureDesc.flags                  = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
            configureDesc.frameID                = frame;
            ok                                   = ok && ffxConfigure(&context, &configureDesc.header) == FFX_API_RETURN_OK;

            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            ok                            = ok && unit_test::vk::BeginCommandBuffer(vkContext, commandBuffer, &errorMessage);
            if (!ok)
            {
                result.message = "frame setup failed: " + errorMessage;
                break;
            }

            ffxApiDispatchDescFrameGenerationPrepare prepareDesc{};
            prepareDesc.header.type             = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            prepareDesc.frameID                 = frame;
            prepareDesc.commandList             = commandBuffer;
            prepareDesc.motionVectorScale       = {static_cast<float>(kRenderW), static_cast<float>(kRenderH)};
            prepareDesc.frameTimeDelta          = 16.6f;
            prepareDesc.cameraNear              = 0.1f;
            prepareDesc.cameraFar               = 1000.0f;
            prepareDesc.cameraFovAngleVertical  = 1.2f;
            prepareDesc.viewSpaceToMetersFactor = 1.0f;
            prepareDesc.depth = unit_test::vk::MakeApiImageResource(depth, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            prepareDesc.motionVectors =
                unit_test::vk::MakeApiImageResource(motionVectors, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            prepareDesc.depthTm1 = unit_test::vk::MakeApiImageResource(depthTm1, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            prepareDesc.colorTm1 = unit_test::vk::MakeApiImageResource(colorTm1, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            unit_test::vk::SetIdentityMatrix(prepareDesc.viewProjection);

            ffxApiDispatchDescFrameGeneration dispatchDesc{};
            dispatchDesc.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
            dispatchDesc.commandList = commandBuffer;
            dispatchDesc.presentColor =
                unit_test::vk::MakeApiImageResource(presentColor, FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
#if NFRU_MFG_API_AVAILABLE
            // Frame 5 (6x) is carried by a chained descriptor; frames 1-4 use the inline outputs.
            ffxApiDispatchDescFrameGenerationExtraOutputs extraOutputs{};
            extraOutputs.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_EXTRA_OUTPUTS;
            if (generatedFrames > FFX_API_FRAMEGENERATION_INLINE_OUTPUTS)
            {
                dispatchDesc.header.pNext = &extraOutputs.header;
            }
#endif
            for (uint32_t k = 0; k < generatedFrames; ++k)
            {
                const FfxApiResource resource =
                    unit_test::vk::MakeApiImageResource(outputs[k],
                                                        postprocessCompute ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_RENDERTARGET,
                                                        postprocessCompute ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_RENDER_TARGET);
#if NFRU_MFG_API_AVAILABLE
                if (k >= FFX_API_FRAMEGENERATION_INLINE_OUTPUTS)
                {
                    extraOutputs.outputs[k - FFX_API_FRAMEGENERATION_INLINE_OUTPUTS] = resource;
                    continue;
                }
#endif
                dispatchDesc.outputs[k] = resource;
            }
            dispatchDesc.numGeneratedFrames = generatedFrames;
            dispatchDesc.reset              = frame == 1;
            dispatchDesc.frameID            = frame;

            const auto            frameStart     = std::chrono::steady_clock::now();
            const ffxReturnCode_t prepareResult  = ffxDispatch(&context, &prepareDesc.header);
            const ffxReturnCode_t dispatchResult = ffxDispatch(&context, &dispatchDesc.header);
            const bool            submitted      = unit_test::vk::SubmitCommandBuffer(vkContext, commandBuffer, &errorMessage);
            result.frameMilliseconds.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count());
            if (prepareResult != FFX_API_RETURN_OK || dispatchResult != FFX_API_RETURN_OK || !submitted)
            {
                char message[160];
                std::snprintf(message, sizeof(message), "frame %u: prepare=%d dispatch=%d submit=%d", frame, prepareResult, dispatchResult, submitted ? 1 : 0);
                result.message = message;
                ok             = false;
                break;
            }

            std::vector<std::vector<uint8_t>> frameOutputs(options.readback ? generatedFrames : 0);
            for (uint32_t k = 0; ok && options.readback && k < generatedFrames; ++k)
            {
                ok = unit_test::vk::ReadbackRgba8Image(vkContext, outputs[k], frameOutputs[k], &errorMessage);
            }
            if (!ok)
            {
                result.message = "readback failed: " + errorMessage;
                break;
            }
            result.outputs.push_back(std::move(frameOutputs));
        }

        for (auto& output : outputs)
        {
            unit_test::vk::DestroyImage(vkContext, output);
        }
        unit_test::vk::DestroyImage(vkContext, colorTm1);
        unit_test::vk::DestroyImage(vkContext, depthTm1);
        unit_test::vk::DestroyImage(vkContext, motionVectors);
        unit_test::vk::DestroyImage(vkContext, depth);
        unit_test::vk::DestroyImage(vkContext, presentColor);
        if (context)
        {
            ffxDestroyContext(&context, nullptr);
        }
        unit_test::vk::DestroyVulkanContext(vkContext);
        return ok;
    }

    bool HasNonClearPixel(const std::vector<uint8_t>& rgba)
    {
        for (size_t i = 0; i + 3 < rgba.size(); i += 4)
        {
            if (rgba[i] > 8 || rgba[i + 1] > 8 || rgba[i + 2] > 8)
            {
                return true;
            }
        }
        return false;
    }
}  // namespace

// Reference for the single-generated-frame (2x) path. Prints a hash of every output so the same scene can be compared
// with an SDK build that predates multi-frame generation; N = 1 must stay bit-identical.
UNIT_TEST_CASE(nfru_mfg, gpu_single_frame_reference_hash)
{
    MfgSceneResult result;
    if (!RunMfgScene(1, 0, 3, result))
    {
        UNIT_TEST_FAIL(result.message);
    }
    for (size_t frame = 0; frame < result.outputs.size(); ++frame)
    {
        std::printf("NFRU_MFG_REFERENCE frame=%zu k=1 hash=0x%016" PRIx64 "\n", frame + 1, Fnv1a(result.outputs[frame][0]));
    }
    UNIT_TEST_CHECK(result.outputs.size() == 3);
    UNIT_TEST_CHECK(result.outputs[0][0] == result.presentColor);
    UNIT_TEST_CHECK(HasNonClearPixel(result.outputs[2][0]));
    return true;
}

#if NFRU_MFG_API_AVAILABLE

namespace
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Host fixture: real component prepare/dispatch, stub backend. Mirrors the jitter fixture in nfru_suite.cpp.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct MfgHostFixture
    {
        struct CopyJob
        {
            int32_t src;
            int32_t dst;
        };

        std::unique_ptr<FfxFrameInterpolationContext> context = std::make_unique<FfxFrameInterpolationContext>();
        FfxFrameInterpolationPrepareDescription       prepare{};
        uint32_t                                      registrations = FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT;
        std::vector<std::pair<const void*, int32_t>>  registered;
        std::vector<float>                            stagedTimesteps;
        std::vector<uint32_t>                         stagedInputChannels;
        std::vector<CopyJob>                          copies;
        std::vector<int32_t>                          postprocessTargets;
        uint32_t                                      executeCalls      = 0;
        uint32_t                                      scheduledJobs     = 0;
        uint32_t                                      maxJobsPerExecute = 0;
        uint32_t                                      jobsSinceExecute  = 0;

        static constexpr uintptr_t kPostprocessPipeline = 0x5050;

        explicit MfgHostFixture(uint32_t inputTensorChannels = FFX_FRAMEINTERPOLATION_INPUT_TENSOR_CHANNELS_BASE)
        {
            const FfxDimensions2D render{96, 96}, display{192, 192};
            auto&                 state            = *reinterpret_cast<FfxArmFrameInterpolationContext_Private*>(context.get());
            state.contextDescription.maxRenderSize = render;
            state.contextDescription.mvDepthSize   = render;
            state.contextDescription.displaySize   = display;
            state.resolutionPlan                   = ffxNfruDeriveResolutionPlan(display, render, FFX_FRAMEINTERPOLATION_DATA_GRAPH_ALIGNMENT);
            state.mvDepthLane                      = state.resolutionPlan.mvDepthLane;
            state.flowLane                         = state.resolutionPlan.flowLane;
            state.enableDataGraphFI                = false;
            state.inputTensorChannels              = inputTensorChannels;
            state.constants._InputTensorChannels   = inputTensorChannels;
            state.constants._RenderSize[0] = state.constants._RenderSize[1] = 96;
            state.constants._DisplaySize[0] = state.constants._DisplaySize[1] = 192;
            state.lastFrameViewProjection[0] = state.lastFrameViewProjection[5] = state.lastFrameViewProjection[10] = state.lastFrameViewProjection[15] = 1.0f;
            for (uint32_t i = 0; i < FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT; ++i)
            {
                state.srvResources[i].internalIndex = static_cast<int32_t>(i + 1);
                state.uavResources[i]               = state.srvResources[i];
            }
            state.pipelineDownsampleInpaintSeed.uavTextureCount = 1;
            for (FfxPipelineState& mipPipeline : state.pipelineDownsampleInpaintMip)
            {
                mipPipeline.srvTextureCount = 1;
                mipPipeline.uavTextureCount = 1;
            }
            // The default (fragment) postprocess writes OUTPUT as its render target; tag the job so the test can see
            // which output each timestep renders into.
            state.pipelinePostprocess.pipeline                         = reinterpret_cast<FfxPipeline>(kPostprocessPipeline);
            state.pipelinePostprocess.rtCount                          = 1;
            state.pipelinePostprocess.rtBindings[0].resourceIdentifier = FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT;

            auto& backend              = state.contextDescription.backendInterface;
            backend.scratchBuffer      = this;
            backend.fpRegisterResource = [](FfxInterface* iface, const FfxResource* resource, FfxUInt32, FfxResourceInternal* out) -> FfxErrorCode {
                auto& self         = *static_cast<MfgHostFixture*>(iface->scratchBuffer);
                out->internalIndex = static_cast<int32_t>(++self.registrations);
                self.registered.emplace_back(resource->resource, out->internalIndex);
                return FFX_OK;
            };
            backend.fpStageConstantBufferDataFunc = [](FfxInterface* iface, void* data, FfxUInt32 size, FfxConstantBuffer*) -> FfxErrorCode {
                auto& self = *static_cast<MfgHostFixture*>(iface->scratchBuffer);
                if (size != sizeof(FrameInterpolationConstants))
                    return FFX_ERROR_INVALID_ARGUMENT;
                FrameInterpolationConstants constants{};
                std::memcpy(&constants, data, size);
                self.stagedTimesteps.push_back(constants._Timestep);
                self.stagedInputChannels.push_back(constants._InputTensorChannels);
                return FFX_OK;
            };
            backend.fpScheduleGpuJob = [](FfxInterface* iface, const FfxGpuJobDescription* job) -> FfxErrorCode {
                auto& self = *static_cast<MfgHostFixture*>(iface->scratchBuffer);
                ++self.scheduledJobs;
                self.maxJobsPerExecute = std::max(self.maxJobsPerExecute, ++self.jobsSinceExecute);
                if (job->jobType == FFX_GPU_JOB_COPY)
                {
                    self.copies.push_back({job->copyJobDescriptor.src.internalIndex, job->copyJobDescriptor.dst.internalIndex});
                }
                if (job->jobType == FFX_GPU_JOB_FRAGMENT && job->fragmentJobDescriptor.pipeline.pipeline == reinterpret_cast<FfxPipeline>(kPostprocessPipeline))
                {
                    self.postprocessTargets.push_back(job->fragmentJobDescriptor.rtTextures[0].resource.internalIndex);
                }
                return FFX_OK;
            };
            backend.fpExecuteGpuJobs = [](FfxInterface* iface, FfxCommandList, FfxUInt32) -> FfxErrorCode {
                auto& self = *static_cast<MfgHostFixture*>(iface->scratchBuffer);
                ++self.executeCalls;
                self.jobsSinceExecute = 0;
                return FFX_OK;
            };
            backend.fpUnregisterResources = [](FfxInterface*, FfxCommandList, FfxUInt32) -> FfxErrorCode { return FFX_OK; };

            prepare.commandList            = this;
            prepare.cameraNear             = 0.1f;
            prepare.cameraFar              = 1000.0f;
            prepare.cameraFovAngleVertical = 1.2f;
            prepare.motionVectorScale      = {1.0f, 1.0f};
            prepare.viewProjection[0] = prepare.viewProjection[5] = prepare.viewProjection[10] = prepare.viewProjection[15] = 1.0f;
            prepare.depth.resource                                                                                          = this;
            prepare.depth.description.width                                                                                 = render.width;
            prepare.depth.description.height                                                                                = render.height;
            prepare.motionVectors                                                                                           = prepare.depth;
            prepare.depthTm1                                                                                                = prepare.depth;
            prepare.depthTm1.description.width                                                                              = state.mvDepthLane.width;
            prepare.depthTm1.description.height                                                                             = state.mvDepthLane.height;
            prepare.depthTm1Next                                                                                            = prepare.depthTm1;
        }

        void ResetRecords()
        {
            registered.clear();
            stagedTimesteps.clear();
            stagedInputChannels.clear();
            copies.clear();
            postprocessTargets.clear();
            executeCalls      = 0;
            scheduledJobs     = 0;
            maxJobsPerExecute = 0;
            jobsSinceExecute  = 0;
        }

        int32_t InternalIndexOf(const void* resource) const
        {
            for (auto it = registered.rbegin(); it != registered.rend(); ++it)
            {
                if (it->first == resource)
                {
                    return it->second;
                }
            }
            return -1;
        }

        // Distinct fake handles: backbuffer 0x100, outputs 0x200 + k.
        static void* Handle(uintptr_t value)
        {
            return reinterpret_cast<void*>(value);
        }

        FfxFrameInterpolationDispatchDescription MakeDispatch(uint64_t frameID, uint32_t generatedFrames, bool reset) const
        {
            FfxFrameInterpolationDispatchDescription dispatch{};
            dispatch.commandList                = const_cast<MfgHostFixture*>(this);
            dispatch.frameID                    = frameID;
            dispatch.reset                      = reset;
            dispatch.numGeneratedFrames         = generatedFrames;
            dispatch.currentBackBuffer.resource = Handle(0x100);
            dispatch.output.resource            = Handle(0x201);
            for (uint32_t k = 2; k <= FFX_FRAMEINTERPOLATION_MAX_GENERATED_FRAMES; ++k)
            {
                dispatch.additionalOutputs[k - 2].resource = k <= generatedFrames ? Handle(0x200 + k) : nullptr;
            }
            return dispatch;
        }

        FfxErrorCode Frame(uint64_t frameID, uint32_t generatedFrames, bool reset)
        {
            prepare.frameID = frameID;
            FFX_VALIDATE(ffxFrameInterpolationPrepare(context.get(), &prepare));
            ResetRecords();
            const FfxFrameInterpolationDispatchDescription dispatch = MakeDispatch(frameID, generatedFrames, reset);
            return ffxFrameInterpolationDispatch(context.get(), &dispatch);
        }
    };
}  // namespace

// t_k = k / (N + 1) from the previous frame, in presentation order; N = 1 keeps exactly 0.5.
UNIT_TEST_CASE(nfru_mfg, timestep_per_generated_frame)
{
    UNIT_TEST_CHECK(ffxFrameInterpolationTimestep(1, 1) == 0.5f);
    for (uint32_t generated = 1; generated <= FFX_FRAMEINTERPOLATION_MAX_GENERATED_FRAMES; ++generated)
    {
        MfgHostFixture fixture;
        UNIT_TEST_CHECK(fixture.Frame(1, generated, true) == FFX_OK);
        UNIT_TEST_CHECK(fixture.Frame(2, generated, false) == FFX_OK);

        // One staged constant block per generated frame, each carrying its own timestep.
        UNIT_TEST_CHECK(fixture.stagedTimesteps.size() == generated);
        for (uint32_t k = 1; k <= generated; ++k)
        {
            const float expected = static_cast<float>(k) / static_cast<float>(generated + 1);
            UNIT_TEST_CHECK(fixture.stagedTimesteps[k - 1] == expected);
            UNIT_TEST_CHECK(std::abs(fixture.stagedTimesteps[k - 1] - static_cast<float>(static_cast<double>(k) / (generated + 1))) == 0.0f);
            UNIT_TEST_CHECK(fixture.stagedTimesteps[k - 1] > 0.0f && fixture.stagedTimesteps[k - 1] < 1.0f);
        }
        if (generated == 1)
        {
            UNIT_TEST_CHECK(fixture.stagedTimesteps[0] == 0.5f);
        }
        // Each timestep is recorded separately, keeping every execute within the backend job budget.
        UNIT_TEST_CHECK(fixture.executeCalls == generated);
        UNIT_TEST_CHECK(fixture.maxJobsPerExecute <= FFX_MAX_GPU_JOBS);
    }
    return true;
}

// Generated frame k renders into outputs[k - 1]: output for k = 1, additionalOutputs[k - 2] otherwise.
UNIT_TEST_CASE(nfru_mfg, each_timestep_writes_its_own_output)
{
    for (uint32_t generated = 1; generated <= FFX_FRAMEINTERPOLATION_MAX_GENERATED_FRAMES; ++generated)
    {
        MfgHostFixture fixture;
        UNIT_TEST_CHECK(fixture.Frame(1, generated, true) == FFX_OK);
        UNIT_TEST_CHECK(fixture.Frame(2, generated, false) == FFX_OK);
        UNIT_TEST_CHECK(fixture.postprocessTargets.size() == generated);
        for (uint32_t k = 1; k <= generated; ++k)
        {
            const int32_t expected = fixture.InternalIndexOf(MfgHostFixture::Handle(0x200 + k));
            UNIT_TEST_CHECK(expected > 0);
            UNIT_TEST_CHECK(fixture.postprocessTargets[k - 1] == expected);
        }
    }
    return true;
}

// Reset (first dispatch, explicit reset, or a frame-ID discontinuity): every output is a copy of presentColor.
UNIT_TEST_CASE(nfru_mfg, reset_copies_present_color_to_every_output)
{
    for (uint32_t generated = 1; generated <= FFX_FRAMEINTERPOLATION_MAX_GENERATED_FRAMES; ++generated)
    {
        for (const int scenario : {0, 1, 2})  // 0: first dispatch, 1: explicit reset, 2: skipped frame ID
        {
            MfgHostFixture fixture;
            if (scenario == 0)
            {
                UNIT_TEST_CHECK(fixture.Frame(1, generated, false) == FFX_OK);
            }
            else
            {
                UNIT_TEST_CHECK(fixture.Frame(1, generated, true) == FFX_OK);
                UNIT_TEST_CHECK(fixture.Frame(scenario == 1 ? 2 : 5, generated, scenario == 1) == FFX_OK);
            }

            const int32_t present = fixture.InternalIndexOf(MfgHostFixture::Handle(0x100));
            UNIT_TEST_CHECK(present > 0);
            // No interpolation work on reset; one staged block (t_1) for the history-seeding warp.
            UNIT_TEST_CHECK(fixture.postprocessTargets.empty());
            UNIT_TEST_CHECK(fixture.stagedTimesteps.size() == 1);
            for (uint32_t k = 1; k <= generated; ++k)
            {
                const int32_t output = fixture.InternalIndexOf(MfgHostFixture::Handle(0x200 + k));
                const bool    copied = std::any_of(fixture.copies.begin(), fixture.copies.end(), [&](const MfgHostFixture::CopyJob& copy) {
                    return copy.src == present && copy.dst == output;
                });
                UNIT_TEST_CHECK(copied);
            }
        }
    }
    return true;
}

// numGeneratedFrames: 0 keeps the single-frame default at the component level, > 5 is rejected, and every generated
// frame needs an output.
UNIT_TEST_CASE(nfru_mfg, component_validates_generated_frame_count_and_outputs)
{
    MfgHostFixture fixture;
    UNIT_TEST_CHECK(fixture.Frame(1, 1, true) == FFX_OK);
    UNIT_TEST_CHECK(fixture.Frame(2, 0, false) == FFX_OK);
    UNIT_TEST_CHECK(fixture.stagedTimesteps.size() == 1 && fixture.stagedTimesteps[0] == 0.5f);

    fixture.prepare.frameID = 3;
    UNIT_TEST_CHECK(ffxFrameInterpolationPrepare(fixture.context.get(), &fixture.prepare) == FFX_OK);
    FfxFrameInterpolationDispatchDescription tooMany = fixture.MakeDispatch(3, FFX_FRAMEINTERPOLATION_MAX_GENERATED_FRAMES, false);
    tooMany.numGeneratedFrames                       = FFX_FRAMEINTERPOLATION_MAX_GENERATED_FRAMES + 1;
    UNIT_TEST_CHECK(ffxFrameInterpolationDispatch(fixture.context.get(), &tooMany) == FFX_ERROR_INVALID_ARGUMENT);

    FfxFrameInterpolationDispatchDescription missingOutput = fixture.MakeDispatch(3, 4, false);
    missingOutput.additionalOutputs[2].resource            = nullptr;  // frame 4
    UNIT_TEST_CHECK(ffxFrameInterpolationDispatch(fixture.context.get(), &missingOutput) == FFX_ERROR_INVALID_POINTER);

    // A rejected dispatch leaves the prepared frame dispatchable.
    fixture.ResetRecords();
    const FfxFrameInterpolationDispatchDescription valid = fixture.MakeDispatch(3, 4, false);
    UNIT_TEST_CHECK(ffxFrameInterpolationDispatch(fixture.context.get(), &valid) == FFX_OK);
    UNIT_TEST_CHECK(fixture.stagedTimesteps.size() == 4);
    return true;
}

// The input-tensor layout constant reaches every staged block (16: original graph, 20: timestep channel).
UNIT_TEST_CASE(nfru_mfg, input_tensor_channels_reach_shader_constants)
{
    for (const uint32_t channels : {static_cast<uint32_t>(FFX_FRAMEINTERPOLATION_INPUT_TENSOR_CHANNELS_BASE),
                                    static_cast<uint32_t>(FFX_FRAMEINTERPOLATION_INPUT_TENSOR_CHANNELS_TIMESTEP)})
    {
        MfgHostFixture fixture(channels);
        UNIT_TEST_CHECK(fixture.Frame(1, 3, true) == FFX_OK);
        UNIT_TEST_CHECK(fixture.Frame(2, 3, false) == FFX_OK);
        UNIT_TEST_CHECK(fixture.stagedInputChannels.size() == 3);
        for (const uint32_t staged : fixture.stagedInputChannels)
        {
            UNIT_TEST_CHECK(staged == channels);
        }
        uint32_t queried = 0;
        UNIT_TEST_CHECK(ffxFrameInterpolationGetInputTensorChannels(fixture.context.get(), &queried) == FFX_OK);
        UNIT_TEST_CHECK(queried == channels);
    }
    return true;
}

namespace
{
    // Full multi-frame scene check shared by the fragment and compute variants.
    bool CheckGpuMultiFrame(uint32_t extraFlags)
    {
        constexpr uint32_t kFrames = 3;
        MfgSceneResult     single, multi;
        if (!RunMfgScene(1, extraFlags, kFrames, single))
        {
            UNIT_TEST_FAIL("N=1 scene: " + single.message);
        }
        if (!RunMfgScene(FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES, extraFlags, kFrames, multi))
        {
            UNIT_TEST_FAIL("N=5 scene: " + multi.message);
        }
        UNIT_TEST_CHECK(single.outputs.size() == kFrames && multi.outputs.size() == kFrames);

        // Reset frame: every generated output is an exact copy of presentColor.
        for (uint32_t k = 0; k < FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES; ++k)
        {
            UNIT_TEST_CHECK(multi.outputs[0][k] == multi.presentColor);
        }

        for (uint32_t frame = 1; frame < kFrames; ++frame)
        {
            const auto& outputs = multi.outputs[frame];
            for (uint32_t k = 0; k < outputs.size(); ++k)
            {
                UNIT_TEST_CHECK(HasNonClearPixel(outputs[k]));
                std::printf("NFRU_MFG frame=%u N=5 k=%u t=%.4f hash=0x%016" PRIx64 " mad_vs_present=%.3f mad_vs_N1=%.3f\n",
                            frame + 1,
                            k + 1,
                            ffxFrameInterpolationTimestep(k + 1, 5),
                            Fnv1a(outputs[k]),
                            MeanAbsDifference(outputs[k], multi.presentColor),
                            MeanAbsDifference(outputs[k], single.outputs[frame][0]));
            }
            // t_3 of N = 5 is exactly 0.5: it must reproduce the single-frame output bit for bit, which also shows
            // that repeating the per-timestep passes leaves the per-frame state intact.
            UNIT_TEST_CHECK(outputs[2] == single.outputs[frame][0]);
        }
        // The timesteps sample different positions of the moving content.
        UNIT_TEST_CHECK(multi.outputs[kFrames - 1][0] != multi.outputs[kFrames - 1][4]);
        return true;
    }
}  // namespace

UNIT_TEST_CASE(nfru_mfg, gpu_five_generated_frames_fragment)
{
    return CheckGpuMultiFrame(0);
}

UNIT_TEST_CASE(nfru_mfg, gpu_five_generated_frames_compute)
{
    return CheckGpuMultiFrame(FFX_API_FG_CONTEXT_FLAG_ALL_STAGES_COMPUTE);
}

// Every multiplier 2x..6x dispatches and produces written outputs.
UNIT_TEST_CASE(nfru_mfg, gpu_every_multiplier_dispatches)
{
    for (uint32_t generated = 1; generated <= FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES; ++generated)
    {
        MfgSceneResult result;
        if (!RunMfgScene(generated, 0, 2, result))
        {
            UNIT_TEST_FAIL("N=" + std::to_string(generated) + ": " + result.message);
        }
        for (const auto& output : result.outputs[1])
        {
            UNIT_TEST_CHECK(HasNonClearPixel(output));
        }
    }
    return true;
}

// Wall-clock cost per generated frame (prepare + dispatch + submit + wait) at Steam Deck resolution. Opt-in with
// NFRU_MFG_BENCHMARK=1: under the ML emulation layer the data graph is emulated, so the absolute numbers are not
// representative of hardware, only the fixed / per-timestep split.
UNIT_TEST_CASE(nfru_mfg, gpu_benchmark_cost_per_generated_frame)
{
    const char* enabled = std::getenv("NFRU_MFG_BENCHMARK");
    if (!enabled || std::string(enabled) != "1")
    {
        UNIT_TEST_SKIP("set NFRU_MFG_BENCHMARK=1 to run");
    }
    // Steam Deck panel resolution; the 540p internal lane cap applies (preprocess lane 864x540).
    MfgSceneOptions options{};
    options.displayWidth  = 1280;
    options.displayHeight = 800;
    options.renderWidth   = 1280;
    options.renderHeight  = 800;
    options.readback    = false;
    double medians[FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES + 1] = {};
    for (uint32_t generated = 1; generated <= FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES; ++generated)
    {
        MfgSceneResult result;
        if (!RunMfgScene(generated, 0, 12, result, options))
        {
            UNIT_TEST_FAIL(result.message);
        }
        std::vector<double> steady(result.frameMilliseconds.begin() + 2, result.frameMilliseconds.end());
        std::sort(steady.begin(), steady.end());
        medians[generated] = steady[steady.size() / 2];
        std::printf("NFRU_MFG_BENCH size=1280x800 N=%u multiplier=%ux median_ms=%.3f min_ms=%.3f\n", generated, generated + 1, medians[generated], steady.front());
    }
    // Least-squares fit T(N) = F + N * C over N = 1..5.
    double sumN = 0, sumT = 0, sumNN = 0, sumNT = 0;
    for (uint32_t n = 1; n <= FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES; ++n)
    {
        sumN += n;
        sumT += medians[n];
        sumNN += double(n) * n;
        sumNT += n * medians[n];
    }
    const double count     = FFX_API_FRAMEGENERATION_MAX_GENERATED_FRAMES;
    const double perFrame  = (count * sumNT - sumN * sumT) / (count * sumNN - sumN * sumN);
    const double fixedCost = (sumT - perFrame * sumN) / count;
    std::printf("NFRU_MFG_BENCH fit fixed_ms=%.3f per_generated_frame_ms=%.3f\n", fixedCost, perFrame);
    return true;
}

#endif  // NFRU_MFG_API_AVAILABLE
