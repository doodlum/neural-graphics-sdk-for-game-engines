// Contract between the NGX proxy (a Windows DLL, calling through winevulkan) and VK_LAYER_AISS_ngx (native) for
// splitting a game's command buffer at an NGX Evaluate, so the D3D12 FSR 4 work can run between the two halves
// (see "Command buffer split" in aiss_layer.c). winevulkan does not forward private entry points, so the proxy talks to
// the layer through ordinary Vulkan calls the layer recognises and swallows:
//
//  register   vkWaitSemaphores(device, {1 timeline semaphore, value 0}, timeout = AISS_SPLIT_REGISTER | token)
//             The layer answers VK_TIMEOUT (a driver returns VK_SUCCESS: value 0 is always reached): that is how the
//             proxy knows the layer is there and splits on this device.
//  unregister the same with AISS_SPLIT_UNREGISTER | token.
//  marker     a VkBuffer created with size AISS_SPLIT_BUFFER_SIZE and usage TRANSFER_DST only (never bound or used).
//  evaluate   vkCmdFillBuffer(cmd, marker, 0, 0, AISS_SPLIT_FILL | token) before the proxy records anything: the layer
//             ends a render pass the game left open (NGX evaluates are recorded outside render passes, but engines that
//             end passes lazily, e.g. NVRHI, leave one open) and drops the game's own end of it later.
//  split      vkCmdFillBuffer(cmd, marker, dstOffset = signal value, size = wait value, data = AISS_SPLIT_FILL | token)
//             The layer ends the real command buffer here and continues the recording in a fresh one.
//  capture    right after the split, vkQueueSubmit2 on any queue of the device with one batch, no command buffers,
//             one wait {fence semaphore, wait value} and one signal {fence semaphore, signal value}, both with
//             stageMask = AISS_SPLIT_STAGE | (token << 32). The layer does not submit it: it keeps the host semaphores
//             and values winevulkan converted the D3D12-fence semaphore operations to and puts them on the cut (the
//             part before the cut signals, the part after waits). Proton's winevulkan emulates D3D12 fences with a
//             temporary host semaphore per operation and a worker thread, so a semaphore operation has to come
//             through winevulkan to take effect; this is how the layer gets ones that did.
// SPDX-License-Identifier: MIT
#pragma once

#define AISS_SPLIT_REGISTER    0x7A155E3A00000000ull
#define AISS_SPLIT_UNREGISTER  0x7A155E3B00000000ull
#define AISS_SPLIT_CMD_MASK    0xFFFFFFFF00000000ull
#define AISS_SPLIT_TOKEN_COUNT 64u
#define AISS_SPLIT_BUFFER_SIZE 0x0A155000ull
#define AISS_SPLIT_FILL        0xA1550000u
#define AISS_SPLIT_FILL_MASK   0xFFFF0000u
#define AISS_SPLIT_STAGE       0xA150000000000000ull
#define AISS_SPLIT_STAGE_MASK  0xFFF0000000000000ull
