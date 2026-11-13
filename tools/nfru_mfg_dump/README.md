<!--
/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: MIT
 */
-->

# NFRU multi-frame generation dump harness

Tools to replay a few frames of a model-gym NFRU safetensors sequence through the SDK for every generated-frame count
N = 1..5 and dump the generated frames plus per-timestep intermediates, for parity checks against a reference
implementation.

## 1. Extract frames

```bash
python -I tools/nfru_mfg_dump/extract_safetensors_frames.py \
    --safetensors <seq>.safetensors --start 2 --count 12 --out <dataset>
```

The safetensors store 60 fps entries; rendered frames are entries `start, start + 2, ...` (f30 motion spans two
entries) and the entry in between is written as the t = 0.5 ground truth. Files (index `NNNN` = rendered frame):

| File | Format | Notes |
|---|---|---|
| `color_NNNN.bin` | RGBA8 UNORM, display size | `rgb_reinhard` clamped to [0, 1], x255 rounded, alpha 255 |
| `depth_NNNN.bin` | float32, MV/depth size | SDK device depth `1 - depth` (reverse-Z, 0 = far; infinite far) |
| `mv_NNNN.bin` | float32 x2, MV/depth size | SDK backward motion in MV/depth pixels: R = channel 1, G = channel 0 of `mv_{}_f30_m1` (chosen by a warp check recorded in `meta.json`) |
| `gt_t0.5_NNNN.bin` | RGBA8 | 60 fps entry between rendered frames NNNN-1 and NNNN |
| `meta.json`, `meta.txt` | text | sizes, camera, view-projection in SDK column-major storage (`raw[c*4+r] = M[r][c]`) |

Jitter is not applied (`jitterOffset = 0`, no jitter cancellation).

## 2. Run the SDK

`nfru_mfg.gpu_dataset_dump` in `sdk_unit_test` (skipped unless the variables below are set):

| Variable | Meaning |
|---|---|
| `NFRU_MFG_DATASET` | directory from step 1 |
| `NFRU_MFG_DUMP_DIR` | output root; results go to `<root>/N<n>/` |
| `NFRU_MFG_DUMP_N` | counts to run, e.g. `1,2,3,4,5` |
| `NFRU_MFG_DUMP_PATH` | `fragment` (default stages) or `compute` (`ALL_STAGES_COMPUTE`) |
| `NFRU_MFG_DUMP_CAPTURE` | `1` dumps intermediates (requires the MFG SDK; use the compute path so every captured image is a storage image) |
| `FFX_NFRU_DEBUG_CAPTURE_TENSORS` | `1` at context creation adds a buffer alias to the input tensor so it can be dumped |
| `NFRU_MFG_DUMP_FRAMES` | number of rendered frames (default: all) |

Context: display = render = colour size, `mvDepthSize` = MV/depth size, `DEPTH_INVERTED` (+ `DEPTH_INFINITE`),
`MANAGE_PREVIOUS_DEPTH | MANAGE_PREVIOUS_COLOR`, MV hints off (SDK default), `motionVectorScale = {1, 1}`,
`frameTimeDelta = 33.3`. Frame 0 is dispatched with `reset = true`.

## 3. Output (`<root>/N<n>/`)

All files are raw, tightly packed, row-major, top row first.

| File | Format | Size |
|---|---|---|
| `index.txt` | text | N, path, display / MV-depth lane / flow lane / tensor lane / params lane sizes |
| `sdk_log.txt` | text | SDK messages with debug checking (`event=dispatch_timesteps ... timesteps=...`) |
| `frame_FFFF_kK.rgba8` | RGBA8 | display; generated frame K of N at t = K / (N + 1) after rendered frame FFFF |
| `frame_FFFF_kK_qmvd.u32` | uint32 | MV/depth lane; packed motion + depth scattered to t (pass 10 output, `quant.h` `qFloat` layout) |
| `frame_FFFF_kK_flow.f16x2` | float16 x2 | flow lane; optical flow scattered to t and hole-filled (pass 21 output), UV units |
| `frame_FFFF_kK_tensor.i8xC` | int8 x C | padded tensor lane; network input NHWC, C = 16 or 20 |
| `frame_FFFF_kK_params.i8x4` | int8 x 4 | params lane (padded tensor lane / 2); raw SINT network output 0 (blend logits) |
| `frame_FFFF_kK_gate.i8` | int8 | params lane; raw SINT network output 1 (inpaint gate) |
| `frame_FFFF_kK_inpaint.rgba8` | RGBA8 | flow lane; resolved inpaint colour (alpha = coverage) |
| `frame_FFFF_hist_dynmask.u8` | R8 | MV/depth lane; dynamic mask written this frame (becomes TM1 next frame) |
| `frame_FFFF_hist_depth.f32` | float32 | MV/depth lane; depth history written this frame |
| `frame_FFFF_hist_colour.rgba8` | RGBA8 | display; previous-colour history written this frame |

Per-timestep intermediates are not written for frame 0 (reset: the outputs are copies of the colour input). The
dequantization parameters of the network outputs are in the selected data graph's `<stem>_metadata.json`.

## 4. Check

```bash
python -I tools/nfru_mfg_dump/check_dumps.py --dumps <root> --baseline <baseline root> --dataset <dataset>
```
