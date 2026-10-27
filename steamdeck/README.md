# Steam Deck: DLSS → Arm NSS under Proton

This branch turns the SDK into a drop-in DLSS Super Resolution replacement for Steam Deck games running under Proton.
Games keep calling DLSS (directly through NGX, or through Streamline), and the frames are upscaled by Arm's Neural
Super Sampling network, running on the Deck's GPU through the native compute ML path of the `universal` branch.

## Based on OptiScaler's DLSS proxy

The design follows OptiScaler's DLSS (NGX) input path, and only that: no other upscalers, inputs, overlay or frame
generation. The pieces carried over:

- Replacing the NGX core (`nvngx.dll` / `_nvngx.dll`) with one that implements the NGX D3D12 / Vulkan / D3D11 entry points,
  answers DLSS capability and optimal-settings queries, and serves the SuperSampling feature with another upscaler.
- Getting NGX to load an unsigned core by pointing its file-signature check at an NVIDIA-signed DLL the game already ships
  (`nvngx_dlss.dll`), from a DLL loaded early as a system-DLL proxy (here `dbghelp.dll` / `dxgi.dll`).
- On Linux, leaving NVIDIA spoofing to Proton/dxvk-nvapi (`PROTON_FORCE_NVAPI=1`), as OptiScaler's Linux setup does.
- A Decky plugin that sets games up, as Decky-Framegen does for OptiScaler.

OptiScaler is GPL-3.0 and this SDK is MIT, so nothing is copied from it. The NGX interface declarations and all the code
are written from scratch.

## Steam Deck / Proton only, by design

- Each graphics API runs only on Proton's translation layers. D3D12 needs vkd3d-proton's `ID3D12DXVKInteropDevice1`, and
  D3D11 needs DXVK's `IDXGIVkInteropDevice`. Without them, NGX init returns `FAIL_PlatformError` ("this proxy only runs
  under Proton"), so the game falls back.
- The Vulkan layer is a native Linux implicit layer, and it assumes RADV for which NVIDIA extensions to emulate.
- The ML kernels are tuned for the Deck's RDNA2 GPU (Van Gogh: int8 `dot4`, wave64, 8 CUs). The runtime picks them
  from the Vulkan driver ID, because Proton reports the GPU as NVIDIA.
- Setup and launch options target Steam/Proton prefixes, driven by the Decky Loader plugin.

## How it works

| Piece | Where | What it does |
| --- | --- | --- |
| `nvngx.dll` | `steamdeck/nvngx` | Replacement NGX core. Implements the NGX D3D12, Vulkan and D3D11 entry points and runs NSS for the SuperSampling feature. Installed as `C:\windows\system32\_nvngx.dll` in the game's prefix, where NGX looks for the driver's core. |
| Loader shim | `steamdeck/loader` | `dbghelp.dll` (Streamline games) or `dxgi.dll` (others), placed next to the game. Forwards to the system DLL and redirects NGX's signature check of the core to an NVIDIA-signed DLL the game ships (its `nvngx_dlss.dll`), so the unsigned core is accepted. Only import tables are patched; `LoadLibrary` still loads the proxy. |
| Vulkan layer | `steamdeck/layer` | Native Linux implicit layer `VK_LAYER_AISS_ngx`, active only with `AISS_LAYER=1`. Emulates `VK_NVX_binary_import`, `VK_NVX_image_view_handle` and `VK_NV_low_latency(2)`, which Vulkan DLSS/Reflex integrations require, and enables the int8 / dot-product / subgroup-size features the ML kernels need on every device (DXVK's and vkd3d-proton's included). |
| Decky plugin | `steamdeck/decky` | Finds installed games that ship DLSS and sets them up (files, prefix, launch options) per game. |

Each API runs NSS on the game's own Vulkan device, with no copies:

- **D3D12**, through vkd3d-proton's `ID3D12DXVKInteropDevice1`: VkImages and layouts behind the game's resources, and the VkCommandBuffer behind the game's command list, so NSS is recorded inline in the game's frame.
- **Vulkan**: the game's VkDevice and command buffer, straight from NGX.
- **D3D11**, through DXVK's `IDXGIVkInteropDevice` / `IDXGIVkInteropSurface`: NSS goes into the proxy's own command buffer, submitted on DXVK's queue right after the game's work is flushed.

Spoofing the GPU as NVIDIA is left to Proton: `PROTON_FORCE_NVAPI=1` enables dxvk-nvapi and reports the AMD GPU as
NVIDIA. The ML runtime still picks its kernels from the real hardware (Vulkan driver ID / device name).

## Performance (Steam Deck, RDNA2)

The int8 `dot4` path has an RDNA2-specific register-tiled kernel, and the pre/post-process shaders were reworked
for the Deck, bit-exact against the original code (`ffx_nss_replay_test --suite --check`; see `docs/nss_gpu_tuning.md`).
NSS mid_low network, 640x400 → 1280x800 (DLSS Performance), per-pass GPU time in a real D3D12 frame of the Streamline
sample:

| Pass | Before tuning (µs) | Now (µs) |
| --- | --- | --- |
| Depth scatter | 32 | 30 |
| Disocclusion mask | 204 | 24 |
| Preprocess | 186 | 138 |
| ML graph | ~3000 (original dot4 kernel), 452 (first RDNA2 kernel) | 407 |
| Postprocess | 806 | 371 |
| **Total** | **1683** (with the RDNA2 ML kernel) | **973** |

Streamline sample (ray tracing on), 1280x800 fullscreen on the Deck, installed exactly as the Decky plugin does it
(its bundled binaries and launch options), frame rate over 900 frames:

| API | Native 1280x800 | DLSS MaxPerf → NSS (640x400 → 1280x800) |
| --- | --- | --- |
| D3D12 | 9.9 FPS | **29.9 FPS** |
| Vulkan | 12.1 FPS | **36.5 FPS** |

Every DLSS mode uses NSS's full network by default. On the Deck it costs about 1.3 ms more per frame than the
small one at 640x400 → 1280x800, and the small network ghosts visibly. The Decky plugin switches between them while the
game runs: it writes the choice to a control file that the proxy re-reads (`AISS_CONTROL`, 0 = full, 2 = fast,
3 = XeSS).

## Intel XeSS (alternative upscaler)

The proxy can run Intel XeSS Super Resolution (Vulkan path of `libxess.dll`, XeSS SDK 3.0.2, SR library 2.0.2) instead of
NSS: plugin "Upscaler" = XeSS, control file value 3, or `AISS_UPSCALER=xess` at launch. It switches live like the NSS
models. `libxess.dll` is loaded at runtime from the proxy's own directory (the prefix's `system32`, next to
`_nvngx.dll`) and used only through its public API; when it is missing or fails to initialise the proxy logs it and
stays on NSS.

- Inputs map as for NSS: colour, depth (passed directly, depth aspect, with the DLSS depth subrect offset), motion
  vectors (render resolution by default, `XESS_INIT_FLAG_HIGH_RES_MV` when the game does not set `MVLowRes`), jitter
  (DLSS and XeSS share the convention; checked visually, the other three sign combinations blur static detail),
  `xessSetVelocityScale` = the DLSS MV scale, reset, HDR (`LDR_INPUT_COLOR` otherwise), inverted depth. DLSS
  AutoExposure maps to `XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE`; otherwise the exposure value is 1 (or `AISS_EXPOSURE`).
- Dynamic resolution works: the per-frame render subrect is the XeSS input size. The context uses the Ultra
  Performance preset, whose input range is 1/3 of the output up to the output (presets only set the range: at a
  given input size the output is bit-identical across presets).
- Device requirements (`xessVKGetRequired*` under Proton on the Deck): no instance extensions, Vulkan 1.3,
  `VK_EXT_mutable_descriptor_type` + `mutableDescriptorType`, `shaderStorageImageWriteWithoutFormat`,
  `scalarBlockLayout`, `shaderInt8`, `shaderIntegerDotProduct`. The Deck supports all of them; the Vulkan layer enables
  them on every device when `AISS_LAYER=1` (`AISS_LAYER_XESS=0` to skip). vkd3d-proton and DXVK already enable enough
  for XeSS to run without the layer.
- On RDNA2 XeSS uses its DP4a (int8 dot product) path, not XMX. Pipelines are built up front
  (`xessVKBuildPipelines`, blocking): ~260 ms for the first context in a process, 45-110 ms for later ones. Its
  temporary memory is 28-33 MB at 1280x722-1280x800, plus its persistent history and weights.

GPU time per upscale (`AISS_GPU_TIMING=120`: timestamps around the evaluate in the game's command buffer) and frame
rate, Streamline sample (ray tracing on), Steam Deck:

| API | Mode | NSS full | NSS fast | XeSS |
| --- | --- | --- | --- | --- |
| D3D12 | 640x361 → 1280x722 (2x) | 2.20 ms, 25.4 FPS | 1.01 ms, 26.3 FPS | 2.88 ms, 24.8 FPS |
| D3D12 | 853x481 → 1280x722 (1.5x) | 3.53 ms, 17.8 FPS | 2.10 ms, 18.1 FPS | 3.02 ms, 18.0 FPS |
| Vulkan | 640x392 → 1280x785 (2x) | 2.58 ms, 26.6 FPS | 1.29 ms, 27.5 FPS | 3.19 ms, 26.2 FPS |
| Vulkan | 853x523 → 1280x785 (1.5x) | 3.79 ms, 19.2 FPS | 2.29 ms, 19.8 FPS | 3.36 ms, 19.3 FPS |

XeSS's cost depends mostly on the output size; NSS's on the input size. In screenshots XeSS is softer than NSS full
on static detail, and the moving robot shows some smearing.

Intel XeSS licence notice: the plugin ships `libxess.dll` unmodified with its licence (`libxess-LICENSE.txt`):

> Intel(R) Xe Super Sampling (XeSS) SDK: Copyright (C) 2025 Intel Corporation. Licensed under the Intel Simplified
> Software License (Version October 2022). Redistributed in binary form only, without modification; no reverse
> engineering, decompilation, disassembly, modification or alteration of the Software is permitted. Neither the name of
> Intel nor the names of its suppliers may be used to endorse or promote products derived from this Software. THE
> SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND; see the full licence text shipped with it.

The XeSS SDK is not part of this repository: the build takes its headers from `../third_party/xess/sdk` (CMake
`XESS_SDK`), and `steamdeck/decky/build.sh` copies `bin/libxess.dll` and `LICENSE.txt` from there.

## Build

Inside any Linux environment with `mingw-w64-gcc`, `cmake`, `ninja`, `make`, `glslc`, `python`, `git-lfs` (for example an
Arch distrobox: `pacman -S mingw-w64-gcc cmake ninja make shaderc python git-lfs`):

```bash
git lfs pull && git submodule update --init --recursive
steamdeck/build.sh
```

The SDK's own Linux build (`python build.py -t Release -b vk_linux_x64`) builds `ffx_ml_runtime_test`, which checks the
ML kernels bit-for-bit and times them on the Deck.

## Install for one game (manual)

Use the Decky plugin for this. By hand:

1. Copy `build_win_nvngx/nvngx.dll` to `steamapps/compatdata/<appid>/pfx/drive_c/windows/system32/_nvngx.dll`. The prefix exists after the first launch.
2. Copy the loader shim next to the game executable:
   - `build_win_loader/dbghelp/dbghelp.dll` if the game ships `sl.interposer.dll`;
   - `build_win_loader/dxgi/dxgi.dll` otherwise.
3. Copy `build_layer/libVkLayer_aiss_ngx.so` and `VkLayer_aiss_ngx.json` to `~/.local/share/vulkan/implicit_layer.d/`.
4. Launch options: `PROTON_FORCE_NVAPI=1 AISS_LAYER=1 WINEDLLOVERRIDES="dbghelp=n,b" %command%`. Use `dxgi=n,b` for the dxgi shim.
5. Enable DLSS in the game.

## Options (launch-option environment variables)

| Variable | Effect |
| --- | --- |
| `AISS_NSS_QUALITY=0/1/2` | NSS model at launch: 0 = full network (default), 1/2 = small network |
| `AISS_CONTROL=/path` | File holding 0/1/2 (NSS model) or 3 (XeSS), re-read twice a second; switches the upscaler while the game runs (set by the plugin) |
| `AISS_UPSCALER=xess` | XeSS instead of NSS at launch (the control file overrides it) |
| `AISS_XESS_QUALITY=100..105` | XeSS preset (default 100, Ultra Performance: the widest input range) |
| `AISS_GPU_TIMING=N` | Log the GPU time of the upscale and the evaluations per second, averaged over N evaluations |
| `AISS_CAMERA_NEAR`, `AISS_CAMERA_FAR`, `AISS_CAMERA_FOV` | Camera values NGX does not pass. Defaults: 0.1, 100000, 60° (reversed-Z, infinite far plane). |
| `AISS_EXPOSURE` | Exposure passed to NSS (default 1.0) |
| `AISS_LOG=0..4` | Log to `aiss_nvngx.log` next to the exe: 1 lifecycle (default), 2 per-frame, 3 NGX calls, 4 every parameter |
| `AISS_DEBUG_VIEW=0..16` | Write NSS's internal views instead of the image (0 = 4x4 grid) |
| `AISS_DUMP_FRAME=N` | Write output frame N to `aiss_frame_N.ppm` |
| `FFX_PROFILE_PASSES=file` | Per-pass GPU timings |
| `FFX_ML_RESCALE_FLOAT=1` | Fast maths: fp32 requantisation in the ML graph, ~1-2% faster, not bit exact (the plugin's "Fast maths" toggle sets it live through the control file) |
| `AISS_DUMP_INPUTS=first,count` | Raw colour/depth/MV/output per dispatch next to the game exe (the plugin's "Capture frames" button does the same for the next 8 frames of a running game, into `~/.local/share/aiss-dlss-nss/captures/`) |
| `AISS_JITTER_SIGN=x,y`, `AISS_MV_SIGN=x,y` | Multipliers on the jitter / motion vector scale handed to NSS (convention debugging) |
| `AISS_LAYER_LOG=1` | Vulkan layer: log instances, device extensions and NULL proc lookups to stderr |
| `AISS_LAYER_FEATURES=0`, `AISS_LAYER_EMULATE=0` | Vulkan layer: don't force ML features / don't emulate the NVIDIA extensions (debugging) |
| `AISS_LAYER_XESS=0` | Vulkan layer: don't enable the XeSS device features (debugging) |

## Tested

All tests use the original Streamline Sample 2.14.16 with the production Streamline plugins, cross-built with
clang-cl, under Proton Experimental:

- **D3D12** (with and without the Vulkan layer): DLSS reported supported. NSS runs inline on vkd3d-proton; the output, the debug view (depth, motion vectors, disocclusion) and the exit are clean. The layer hides its emulated NVIDIA extensions from DXVK and vkd3d-proton, which would otherwise switch on Reflex/CUDA interop paths that nothing implements.
- **Vulkan** (`-vk`): DLSS reported supported; output frame verified with `AISS_DUMP_FRAME`; clean exit.
- **D3D11**: the sample's D3D11 path needs `fxc`, so `steamdeck/tests/ngx_d3d11_test` drives the NGX D3D11 API on DXVK instead (64 evaluations, upscaled output within 0.009 mean error of the reference).
- **Inputs larger than the render size** (e.g. Galactic Racer: 428x268 colour for a 427x267 render): NSS samples colour and motion vectors with UVs from the render size, so the proxy copies the render-size region of an oversized input into an exact-size image of the same format first (depth always goes through such a copy). `ngx_d3d11_test --pad N` pads the inputs with junk and must match the unpadded run bit for bit; `--replay DIR [--crop]` replays a capture or `AISS_DUMP_INPUTS` dump, as dumped or cropped to the render size.

The sample's `SLWrapper::UpdateFeatureAvailable` reads a LUID through a pointer to a variable whose scope has ended.
MSVC builds happen to work; an optimised clang build makes Streamline see LUID 0 and report DLSS as unsupported.
Building that one file without optimisation keeps the original source unchanged.

## Limitations

- DLSS Super Resolution only. Ray Reconstruction and Frame Generation report "not supported", and games fall back.
- Colour/output subrect offsets are ignored, and display-resolution motion vectors are not supported (both logged).
- No dynamic resolution with NSS: it requires every dispatch's render size to equal the size the context was created with, so a smaller render subrect fails the dispatch. XeSS supports it.
- Online games with anti-cheat may object to the extra DLLs; don't use it there.
