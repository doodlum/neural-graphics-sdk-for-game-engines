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
3 = XeSS, 4-6 = FSR 4).

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

## AMD FSR 4 (alternative upscaler)

The proxy can also run AMD FSR 4.1.2 from `amdvkffx64.dll`, the FSR 4 component of AMD's Windows Vulkan driver
(Adrenalin 26.10.1). The plugin's "Upscaler" dropdown offers **FSR 4 Lite**, **FSR 4 INT8** and **FSR 4 INT8 RDNA 2**.
The control file values are 4, 5 and 6. The launch options are `AISS_UPSCALER=fsr4` with `AISS_FSR4_MODEL=lite|int8|rdna2|auto`,
or `AISS_UPSCALER=fsr4-lite` and so on. FSR 4 switches live like the other upscalers. The DLL is loaded at runtime
from the proxy's own directory (the prefix's `system32`, next to `_nvngx.dll`). When it is missing, hands out no
provider, or refuses the requested network, the proxy logs the reason and stays on NSS, with output bit-identical to
plain NSS (checked with the Streamline replay).

### How it is driven

The proxy uses the DLL only through the FidelityFX API provider it exports (`UpdateFfxApiProvider`) and AMD's public
FidelityFX SDK 2.3 types (`steamdeck/nvngx/amd_ffx/`, MIT). The file on disk is never changed. Four things are done
in memory, and they are what make it work under Proton on a Steam Deck:

1. **SDK module check.** The DLL hands out providers only when it finds signed FidelityFX SDK modules
   (`amd_fidelityfx_loader_vk.dll`, `..._upscaler_vk.dll`, `..._framegeneration_vk.dll`) through `GetModuleHandleW`.
   The proxy points the DLL's `GetModuleHandleW` import at a lookup that answers those names with the DLL itself.
   AMD signed it with the same certificate, and Wine's `WinVerifyTrust` accepts the signature.
2. **GPU family.** AMD's device checks read the GPU family and revision from the Windows kernel driver
   (`D3DKMTEnumAdapters` + `D3DKMTQueryAdapterInfo`, AMD private data, looked up with `GetProcAddress`). Wine has no
   answer to that query, so the DLL refuses every FSR 4 network on its own. This is the same problem FSR4Injector
   found for the D3D12 DLL. The proxy points the DLL's `GetProcAddress` import at a lookup that returns its own
   D3DKMT pair, which answers with the amdgpu family and external revision: 144 / 1 on Van Gogh (both Steam Deck
   models; read with `DRM_IOCTL_AMDGPU_INFO` on an OLED; `AISS_FSR4_GPU=family,revision` overrides it).
3. **Model selection.** Each network is its own provider class in the DLL (RTTI `ffxProvider_FSR4`,
   `..._FSR4_Int8`, `..._FSR4_Int8_Navi2`, `..._FSR4Lite_Int8`). Each class has a device check, which is the third
   virtual function. FidelityFX asks FP8, INT8, INT8 RDNA 2 and then Lite, and runs the first network that accepts.
   AMD's checks accept Lite for family 144 revision 1-0x7F (and Strix Point), INT8 for RDNA 3 / Strix Halo, INT8 RDNA 2
   for family 143 revision 0x28-0x4F, and FP8 for RDNA 4. The proxy wraps those vtable entries: with a model selected,
   only that network's check accepts; with `auto`, AMD's own answer goes through unchanged. AMD's checks run only for
   an AMD vendor ID, and under `PROTON_FORCE_NVAPI=1` the Vulkan device reports NVIDIA (10de), so they accept nothing.
   In that case `auto` falls back to the network AMD picks for the family, which is Lite on the Deck. Without the
   spoof (the D3D11 harness), AMD's checks pick Lite by themselves.
4. **Shaders.** FSR 4's SPIR-V declares images with 16-bit float texels, which only AMD's own driver accepts. RADV
   crashes compiling the pipeline (`vkCreateComputePipelines` access violation). The DLL gets a `vkGetDeviceProcAddr`
   whose `vkCreateShaderModule` rewrites those images to 32-bit float texels, with `OpFConvert` after each read and
   before each write (`spirv_half_images.cpp`; the results pass `spirv-val --target-env vulkan1.3`).

Proof of which network ran:

- The log names the provider (`"4.1.2 *"`), the network that accepted, and AMD's own choice.
- `AISS_FSR4_WATERMARK=1` (or `FSR_WATERMARK=1`) sets `MLSR-WATERMARK`, the variable this DLL reads, and the DLL then
  draws its banner on the output. The banners seen on the Deck were `FSR4L-i8` (Lite), `FSR4-i8` (INT8) and `FSR4-i8-2`
  (INT8 RDNA 2), each followed by `Upscale 4.1.2 DRS 2.00x Source: Driver`.
- The networks also differ in their pipelines: Lite builds 13 shaders (4 network passes), and INT8 and INT8 RDNA 2
  build 30 each (13 passes), with different code (`AISS_FSR4_DUMP_SPIRV=<dir>` writes them out).

### Inputs and device features

- Inputs are colour, depth (passed directly), motion vectors with the DLSS MV scale, and jitter. DLSS and FSR share
  the jitter convention; this was checked visually, and the other three sign combinations leave jagged edges on static
  detail (`testlogs/fsr4/jitter_sign_grid.png`). Reset, HDR, inverted depth (near and far swapped, the FidelityFX
  convention), display-resolution motion vectors when `MVLowRes` is not set, and sharpening off are also passed.
- FSR 4 needs an exposure. DLSS AutoExposure maps to FidelityFX auto exposure; otherwise the proxy passes a 1x1
  exposure texture holding 1 (or `AISS_EXPOSURE`).
- Dynamic resolution works. The context is created with `FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION`, and every dispatch
  passes the DLSS render subrect as `renderSize` (`ngx_d3d11_test --dyn` passes with all three networks).
- The shaders use `SPV_KHR_integer_dot_product` (DotProductInput4x8BitPacked), `Int8`, `Float16`,
  `SPV_KHR_compute_shader_derivatives` (DerivativeGroupLinear), `GroupNonUniformQuad`,
  `StorageImageWriteWithoutFormat` and `StorageImageExtendedFormats`. With `AISS_LAYER=1` the Vulkan layer now also
  enables `VK_KHR_compute_shader_derivatives` with `computeDerivativeGroupLinear`, and
  `shaderStorageImageExtendedFormats`, alongside the int8 / fp16 / dot-product features it already enabled
  (`AISS_LAYER_FSR4=0` skips the FSR 4 additions). RADV runs the shaders without the layer too.
- The same device and command buffer path as NSS and XeSS is used on D3D12 (vkd3d-proton interop), Vulkan and D3D11
  (DXVK interop, the proxy's own command buffer).

### Performance

GPU time per upscale (`AISS_GPU_TIMING=120`) and frame rate, Streamline sample (ray tracing on), Steam Deck, with
`amdvkffx64.dll` (Vulkan DLL path) on every API:

| API | Mode | NSS full | NSS fast | XeSS | FSR 4 Lite | FSR 4 INT8 | FSR 4 INT8 RDNA 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D3D12 | 640x361 → 1280x722 (2x) | 2.21 ms, 24.9 FPS | 1.01 ms, 25.8 FPS | 2.88 ms, 24.7 FPS | 6.69 ms, 22.8 FPS | 11.33 ms, 20.5 FPS | 9.48 ms, 21.3 FPS |
| D3D12 | 853x481 → 1280x722 (1.5x) | 3.53 ms, 17.7 FPS | 2.10 ms, 18.2 FPS | 3.03 ms, 17.8 FPS | 7.02 ms, 16.6 FPS | 11.54 ms, 15.4 FPS | 9.62 ms, 15.9 FPS |
| Vulkan | 640x392 → 1280x785 (2x) | 2.58 ms, 27.5 FPS | 1.29 ms, 27.3 FPS | 3.19 ms, 26.1 FPS | 7.34 ms, 23.8 FPS | 12.34 ms, 21.0 FPS | 10.20 ms, 22.0 FPS |
| Vulkan | 853x523 → 1280x785 (1.5x) | 3.80 ms, 18.7 FPS | 2.29 ms, 19.6 FPS | 3.36 ms, 19.1 FPS | not measured | 12.98 ms, 15.5 FPS | not measured |

On the Deck FSR 4 costs several times more than NSS or XeSS: Lite about 7 ms, INT8 RDNA 2 about 9.5-10 ms and INT8
about 11.5-13 ms per upscale. Its cost depends little on the input size.

### Limitations

- **FP8 cannot run on the Deck.** Its shaders need `SPV_EXT_float8`, `SPV_KHR_cooperative_matrix` and
  `SPV_NV_cooperative_matrix2`, and RADV on Van Gogh has neither `VK_EXT_shader_float8` nor
  `VK_KHR_cooperative_matrix`. Forcing it crashes in pipeline creation, so the proxy refuses it unless the device has
  both extensions (`AISS_FSR4_FORCE_FP8=1` overrides that). The control file value is 7 and it is not offered in the
  plugin.
- INT8 and INT8 RDNA 2 run on the Deck only because the proxy overrides AMD's device check; AMD limits them to RDNA 3
  and to RDNA 2 desktop GPUs. They are much slower than Lite there (see the table).
- No reactive or transparency mask is passed (DLSS has none), and colour/output subrect offsets are ignored, as for
  the other upscalers.
- Each live switch creates a new FidelityFX context (one effect context per context). The DLL's Vulkan backend is
  known to run out of pipeline-layout slots when one context rebuilds its network many times on preset changes. The
  dynamic-resolution flag keeps it on one preset, and 30 live switches in a row ran cleanly.

### Licence

`amdvkffx64.dll` is AMD's file, unmodified, under AMD's licence (it ships with AMD Software: Adrenalin Edition). It is
not in this repository: `.gitignore` excludes it, and `steamdeck/decky/build.sh` bundles it from
`../third_party/amd/amdvkffx64.dll` (`FSR4_DLL=` overrides the path). When it is absent, the build warns and the
plugin hides the FSR 4 options. The headers in `steamdeck/nvngx/amd_ffx/` are AMD's MIT-licensed FidelityFX SDK
headers. The ABI facts above (provider block layout, the Vulkan backend description type `0x01000003`, the upscaler
version description `0x0001000B`, the provider classes and the model check order) come from reading OptiScaler's FSR 4
notes and FSR4Injector's README, and from disassembling the DLL. None of OptiScaler's or FSR4Injector's code is used.

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

1. Copy `build_win_nvngx/nvngx.dll` to `steamapps/compatdata/<appid>/pfx/drive_c/windows/system32/_nvngx.dll`. The prefix exists after the first launch. For XeSS / FSR 4, put `libxess.dll` / `amdvkffx64.dll` next to it.
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
| `AISS_CONTROL=/path` | File holding "upscaler [fast maths] [capture]": 0/1/2 (NSS model), 3 (XeSS), 4/5/6 (FSR 4 Lite / INT8 / INT8 RDNA 2), 7 (FSR 4 FP8), 8 (FSR 4, AMD's choice); re-read twice a second; switches the upscaler while the game runs (set by the plugin) |
| `AISS_UPSCALER=xess` | XeSS instead of NSS at launch (the control file overrides it) |
| `AISS_UPSCALER=fsr4`, `AISS_FSR4_MODEL=auto/lite/int8/rdna2/fp8` | AMD FSR 4 (`amdvkffx64.dll` next to the proxy) at launch, with that network (default auto: AMD's choice, Lite on a Steam Deck); also `AISS_UPSCALER=fsr4-lite` etc. |
| `AISS_FSR4_WATERMARK=1` (or `FSR_WATERMARK=1`) | FSR 4: have AMD's DLL draw its banner (network name, version) on the output |
| `AISS_FSR4_GPU=family,revision` | FSR 4: amdgpu family / revision handed to AMD's GPU query (default 144,1 on a Steam Deck, nothing elsewhere) |
| `AISS_FSR4_DUMP_SPIRV=C:\dir` | FSR 4: write every shader module the DLL creates (before the half-float image rewrite) |
| `AISS_FSR4_NO_SPIRV_FIX=1`, `AISS_FSR4_FORCE_FP8=1` | FSR 4 debugging: skip the shader rewrite (RADV then crashes) / allow FP8 without float8 + cooperative matrix support |
| `AISS_XESS_QUALITY=100..105` | XeSS preset (default 100, Ultra Performance: the widest input range) |
| `AISS_GPU_TIMING=N` | Log the GPU time of the upscale and the evaluations per second, averaged over N evaluations |
| `AISS_CAMERA_NEAR`, `AISS_CAMERA_FAR`, `AISS_CAMERA_FOV` | Camera values NGX does not pass. Defaults: 0.1, 100000, 60° (reversed-Z, infinite far plane). |
| `AISS_EXPOSURE` | Exposure passed to NSS (default 1.0) |
| `AISS_LOG=0..4` | Log to `aiss_nvngx.log` next to the exe: 1 lifecycle (default), 2 per-frame, 3 NGX calls, 4 every parameter |
| `AISS_DEBUG_VIEW=0..16` | Write NSS's internal views instead of the image (0 = 4x4 grid) |
| `AISS_DUMP_FRAME=N` | Write output frame N to `aiss_frame_N.ppm` (any upscaler) |
| `FFX_PROFILE_PASSES=file` | Per-pass GPU timings |
| `FFX_ML_RESCALE_FLOAT=1` | Fast maths: fp32 requantisation in the ML graph, ~1-2% faster, not bit exact (the plugin's "Fast maths" toggle sets it live through the control file) |
| `AISS_DUMP_INPUTS=first,count` | Raw colour/depth/MV/output per dispatch next to the game exe (the plugin's "Capture frames" button does the same for the next 8 frames of a running game, into `~/.local/share/aiss-dlss-nss/captures/`) |
| `AISS_JITTER_SIGN=x,y`, `AISS_MV_SIGN=x,y` | Multipliers on the jitter / motion vector scale handed to NSS (convention debugging) |
| `AISS_LAYER_LOG=1` | Vulkan layer: log instances, device extensions and NULL proc lookups to stderr |
| `AISS_LAYER_FEATURES=0`, `AISS_LAYER_EMULATE=0` | Vulkan layer: don't force ML features / don't emulate the NVIDIA extensions (debugging) |
| `AISS_LAYER_XESS=0` | Vulkan layer: don't enable the XeSS device features (debugging) |
| `AISS_LAYER_FSR4=0` | Vulkan layer: don't enable the FSR 4 device features (debugging) |

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
- No dynamic resolution with NSS: it requires every dispatch's render size to equal the size the context was created with, so a smaller render subrect fails the dispatch. XeSS and FSR 4 support it.
- Online games with anti-cheat may object to the extra DLLs; don't use it there.
