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
| Vulkan layer | `steamdeck/layer` | Native Linux implicit layer `VK_LAYER_AISS_ngx`, active only with `AISS_LAYER=1`. Emulates `VK_NVX_binary_import`, `VK_NVX_image_view_handle` and `VK_NV_low_latency(2)`, which Vulkan DLSS/Reflex integrations require, and enables the int8 / dot-product / subgroup-size features the ML kernels need on every device (DXVK's and vkd3d-proton's included). Splits a Vulkan game's command buffer at the evaluate for FSR 4 (D3D12 interop). |
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

Every DLSS mode uses NSS's full network by default. At 640x400 → 1280x800 its passes total 2.22 ms on the Deck
(ML graph 1.36, postprocess 0.57, preprocess 0.25, depth scatter 0.03; bit exact, `docs/nss_gpu_tuning.md` round 3),
about 1.2 ms more per frame than the small network, which ghosts visibly. The Decky plugin switches between them while the
game runs: the plugin's Quick Access panel carries every option (upscaler, NSS quality, FSR 4 model and watermark,
XeSS preset, equal-cost render scale, GPU-time logging, log level, capture) through a control file the proxy re-reads twice a second
(`AISS_CONTROL`). The file is `key=value` lines (upscaler 0 = NSS Quality, 1 = Balanced, 2 = Performance, 3 = XeSS,
4-6/8 = FSR 4), with a legacy first line `<upscaler> <fast_math> <capture>` so older proxies still parse it.

## Equal-cost render scale

The upscalers cost very different amounts on the Deck: FSR 4 INT8 through the Vulkan interop takes about 8 ms of frame
time against NSS Quality's 2.5 ms at 640x392 → 1280x785. With equal cost on (the default; plugin toggle "Equal-cost
render scale", control key `equal_cost`, launch option `AISS_EQUAL_COST`), the proxy answers the game's DLSS
optimal-settings query with a render size per upscaler such that the whole frame (the game's render-resolution work
plus the upscale) costs the same GPU time as NSS Quality at the DLSS mode's standard ratio. The expensive upscalers get
fewer pixels, and the cheap NSS Balanced / Performance networks get more. DLAA is left alone, and the ratio stays
between 1 and 4.

- **The upscale cost** is a table measured on the Deck at render ratios 3, 2.5, 2 and 1.5 per upscaler and API
  (`steamdeck/nvngx/cost_model.cpp`, from the sweep below), interpolated in render pixels and scaled by output pixels.
  The FSR 4 interop rows include the ~2.8 ms per frame the GPU idles while the Vulkan and D3D12 queues wait on each
  other; it shows up as frame time, not as upscale GPU time.
- **The game's cost per extra render megapixel** is measured while the game runs: a thread samples amdgpu's
  `gpu_busy_percent` and shader clock every 20 ms, the busy time per frame minus the modelled upscale is kept per
  render size, and the slope across the sizes seen is the game's cost (until two sizes are seen, 0.55 of the average,
  the share measured in the Streamline sample). It is saved per game under the plugin's `cost/` folder, so the next
  launch starts from it.
- **It applies when the game next asks** for optimal settings: a DLSS mode change, a resolution change or a restart.

Upscale cost at 1280x800 output, ms of frame time (Streamline sample; NSS at exactly 2x takes its faster static-LUT
path):

| Upscaler | D3D12 3x | 2.5x | 2x | 1.5x | Vulkan 3x | 2.5x | 2x | 1.5x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NSS Quality | 1.56 | 1.88 | 2.22 | 3.72 | 1.58 | 1.90 | 2.46 | 3.68 |
| NSS Balanced | 1.29 | 1.18 | 1.22 | 2.41 | 1.49 | 1.21 | 1.43 | 2.43 |
| NSS Performance | 1.16 | 1.05 | 1.09 | 2.28 | 1.36 | 1.07 | 1.29 | 2.30 |
| XeSS | 3.10 | 3.13 | 3.19 | 3.36 | 3.16 | 3.19 | 3.25 | 3.43 |
| FSR 4 Lite | 3.05 | 3.05 | 3.06 | 3.10 | 6.37 | 6.37 | 6.40 | 6.47 |
| FSR 4 INT8 | 5.04 | 5.01 | 5.02 | 5.05 | 8.31 | 8.33 | 8.34 | 8.45 |
| FSR 4 INT8 RDNA 2 | 4.95 | 4.96 | 4.98 | 5.02 | 8.20 | 8.27 | 8.26 | 8.37 |

Result in the Streamline sample, DLSS Performance (2x for NSS Quality), equal cost on, frames per second over the
last 360 upscales. Before, at the standard 2x, FSR 4 Lite on Vulkan ran at 24.5 FPS against NSS Quality's 26.5.

| Upscaler | D3D12 render size | FPS | Vulkan render size | FPS |
| --- | --- | --- | --- | --- |
| NSS Quality (reference) | 640x361 | 25.6 | 640x392 | 26.1 |
| NSS Performance | 650x366 | 25.6 | 656x402 | 26.8 |
| XeSS | 626x353 | 25.4 | 623x382 | 26.5 |
| FSR 4 Lite | 625x353 | 25.6 | 553x339 | 27.2 |
| FSR 4 INT8 | 592x333 | 25.9 | 543x333 | 26.1 |

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

| API | Mode | NSS Quality | NSS Performance | XeSS |
| --- | --- | --- | --- | --- |
| D3D12 | 640x361 → 1280x722 (2x) | 2.20 ms, 25.4 FPS | 1.01 ms, 26.3 FPS | 2.88 ms, 24.8 FPS |
| D3D12 | 853x481 → 1280x722 (1.5x) | 3.53 ms, 17.8 FPS | 2.10 ms, 18.1 FPS | 3.02 ms, 18.0 FPS |
| Vulkan | 640x392 → 1280x785 (2x) | 2.58 ms, 26.6 FPS | 1.29 ms, 27.5 FPS | 3.19 ms, 26.2 FPS |
| Vulkan | 853x523 → 1280x785 (1.5x) | 3.79 ms, 19.2 FPS | 2.29 ms, 19.8 FPS | 3.36 ms, 19.3 FPS |

XeSS's cost depends mostly on the output size; NSS's on the input size. In screenshots XeSS is softer than NSS Quality
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

The proxy runs AMD FSR 4.1.2. The main path, for every API, is **AMD's FSR SDK 2.4.0 Vulkan upscaler**
(`amd_fidelityfx_upscaler_vk.dll`, signed SDK binary, file version 4.1.2.4555) on the game's own VkDevice
(`fsr4_vk.cpp`), recorded into the game's command buffer:

- **Vulkan games**: on the game's device.
- **D3D12 games**: on vkd3d-proton's VkDevice, recorded into the game's command list (like NSS).
- **D3D11 games**: on DXVK's VkDevice, in the proxy's own command buffer.

When that DLL is missing or refuses the device, the older paths take over through `amdxcffx64.dll`, the FSR 4 component
of AMD's Windows D3D12 driver (Adrenalin 26.10.1, FidelityFX 2.4.0.4370):

- **D3D12 games**: natively on the game's own `ID3D12Device` and command list (`fsr4_dx12.cpp`), no copies.
- **D3D11 games** (DXVK): through **D3D12 interop** (`fsr4_interop.cpp`): an in-process vkd3d-proton D3D12 device that
  shares textures and a fence with the game's VkDevice (below).
- **Vulkan games**: through the same D3D12 interop, with the Vulkan layer splitting the game's command buffer at the
  evaluate (see "FSR 4 for Vulkan games"); needs `AISS_LAYER=1` (the plugin sets it).

### FSR SDK 2.4 Vulkan DLL on RADV

Out of the box the SDK DLL offers only FSR 3.1.6 and 2.3.4 on a Steam Deck. Three things make it run FSR 4 Lite, and fast:

- **FSR 4 on RADV.** The DLL asks the device for `vkUpdateFfxApiProviderAMD`, through which AMD's Windows Vulkan driver
  hands it a driver-side FSR 4 provider; RADV has no such entry point. The DLL also carries the same FSR 4 networks
  itself, but their device checks (vtable slot 2 of `ffxProvider_FSR4*`, found via RTTI) accept nothing on RADV. The proxy
  wraps those checks (shared with the amdxcffx64 path in `fsr4_common.cpp`) so the requested network accepts; auto is
  the GPU's own network, Lite on a Steam Deck.
- **Cached memory.** The DLL allocates its weights and tensors from the first memory type its filter allows, which on
  RADV is `DEVICE_COHERENT_AMD | DEVICE_UNCACHED_AMD`: every access bypasses the GPU's L2. The proxy hands the DLL its own
  `vkGetDeviceProcAddr` (both the device context's pointer and the DLL's `vulkan-1.dll` import) and moves those
  allocations to the same heap's plain type. That alone takes Lite from **7.3 to 3.0 ms** at 640x392 → 1280x785.
- **Image layouts.** Inputs in layouts the FFX backend does not expect (DXVK can hand over depth as
  `DEPTH_STENCIL_READ_ONLY_OPTIMAL`) are moved to a supported one for the dispatch and back.

**Replacement shader modules.** The proxy can replace any shader module the DLL creates, keyed by an FNV-1a hash of
the original SPIR-V, so a replacement only ever applies to the exact module it was made for. They come from
`system32\aiss\fsr4_shaders.bin` (format in `tools/pack_fsr4_shaders.py`; the plugin installs it when its bundle has
one) or from `AISS_FSR4_VK_SHADER_DIR`. **No replacement shaders are part of this repository.** The ones used on the
Deck are bit-identical in output to AMD's (all 16 combinations of HDR, motion-vector resolution, depth direction and
auto exposure) and about 10% faster:

| FSR 4 Lite, 2x | AMD's shaders | Replacement shaders |
| --- | --- | --- |
| Vulkan (Streamline sample, 640x392 → 1280x785) | 3.00 ms | 2.71 ms |
| D3D12 (Streamline sample, 640x361 → 1280x722) | 2.79 ms | 2.50 ms |
| D3D11 (replay harness, 640x361 → 1280x722) | 3.45 ms | 3.11 ms |

For comparison, the D3D12 interop that Vulkan games used before cost 4.6 ms plus about 2.8 ms of cross-queue idle
per frame, and AMD's D3D12 DLL natively 2.8 ms.

Per-pass diagnostics: `AISS_FSR4_VK_PROFILE=N` logs the GPU time of each dispatch the DLL records (by shader entry
point), `AISS_FSR4_DX12_PROFILE=N` the same for the D3D12 DLL, `AISS_FSR4_VK_DUMP=dir` writes each original module as
`<hash>_<entry point>.spv`, and `AISS_FSR4_VK_SHADERS=0` turns replacement off.

The plugin's "Upscaler" dropdown offers **FSR 4**, with an "FSR 4 model" choice of **Auto** (AMD's choice: Lite on a
Steam Deck), **Lite**, **INT8** and **INT8 RDNA 2**. Control-file upscaler values: 4 Lite, 5 INT8, 6 INT8 RDNA 2, 8 Auto;
launch options `AISS_UPSCALER=fsr4` with `AISS_FSR4_MODEL=lite|int8|rdna2|auto`. FSR 4 switches live like the other
upscalers.

**4.1.2 is required.** Both DLLs (`amd_fidelityfx_upscaler_vk.dll` must be file version 4.1.2.x) are loaded by full path from a private `aiss` subfolder of the prefix's `system32`
(`system32\aiss\`, next to `_nvngx.dll`), not from `system32` itself, because Proton ships its own older
`amdxcffx64.dll` (FSR 4.1.1, sha256 4e7dc37a…) there and reinstalls it on every launch, which would shadow ours. The
proxy logs the DLL's file version, loads it, and checks the FidelityFX provider's version name: anything that is not
4.1.2 (an older 4.0.x/4.1.1, or an FSR 3.1 fallback) is **refused loudly**. If the DLL is missing, refused, hands out no
provider, rejects the network, or interop fails at any step, the proxy logs why and **falls back to NSS**, with output
bit-identical to plain NSS (Streamline replay hash a4958bfb with the DLL removed).

### How amdxcffx64 is driven

The proxy uses the DLL only through the FidelityFX provider it exports (`UpdateFfxApiProvider`) and AMD's public
FidelityFX SDK 2.3 types (`steamdeck/nvngx/amd_ffx/`, MIT); the file is never changed. In memory
(`fsr4_common.cpp`):

1. **SDK module check.** The DLL hands out providers only when it finds signed FidelityFX SDK modules
   (`amd_fidelityfx_*_dx12.dll`) through `GetModuleHandleW`. The proxy points the DLL's `GetModuleHandleW` import at a
   lookup that answers those names with the DLL itself (same signing certificate; Wine's `WinVerifyTrust` accepts it).
2. **GPU family.** AMD's device checks read the GPU family and revision from the Windows kernel driver
   (`D3DKMTEnumAdapters` + `D3DKMTQueryAdapterInfo`, AMD private data, looked up with `GetProcAddress`; it also uses
   AMD's `amdxc64.dll` extension, which Proton ships). Wine has no answer to the D3DKMT query, so the DLL would refuse
   every FSR 4 network — the problem FSR4Injector solved with a GE-Proton `amdxc64` shim. The proxy points the DLL's
   `GetProcAddress` import at its own D3DKMT pair answering with the amdgpu family / external revision: 144 / 1 on Van
   Gogh (read with `DRM_IOCTL_AMDGPU_INFO`; `AISS_FSR4_GPU=family,revision` overrides). No extra shim DLL is needed.
3. **Model selection.** Each network is its own provider class (RTTI `ffxProvider_FSR4`, `..._FSR4_Int8`,
   `..._FSR4_Int8_Navi2`, `..._FSR4Lite_Int8`) whose device check is the third virtual function; FidelityFX asks FP8,
   INT8, INT8 RDNA 2, then Lite and runs the first that accepts. The proxy wraps those checks: with a model selected only
   that network accepts; with Auto, AMD's answer goes through (AMD's checks run only for an AMD vendor ID, so under
   `PROTON_FORCE_NVAPI=1` Auto falls back to the family's network, Lite on the Deck).
4. The upscaler version description is left out: this DLL's `createContext` returns `ERROR_PARAMETER` with it.

Proof of which network and version ran:

- The log names the file version (2.4.0.4370), the provider (`"4.1.2 *"`, id `…1002`), the network that runs, and AMD's
  own choice.
- `AISS_FSR4_WATERMARK=1` / `FSR_WATERMARK=1` / the plugin's "FSR 4 watermark" set `MLSR-WATERMARK`, and the DLL draws
  `FSR4L-i8` (Lite), `FSR4-i8` (INT8) or `FSR4-i8-2` (INT8 RDNA 2) followed by `Upscale 4.1.2 DRS 2.00x Source: Driver`,
  on D3D12 and through interop (`testlogs/fsr4/dx12_lite_watermark.png`, `d3d11_interop_lite_watermark.png`).

### Inputs

Colour, depth (passed directly natively; as R32F through interop), motion vectors with the DLSS MV scale, jitter (DLSS
and FSR share the convention; the other three sign combinations leave jagged edges on static detail, checked
visually), reset, HDR, inverted depth (near and far swapped, the FidelityFX convention), display-resolution motion
vectors when `MVLowRes` is not set, DLSS AutoExposure → FidelityFX auto exposure, sharpening off. Dynamic resolution:
the context is created with `FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION` and every dispatch passes the DLSS render subrect as
`renderSize` (`ngx_d3d11_test --dyn` passes for Lite, INT8 and INT8 RDNA 2 through interop). FP8 is refused (it needs
RDNA 4).

### D3D12 interop for D3D11 games

- The proxy creates a D3D12 device in-process (vkd3d-proton) on the game's adapter (LUID from
  `VkPhysicalDeviceIDProperties`, matched with `IDXGIFactory4::EnumAdapterByLuid`) with a **compute** queue, the first
  time FSR 4 is used.
- Memory is shared through D3D12 committed **2D textures** created with `D3D12_HEAP_FLAG_SHARED` and
  `ID3D12Device::CreateSharedHandle`, imported into the game's VkDevice as `VkImage`s with `VK_KHR_external_memory_win32`
  (handle type `D3D12_RESOURCE`, dedicated allocation, the usage vkd3d-proton gives the texture so both drivers lay it
  out alike). Under Proton vkd3d-proton returns `S_OK` with a **null handle** for a shared committed *buffer* and
  `E_NOTIMPL` (0x80004001) for `CreateSharedHandle` on a shared *heap*, so textures are what is shared. FSR 4 reads the
  shared colour / depth (copied through a buffer into R32F: a depth aspect cannot be copied to a colour image) /
  motion-vector textures and writes the shared output directly.
- Sync is one shared `ID3D12Fence` imported as a Vulkan **timeline semaphore** (`VK_KHR_external_semaphore_win32`,
  `D3D12_FENCE` handle): the copy-in submission signals N, the D3D12 queue waits N, runs FSR 4 and signals N+1, the
  copy-out submission waits N+1. The proxy owns these submissions on DXVK's queue (it already submits its own command
  buffers there), so all waits are on the GPU / in the kernel and the CPU never waits per frame. A warm-up dispatch at
  creation builds FSR 4's pipelines once.

### Performance

GPU time per upscale (`AISS_GPU_TIMING`), Steam Deck. Native D3D12 is the Streamline sample (ray tracing on); interop is
the D3D11 harness (DXVK, `AISS_LAYER=1`, 64 evaluations), which does little else, so the GPU runs at lower clocks there:
compare a row's interop columns with each other, and with native only through the "normalised" column.

| Ratio | Network | Native D3D12 (sample) | Interop total (harness) | of which: copy-in | D3D12 dispatch | sync gap | copy-out | Interop normalised to sample clocks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2x | Lite | 2.77 ms, 24.9 FPS | 5.71 ms | 0.16 | 4.81 | 0.50 | 0.24 | ~3.5 ms |
| 2x | INT8 | 4.53 ms, 23.8 FPS | 9.37 ms | 0.18 | 8.03 | 0.90 | 0.26 | ~5.7 ms |
| 2x | INT8 RDNA 2 | 4.50 ms, 23.9 FPS | 8.77 ms | 0.17 | 7.88 | 0.47 | 0.25 | ~5.4 ms |
| 1.5x | Lite | 2.79 ms, 17.8 FPS | 5.93 ms | 0.29 | 4.86 | 0.54 | 0.24 | ~3.7 ms |
| 1.5x | INT8 | 4.56 ms, 17.3 FPS | 9.12 ms | 0.29 | 7.98 | 0.61 | 0.24 | ~5.7 ms |
| 1.5x | INT8 RDNA 2 | 4.53 ms, 17.3 FPS | 9.11 ms | 0.30 | 7.63 | 0.94 | 0.24 | ~5.7 ms |

2x = 640x361 → 1280x722 in the sample, 640x400 → 1280x800 in the harness; 1.5x = 853x481 / 853x533 → 1280x722 / 1280x800.
The normalised column divides by the clock ratio measured with NSS Quality on both (harness 3.61 / 5.65 ms vs sample
2.21 / 3.53 ms: 1.63 / 1.60), so it is an estimate; everything else is measured. Interop's own measured overhead
(copies + sync) is 0.9-1.5 ms per frame at harness clocks (~0.6-0.9 ms normalised); the D3D12 dispatch itself scales like
the native one. For reference: NSS Quality 2.21 / 3.53 ms, XeSS 2.88 / 3.03 ms in the sample (D3D12, 2x / 1.5x).

### FSR 4 for Vulkan games

NGX gives the proxy only the game's Vulkan command buffer, which the game submits later, and a command buffer cannot
signal a semaphore halfway. So the native layer (`VK_LAYER_AISS_ngx`) **cuts the game's command buffer in two at the
evaluate**; the submissions then look exactly like the D3D11 path's, in the same frame:

```
game cb:  [ game ... | proxy copy-in ] CUT [ proxy copy-out | game ... ]
submit:   { game's waits, part 1, signal fence=N }  ->  D3D12 queue: wait N, FSR 4, signal N+1
          { wait fence>=N+1, part 2, game's signals + fence }
```

- The proxy talks to the layer with ordinary Vulkan calls the layer recognises and swallows (winevulkan does not
  forward private entry points; contract in `steamdeck/layer/aiss_split_abi.h`): a `vkWaitSemaphores` with a magic
  timeout registers (the layer answers `VK_TIMEOUT`, so without the layer the game gets NSS); `vkCmdFillBuffer` on a
  magic marker buffer marks the evaluate start and the cut.
- After the cut the layer records the game's commands into a fresh command buffer of the same queue family: every
  `vkCmd*` entry point goes through a generated trampoline (`aiss_cmd_trampolines.inc`, from `vulkan_core.h` by
  `gen_cmd_trampolines.py`). At `vkQueueSubmit` / `vkQueueSubmit2` the game's batch is expanded into the parts; the
  game's waits stay on the first, its signals and fence on the last.
- Proton's winevulkan emulates D3D12 fences: each wait or signal of a D3D12-fence semaphore becomes a temporary host
  semaphore that a winevulkan worker thread (`signaller_worker`) bridges to the shared fence. A semaphore operation the
  layer adds on its own is never seen (the first attempt hung that way: `err:winevulkan:signaller_worker wait timed out
  with non-empty poll list`). So right after the cut the proxy makes a "capture" `vkQueueSubmit2` (signal N, wait N+1)
  that goes through winevulkan's conversion; the layer keeps the converted host semaphores for the cut and does not
  submit it.
- Engines that end render passes lazily (NVRHI: the Streamline sample) call evaluate with a render pass still open.
  The layer ends it at the evaluate marker and drops the game's own end later, so the copies and the cut are outside
  any render pass. A cut is refused (the fence is then signalled after the whole command buffer, a frame-late input but
  never a hang) in secondary command buffers and ones allocated before the layer saw them. A command buffer submitted
  twice without re-recording runs its parts without the fence operations; one reset before it was ever submitted
  releases the D3D12 wait on the host.
- Only native Vulkan games' devices get this (not DXVK / vkd3d-proton); `AISS_LAYER_SPLIT=0` turns it off (FSR 4 then
  falls back to NSS with a log line). The layer also enables `timelineSemaphore`; the proxy requests
  `VK_KHR_external_memory_win32`, `VK_KHR_external_semaphore_win32` and `VK_KHR_timeline_semaphore` through NGX's
  device-extension query.

Measured in the Streamline sample (Vulkan, ray tracing on, 1200 frames, `AISS_GPU_TIMING`), game-queue GPU time per
upscale, with the D3D12 dispatch timed on the interop queue:

| Ratio | Network | Total (game queue) | copy-in | wait for D3D12 | of which D3D12 dispatch | copy-out | FPS | NSS Quality (same run setup) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2x | Lite | 4.59 ms | 0.13 | 4.25 | 3.47 | 0.21 | 22.8 | 2.57 ms, 25.4 FPS |
| 2x | INT8 | 6.59 ms | 0.13 | 6.25 | 5.51 | 0.21 | 21.8 | |
| 2x | INT8 RDNA 2 | 6.58 ms | 0.13 | 6.24 | 5.50 | 0.21 | 22.0 | |
| 1.5x | Lite | 4.63 ms | 0.23 | 4.19 | 3.44 | 0.21 | 17.8 | 3.78 ms, 18.5 FPS |
| 1.5x | INT8 | 6.77 ms | 0.23 | 6.32 | 5.55 | 0.21 | 16.7 | |
| 1.5x | INT8 RDNA 2 | 6.71 ms | 0.23 | 6.26 | 5.50 | 0.21 | 16.8 | |

2x = 640x392 → 1280x785, 1.5x = 853x523 → 1280x785. Overhead against FSR 4 itself: copies ~0.35-0.45 ms and a sync gap
of ~0.75 ms between the first part's end and the second part's start (winevulkan's worker thread relays both fence
hops). The D3D12 dispatch on the interop compute queue is slower than native D3D12 FSR 4 on the game's own queue
(Lite 3.47 ms vs 2.77 ms in the D3D12 sample). The watermark reads "FSR4L-i8 Upscale 4.1.2 DRS 2.00x Source: Driver".

GPU-side waits inside one command buffer were tried first and do not work (removed): `vkCmdWaitEvents` on a host-set
event with shared D3D12 textures ran into amdgpu's 5 s gfx lockup timeout (`ring gfx_0.0.0 timeout ... ring reset`,
three times); host-memory staging with a 750 ms release never saw the D3D12 work finish while the game's job waited.

### Limitations

- FP8 needs RDNA 4 and is refused.
- INT8 and INT8 RDNA 2 run on the Deck only because the proxy overrides AMD's device check (AMD limits them to RDNA 3 /
  RDNA 2 desktop GPUs); they cost about 1.6x Lite.
- Interop supports colour/motion formats RGBA16F/32F, R11G11B10, RGB10A2, RGBA8/BGRA8 (+sRGB), RG16F/32F and depth
  D32 / D32S8; anything else falls back to NSS with a log line.
- No reactive or transparency mask is passed (DLSS has none), and colour/output subrect offsets are ignored, as for the
  other upscalers.

### Licence

`amdxcffx64.dll` is AMD's file, unmodified, under AMD's licence (it ships with AMD Software: Adrenalin Edition). It is
not in this repository: `.gitignore` excludes it, and `steamdeck/decky/build.sh` bundles it from
`../third_party/amd/amdxcffx64.dll` (`FSR4_DX12_DLL=` overrides). When it is absent the build warns and the plugin hides
the FSR 4 option. The headers in `steamdeck/nvngx/amd_ffx/` are AMD's MIT-licensed FidelityFX SDK headers. The ABI
facts above (provider block layout, the D3D12 backend description type `0x00000002`, the provider classes and their
check order, the D3DKMT private-data offsets, `MLSR-WATERMARK`) come from reading OptiScaler's FSR 4 notes,
FSR4Injector's README and GE-Proton's amdxc64, and from disassembling the DLL. None of their code is used.

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

1. Copy `build_win_nvngx/nvngx.dll` to `steamapps/compatdata/<appid>/pfx/drive_c/windows/system32/_nvngx.dll`. The prefix exists after the first launch. For XeSS, put `libxess.dll` next to it; for FSR 4, put `amdxcffx64.dll` in a `system32\aiss\` subfolder (not `system32` itself, where Proton's own older `amdxcffx64.dll` would shadow it).
2. Copy the loader shim next to the game executable:
   - `build_win_loader/dbghelp/dbghelp.dll` if the game ships `sl.interposer.dll`;
   - `build_win_loader/dxgi/dxgi.dll` otherwise.
3. Copy `build_layer/libVkLayer_aiss_ngx.so` and `VkLayer_aiss_ngx.json` to `~/.local/share/vulkan/implicit_layer.d/`.
4. Launch options: `PROTON_FORCE_NVAPI=1 AISS_LAYER=1 WINEDLLOVERRIDES="dbghelp=n,b" %command%`. Use `dxgi=n,b` for the dxgi shim.
5. Enable DLSS in the game.

## Options (launch-option environment variables)

| Variable | Effect |
| --- | --- |
| `AISS_NSS_QUALITY=0/1/2` | NSS shader quality at launch: 0 = Quality network (default), 1 = Balanced, 2 = Performance |
| `AISS_CONTROL=/path` | File holding "upscaler [fast maths] [capture]": 0/1/2 (NSS model), 3 (XeSS), 4/5/6 (FSR 4 Lite / INT8 / INT8 RDNA 2), 7 (FSR 4 FP8), 8 (FSR 4, AMD's choice); re-read twice a second; switches the upscaler while the game runs (set by the plugin) |
| `AISS_UPSCALER=xess` | XeSS instead of NSS at launch (the control file overrides it) |
| `AISS_UPSCALER=fsr4`, `AISS_FSR4_MODEL=auto/lite/int8/rdna2` | AMD FSR 4 (amdxcffx64.dll: native on D3D12, D3D12 interop on D3D11 and Vulkan) at launch, with that network (default auto: Lite on a Steam Deck); also `AISS_UPSCALER=fsr4-lite` etc. |
| `AISS_FSR4_DIR=C:\path` | Folder amd_fidelityfx_upscaler_vk.dll and amdxcffx64.dll are loaded from (default: the `aiss` subfolder next to `_nvngx.dll`) |
| `AISS_FSR4_VK_SHADERS=0`, `AISS_FSR4_VK_SHADER_DIR=C:\path` | FSR 4 on the SDK Vulkan DLL: no replacement shader modules / take them from `<path>\<hash>.spv` instead of `aiss\fsr4_shaders.bin` |
| `AISS_FSR4_VK_PROFILE=N`, `AISS_FSR4_DX12_PROFILE=N`, `AISS_FSR4_VK_DUMP=dir` | FSR 4 diagnostics: per-dispatch GPU time (SDK Vulkan / D3D12 DLL), dump the SDK DLL's shader modules |
| `AISS_FSR4_ALLOW_ANY_VERSION=1` | Accept an FSR 4 provider that is not 4.1.2 (diagnostics; normally such a provider is refused and the proxy uses NSS) |
| `AISS_FSR4_WATERMARK=1` (or `FSR_WATERMARK=1`) | FSR 4: have AMD's DLL draw its banner (network name, version) on the output |
| `AISS_FSR4_GPU=family,revision` | FSR 4: amdgpu family / revision handed to AMD's GPU query (default 144,1 on a Steam Deck, nothing elsewhere) |
| `AISS_FSR4_DX12_VER=0xVER` | FSR 4 D3D12: chain an upscaler version desc with this value (default: none; the 4.1.2 DLL rejects it) |
| `AISS_EQUAL_COST=0/1` | Equal-cost render scale (default 1): each upscaler gets the render size at which game + upscale costs the same GPU time as NSS Quality at the DLSS mode's ratio (see *Equal-cost render scale*); 0 = the standard DLSS ratios for every upscaler. The control file's `equal_cost=` overrides it |
| `AISS_RATIO=r` | Force one render-scale divisor for every DLSS mode (testing); turns equal cost off |
| `AISS_XESS_QUALITY=100..105` | XeSS preset (default 100, Ultra Performance: the widest input range) |
| `AISS_GPU_TIMING=N` | Log the GPU time of the upscale and the evaluations per second, averaged over N evaluations |
| `AISS_CAMERA_NEAR`, `AISS_CAMERA_FAR`, `AISS_CAMERA_FOV` | Camera values NGX does not pass. Defaults: 0.1, 100000, 60° (reversed-Z, infinite far plane). |
| `AISS_EXPOSURE` | Exposure passed to NSS (default 1.0) |
| `AISS_LOG=0..4` | Log to `aiss_nvngx.log` next to the exe: 1 lifecycle (default), 2 per-frame, 3 NGX calls, 4 every parameter |
| `AISS_DEBUG_VIEW=0..16` | Write NSS's internal views instead of the image (0 = 4x4 grid) |
| `AISS_DUMP_FRAME=N` | Write output frame N to `aiss_frame_N.ppm` (any upscaler) |
| `FFX_PROFILE_PASSES=file` | Per-pass GPU timings |
| `FFX_ML_RESCALE_FLOAT=1` | Fast maths: fp32 requantisation in the ML graph, ~1-2% faster, not bit exact; launch option only (the plugin no longer offers it and always writes fast maths off) |
| `AISS_DUMP_INPUTS=first,count` | Raw colour/depth/MV/output per dispatch next to the game exe (the plugin's "Capture frames" button does the same for the next 8 frames of a running game, into `~/.local/share/aiss-dlss-nss/captures/`) |
| `AISS_JITTER_SIGN=x,y`, `AISS_MV_SIGN=x,y` | Multipliers on the jitter / motion vector scale handed to NSS (convention debugging) |
| `AISS_LAYER_LOG=1` | Vulkan layer: log instances, device extensions and NULL proc lookups to stderr |
| `AISS_LAYER_FEATURES=0`, `AISS_LAYER_EMULATE=0` | Vulkan layer: don't force ML features / don't emulate the NVIDIA extensions (debugging) |
| `AISS_LAYER_XESS=0` | Vulkan layer: don't enable the XeSS device features (debugging) |
| `AISS_LAYER_SPLIT=0` | Vulkan layer: no command buffer split (FSR 4 on Vulkan games then falls back to NSS) |
| `AISS_LAYER_RPLOG=1` | Vulkan layer: log render pass begin/end on split devices (debugging) |

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
