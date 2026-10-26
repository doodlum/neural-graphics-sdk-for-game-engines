# Steam Deck: DLSS → Arm NSS under Proton

This branch turns the SDK into a drop-in DLSS Super Resolution replacement for Steam Deck games running under Proton.
Games keep calling DLSS (directly through NGX, or through Streamline), and the frames are upscaled by Arm's Neural
Super Sampling network, running on the Deck's GPU through the native compute ML path of the `universal` branch.

Only the DLSS proxy idea is taken from OptiScaler. There are no other upscalers, no overlay and no frame generation.

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

In the Streamline sample with ray tracing on, D3D12 runs at 9 FPS natively at 1280x722 and 22-24 FPS with DLSS
Performance → NSS at 1280x800 (measured before the pre/post-process tuning).

DLSS quality modes map to NSS's small (mid_low) network. NSS's QUALITY mode switches to the large network, which costs
about 5x more on the Deck; it is only used with `AISS_NSS_QUALITY=0`.

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
| `AISS_RATIO=2.0` | Render-scale divisor for every DLSS mode (default: DLSS's own ratios) |
| `AISS_NSS_QUALITY=0/1/2` | Force NSS's shader quality (0 = large network). Default: follow the game's DLSS mode, on the small network. |
| `AISS_CAMERA_NEAR`, `AISS_CAMERA_FAR`, `AISS_CAMERA_FOV` | Camera values NGX does not pass. Defaults: 0.1, 100000, 60° (reversed-Z, infinite far plane). |
| `AISS_EXPOSURE` | Exposure passed to NSS (default 1.0) |
| `AISS_LOG=0..4` | Log to `aiss_nvngx.log` next to the exe: 1 lifecycle (default), 2 per-frame, 3 NGX calls, 4 every parameter |
| `AISS_DEBUG_VIEW=0..16` | Write NSS's internal views instead of the image (0 = 4x4 grid) |
| `AISS_DUMP_FRAME=N` | Write output frame N to `aiss_frame_N.ppm` |
| `FFX_PROFILE_PASSES=file` | Per-pass GPU timings |
| `AISS_LAYER_LOG=1` | Vulkan layer: log instances, device extensions and NULL proc lookups to stderr |
| `AISS_LAYER_FEATURES=0`, `AISS_LAYER_EMULATE=0` | Vulkan layer: don't force ML features / don't emulate the NVIDIA extensions (debugging) |

## Tested

All tests use the original Streamline Sample 2.14.16 with the production Streamline plugins, cross-built with
clang-cl, under Proton Experimental:

- **D3D12** (with and without the Vulkan layer): DLSS reported supported. NSS runs inline on vkd3d-proton; the output, the debug view (depth, motion vectors, disocclusion) and the exit are clean. The layer hides its emulated NVIDIA extensions from DXVK and vkd3d-proton, which would otherwise switch on Reflex/CUDA interop paths that nothing implements.
- **Vulkan** (`-vk`): DLSS reported supported; output frame verified with `AISS_DUMP_FRAME`; clean exit.
- **D3D11**: the sample's D3D11 path needs `fxc`, so `steamdeck/tests/ngx_d3d11_test` drives the NGX D3D11 API on DXVK instead (64 evaluations, upscaled output within 0.009 mean error of the reference).

The sample's `SLWrapper::UpdateFeatureAvailable` reads a LUID through a pointer to a variable whose scope has ended.
MSVC builds happen to work; an optimised clang build makes Streamline see LUID 0 and report DLSS as unsupported.
Building that one file without optimisation keeps the original source unchanged.

## Limitations

- DLSS Super Resolution only. Ray Reconstruction and Frame Generation report "not supported", and games fall back.
- Colour/output subrect offsets are ignored, and display-resolution motion vectors are not supported (both logged).
- Online games with anti-cheat may object to the extra DLLs; don't use it there.
