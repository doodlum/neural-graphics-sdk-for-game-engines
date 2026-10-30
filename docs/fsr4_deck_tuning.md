# FSR 4 Lite tuning notes (Steam Deck, RADV)

How AMD FSR 4.1.2 Lite was made to run, and then run fast, on the Steam Deck (Van Gogh: RDNA2, 8 CUs, Mesa RADV/ACO)
under Proton, for D3D11, D3D12 and Vulkan games. Integration code: `steamdeck/nvngx/fsr4_vk.cpp` (overview in
`steamdeck/README.md`, "AMD FSR 4").

**No AMD shader code is in this repository.** The faster shaders are replacement SPIR-V modules that the proxy loads
from packs next to AMD's DLL (`system32\aiss\fsr4_shaders.bin`, `fsr4_shaders_dot2.bin`; format in
`steamdeck/tools/pack_fsr4_shaders.py`). They are built and kept outside the repository. This file documents the method,
the measurements and the techniques, not the shaders.

## Result

FSR 4 Lite, 2x (render half the output size per axis), GPU time per upscale:

| API | AMD's shaders | Replacement shaders | Change |
| --- | --- | --- | --- |
| Vulkan (Streamline sample, 640x392 → 1280x785) | 3.00 ms | 2.12 ms | −29% |
| D3D12 (Streamline sample, 640x361 → 1280x722) | 2.79 ms | 1.96 ms | −30% |
| D3D11 (replay harness, 640x361 → 1280x722) | 3.30 ms | 2.30 ms | −30% |

Quality against AMD's own shaders: 69.1–72.7 dB tonemapped PSNR on all 16 combinations of the context flags (below);
the largest differences are single pixels in a dithered shadow edge, invisible at 1:1. Frame-to-frame flicker on static
content stays inside AMD's own run-to-run spread. A bit-identical set (all 16 combinations identical) runs at 2.56 ms on
Vulkan.

For context: before this work, Vulkan games ran FSR 4 through a D3D12 interop at 4.6 ms plus ~2.8 ms of cross-queue
idle per frame, and AMD's SDK Vulkan DLL ran Lite at 7.3 ms on RADV.

## Getting AMD's SDK Vulkan DLL to run FSR 4 on RADV

AMD's FSR SDK 2.4.0 ships a signed `amd_fidelityfx_upscaler_vk.dll` (file version 4.1.2.4555). On the Deck it first offers
only FSR 3.1.6 and 2.3.4.

| Problem | Finding | Fix (proxy) | Effect |
| --- | --- | --- | --- |
| No FSR 4 offered | The DLL asks the device for `vkUpdateFfxApiProviderAMD`, through which AMD's Windows Vulkan driver hands it a driver-side FSR 4 provider (same block layout as `amdxcffx64`'s `UpdateFfxApiProvider`). RADV has no such entry point. The DLL also carries the same FSR 4 networks itself, but their device checks reject RADV. | Wrap the per-network device checks (found by RTTI, shared with the `amdxcffx64` path) so the chosen network accepts; auto = Lite on Van Gogh. | 4.1.2 offered and created on the game's device |
| 7.3 ms for Lite | Per-dispatch profiling showed the memory-heavy passes 3–4× slower than the same network through AMD's D3D12 DLL, with near-identical ACO stats. The DLL allocates weights and tensors from the first allowed memory type: on RADV `DEVICE_COHERENT_AMD | DEVICE_UNCACHED_AMD`, which bypasses L2. | Hand the DLL our own `vkGetDeviceProcAddr` (its device context and its `vulkan-1.dll` import) and move such allocations to the same heap's plain type. | 7.3 → 3.0 ms |
| DXVK image layouts | D3D11 inputs can arrive in layouts the FFX backend does not expect (depth as `DEPTH_STENCIL_READ_ONLY_OPTIMAL`). | Transition for the dispatch and back. | D3D11 works natively |
| One path for every API | The DLL records into any VkCommandBuffer. | D3D12 games run it on vkd3d-proton's device (recorded into the game's command list), D3D11 games on DXVK's device. | Shader work applies to all three APIs |

## Method

- **Replacement hook.** Every `vkCreateShaderModule` from the DLL is keyed by an FNV-1a hash of the original SPIR-V; a
  replacement applies only to the exact module it was made for. `AISS_FSR4_VK_DUMP=dir` writes the originals.
- **Variants.** The DLL builds different prepass/postpass modules depending on the context flags. The D3D11 harness
  (`steamdeck/tests/ngx_d3d11_test`, `--flags`) covers all 16 combinations of HDR, low-res motion vectors, inverted
  depth and auto exposure; the Lite network uses 11 distinct modules for them (6 prepass, 2 postpass, pass1/2/3).
- **Quality gate.** Replay of 8 captured real-game frames ×4 loops through the D3D11 path (deterministic; loop wrap
  forces history rejection), per flag combination, against AMD-shader baselines: bit-identical, or tonemapped PSNR and
  max/mean difference. Accepted lossy changes had to stay ≥ 45 dB on every combination with no visible difference in
  crops and ×50/×500 amplified diffs. Synthetic scenes at render ratios 1.0–3.0 as a second check.
- **Flicker.** `AISS_DUMP_FRAME=N AISS_DUMP_FRAME_COUNT=16` dumps consecutive frames; mean frame-to-frame luma change
  on static content and edges of the Streamline sample, against several AMD runs (the sample itself varies about one
  level per frame between runs).
- **Timing.** `AISS_FSR4_VK_PROFILE=N` (and `AISS_FSR4_DX12_PROFILE=N` for AMD's D3D12 DLL) timestamps every dispatch
  the DLL records, by shader entry point. Totals from `AISS_GPU_TIMING`. The replay harness runs at ~350 fps with
  moving clocks, so D3D11 numbers are medians of 14 windows of 64 evaluations.
- **ISA.** Every change was judged by the RDNA2 code ACO emits (`RADV_DEBUG=shaders,shaderstats`): VALU/VMEM counts,
  waits, VGPRs and resulting waves per SIMD, ACO's latency/throughput estimates; measured time decides.

## Where the time goes (AMD's shaders, Vulkan 2x, after the memory fix)

| Pass | µs | What it does |
| --- | --- | --- |
| prepass | 939 | One thread per output pixel: colour reconstruction from the render-size input, depth dilation, history reprojection, input convolution to int8 features |
| pass1 / pass2 / pass3 | 530 / 203 / 325 | int8 convolution layers (`v_dot4_i32_i8`) |
| postpass | 930 | Final layers, colour output, history and recurrent state writes at output resolution |
| other (exposure, small passes) | ~45 | |

All big passes end up VALU-issue bound; an issue-bound model (waves × VALU per thread ÷ 16 SIMD32 × 1.6 GHz)
reproduces the measured pass times within ~5–15%.

## Round 1 (bit-identical): 3.00 → 2.72 ms

| Change | Pass | µs |
| --- | --- | --- |
| Convolution weights made wave-uniform (each wave handles one output-channel group via an LDS exchange), so they load into SGPRs with scalar loads instead of 16 vector loads per thread | prepass | 939 → 842 |
| Colour taps shared through LDS, stores issued per pixel (VGPRs 128 → 80) | postpass | 930 → 839 |
| Lane remap for wave-uniform weights; exact reordering of an integer accumulation removing 48 SGPR spills | pass1 | 530 → 483 |
| A loop bound that depended on the workgroup index blocked unrolling; specialised path with constant trip counts | pass3 | 325 → 287 |
| Rounding bit via one `v_bfe_u32`; byte packing via SDWA moves | int8 passes | ~−35 total |

Two toolchain pitfalls found here: a recompiled module must keep the original entry-point name (otherwise the process
dies in pipeline creation), and decompiled vec3 image stores must not turn into `.xyzz` writes (they write alpha the
original leaves alone).

## Round 2 (bit-exact + first lossy changes): 2.72 → 2.49 ms

| Change | Exact? | µs | Quality |
| --- | --- | --- | --- |
| Integer residual tails: float combining of power-of-two-scaled integers done as integer shifts with round-to-nearest-even | yes | −46 | identical |
| Requantisation: clamp before the 8-bit shift, so the result is a byte select; ReLU folded into the clamp | yes | −44 | identical |
| Prepass convolution: exact f16 products accumulated in f32 (`v_fma_mix_f32`) instead of rounding each product to f16 | no | −67 | ~68–72 dB |
| …the same on `v_dot2_f32_f16` (the instruction AMD's D3D12 build compiles to) | no | further −44 | ~68–72 dB |
| History reprojection: 5-tap instead of 9-tap Catmull-Rom (corners dropped, weights renormalised) | no | −30 | 73.2 / 71.7 dB alone |

`v_dot2_f32_f16` needs `VK_VALVE_shader_mixed_float_dot_product` enabled on the device: vkd3d-proton enables it; the
Vulkan layer now enables it on Vulkan and DXVK devices, and the proxy uses the dot2 pack only when that holds.

## Round 3: 2.49 → 2.34 ms

| Change | Exact? | µs |
| --- | --- | --- |
| Large bias literals routed through an SGPR (removes 112 `v_mov` per thread) | yes | postpass −18 |
| Quarter-rate `v_mul_lo_u32` in LDS/row addressing replaced by 24-bit multiplies and shifts | yes | −6 + pass1 share |
| Dead zeroing before stores that already go out of bounds removed, enabling the byte-select requantisation on more layers | yes | pass3 −9, pass2 −5 |
| Residual add folded into the dot4 chain (weight 2^shift) | yes | pass1 −12 (with the above) |
| Postpass LDS-fill loads issued together | yes | −16 |
| Postpass occupancy tuning | yes | 813 → 773 |
| Round-half-up requantisation (rounding constant folded into the bias; only exact ties differ, by +1 LSB) | no | −67 |
| Prepass micro-changes (`v_fma_mix` forms, dot2 chains starting from the bias) | no | −7.5 |

## Round 4: prepass and postpass, 2.39 → 2.26 ms

The round-3 set measured 2.39 ms on Vulkan this time (2.34 ms before: run-to-run drift of the machine), so the deltas
below are against it, run interleaved in the same session.

A per-stage breakdown (one stage stubbed at a time, everything else live) showed that neither pass is VALU-bound any
more. In the prepass the costs are the history colour fetch (113 µs), the int8 tensor and reprojected-colour stores
(112 together), depth dilation (74) and the convolution's LDS barrier (30 of its 69). In the postpass the network's
dot4 work now hides behind memory: the three output stores (100), the 9-tap filter (61) and the history blend (44).

| Change | Exact? | µs | Quality |
| --- | --- | --- | --- |
| Prepass pixel-to-wave mapping: 8×4 pixels per wave instead of 16×2 | yes | −36 | identical |
| Prepass tensor stores transposed, combined with the round-3 store changes | yes | −8 | identical |
| History reprojection: one bilinear fetch instead of the 5-tap Catmull-Rom | no | −78 | 69.1–72.7 dB; flicker and static-edge sharpness unchanged |

Same session, the round-3 set against round 4: Vulkan 2.39 → 2.26 ms, D3D12 sample 2.19 → 2.09, D3D11 replay
2.50 → 2.39, D3D12 replay 2.48 → 2.40 and 2.72 → 2.63 (two captured scenes).

## Round 5: 2.26 → 2.13 ms (output unchanged)

All three changes leave the output bit-identical to round 4 (16 flag combinations, the replay dumps with and without
auto exposure, D3D11 and D3D12), and the Inf / NaN replay dumps give the same frames.

| Change | µs (Vulkan) |
| --- | --- |
| Prepass with 2×2 pixels per lane: 64-lane workgroups for the same 16×16 tile, so the convolution needs no workgroup barrier; each lane runs the per-pixel stage on four pixels of a contiguous 8×4 block, then computes all 16 channels itself (one pixel of each 2×2 block per iteration, with stride-2 memory access, was 338 µs slower) | −50 |
| The proxy's colour clean-up for auto exposure moved into the shaders that read the colour: the prepass's colour transform is monotone and already maps NaN to the value of 0, so cleaning is one `v_min` per channel against the transform of 65504; AMD's exposure pass gets an Inf clamp (its luminance already goes through a NaN-ignoring max; a NaN select there changed finite results) | −35 net (pass 49) |
| Four of the five small `*_post` dispatches only zero a one-cell border ring around tensors that share memory: two rings are never read, two are zeroed by their producers (prepass, pass 2); the fifth stays (pass 1 has no binding with that tensor's size). The DLL records them, so the proxy drops them (pack options) | −23 |
| (the rest: barriers around the dropped dispatches) | ≈ −20 |

The DLL's command stream is 11 dispatches and 12 barriers, no copies or clears. Same session: round 4 2.26 → 2.13 ms
on Vulkan, 2.08 → 1.96 ms on the D3D12 sample, 2.45 → 2.32 ms on the D3D11 replay with auto exposure.

Lossy options measured on top (not in the shipped set), quality as the minimum over the 16 flag combinations:

| Change | µs | dB | Note |
| --- | --- | --- | --- |
| Postpass filter on 5 of the network's 9 taps (cross) | −82 | 65.6 | static edges 0.35% sharper; in motion ~25× the visibly-off pixels of the shipped set (below) |
| …and no depth dilation in the prepass | −128 | 63.0 | Vulkan 2.00 ms, D3D12 1.85 ms |
| Prepass colour taps reduced to one | −25 alone | 64.8 | worst µs per dB |

## Round 6: what bounds the postpass

Round 4's per-stage stubs suggested the postpass was not limited by instruction issue; that was wrong. A stub writes
garbage, the output images then compress worse (DCC), and the extra bandwidth hid the saved instructions. With DCC off
(`RADV_DEBUG=nodcc`; the baseline is unchanged at 750 µs) the stubs give: network trunk −211, filter −185, LDS fill
−76, stores −49, per-sub-pixel heads −48, history blend −21. Adding 256 dependent VALU instructions per thread costs
+79 µs, 0.31 µs each, as the issue model predicts. The executed path is about 2115 VALU per thread (the static count of
3548 includes fallback blocks that never run), which models 656 of the 734 µs; image traffic (about 32 bytes per
pixel) is the rest. Transcendentals cost about one issue slot each, occupancy is not the limit (80 VGPRs is slower), and
the tail of the last workgroups is about 10 µs.

| Change | Exact? | µs (Vulkan) |
| --- | --- | --- |
| Postpass waves entirely outside the image (8 of 400 rows) skip the network and sub-pixel loop (VGPR 128 → 96) | yes | −10 to −16 |
| Filter weights re-associated as `exp2(fma(..))` with shared per-row and per-column factors | no (about 72 dB from the shipped set; same results in the motion test) | a further −5 to −13, not shipped |
| Prepass colour taps from a per-wave LDS tile (−76 VALU per pixel) | yes | +36 or more: the prepass is latency- and bandwidth-bound at full occupancy |

## Quality in motion

The 16-combination test above replays 8 frames with little motion (median under a pixel per frame). A second test
replays captured camera pans (about 4, 8 and 16 px per frame, 32 frames each) and the moving-character scene through
the D3D12 path and compares 12 steady frames of each set against AMD's own shaders, in 8-bit display space:

| Set | Pans: PSNR / pixels off by more than 8/255 / frame-to-frame change vs AMD | Moving character |
| --- | --- | --- |
| AMD's shaders recompiled unchanged (method check) | 99 dB / 0% / 1.000× | 99 dB / 0% |
| Shipped (rounds 3–5 alike) | 54.5–55.4 dB / 0.003–0.009% / 0.995–1.000× | 56.1 dB / 0.008% |
| + 5-tap postpass filter | 50.0–50.9 dB / 0.18–0.24% (worst frame 0.69%) / 1.009–1.012× | 54.4 dB / 0.037% |
| + 5-tap filter, no depth dilation | 50.0–50.9 dB / 0.18–0.24% / 1.009–1.012× | 52.1 dB / 0.11% |

The 1-bilinear-tap history of round 4 costs nothing measurable in motion. The 5-tap filter's differences sit on thin
high-contrast detail while the camera moves (not visible side by side at 1:1, but about a quarter of a percent of the
pixels, and the static test understates it about 5×), so it is not shipped.

## What did not work (and why)

| Idea | Result |
| --- | --- |
| Coalescing the postpass output stores (as reported for the full INT8 network on RDNA3) | Stores alone are 3× faster as 8×4 blocks, but in the real shader they hide behind VALU work (stores off: only −56 µs), and every transpose variant raised VGPRs 80 → 128: slower overall |
| LDS-shared colour/depth taps in the prepass | Barrier and occupancy cost more than the saved fetches (+80–120 µs) |
| Pre-transforming the colour input at render resolution (an extra proxy pass) | Ceiling −36 µs, net ~−10–20 µs, impossible with auto exposure (its value is computed inside the same evaluate) |
| Packed f16 maths in the convolution or colour paths | Conversions cost as much as they save; f16 log/exp no faster than f32; HDR highlights overflow f16 |
| Postpass VGPRs 80 → 64 | ACO still allocates 80; a probe showed more waves would not help |
| LDS bank-conflict padding/swizzle in the prepass | Not LDS-bound: no gain or slower |
| Wave64 | All passes already compile as wave32; RDNA2 has no dual issue, so wave64 cannot add VALU throughput and would undo the wave-uniform weights |
| int4 weights (`v_dot8_i32_i4`) | Not reachable: RADV has no 8×4-bit dot operation and SPIR-V cannot express one; it also needs int4 activations. Ceiling if it were: 2.49 → 1.97 ms. Weight-only int4 quality per layer was measured anyway: 49–60 dB |
| Frame skipping (network on alternate frames) | Ruled out on purpose |
| Postpass filter with 5 or 4 taps instead of the network's 9 | −53 µs but 65.7–67.4 dB, and the 4-tap version shimmers more on static content |
| Postpass filter on the k largest of the 9 weights | Selecting them costs more than the skipped taps (+82 µs) |
| No depth dilation in the prepass | −70 µs, 65.1 dB worst case |
| Fusing pass3 into the postpass | Each pass reads a 3×3 neighbourhood of the previous one's output, so a fused shader must recompute a halo of pass3 per tile: at least +130 µs against a −47 µs ceiling |
| Fusing pass2 and pass3 | Ceiling only −7 µs before any halo recompute |
| Narrower intermediate formats | History and reprojected colour are RGBA16F holding three channels (no 3-channel half format exists; R11G11B10 is too coarse for log colour), the recurrent state is already 4 bytes per pixel; DCC off: 0 to −9 µs |
| Recomputing the reprojected colour in the postpass instead of storing it | The postpass's descriptor layout (the DLL's) has no history, velocity or depth bindings, and it would read as many bytes as it saves |
| Exact de-duplication of depth taps at exactly 2x | The sample's odd height (785) makes the y scale 0.4994, so the fast path never runs (+3 µs) |

## Limits

The int8 dot4 instructions fixed by the network are 65–73% of the executed VALU in pass1–3 and 41% in the postpass;
the dot-only floors are about pass1 300, pass2 120, pass3 160 and postpass 300 µs. What remains above them is
requantisation, residual shifts and address maths, close to minimal per output, plus the postpass's float colour maths.
Every pass now runs at 89–96% of its instruction-issue model or against its image traffic: pass 1–3 at 95–96% (their
floors set by the network's dot4 count; the int8 weights are dense, so pruning would remove almost no dot4s), the
postpass at about 89% plus its image traffic, and the prepass (2×2 pixels per lane, ~505 µs) latency- and
bandwidth-bound at full occupancy above a ~360 µs memory floor. What is left changes the output: fewer filter taps,
less depth dilation (both measured in round 5 and not shipped after the motion test), or formats and bindings fixed
by the DLL (the reprojected colour's round trip is about 16 MB per frame, and the postpass has no binding to recompute
it). Without a driver change exposing `v_dot8` (and an int4-activation network), FSR 4 Lite at 2x on Van Gogh/RADV
stays at about 2.1 ms for a 1280x785 output with the output unchanged.
