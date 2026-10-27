# NSS GPU tuning notes (RTX 4080 reference)

How the native compute ML path (`sdk/src/backends/vk/ml`) was tuned on an RTX 4080, and how to repeat the process on
another GPU. Work in progress: the remaining ideas are listed at the end.

**Rule: output must stay bit-identical to the original.** Every ML-kernel change is checked with `ffx_ml_runtime_test`
(0 mismatches vs the CPU TOSA reference on every path). Every pre/post-process shader change must be checked with
`ffx_nss_replay_test --suite --check` (`ALL IDENTICAL`, see the Steam Deck pre/post-process section) or the in-app
replay below. Lossy types (fp8 etc.) are not used.

## Tools

| Tool | What it gives |
| --- | --- |
| `bin/ffx_ml_runtime_test.exe [--gpu N] [--width W --height H]` | Exactness of every path/type, plus timings. Use the **batched** column (64 graph runs per submit, 12 warm-up batches): single-submit timings drift with GPU clocks. |
| `ML_PROFILE=1 ffx_ml_runtime_test.exe --iterations 300` | Per-layer times (timestamps between layers; ~1 us granularity on NVIDIA). |
| `--width 64 --height 64` | Tiny graph: measures the fixed per-layer cost (barrier + one workgroup's latency chain). |
| `FFX_PROFILE_PASSES=<file>` (any app using the VK backend) | Per-job GPU time of every NSS pass (labelled `<index>_<job type>`), medians rewritten every 250 executions. |
| `FFX_PROFILE_BARRIERS=1` (with `FFX_PROFILE_PASSES`) | Adds a `<n>_barriers` timestamp after each compute job's barriers, splitting barrier/transition cost from the dispatch. |
| `bin/ffx_nss_replay_test` (Linux, native VK, `sdk/src/components/nss/test`) | Deterministic synthetic replay through ffx-api: bit-exact gate for all Deck permutations (`--suite --save/--check DIR`, QUALITY/BALANCED/PERFORMANCE) and per-pass timings without a game (`--bench N --render WxH --quality 0\|1\|2`). |
| Streamline_Sample `-benchmark` + `tools/nss/run_nss_vs_dlss.sh` (in the Streamline_Sample nss-upscaler branch) | Upscaler GPU time in a real frame vs DLSS. |
| `tools/nss/nss_replay.sh record/replay` + `compare_raw.py` (Streamline_Sample nss-upscaler branch) | Bit-exact output check on identical recorded inputs (the sample's renderer is not deterministic run to run, so plain captures cannot be compared). |

Env overrides for experiments: `FFX_ML_PATH`, `FFX_ML_MATTYPE`, `FFX_ML_TILE_ROWS`, `FFX_ML_WEIGHTS_SHARED_LIMIT`,
`FFX_ML_N_SPLIT`, `FFX_ML_N_SPLIT_TARGET`, `FFX_ML_EXPERIMENT_NO_BARRIERS` (timing only, wrong results). Tiled dot4
kernel (AMD, see the Steam Deck section): `FFX_ML_DOT4_TILED`, `FFX_ML_DOT4_SUBGROUP`, `FFX_ML_DOT4_ACC`, `FFX_ML_DOT4_PX`,
`FFX_ML_DOT4_OCB`, `FFX_ML_DOT4_WGX`/`WGY`, `FFX_ML_DOT4_TARGET_THREADS`, `FFX_ML_DOT4_KSPLIT`/`KSPLIT_MAX`/`KSPLIT_FIRST`,
`FFX_ML_NO_SCALE_HI_ONLY`; `FFX_ML_DEBUG_LAYERS=1` prints each layer's tile and rescale parameters.

Runtime test helpers: `FFX_ML_TEST_PATHS=dot4,scalar,...` and `FFX_ML_TEST_MODELS=high|mid_low` limit what runs.
`ML_PROFILE=1` records 16 graph runs per submit and skips the first third of the iterations (clock ramp), so per-layer
numbers are taken at steady clocks.

## Where the time goes (Balanced, 960x540 -> 1920x1080, in-app, before this round)

| Pass | us |
| --- | --- |
| DepthScatter | 7 |
| DisocclusionMask | 8 |
| Preprocess | 28 |
| ML graph (14 conv layers) | 181 (151 isolated) |
| Postprocess | 69 |
| **Total** | **~296** (DLSS preset E: ~230) |

The graph is latency-bound, not compute-bound: ~150M MACs is ~1 us of tensor-core time. At a 32x32 input the graph
still costs ~85 us, and removing the inter-layer barriers (timing experiment) drops full size from 160 us to ~95 us.
So about 65 us is serialisation across the 14 dependent dispatches.

## What worked on the RTX 4080 (all bit-exact)

| Change | mid_low graph (isolated, batched) |
| --- | --- |
| Start of round (kernel v3, int8) | 0.155-0.163 ms |
| Patch staging issues all of a thread's global loads before any shared store | 0.154 (int8), 0.147 (fp16) |
| Output-channel split (grid z) for layers with < 128 workgroups (`splitTargetGroups`) | 0.141 (int8), 0.135 (fp16) |
| fp16 preferred over int8 on NVIDIA (`VendorTuning::matOrder`) | **0.134** auto-selected |

The high model went from 0.441/0.419 ms (int8/fp16) to 0.430/0.399 ms.

## What did not work on the RTX 4080

- Larger tiles (several rows per subgroup): occupancy collapse, 3.5x slower. One row per subgroup.
- Weights staged in shared memory per workgroup: more L2 traffic than it saves. Opt-in only.
- Software-pipelining the weight loads (next step's B tiles prefetched): no gain; the driver already overlaps them.
- Unrolling the tap/k loops: neutral for int8, fp16 regressed (register pressure).
- Splitting output channels on large layers: re-stages the same patch per split, slower (hence the target).
- vec4-typed shared arrays for `coopMatLoad`: wrong results on this driver (not portable). Scalar arrays only.

## Procedure for a new GPU

1. Run `ffx_ml_runtime_test` on it: every path must be EXACT. Note the batched times per matrix type and set the
   vendor's `matOrder` fastest-first in `TuningFor` (ffx_vk_ml_runtime.cpp).
2. Sweep `FFX_ML_N_SPLIT_TARGET` (64..2048) on both models; set `splitTargetGroups` (about 2 workgroups per
   SM/CU worked on the 4080).
3. Try `FFX_ML_TILE_ROWS=2` and `FFX_ML_WEIGHTS_SHARED_LIMIT=65536`; they lost on the 4080 but may win on GPUs with
   more shared memory per CU or slower L2.
4. Measure the fixed cost: run `--width 64 --height 64` with and without `FFX_ML_EXPERIMENT_NO_BARRIERS`. If
   serialisation dominates, layer fusion is the next lever (see below).
5. In-app: `FFX_PROFILE_PASSES` for the per-pass split, then `run_nss_vs_dlss.sh` for the end result.
6. For any shader change outside the graph: `nss_replay.sh record base`, apply the change, `nss_replay.sh replay new base`,
   `compare_raw.py` must print IDENTICAL.

## Next ideas (not done yet)

- Fuse consecutive same-resolution conv pairs ([5,6], [8,9], [11,12], ...) with halo recompute: removes a barrier,
  a dispatch and a global round trip per pair (~4-5 us each on the 4080).
- Postprocess (69 us at 1080p) is the largest non-graph pass: its per-pixel KPN weight reads are single bytes from an
  SSBO and it writes output and history separately. Any change must pass the replay check.
- Re-run the in-app benchmark after these graph changes (not re-measured yet).

## Steam Deck (AMD Van Gogh, RDNA2, 8 CUs, RADV)

No cooperative matrix on RDNA2, so the dot4 path runs. The device reports
`integerDotProduct4x8BitPackedSignedAccelerated`, and ACO emits `v_dot4c_i32_i8` (check with
`RADV_DEBUG=shaders ffx_ml_runtime_test ... 2> isa.txt`; this Mesa build prints ACO IR rather than LLVM disassembly).
Peak int8 dot4 is about 3.3 TMAC/s (8 CUs x 64 lanes x 4 MACs x 1.6 GHz).

### Before / after (batched ms, bit exact on every path)

| Graph input (test `--width x --height`) | high before | high after | mid_low before | mid_low after |
| --- | --- | --- | --- | --- |
| 960x544 (high 544x960, mid_low 272x480) | 33.2 | **2.92** | 6.72 | **0.77** |
| 640x400, Deck Performance (high 400x640, mid_low 200x320) | 14.7 | **1.50** | 3.12 | **0.47** |
| 864x544, Deck Quality (high 544x864, mid_low 272x432) | 31.7 | **2.64** | 6.05 | **0.69** |

"Before" is the untiled dot4 kernel (`FFX_ML_DOT4_TILED=0`), which other vendors still use. The batched numbers
include the Deck's sustained-load clocks and the shared 15 W CPU+GPU budget; expect +-5% between runs.

Per layer (ML_PROFILE, ms; old -> new):

| Layer | 640x400 high | 640x400 mid_low | 864x544 high | 864x544 mid_low |
| --- | --- | --- | --- | --- |
| 0 conv s2 12->32 | 1.014 -> 0.131 | 0.292 -> 0.054 | 1.711 -> 0.216 | 0.537 -> 0.069 |
| 1 conv 32->32 | 2.980 -> 0.238 | 0.766 -> 0.106 | 5.382 -> 0.429 | 1.393 -> 0.138 |
| 2 conv s2 32->32 | 1.651 -> 0.076 | 0.222 -> 0.034 | 4.707 -> 0.129 | 0.576 -> 0.051 |
| 3 conv 32->32 | 0.760 -> 0.068 | 0.104 -> 0.033 | 1.383 -> 0.115 | 0.230 -> 0.047 |
| 4 conv s2 32->64 | 0.451 -> 0.050 | 0.062 -> 0.027 | 1.180 -> 0.082 | 0.173 -> 0.032 |
| 5 conv 64->64 | 0.825 -> 0.092 | 0.112 -> 0.035 | 2.166 -> 0.180 | 0.335 -> 0.048 |
| 6 conv 64->32 | 0.409 -> 0.038 | 0.065 -> 0.029 | 1.048 -> 0.084 | 0.169 -> 0.027 |
| 7 conv 32->16 (resize) | 0.332 -> 0.040 | 0.096 -> 0.020 | 0.593 -> 0.065 | 0.162 -> 0.024 |
| 8 conv 48->32 (concat) | 0.960 -> 0.098 | 0.152 -> 0.044 | 1.770 -> 0.164 | 0.342 -> 0.061 |
| 9 conv 32->36/16 | 0.855 -> 0.088 | 0.062 -> 0.020 | 1.552 -> 0.153 | 0.117 -> 0.027 |
| 10 conv 32->16 (resize) | 1.275 -> 0.134 | 0.330 -> 0.061 | 2.319 -> 0.240 | 0.592 -> 0.076 |
| 11 conv 48->16 (concat) | 1.910 -> 0.179 | 0.479 -> 0.082 | 3.432 -> 0.322 | 0.903 -> 0.109 |
| 12 conv 16->16 | 0.384 -> 0.072 | 0.103 -> 0.035 | 0.697 -> 0.129 | 0.182 -> 0.046 |
| 13 conv 16->4 (resize) | 0.672 -> 0.107 | 0.177 -> 0.046 | 1.219 -> 0.198 | 0.314 -> 0.067 |
| Sum | 14.48 -> 1.41 | 3.02 -> 0.62 | 29.16 -> 2.51 | 6.02 -> 0.82 |

The large layers now reach about 2.4-2.5 TMAC/s (~75% of the dot4 peak). Layer 0 (12 input channels) and layer 13
(4 output channels, 1 MB in and out) are close to memory bound. At Deck sizes, removing every barrier
(`FFX_ML_EXPERIMENT_NO_BARRIERS`) would save only ~7% (high) and ~16% (mid_low), so serialisation is no longer the main cost.

### What the old kernel did wrong

One thread per (pixel, 4 output channels): every input word was loaded once per 4 dot4s, each dot4 sat behind its own
branch and an `s_waitcnt`, and a tiny graph (32x32) cost ~20 us per layer, because each thread's dependent load chain was
9 x IC/4 steps long.

### What worked (all bit exact)

| Change | Effect |
| --- | --- |
| New `dot4_tiled` kernel (`ML_DOT4_TILED`): each thread owns PX pixels x OCB output channels; all threads of a workgroup share the channel block, so weight addresses are wave uniform and become `s_buffer_load_dwordx16` + SGPR dot4 operands; weights laid out `[block][tap][icWord][OCB]` | 960x544: 33.2 -> 6.6 ms (high), 6.7 -> 2.2 ms (mid_low) |
| Loads made unconditional (clamped addresses, select afterwards; a `?:` around a load compiled to a branch per load), next input word prefetched, separate branch-free loops per concat source, zero-weight padding words skipped | concat layers 2-3x faster; high 6.6 -> 5.5 ms |
| Tile shape sweep: 1 pixel x 32 channels per thread for wide layers, wave64 (`requiredSubgroupSize` 64) | high 5.5 -> ~3.0 ms |
| Pixels per thread = 32 accumulators / OCB (OC 16 layers take 2 pixels, the OC 4 output layer takes 8) | 640x400: high 2.28 -> 1.59, mid_low 0.71 -> 0.48 ms |
| Small layers get narrower channel blocks (more threads) up to 4096 threads; on the tiniest layers the input words are additionally split across up to 4 subgroups of one workgroup (local z) with an exact shared-memory reduction (`subgroupBroadcastFirst` keeps the split index scalar) | mid_low at 640x400 ~ -12%; 128x128 graph 0.34 -> 0.17 ms |
| Epilogue: all NSS rescale shifts are 35..49 with no double rounding, so the rounding constant lies wholly in the high word of the 64-bit product: `(mulhi + 2^(s-33)) >> (s-32)` is exact and drops the quarter-rate `v_mul_lo_u32` and the carry chain (host checks per layer, spec constant `SCALE_HI_ONLY`) | 5-7% |

### What did not work

- Splitting the input words across adjacent *lanes* (subgroup-shuffle reduce-scatter): lanes then read different weight
  words, which turns every scalar weight load into a vector load. 2-5x slower; replaced by the per-subgroup split.
- Large register tiles (PX x OCB = 64..256 accumulators): occupancy collapses; 256 accumulators in wave64 spill, and
  that configuration (wave64, PX 8, OCB 32) also produced wrong results on this driver, so tiles are capped by the
  32-accumulator budget.
- Splitting input words before narrowing the channel blocks (`FFX_ML_DOT4_KSPLIT_FIRST=1`): slower on every size tried.
- Workgroup shape: 8x8, 16x4 and 16x8 are within noise; 64x1 and 4x16 are 5-10% slower. 8x8 is kept.
- wave32 vs wave64: within a few percent on the big layers. wave64 wins on mid_low with the final tiles.

### Ideas left

- Fusing consecutive same-resolution layers with halo recompute: at most the barrier share above (7-16%), plus the
  saved intermediate traffic for the memory-bound layers 0/1 and 12/13.
- Table lookups (layers 9 and 13) still go through a per-output vector load; an LDS copy of the 256-byte table may help those two layers.
- Layer 0 (12 input channels, stride 2) runs at ~1.6 TMAC/s: a dedicated variant that stages the 3-word pixels in LDS
  could approach the memory bound.
- A weights-in-LDS variant (VGPR weight operands, `ds_read_b128` broadcast) to take the scalar-cache latency off the
  `s_waitcnt` in the inner loop; not tried.
- Re-measure in a real frame (`FFX_PROFILE_PASSES`) on the Deck: the pre/post passes were not touched here.

### Pre/post-process passes (non-ML), Steam Deck

In a real frame (Streamline sample under Proton, DLSS Performance = NSS PERFORMANCE 640x400 -> 1280x800, mid_low,
compute path) the non-ML passes were ~73% of the 1.68 ms: depth scatter 32 us, disocclusion mask 204 us, preprocess
186 us, ML graph 452 us, postprocess 806 us.

**Harness.** `bin/ffx_nss_replay_test` creates the NSS context through ffx-api with the proxy's flags (QUANTIZED,
MANAGE_HISTORY, DEPTH_INVERTED, DEPTH_INFINITE, HDR on/off, compute path), feeds 16 synthetic frames (gradients, stripes
and noise, a depth field with box discontinuities, a panning camera plus a moving textured object with its own motion,
the NSS jitter sequence, resets on frames 0 and 9) and reads the output back after every frame (RGBA16F, so stores are
not hidden by the R11G11B10 rounding; one config also with an R11G11B10 output).

    # once, with the unmodified code
    ./bin/ffx_nss_replay_test --suite --save /path/base
    # after every change: every frame, pixel and channel of 9 configs
    # (BALANCED/PERFORMANCE x 640x400/742x464 -> 1280x800 x HDR off/on, + R11G11B10 output)
    ./bin/ffx_nss_replay_test --suite --check /path/base        # must print ALL IDENTICAL
    # per-pass GPU medians (FFX_PROFILE_PASSES), run >= 1500 frames so the Deck's clocks settle
    ./bin/ffx_nss_replay_test --bench 2000 --render 640x400 --quality 2

Labels in the bench output (no reset): `0_clear_uint`, `1_compute` depth scatter, `2_compute` disocclusion mask,
`3_compute` preprocess, `4_data_graph`, (`5_compute` dynamic LUT for non-2x), then postprocess. The Deck drifts by
~3% with temperature; compare builds by running them interleaved (copy `bin/libngsdk_linux_x64.so` +
`bin/ffx_nss_replay_test` per build and alternate). `RADV_DEBUG=shaders,shaderstats` dumps ACO IR and VGPR/occupancy
per pipeline.

**Before / after** (us, medians of interleaved harness runs, PERFORMANCE unless noted; the harness reproduces the
in-game numbers except the disocclusion mask, whose old cost depended on the scene):

| Pass | 640x400 before | 640x400 after | 742x464 before | 742x464 after |
| --- | --- | --- | --- | --- |
| Depth scatter | 28 | 28 | 38 | 38 |
| Disocclusion mask | 413 | **29** | 468 | **39** |
| Preprocess | 202 | **167** | 255 | **212** |
| ML graph (unchanged code) | 453 | 471 | 587 | 598 |
| Dynamic offset LUT | - | - | 10 | 10 |
| Postprocess | 803 | **415** | 886 | **634** |
| Total | 1909 | **1115** | 2251 | **1535** |
| Postprocess, BALANCED (Catmull history) | 932 | **544** | 1020 | **757** |

The ML graph reads 2-4% slower only because the GPU is busier per unit time (shared 15 W budget, lower clocks); its
code is untouched. All rows are bit exact against the original on every harness configuration.

What each change gave (all bit exact, A/B on the Deck):

| Change | Effect |
| --- | --- |
| Disocclusion mask: the 4x4 nearest-depth search loaded each depth and then, behind a branch, its motion vector (~36 dependent fetches per thread). Load the 17 depths up front, pick the same sample with the same comparison, fetch one motion vector; issue the 4 previous-depth loads together | 413 -> 27 us |
| Postprocess: the LR->KPN coordinate mapping used a generic 32-bit integer divide per tap (8 per pixel, ~20 VALU each with quarter-rate multiplies). Float quotient of `n*k + 0.5` times `1/d` is exact while `n*k < 2^20` (Vulkan's 2.5 ulp division included); integer divide otherwise | 803 -> 645 us |
| Postprocess 2x: the four sparse taps are always the 2x2 LR block {lo, lo+1}^2, so map 2 values per axis instead of 4 taps; KPN coefficients read as 32-bit words + one `bitfieldExtract` instead of a per-lane select between bytes (all modes) | 645 -> 511 us (2x), 729 -> 710 (742x464) |
| Postprocess: power-of-two input/KPN ratio (640 -> 80, 400 -> 50, 464 -> 58) maps with a shift (uniform check) | 511 -> 469 us |
| Preprocess (half res): the packed nearest-offset quad recomputed the 4 footprint nearest-depth searches the history lanes had just done; the centre search equals lane (1,1)'s for even sizes (checked per pixel) | 201 -> 166 us |
| Preprocess: 8x8 workgroups (320x200 was 260 16x16 groups with a half-empty last row; 1000 groups now) | -3.5% |
| Postprocess 2x sparse: one thread per 2x2 output quad; at exactly 2x the quad's pixels share one input pixel, so the nearest-offset decode and dilated motion fetch run once (per-pixel fallback otherwise); the tap coordinates of the quad come from {q-1, q, q+1} per axis | 478 -> 415 us |
| Postprocess dynamic LUT: `output_px % IndexModulo` and the per-tap KPN mapping without integer divides | 683 -> 629 us (742x464) |

How the bottlenecks were found: with `FFX_PROFILE_BARRIERS` the barriers before each pass are 1-5 us, so the time
is in the shaders. Stubbing parts of the postprocess showed it was VALU bound (~530 VALU per pixel at the start: an
empty shader is 21 us, and removing the loads changed nothing once the integer divides were gone); the disocclusion
mask was latency bound (a chain of dependent fetches in 70 workgroups).

What did not help (measured, reverted):

- Workgroup shape for postprocess (8x8, 32x8 vs 16x16) and for depth scatter / disocclusion (8x8): within noise.
- wave32 (`RADV_PERFTEST=cswave32`): postprocess unchanged, preprocess -2%; not worth a pipeline change.
- Output format/bandwidth: RGBA16F vs R11G11B10 output costs the same, the postprocess is not store bound.
- DCC: `RADV_DEBUG=nodcc` changes nothing (the history GENERAL <-> SHADER_READ_ONLY transitions are cheap).
- Tracking the preprocess nearest offset as one integer code instead of a float2 blend: -5% VALU, no time change
  (preprocess is latency/occupancy bound, 64 VGPRs = 16 waves/SIMD).
- Precomputing KPN row offsets per quad / via fp32 FMA: within noise.

Ideas left:

- Preprocess (~167 us, latency bound): the four footprint searches could share one 5x5 depth window (9 gathers instead
  of 16) and the 4-lane history chain (gathers -> nearest -> motion -> history sample, ~19 us) is the longest
  dependency; lowering VGPRs below 64 would raise occupancy.
- Postprocess (~415 us at 2x, VALU bound, ~300 VALU/pixel left, most of it the moments/history clamp/tonemap math
  that must stay as is): the remaining shareable work in the quad is small. The dynamic-LUT (non-2x) path still maps
  4 arbitrary taps per pixel.
- Depth scatter + disocclusion (~57 us together) need a global barrier between them (scattered atomics), so they
  cannot simply be merged.
- The two history clears on a reset frame cost ~66 us each (1280x800 R11G11B10 `vkCmdClearColorImage`); a faster
  clear would shave a one-frame spike on camera cuts (they cannot simply be skipped: the reset blend multiplies the
  old history by 0, and uncleared memory could hold NaNs).
- Re-measure in the game: the harness scene's disocclusion cost was 2x the game's, the other passes matched within ~5%.


### Full network (QUALITY), Steam Deck

The Steam Deck proxy now runs QUALITY (the `high` graph, full-res preprocess, dense 9-tap filter, Catmull history) for
every DLSS mode. Measured with the replay harness, 1280x800 output, interleaved runs of the original `steamdeck` build
and this one (us, steady-state medians of 1500 frames):

| Pass | 640x400 before | 640x400 after | 853x533 before | 853x533 after |
| --- | --- | --- | --- | --- |
| Depth scatter (full res) | 32 | 32 | 64 | 64 |
| Preprocess (full res) | 245 | 247 | 432 | 442 |
| ML graph (high) | 1531 | 1529 | 2640 | 2630 |
| Dynamic offset LUT | - | - | **2153** | **0.1** |
| Postprocess | 764 | 769 | **1682** | **1111** |
| Total | 2577 | 2581 | **6981** | **4251** |

640x400 is an exact 2x ratio (static LUT, NSS_FILTER_MODE 0); 853x533 is a general ratio (dynamic dense LUT, mode 2).
All changes are bit identical: the replay suite (now 13 configs, with QUALITY at 640x400 and 853x533 -> 1280x800, HDR
off/on), 64-frame runs at eight more QUALITY ratios (640x400, 720x450, 742x464, 853x533, 960x600, 1000x700,
1066x666) plus BALANCED/PERFORMANCE spot checks compared by hash, and `ffx_ml_runtime_test` EXACT on every path.
Bench labels for QUALITY: `1_compute` depth scatter, `2_compute` preprocess, `3_data_graph`, `4_compute` dynamic LUT
(non-2x only), then postprocess.

What worked:

| Change | Effect |
| --- | --- |
| Dynamic dense LUT made separable. The LUT held 9 packed taps for each tile of the reduced fraction, which is the full 1280x800 for 853 -> 1280 (853 is prime), 48 B each, rebuilt every frame with up to 36 candidates per thread. Tap validity and LR offsets depend on x and y independently and the candidate order is fixed, so the LUT now holds one texel per x tile and one per y tile (6 offsets + a 36-bit mask over candidate ranks). The postprocess ANDs the two masks and takes the first 9 ranks: the same taps in the same order, with the per-axis float math unchanged | LUT 2153 -> 0.1 us, postprocess 1682 -> 1427 us |
| Postprocess (mode 2): the 9 taps are resolved before any fetch, and the LR clamping + KPN mapping (~30 VALU per tap) is done once per workgroup column and row into LDS (same functions); each tap reads 2 words | 1427 -> 1100 us |
| Graph: the activation TABLE is copied to shared memory per workgroup (one SSBO load per output value before), and at most 4 pixels per thread (the 16 -> 4 output layer) | layer 13 at 544x864: 198 -> 185 us |
| Graph: channel blocks up to 40, so the 36-channel KPN layer is one block instead of 2 x 20 | layer 9 at 136x216: 155 -> 140 us |

What did not work (measured, reverted):

- Mode 2 postprocess: only unrolling the tap loop and issuing all fetches first (80 VGPRs, 12 waves): no change. The
  pass is VALU bound (~800 VALU/pixel, which predicts ~0.97 ms against 1.1 ms measured), not latency bound.
- Exact shift for the KPN mapping (`floor((t + 0.501) * 0.25) == t >> 2`; `_KpnScale` is always exactly 1/4): slower
  (765 -> 808 us at 640x400) as a uniform branch inside the tap code, because ACO does not schedule loads across basic
  blocks; neutral with the branch hoisted around the whole filter. The mapping is not the cost.
- Mode 0 (exact 2x) postprocess, VALU bound (537 VALU/pixel, ~88% of the measured time). Timing stubs: the KPN weight
  path (coordinates, address, byte read, dequant) is ~230 of 764 us; the colour loads only ~20 us. ACO already reduces
  the 9 taps to 3 + 3 distinct coordinates. Staging the 11x11 LR window and its KPN words in LDS: 764 -> 891 us; KPN
  words only in LDS: 886 us; per-column/row coordinates in LDS: neutral.
- Preprocess (full res): sharing the 5-tap YCoCg derivative neighbours through LDS: 442 -> 458 us. The pass is at the
  memory floor: ~55 B of unique traffic per pixel (colour, depth, motion, bilinear history at output resolution,
  derivative/feedback/depth history, 17 B of outputs) is ~26 MB at 853x533, i.e. ~0.43 ms at ~60 GB/s.
- Graph: 16-channel blocks for the high model at 640x400: 1.44 -> 1.84 ms. 16 pixels per thread on the narrow layers
  (256 accumulators, spills) gave wrong results on this driver, like the earlier wave64 PX 8 x OCB 32 case; the tile
  choice caps accumulators at 32 so it never goes there.

The graph's big layers stay at ~75% of the int8 dot4 peak; the rest of its time is the quarter-rate `v_mul_hi_i32` of
the exact rescale, the low-IC first layer and the memory-bound full-res last layer.

Ideas left:

- Mode 2 postprocess: the selected ranks depend only on the 6-bit x and 6-bit y validity patterns, so a 4096-entry
  table (9 x 6-bit ranks + count) generated with the LUT would replace the per-tap `findLSB` extraction (~9% of the
  pass by count).
- Mode 0 postprocess: the KPN weight path is the largest exact-reducible part (~30%); a layout that groups the 9
  channels a pixel reads, or reading the KPN as 32-bit words per tap pair, could cut the address/extract work.
- Async compute: the history warp/Catmull part of the postprocess does not depend on the graph and could overlap the
  graph's barriers on a second queue.
- Graph: fusing the full-res tail (layers 12 -> 13) would remove one 1.1 MB round trip and a barrier at 640x400.

### Full network (QUALITY), Steam Deck, round 2

Interleaved runs of `steamdeck` c6bc270 and this round (us, 1500-frame medians, 3 alternating runs averaged,
1280x800 output):

| Pass | 640x400 before | 640x400 after | 853x533 before | 853x533 after |
| --- | --- | --- | --- | --- |
| Depth scatter | 32 | 32 | 64 | 64 |
| Preprocess | 245 | 246 | 443 | 445 |
| ML graph (high) | 1515 | **1459** | 2630 | **2547** |
| Dynamic offset LUT | - | - | 0.1 | 0.1 |
| Postprocess | 762 | **675** | 1110 | **902** |
| Total | 2560 | **2417** | 4253 | **3969** |

The graph changes also apply to BALANCED/PERFORMANCE (mid_low): PERFORMANCE 640x400 graph 464 -> 435 us, BALANCED
742x464 graph 593 -> 555 us. All bit exact: replay suite (quality 0/1/2) ALL IDENTICAL against a c6bc270 baseline,
64-frame hashes identical at 8 more QUALITY ratios, `ffx_ml_runtime_test` EXACT on every path at 960x544, 864x544,
640x400 and 128x128.

The postprocess is VALU bound, and its time follows the VALU count closely: at 1280x800, one VALU per pixel costs about
1.2 us (1M pixels / 16 SIMDs, wave64 = 2 cycles, 1.6 GHz). A timing stub that only removed the 9 KPN loads (adding 9
multiplies) got 44 us slower, so memory is not the limit there.

What worked:

| Change | Effect |
| --- | --- |
| KPN tap mapping `floor((t + 0.5 + 1e-3) * scale)` replaced by `t >> 2`: `_KpnScale` is exactly 1/4 by construction (KPN = dataGraph / 4, dataGraph a multiple of 8; now asserted on the host). Last round's attempt was a runtime branch; this is unconditional (`NSS_KPN_SCALE_QUARTER`) | exact-2x postprocess 537 -> 513 VALU, 761 -> 733 us |
| Exact-2x filter: the channel arithmetic of the 9 KPN reads (base + c, >> 2, (& 3) << 3) from 4 per-pixel values, the constant part folded into the load offset; the double clamp of KPN coordinates removed (taps are >= 0) | 513 -> 472 VALU, 735 -> 683 us |
| KPN pixel word as y * (w * 9) + x * 9 (shared row/column parts) | 472 -> 466 VALU, ~ -5 us |
| Dynamic-LUT filter: a 64x64 selection table (in rows 2..65 of the LUT image, written by the LUT pass) gives a pixel's taps from the 6-bit validity of its x and y tiles; replaces 9 x (findLSB on a 36-bit mask + constant-table lookup) | 853x533 postprocess 1100 -> 882 us |
| Graph epilogue: output zero point folded into the rounding constant (`(hi + r + (zp << k)) >> k`) | high graph batched -2.5% |
| Graph: zero-point pixel after every internal tensor (filled once), padding taps point at it, so the zero point is selected once per column instead of once per loaded word | -2% |
| Graph: when OUT_C is a multiple of the channel block, the per-output `oc < OUT_C` guard is compile-time true (no per-output branch, descriptor reload and scalar loads) | -2% |

What did not work (measured, reverted):

- Dequantised KPN weights as a 256-entry LDS table (exact-2x filter): -5 us on the pass, within noise of the total.
- Folding the conv bias into the accumulator start: slower (scalar loads + moves for 32 accumulators).
- Hoisting the column part of the input addresses out of the kernel-row loop: no change (ACO already shares it).
- Thread target 8192/16384 with a 16-channel floor for small layers, and wave32 for the high graph: faster per layer
  in ML_PROFILE, not in the batched/in-frame runs.
- Not attempted, with reasons: fusing graph layers 12 -> 13 (a 16x16 tile of layer 13 needs 10x10 layer-12 pixels
  instead of 8x8: ~1.56x recompute of a compute-bound layer, more than the saved round trip); async compute for the
  history warp (the ffx-api VK backend records into the caller's command buffer and never sees a queue, so this
  needs API/proxy work); quad-per-thread exact-2x postprocess (estimated ~17 VALU/pixel, ~20 us, after the changes
  above).

Floors at 640x400: the preprocess moves ~61 B per LR pixel (~15.6 MB per frame, ~0.25 ms at ~63 GB/s); the graph's
large layers run at ~75% of the int8 dot4 peak; the postprocess is down to ~466 VALU per pixel, most of it the
moments/history/tonemap math that must stay as written.

### Non-exact experiment (QUALITY, Steam Deck): what dropping bit exactness buys

Everything here is opt-in and off by default (the replay suite stays ALL IDENTICAL and `ffx_ml_runtime_test` EXACT
with the switches unset). Switches: `--fp16` in the harness / `FFX_API_NSS_CONTEXT_FLAG_ALLOW_16BIT` in the API (fp16
shader permutations for every NSS pass), `FFX_NSS_FP16_PASSES=pre,post,...` (fp16 permutation per pass) and
`FFX_ML_RESCALE_FLOAT=1` (graph requantisation as `round(acc * s + o)` in fp32 instead of the exact 64-bit TOSA
rescale; same int8 weights). Error is measured with `ffx_nss_replay_test --compare DIR` against the exact output: RGB
of the RGBA16F output, PSNR with peak 1 (SDR) or the reference maximum (~36, HDR). The 64-frame runs use exact
64-frame baselines; frame 16 of the 64-frame sequence differs from the 16-frame suite sequence (resets move).

Time saved (interleaved A/B, 1500 frames; negative = faster) and error (SDR / HDR PSNR):

| Candidate | 640x400 total | 853x533 total | SDR frame 16 / 64 | HDR frame 16 / 64 | SDR mean abs diff @64 |
| --- | --- | --- | --- | --- | --- |
| fp16 everywhere (`--fp16`) | **+88 us** (post +90, pre -11) | +30 us (post +205, pre -39) | 45.8 / 45.1 dB (640), 47.5 / 46.8 (853) | 59.6 / 59.1, 61.2 / 60.5 | 0.0037 / 0.0028 |
| fp16 preprocess only | -11 us (pass), total in noise | -22 us (pass), total in noise | - / 49.5 dB, - / 50.5 | - / 60.5, - / 61.1 | 0.0020 / 0.0017 |
| fp16 postprocess only | slower (see first row) | slower | - / 45.2 dB, - / 47.0 | - / 59.4, - / 60.7 | 0.0036 / 0.0028 |
| fp32 requantisation (`FFX_ML_RESCALE_FLOAT`) | **-40 us** (graph -36) | **-96 us** (graph -83) | 50.5 / 49.8 dB, 51.6 / 50.6 | 61.1 / 60.5, 61.9 / 61.2 | 0.0019 / 0.0016 |
| fp32 requant + fp16 preprocess | ~-30 us | -105 us | 50.2 / 49.7 dB, 50.8 / 50.4 | 60.4 / 60.8, 61.4 / 61.1 | 0.0019 / 0.0017 |

In the 16-frame suite (same sequence as the exact gate) fp32 requantisation gives 56.4 / 57.1 dB SDR at frame 16
(640x400 / 853x533); it changes 1.4% of the graph's int8 outputs, by up to 11 steps after propagation. The SDR mean abs
error of ~0.002 is about half an 8-bit step; maxima are isolated pixels (0.05-0.9 in SDR).

Why fp16 does not pay here: the stock fp16 postprocess has more VALU than the exact fp32 one (520 vs 466): texel
fetches return fp32 and are converted, the dequantisation is written in fp32 and converted, and integer KPN addressing
picks up quarter-rate `v_mul_lo_u32`; the packed `v_pk_fma_f16` savings are smaller than that. A hand-written packed
filter would save at most ~6 of ~9 accumulate ops per tap (~30 us at 640x400) at a similar error.

Recommendation: keep exact. The only consistent win is fp32 requantisation, 1.6% (640x400) / 2.4% (853x533) of the
frame for ~50 dB PSNR after 64 frames; with temporal feedback any perturbation settles around that level.
