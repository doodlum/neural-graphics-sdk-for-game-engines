# NSS GPU tuning notes (RTX 4080 reference)

How the native compute ML path (`sdk/src/backends/vk/ml`) was tuned on an RTX 4080, and how to repeat the process on
another GPU. Work in progress: the remaining ideas are listed at the end.

**Rule: output must stay bit-identical to the original.** Every ML-kernel change is checked with `ffx_ml_runtime_test`
(0 mismatches vs the CPU TOSA reference on every path). Every pre/post-process shader change must be checked with the
input replay harness below (`IDENTICAL` on all frames). Lossy types (fp8 etc.) are not used.

## Tools

| Tool | What it gives |
| --- | --- |
| `bin/ffx_ml_runtime_test.exe [--gpu N] [--width W --height H]` | Exactness of every path/type, plus timings. Use the **batched** column (64 graph runs per submit, 12 warm-up batches): single-submit timings drift with GPU clocks. |
| `ML_PROFILE=1 ffx_ml_runtime_test.exe --iterations 300` | Per-layer times (timestamps between layers; ~1 us granularity on NVIDIA). |
| `--width 64 --height 64` | Tiny graph: measures the fixed per-layer cost (barrier + one workgroup's latency chain). |
| `FFX_PROFILE_PASSES=<file>` (any app using the VK backend) | Per-job GPU time of every NSS pass (labelled `<index>_<job type>`), medians rewritten every 250 executions. |
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
