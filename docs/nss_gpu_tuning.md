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
`FFX_ML_N_SPLIT`, `FFX_ML_N_SPLIT_TARGET`, `FFX_ML_EXPERIMENT_NO_BARRIERS` (timing only, wrong results).

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
