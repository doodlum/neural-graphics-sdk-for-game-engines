#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Acceptance checks on nfru_mfg.gpu_dataset_dump output (see README.md).

  --dumps   <root> containing N1..N5 (MFG build)
  --baseline <root> containing N1 from an unmodified v1.1.1 build (optional)
  --dataset  extracted input directory (reset copies, ground truth)
"""

import argparse
import math
import os
import re

import numpy as np


def load(path, dtype=np.uint8):
    return np.fromfile(path, dtype=dtype) if os.path.exists(path) else None


def psnr(a, b):
    mse = np.mean((a[..., :3].astype(np.float64) - b[..., :3].astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def frames_in(directory):
    return sorted({int(m.group(1)) for f in os.listdir(directory) for m in [re.match(r"frame_(\d+)_k1\.rgba8$", f)] if m})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    results = []

    def report(name, ok, detail=""):
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    counts = [n for n in range(1, 6) if os.path.isdir(os.path.join(a.dumps, f"N{n}"))]
    frames = frames_in(os.path.join(a.dumps, "N1"))
    w, h = map(int, open(os.path.join(a.dataset, "meta.txt")).readline().split()[1:3])

    # (a) N = 1 bit-identical to the baseline build
    if a.baseline:
        same = all(np.array_equal(load(os.path.join(a.dumps, "N1", f"frame_{f:04d}_k1.rgba8")),
                                  load(os.path.join(a.baseline, "N1", f"frame_{f:04d}_k1.rgba8"))) for f in frames)
        report("a_n1_bit_identical_to_baseline", same, f"({len(frames)} frames)")

    # (b) timesteps logged by the SDK equal k / (N + 1), outputs in presentation order
    for n in counts:
        log = open(os.path.join(a.dumps, f"N{n}", "sdk_log.txt")).read()
        lines = re.findall(r"event=dispatch_timesteps generated_frames=(\d+) multiplier=\d+x input_tensor_channels=(\d+) timesteps=([0-9.,]+)", log)
        expected = ",".join(f"{k / (n + 1):.4f}" for k in range(1, n + 1))
        ok = len(lines) >= len(frames) and all(int(g) == n and ts == expected for g, _, ts in lines)
        report(f"b_timesteps_N{n}", ok, f"expected {expected}, {len(lines)} dispatches")
        # presentation order: for interior frames, each output is closer to the previous rendered frame than the next one
        # in a monotone way when compared with the rendered frames (mean abs distance to previous colour increases with k)
        if n > 1:
            mono = 0
            total = 0
            for f in frames[1:]:
                prev = load(os.path.join(a.dataset, f"color_{f - 1:04d}.bin")).reshape(h, w, 4).astype(np.int16)
                d = [np.abs(load(os.path.join(a.dumps, f"N{n}", f"frame_{f:04d}_k{k}.rgba8")).reshape(h, w, 4).astype(np.int16) - prev).mean()
                     for k in range(1, n + 1)]
                mono += all(d[i] <= d[i + 1] + 1e-9 for i in range(n - 1))
                total += 1
            report(f"b_presentation_order_N{n}", mono == total, f"distance to previous frame increases with k on {mono}/{total} frames")

    # t = 0.5 of odd N equals N = 1 (same per-frame state, same timestep)
    for n in (3, 5):
        if n in counts:
            k = (n + 1) // 2
            same = all(np.array_equal(load(os.path.join(a.dumps, f"N{n}", f"frame_{f:04d}_k{k}.rgba8")),
                                      load(os.path.join(a.dumps, "N1", f"frame_{f:04d}_k1.rgba8"))) for f in frames)
            report(f"t_half_N{n}_equals_N1", same)

    # (d) histories identical for N = 1 and N = 5
    if 5 in counts:
        for item, dtype in (("dynmask.u8", np.uint8), ("depth.f32", np.float32), ("colour.rgba8", np.uint8)):
            paths = [(os.path.join(a.dumps, "N1", f"frame_{f:04d}_hist_{item}"), os.path.join(a.dumps, "N5", f"frame_{f:04d}_hist_{item}")) for f in frames]
            if all(os.path.exists(p) and os.path.exists(q) for p, q in paths):
                same = all(np.array_equal(load(p, dtype), load(q, dtype)) for p, q in paths)
                report(f"d_history_{item.split('.')[0]}_N1_equals_N5", same)

    # (e) reset frame: every output is a copy of presentColor
    for n in counts:
        present = load(os.path.join(a.dataset, "color_0000.bin"))
        ok = all(np.array_equal(load(os.path.join(a.dumps, f"N{n}", f"frame_0000_k{k}.rgba8")), present) for k in range(1, n + 1))
        report(f"e_reset_copies_N{n}", ok)

    # quality context: PSNR of t = 0.5 against the 60 fps ground truth
    vals = []
    for f in frames[1:]:
        gt = load(os.path.join(a.dataset, f"gt_t0.5_{f:04d}.bin"))
        if gt is not None:
            out = load(os.path.join(a.dumps, "N1", f"frame_{f:04d}_k1.rgba8")).reshape(h, w, 4)
            prev = load(os.path.join(a.dataset, f"color_{f - 1:04d}.bin")).reshape(h, w, 4)
            vals.append((psnr(out, gt.reshape(h, w, 4)), psnr(prev, gt.reshape(h, w, 4))))
    if vals:
        print("info: t=0.5 PSNR vs ground truth (NFRU, repeat-previous):", ", ".join(f"{x:.2f}/{y:.2f}" for x, y in vals))

    print(f"{sum(ok for _, ok in results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
