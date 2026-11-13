#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Extract consecutive frames of an NFRU model-gym safetensors sequence into raw files for the SDK dump harness.

Run with ``python -I`` (it reads downloaded data). Writes, for frames start..start+count-1 of the sequence:

  Rendered frames are the 30 fps entries start, start + 2, ... (the safetensors store 60 fps; f30 motion spans two).

  color_NNNN.bin   RGBA8 UNORM, display size (rgb_reinhard, clamped to [0, 1], x255 rounded, alpha 255)
  depth_NNNN.bin   float32, MV/depth size: SDK device depth = 1 - safetensors depth (reverse-Z, 0 = far)
  mv_NNNN.bin      float32 x2 (R, G), MV/depth size, SDK backward motion in MV/depth pixels
                   (UV(previous) = UV(current) + motion / size); channel mapping chosen by a warp check, see meta.json
  gt_t0.5_NNNN.bin RGBA8, the 60 fps entry between rendered frames NNNN-1 and NNNN (ground truth for t = 0.5)
  meta.json        sizes, camera, per-frame view-projection (SDK column-major storage) and the MV mapping

The view-projection is stored so that SDK raw[c * 4 + r] = M[r][c] for the gym matrix M (clip = M @ point).
"""

import argparse
import json
import os
import struct

import numpy as np

DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "BOOL": np.bool_}


class SafeTensors:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n

    def get(self, key):
        e = self.header[key]
        return np.memmap(self.path, dtype=DTYPES[e["dtype"]], mode="r", offset=self.base + e["data_offsets"][0], shape=tuple(e["shape"]))


def sample_nearest(img, uv):
    h, w = img.shape[:2]
    x = np.clip(np.floor(uv[..., 0] * w).astype(np.int64), 0, w - 1)
    y = np.clip(np.floor(uv[..., 1] * h).astype(np.int64), 0, h - 1)
    return img[y, x]


def choose_mv_mapping(st, frame):
    """Pick the (channel order, sign) that best warps the previous colour onto the current one."""
    mv = np.asarray(st.get("mv_{}_f30_m1")[frame], dtype=np.float32)  # [2, H, W] pixels
    h, w = mv.shape[1:]
    cur = np.asarray(st.get("rgb_reinhard")[frame], dtype=np.float32)[:, ::2, ::2].transpose(1, 2, 0)
    # f30 motion points two entries back: the sequence is stored at 60 fps (odd entries are the 2x ground truth).
    prev = np.asarray(st.get("rgb_reinhard")[frame - 2], dtype=np.float32)[:, ::2, ::2].transpose(1, 2, 0)
    ys, xs = np.mgrid[0:h, 0:w]
    uv = np.stack([(xs + 0.5) / w, (ys + 0.5) / h], axis=-1)
    results = []
    for order in ("ch1_ch0", "ch0_ch1"):
        mx, my = (mv[1], mv[0]) if order == "ch1_ch0" else (mv[0], mv[1])
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                motion = np.stack([sx * mx / w, sy * my / h], axis=-1)
                err = float(np.abs(sample_nearest(prev, uv + motion) - cur).mean())
                results.append((err, order, sx, sy))
    zero = float(np.abs(prev - cur).mean())
    results.sort()
    return results[0], zero, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default=r"C:\dev\modelgym\data\arm-nfru\nfru\test\0000.safetensors")
    ap.add_argument("--start", type=int, default=2, help="first rendered (30 fps) entry; rendered frames are start, start + 2, ...")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out", default=r"C:\dev\modelgym\output\sdk_mfg\input")
    a = ap.parse_args()

    st = SafeTensors(a.safetensors)
    os.makedirs(a.out, exist_ok=True)
    (best_err, order, sx, sy), zero_err, all_results = choose_mv_mapping(st, a.start + 2)
    print(f"mv mapping: order={order} sign=({sx},{sy}) warp_err={best_err:.5f} (no-motion err {zero_err:.5f})")

    color_shape = st.header["rgb_reinhard"]["shape"]
    depth_shape = st.header["depth"]["shape"]
    frames = []
    for idx in range(a.count):
        i = a.start + 2 * idx
        rgb = np.asarray(st.get("rgb_reinhard")[i], dtype=np.float32).transpose(1, 2, 0)
        rgba = np.empty(rgb.shape[:2] + (4,), np.uint8)
        rgba[..., :3] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgba[..., 3] = 255
        rgba.tofile(os.path.join(a.out, f"color_{idx:04d}.bin"))
        if idx > 0:
            # 60 fps entry between this rendered frame and the previous one: ground truth for t = 0.5.
            gt = np.asarray(st.get("rgb_reinhard")[i - 1], dtype=np.float32).transpose(1, 2, 0)
            gt8 = np.empty(gt.shape[:2] + (4,), np.uint8)
            gt8[..., :3] = np.round(np.clip(gt, 0.0, 1.0) * 255.0).astype(np.uint8)
            gt8[..., 3] = 255
            gt8.tofile(os.path.join(a.out, f"gt_t0.5_{idx:04d}.bin"))

        depth = 1.0 - np.asarray(st.get("depth")[i, 0], dtype=np.float32)
        depth.astype(np.float32).tofile(os.path.join(a.out, f"depth_{idx:04d}.bin"))

        mv = np.asarray(st.get("mv_{}_f30_m1")[i], dtype=np.float32)
        mx, my = (mv[1], mv[0]) if order == "ch1_ch0" else (mv[0], mv[1])
        np.stack([sx * mx, sy * my], axis=-1).astype(np.float32).tofile(os.path.join(a.out, f"mv_{idx:04d}.bin"))

        m = np.asarray(st.get("ViewProj")[i], dtype=np.float32)
        frames.append({
            "source_frame": i,
            "img_id": int(st.get("img_id")[i]),
            "view_projection_column_major": [float(v) for v in m.T.reshape(-1)],
            "near": float(st.get("NearPlane")[i]),
            "far": float(st.get("FarPlane")[i]),
            "fov_y": float(st.get("FovY")[i]),
            "infinite_far": bool(st.get("infinite_zFar")[i]),
        })

    meta = {
        "source": os.path.abspath(a.safetensors),
        "display_size": [int(color_shape[3]), int(color_shape[2])],
        "mv_depth_size": [int(depth_shape[3]), int(depth_shape[2])],
        "depth": "SDK device depth = 1 - safetensors depth; reverse-Z (0 = far), use DEPTH_INVERTED (+ DEPTH_INFINITE when infinite_far)",
        "color": "rgb_reinhard clamped to [0,1], x255 rounded, RGBA8 UNORM",
        "motion_vectors": {
            "units": "MV/depth pixels, SDK backward motion (motionVectorScale = {1, 1})",
            "mapping": {"order": order, "sign_x": sx, "sign_y": sy},
            "warp_check": {"best_mean_abs_error": best_err, "zero_motion_error": zero_err,
                           "candidates": [list(r) for r in all_results]},
        },
        "jitter": "not applied (jitterOffset = 0, no jitter cancellation)",
        "frames": frames,
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    # Plain-text copy for the C++ harness: one key per line, frames as
    # "frame <index> <near> <far> <fov_y> <infinite 0/1> <16 floats, SDK column-major view-projection>".
    with open(os.path.join(a.out, "meta.txt"), "w") as f:
        f.write(f"display {meta['display_size'][0]} {meta['display_size'][1]}\n")
        f.write(f"mvdepth {meta['mv_depth_size'][0]} {meta['mv_depth_size'][1]}\n")
        f.write(f"frames {len(frames)}\n")
        for idx, fr in enumerate(frames):
            vp = " ".join(repr(v) for v in fr["view_projection_column_major"])
            f.write(f"frame {idx} {fr['near']!r} {fr['far']!r} {fr['fov_y']!r} {int(fr['infinite_far'])} {vp}\n")
    print("wrote", a.count, "frames to", a.out)


if __name__ == "__main__":
    main()
