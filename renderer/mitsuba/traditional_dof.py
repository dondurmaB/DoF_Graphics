#!/usr/bin/env python3
"""Legacy approximate preview gather used by the browser viewer.

It is NOT the production OpenGL shader and is NOT used for stage-2 results.
The command-line comparison now requires authenticated stage2_experiment.py
artifacts, including independently sampled references and measured noise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Legacy preview pattern; NOT equivalent to the production OpenGL gather.
_I = np.arange(64) + 0.5
DISK = np.stack([np.cos(_I * 2.39996323), np.sin(_I * 2.39996323)], 1) * np.sqrt(_I / 64)[:, None]
DISK_LEN = np.linalg.norm(DISK, axis=1)
MAX_RADIUS_PX = 32.0
SKY_DEPTH_M = 1e4


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def signed_coc_radius_px(depth, focal_m, sensor_m, f_number, focus_m, height_px):
    """Thin-lens CoC radius in pixels; negative in front of the focus plane."""
    aperture = focal_m / f_number
    coc_sensor = aperture * focal_m * (depth - focus_m) / (depth * focus_m)
    return 0.5 * coc_sensor / sensor_m * height_px


def bilinear(img, x, y):
    """Sample img (H, W, C) at float pixel coords with clamp-to-edge."""
    h, w = img.shape[:2]
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def gather_dof(rgb, depth, focal_m, sensor_m, f_number, focus_m, max_radius=MAX_RADIUS_PX):
    """Legacy preview approximation; not accepted as experimental evidence."""
    h, w = depth.shape
    rgb = rgb.astype(np.float32)
    depth = np.where(np.isfinite(depth), depth, SKY_DEPTH_M).astype(np.float32)
    radius = signed_coc_radius_px(depth, focal_m, sensor_m, f_number, focus_m, h).astype(np.float32)
    blur = np.minimum(np.abs(radius), min(max_radius, 32.0))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    packed = np.concatenate([rgb, depth[..., None], radius[..., None]], -1)
    centre_fg = radius < 0.0

    total = rgb.copy()
    weight_sum = np.ones((h, w, 1), np.float32)
    for (dx, dy), dl in zip(DISK, DISK_LEN):
        s = bilinear(packed, xs + dx * blur, ys + dy * blur)
        s_rgb, s_depth, s_radius = s[..., :3], s[..., 3], s[..., 4]
        delta = s_depth - depth
        similar = 1.0 - smoothstep(0.05, 1.4, np.abs(delta))
        keep_bg_off_fg = np.where(centre_fg, 1.0 - smoothstep(-0.05, 0.25, delta), 1.0)
        big_fg_occluder = np.where(s_radius < -0.5, smoothstep(0.2, 1.2, np.abs(s_radius)), 0.0)
        reach = smoothstep(dl * blur - 0.75, dl * blur + 0.75, np.abs(s_radius))
        wgt = (0.18 + 0.82 * similar)
        wgt *= np.maximum(keep_bg_off_fg, big_fg_occluder * 0.85)
        wgt *= np.maximum(reach, similar * 0.65)
        wgt = np.where(blur >= 0.5, wgt, 0.0)[..., None].astype(np.float32)
        total += s_rgb * wgt
        weight_sum += wgt
    return total / weight_sum


def gather_dof_gpu(rgb, depth, focal_m, sensor_m, f_number, focus_m, max_radius=MAX_RADIUS_PX):
    """gather_dof() on the active Mitsuba/Dr.Jit backend (CUDA or Metal), for the live viewer.

    Same taps and weights; bilinear sampling comes from a clamped Texture2f,
    and the 64-tap loop fuses into a single kernel.
    """
    import drjit as dr
    import mitsuba as mi

    h, w = depth.shape
    depth = np.where(np.isfinite(depth), depth, SKY_DEPTH_M).astype(np.float32)
    radius = signed_coc_radius_px(depth, focal_m, sensor_m, f_number, focus_m, h).astype(np.float32)
    packed = np.concatenate([rgb.astype(np.float32), depth[..., None], radius[..., None]], -1)
    tex = mi.Texture2f(mi.TensorXf(packed), filter_mode=dr.FilterMode.Linear, wrap_mode=dr.WrapMode.Clamp)

    idx = dr.arange(mi.UInt32, h * w)
    px, py = mi.Float(idx % w), mi.Float(idx // w)
    c_depth, c_radius = mi.Float(depth.ravel()), mi.Float(radius.ravel())
    blur = dr.minimum(dr.abs(c_radius), min(max_radius, 32.0))
    centre_fg = c_radius < 0.0
    active = blur >= 0.5

    def ss(e0, e1, x):
        t = dr.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    c = [mi.Float(rgb[..., k].astype(np.float32).ravel()) for k in range(3)]
    total, wsum = list(c), mi.Float(1.0)
    for (dx, dy), dl in zip(DISK, DISK_LEN):
        u = (px + float(dx) * blur + 0.5) / w
        v = (py + float(dy) * blur + 0.5) / h
        sr, sg, sb, s_depth, s_radius = tex.eval(mi.Point2f(u, v))
        delta = s_depth - c_depth
        similar = 1.0 - ss(0.05, 1.4, dr.abs(delta))
        keep_bg_off_fg = dr.select(centre_fg, 1.0 - ss(-0.05, 0.25, delta), 1.0)
        big_fg_occluder = dr.select(s_radius < -0.5, ss(0.2, 1.2, dr.abs(s_radius)), 0.0)
        reach = ss(float(dl) * blur - 0.75, float(dl) * blur + 0.75, dr.abs(s_radius))
        wgt = (0.18 + 0.82 * similar) * dr.maximum(keep_bg_off_fg, big_fg_occluder * 0.85)
        wgt = dr.select(active, wgt * dr.maximum(reach, similar * 0.65), 0.0)
        total = [t + ch * wgt for t, ch in zip(total, (sr, sg, sb))]
        wsum += wgt
    out = np.stack([np.asarray(t / wsum) for t in total], -1)
    return out.reshape(h, w, 3)


def to_8bit(rgb):
    import render as R

    return np.round(R.tonemap(rgb, 0.0) * 255.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("render_dir", type=Path, help="qualified stage-2 per-stop directory")
    args = ap.parse_args(argv)
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    from experiment_contract import compare
    from stage2_experiment import source_manifest
    meta = json.loads((args.render_dir / "context.json").read_text())
    if meta.get("sources") != source_manifest():
        raise ValueError("Stale experiment sources")
    result = compare(args.render_dir, meta)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
