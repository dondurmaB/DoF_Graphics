#!/usr/bin/env python3
"""Traditional screen-space depth of field applied to a Mitsuba render.

Takes the all-in-focus `sharp.exr` and the planar `depth.npy` that render.py
writes, blurs them with the same depth-aware 64-tap disk gather as the OpenGL
DoFScene (shaders/dof_scene/screen.frag), and compares the result with the
path-traced thin-lens `dof.exr` from the same camera:

    python renderer/mitsuba/traditional_dof.py output/mitsuba/cafe

writes `traditional.exr/.png`, `compare.png` (sharp | traditional | path-traced
| error) and `traditional_metrics.json` into that directory.

The comparison is fair by construction: all three images come from one camera
pose, one lens model and one scene, so the only difference is how visibility
through the aperture is resolved - a single pinhole image plus depth, versus
rays traced from every point of the lens.
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

# Golden-angle disk, identical to diskSamples[] in shaders/dof_scene/screen.frag.
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
    coc_sensor = aperture * focal_m * (depth - focus_m) / (depth * (focus_m - focal_m))
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
    """Port of DoFScene's gatherDof(): per-pixel disk gather with depth-aware weights."""
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render_dir", help="directory written by render.py (needs sharp.exr, depth.npy, dof.exr)")
    ap.add_argument("--max-radius", type=float, default=MAX_RADIUS_PX)
    args = ap.parse_args(argv)
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    import render as R
    from PIL import Image

    d = Path(args.render_dir)
    meta = json.loads((d / "metadata.json").read_text())
    cam = meta["camera"]
    sharp = np.asarray(mi.Bitmap(str(d / "sharp.exr")), np.float32)[..., :3]
    truth = np.asarray(mi.Bitmap(str(d / "dof.exr")), np.float32)[..., :3]
    depth = np.load(d / "depth.npy")

    trad = gather_dof(sharp, depth, cam["focal_length_mm"] / 1000, cam["sensor_height_mm"] / 1000,
                      cam["f_number"], cam["focus_distance_m"], args.max_radius)
    mi.Bitmap(trad.astype(np.float32)).write(str(d / "traditional.exr"))
    R.save_png(d / "traditional.png", R.tonemap(trad, 0.0))

    s8, t8, g8 = to_8bit(sharp), to_8bit(trad), to_8bit(truth)
    err = np.abs(t8 - g8).mean(-1)
    # Depth discontinuities: where the depth changes by >30% within a pixel,
    # dilated by the local blur radius, which is where screen-space DoF breaks.
    finite = np.where(np.isfinite(depth), depth, SKY_DEPTH_M)
    ld = np.log(finite)
    edge = np.zeros_like(ld, bool)
    edge[:, 1:] |= np.abs(np.diff(ld, axis=1)) > math.log(1.3)
    edge[1:, :] |= np.abs(np.diff(ld, axis=0)) > math.log(1.3)
    from PIL import ImageFilter

    band = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(15))) > 0
    metrics = {
        "method": "DoFScene depth-aware 64-tap gather on the pinhole render + planar depth",
        "reference": "path-traced thin lens (dof.exr), same camera",
        "spp": {k: v["spp"] for k, v in meta["passes"].items()},
        "mae_0_255": {"traditional_vs_reference": float(err.mean()),
                      "sharp_vs_reference": float(np.abs(s8 - g8).mean())},
        "mae_0_255_depth_edges": {"traditional_vs_reference": float(err[band].mean()),
                                  "sharp_vs_reference": float(np.abs(s8 - g8).mean(-1)[band].mean()),
                                  "edge_pixel_fraction": float(band.mean())},
        "mae_0_255_away_from_edges": {"traditional_vs_reference": float(err[~band].mean())},
    }
    (d / "traditional_metrics.json").write_text(json.dumps(metrics, indent=2))

    # sharp | traditional | path-traced, with a 4x amplified error map beneath.
    h, w = sharp.shape[:2]
    heat = np.clip(err * 4.0 / 255.0, 0, 1)
    heat_rgb = np.stack([heat, heat ** 2 * 0.6, heat ** 4 * 0.2], -1)
    sheet = np.full((2 * h + 12, 2 * w + 12, 3), 1.0, np.float32)
    sheet[:h, :w] = s8 / 255
    sheet[:h, w + 12:] = t8 / 255
    sheet[h + 12:, :w] = g8 / 255
    sheet[h + 12:, w + 12:] = heat_rgb
    R.save_png(d / "compare.png", sheet)
    print(json.dumps(metrics["mae_0_255"]), json.dumps(metrics["mae_0_255_depth_edges"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
