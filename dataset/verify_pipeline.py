#!/usr/bin/env python3
"""Check the dataset pipeline on this machine. Needs no Mitsuba and no scene.

    python dataset/verify_pipeline.py

Run this before committing cluster hours. It exercises the parts that actually
break when moving between machines -- the OpenGL context, the production shader,
and the CoC convention -- using synthetic two-plane images, so a failure points
at the environment rather than at the renderer.

What it does NOT check: Mitsuba, the café scene, or whether a reference is
converged. Those need a GPU node and a real render.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "renderer" / "mitsuba"))
sys.path.insert(0, str(HERE))

CAMERA = {"focus_distance_m": 1.0, "focal_length_mm": 85.0,
          "sensor_height_mm": 24.0, "f_number": 1.2}


def synthetic(height=96, width=128):
    """A near plane at the focus distance against a far plane with a highlight."""
    depth = np.full((height, width), 8.0, np.float32)
    depth[:, :width // 3] = 1.0
    rgb = np.zeros((height, width, 3), np.float32)
    rgb[:, :width // 3] = (1.0, 0.3, 0.1)
    rgb[:, width // 3:] = (0.05, 0.08, 0.12)
    rgb[height // 2 - 3:height // 2 + 3, width // 2 - 3:width // 2 + 3] = 40.0
    return rgb, depth


def check(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    return bool(condition)


def main():
    ok = True
    print("1. CoC convention")
    from build_dataset import signed_coc_radius_px
    # The audited reference figure for the Mitsuba thinlens sensor.
    value = signed_coc_radius_px(np.array([[10.0]], np.float32),
                                 0.050, 0.024, 1.8, 1.887372, 540)[0, 0]
    ok &= check("matches the audited sensor convention", abs(value - 6.7162) < 1e-3,
                f"{value:+.4f} px, expected +6.7162")
    near = signed_coc_radius_px(np.array([[0.5]], np.float32), 0.050, 0.024, 1.8, 1.887372, 540)[0, 0]
    at = signed_coc_radius_px(np.array([[1.887372]], np.float32), 0.050, 0.024, 1.8, 1.887372, 540)[0, 0]
    ok &= check("negative in front of focus, zero at it", near < 0 and abs(at) < 1e-4,
                f"{near:+.2f} px near, {at:+.4f} px at focus")

    print("2. OpenGL context")
    try:
        from gl_context import create_context
        context = create_context()
        info = context.describe()
        context.destroy()
    except Exception as error:                                       # noqa: BLE001
        print(f"  FAIL  no offscreen context — {error}")
        return 1
    print(f"        {info['backend']} / {info['vendor']} / {info['renderer']} / {info['version']}")
    software = any(m in info["renderer"].lower()
                   for m in ("llvmpipe", "softpipe", "swrast", "software"))
    ok &= check("OpenGL 3.3 core or newer", info["version"].startswith(("3.3", "4.")))
    if software:
        print("  WARN  software rasterizer: correct but far too slow for a dataset. "
              "On a cluster this means the GPU driver is not in use.")

    print("3. Production shader")
    from gl_gather import Gather
    rgb, depth = synthetic()
    try:
        gather = Gather()
        images = {kind: gather.run(rgb, depth, CAMERA, kind, 120.0)
                  for kind in ("naive", "weighted")}
        gather.close()
    except Exception as error:                                       # noqa: BLE001
        print(f"  FAIL  shader did not run — {error}")
        return 1
    for kind, image in images.items():
        ok &= check(f"{kind} output finite and correctly shaped",
                    image.shape == rgb.shape and np.isfinite(image).all())
    # The near plane sits exactly at the focus distance, so its CoC is zero and
    # the gather must return it untouched. A failure here means the lens
    # parameters are not reaching the shader.
    column = slice(0, rgb.shape[1] // 3 - 8)
    for kind, image in images.items():
        error = float(np.abs(image[:, column] - rgb[:, column]).max())
        ok &= check(f"{kind} leaves the in-focus plane untouched", error < 1e-5,
                    f"max error {error:.2e}")
    # The far plane must actually blur: the bright highlight spreads and its peak drops.
    ok &= check("far field is blurred", images["naive"].max() < 0.5 * rgb.max(),
                f"peak {images['naive'].max():.2f} from {rgb.max():.0f}")
    # The two variants must differ, or the gather-mode uniform is not connected.
    difference = float(np.abs(images["naive"] - images["weighted"]).mean())
    ok &= check("naive and weighted differ", difference > 1e-6, f"mean |diff| {difference:.2e}")

    print("4. Dataset plan")
    import build_dataset as B
    config, samples, mine = B.build_plan(
        B.parse_args(["--dry-run", "--out", "/tmp/_verify", "--scenes", "12",
                      "--shards", "64", "--shard", "0"]))
    splits = config["splits"]
    overlap = (set(splits["train"]) & set(splits["val"]) |
               set(splits["train"]) & set(splits["test"]) |
               set(splits["val"]) & set(splits["test"]))
    ok &= check("splits share no scene", not overlap)
    ok &= check("shard covers the configuration space",
                len({s["scene_seed"] for s in mine}) == len(config["scene_seeds"]),
                f"{len(mine)} samples over {len({s['scene_seed'] for s in mine})} scenes")

    print()
    if ok and not software:
        print("READY: this machine can generate the dataset.")
        return 0
    if ok:
        print("USABLE BUT SLOW: everything is correct, but OpenGL is running in software. "
              "Fine for a local smoke test; use a GPU node for the real run.")
        return 0
    print("NOT READY: see the failures above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
