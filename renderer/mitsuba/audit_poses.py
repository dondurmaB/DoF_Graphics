#!/usr/bin/env python3
"""Contact sheet of the sampler's random poses for a scene: exactly what the dataset will see.

  python audit_poses.py --scene courtyard --env clear --count 24 --out output/dev/audit

Poses come from sampler.py's own draw and rejection logic with the same seeds, so sample i here is
sample i of `sampler.py --sampler-seed <same>`. Each tile is a low-spp sharp render labelled with
index, focal length, roll and median luminance; a red label marks a pose outside the exposure band.
Use it to find ugly angles (unfinished edges, backs of props, empty views) before generating data.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render as R  # noqa: E402
import sampler as SM  # noqa: E402
import scene_api as S  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--env", default="clear")
    ap.add_argument("--scene-seed", type=int, default=None)
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--res", default="384x216")
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--assets", default=str(HERE / "generated"))
    ap.add_argument("--out", default=str(R.REPO / "output" / "dev" / "audit"))
    args = ap.parse_args(argv)
    w, h = (int(v) for v in args.res.lower().split("x"))

    R.pick_variant(args.variant)
    import mitsuba as mi
    from PIL import Image, ImageDraw

    scene_def = S.load_scene(args.scene)
    ctx = S.make_context(scene_def, Path(args.assets), args.scene_seed, args.env)
    args.scene_seed = ctx.seed
    bundle = S.build_checked(scene_def, ctx)
    scene_dict = dict(bundle.scene)
    scene_dict["integrator"] = {"type": "path", "max_depth": scene_def.max_depth, "rr_depth": scene_def.rr_depth}
    first = scene_def.views[scene_def.default_view]
    scene_dict["sensor"] = R.Lens(first.origin, first.target, 50.0, S.SENSOR_HEIGHT_MM, 1e5, 1.0,
                                  far_clip=scene_def.far_clip).sensor(w, h, 1, False)
    scene = mi.load_dict(scene_dict)

    tiles, rejected, dark = [], Counter(), 0
    for i in range(args.start, args.start + args.count):
        reasons, pose = Counter(), None
        for k in range(SM.MAX_POSE_TRIES):
            rng = SM.rng_for(scene_def, args, i, 0, k)
            drawn = SM.draw_pose(scene_def, rng, reasons)
            if drawn is None:
                continue
            origin, target, roll, focal = drawn
            lens = R.Lens(origin, target, focal, S.SENSOR_HEIGHT_MM, 1e5, 1.0, roll, scene_def.far_clip)
            depth, layers, _ = R.render_gbuffer(scene, lens.sensor(w, h, 1, False, "box"), lens,
                                                int(rng.integers(0, 2 ** 31 - 1)))
            if SM.check_depth(depth, reasons, layers["nn"]) and SM.bright_enough(scene, lens, w, h, rng, reasons):
                pose = (focal, roll, lens)
                break
        rejected.update(reasons)
        if pose is None:
            tile = Image.new("RGB", (w, h), (60, 0, 0))
            ImageDraw.Draw(tile).text((6, 6), f"{i} no valid pose", fill=(255, 255, 255))
            tiles.append(tile)
            continue
        focal, roll, lens = pose
        rgb, _ = R.render_beauty(scene, lens.sensor(w, h, args.spp, False), args.spp, args.spp, i % 10000, "audit")
        med = float(np.median(rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)))
        ok = 0.02 <= med <= 0.5 or "night" in scene_def.tags
        dark += not ok
        tile = Image.fromarray((np.clip(R.tonemap(rgb, 0.0), 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(tile)
        label = f"{i}  {focal:.0f}mm  roll {roll:+.0f}  lum {med:.3f}"
        d.rectangle((0, 0, 7 * len(label) + 8, 16), fill=(0, 0, 0) if ok else (170, 0, 0))
        d.text((4, 3), label, fill=(255, 255, 255))
        tiles.append(tile)

    rows = (len(tiles) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * (w + 4), rows * (h + 4)), (20, 20, 20))
    for n, tile in enumerate(tiles):
        sheet.paste(tile, ((n % args.cols) * (w + 4), (n // args.cols) * (h + 4)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.scene}_{args.env}.png"
    sheet.save(path)
    print(f"{args.scene}/{args.env}: {len(tiles)} poses, rejections {dict(rejected)}, outside exposure band {dark} "
          f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
