#!/usr/bin/env python3
"""Generate the stage-3 depth-of-field dataset.

One sample is one (scene, view, lens) configuration rendered into the tensors the
learned stage consumes:

    sharp.exr      pinhole, path traced, linear float32       INPUT
    depth.npy      planar z in metres along the optical axis  INPUT
    coc.npy        signed CoC radius in pixels                INPUT (derived)
    dof.exr        thin lens, path traced, linear float32     TARGET
    gather-naive.exr     production shader, equal weights     BASELINE
    gather-weighted.exr  production shader, CoC weights       BASELINE
    sample.json    camera, lens, hashes, timings, spp         PROVENANCE

The contract these obey is written out in dataset/README.md. Read that before
writing a dataloader; several of the conventions are load-bearing and a couple
are counter-intuitive (sky depth, signed CoC, linear-not-sRGB).

Design decisions worth knowing, because they are not reversible after a long run:

* **Splits are disjoint by scene seed, not by sample.** Two samples of the same
  café from different camera angles share furniture, materials and light
  positions. Splitting per sample would put near-duplicates on both sides of the
  train/test line and report a validation score that means nothing.

* **The target is the thin-lens render, not the gather.** The gathers are stored
  as a baseline to measure against, never as supervision. If a model is trained
  on gather output it can at best learn to reproduce the method's errors.

* **Every file is sealed with the scene and camera hash** through
  experiment_contract.seal, so a resumed or re-sharded run cannot pair a sharp
  image with a reference from different settings. That failure is silent
  otherwise: the images are the same size and look plausible together.

* **Sharding is by sample index modulo shard count**, so any subset of array
  tasks that completes still gives a dataset covering the whole configuration
  space rather than a contiguous block of one scene.

Usage:

    # plan only; needs no Mitsuba, prints what would be rendered
    python dataset/build_dataset.py --dry-run

    # one shard of a SLURM array
    python dataset/build_dataset.py --out data/dof-v1 --shard $SLURM_ARRAY_TASK_ID --shards 64

    # local smoke test at low quality
    python dataset/build_dataset.py --out /tmp/ds --preset smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "renderer" / "mitsuba"))

SCHEMA = 1

# --- Configuration space ----------------------------------------------------
# Deliberately coarse and explicit rather than randomly sampled: a reviewer can
# read this and know exactly what the dataset covers, and a rerun reproduces it.
#
# Lens settings span from a near-pinhole control to wide open. f/16 is included
# on purpose even though it is almost sharp: it is the control that shows the
# pipeline agrees when depth of field is barely present, and a model that cannot
# pass through a near-sharp image is broken in an obvious way.
F_NUMBERS = (1.2, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 16.0)
FOCAL_LENGTHS_MM = (28.0, 50.0, 85.0)
VIEWS = ("home", "close", "wide")

PRESETS = {
    # name: (width, height, sharp spp, dof spp, max_depth)
    "smoke": (160, 96, 16, 16, 6),
    "draft": (480, 270, 256, 512, 12),
    "train": (960, 540, 1024, 4096, 12),
    "final": (1280, 720, 2048, 8192, 12),
}


def stable_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=float).encode()
    ).hexdigest()


def enumerate_samples(scene_seeds, focus_fractions):
    """Every (scene, view, lens, focus) combination, in a fixed order.

    The order is part of the contract: shard assignment is index modulo shard
    count, so a different enumeration order would scatter a resumed run's work
    across different shards and leave gaps.
    """
    samples = []
    for seed in scene_seeds:
        for view in VIEWS:
            for lens_mm in FOCAL_LENGTHS_MM:
                for f_number in F_NUMBERS:
                    for focus_name, _ in focus_fractions:
                        samples.append({
                            "scene_seed": int(seed),
                            "view": view,
                            "focal_length_mm": float(lens_mm),
                            "f_number": float(f_number),
                            "focus_target": focus_name,
                        })
    for index, sample in enumerate(samples):
        sample["index"] = index
        sample["id"] = (f"s{sample['scene_seed']:03d}_{sample['view']}"
                        f"_{int(sample['focal_length_mm']):03d}mm"
                        f"_f{sample['f_number']:g}_{sample['focus_target']}")
    return samples


def assign_splits(samples, scene_seeds, val_scenes, test_scenes):
    """Disjoint by scene, so no furniture layout appears in two splits."""
    ordered = sorted(scene_seeds)
    if len(ordered) < val_scenes + test_scenes + 1:
        raise SystemExit(
            f"{len(ordered)} scenes cannot be split into train/val/test with "
            f"{val_scenes} validation and {test_scenes} test scenes. Raise --scenes."
        )
    test = set(ordered[:test_scenes])
    validation = set(ordered[test_scenes:test_scenes + val_scenes])
    for sample in samples:
        seed = sample["scene_seed"]
        sample["split"] = "test" if seed in test else "val" if seed in validation else "train"
    return {"test": sorted(test), "val": sorted(validation),
            "train": sorted(set(ordered) - test - validation)}


def signed_coc_radius_px(depth_m, lens_m, sensor_m, f_number, focus_m, height_px):
    """Signed CoC RADIUS in pixels. Negative in front of the focus plane.

    This is the thin-lens sensor convention Mitsuba's `thinlens` implements,

        coc_sensor = (f / N) * f * (z - s) / (z * s)

    and NOT the textbook f(z-s)/(z(s-f)) form. The two differ by s/(s-f), which
    was measured at 2.72% on the stage-2 configuration. Using the textbook form
    here would bake that bias into every label in the dataset.

    Radius, not diameter: the circle of confusion is defined as a diameter, and
    passing it in as a radius doubles the blur. That was a real bug in this
    project's history and is the reason for the explicit 0.5.
    """
    aperture = lens_m / f_number
    coc_sensor = aperture * lens_m * (depth_m - focus_m) / (depth_m * focus_m)
    return (0.5 * coc_sensor / sensor_m * height_px).astype(np.float32)


def finite_depth_for_gather(depth, sky_depth_m):
    """The shader needs finite positive depth; the sky is infinitely far.

    Mapped to an explicit large constant rather than clamped silently, and the
    constant is recorded in the sample metadata so a dataloader can reconstruct
    which pixels were sky.
    """
    return np.where(np.isfinite(depth) & (depth > 0), depth, sky_depth_m).astype(np.float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO / "data" / "dof-v1"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="train")
    parser.add_argument("--res", help="WIDTHxHEIGHT, overrides the preset")
    parser.add_argument("--spp", type=int, help="sharp-pass spp, overrides the preset")
    parser.add_argument("--dof-spp", type=int, help="reference spp, overrides the preset")
    parser.add_argument("--spp-per-pass", type=int, default=256)
    parser.add_argument("--max-depth", type=int, help="path depth, overrides the preset")
    parser.add_argument("--scenes", type=int, default=12, help="number of distinct café seeds")
    parser.add_argument("--scene-seed-base", type=int, default=1000)
    parser.add_argument("--val-scenes", type=int, default=2)
    parser.add_argument("--test-scenes", type=int, default=2)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="stop after this many samples (debugging)")
    parser.add_argument("--variant", default="auto")
    parser.add_argument("--gl-backend", default=None, choices=(None, "egl", "cgl"))
    parser.add_argument("--egl-device", type=int, default=None)
    parser.add_argument("--no-gather", action="store_true",
                        help="skip the OpenGL baselines; renders inputs and target only")
    parser.add_argument("--max-radius-px", type=float, default=120.0)
    parser.add_argument("--gather-samples", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the manifest; render nothing, needs no Mitsuba")
    args = parser.parse_args(argv)

    width, height, spp, dof_spp, max_depth = PRESETS[args.preset]
    if args.res:
        width, height = (int(v) for v in args.res.lower().split("x"))
    args.width, args.height = width, height
    args.spp = args.spp or spp
    args.dof_spp = args.dof_spp or dof_spp
    args.max_depth = args.max_depth or max_depth
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    return args


def build_plan(args):
    """The full sample list, splits and this shard's share. No rendering."""
    scene_seeds = [args.scene_seed_base + i for i in range(args.scenes)]
    # Focus targets are named points in the café; the scene builder returns their
    # positions, and the depth is measured along the optical axis at render time.
    focus_fractions = [("teapot", None), ("books", None), ("espresso", None)]
    samples = enumerate_samples(scene_seeds, focus_fractions)
    splits = assign_splits(samples, scene_seeds, args.val_scenes, args.test_scenes)
    mine = [s for s in samples if s["index"] % args.shards == args.shard]
    if args.limit:
        mine = mine[:args.limit]
    config = {
        "schema": SCHEMA,
        "resolution": [args.width, args.height],
        "sharp_spp": args.spp, "reference_spp": args.dof_spp,
        "max_depth": args.max_depth,
        "f_numbers": list(F_NUMBERS), "focal_lengths_mm": list(FOCAL_LENGTHS_MM),
        "views": list(VIEWS), "focus_targets": [name for name, _ in focus_fractions],
        "scene_seeds": scene_seeds, "splits": splits,
        "gather": {"max_radius_px": args.max_radius_px, "samples": args.gather_samples,
                   "variants": [] if args.no_gather else ["naive", "weighted"]},
        "sky_depth_m": 1.0e4,
    }
    config["config_hash"] = stable_hash(config)
    return config, samples, mine


def main(argv=None):
    args = parse_args(argv)
    config, samples, mine = build_plan(args)
    out = Path(args.out)

    print(f"dataset   {out}")
    print(f"preset    {args.preset}  {args.width}x{args.height}  "
          f"sharp {args.spp} spp, reference {args.dof_spp} spp, depth {args.max_depth}")
    print(f"scenes    {args.scenes} seeds -> train {len(config['splits']['train'])}, "
          f"val {len(config['splits']['val'])}, test {len(config['splits']['test'])} (disjoint)")
    print(f"samples   {len(samples)} total, {len(mine)} in shard {args.shard}/{args.shards}")
    print(f"config    {config['config_hash'][:16]}")

    if args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset.json").write_text(json.dumps(
            {"config": config, "samples": samples}, indent=2) + "\n")
        print(f"\nwrote {out / 'dataset.json'} (plan only; nothing rendered)")
        for sample in mine[:6]:
            print(f"  {sample['split']:5s} {sample['id']}")
        if len(mine) > 6:
            print(f"  ... {len(mine) - 6} more")
        return 0

    # --- Everything below needs Mitsuba and, unless --no-gather, a GL context.
    import render as R
    from cafe_scene import build_scene
    from experiment_contract import seal, write_json

    variant = R.pick_variant(args.variant)
    import mitsuba as mi
    print(f"mitsuba   {mi.__version__} variant {variant}")

    gather = None
    if not args.no_gather:
        from gl_gather import Gather
        gather = Gather(backend=args.gl_backend, device_index=args.egl_device)
        print(f"gl        {gather.gl_info['backend']} {gather.renderer}")
        if any(mark in gather.renderer.lower()
               for mark in ("llvmpipe", "softpipe", "swrast")):
            raise SystemExit(
                "Refusing to generate a dataset with a software OpenGL rasterizer: it would "
                "take days and indicates the GPU driver is not in use. Check "
                "`python renderer/mitsuba/gl_context.py` on this node."
            )

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset.json", {"config": config, "samples": samples})

    assets = REPO / "renderer" / "mitsuba" / "generated"
    scene_cache = {}
    written, skipped, failed = 0, 0, []
    started = time.time()

    for position, sample in enumerate(mine, 1):
        directory = out / sample["split"] / sample["id"]
        done_marker = directory / "sample.json"
        if done_marker.is_file() and not args.overwrite:
            skipped += 1
            continue
        directory.mkdir(parents=True, exist_ok=True)

        try:
            seed = sample["scene_seed"]
            if seed not in scene_cache:
                # Building a café costs seconds and every lens setting reuses it,
                # so the loop is ordered scene-outermost and the scene is cached.
                scene_dict, focus_points = build_scene(assets, False, seed=seed)
                scene_dict["integrator"] = {
                    "type": "path", "max_depth": args.max_depth,
                    "rr_depth": 6, "hide_emitters": False,
                }
                scene_cache.clear()          # one at a time: these are large
                scene_cache[seed] = (mi.load_dict(scene_dict), focus_points,
                                     stable_hash(sorted(scene_dict.keys())))
            scene, focus_points, scene_hash = scene_cache[seed]

            view = R.VIEWS[sample["view"]]
            point = focus_points[sample["focus_target"]]
            lens = R.Lens(view["origin"], view["target"], sample["focal_length_mm"],
                          24.0, sample["f_number"], 1.0)
            lens.focus_m = max(0.15, lens.depth_of(point))

            context = {
                "schema": SCHEMA, "scene_hash": scene_hash, "scene_seed": seed,
                "camera": lens.metadata(args.width, args.height),
                "resolution": [args.width, args.height],
                "sharp_spp": args.spp, "reference_spp": args.dof_spp,
                "max_depth": args.max_depth, "config_hash": config["config_hash"],
                "integrator": {"type": "path", "max_depth": args.max_depth,
                               "rr_depth": 6, "hide_emitters": False},
                "beauty_filter": "gaussian", "depth_filter": "box",
                "qualified_reference": False, "reference_noise_rms": None,
            }

            sharp_sensor = lens.sensor(args.width, args.height, args.spp, thin_lens=False)
            dof_sensor = lens.sensor(args.width, args.height, args.dof_spp, thin_lens=True)
            sharp, sharp_seconds = R.render_beauty(scene, sharp_sensor, args.spp,
                                                   args.spp_per_pass, seed, "sharp")
            reference, dof_seconds = R.render_beauty(scene, dof_sensor, args.dof_spp,
                                                     args.spp_per_pass, seed + 1, "dof")
            depth_sensor = lens.sensor(args.width, args.height, 1,
                                       thin_lens=False, rfilter="box")
            depth, layers, depth_seconds = R.render_gbuffer(scene, depth_sensor, lens, seed)

            coc = signed_coc_radius_px(
                finite_depth_for_gather(depth, config["sky_depth_m"]),
                lens.lens_m, lens.sensor_m, lens.f_number, lens.focus_m, args.height)

            mi.Bitmap(sharp).write(str(directory / "sharp.exr"))
            mi.Bitmap(reference).write(str(directory / "dof.exr"))
            np.save(directory / "depth.npy", depth)
            np.save(directory / "coc.npy", coc)
            for name in ("sharp.exr", "dof.exr"):
                seal(directory / name, context, name.split(".")[0])

            baselines = {}
            if gather is not None:
                safe_depth = finite_depth_for_gather(depth, config["sky_depth_m"])
                for kind in ("naive", "weighted"):
                    image = gather.run(sharp, safe_depth, context["camera"], kind,
                                       args.max_radius_px)
                    path = directory / f"gather-{kind}.exr"
                    mi.Bitmap(image).write(str(path))
                    seal(path, context, f"gather-{kind}", variant=kind,
                         renderer=gather.renderer)
                    baselines[kind] = float(np.abs(image - reference).mean())

            finite = np.isfinite(depth)
            record = dict(
                sample, **{
                    "schema": SCHEMA, "context": context,
                    "files": sorted(p.name for p in directory.iterdir()),
                    "timings_s": {"sharp": sharp_seconds, "reference": dof_seconds,
                                  "gbuffer": depth_seconds},
                    "depth_m": {"min": float(depth[finite].min()) if finite.any() else None,
                                "max": float(depth[finite].max()) if finite.any() else None,
                                "sky_fraction": float((~finite).mean())},
                    "coc_radius_px": {"min": float(coc.min()), "max": float(coc.max()),
                                      "clipped_fraction": float(
                                          (np.abs(coc) > args.max_radius_px).mean())},
                    "baseline_linear_mae": baselines,
                    "mitsuba_variant": variant, "host": platform.node(),
                })
            write_json(done_marker, record)
            written += 1
            elapsed = time.time() - started
            print(f"[{position}/{len(mine)}] {sample['id']}  "
                  f"CoC {coc.min():+.1f}..{coc.max():+.1f} px  "
                  f"{elapsed / max(written, 1):.0f}s/sample", flush=True)
        except Exception as error:                                  # noqa: BLE001
            # One bad configuration must not kill a six-hour array task. It is
            # recorded and skipped; the manifest at the end lists every failure.
            failed.append({"id": sample["id"], "error": repr(error)})
            print(f"[{position}/{len(mine)}] FAILED {sample['id']}: {error!r}", flush=True)

    if gather is not None:
        gather.close()
    summary = {"schema": SCHEMA, "shard": args.shard, "shards": args.shards,
               "written": written, "skipped": skipped, "failed": failed,
               "seconds": time.time() - started, "config_hash": config["config_hash"]}
    write_json(out / f"shard-{args.shard:04d}.json", summary)
    print(f"\nwrote {written}, skipped {skipped}, failed {len(failed)} "
          f"in {summary['seconds'] / 60:.1f} min")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
