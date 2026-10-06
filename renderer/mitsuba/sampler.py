#!/usr/bin/env python3
"""Deterministic dataset sampler: random camera poses and lens settings -> sample directories.

Sample i of a (scene, scene seed, env, sampler seed) is a pure function of those
and i, so any index can be regenerated alone and array jobs can split the range.

  python sampler.py --scene cafe --count 4 --res 640x360 --spp 64 --out dataset/mitsuba
  python sampler.py --scene cafe --count 500 --dry-run      # poses and lenses only: rejection stats

Per sample index it draws a *pose* (position in camera_box, target in target_box,
roll, focal length), renders the 1-spp G-buffer to accept or reject it, then draws
`--lenses-per-pose` *lens variants* (focus distance, f-number). Focal length belongs
to the pose because it changes the field of view; f-number and focus only change blur.
Layout (SCENE_CONTRACT.md section 6):

  <out>/<scene>/s<seed>/<env>/p<index>_<hash>/{pose.json, depth.npy, gbuffer.exr, sharp.exr,
                                              lens_<hash>/{dof.exr, coc.npy, metadata.json}, .done}
  <out>/<scene>/s<seed>/<env>/manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render as R  # noqa: E402
import scene_api as S  # noqa: E402

# Sampling policy. These are defaults to tune, not physics.
PITCH_MAX_DEG = 75.0
MIN_CLEARANCE_M = 0.3          # 1st percentile of surface depth
MAX_SKY_FRACTION = 0.5
MIN_DEPTH_SPREAD = 1.5         # p95 / p5 of surface depth: rejects a single flat surface
MIN_VIEW_DISTANCE_M = 0.5      # between camera and target
ROLL_PROB, ROLL_MAX_DEG = 0.5, 25.0
LENS_MM_GLOBAL = (24.0, 135.0)
F_NUMBER_GLOBAL = (1.2, 16.0)
COC_FRAC_RANGE = (0.0005, 0.04)  # target p95 |CoC| as a fraction of image height (0.5 to 43 px at 1080p)
COC_BINS = 8
FOCUS_NAMED_PROB = 0.5
FOCUS_VISIBLE_TOL_M = 0.15     # a named point counts as visible if a surface pixel lies this close
MAX_POSE_TRIES, MAX_LENS_TRIES = 60, 30


def rng_for(scene: S.SceneDef, args, index: int, stream: int, attempt: int = 0) -> np.random.Generator:
    key = [zlib.crc32(scene.id.encode()), args.scene_seed, ENV_INDEX[args.env], args.sampler_seed, index, stream, attempt]
    return np.random.default_rng(np.random.SeedSequence(key))


ENV_INDEX = {e: i for i, e in enumerate(S.ENVS)}


def short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


def in_any(p, boxes) -> bool:
    return any(all(lo <= v <= hi for v, lo, hi in zip(p, *b)) for b in boxes)


def draw_pose(scene: S.SceneDef, rng, reasons: Counter):
    """Position, target, roll and focal length, or None if this attempt is unusable before rendering."""
    for _ in range(100):
        origin = rng.uniform(*scene.camera_box)
        if not in_any(origin, scene.exclude_boxes):
            break
    else:
        reasons["no_free_position"] += 1
        return None
    target = rng.uniform(*scene.target_box)
    d = target - origin
    dist = float(np.linalg.norm(d))
    if dist < MIN_VIEW_DISTANCE_M:
        reasons["too_close_target"] += 1
        return None
    if abs(math.degrees(math.asin(d[1] / dist))) > PITCH_MAX_DEG:
        reasons["pitch"] += 1
        return None
    roll = float(rng.uniform(-ROLL_MAX_DEG, ROLL_MAX_DEG)) if rng.random() < ROLL_PROB else 0.0
    lo, hi = max(LENS_MM_GLOBAL[0], scene.lens_mm[0]), min(LENS_MM_GLOBAL[1], scene.lens_mm[1])
    lo, hi = (lo, hi) if lo <= hi else scene.lens_mm
    focal = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return origin, target, roll, focal


def check_depth(depth: np.ndarray, reasons: Counter) -> bool:
    finite = depth[np.isfinite(depth)]
    if finite.size < 0.5 * depth.size or 1.0 - finite.size / depth.size > MAX_SKY_FRACTION:
        reasons["sky"] += 1
        return False
    p1, p5, p95 = np.percentile(finite, (1, 5, 95))
    if p1 < MIN_CLEARANCE_M:
        reasons["clearance"] += 1
        return False
    if p95 / max(p5, 1e-6) < MIN_DEPTH_SPREAD:
        reasons["flat"] += 1
        return False
    return True


def draw_lens(scene: S.SceneDef, rng, pose_lens: R.Lens, depth, pos, focus_points, height_px: int):
    """Focus distance and f-number. The target p95 CoC is drawn log-uniformly from what this
    pose's focal length and focus can reach with the allowed f-numbers, so blur sizes spread out
    without asking a wide lens for a huge blur (physically unreachable). `stratum` is the achieved bin."""
    finite = depth[np.isfinite(depth)]
    lo, hi = max(F_NUMBER_GLOBAL[0], scene.f_number[0]), min(F_NUMBER_GLOBAL[1], scene.f_number[1])
    lo, hi = (lo, hi) if lo <= hi else scene.f_number
    log_span = math.log(COC_FRAC_RANGE[1] / COC_FRAC_RANGE[0])
    fallback = None
    for _ in range(MAX_LENS_TRIES):
        focus, source = None, "depth_quantile"
        if rng.random() < FOCUS_NAMED_PROB:
            name = list(focus_points)[int(rng.integers(len(focus_points)))]
            p = np.asarray(focus_points[name][int(rng.integers(len(focus_points[name])))])
            z = pose_lens.depth_of(p)
            if z > MIN_CLEARANCE_M and (np.linalg.norm(pos - p, axis=-1) < FOCUS_VISIBLE_TOL_M).any():
                focus, source = z, f"point:{name}"
        if focus is None:
            focus = float(np.quantile(finite, rng.uniform(0.05, 0.95)))
        unit = R.Lens(pose_lens.origin, pose_lens.target, pose_lens.lens_m * 1000, 24.0, 1.0, focus, pose_lens.roll_deg)
        p95_at_f1 = float(np.percentile(np.abs(unit.coc_map(depth, height_px)), 95)) / height_px
        window = (max(p95_at_f1 / hi, COC_FRAC_RANGE[0]), min(p95_at_f1 / lo, COC_FRAC_RANGE[1]))
        if window[0] >= window[1]:
            n = min(max(1.0, lo), hi)
            fallback = fallback or dict(focus=float(focus), f_number=float(n), focus_source=source,
                                        achieved_frac=p95_at_f1 / n, coc_out_of_range=True)
            continue
        achieved = math.exp(rng.uniform(math.log(window[0]), math.log(window[1])))
        stratum = min(int(COC_BINS * math.log(achieved / COC_FRAC_RANGE[0]) / log_span), COC_BINS - 1)
        return dict(focus=float(focus), f_number=float(p95_at_f1 / achieved), focus_source=source,
                    achieved_frac=achieved, stratum=stratum, coc_out_of_range=False)
    fallback["stratum"] = -1
    return fallback


def write_exr(path: Path, rgb: np.ndarray) -> None:
    import mitsuba as mi

    mi.Bitmap(rgb).write(str(path))


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def run_sample(index, scene_def, bundle, scene, ctx, args, base: Path, fp: str, git: dict, variant: str, mi_version: str):
    w, h = args.width, args.height
    reasons: Counter = Counter()
    pose = None
    for k in range(MAX_POSE_TRIES):
        rng = rng_for(scene_def, args, index, 0, k)
        drawn = draw_pose(scene_def, rng, reasons)
        if drawn is None:
            continue
        origin, target, roll, focal = drawn
        pose_lens = R.Lens(origin, target, focal, S.SENSOR_HEIGHT_MM, 1e5, 1.0, roll)
        depth, layers, g_sec = R.render_gbuffer(scene, pose_lens.sensor(w, h, 1, False, "box"), pose_lens,
                                                int(rng.integers(0, 2 ** 31 - 1)))
        if check_depth(depth, reasons):
            pose = dict(origin=origin, target=target, roll=roll, focal=focal, attempt=k)
            break
    if pose is None:
        return {"index": index, "failed": True, "reasons": dict(reasons)}

    pose_key = {"scene": scene_def.id, "seed": args.scene_seed, "env": args.env, "sampler_seed": args.sampler_seed,
                "origin": np.round(origin, 4).tolist(), "target": np.round(target, 4).tolist(),
                "roll": round(roll, 3), "focal": round(focal, 3), "res": [w, h]}
    pose_id = f"p{index:06d}_{short_hash(pose_key)[:8]}"
    lens_rng = rng_for(scene_def, args, index, 1)
    lenses = [draw_lens(scene_def, rng_for(scene_def, args, index, 2 + j), pose_lens, depth, layers["pos"],
                        bundle.focus_points, h) for j in range(args.lenses_per_pose)]
    out = {"index": index, "pose_id": pose_id, "reasons": dict(reasons), "lenses": lenses, "pose": pose_key}
    if args.dry_run:
        return out

    pdir = base / pose_id
    if (pdir / ".done").exists():
        out["skipped"] = True
        return out
    pdir.mkdir(parents=True, exist_ok=True)
    sharp_seed = int(rng_for(scene_def, args, index, 100).integers(0, 10000))
    sharp, sharp_sec = R.render_beauty(scene, pose_lens.sensor(w, h, args.spp, False), args.spp, args.spp_per_pass,
                                       sharp_seed, "sharp")
    np.save(pdir / "depth.npy", depth)
    R.write_gbuffer(pdir / "gbuffer.exr", layers)
    write_exr(pdir / "sharp.exr", sharp)
    lum = luminance(sharp)
    pose_meta = {**pose_key, "pose_id": pose_id, "rejected_before": dict(reasons),
                 "sharp": {"spp": args.spp, "seconds": round(sharp_sec, 2), "seed": sharp_seed},
                 "sharp_median_luminance": float(np.median(lum)),
                 "exposure_gate_ok": bool(0.02 <= np.median(lum) <= 0.5),
                 "camera": pose_lens.metadata(w, h)}
    (pdir / "pose.json").write_text(json.dumps(pose_meta, indent=2))

    dof_spp = args.dof_spp or max(args.spp, scene_def.spp_hint)
    manifest = []
    for j, lens_spec in enumerate(lenses):
        lens = R.Lens(origin, target, focal, S.SENSOR_HEIGHT_MM, lens_spec["f_number"], lens_spec["focus"], roll)
        lens_id = "lens_" + short_hash({"focus": round(lens_spec["focus"], 4), "n": round(lens_spec["f_number"], 4),
                                        "aperture": "disc"})[:8]
        ldir = pdir / lens_id
        ldir.mkdir(exist_ok=True)
        dof_seed = 10000 + int(rng_for(scene_def, args, index, 200 + j).integers(0, 10000))
        dof, dof_sec = R.render_beauty(scene, lens.sensor(w, h, dof_spp, True), dof_spp, args.spp_per_pass, dof_seed, "dof")
        write_exr(ldir / "dof.exr", dof)
        np.save(ldir / "coc.npy", lens.coc_map(depth, h))
        sample_id = short_hash({**pose_key, "focus": round(lens_spec["focus"], 4),
                                "f_number": round(lens_spec["f_number"], 4), "aperture": "disc"})
        meta = {"contract_version": S.CONTRACT_VERSION, "sample_id": sample_id, "pose_id": pose_id, "lens_id": lens_id,
                "scene": {"id": scene_def.id, "seed": args.scene_seed, "env": args.env, "fingerprint": fp,
                          "asset_version": scene_def.asset_version, **bundle.meta},
                "git": git, "variant": variant, "mitsuba": mi_version, "resolution": [w, h],
                "camera": lens.metadata(w, h), "lens_draw": lens_spec,
                "coc": {"definition": "signed diameter in px, positive behind the focus plane; sky = at infinity",
                        "formula": "H f^2 (z - s) / (N z s sensor_h)"},
                "passes": {"sharp": pose_meta["sharp"], "dof": {"spp": dof_spp, "seconds": round(dof_sec, 2),
                                                                "seed": dof_seed}},
                "sampler": {"sampler_seed": args.sampler_seed, "index": index}}
        (ldir / "metadata.json").write_text(json.dumps(meta, indent=2))
        manifest.append({"sample_id": sample_id, "path": f"{pose_id}/{lens_id}", "stratum": lens_spec["stratum"],
                         "focal_mm": round(focal, 2), "f_number": round(lens_spec["f_number"], 3),
                         "focus_m": round(lens_spec["focus"], 3), "roll_deg": round(roll, 2),
                         "coc_p95_frac": round(lens_spec["achieved_frac"], 5), "coc_out_of_range": lens_spec["coc_out_of_range"],
                         "exposure_gate_ok": pose_meta["exposure_gate_ok"]})
    (pdir / ".done").write_text("")
    with open(base / "manifest.jsonl", "a") as f:
        f.write("".join(json.dumps(m) + "\n" for m in manifest))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="cafe")
    ap.add_argument("--scene-seed", type=int, default=None)
    ap.add_argument("--env", default="clear")
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first sample index")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--lenses-per-pose", type=int, default=2)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--spp", type=int, default=1024, help="sharp pass spp")
    ap.add_argument("--dof-spp", type=int, default=None, help="default: max(--spp, the scene's spp_hint)")
    ap.add_argument("--spp-per-pass", type=int, default=64)
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--out", default=str(R.REPO / "dataset" / "mitsuba"))
    ap.add_argument("--assets", default=str(HERE / "generated"))
    ap.add_argument("--dry-run", action="store_true", help="draw and check poses and lenses, render no beauty passes")
    args = ap.parse_args(argv)
    args.width, args.height = (int(v) for v in args.res.lower().split("x"))

    variant = R.pick_variant(args.variant)
    import mitsuba as mi

    scene_def = S.load_scene(args.scene)
    ctx = S.make_context(scene_def, Path(args.assets), args.scene_seed, args.env)
    args.scene_seed = ctx.seed
    bundle = S.build_checked(scene_def, ctx)
    fp = S.fingerprint(bundle.scene, [ctx.shared_dir, ctx.assets_dir])
    scene_dict = dict(bundle.scene)
    scene_dict["integrator"] = {"type": "path", "max_depth": scene_def.max_depth, "rr_depth": scene_def.rr_depth}
    first = scene_def.views[scene_def.default_view]
    scene_dict["sensor"] = R.Lens(first.origin, first.target, 50.0, S.SENSOR_HEIGHT_MM, 1e5, 1.0).sensor(
        args.width, args.height, 1, False)
    scene = mi.load_dict(scene_dict)
    git = R.git_state()
    base = Path(args.out) / scene_def.id / f"s{ctx.seed}" / args.env
    base.mkdir(parents=True, exist_ok=True)

    reasons, strata, failed, missed, n_lenses, t0 = Counter(), Counter(), 0, 0, 0, time.time()
    for index in range(args.start, args.start + args.count):
        r = run_sample(index, scene_def, bundle, scene, ctx, args, base, fp, git, variant, mi.__version__)
        reasons.update(r["reasons"])
        if r.get("failed"):
            failed += 1
            print(f"[{index}] FAILED after {MAX_POSE_TRIES} poses: {r['reasons']}")
            continue
        for ls in r["lenses"]:
            strata[ls["stratum"]] += 1
            missed += ls["coc_out_of_range"]
            n_lenses += 1
        tag = "skipped (done)" if r.get("skipped") else ("dry" if args.dry_run else "written")
        l0 = r["lenses"][0]
        print(f"[{index}] {r['pose_id']} {tag}: f/{l0['f_number']:.1f} focus {l0['focus']:.2f} m "
              f"({l0['focus_source']}) CoC p95 {l0['achieved_frac'] * args.height:.1f} px", flush=True)
    print(f"\n{args.count} poses in {time.time() - t0:.1f}s | pose rejections {dict(reasons)} | failed {failed}")
    print(f"CoC strata (achieved p95 bins, low to high): {[strata[i] for i in range(COC_BINS)]} | "
          f"outside the CoC range {missed}/{n_lenses}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
