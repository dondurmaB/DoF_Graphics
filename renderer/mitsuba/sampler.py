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
`--topup NAME` adds one more lens to each existing pose instead (experiment_plan.md D2): the pose is redrawn
exactly as before (same random streams, so the same pose_id), only the new lens's DoF pass is rendered, and the
output holds `p<index>_<hash>/{topup.json, lens_<hash>/{dof.exr, coc.npy, metadata.json}, .done}` to be merged
into the original dataset (merge_topup.py). The sharp image, depth and G-buffer already exist there.

Layout (SCENE_CONTRACT.md section 6):

  <out>/<scene>/s<seed>/<env>/p<index>_<hash>/{pose.json, depth.npy, gbuffer.exr, sharp.exr,
                                              lens_<hash>/{dof.exr, coc.npy, metadata.json}, .done}
  <out>/<scene>/s<seed>/<env>/manifest.jsonl     (or --manifest-name: one file per parallel worker,
                                               so workers never append to the same file over NFS)
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
MAX_SAME_FACING = 0.8          # share of pixels within 15 deg of the dominant normal: rejects a wall filling the frame
AUTO_EXPOSURE_KEY = 0.18       # mid-grey a camera meters to; recorded per sample, never baked into the EXRs
MIN_MEDIAN_LUMINANCE = 0.002   # below this the frame is essentially black and rejected after the sharp pass
MIN_VIEW_DISTANCE_M = 0.5      # between camera and target
# Camera roll, as people hold cameras: mostly level with handheld wobble, often tilted, sometimes portrait.
ROLL_LEVEL_P, ROLL_LEVEL_JITTER_DEG = 0.6, 3.0
ROLL_TILT_P, ROLL_TILT_MAX_DEG = 0.3, 45.0
ROLL_PORTRAIT_JITTER_DEG = 5.0                      # remaining 10%: +-90 degrees, either way
LENS_MM_GLOBAL = (24.0, 135.0)
F_NUMBER_GLOBAL = (1.2, 16.0)
COC_FRAC_RANGE = (0.005, 0.04)   # target p95 |CoC| as a fraction of image height (5.4 to 43 px at 1080p); the floor
                                 # was 0.0005 until 2026-10-20, when a third of the pilot had no visible blur (p95 < 3 px)
COC_BINS = 8
FOCUS_NAMED_PROB = 0.5
FOCUS_VISIBLE_TOL_M = 0.15     # a named point counts as visible if a surface pixel lies this close
MAX_POSE_TRIES, MAX_LENS_TRIES = 60, 30
# Top-up lenses (experiment_plan.md D2): realistic setups the main draw leaves out. Scenes allow f/1.4-8 and the main
# draw aims every sample at visible blur, so wide-open close-focus shots and small-aperture deep-focus shots are
# rare or missing (E1 audit, 2026-10-29). These override the scene's f-number range, within what real lenses do.
TOPUP_KINDS = (("wide", 0.65), ("small", 0.35))
TOPUP_WIDE_N_MAX = 2.8                 # wide open up to f/2.8
TOPUP_WIDE_FOCUS_Q = (0.02, 0.4)       # focus in the near part of the scene: a subject in front
TOPUP_WIDE_NAMED_P = 0.7               # ... preferably a named focus point (a subject)
TOPUP_WIDE_P95_MAX = 0.10              # a photographer stops down before p95 blur passes 10% of the frame (108 px)
TOPUP_SMALL_N = (8.0, 16.0)            # stopped down: nearly everything sharp
TOPUP_SMALL_FOCUS_Q = (0.2, 0.6)
TOPUP_SMALL_NAMED_P = 0.3


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
    u = rng.random()
    if u < ROLL_LEVEL_P:
        roll = float(rng.normal(0.0, ROLL_LEVEL_JITTER_DEG))
    elif u < ROLL_LEVEL_P + ROLL_TILT_P:
        roll = float(rng.uniform(-ROLL_TILT_MAX_DEG, ROLL_TILT_MAX_DEG))
    else:
        roll = float(rng.choice([-90.0, 90.0]) + rng.normal(0.0, ROLL_PORTRAIT_JITTER_DEG))
    lo, hi = max(LENS_MM_GLOBAL[0], scene.lens_mm[0]), min(LENS_MM_GLOBAL[1], scene.lens_mm[1])
    lo, hi = (lo, hi) if lo <= hi else scene.lens_mm
    focal = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return origin, target, roll, focal


def check_depth(depth: np.ndarray, reasons: Counter, normals: np.ndarray | None = None) -> bool:
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
    if normals is not None:
        n = normals[np.isfinite(depth)].reshape(-1, 3)[::7]
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
        dominant = np.median(n, axis=0)
        dominant /= max(np.linalg.norm(dominant), 1e-6)
        if np.mean(n @ dominant > np.cos(np.radians(15))) > MAX_SAME_FACING:
            reasons["single_surface"] += 1
            return False
    return True


def bright_enough(scene, lens, w, h, rng, reasons: Counter) -> bool:
    """4-spp probe: rejects frames that are essentially black before the expensive passes."""
    probe, _ = R.render_beauty(scene, lens, w, h, False, 4, 4, int(rng.integers(0, 10000)), "probe")
    if np.median(luminance(probe)) < MIN_MEDIAN_LUMINANCE:
        reasons["black"] += 1
        return False
    return True


def auto_exposure(lum: np.ndarray) -> float:
    """Linear scale a camera's meter would apply: brings the log-average luminance to mid-grey."""
    return float(AUTO_EXPOSURE_KEY / np.exp(np.mean(np.log(np.maximum(lum, 1e-4)))))


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


def widest_f_number(focal_mm: float) -> float:
    """Widest aperture common primes reach at this focal length (24/1.4, 35-85/1.2, 105/1.4, 135/1.8)."""
    if focal_mm < 30:
        return 1.4
    if focal_mm <= 90:
        return 1.2
    return 1.4 if focal_mm <= 110 else 1.8


def draw_lens_topup(kind: str, rng, pose_lens: R.Lens, depth, pos, focus_points, height_px: int):
    """One realistic top-up lens: 'wide' (wide open, focused on something near, as for a portrait or a close-up)
    or 'small' (f/8-16, focused a third of the way in, as for a landscape). Blur is whatever that setup gives."""
    finite = depth[np.isfinite(depth)]
    wide = kind == "wide"
    named_p, q = (TOPUP_WIDE_NAMED_P, TOPUP_WIDE_FOCUS_Q) if wide else (TOPUP_SMALL_NAMED_P, TOPUP_SMALL_FOCUS_Q)
    focus, source = None, "depth_quantile"
    if focus_points and rng.random() < named_p:
        for _ in range(MAX_LENS_TRIES):
            name = list(focus_points)[int(rng.integers(len(focus_points)))]
            p = np.asarray(focus_points[name][int(rng.integers(len(focus_points[name])))])
            z = pose_lens.depth_of(p)
            if z > MIN_CLEARANCE_M and (np.linalg.norm(pos - p, axis=-1) < FOCUS_VISIBLE_TOL_M).any():
                focus, source = z, f"point:{name}"
                break
    if focus is None:
        focus = float(np.quantile(finite, rng.uniform(*q)))
    focal_mm = pose_lens.lens_m * 1000
    unit = R.Lens(pose_lens.origin, pose_lens.target, focal_mm, S.SENSOR_HEIGHT_MM, 1.0, focus, pose_lens.roll_deg)
    p95_at_f1 = float(np.percentile(np.abs(unit.coc_map(depth, height_px)), 95)) / height_px
    if wide:
        n = math.exp(rng.uniform(math.log(widest_f_number(focal_mm)), math.log(TOPUP_WIDE_N_MAX)))
        n = max(n, p95_at_f1 / TOPUP_WIDE_P95_MAX)
    else:
        n = math.exp(rng.uniform(math.log(TOPUP_SMALL_N[0]), math.log(TOPUP_SMALL_N[1])))
    return dict(focus=float(focus), f_number=float(n), focus_source=source, achieved_frac=p95_at_f1 / n,
                stratum=-1, coc_out_of_range=False, topup_kind=kind)


def lens_key(spec: dict) -> dict:
    return {"focus": round(spec["focus"], 4), "n": round(spec["f_number"], 4), "aperture": "disc"}


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
        pose_lens = R.Lens(origin, target, focal, S.SENSOR_HEIGHT_MM, 1e5, 1.0, roll, scene_def.far_clip)
        depth, layers, g_sec = R.render_gbuffer(scene, pose_lens, w, h, int(rng.integers(0, 2 ** 31 - 1)))
        if not (check_depth(depth, reasons, layers["nn"]) and bright_enough(scene, pose_lens, w, h, rng, reasons)):
            continue
        # A pose whose depth range cannot give a blur inside COC_FRAC_RANGE at any allowed f-number (a wide lens
        # facing only distant surfaces) would yield a sample below the floor: draw another pose instead.
        lenses = []
        for j in range(args.lenses_per_pose):
            # two draws of one pose can land on the same focus point with a collapsed blur window and give the
            # same lens (cafe p000281 in train10k): redraw from a fresh stream until the lens is new
            for t in range(10):
                ls = draw_lens(scene_def, rng_for(scene_def, args, index, 2 + j, t), pose_lens, depth, layers["pos"],
                               bundle.focus_points, h)
                if lens_key(ls) not in [lens_key(o) for o in lenses]:
                    break
            lenses.append(ls)
        if any(ls["coc_out_of_range"] for ls in lenses):
            reasons["blur_unreachable"] += 1
            continue
        pose = dict(origin=origin, target=target, roll=roll, focal=focal, attempt=k)
        break
    if pose is None:
        return {"index": index, "failed": True, "reasons": dict(reasons)}

    pose_key = {"scene": scene_def.id, "seed": args.scene_seed, "env": args.env, "sampler_seed": args.sampler_seed,
                "origin": np.round(origin, 4).tolist(), "target": np.round(target, 4).tolist(),
                "roll": round(roll, 3), "focal": round(focal, 3), "res": [w, h]}
    pose_id = f"p{index:06d}_{short_hash(pose_key)[:8]}"
    out = {"index": index, "pose_id": pose_id, "reasons": dict(reasons), "lenses": lenses, "pose": pose_key}
    if args.dry_run:
        if args.topup:                         # the top-up lens this pose would get, without rendering it
            out["lenses"] = [topup_spec(index, scene_def, bundle, args, pose_lens, depth, layers, lenses, h)[1]]
            out["original_lenses"] = lenses
        return out

    pdir = base / pose_id
    if (pdir / ".done").exists():
        out["skipped"] = True
        return out
    if args.topup:
        return run_topup(index, scene_def, bundle, scene, args, pdir, pose_key, pose_lens, depth, layers, lenses,
                         out, fp, git, variant, mi_version)
    pdir.mkdir(parents=True, exist_ok=True)
    sharp_seed = int(rng_for(scene_def, args, index, 100).integers(0, 10000))
    sharp, sharp_sec = R.render_beauty(scene, pose_lens, w, h, False, args.spp, args.spp_per_pass, sharp_seed,
                                       "sharp")
    np.save(pdir / "depth.npy", depth)
    np.save(pdir / "coc_valid.npy", layers["coc_valid"][..., 0].astype(np.uint8))
    R.write_gbuffer(pdir / "gbuffer.exr", layers)
    write_exr(pdir / "sharp.exr", sharp)
    lum = luminance(sharp)
    pose_meta = {**pose_key, "pose_id": pose_id, "rejected_before": dict(reasons),
                 "sharp": {"spp": args.spp, "seconds": round(sharp_sec, 2), "seed": sharp_seed},
                 "sharp_median_luminance": float(np.median(lum)),
                 "exposure_gate_ok": bool(0.02 <= np.median(lum) <= 0.5),
                 "auto_exposure_scale": auto_exposure(lum),
                 "coc_valid_fraction": float(layers["coc_valid"].mean()),
                 "camera": pose_lens.metadata(w, h)}
    (pdir / "pose.json").write_text(json.dumps(pose_meta, indent=2))

    dof_spp = args.dof_spp or max(args.spp, scene_def.spp_hint)
    manifest = []
    for j, lens_spec in enumerate(lenses):
        lens = R.Lens(origin, target, focal, S.SENSOR_HEIGHT_MM, lens_spec["f_number"], lens_spec["focus"], roll,
                      scene_def.far_clip)
        lens_id = "lens_" + short_hash({"focus": round(lens_spec["focus"], 4), "n": round(lens_spec["f_number"], 4),
                                        "aperture": "disc"})[:8]
        ldir = pdir / lens_id
        ldir.mkdir(exist_ok=True)
        dof_seed = 10000 + int(rng_for(scene_def, args, index, 200 + j).integers(0, 10000))
        dof, dof_sec = R.render_beauty(scene, lens, w, h, True, dof_spp, args.spp_per_pass, dof_seed, "dof")
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
                         "exposure_gate_ok": pose_meta["exposure_gate_ok"],
                         "auto_exposure_scale": round(pose_meta["auto_exposure_scale"], 5)})
    (pdir / ".done").write_text("")
    with open(base / args.manifest_name, "a") as f:
        f.write("".join(json.dumps(m) + "\n" for m in manifest))
    return out


def topup_spec(index, scene_def, bundle, args, pose_lens, depth, layers, lenses, h):
    """The top-up lens of a pose: its kind, then a draw that differs from the pose's existing lenses."""
    u = rng_for(scene_def, args, index, 300).random()
    kind = next(k for k, c in zip((k for k, _ in TOPUP_KINDS), np.cumsum([p for _, p in TOPUP_KINDS])) if u < c)
    taken = [lens_key(o) for o in lenses]
    for t in range(10):
        spec = draw_lens_topup(kind, rng_for(scene_def, args, index, 301, t), pose_lens, depth, layers["pos"],
                               bundle.focus_points, h)
        if lens_key(spec) not in taken:
            break
    spec["topup"] = args.topup
    lens = R.Lens(pose_lens.origin, pose_lens.target, pose_lens.lens_m * 1000, S.SENSOR_HEIGHT_MM, spec["f_number"],
                  spec["focus"], pose_lens.roll_deg)
    coc = np.abs(lens.coc_map(depth, h))
    spec["coc_max_px"] = float(coc.max())
    spec["frac_ge32"], spec["frac_ge64"] = float((coc >= 32).mean()), float((coc >= 64).mean())
    return kind, spec, taken


def run_topup(index, scene_def, bundle, scene, args, pdir, pose_key, pose_lens, depth, layers, lenses, out, fp, git,
              variant, mi_version):
    """Render one extra, realistic lens for an existing pose (see the module docstring)."""
    w, h = args.width, args.height
    kind, spec, taken = topup_spec(index, scene_def, bundle, args, pose_lens, depth, layers, lenses, h)
    out["lenses"] = [spec]
    pdir.mkdir(parents=True, exist_ok=True)
    lens = R.Lens(pose_lens.origin, pose_lens.target, pose_lens.lens_m * 1000, S.SENSOR_HEIGHT_MM, spec["f_number"],
                  spec["focus"], pose_lens.roll_deg, scene_def.far_clip)
    lens_id = "lens_" + short_hash(lens_key(spec))[:8]
    ldir = pdir / lens_id
    ldir.mkdir(exist_ok=True)
    dof_spp = args.dof_spp or max(args.spp, scene_def.spp_hint)
    dof_seed = 20000 + int(rng_for(scene_def, args, index, 400).integers(0, 10000))
    dof, dof_sec = R.render_beauty(scene, lens, w, h, True, dof_spp, args.spp_per_pass, dof_seed, "dof")
    write_exr(ldir / "dof.exr", dof)
    np.save(ldir / "coc.npy", lens.coc_map(depth, h))
    sample_id = short_hash({**pose_key, "focus": round(spec["focus"], 4), "f_number": round(spec["f_number"], 4),
                            "aperture": "disc"})
    meta = {"contract_version": S.CONTRACT_VERSION, "sample_id": sample_id, "pose_id": pdir.name, "lens_id": lens_id,
            "scene": {"id": scene_def.id, "seed": args.scene_seed, "env": args.env, "fingerprint": fp,
                      "asset_version": scene_def.asset_version, **bundle.meta},
            "git": git, "variant": variant, "mitsuba": mi_version, "resolution": [w, h],
            "camera": lens.metadata(w, h), "lens_draw": spec,
            "coc": {"definition": "signed diameter in px, positive behind the focus plane; sky = at infinity",
                    "formula": "H f^2 (z - s) / (N z s sensor_h)"},
            "passes": {"sharp": "see pose.json", "dof": {"spp": dof_spp, "seconds": round(dof_sec, 2), "seed": dof_seed}},
            "sampler": {"sampler_seed": args.sampler_seed, "index": index, "topup": args.topup}}
    (ldir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (pdir / "topup.json").write_text(json.dumps({
        "topup": args.topup, "pose_id": pdir.name, "pose": pose_key, "kind": kind, "lens_id": lens_id,
        "original_lenses": ["lens_" + short_hash(k)[:8] for k in taken],
        "depth_md5": hashlib.md5(np.ascontiguousarray(depth).tobytes()).hexdigest()}, indent=1))
    (pdir / ".done").write_text("")
    entry = {"sample_id": sample_id, "path": f"{pdir.name}/{lens_id}", "stratum": -1, "topup": args.topup,
             "topup_kind": kind, "focal_mm": round(pose_lens.lens_m * 1000, 2), "f_number": round(spec["f_number"], 3),
             "focus_m": round(spec["focus"], 3), "roll_deg": round(pose_lens.roll_deg, 2),
             "coc_p95_frac": round(spec["achieved_frac"], 5), "coc_out_of_range": False}
    with open(pdir.parent / args.manifest_name, "a") as f:
        f.write(json.dumps(entry) + "\n")
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
    ap.add_argument("--topup", default=None, help="name of a top-up: one extra realistic lens per existing pose")
    ap.add_argument("--dry-run", action="store_true", help="draw and check poses and lenses, render no beauty passes")
    ap.add_argument("--manifest-name", default="manifest.jsonl",
                    help="manifest file in the env directory; give each parallel worker its own (manifest.<chunk>.jsonl)")
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
    scene_dict["sensor"] = R.Lens(first.origin, first.target, 50.0, S.SENSOR_HEIGHT_MM, 1e5, 1.0,
                                   far_clip=scene_def.far_clip).sensor(
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
        if args.topup and args.dry_run:
            print("TOPUP " + json.dumps({"index": index, "pose_id": r["pose_id"], "focal": r["pose"]["focal"], **l0,
                                         "orig_p95_px": [o["achieved_frac"] * args.height
                                                         for o in r["original_lenses"]]}), flush=True)
    print(f"\n{args.count} poses in {time.time() - t0:.1f}s | pose rejections {dict(reasons)} | failed {failed}")
    print(f"CoC strata (achieved p95 bins, low to high): {[strata[i] for i in range(COC_BINS)]} | "
          f"outside the CoC range {missed}/{n_lenses}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
