#!/usr/bin/env python3
"""Run the SCENE_CONTRACT.md acceptance checklist (section 9) on one scene.

  python renderer/mitsuba/check_scene.py cafe
  python renderer/mitsuba/check_scene.py cafe --views home --envs clear --keep /tmp/check

Prints PASS / WARN / FAIL per check and exits 1 if anything FAILs. Previews are
noisy by design (default 64 spp), so noise-sensitive checks only WARN.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene_api as S  # noqa: E402

MEDIAN_LUMINANCE_BAND = (0.02, 0.5)
MAX_TRIANGLES = 8_000_000
MAX_WARM_BUILD_S = 60.0


# --- pure metrics (no Mitsuba) ------------------------------------------------

def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def median5x5(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    p = np.pad(img.astype(np.float32), 2, mode="reflect")
    return np.median(np.stack([p[dy:dy + h, dx:dx + w] for dy in range(5) for dx in range(5)]), axis=0)


def firefly_fraction(lum: np.ndarray) -> float:
    """Pixels brighter than 4x their 5x5 median plus 0.5. A lit bulb has bright
    neighbours, so only isolated outliers count (3 pixels at each corner of a
    rectangular emitter still do, which is why this only warns)."""
    return float(np.mean(lum > 4.0 * median5x5(lum) + 0.5))


def sky_fraction(depth: np.ndarray) -> float:
    return float(1.0 - np.isfinite(depth).mean())


# --- report ---------------------------------------------------------------------

class Report:
    def __init__(self):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.counts[status] += 1
        print(f"{status:<5} {name:<22} {detail}", flush=True)

    def check(self, ok: bool, name: str, detail: str = "", otherwise: str = "FAIL") -> None:
        self.add("PASS" if ok else otherwise, name, detail)


# --- checks ---------------------------------------------------------------------

def check_declaration(rep: Report, scene: S.SceneDef) -> None:
    errs = S.validate_definition(scene)
    rep.check(not errs, "definition", "; ".join(errs) or "valid")
    split_errs = [e for e in S.validate_splits(S.list_scenes()) if repr(scene.id) in e]
    rep.check(not split_errs, "splits.json", "; ".join(split_errs) or f"{scene.id} is assigned to a split")


def check_build(rep: Report, scene: S.SceneDef, assets: Path) -> None:
    def build(seed, timed=False):
        ctx = S.make_context(scene, assets, seed, "clear")
        t0 = time.perf_counter()
        bundle = scene.build(ctx)
        return bundle, S.fingerprint(bundle.scene, [ctx.shared_dir, ctx.assets_dir]), time.perf_counter() - t0

    seed = scene.default_seed
    try:
        build(seed)                                   # cold: fills the shared cache
        bundle, fp_a, warm_s = build(seed)
        _, fp_b, _ = build(seed)
        _, fp_c, _ = build(seed + 1)
    except Exception as e:                            # a scene's own bug: report, don't crash the checker
        rep.add("FAIL", "build", f"{type(e).__name__}: {e}")
        return
    errs = S.validate_bundle(scene, bundle)
    rep.check(not errs, "bundle", "; ".join(errs) or f"{len(bundle.focus_points)} focus points")
    rep.check(fp_a == fp_b, "deterministic", f"same seed {seed}: fingerprint {fp_a} vs {fp_b}")
    rep.check(fp_a != fp_c, "seed changes scene", f"seed {seed}: {fp_a}, seed {seed + 1}: {fp_c}")
    rep.check(warm_s <= MAX_WARM_BUILD_S, "build time (warm)", f"{warm_s:.1f}s of {MAX_WARM_BUILD_S:.0f}s")


def render_preview(R, scene_id, env, view, out: Path, args) -> tuple[dict | None, str]:
    argv = ["--scene", scene_id, "--env", env, "--view", view, "--spp", str(args.spp), "--res", args.res,
            "--passes", "sharp,dof,gbuffer", "--out", str(out), "--assets", args.assets]
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            R.main(argv)
    except (Exception, SystemExit) as e:
        return None, f"{type(e).__name__}: {e} | {log.getvalue()[-300:]}"
    return json.loads((out / "metadata.json").read_text()), ""


def check_preview(rep: Report, mi, scene: S.SceneDef, tag: str, out: Path, meta: dict, height: int) -> None:
    sharp = np.array(mi.Bitmap(str(out / "sharp.exr")))[..., :3]
    dof = np.array(mi.Bitmap(str(out / "dof.exr")))[..., :3]
    depth = np.load(out / "depth.npy")

    rep.check(bool(np.isfinite(sharp).all() and np.isfinite(dof).all()), tag + " finite", "no NaN/inf in sharp, dof")

    lo, hi = MEDIAN_LUMINANCE_BAND
    med = float(np.median(luminance(sharp)))
    rep.check(lo <= med <= hi, tag + " exposure", f"median luminance {med:.3f} (band {lo}-{hi})",
              otherwise="WARN" if "night" in scene.tags else "FAIL")

    ratio = float(dof.mean() / sharp.mean() - 1.0)
    rep.add("PASS" if abs(ratio) < 0.01 else "WARN" if abs(ratio) <= 0.03 else "FAIL", tag + " energy",
            f"mean(dof)/mean(sharp) - 1 = {ratio:+.4f}")

    ff = max(firefly_fraction(luminance(sharp)), firefly_fraction(luminance(dof)))
    rep.check(ff <= 0.001, tag + " fireflies", f"{ff * 100:.3f}% of pixels (target <= 0.1%, noisy at {meta['passes']['sharp']['spp']} spp)",
              otherwise="WARN")

    sky = sky_fraction(depth)
    rep.check(sky <= 0.5, tag + " sky", f"sky fraction {sky:.2f}", otherwise="WARN")
    near, far = meta["depth_range_m"]
    rep.add("PASS", tag + " depth", f"{near:.2f} - {far:.2f} m")
    coc = meta["coc_px_percentiles"]
    p95 = coc["95"]
    rep.check(1.0 <= p95 <= 0.05 * height, tag + " CoC",
              f"p5/p50/p95 = {coc['5']:.1f}/{coc['50']:.1f}/{p95:.1f} px (want p95 in 1 - {0.05 * height:.0f})",
              otherwise="WARN")
    rep.check(meta["triangles"] <= MAX_TRIANGLES, tag + " triangles", f"{meta['triangles']:,} of {MAX_TRIANGLES:,}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--envs", help="comma list (default: all the scene declares)")
    ap.add_argument("--views", help="comma list (default: all the scene's views)")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--res", default="640x360")
    ap.add_argument("--keep", help="keep the preview renders in this directory")
    ap.add_argument("--assets", default=str(HERE / "generated"))
    args = ap.parse_args(argv)

    import render as R

    scene = S.load_scene(args.scene)
    envs = args.envs.split(",") if args.envs else list(scene.envs)
    views = args.views.split(",") if args.views else list(scene.views)
    if not set(envs) <= set(scene.envs) or not set(views) <= set(scene.views):
        ap.error(f"{scene.id} declares envs {scene.envs} and views {tuple(scene.views)}")
    height = int(args.res.lower().split("x")[1])

    rep = Report()
    check_declaration(rep, scene)
    variant = R.pick_variant("auto")
    import mitsuba as mi

    print(f"-- {scene.id} ({scene.group}, owner {scene.owner}) on {variant}")
    check_build(rep, scene, Path(args.assets))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(args.keep) if args.keep else Path(tmp)
        for env in envs:
            for view in views:
                tag, out = f"{env}/{view}", root / env / view
                meta, err = render_preview(R, scene.id, env, view, out, args)
                if meta is None:
                    rep.add("FAIL", tag + " render", err)
                else:
                    check_preview(rep, mi, scene, tag, out, meta, height)

    c = rep.counts
    print(f"-- {c['PASS']} pass, {c['WARN']} warn, {c['FAIL']} fail")
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
