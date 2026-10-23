"""Temporary noise ablation for night_street (run on the cluster via remote_run.sh; delete when done)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import render as R  # noqa: E402
import scene_api as S  # noqa: E402

out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
R.pick_variant("cuda_ad_rgb")
import mitsuba as mi  # noqa: E402

sc = S.load_scene("night_street")
bundle = S.build_checked(sc, S.make_context(sc, HERE / "generated", None, "wet"))
v = sc.views["kerb_wide"]
L = R.Lens(v.origin, v.target, 50.0, 24.0, 1e5, 10.0)
W, H, SPP = 960, 540, 256
lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])


def variant(name):
    d = dict(bundle.scene)
    for k, val in list(d.items()):
        if not isinstance(val, dict):
            continue
        if name == "nodelta" and val.get("type") in ("plastic", "thindielectric"):
            d[k] = {"type": "roughplastic", "diffuse_reflectance": val.get("diffuse_reflectance", {"type": "rgb", "value": [0.01] * 3}), "alpha": 0.12}
        if name == "lantern_boost" and k.startswith("lamp_") and "emitter" in val:
            d[k] = dict(val, emitter=dict(val["emitter"], sampling_weight=val["emitter"]["sampling_weight"] * 20))
        if name == "only_lamps" and "emitter" in val and not k.startswith("lamp_"):
            d[k] = {kk: vv for kk, vv in val.items() if kk != "emitter"}
    if name == "only_lamps":
        d["sky"] = {"type": "constant", "radiance": {"type": "rgb", "value": [0.0] * 3}}
    d["integrator"] = {"type": "path", "max_depth": sc.max_depth, "rr_depth": sc.rr_depth}
    d["sensor"] = L.sensor(W, H, 1, False)
    return mi.load_dict(d)


crop = (slice(200, 520), slice(200, 880))
args = sys.argv[1:sys.argv.index("--out")]
if args and args[0] == "cam":                                   # cam <env> label ox,oy,oz tx,ty,tz lens spp ...
    env = args[1]
    bundle = S.build_checked(sc, S.make_context(sc, HERE / "generated", None, env))
    rest = args[2:]
    for i in range(0, len(rest), 5):
        label, o, tg, lens_mm, spp = rest[i:i + 5]
        O, T = [tuple(float(x) for x in s.split(",")) for s in (o, tg)]
        Lc = R.Lens(O, T, float(lens_mm), 24.0, 1e5, 10.0)
        d = dict(bundle.scene)
        d["integrator"] = {"type": "path", "max_depth": sc.max_depth, "rr_depth": sc.rr_depth}
        d["sensor"] = Lc.sensor(W, H, 1, False)
        scene = mi.load_dict(d)
        img, sec = R.render_beauty(scene, Lc.sensor(W, H, int(spp), False), int(spp), 64, 1, label)
        R.save_png(out / f"{label}.png", R.tonemap(img, 0.0))
        print(f"{label}: {sec:.1f}s median lum {np.median(lum(img)):.4f}", flush=True)
    args = []
for name in args:
    scene = variant(name)
    a, _ = R.render_beauty(scene, L.sensor(W, H, SPP, False), SPP, 64, 1, name)
    b, _ = R.render_beauty(scene, L.sensor(W, H, SPP, False), SPP, 64, 2, name)
    la, lb = lum(a)[crop], lum(b)[crop]
    m = 0.5 * (la + lb)
    rel = np.abs(la - lb) / (m + 0.01)
    print(f"{name:14s} car-crop: mean lum {m.mean():.4f}  rel diff mean {rel.mean():.3f}  p99 {np.percentile(rel, 99):.2f}  "
          f"p99.9 {np.percentile(rel, 99.9):.2f}", flush=True)
    R.save_png(out / f"{name}.png", R.tonemap(0.5 * (a + b), 0.0))
