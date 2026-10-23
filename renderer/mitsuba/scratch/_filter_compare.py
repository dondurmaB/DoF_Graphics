"""Scratch: gaussian pixel filter (mi.render, current dataset path) vs box filter (lens_trace), same view."""
import json
import sys
import time
from pathlib import Path

import numpy as np

import render as R
import scene_api as S
import lens_trace as LT

SCENE = sys.argv[1] if len(sys.argv) > 1 else "desk"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else f"output/dev/filter-compare/{SCENE}")
W, H = 1280, 720
SPP = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

args = R.parse_args(["--scene", SCENE])
R.pick_variant("auto")
import mitsuba as mi  # noqa: E402

sd = S.load_scene(SCENE)
view = sd.views[sd.default_view]
b = S.build_checked(sd, S.make_context(sd, R.HERE / "generated", None, "clear"))
target = view.focus or next(iter(b.focus_points))
probe = R.Lens(view.origin, view.target, args.lens, args.sensor_height, args.f_number, 1.0, view.roll_deg, sd.far_clip)
focus = probe.depth_of(b.focus_points[target][0])
lens = R.Lens(view.origin, view.target, args.lens, args.sensor_height, args.f_number, focus, view.roll_deg, sd.far_clip)
d = dict(b.scene)
d["integrator"] = {"type": "path", "max_depth": sd.max_depth, "rr_depth": sd.rr_depth}
d["sensor"] = lens.sensor(W, H, 1, thin_lens=False)
scene = mi.load_dict(d)
OUT.mkdir(parents=True, exist_ok=True)

imgs = {}
for kind, thin in (("sharp", False), ("dof", True)):
    imgs[f"{kind}_gauss"], sec = R.render_beauty(scene, lens.sensor(W, H, SPP, thin), SPP, 256, 7, f"{kind} gaussian")
    t0, acc = time.time(), np.zeros((H, W, 3))
    for k in range(SPP // 64):
        acc += LT.trace(scene, lens, W, H, 64, 7 * 1000 + k, aperture=thin)
    imgs[f"{kind}_box"] = (acc / (SPP // 64)).astype(np.float32)
    print(f"{kind}: gaussian {sec:.1f}s, box {time.time() - t0:.1f}s", flush=True)


def lum(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def grad_energy(x):
    y = R.tonemap(x, 0)
    l = lum(y)
    return float(np.mean(np.diff(l, axis=0) ** 2) + np.mean(np.diff(l, axis=1) ** 2))


def gblur(x, s):
    if s <= 0:
        return x
    r = int(np.ceil(4 * s))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    y = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, x)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 1, y)


stats = {}
for kind in ("sharp", "dof"):
    g, bx = imgs[f"{kind}_gauss"], imgs[f"{kind}_box"]
    lg, lb = lum(R.tonemap(g, 0)), lum(R.tonemap(bx, 0))
    # Extra gaussian blur on the box image that best reproduces the gaussian-filter image.
    errs = {s: float(np.mean(np.abs(gblur(lb, s) - lg))) for s in np.arange(0, 0.81, 0.05).round(2)}
    best = min(errs, key=errs.get)
    stats[kind] = {"edge_energy_gauss": grad_energy(g), "edge_energy_box": grad_energy(bx),
                   "box_sharper_by_pct": 100 * (grad_energy(bx) / grad_energy(g) - 1),
                   "mean_abs_diff_8bit": 255 * float(np.mean(np.abs(lb - lg))),
                   "extra_blur_sigma_px_that_matches": best, "fit_err_8bit": {str(k): 255 * v for k, v in errs.items()}}
    print(kind, json.dumps({k: v for k, v in stats[kind].items() if k != "fit_err_8bit"}), flush=True)
(OUT / "stats.json").write_text(json.dumps(stats, indent=1))

# Figure: crop where the DoF image has the most in-focus detail, shown 4x with nearest-neighbour.
from PIL import Image, ImageDraw  # noqa: E402

C, Z = 120, 4
le = lum(R.tonemap(imgs["dof_gauss"], 0))
e = np.abs(np.diff(le, axis=0))[:, :-1] + np.abs(np.diff(le, axis=1))[:-1, :]
ii = np.cumsum(np.cumsum(np.pad(e, ((1, 0), (1, 0))), 0), 1)
best, at = -1, (0, 0)
for y in range(0, H - C - 1, 8):
    for x in range(0, W - C - 1, 8):
        s = ii[y + C, x + C] - ii[y, x + C] - ii[y + C, x] + ii[y, x]
        if s > best:
            best, at = s, (y, x)
y0, x0 = at


def png(x):
    return Image.fromarray((np.clip(R.tonemap(x, 0), 0, 1) * 255 + 0.5).astype(np.uint8))


for name, x in imgs.items():
    png(x).save(OUT / f"{name}.png")
pad, lab = 12, 26
cw = C * Z
fig = Image.new("RGB", (pad + 3 * (cw + pad), pad + 2 * (lab + cw + pad) + lab + 360 + pad), (24, 24, 26))
dr_ = ImageDraw.Draw(fig)
for r, kind in enumerate(("sharp", "dof")):
    g, bx = imgs[f"{kind}_gauss"], imgs[f"{kind}_box"]
    bb = np.stack([gblur(bx[..., i], 0.41) for i in range(3)], -1).astype(np.float32)
    tiles = [(f"{kind.upper()}  gaussian filter (now)", png(g).crop((x0, y0, x0 + C, y0 + C))),
             (f"{kind.upper()}  box filter (fast)", png(bx).crop((x0, y0, x0 + C, y0 + C))),
             ("box + 0.41 px gaussian blur (should match left)", png(bb).crop((x0, y0, x0 + C, y0 + C)))]
    top = pad + r * (lab + cw + pad)
    for c, (t, im) in enumerate(tiles):
        x = pad + c * (cw + pad)
        dr_.text((x, top + 6), t, fill=(230, 227, 220))
        fig.paste(im.resize((cw, cw), Image.NEAREST), (x, top + lab))
top = pad + 2 * (lab + cw + pad)
dr_.text((pad, top + 6), f"{SCENE}, {W}x{H}, {SPP} spp each; red box = crop above (shown {Z}x)", fill=(230, 227, 220))
full = png(imgs["dof_gauss"]).resize((640, 360))
fd = ImageDraw.Draw(full)
fd.rectangle([x0 / 2, y0 / 2, (x0 + C) / 2, (y0 + C) / 2], outline=(255, 60, 60), width=2)
fig.paste(full, (pad, top + lab))
fig.save(OUT / "compare.png")
print("crop at", x0, y0, "->", OUT / "compare.png")
