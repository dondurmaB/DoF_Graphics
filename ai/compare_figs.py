"""Side-by-side examples: sharp input, the classic method, learned models and the path-traced target, per frame.

    python ai/compare_figs.py ROOT --ckpts "U-Net w24=runs/smoke-train-358859-0/last.pt" \
        "Pyramid w24=runs/smoke-train-358859-1/last.pt" --exclude runs/bad_samples.txt --out runs/compare_figs

Picks three held-out frames (never trained on): the held-out top-up lens with the most blur >= 64 px, the hard
held-out crop frame with the most blur >= 32 px, and a regular held-out frame. Per frame it writes
`<k>_full.png` (whole frame at half resolution, one column per method), `<k>_zoom.png` (512 x 512 window with the
most blur, full resolution) and adds a line to `frames.json` with each method's error relative to doing nothing
on the whole frame. Display only: Reinhard tone map and gamma 2.2 on the HDR values.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D              # noqa: E402
import smoke_train as S       # noqa: E402
from classic_triton import gather_triton  # noqa: E402
from model import build       # noqa: E402


def to_png(t):
    lin = np.expm1(np.maximum(t, 0)).transpose(1, 2, 0)
    return (np.clip(lin / (1 + lin), 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 + 7 * len(text), 18), fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ckpts", nargs="+", required=True, help='"label=path/to/last.pt"')
    ap.add_argument("--exclude")
    ap.add_argument("--out", default="runs/compare_figs")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    models = []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = build(a.get("arch", "unet"), a.get("width"), a.get("film", False)).to(dev).eval()
        m.load_state_dict(ck["model"])
        models.append((name, m))
    full, _, _, _, va, hard = S.datasets(args)

    def frac(i, t):
        return float((np.abs(np.load(full.samples[i].lens_dir / "coc.npy")) >= t).mean())
    picks = []
    if args.va_topup:
        picks.append(("wide-open top-up lens", max(args.va_topup, key=lambda i: frac(i, 64))))
    picks.append(("strong-blur frame", max(hard, key=lambda i: frac(i, 32))))
    rest = [i for i in va if i not in hard]
    picks.append(("regular frame", rest[len(rest) // 2]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (kind, i) in enumerate(picks):
        s = full.samples[i]
        ex = D.load_example(s, crop=None, augment=False)
        m, x, y = ex["meta"], ex["x"], ex["y"]
        h, w = x.shape[-2:]
        xt = F.pad(torch.from_numpy(x)[None].to(dev), (0, -w % 16, 0, -h % 16), mode="reflect")
        preds = {}
        sharp = torch.from_numpy(D.read_exr(s.pose_dir / "sharp.exr").transpose(2, 0, 1).copy()).to(dev)
        depth = torch.from_numpy(np.load(s.pose_dir / "depth.npy").astype(np.float32)).to(dev)
        cl = gather_triton(sharp, depth, m["focal_mm"] / 1000, m["sensor_mm"] / 1000, m["f_number"], m["focus_m"])
        preds["Classic method"] = torch.log1p(cl.clamp(min=0) * m["auto_exposure_scale"]).cpu().numpy()
        for name, mdl in models:
            with torch.autocast("cuda", torch.bfloat16):
                preds[name] = mdl(xt, xt[:, 4:, 0, 0]).float()[0, :, :h, :w].cpu().numpy()
        e_id = float(np.abs(x[:3] - y).mean())
        cols = [("Input (sharp)", x[:3])] + list(preds.items()) + [("Ground truth (path traced)", y)]
        rec = {"k": k, "kind": kind, "lens_dir": m["lens_dir"], "scene": m["scene"], "focal_mm": m["focal_mm"],
               "f_number": m["f_number"], "focus_m": m["focus_m"],
               "max_blur_px": float(np.abs(np.load(s.lens_dir / "coc.npy")).max()),
               "error_vs_nothing": {n: float(np.abs(p - y).mean() / e_id) for n, p in preds.items()}}
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        Image.fromarray(np.concatenate([label(to_png(t[:, ::2, ::2]), n) for n, t in cols], 1)).save(out / f"{k}_full.png")
        y0, x0, _ = D.hard_crop_at(np.load(s.lens_dir / "coc.npy"), args.crop, 32.0)
        win = (slice(None), slice(y0, y0 + args.crop), slice(x0, x0 + args.crop))
        Image.fromarray(np.concatenate([label(to_png(t[win]), n) for n, t in cols], 1)).save(out / f"{k}_zoom.png")
    (out / "frames.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
