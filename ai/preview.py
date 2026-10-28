"""Look at a trained model: full-frame predictions on held-out samples next to the sharp input, the classic gather
and the path-traced target.

    python ai/preview.py runs/smoke-train-347294/last.pt ROOT --exclude runs/bad_samples.txt --out runs/preview-K

Per sample it writes `<n>_full.png` (half resolution: sharp | target over classic | model), `<n>_zoom.png` (full
resolution on the window with the most blur >= 32 px: sharp | classic | model | target | |model - target| x 8) and
`<n>_pred.npy` (the model's full frame, linear radiance, float32 HDR). `metrics.json` holds full-frame L1 and its
ratio to identity per sample for the model and the classic gather (the model runs on the whole 1080p frame here,
not on 512 crops as in training).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "renderer" / "mitsuba"))
import data as D                     # noqa: E402
import smoke_train as S              # noqa: E402
from model import build              # noqa: E402
from traditional_dof import gather_dof  # noqa: E402


def to_png(t):
    """Model-space [3, H, W] -> 8-bit display image (Reinhard, gamma 2.2)."""
    lin = np.expm1(np.maximum(t, 0)).transpose(1, 2, 0)
    return (np.clip(lin / (1 + lin), 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def err_png(a, b, gain=8.0):
    e = np.clip(np.abs(a - b).mean(0) * gain, 0, 1)
    return (np.repeat(e[..., None], 3, -1) * 255 + 0.5).astype(np.uint8)


@torch.no_grad()
def predict_full(model, x, dev, mult=16):
    """Run on the whole frame: reflect-pad H and W up to a multiple of `mult`, crop back."""
    h, w = x.shape[-2:]
    ph, pw = -h % mult, -w % mult
    xt = F.pad(torch.from_numpy(x)[None].to(dev), (0, pw, 0, ph), mode="reflect")
    with torch.autocast("cuda", torch.bfloat16):
        y = model(xt, xt[:, 4:, 0, 0]).float()
    return y[0, :, :h, :w].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("root")
    ap.add_argument("--exclude")
    ap.add_argument("--out", default="runs/preview")
    ap.add_argument("--n", type=int, default=8, help="regular held-out samples, evenly spaced")
    ap.add_argument("--n-hard", type=int, default=4, help="extra samples from the hard (blur >= 32 px) set")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = build(a.get("arch", "unet"), a.get("width"), a.get("film", False)).to(dev).eval()
    model.load_state_dict(ck["model"])
    full, _, _, _, va, hard = S.datasets(args)
    pick = [va[i] for i in np.linspace(0, len(va) - 1, args.n).astype(int)]
    pick += [i for i in hard if i not in pick][:args.n_hard]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, i in enumerate(pick):
        s = full.samples[i]
        ex = D.load_example(s, crop=None, augment=False)
        m, x, y = ex["meta"], ex["x"], ex["y"]
        yh = predict_full(model, x, dev)
        sharp = D.read_exr(s.pose_dir / "sharp.exr")
        depth = np.load(s.pose_dir / "depth.npy").astype(np.float32)
        cl = D.to_model(gather_dof(sharp, depth, m["focal_mm"] / 1000, m["sensor_mm"] / 1000, m["f_number"],
                                   m["focus_m"]), m["auto_exposure_scale"]).transpose(2, 0, 1)
        e_id = np.abs(x[:3] - y).mean()
        rec = {"n": n, "lens_dir": m["lens_dir"], "scene": m["scene"], "f_number": m["f_number"],
               "focal_mm": m["focal_mm"], "focus_m": m["focus_m"], "hard": i in hard, "identity_l1": float(e_id),
               "model_ratio": float(np.abs(yh - y).mean() / e_id), "classic_ratio": float(np.abs(cl - y).mean() / e_id)}
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        np.save(out / f"{n:02d}_pred.npy", D.from_model(yh.transpose(1, 2, 0), m["auto_exposure_scale"]))
        half = [to_png(t[:, ::2, ::2]) for t in (x[:3], y, cl, yh)]
        grid = np.concatenate([np.concatenate(half[:2], 1), np.concatenate(half[2:], 1)], 0)
        Image.fromarray(grid).save(out / f"{n:02d}_full.png")
        y0, x0, _ = D.hard_crop_at(np.load(s.lens_dir / "coc.npy"), args.crop, S.HARD_COC)
        win = (slice(None), slice(y0, y0 + args.crop), slice(x0, x0 + args.crop))
        zoom = [to_png(t[win]) for t in (x[:3], cl, yh, y)] + [err_png(yh[win], y[win])]
        Image.fromarray(np.concatenate(zoom, 1)).save(out / f"{n:02d}_zoom.png")
    (out / "metrics.json").write_text(json.dumps(rows, indent=1))
    print(f"model / identity, mean over samples: {np.mean([r['model_ratio'] for r in rows]):.3f}; "
          f"classic: {np.mean([r['classic_ratio'] for r in rows]):.3f}", flush=True)


if __name__ == "__main__":
    main()
