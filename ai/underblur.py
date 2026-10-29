"""E0 (experiment_plan.md): do the models under-blur large defocus? Splits the error into coarse and fine detail.

    python ai/underblur.py ROOT --ckpts K=runs/smoke-train-347294/last.pt w24=runs/smoke-train-350202/last.pt \
        --exclude runs/bad_samples.txt --out runs/underblur.json

On the fixed held-out crops (regular and hard), per blur bin, for the sharp input x, the target y, each model's
prediction and the classic gather (`speed.gather_torch` on the full frame, cropped):

  detail     mean |t - G(t)|, the fine detail left in an image (G = Gaussian, sigma 2 px)
  detail_vs_target   detail(pred) / detail(y): > 1 means sharper than the lens image (under-blur), < 1 over-blur
  err_coarse mean |G4(pred) - G4(y)| (sigma 4 px): wrong brightness or wrong large-scale shape
  err_fine   mean |(pred - G4 pred) - (y - G4 y)|: wrong fine detail
  bias       mean (pred - y)

All in model space (log(1 + L e)), pixel-weighted over the bin.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402
import smoke_train as S   # noqa: E402
from model import build   # noqa: E402
from speed import gather_torch  # noqa: E402

BINS = {"in_focus": (0, 1), "coc_1_4": (1, 4), "coc_4_16": (4, 16), "coc_16_64": (16, 64), "coc_64_up": (64, 1e9)}


def gauss(t, sigma):
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=t.device, dtype=t.dtype) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())
    c = t.shape[1]
    t = F.pad(t, (r, r, r, r), mode="reflect")
    t = F.conv2d(t, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(t, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def classic_crop(lens_dir, meta, crop, dev):
    s = Path(lens_dir).parent
    sharp = torch.from_numpy(D.read_exr(s / "sharp.exr").transpose(2, 0, 1).copy()).to(dev)
    depth = torch.from_numpy(np.load(s / "depth.npy").astype(np.float32)).to(dev)
    out = gather_torch(sharp, depth, meta["focal_mm"] / 1000, meta["sensor_mm"] / 1000, meta["f_number"],
                       meta["focus_m"])
    y0, x0 = meta["crop_yx"]
    lin = out[:, y0:y0 + crop, x0:x0 + crop].clamp(min=0)
    return torch.log1p(lin * meta["auto_exposure_scale"])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path/to/last.pt")
    ap.add_argument("--exclude")
    ap.add_argument("--out", default="runs/underblur.json")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    models = {}
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = build(a.get("arch", "unet"), a.get("width"), a.get("film", False)).to(dev).eval()
        m.load_state_dict(ck["model"])
        models[name] = m
    _, evalset, hardset, _, va, hard = S.datasets(args)
    result = {}
    for setname, ds, idx in (("regular", evalset, va), ("hard", hardset, hard)):
        acc = {}
        for i in idx:
            item = ds[i]
            x = item["x"][None].to(dev)
            y = item["y"][None].to(dev)
            own = S.regions(item["coc"][None].to(dev))["own"]
            preds = {"sharp": x[:, :3], "target": y, "classic": classic_crop(item["meta"]["lens_dir"], item["meta"],
                                                                              args.crop, dev)[None]}
            with torch.autocast("cuda", torch.bfloat16):
                for name, m in models.items():
                    preds[name] = m(x, item["settings"][None].to(dev)).float()
            g4y = gauss(y, 4.0)
            for name, p in preds.items():
                detail = (p - gauss(p, 2.0)).abs().mean(1, keepdim=True)
                g4p = gauss(p, 4.0)
                coarse = (g4p - g4y).abs().mean(1, keepdim=True)
                fine = ((p - g4p) - (y - g4y)).abs().mean(1, keepdim=True)
                bias = (p - y).mean(1, keepdim=True)
                for b, (lo, hi) in BINS.items():
                    msk = ((own >= lo) & (own < hi)).float()
                    n = msk.sum().item()
                    if n == 0:
                        continue
                    a = acc.setdefault(b, {}).setdefault(name, np.zeros(5))
                    a += [(detail * msk).sum().item(), (coarse * msk).sum().item(), (fine * msk).sum().item(),
                          (bias * msk).sum().item(), n]
        res = {}
        for b, per in acc.items():
            dt = per["target"][0] / per["target"][4]
            res[b] = {name: {"detail": v[0] / v[4], "detail_vs_target": v[0] / v[4] / dt, "err_coarse": v[1] / v[4],
                             "err_fine": v[2] / v[4], "bias": v[3] / v[4], "pixels": int(v[4])}
                      for name, v in per.items()}
        result[setname] = res
        print(f"== {setname} crops", flush=True)
        for b, per in res.items():
            print(f"  {b}  ({per['target']['pixels']} px)", flush=True)
            for name, v in per.items():
                print(f"    {name:8s} detail/target {v['detail_vs_target']:6.2f}  coarse err {v['err_coarse']:.4f}  "
                      f"fine err {v['err_fine']:.4f}  bias {v['bias']:+.4f}", flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
