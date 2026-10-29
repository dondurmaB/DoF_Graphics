"""What each arm of a twoarm2 model does, per blur bin, on the fixed held-out crops (regular and hard).

    python ai/arm_diag.py ROOT --ckpt runs/smoke-train-352594/last.pt --exclude runs/bad_samples.txt

Per bin: L1 ratio to identity of the final output, of arm A (U-Net) alone and of arm B (large-blur arm) alone,
the mean mask m (share given to arm B), and the share of pixels where B is closer to the target than A.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D          # noqa: E402
import smoke_train as S   # noqa: E402
from model import build   # noqa: E402

BINS = {"in_focus": (0, 1), "coc_1_4": (1, 4), "coc_4_16": (4, 16), "coc_16_32": (16, 32), "coc_32_64": (32, 64),
        "coc_64_up": (64, 1e9)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exclude")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build(a["arch"], a.get("width"), a.get("film", False)).to(dev).eval()
    model.load_state_dict(ck["model"])
    _, evalset, hardset, _, va, hard = S.datasets(args)
    for setname, ds, idx in (("regular", evalset, va), ("hard", hardset, hard)):
        acc = {}
        for i in idx:
            item = ds[i]
            x, y = item["x"][None].to(dev), item["y"][None].to(dev)
            own = S.regions(item["coc"][None].to(dev))["own"]
            with torch.autocast("cuda", torch.bfloat16):
                out = model(x, item["settings"][None].to(dev)).float()
            aux = model.aux
            err = {k: (v - y).abs().mean(1, keepdim=True) for k, v in
                   (("out", out), ("A", aux["A"]), ("B", aux["B"]), ("id", x[:, :3]))}
            for b, (lo, hi) in BINS.items():
                msk = ((own >= lo) & (own < hi)).float()
                if msk.sum() == 0:
                    continue
                r = acc.setdefault(b, {k: 0.0 for k in ("out", "A", "B", "id", "m", "b_wins", "n")})
                for k in ("out", "A", "B", "id"):
                    r[k] += (err[k] * msk).sum().item()
                r["m"] += (aux["m"] * msk).sum().item()
                r["b_wins"] += ((err["B"] < err["A"]).float() * msk).sum().item()
                r["n"] += msk.sum().item()
        print(f"== {setname} crops   (ratio to identity; m = mean share of arm B)", flush=True)
        print(f"  {'bin':10s} {'pixels':>9s} {'output':>7s} {'arm A':>7s} {'arm B':>7s} {'m':>6s} {'B better':>9s}")
        for b, r in acc.items():
            print(f"  {b:10s} {int(r['n']):9d} {r['out'] / r['id']:7.3f} {r['A'] / r['id']:7.3f} {r['B'] / r['id']:7.3f} "
                  f"{r['m'] / r['n']:6.3f} {r['b_wins'] / r['n']:9.1%}", flush=True)


if __name__ == "__main__":
    main()
