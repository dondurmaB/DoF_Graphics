"""Score the classic screen-space DoF (renderer/mitsuba/traditional_dof.py) on the same held-out crops and regions
as ai/smoke_train.py, so learned models have a non-learned reference.

    python ai/classic_eval.py ROOT --exclude runs/bad_samples.txt --out runs/classic

The gather needs the whole frame (blur reaches in from outside a crop), so each held-out sample is blurred at full
resolution from its sharp image, depth and lens settings (the model's inputs, nothing else), converted to model
space and cut at the sample's fixed evaluation crop.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "renderer" / "mitsuba"))
import data as D                     # noqa: E402
import smoke_train as S              # noqa: E402
from traditional_dof import gather_dof  # noqa: E402


class WithClassic(Dataset):
    """An evaluation dataset item plus `yh`: the classic gather of the full frame, cropped like the item."""

    def __init__(self, base):
        self.base = base
        self.samples = base.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        m, s = item["meta"], self.samples[i]
        sharp = D.read_exr(s.pose_dir / "sharp.exr")
        depth = np.load(s.pose_dir / "depth.npy").astype(np.float32)
        out = gather_dof(sharp, depth, m["focal_mm"] / 1000, m["sensor_mm"] / 1000, m["f_number"], m["focus_m"])
        y0, x0 = m["crop_yx"]
        c = item["x"].shape[-1]
        yh = D.to_model(out[y0:y0 + c, x0:x0 + c], m["auto_exposure_scale"])
        return {**item, "yh": torch.from_numpy(np.ascontiguousarray(yh.transpose(2, 0, 1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--exclude")
    ap.add_argument("--out", default="runs/classic")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, evalset, hardset, _, va, hard = S.datasets(args)
    ev = {}
    for prefix, ds, idx in (("", evalset, va), ("hard_", hardset, hard)):
        dl = DataLoader(Subset(WithClassic(ds), idx), batch_size=args.batch, num_workers=args.workers,
                        collate_fn=S.collate, worker_init_fn=S.one_thread)
        ev.update({prefix + k: v for k, v in S.evaluate(None, dl, dev, pred=lambda b: b["yh"]).items()})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.jsonl").write_text(json.dumps({"eval": {"step": 0, **ev}}) + "\n")
    for k, v in ev.items():
        print(f"  {k:18s} frac {v['frac']:.4f}  l1 {v['l1']:.5f}  identity {v['identity']:.5f}  ratio {v['ratio']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
