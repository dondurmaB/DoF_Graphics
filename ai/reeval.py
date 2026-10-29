"""Re-score saved checkpoints with the current metrics (for runs trained before a metric was added).

    python ai/reeval.py ROOT --ckpts w24=runs/smoke-train-350202/last.pt --exclude runs/bad_samples.txt

Writes runs/reeval-<name>/log.jsonl with one "eval" record (regular and hard_ crops), the same format as
smoke_train.py, so runs/cmp.py can read it.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D          # noqa: E402
import smoke_train as S   # noqa: E402
from model import build   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path/to/last.pt")
    ap.add_argument("--exclude")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    _, evalset, hardset, _, va, hard = S.datasets(args)
    dl_va, dl_hard = S.loader(evalset, va, args, False), S.loader(hardset, hard, args, False)
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        model = build(a.get("arch", "unet"), a.get("width"), a.get("film", False)).to(dev)
        model.load_state_dict(ck["model"])
        ev = S.evaluate(model, dl_va, dev)
        ev.update({"hard_" + k: v for k, v in S.evaluate(model, dl_hard, dev).items()})
        out = Path(f"runs/reeval-{name}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "log.jsonl").write_text(json.dumps({"eval": {"step": ck.get("step", 0), **ev}}) + "\n")
        print(name, {k: round(ev[k]["ratio"], 3) for k in ("all", "highlight_blur16", "energy_all",
                                                            "hard_energy_coc_32_up") if k in ev}, flush=True)


if __name__ == "__main__":
    main()
