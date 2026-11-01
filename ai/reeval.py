"""Re-score saved checkpoints with the current metrics (for runs trained before a metric was added).

    python ai/reeval.py ROOT --ckpts w24=runs/smoke-train-350202/last.pt --exclude runs/bad_samples.txt

Writes runs/reeval-<name>/log.jsonl with one "eval" record (regular, hard_ and top_ crops: top_ = held-out top-up
lenses, experiment_plan.md D2), the same format as smoke_train.py, so runs/cmp.py can read it. `--classic NAME` also
scores the classic gather (speed.gather_torch on the GPU, full frame, cropped like each item).
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
    ap.add_argument("--ckpts", nargs="*", default=[], help="name=path/to/last.pt")
    ap.add_argument("--classic", default=None, help="also score the classic gather under this name")
    ap.add_argument("--exclude")
    ap.add_argument("--crop", type=int, default=D.CROP)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    _, evalset, hardset, _, va, hard = S.datasets(args)
    dl_va, dl_hard = S.loader(evalset, va, args, False), S.loader(hardset, hard, args, False)
    dl_top = S.loader(evalset, args.va_topup, args, False) if args.va_topup else None
    sets = [("", dl_va, evalset, va), ("hard_", dl_hard, hardset, hard)] + \
        ([("top_", dl_top, evalset, args.va_topup)] if dl_top is not None else [])

    def save(name, ev, step=0):
        out = Path(f"runs/reeval-{name}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "log.jsonl").write_text(json.dumps({"eval": {"step": step, **ev}}) + "\n")
        print(name, {k: round(ev[k]["ratio"], 3) for k in ("all", "highlight_blur16", "hard_coc_64_up",
                                                            "top_all", "top_coc_64_up") if k in ev}, flush=True)

    if args.classic:
        from underblur import classic_crop

        def batches(ds, idx):
            for i in idx:
                item = ds[i]
                b = S.collate([item])
                b["yh"] = classic_crop(item["meta"]["lens_dir"], item["meta"], args.crop, dev)[None]
                yield b
        ev = {}
        for prefix, _, ds, idx in sets:
            ev.update({prefix + k: v for k, v in S.evaluate(None, batches(ds, idx), dev, pred=lambda b: b["yh"]).items()})
        save(args.classic, ev)
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        model = build(a.get("arch", "unet"), a.get("width"), a.get("film", False)).to(dev)
        model.load_state_dict(ck["model"])
        ev = {}
        for prefix, dl, _, _ in sets:
            ev.update({prefix + k: v for k, v in S.evaluate(model, dl, dev).items()})
        save(name, ev, ck.get("step", 0))


if __name__ == "__main__":
    main()
