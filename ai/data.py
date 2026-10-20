"""Loader for the DoF dataset, implementing ai/DATA_PROTOCOL.md.

One example is `x` [4, H, W] (sharp RGB in model space + signed CoC / COC_NORM), `y` [3, H, W] (DoF RGB in
model space), `mask` [1, H, W] (coc_valid) and `meta`. Model space is log(1 + radiance * auto_exposure_scale);
`from_model` inverts it. The numpy core has no framework dependency; `DofDataset` wraps it for PyTorch.

    from ai.data import find_samples, load_example
    samples = find_samples("dataset/mitsuba", split="train")
    ex = load_example(samples[0], crop=512, rng=np.random.default_rng(0))

    python ai/data.py dataset/mitsuba --split train      # check every sample, print a summary
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "renderer" / "mitsuba" / "scenes" / "splits.json"

COC_NORM = 16.0            # px; channel 3 of x is CoC / COC_NORM
CROP = 512
HFLIP_PROB = 0.5
ROT90_PROB = 0.1           # sideways pictures break gravity and sun cues (SCENE_CONTRACT.md section 7)
VFLIP_PROB = 0.1
MIN_DEPTH_M = 0.05
COC_P95_FRAC = (0.0005, 0.04)   # the sampler's target range for p95 |CoC| / image height


@dataclass
class Sample:
    lens_dir: Path            # .../p<index>_<hash>/lens_<hash>
    scene: str
    env: str
    entry: dict = field(default_factory=dict)   # its manifest.jsonl line

    @property
    def pose_dir(self) -> Path:
        return self.lens_dir.parent


def find_samples(root, split: str | None = "train", scenes=None) -> list[Sample]:
    """Samples under root/<scene>/s<seed>/<env>/ listed in manifest.jsonl whose pose finished (.done)."""
    root = Path(root)
    allowed = set(scenes) if scenes else None
    if split is not None:
        in_split = set(json.loads(SPLITS.read_text())[split])
        allowed = in_split if allowed is None else allowed & in_split
    out = []
    for manifest in sorted(root.glob("*/s*/*/manifest.jsonl")):
        env_dir = manifest.parent
        scene, env = env_dir.parent.parent.name, env_dir.name
        if allowed is not None and scene not in allowed:
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            lens_dir = env_dir / entry["path"]
            if (lens_dir.parent / ".done").exists() and (lens_dir / "dof.exr").exists():
                out.append(Sample(lens_dir, scene, env, entry))
    return out


def read_exr(path) -> np.ndarray:
    """Linear RGB float32 (H, W, 3). Uses the OpenEXR package if installed, else Mitsuba's reader."""
    try:
        import OpenEXR

        with OpenEXR.File(str(path)) as f:
            ch = f.channels()
            if "RGB" in ch:
                return np.asarray(ch["RGB"].pixels, np.float32)
            return np.stack([np.asarray(ch[c].pixels, np.float32) for c in "RGB"], -1)
    except ImportError:
        import mitsuba as mi

        if mi.variant() is None:
            mi.set_variant("scalar_rgb")
        return np.asarray(mi.Bitmap(str(path)), np.float32)[..., :3]


def to_model(radiance: np.ndarray, exposure_scale: float) -> np.ndarray:
    """Metered linear radiance, log-compressed: exactly invertible by from_model, no clipping."""
    return np.log1p(np.maximum(radiance, 0.0) * exposure_scale).astype(np.float32)


def from_model(t: np.ndarray, exposure_scale: float = 1.0) -> np.ndarray:
    """Model space back to linear radiance (exposure_scale=1 keeps it metered)."""
    return (np.expm1(t) / exposure_scale).astype(np.float32)


def check(sharp, dof, depth, coc, height: int) -> list[str]:
    """Data problems that make a sample unusable (DATA_PROTOCOL.md section 6); empty when fine."""
    problems = []
    for name, a in (("sharp", sharp), ("dof", dof), ("coc", coc)):
        if not np.all(np.isfinite(a)):
            problems.append(f"{name} has NaN or inf")
    if np.any(np.isnan(depth)):
        problems.append("depth has NaN")
    finite = depth[np.isfinite(depth)]
    if finite.size and finite.min() < MIN_DEPTH_M:
        problems.append(f"depth {finite.min():.3f} m < {MIN_DEPTH_M} m")
    p95 = float(np.percentile(np.abs(coc), 95)) / height
    lo, hi = COC_P95_FRAC
    if not lo * 0.5 <= p95 <= hi * 2:
        problems.append(f"CoC p95 {p95:.4f} of image height is far outside [{lo}, {hi}]")
    m_s, m_d = float(sharp.mean()), float(dof.mean())
    if abs(m_d - m_s) > 0.05 * max(m_s, 1e-6):
        problems.append(f"mean(dof) {m_d:.4g} differs from mean(sharp) {m_s:.4g} by more than 5%")
    return problems


def load_example(sample: Sample, crop: int | None = CROP, rng: np.random.Generator | None = None,
                 augment: bool = True, validate: bool = False) -> dict:
    """x [4, H, W], y [3, H, W], mask [1, H, W] float32 and meta. crop=None keeps the full frame."""
    rng = rng or np.random.default_rng()
    meta = json.loads((sample.lens_dir / "metadata.json").read_text())
    scale = float(sample.entry.get("auto_exposure_scale")
                  or json.loads((sample.pose_dir / "pose.json").read_text())["auto_exposure_scale"])
    sharp = read_exr(sample.pose_dir / "sharp.exr")
    dof = read_exr(sample.lens_dir / "dof.exr")
    coc = np.load(sample.lens_dir / "coc.npy").astype(np.float32)
    mask_path = sample.pose_dir / "coc_valid.npy"
    mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else np.ones(coc.shape, np.float32)
    h, w = coc.shape
    if validate:
        problems = check(sharp, dof, np.load(sample.pose_dir / "depth.npy"), coc, h)
        if problems:
            raise ValueError(f"{sample.lens_dir}: " + "; ".join(problems))

    # Order fixed by the protocol: colour transform, crop, flips / rot90, CoC normalisation.
    planes = np.concatenate([to_model(sharp, scale), coc[..., None], mask[..., None], to_model(dof, scale)], -1)
    if crop is not None:
        if h < crop or w < crop:
            raise ValueError(f"{sample.lens_dir}: {w}x{h} is smaller than the {crop} crop")
        y0, x0 = int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))
        planes = planes[y0:y0 + crop, x0:x0 + crop]
    ops = []
    if augment:
        if rng.random() < HFLIP_PROB:
            planes, ops = planes[:, ::-1], ops + ["hflip"]
        if rng.random() < VFLIP_PROB:
            planes, ops = planes[::-1], ops + ["vflip"]
        if planes.shape[0] == planes.shape[1] and rng.random() < ROT90_PROB:
            k = int(rng.choice([1, 3]))
            planes, ops = np.rot90(planes, k), ops + [f"rot90x{k}"]
    planes = np.ascontiguousarray(planes.transpose(2, 0, 1))      # HWC -> CHW
    x = planes[0:4].copy()
    x[3] /= COC_NORM
    cam = meta["camera"]
    info = {"sample_id": meta["sample_id"], "scene": sample.scene, "env": sample.env,
            "focal_mm": cam["focal_length_mm"], "f_number": cam["f_number"], "focus_m": cam["focus_distance_m"],
            "coc_p95_frac": sample.entry.get("coc_p95_frac"), "stratum": sample.entry.get("stratum"),
            "auto_exposure_scale": scale, "has_coc_valid": mask_path.exists(), "augment": ops,
            "lens_dir": str(sample.lens_dir)}
    return {"x": x, "y": planes[5:8].copy(), "mask": planes[4:5].copy(), "meta": info}


def DofDataset(root, split="train", crop=CROP, augment=True, seed=0, scenes=None):
    """PyTorch Dataset over find_samples(); each worker gets its own random stream."""
    import torch
    from torch.utils.data import Dataset

    class _Dof(Dataset):
        def __init__(self):
            self.samples = find_samples(root, split, scenes)
            self.seed = seed

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            info = torch.utils.data.get_worker_info()
            rng = np.random.default_rng([self.seed, i, info.id if info else 0, torch.initial_seed() % 2 ** 32])
            ex = load_example(self.samples[i], crop, rng, augment)
            return {"x": torch.from_numpy(ex["x"]), "y": torch.from_numpy(ex["y"]),
                    "mask": torch.from_numpy(ex["mask"]), "meta": ex["meta"]}

    return _Dof()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check every sample of a split and summarise it.")
    ap.add_argument("root", nargs="?", default=str(REPO / "dataset" / "mitsuba"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--crop", type=int, default=CROP)
    args = ap.parse_args(argv)
    samples = find_samples(args.root, args.split)
    if not samples:
        print(f"no finished samples under {args.root} for split {args.split}")
        return 1
    rng = np.random.default_rng(0)
    bad, per_scene, stats = 0, {}, {"x_rgb": [], "coc": [], "y": [], "mask": []}
    for s in samples:
        try:
            ex = load_example(s, args.crop, rng, augment=True, validate=True)
        except ValueError as e:
            bad += 1
            print("BAD", e)
            continue
        assert ex["x"].shape == (4, args.crop, args.crop) and ex["y"].shape == (3, args.crop, args.crop)
        assert ex["mask"].shape == (1, args.crop, args.crop)
        per_scene[s.scene] = per_scene.get(s.scene, 0) + 1
        stats["x_rgb"].append(ex["x"][:3].mean())
        stats["coc"].append(np.percentile(np.abs(ex["x"][3]), 95))
        stats["y"].append(ex["y"].mean())
        stats["mask"].append(ex["mask"].mean())
    print(f"{len(samples)} samples, {bad} bad | per scene {per_scene}")
    for k, v in stats.items():
        if v:
            print(f"  {k}: mean {np.mean(v):.3f}  min {np.min(v):.3f}  max {np.max(v):.3f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
