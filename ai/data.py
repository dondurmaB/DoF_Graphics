"""Loader for the DoF dataset, implementing ai/DATA_PROTOCOL.md.

One example is `x` [10, H, W] (sharp RGB in model space, inverse depth, 6 camera settings as constant planes),
`y` [3, H, W] (DoF RGB in model space), `settings` [6] and `meta`. Model space is
log(1 + radiance * auto_exposure_scale); `from_model` inverts it. The numpy core has no framework dependency;
`DofDataset` wraps it for PyTorch.

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

CROP = 512
HFLIP_PROB = 0.5
ROT90_PROB = 0.1           # sideways pictures break gravity and sun cues (SCENE_CONTRACT.md section 7)
VFLIP_PROB = 0.1
MIN_DEPTH_M = 0.05
SETTINGS = ("log_focal_50mm", "log_fnumber_4", "log_aperture_10mm", "inv_focus_m", "sensor_24mm", "frame_1080px")


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
    """Samples under root/<scene>/s<seed>/<env>/ listed in manifest*.jsonl whose pose finished (.done)."""
    root = Path(root)
    allowed = set(scenes) if scenes else None
    if split is not None:
        in_split = set(json.loads(SPLITS.read_text())[split])
        allowed = in_split if allowed is None else allowed & in_split
    out = []
    seen = set()
    for manifest in sorted(root.glob("*/s*/*/manifest*.jsonl")):
        env_dir = manifest.parent
        scene, env = env_dir.parent.parent.name, env_dir.name
        if allowed is not None and scene not in allowed:
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            lens_dir = env_dir / entry["path"]
            if lens_dir in seen:          # a chunk rerun after a crash can list a sample twice
                continue
            seen.add(lens_dir)
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


def check(sharp, dof, depth) -> list[str]:
    """Data problems that make a sample unusable (DATA_PROTOCOL.md section 6); empty when fine."""
    problems = []
    for name, a in (("sharp", sharp), ("dof", dof)):
        if not np.all(np.isfinite(a)):
            problems.append(f"{name} has NaN or inf")
    if np.any(np.isnan(depth)):
        problems.append("depth has NaN")
    finite = depth[np.isfinite(depth)]
    if finite.size and finite.min() < MIN_DEPTH_M:
        problems.append(f"depth {finite.min():.3f} m < {MIN_DEPTH_M} m")
    m_s, m_d = float(sharp.mean()), float(dof.mean())
    if abs(m_d - m_s) > 0.05 * max(m_s, 1e-6):
        problems.append(f"mean(dof) {m_d:.4g} differs from mean(sharp) {m_s:.4g} by more than 5%")
    return problems


def settings_vector(meta: dict) -> np.ndarray:
    """The 6 normalised camera settings of DATA_PROTOCOL.md section 1, in SETTINGS order."""
    cam = meta["camera"]
    f, n, s = cam["focal_length_mm"], cam["f_number"], cam["focus_distance_m"]
    return np.array([np.log(f / 50.0), np.log(n / 4.0), np.log(f / n / 10.0), 1.0 / s,
                     cam["sensor_height_mm"] / 24.0, meta["resolution"][1] / 1080.0], np.float32)


def load_example(sample: Sample, crop: int | None = CROP, rng: np.random.Generator | None = None,
                 augment: bool = True, validate: bool = False, coc_margin: int | None = None,
                 crop_at: tuple[int, int] | None = None) -> dict:
    """x [10, H, W], y [3, H, W], settings [6] float32 and meta. crop=None keeps the full frame.

    coc_margin=M also returns `coc` [H + 2M, W + 2M]: the lens's signed CoC diameter in px around the same
    window (0 outside the frame), flipped and turned with the example. Loss weights and metrics only
    (DATA_PROTOCOL.md v1.1), never an input. It draws nothing from rng, so crops are the same with or without it.
    crop_at=(y0, x0) puts the crop there instead of drawing it (fixed evaluation crops).
    """
    rng = rng or np.random.default_rng()
    meta = json.loads((sample.lens_dir / "metadata.json").read_text())
    scale = float(sample.entry.get("auto_exposure_scale")
                  or json.loads((sample.pose_dir / "pose.json").read_text())["auto_exposure_scale"])
    sharp = read_exr(sample.pose_dir / "sharp.exr")
    dof = read_exr(sample.lens_dir / "dof.exr")
    depth = np.load(sample.pose_dir / "depth.npy").astype(np.float32)
    h, w = depth.shape
    if validate:
        problems = check(sharp, dof, depth)
        if problems:
            raise ValueError(f"{sample.lens_dir}: " + "; ".join(problems))

    # Order fixed by the protocol: colour transform and inverse depth, crop, flips / rot90, settings planes.
    inv_depth = np.where(np.isfinite(depth), 1.0 / np.maximum(depth, MIN_DEPTH_M), 0.0).astype(np.float32)
    planes = np.concatenate([to_model(sharp, scale), inv_depth[..., None], to_model(dof, scale)], -1)
    coc = None
    if coc_margin is not None:
        coc = np.pad(np.load(sample.lens_dir / "coc.npy").astype(np.float32), coc_margin)
    if crop is not None:
        if h < crop or w < crop:
            raise ValueError(f"{sample.lens_dir}: {w}x{h} is smaller than the {crop} crop")
        y0, x0 = crop_at if crop_at is not None else (int(rng.integers(0, h - crop + 1)),
                                                       int(rng.integers(0, w - crop + 1)))
        planes = planes[y0:y0 + crop, x0:x0 + crop]
        if coc is not None:
            coc = coc[y0:y0 + crop + 2 * coc_margin, x0:x0 + crop + 2 * coc_margin]
    ops = []
    if augment:
        if rng.random() < HFLIP_PROB:
            planes, ops = planes[:, ::-1], ops + ["hflip"]
            coc = None if coc is None else coc[:, ::-1]
        if rng.random() < VFLIP_PROB:
            planes, ops = planes[::-1], ops + ["vflip"]
            coc = None if coc is None else coc[::-1]
        if planes.shape[0] == planes.shape[1] and rng.random() < ROT90_PROB:
            k = int(rng.choice([1, 3]))
            planes, ops = np.rot90(planes, k), ops + [f"rot90x{k}"]
            coc = None if coc is None else np.rot90(coc, k)
    planes = np.ascontiguousarray(planes.transpose(2, 0, 1))      # HWC -> CHW
    settings = settings_vector(meta)
    x = np.concatenate([planes[0:4], np.broadcast_to(settings[:, None, None], (6, *planes.shape[1:]))])
    cam = meta["camera"]
    info = {"sample_id": meta["sample_id"], "scene": sample.scene, "env": sample.env,
            "focal_mm": cam["focal_length_mm"], "f_number": cam["f_number"],
            "aperture_mm": cam["focal_length_mm"] / cam["f_number"], "focus_m": cam["focus_distance_m"],
            "sensor_mm": cam["sensor_height_mm"], "frame_px": meta["resolution"][1],
            "auto_exposure_scale": scale, "augment": ops, "lens_dir": str(sample.lens_dir)}
    out = {"x": x.astype(np.float32), "y": planes[4:7].copy(), "settings": settings, "meta": info}
    if coc is not None:
        out["coc"] = np.ascontiguousarray(coc)
    return out


def hard_crop_at(coc: np.ndarray, crop: int, min_coc: float, step: int = 8):
    """(y0, x0, n) of the crop window holding the most pixels with |CoC| >= min_coc (n counted on a step grid)."""
    m = (np.abs(coc[::step, ::step]) >= min_coc).astype(np.int64)
    cs = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = crop // step
    win = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
    iy, ix = np.unravel_index(int(win.argmax()), win.shape)
    h, w = coc.shape
    return min(iy * step, h - crop), min(ix * step, w - crop), int(win.max()) * step * step


def DofDataset(root, split="train", crop=CROP, augment=True, seed=0, scenes=None, coc_margin=None,
               deterministic=False, hard_coc=None):
    """PyTorch Dataset over find_samples(); each worker gets its own random stream.

    deterministic=True ties each sample's crop to (seed, index) only, so every run and epoch sees the same
    pixels (held-out evaluation). coc_margin adds the `coc` field (see load_example). hard_coc=T centres every
    crop on the window with the most pixels of |CoC| >= T px (the blurs a random crop rarely contains)."""
    import torch
    from torch.utils.data import Dataset

    class _Dof(Dataset):
        def __init__(self):
            self.samples = find_samples(root, split, scenes)
            self.seed = seed

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            if deterministic:
                rng = np.random.default_rng([self.seed, i])
            else:
                info = torch.utils.data.get_worker_info()
                rng = np.random.default_rng([self.seed, i, info.id if info else 0, torch.initial_seed() % 2 ** 32])
            at = None
            if hard_coc is not None:
                at = hard_crop_at(np.load(self.samples[i].lens_dir / "coc.npy"), crop, hard_coc)[:2]
            ex = load_example(self.samples[i], crop, rng, augment, coc_margin=coc_margin, crop_at=at)
            out = {"x": torch.from_numpy(ex["x"]), "y": torch.from_numpy(ex["y"]),
                   "settings": torch.from_numpy(ex["settings"]), "meta": ex["meta"]}
            if "coc" in ex:
                out["coc"] = torch.from_numpy(ex["coc"])
            return out

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
    bad, per_scene, stats = 0, {}, {"x_rgb": [], "inv_depth": [], "y": []}
    settings = []
    for s in samples:
        try:
            ex = load_example(s, args.crop, rng, augment=True, validate=True)
        except ValueError as e:
            bad += 1
            print("BAD", e)
            continue
        assert ex["x"].shape == (10, args.crop, args.crop) and ex["y"].shape == (3, args.crop, args.crop)
        assert np.allclose(ex["x"][4:].mean((1, 2)), ex["settings"])
        per_scene[s.scene] = per_scene.get(s.scene, 0) + 1
        stats["x_rgb"].append(ex["x"][:3].mean())
        stats["inv_depth"].append(ex["x"][3].max())
        stats["y"].append(ex["y"].mean())
        settings.append(ex["settings"])
    print(f"{len(samples)} samples, {bad} bad | per scene {per_scene}")
    for k, v in stats.items():
        if v:
            print(f"  {k}: mean {np.mean(v):.3f}  min {np.min(v):.3f}  max {np.max(v):.3f}")
    if settings:
        a = np.stack(settings)
        for i, name in enumerate(SETTINGS):
            print(f"  {name}: min {a[:, i].min():.3f}  max {a[:, i].max():.3f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
