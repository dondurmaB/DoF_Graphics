"""Throwaway: tone-map sampler outputs into sharp|dof PNG pairs (run on the cluster)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mitsuba as mi
mi.set_variant("scalar_rgb")
import numpy as np
from PIL import Image
import render as R

root = Path(sys.argv[1])
for pose in sorted(root.rglob("pose.json")):
    d = pose.parent
    sharp = np.array(mi.Bitmap(str(d / "sharp.exr")))[..., :3]
    for lens in sorted(d.glob("lens_*")):
        dof = np.array(mi.Bitmap(str(lens / "dof.exr")))[..., :3]
        img = np.concatenate([R.tonemap(sharp, 0.0), R.tonemap(dof, 0.0)], 1)
        Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(lens / "sharp_dof.png")
        print("wrote", lens / "sharp_dof.png")
