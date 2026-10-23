import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import numpy as np
from PIL import Image
import procedural as G, web_assets as W, env_kit as E
from scene_kit import Assets, SceneBuilder

out = HERE.parent.parent / "output" / "dev" / "texprobe"
out.mkdir(parents=True, exist_ok=True)
b = SceneBuilder(Assets(out / "assets"))
b.material("floor", W.surface("concrete_floor_02"))
b.material("tinted", W.surface("concrete_floor_02", tint=(0.8, 0.6, 0.45)))
b.box_mesh("floor", (0, -0.05, 0), (12, 0.1, 12), "floor")
b.box_mesh("block", (0.8, 0.5, -1.0), (1.0, 1.0, 1.0), "tinted")
m = W.model("bench_vice_01")
W.add(b, m, W.place(m, at=(-0.4, 0.0, 0.2), yaw=30))
b.d["sky"] = E.sky("clear", sun_direction=E.default_sun_direction(30, 50))
d = dict(b.d)
d["integrator"] = {"type": "path", "max_depth": 8}
d["sensor"] = {"type": "perspective", "fov": 50,
               "to_world": mi.ScalarTransform4f().look_at(origin=[0.3, 1.1, 2.4], target=[0, 0.2, -0.5], up=[0, 1, 0]),
               "film": {"type": "hdrfilm", "width": 960, "height": 540}}
img = np.array(mi.render(mi.load_dict(d), spp=256))[..., :3]
x = np.clip(img * 0.8 * (2.51 * img * 0.8 + 0.03) / (img * 0.8 * (2.43 * img * 0.8 + 0.59) + 0.14), 0, 1)
Image.fromarray((np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)).save(out / "tex.png")
print("ok, median", float(np.median(img.mean(-1))))
