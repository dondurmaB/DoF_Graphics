import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import numpy as np
from PIL import Image
import procedural as G, web_assets as W, env_kit as E
from scene_kit import Assets, SceneBuilder

out = HERE.parent.parent / "output" / "dev" / "sourceprobe"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
bust = W.model("bench_vice_01")
(x0, y0, z0), (x1, y1, z1) = bust.bounds
print(f"bust: {bust.triangles:,} tris, bounds {x1-x0:.3f} x {y1-y0:.3f} x {z1-z0:.3f}, parts {[p.name for p in bust.parts]}, {time.time()-t0:.0f}s")
b = SceneBuilder(Assets(out / "assets"))
b.material("floor", W.surface("acg:Tiles107"))
b.box_mesh("floor", (0, -0.05, 0), (8, 0.1, 8), "floor")
W.add(b, bust, W.place(bust, at=(0, 0, 0), height=0.4))
W.add(b, bust, W.place(bust, at=(0.9, 0, -0.6), yaw=40, height=0.4))
print("instances:", sum(1 for v in b.d.values() if isinstance(v, dict) and v.get("type") == "instance"))
b.d["sky"] = E.sky("clear", sun_direction=E.default_sun_direction(35, 40))
d = dict(b.d)
d["integrator"] = {"type": "path", "max_depth": 8}
d["sensor"] = {"type": "perspective", "fov": 40,
               "to_world": mi.ScalarTransform4f().look_at(origin=[0.0, 0.8, 2.0], target=[0, 0.35, 0], up=[0, 1, 0]),
               "film": {"type": "hdrfilm", "width": 800, "height": 600}}
img = np.array(mi.render(mi.load_dict(d), spp=128))[..., :3]
x = np.clip(img * 0.8 * (2.51 * img * 0.8 + 0.03) / (img * 0.8 * (2.43 * img * 0.8 + 0.59) + 0.14), 0, 1)
Image.fromarray((np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)).save(out / "sources.png")
print("rendered")
