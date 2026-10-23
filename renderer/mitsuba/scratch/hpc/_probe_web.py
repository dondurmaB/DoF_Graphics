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

out = HERE.parent.parent / "output" / "dev" / "webprobe"
out.mkdir(parents=True, exist_ok=True)

# 1. uv orientation: quad with uv (0,0) at the bottom-left vertex; texture top half red, bottom half blue
tex = np.zeros((64, 64, 3), np.uint8); tex[:32] = (255, 0, 0); tex[32:] = (0, 0, 255)
Image.fromarray(tex).save(out / "uvtest.png")
q = G.Mesh(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float), np.tile([0, 0, 1.0], (4, 1)),
           np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), np.array([[0, 1, 2], [0, 2, 3]]))
q.write_ply(out / "uvquad.ply")
sc = mi.load_dict({"type": "scene", "integrator": {"type": "path"}, "l": {"type": "constant"},
    "q": {"type": "ply", "filename": str(out / "uvquad.ply"),
          "bsdf": {"type": "diffuse", "reflectance": {"type": "bitmap", "filename": str(out / "uvtest.png")}}},
    "sensor": {"type": "perspective", "fov": 60,
               "to_world": mi.ScalarTransform4f().look_at(origin=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0]),
               "film": {"type": "hdrfilm", "width": 32, "height": 32}}})
img = np.array(mi.render(sc, spp=4))
top = img[8, 16]
print("uv test: image top pixel", top.round(2), "->",
      "v=1 maps to image TOP: v up, glTF needs FLIP_V=True" if top[0] > top[2] else "v=1 maps to image BOTTOM: FLIP_V=False")

# 2. fetch + convert
models = {}
for aid in ["bench_vice_01", "potted_plant_01", "fern_02", "outdoor_table_chair_set_01", "tree_small_02"]:
    t0 = time.time(); m = W.model(aid); models[aid] = m
    (x0, y0, z0), (x1, y1, z1) = m.bounds
    print(f"{aid}: {m.triangles:,} tris, {x1-x0:.2f} x {y1-y0:.2f} x {z1-z0:.2f} m, {time.time()-t0:.1f}s, "
          f"parts {[p.name for p in m.parts]}")

# 3. render them on a ground plane under the env_kit clear sky
b = SceneBuilder(Assets(out / "assets"))
b.material("ground", E.ground("clear"))
b.rect(G.compose(G.rotate((1, 0, 0), -90), G.scale(20)), "ground")
for aid, at in [("tree_small_02", (-2.6, 0, -3.5)), ("potted_plant_01", (-0.9, 0, -0.6)), ("fern_02", (0.3, 0, -1.6)),
                ("outdoor_table_chair_set_01", (1.6, 0, -1.2)), ("bench_vice_01", (0.0, 0.0, 0.6))]:
    W.add(b, models[aid], W.place(models[aid], at=at, yaw=20))
b.d["sky"] = E.sky("clear", sun_direction=E.default_sun_direction(35, 40))
d = dict(b.d)
d["integrator"] = {"type": "path", "max_depth": 8}
for name, origin, target in [("wide", [0.5, 1.4, 3.2], [-0.5, 1.0, -1.5]), ("vice", [0.35, 0.55, 1.25], [0.0, 0.2, 0.6])]:
    d["sensor"] = {"type": "perspective", "fov": 45,
                   "to_world": mi.ScalarTransform4f().look_at(origin=origin, target=target, up=[0, 1, 0]),
                   "film": {"type": "hdrfilm", "width": 960, "height": 540}}
    t0 = time.time(); img = np.array(mi.render(mi.load_dict(d), spp=256))[..., :3]
    x = np.clip(img * 0.8 * (2.51 * img * 0.8 + 0.03) / (img * 0.8 * (2.43 * img * 0.8 + 0.59) + 0.14), 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    Image.fromarray((srgb * 255).astype(np.uint8)).save(out / f"{name}.png")
    print(f"rendered {name} in {time.time()-t0:.1f}s, NaN {np.isnan(img).any()}")
