import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import web_assets as W
import render as R
from scene_kit import Assets, SceneBuilder

m = W.model("bench_vice_01")
for n in (1, 3):
    b = SceneBuilder(Assets(HERE / "generated" / "_probe"))
    for k in range(n):
        W.add(b, m, W.place(m, at=(k, 0, 0)))
    d = dict(b.d)
    d["e"] = {"type": "constant"}
    sc = mi.load_dict(d)
    shapes = sc.shapes()
    tris = sum(int(s.face_count()) for s in shapes if hasattr(s, "face_count"))
    print(f"{n} instance(s) of a {m.triangles}-tri model: {len(shapes)} shapes, face_count sum {tris}, "
          f"unique_triangles {R.unique_triangles(sc, d)}")
