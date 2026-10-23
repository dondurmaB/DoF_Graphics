"""Temporary: per-part bounds and the top-section centroid of web models (cluster only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import mitsuba as mi
mi.set_variant("scalar_rgb")
import web_assets as W
for aid in sys.argv[1:]:
    m = W.model(aid)
    allp = []
    for p in m.parts:
        sh = mi.load_dict({"type": "ply", "filename": p.ply})
        v = np.array(mi.traverse(sh)["vertex_positions"]).reshape(-1, 3)
        allp.append(v)
        print(f"  {aid}:{p.name:34s} lo {np.round(v.min(0), 3).tolist()} hi {np.round(v.max(0), 3).tolist()}")
    v = np.vstack(allp)
    lo, hi = v.min(0), v.max(0)
    top = v[v[:, 1] > lo[1] + 0.7 * (hi[1] - lo[1])]
    print(f"{aid}: centre {np.round((lo + hi) / 2, 3).tolist()} top-centroid {np.round(top.mean(0), 3).tolist()}")
