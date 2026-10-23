"""Temporary probe (run on the cluster): variant layout of scanned plant sheets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web_assets as W
from scenes import _greenhouse_props as H
for aid in sys.argv[1:]:
    m = W.model(aid)
    meshes = {p.name: H.read_ply(p.ply) for p in m.parts}
    (x0, y0, z0), (x1, y1, z1) = m.bounds
    print(f"{aid}: {m.triangles:,} tris, {x1-x0:.2f} x {y1-y0:.2f} x {z1-z0:.2f} m, parts {[(p.name, p.bsdf['type']) for p in m.parts]}")
    for axis in (0, 2):
        rs = H.variant_ranges(list(meshes.values()), axis=axis)
        if len(rs) > 1:
            sizes = []
            for r in rs:
                v = H.cut_variant(meshes, r, axis)
                import numpy as np
                allp = np.vstack([mm.p for mm in v.values()]); tri = sum(len(mm.f) for mm in v.values())
                ext = allp.max(0) - allp.min(0)
                sizes.append(f"[{r[0]:.2f},{r[1]:.2f}] h={ext[1]:.2f} w={ext[0]:.2f} d={ext[2]:.2f} {tri}t")
            print(f"   axis {axis}: {len(rs)} variants: " + " | ".join(sizes))
