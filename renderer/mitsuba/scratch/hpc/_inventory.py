"""List every object in every scene: name, instances, triangles, material, procedural or scanned."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mitsuba as mi
mi.set_variant("scalar_rgb")
import scene_api as S

out = HERE.parent.parent / "output" / "dev" / "inventory"
out.mkdir(parents=True, exist_ok=True)


def ply_faces(path):
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("ascii", "ignore").strip()
            if line.startswith("element face"):
                return int(line.split()[-1])
            if line == "end_header":
                return 0
    return 0


for sid in S.list_scenes():
    scene = S.load_scene(sid)
    ctx = S.make_context(scene, HERE / "generated", None, "clear")
    bundle = S.build_checked(scene, ctx)
    groups = defaultdict(lambda: {"instances": 0, "triangles": 0, "materials": set(), "source": "procedural"})
    for key, v in bundle.scene.items():
        if not isinstance(v, dict) or v.get("type") not in ("ply", "obj", "cube", "rectangle", "sphere", "disk", "cylinder"):
            continue
        mat = v.get("bsdf", {}).get("id", "") if isinstance(v.get("bsdf"), dict) else ""
        emitter = "emitter" in v
        if v["type"] in ("ply", "obj"):
            fn = Path(v["filename"])
            if "web_assets" in str(fn):
                name, src = f"web:{fn.parts[fn.parts.index('polyhaven') + 1]}", "scanned"
            else:
                name, src = re.sub(r"_?\d+$", "", fn.stem), "procedural"
            tris = ply_faces(fn)
        else:
            name, src, tris = f"{v['type']}:{mat}", "procedural", {"cube": 12, "rectangle": 2}.get(v["type"], 0)
        if emitter:
            name += " [emitter]"
        g = groups[name]
        g["instances"] += 1
        g["triangles"] += tris
        g["materials"].add(mat)
        g["source"] = src
    rows = sorted(({"name": k, **{kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in g.items()}}
                   for k, g in groups.items()), key=lambda r: (r["source"], -r["triangles"]))
    (out / f"{sid}.json").write_text(json.dumps(rows, indent=1))
    n_proc = sum(r["instances"] for r in rows if r["source"] == "procedural")
    n_scan = sum(r["instances"] for r in rows if r["source"] == "scanned")
    print(f"{sid}: {len(rows)} object groups, {n_proc} procedural instances, {n_scan} scanned instances")
