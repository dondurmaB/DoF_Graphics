#!/usr/bin/env python3
"""CC0 models and textures from the web: pinned in a manifest, fetched on demand, converted for Mitsuba.

Sources (all CC0): Poly Haven models and textures (plain ids), ambientCG materials (`acg:<AssetId>`),
Smithsonian Open Access 3D OBJ zips (`si:<name>`, pinned with --url).

  python web_assets.py search tree shrub         # list matching Poly Haven models (light, run anywhere)
  python web_assets.py search --textures concrete floor   # ... or textures
  python web_assets.py pin bench_vice_01 --res 1k  # record file URLs and md5 in scenes/assets/web_manifest.json
  python web_assets.py pin acg:Tiles107          # ambientCG material (downloads the zip once to record its sha256)
  python web_assets.py pin si:amphora_x --url https://3d-api.si.edu/.../x-obj.zip --name "Amphora"   # Smithsonian
  python web_assets.py stage si:amphora_x        # download locally: the cluster cannot reach 3d-api.si.edu, and
                                                 # remote_run.sh syncs web_assets/ to it (conversion runs there)
  python web_assets.py info bench_vice_01        # fetch + convert, print parts, bounds, triangles (run on the cluster)

In a scene build:

  import web_assets as W
  m = W.model("bench_vice_01")                   # pinned assets only; downloads once, glTF -> PLY + Mitsuba BSDFs
  W.add(b, m, W.place(m, at=(x, 0.9, z), yaw=30))  # registers its materials and shapes on a SceneBuilder
  m.bounds, m.triangles, [p.name for p in m.parts]
  b.material("floor", W.surface("concrete_floor_02"))   # pinned texture -> principled + normal map, real-world scale

Every Poly Haven asset is CC0 (https://polyhaven.com/license). Downloads are md5-checked against the
manifest, which is committed; the downloaded and converted files live in web_assets/ (gitignored, or
$DOF_WEB_ASSETS) on whatever machine builds the scene. Pin locally, fetch on the cluster.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "scenes" / "assets" / "web_manifest.json"
CACHE = Path(os.environ.get("DOF_WEB_ASSETS", HERE / "web_assets"))
API = "https://api.polyhaven.com"
UA = {"User-Agent": "DoF-Graphics-dataset/0.1"}   # the API rejects Python's default agent
CONVERTER_VERSION = 4                              # bump when conversion changes; old conversions are ignored
FLIP_V = False                                     # Mitsuba bitmaps, like glTF, put v = 0 at the image top (measured)
# Leaf-like parts get thin translucent shading; words that also name pots/vases ("plant", "flower") are left out.
FOLIAGE_WORDS = ("leaf", "leaves", "fern", "foliage", "grass", "petal", "needle", "moss")


# --------------------------------------------------------------------------- manifest and download

def _get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return json.loads(r.read())


def _read_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"source": "Poly Haven, https://polyhaven.com", "license": "CC0 1.0 (https://polyhaven.com/license)",
            "assets": {}}


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def search(words: list[str], kind: str = "models") -> None:
    assets = _get(f"{API}/assets?t={kind}")
    for aid, a in sorted(assets.items()):
        text = " ".join([aid] + a.get("tags", []) + a.get("categories", [])).lower()
        if all(w.lower() in text for w in words):
            dims = [round(x / 1000, 2) for x in a.get("dimensions", [])]
            print(f"{aid:34s} {dims} m  {a.get('polycount', '?')} polys  {', '.join(a.get('categories', [])[:4])}")


def _source(aid: str) -> str:
    return aid.split(":", 1)[0] if ":" in aid else "polyhaven"


def _hash_url(url: str) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _store(aid: str, res: str, entry: dict) -> None:
    with _locked(MANIFEST.parent / ".web_manifest.lock"):     # several agents may pin at once
        m = _read_manifest()
        m["assets"].setdefault(aid, {})[res] = entry
        _write_atomic(MANIFEST, (json.dumps(m, indent=1, sort_keys=True) + "\n").encode())
    mb = sum(f["size"] for f in entry["files"].values()) / 1e6
    print(f"pinned {entry['kind']} {aid} @ {res}: {len(entry['files'])} files, {mb:.1f} MB, "
          f"{entry.get('polycount') or '-'} polys, {entry.get('dimensions_m')} m")


def pin(aid: str, res: str = "1k", url: str | None = None, name: str | None = None) -> None:
    src = _source(aid)
    if src == "acg":
        a = _get(f"https://ambientcg.com/api/v2/full_json?id={aid[4:]}&include=downloadData,dimensionsData")["foundAssets"][0]
        dl = next(d for d in a["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
                  if d["attribute"] == f"{res.upper()}-JPG")
        sha, size = _hash_url(dl["downloadLink"])
        dims = [a.get("dimensionX"), a.get("dimensionY")]
        entry = {"name": a.get("displayName", aid), "page": f"https://ambientcg.com/view?id={aid[4:]}", "kind": "texture",
                 "source": "ambientcg", "license": "CC0", "dimensions_m": [d / 100 for d in dims if d] or [1.0],
                 "files": {dl["fileName"]: {"url": dl["downloadLink"], "size": size, "sha256": sha, "unzip": True}}}
        return _store(aid, res, entry)
    if src == "si":
        if not url:
            raise ValueError("Smithsonian assets need --url <full-resolution OBJ zip from 3d-api.si.edu>")
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        size = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
        entry = {"name": name or aid, "page": url, "kind": "obj_zip", "source": "smithsonian",
                 "license": "CC0 (Smithsonian Open Access)",
                 "files": {Path(url).name: {"url": url, "size": size, "unzip": True}}}
        return _store(aid, res, entry)
    api = _get(f"{API}/files/{aid}")
    info = _get(f"{API}/info/{aid}")
    entry = {"name": info.get("name", aid), "page": f"https://polyhaven.com/a/{aid}",
             "dimensions_m": [round(x / 1000, 3) for x in info.get("dimensions", [])],
             "polycount": info.get("polycount"), "authors": sorted(info.get("authors", {}))}
    if info.get("type") == 1:                                       # texture: diffuse, roughness, normal (+ metal)
        roles = {"Diffuse": "diff", "Rough": "rough", "nor_gl": "nor_gl", "Metal": "metal"}
        entry["kind"] = "texture"
        entry["files"] = {Path(api[k][res]["jpg"]["url"]).name: {**{f: api[k][res]["jpg"][f] for f in ("url", "md5", "size")},
                                                                 "role": role}
                          for k, role in roles.items() if k in api and res in api[k]}
    else:
        files = api["gltf"][res]["gltf"]
        entry["kind"] = "model"
        entry["files"] = {Path(files["url"]).name: {k: files[k] for k in ("url", "md5", "size")},
                          **{k: {f: v[f] for f in ("url", "md5", "size")} for k, v in files["include"].items()}}
    _store(aid, res, entry)


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _fetch(aid: str, res: str) -> tuple[Path, dict]:
    entry = _read_manifest()["assets"].get(aid, {}).get(res)
    if entry is None:
        raise KeyError(f"web asset {aid!r} @ {res} is not pinned: run `python renderer/mitsuba/web_assets.py pin "
                       f"{aid} --res {res}` locally and commit scenes/assets/web_manifest.json")
    root = _base(aid, res) / "src"
    for rel, f in entry["files"].items():
        dst = root / rel
        if dst.exists() and dst.stat().st_size == f["size"]:
            _unzip(dst, f)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):                                   # big scans: survive dropped connections
            md5, sha = hashlib.md5(), hashlib.sha256()
            fd, tmp = tempfile.mkstemp(dir=dst.parent)
            try:
                with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
                        urllib.request.Request(f["url"], headers=UA), timeout=600) as r:
                    while chunk := r.read(1 << 20):
                        md5.update(chunk)
                        sha.update(chunk)
                        out.write(chunk)
                break
            except OSError:
                os.unlink(tmp)
                if attempt == 3:
                    raise
                import time
                time.sleep(10 * (attempt + 1))
        bad = (("md5" in f and md5.hexdigest() != f["md5"]) or ("sha256" in f and sha.hexdigest() != f["sha256"])
               or (f.get("size") and os.path.getsize(tmp) != f["size"]))
        if bad:
            os.unlink(tmp)
            raise IOError(f"checksum or size mismatch for {f['url']}")
        os.replace(tmp, dst)
        _unzip(dst, f)
    return root, entry


def _base(aid: str, res: str) -> Path:
    return CACHE / _source(aid) / aid.split(":", 1)[-1] / res


def _unzip(path: Path, f: dict) -> None:
    import zipfile

    marker = path.with_suffix(".unzipped")
    if f.get("unzip") and not marker.exists():
        with zipfile.ZipFile(path) as z:
            z.extractall(path.parent / path.stem)
        marker.write_text("")


# --------------------------------------------------------------------------- glTF -> PLY + BSDF

_COMP = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _accessor(g, bufs, i):
    import numpy as np

    a = g["accessors"][i]
    if "sparse" in a or "bufferView" not in a:
        raise NotImplementedError("sparse or bufferless glTF accessors")
    bv = g["bufferViews"][a["bufferView"]]
    dt, n = np.dtype("<" + _COMP[a["componentType"]]), _NCOMP[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride") or dt.itemsize * n
    arr = np.ndarray((a["count"], n), dt, bufs[bv["buffer"]], off, (stride, dt.itemsize))
    arr = np.array(arr)
    if a.get("normalized"):
        arr = arr.astype(np.float64) / np.iinfo(dt).max
    return arr


def _node_matrix(n):
    import numpy as np

    if "matrix" in n:
        return np.asarray(n["matrix"], float).reshape(4, 4).T
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * np.asarray(n.get("scale", [1, 1, 1]), float)
    m[:3, 3] = n.get("translation", [0, 0, 0])
    return m


def _uv_transform(tex_info):
    import numpy as np

    t = (tex_info or {}).get("extensions", {}).get("KHR_texture_transform")
    if not t:
        return None
    (ox, oy), (sx, sy), r = t.get("offset", [0, 0]), t.get("scale", [1, 1]), t.get("rotation", 0.0)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c * sx, s * sy, ox], [-s * sx, c * sy, oy]])   # glTF spec: T * R * S


def _material(g, src: Path, out: Path, mi_idx, foliage_translucency: float):
    """Mitsuba BSDF dict (texture paths relative to `out`) for glTF material `mi_idx`."""
    import numpy as np
    from PIL import Image

    mat = g["materials"][mi_idx] if mi_idx is not None else {"name": "default"}
    name = mat.get("name") or f"material_{mi_idx}"
    pbr = mat.get("pbrMetallicRoughness", {})
    tdir = out / "textures"
    tdir.mkdir(parents=True, exist_ok=True)

    def image(tex_info) -> Path | None:
        if not tex_info:
            return None
        return src.parent / g["images"][g["textures"][tex_info["index"]]["source"]]["uri"]

    def gray(arr, tag) -> str:
        path = tdir / f"{name}_{mi_idx}_{tag}.png"
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").save(path)
        return str(path.relative_to(out))

    spec_base, alpha = None, None
    base_img = image(pbr.get("baseColorTexture"))
    factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
    if base_img is not None:
        im = Image.open(base_img)
        if im.mode in ("RGBA", "LA") and mat.get("alphaMode") in ("MASK", "BLEND"):
            alpha = gray(np.asarray(im.getchannel("A")), "alpha")
            rgb_path = tdir / f"{name}_{mi_idx}_base.png"
            im.convert("RGB").save(rgb_path)
            base_rel = str(rgb_path.relative_to(out))
        else:
            dst = tdir / base_img.name
            if not dst.exists():
                dst.write_bytes(base_img.read_bytes())
            base_rel = str(dst.relative_to(out))
        spec_base = {"type": "bitmap", "filename": base_rel}
    else:
        spec_base = {"type": "rgb", "value": [float(v) for v in factor[:3]]}

    rough, metal = float(pbr.get("roughnessFactor", 1.0)), float(pbr.get("metallicFactor", 1.0))
    mr_img = image(pbr.get("metallicRoughnessTexture"))
    if mr_img is not None:
        arr = np.asarray(Image.open(mr_img).convert("RGB"), float)
        spec_rough = {"type": "bitmap", "raw": True, "filename": gray(arr[..., 1] * rough, "rough")}
        spec_metal = ({"type": "bitmap", "raw": True, "filename": gray(arr[..., 2] * metal, "metal")}
                      if metal > 0 and arr[..., 2].max() > 0 else 0.0)
    else:
        spec_rough, spec_metal = rough, metal

    foliage = mat.get("alphaMode") in ("MASK", "BLEND") or any(w in name.lower() for w in FOLIAGE_WORDS)
    if foliage:
        bsdf = {"type": "principledthin", "base_color": spec_base, "roughness": spec_rough,
                "diff_trans": float(foliage_translucency)}
    else:
        bsdf = {"type": "principled", "base_color": spec_base, "roughness": spec_rough, "metallic": spec_metal}
    nrm = image(mat.get("normalTexture"))
    if nrm is not None:
        dst = tdir / nrm.name
        if not dst.exists():
            dst.write_bytes(nrm.read_bytes())
        bsdf = {"type": "normalmap", "normalmap": {"type": "bitmap", "raw": True, "filename": str(dst.relative_to(out))},
                "bsdf": bsdf}
    if not foliage and mat.get("doubleSided", False):
        bsdf = {"type": "twosided", "bsdf": bsdf}
    if alpha is not None:
        bsdf = {"type": "mask", "opacity": {"type": "bitmap", "raw": True, "filename": alpha}, "bsdf": bsdf}
    return name, bsdf, _uv_transform(pbr.get("baseColorTexture"))


def _convert(gltf: Path, out: Path, foliage_translucency: float) -> dict:
    """Flatten the glTF scene into one PLY per material (world transforms baked in)."""
    import numpy as np

    sys.path.insert(0, str(HERE))
    import procedural as G

    g = json.loads(gltf.read_text())
    bufs = [(gltf.parent / b["uri"]).read_bytes() for b in g["buffers"]]
    groups: dict = {}

    def visit(ni, parent):
        node = g["nodes"][ni]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in g["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                at = prim["attributes"]
                p = _accessor(g, bufs, at["POSITION"]).astype(np.float64)
                nrm = _accessor(g, bufs, at["NORMAL"]).astype(np.float64) if "NORMAL" in at else None
                uv = _accessor(g, bufs, at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else np.zeros((len(p), 2))
                f = (_accessor(g, bufs, prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim
                     else np.arange(len(p)).reshape(-1, 3))
                groups.setdefault(prim.get("material"), []).append((m, p, nrm, uv, f))
        for c in node.get("children", []):
            visit(c, m)

    for root in g["scenes"][g.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))

    parts, lo, hi, total = [], np.full(3, np.inf), np.full(3, -np.inf), 0
    for mi_idx, prims in groups.items():
        name, bsdf, uvt = _material(g, gltf, out, mi_idx, foliage_translucency)
        if any(p["name"] == name for p in parts):        # glTF allows duplicate material names
            name = f"{name}_{mi_idx}"
        mesh = G.Mesh()
        for m, p, nrm, uv, f in prims:
            pw = p @ m[:3, :3].T + m[:3, 3]
            if nrm is None:
                nrm = np.zeros_like(p)
                fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
                for k in range(3):
                    np.add.at(nrm, f[:, k], fn)
            nw = nrm @ np.linalg.inv(m[:3, :3])
            nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-12)
            if uvt is not None:
                uv = uv @ uvt[:, :2].T + uvt[:, 2]
            if FLIP_V:
                uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])
            if np.linalg.det(m[:3, :3]) < 0:
                f = f[:, ::-1]
            mesh += G.Mesh(pw, nw, uv, f)
        ply = out / f"{name}.ply"
        mesh.write_ply(ply)
        parts.append({"name": name, "ply": ply.name, "bsdf": bsdf, "triangles": int(len(mesh.f))})
        lo, hi, total = np.minimum(lo, mesh.p.min(0)), np.maximum(hi, mesh.p.max(0)), total + len(mesh.f)
    meta = {"converter_version": CONVERTER_VERSION, "parts": parts, "bounds": [lo.tolist(), hi.tolist()],
            "triangles": int(total)}
    _write_atomic(out / "model.json", json.dumps(meta, indent=1).encode())
    return meta


def _convert_obj(root: Path, out: Path) -> dict:
    """Smithsonian-style OBJ (+ MTL with map_Kd, optional normal map) -> one PLY per material."""
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(HERE))
    import procedural as G

    obj = max(root.rglob("*.obj"), key=lambda p: p.stat().st_size)
    mtl_maps: dict = {}
    for mtl in obj.parent.glob("*.mtl"):
        cur = None
        for line in mtl.read_text(errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "newmtl":
                cur = parts[1]
                mtl_maps[cur] = {}
            elif cur and parts[0] in ("map_Kd", "map_Bump", "bump", "norm", "map_Kn"):
                mtl_maps[cur][parts[0]] = obj.parent / parts[-1]
    v, vt, vn, faces, cur = [], [], [], {}, "default"
    for line in obj.read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            v.append(line.split()[1:4])
        elif line.startswith("vt "):
            vt.append(line.split()[1:3])
        elif line.startswith("vn "):
            vn.append(line.split()[1:4])
        elif line.startswith("usemtl"):
            cur = line.split()[1]
        elif line.startswith("f "):
            idx = [tuple(int(x) if x else 0 for x in (tok.split("/") + ["", ""])[:3]) for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.setdefault(cur, []).extend((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(v, float)
    vt = np.asarray(vt, float) if vt else np.zeros((1, 2))
    vn = np.asarray(vn, float) if vn else None
    (out / "textures").mkdir(parents=True, exist_ok=True)
    parts, lo, hi, total = [], np.full(3, np.inf), np.full(3, -np.inf), 0
    for mat, corners in faces.items():
        c = np.asarray(corners, np.int64)
        c = np.where(c < 0, c + np.array([len(v), len(vt), len(vn) if vn is not None else 0]) + 1, c)
        keys, inv = np.unique(c, axis=0, return_inverse=True)
        p = v[keys[:, 0] - 1]
        uv = vt[np.maximum(keys[:, 1] - 1, 0)][:, :2] if len(vt) > 1 else np.zeros((len(keys), 2))
        uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])          # OBJ puts v = 0 at the image bottom
        f = inv.reshape(-1, 3)
        if vn is not None and keys[:, 2].min() > 0:
            n = vn[keys[:, 2] - 1]
        else:
            n = np.zeros_like(p)
            fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
            for k in range(3):
                np.add.at(n, f[:, k], fn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in mat)
        G.Mesh(p, n, uv, f).write_ply(out / f"{name}.ply")
        maps = mtl_maps.get(mat, {})
        base = {"type": "rgb", "value": [0.6, 0.6, 0.6]}
        if maps.get("map_Kd") and maps["map_Kd"].exists():
            dst = out / "textures" / maps["map_Kd"].name
            if not dst.exists():
                Image.open(maps["map_Kd"]).convert("RGB").save(dst)
            base = {"type": "bitmap", "filename": str(dst.relative_to(out))}
        bsdf = {"type": "principled", "base_color": base, "roughness": 0.6}
        nmap = next((maps[k] for k in ("norm", "map_Kn", "map_Bump", "bump") if k in maps and "norm" in maps[k].name.lower()), None)
        if nmap is not None and nmap.exists():
            dst = out / "textures" / nmap.name
            if not dst.exists():
                Image.open(nmap).convert("RGB").save(dst)
            bsdf = {"type": "normalmap", "normalmap": {"type": "bitmap", "raw": True, "filename": str(dst.relative_to(out))},
                    "bsdf": bsdf}
        parts.append({"name": name, "ply": f"{name}.ply", "bsdf": {"type": "twosided", "bsdf": bsdf},
                      "triangles": int(len(f))})
        lo, hi, total = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0)), total + len(f)
    meta = {"converter_version": CONVERTER_VERSION, "parts": parts, "bounds": [lo.tolist(), hi.tolist()],
            "triangles": int(total), "note": "OBJ scans have arbitrary units and up axis: place() with height= and check"}
    _write_atomic(out / "model.json", json.dumps(meta, indent=1).encode())
    return meta


@dataclass
class Part:
    name: str
    ply: str
    bsdf: dict
    triangles: int


@dataclass
class Model:
    id: str
    res: str
    parts: list
    bounds: tuple          # ((x0, y0, z0), (x1, y1, z1)) in meters, y up, before placement
    triangles: int


def _absolute(spec, out: Path):
    if isinstance(spec, dict):
        return {k: (str(out / v) if k == "filename" else _absolute(v, out)) for k, v in spec.items()}
    return spec


def model(aid: str, res: str = "1k", foliage_translucency: float = 0.35) -> Model:
    """Fetch (once) and convert (once) a pinned Poly Haven model."""
    base = _base(aid, res)
    out = base / f"mitsuba_v{CONVERTER_VERSION}_t{foliage_translucency:g}"
    with _locked(base / ".lock"):
        if not (out / "model.json").exists():
            root, entry = _fetch(aid, res)
            if entry.get("kind") == "obj_zip":
                _convert_obj(root, out)
            else:
                _convert(root / next(k for k in entry["files"] if k.endswith(".gltf")), out, foliage_translucency)
    meta = json.loads((out / "model.json").read_text())
    parts = [Part(p["name"], str(out / p["ply"]), _absolute(p["bsdf"], out), p["triangles"]) for p in meta["parts"]]
    return Model(aid, res, parts, tuple(map(tuple, meta["bounds"])), meta["triangles"])


def surface(aid: str, res: str = "1k", uv_scale=None, tint=None) -> dict:
    """Principled BSDF (with normal map) from a pinned Poly Haven texture.

    `uv_scale` repeats the texture per uv unit. For meshes whose uv is in meters (scene_kit box_mesh,
    most procedural meshes) leave it None: it becomes 1 / the texture's real width from the manifest.
    `tint` multiplies the albedo (an rgb triple), for variations without another download."""
    import mitsuba as mi
    import numpy as np
    from PIL import Image

    base = _base(aid, res)
    out = base / f"surface_v{CONVERTER_VERSION}"
    with _locked(base / ".lock"):
        root, entry = _fetch(aid, res)
        if entry.get("kind") != "texture":
            raise ValueError(f"{aid} is a model; use model()")
        if entry.get("source") == "ambientcg":
            suffix = {"_Color.jpg": "diff", "_Roughness.jpg": "rough", "_NormalGL.jpg": "nor_gl", "_Metalness.jpg": "metal"}
            roles = {role: p for p in root.rglob("*.jpg") for end, role in suffix.items() if p.name.endswith(end)}
        else:
            roles = {f["role"]: root / name for name, f in entry["files"].items()}
        if not (out / "rough.png").exists():
            out.mkdir(parents=True, exist_ok=True)
            for role in ("rough", "metal"):
                if role in roles:
                    Image.open(roles[role]).convert("L").save(out / f"{role}.png")
    if uv_scale is None:
        width = (entry.get("dimensions_m") or [1.0])[0] or 1.0
        uv_scale = 1.0 / width
    su, sv = (uv_scale, uv_scale) if np.isscalar(uv_scale) else uv_scale
    to_uv = mi.ScalarTransform4f().scale([su, sv, 1.0])
    tex = lambda path, raw: {"type": "bitmap", "filename": str(path), "raw": raw, "to_uv": to_uv}
    bsdf = {"type": "principled", "base_color": tex(roles["diff"], False), "roughness": tex(out / "rough.png", True)}
    if "metal" in roles:
        bsdf["metallic"] = tex(out / "metal.png", True)
    if tint is not None:
        bsdf["base_color"] = {"type": "bitmap", "filename": str(_tinted(roles["diff"], tint, out)), "to_uv": to_uv}
    return {"type": "normalmap", "normalmap": tex(roles["nor_gl"], True), "bsdf": bsdf} if "nor_gl" in roles else bsdf


def _tinted(path: Path, tint, out: Path) -> Path:
    import numpy as np
    from PIL import Image

    dst = out / f"diff_tint_{'_'.join(f'{c:.3f}' for c in tint)}.png"
    if not dst.exists():
        a = np.asarray(Image.open(path).convert("RGB"), float) * np.asarray(tint, float)
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(dst)
    return dst


def place(m: Model, at=(0.0, 0.0, 0.0), yaw: float = 0.0, height: float | None = None, scale: float = 1.0):
    """Matrix putting the model's footprint centre and lowest point at `at`, turned `yaw` degrees about y,
    uniformly scaled by `scale` or to a total `height` in meters."""
    import numpy as np

    sys.path.insert(0, str(HERE))
    import procedural as G

    lo, hi = np.asarray(m.bounds[0]), np.asarray(m.bounds[1])
    s = height / (hi[1] - lo[1]) if height is not None else scale
    anchor = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return G.compose(G.translate(at), G.rotate((0, 1, 0), yaw), G.scale(s), G.translate(-anchor))


def add(b, m: Model, matrix, prefix: str | None = None, overrides: dict | None = None, instance: bool = True) -> None:
    """Add the model to a scene_kit.SceneBuilder at `matrix`.

    By default the geometry is stored once as a Mitsuba shapegroup per prefix and every call adds an
    instance of it, so ten copies of a 2M-triangle tree cost 2M triangles. Pass a different `prefix`
    for a differently coloured copy (`overrides` = {part name: bsdf}), or instance=False for plain shapes."""
    from scene_kit import xf

    prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in (prefix or f"web_{m.id}"))
    mats = {}
    for part in m.parts:
        mid = f"{prefix}__{part.name}"
        if mid not in b.d:
            b.material(mid, (overrides or {}).get(part.name, part.bsdf))
        mats[part.name] = mid
    if not instance:
        for part in m.parts:
            b.ply(part.ply, matrix, mats[part.name], prefix=prefix)
        return
    group = f"sg_{prefix}"
    if group not in b.d:
        b.d[group] = {"type": "shapegroup", **{f"part_{i}": {"type": "ply", "filename": part.ply,
                                                              "bsdf": {"type": "ref", "id": mats[part.name]}}
                                               for i, part in enumerate(m.parts)}}
    b.d[b._key(f"inst_{prefix}")] = {"type": "instance", "shapegroup": {"type": "ref", "id": group}, "to_world": xf(matrix)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("search", "pin", "stage", "info"))
    ap.add_argument("names", nargs="+")
    ap.add_argument("--res", default="1k")
    ap.add_argument("--textures", action="store_true", help="search textures instead of models")
    ap.add_argument("--url", help="pin si:<name>: the Smithsonian full-resolution OBJ zip URL")
    ap.add_argument("--name", help="pin si:<name>: a human-readable title")
    args = ap.parse_args(argv)
    if args.cmd == "search":
        search(args.names, "textures" if args.textures else "models")
    for aid in args.names if args.cmd != "search" else []:
        if args.cmd == "pin":
            pin(aid, args.res, args.url, args.name)
        elif args.cmd == "stage":                 # download only; remote_run.sh syncs web_assets/ to the cluster
            root, _ = _fetch(aid, args.res)
            print(f"staged {aid} in {root}")
        elif _read_manifest()["assets"].get(aid, {}).get(args.res, {}).get("kind") == "texture":
            _fetch(aid, args.res)
            print(f"{aid} @ {args.res}: texture fetched")
        else:
            m = model(aid, args.res)
            (x0, y0, z0), (x1, y1, z1) = m.bounds
            print(f"{aid} @ {args.res}: {m.triangles:,} triangles, size {x1 - x0:.2f} x {y1 - y0:.2f} x {z1 - z0:.2f} m")
            for p in m.parts:
                print(f"   {p.name:40s} {p.triangles:>9,} tris  {p.bsdf['type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
