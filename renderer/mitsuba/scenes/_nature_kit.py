"""Shared pieces for the natural scenes (owner beiba): terrain height fields, Poisson scattering,
scanned plants cut from Poly Haven variant sheets, and bark-textured fallen logs.

Meters, y up. Everything random takes an explicit numpy Generator, so the scenes stay deterministic.
"""

from __future__ import annotations

import math

import numpy as np

import procedural as G
import web_assets as W
from scene_kit import Assets, SceneBuilder, xf
from scenes import _greenhouse_props as H   # read_ply, split_variants: generic variant-sheet tools


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------
class Terrain:
    """y = h(x, z): a few sinusoids, smooth bowls (negative depth) or mounds, and a planar tilt.

    `waves` is a list of (amplitude m, wavelength m, direction degrees, phase radians); `bumps` a list of
    (x, z, radius m, height m), negative height for a hollow. Seed-independent unless the caller says so.
    """

    def __init__(self, x0, x1, z0, z1, spacing=0.3, waves=(), bumps=(), tilt=(0.0, 0.0), base=0.0):
        self.x0, self.x1, self.z0, self.z1 = x0, x1, z0, z1
        self.spacing = spacing
        self.waves, self.bumps, self.tilt, self.base = list(waves), list(bumps), tilt, base

    def height(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        h = self.base + self.tilt[0] * x + self.tilt[1] * z + np.zeros(np.broadcast(x, z).shape)
        for amp, lam, deg, ph in self.waves:
            a = math.radians(deg)
            h = h + amp * np.sin(math.tau / lam * (math.cos(a) * x + math.sin(a) * z) + ph)
        for bx, bz, r, dh in self.bumps:
            d2 = ((x - bx) ** 2 + (z - bz) ** 2) / (r * r)
            h = h + dh * np.exp(-2.5 * d2)                   # smooth, ~0 beyond the radius
        return h

    def at(self, x, z, dy=0.0) -> list[float]:
        return [float(x), float(self.height(x, z)) + dy, float(z)]

    def mesh(self) -> G.Mesh:
        xs = np.linspace(self.x0, self.x1, int(round((self.x1 - self.x0) / self.spacing)) + 1)
        zs = np.linspace(self.z0, self.z1, int(round((self.z1 - self.z0) / self.spacing)) + 1)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        Y = self.height(X, Z)
        p = np.stack([X, Y, Z], -1).reshape(-1, 3)
        f = G._grid_faces(len(xs), len(zs))
        uv = np.stack([X, Z], -1).reshape(-1, 2)            # metres, for scanned textures at real scale
        mesh = G.Mesh(p, G.vertex_normals(p, f), uv, f)
        return _upward(mesh)


def _upward(mesh: G.Mesh) -> G.Mesh:
    """Make every face's winding and the vertex normals point up (+y), the air side of the ground."""
    p, f = mesh.p, mesh.f.copy()
    fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    flip = fn[:, 1] < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    n = G.vertex_normals(p, f)
    n[n[:, 1] < 0] *= -1
    return G.Mesh(p, n, mesh.uv, f)


def water_sheet(x0, x1, z0, z1, y, seed: int, spacing=0.05, slope_rms=0.02, waves=24, mask=None) -> G.Mesh:
    """A rippled water surface (the pool scene's recipe) clipped to the cells where `mask(x, z)` is true."""
    from scenes import _pool_props as PP

    m = PP.water_surface(x0, x1, z1, z0, y, seed, spacing=spacing, slope_rms=slope_rms, waves=waves)
    if mask is None:
        return m
    c = m.p[m.f].mean(1)
    keep = mask(c[:, 0], c[:, 2])
    used, inv = np.unique(m.f[keep], return_inverse=True)
    return G.Mesh(m.p[used], m.n[used], m.uv[used], inv.reshape(-1, 3))


# --------------------------------------------------------------------------
# Scattering
# --------------------------------------------------------------------------
def scatter(rng, box, n, min_dist, avoid=(), taken=(), tries=40) -> list[tuple[float, float]]:
    """Dart-throwing Poisson disc in box ((x0, z0), (x1, z1)): up to n points, at least min_dist apart and
    from every point in `taken`; `avoid` is a list of (x, z, r) discs and ((x0, z0), (x1, z1)) rectangles."""
    (x0, z0), (x1, z1) = box
    pts: list[tuple[float, float]] = []
    others = [tuple(t[:2]) for t in taken]
    for _ in range(n * tries):
        if len(pts) >= n:
            break
        x, z = float(rng.uniform(x0, x1)), float(rng.uniform(z0, z1))
        if any(_inside(x, z, a) for a in avoid):
            continue
        if any((x - px) ** 2 + (z - pz) ** 2 < min_dist ** 2 for px, pz in pts + others):
            continue
        pts.append((x, z))
    return pts


def _inside(x, z, a) -> bool:
    if len(a) == 3 and not isinstance(a[0], (tuple, list)):
        return (x - a[0]) ** 2 + (z - a[1]) ** 2 < a[2] ** 2
    (ax0, az0), (ax1, az1) = a
    return ax0 <= x <= ax1 and az0 <= z <= az1


# --------------------------------------------------------------------------
# Scanned models
# --------------------------------------------------------------------------
class Library:
    """Scanned Poly Haven models, whole or one plant cut out of a variant sheet (the sheet is split once
    into the shared cache). Every placement is an instance, so repeats cost no geometry."""

    def __init__(self, b: SceneBuilder):
        self.b, self.sheets = b, {}

    def _materials(self, aid, m, overrides):
        mats = {}
        for p in m.parts:
            mid = f"web_{aid}__{p.name}"
            if mid not in self.b.d:
                self.b.material(mid, (overrides or {}).get(p.name, p.bsdf))
            mats[p.name] = mid
        return mats

    def variants(self, aid, overrides=None) -> list:
        """[(parts {name: ply path}, height m, half-width m)] for each plant in the sheet."""
        if aid not in self.sheets:
            m = W.model(aid)
            self._materials(aid, m, overrides)

            def build():
                meshes = {p.name: H.read_ply(p.ply) for p in m.parts}
                return {f"v{k}--{name}": mesh for k, v in enumerate(H.split_variants(meshes)) for name, mesh in v.items()}

            found: dict = {}
            for key, path in self.b.assets.meshes(f"web_{aid}_variants", build).items():
                k, name = key.split("--", 1)
                found.setdefault(int(k[1:]), {})[name] = path
            sheet = []
            for k in sorted(found):
                ps = np.concatenate([H.read_ply(q).p for q in found[k].values()])
                sheet.append((found[k], float(ps[:, 1].max()), float(np.abs(ps[:, [0, 2]]).max())))
            self.sheets[aid] = sheet
        return self.sheets[aid]

    def plant(self, aid, k, at, yaw=0.0, height=None, scale=1.0, tilt=(0.0, 0.0), overrides=None, tag=""):
        """Variant k of a sheet, base at `at`, scaled to `height` (or by `scale`), leaned by tilt (x, z) degrees."""
        self.variants(aid, overrides)
        k, m, placed = self.matrix(aid, k, at, yaw, height, scale, tilt)
        self.instance(aid, k, m, tag)
        return placed

    def instance(self, aid, k, m, tag="") -> None:
        """Instance variant k of a sheet at matrix m (from `matrix`), with the tagged material set if given."""
        k %= len(self.variants(aid))
        parts = self.sheets[aid][k][0]
        group = f"sg_var_{aid}{tag}_{k}"
        if group not in self.b.d:
            self.b.d[group] = {"type": "shapegroup", **{f"part_{i}": {"type": "ply", "filename": path,
                                                                      "bsdf": {"type": "ref", "id": f"web_{aid}{tag}__{name}"}}
                                                       for i, (name, path) in enumerate(parts.items())}}
        self.b.d[self.b._key(f"inst_var_{aid}")] = {"type": "instance", "shapegroup": {"type": "ref", "id": group},
                                                    "to_world": xf(m)}

    def material_set(self, aid, tag, transform) -> None:
        """Register a second set of materials for a scanned model, ids web_<aid><tag>__<part>, each the scan's
        BSDF passed through `transform` (for example a clear coat for wet stones). Plants placed with the same
        tag use them; untagged placements keep the scan's own materials."""
        import copy
        m = W.model(aid)
        for p in m.parts:
            mid = f"web_{aid}{tag}__{p.name}"
            if mid not in self.b.d:
                self.b.material(mid, transform(copy.deepcopy(p.bsdf)))

    def top_point(self, aid, k, m) -> list[float]:
        """World position of the highest vertex of variant k placed by matrix m: a point on the real surface,
        so a focus point there passes the sampler's visibility test (a surface pixel within 0.15 m)."""
        key = (aid, k % len(self.variants(aid)))
        if not hasattr(self, "_tops"):
            self._tops = {}
        if key not in self._tops:
            ps = np.concatenate([H.read_ply(q).p for q in self.variants(aid)[key[1]][0].values()])
            self._tops[key] = ps[np.argmax(ps[:, 1])]
        return (np.asarray(m) @ np.append(self._tops[key], 1.0))[:3].tolist()

    def matrix(self, aid, k, at, yaw=0.0, height=None, scale=1.0, tilt=(0.0, 0.0)):
        """(variant index, placement matrix, placed height) for variant k of a sheet, as `plant` places it."""
        vs = self.variants(aid)
        k %= len(vs)
        s = height / vs[k][1] if height is not None else scale
        m = G.compose(G.translate(at), G.rotate((1, 0, 0), tilt[0]), G.rotate((0, 0, 1), tilt[1]),
                      G.rotate((0, 1, 0), yaw), G.scale(s))
        return k, m, vs[k][1] * s

    def whole(self, aid, at, yaw=0.0, height=None, scale=1.0, drop_parts=(), prefix=None):
        """A whole scanned model, footprint centre and lowest point at `at`. `drop_parts` removes parts whose
        name ends with any of the given suffixes (for example bare trees without `_leaves`)."""
        m = W.model(aid)
        if drop_parts:
            m = W.Model(m.id, m.res, [p for p in m.parts if not p.name.endswith(tuple(drop_parts))], m.bounds, m.triangles)
        M = W.place(m, at=at, yaw=yaw, height=height, scale=scale)
        W.add(self.b, m, M, prefix=prefix)
        lo, hi = np.asarray(m.bounds[0]), np.asarray(m.bounds[1])
        return (hi[1] - lo[1]) * (height / (hi[1] - lo[1]) if height is not None else scale)


class Batch:
    """Variant-sheet plants merged into one plain mesh per material instead of one instance each.

    Measured on forest_floor (M4 Max, Metal, 1920x1080, per 1024 spp): ~1,050 scattered plants as instances
    cost 373 s; the same plants merged into 4 meshes cost 98 s with an identical image (2.9M triangles). Many
    small, overlapping instance boxes are what costs, not triangles. Use a Batch for scattered undergrowth and
    keep instances for a few hero plants. The merged meshes depend on the placements, so `flush` caches them
    under a key the caller makes unique (seed and version) in the seed's own asset directory."""

    def __init__(self, lib: Library):
        self.lib, self.items = lib, []

    def plant(self, aid, k, at, yaw=0.0, height=None, scale=1.0, tilt=(0.0, 0.0), tag="") -> float:
        k, m, placed = self.lib.matrix(aid, k, at, yaw, height, scale, tilt)
        self.items.append((aid, k, m, tag))
        self.last = (aid, k, m)
        return placed

    def flush(self, assets: Assets, key: str) -> None:
        items = list(self.items)

        def build():
            cache, acc = {}, {}
            for aid, k, m, tag in items:
                for name, path in self.lib.variants(aid)[k][0].items():
                    if path not in cache:
                        cache[path] = H.read_ply(path)
                    acc.setdefault(f"web_{aid}{tag}__{name}", []).append(cache[path].transformed(m))
            return {mat: _concat(meshes) for mat, meshes in acc.items()}

        # Sorted, so a cold build (meshes in creation order) and a warm one (cache files, which glob returns
        # sorted) number the shapes alike and give the same fingerprint.
        for mat, path in sorted(assets.meshes(key, build).items()):
            self.lib.b.ply(path, np.eye(4), mat, prefix="batch")
        self.items = []


def _concat(meshes) -> G.Mesh:
    offsets = np.cumsum([0] + [len(m.p) for m in meshes[:-1]])
    return G.Mesh(np.concatenate([m.p for m in meshes]), np.concatenate([m.n for m in meshes]),
                  np.concatenate([m.uv for m in meshes]), np.concatenate([m.f + o for m, o in zip(meshes, offsets)]))


# --------------------------------------------------------------------------
# Distant woodland backdrop
# --------------------------------------------------------------------------
# All geometry, no alpha: render.py's G-buffer takes the first surface a ray hits and ignores mask opacity,
# so a masked canopy edge would label sky-hole pixels with the band's depth while the image shows sky.

class Skyline:
    """Height of the far canopy edge along a ring: a sum of sinusoids in arc length, 60 m down to 0.6 m
    wavelengths, so the silhouette is lumpy at every scale instead of a few straight-edged teeth."""

    def __init__(self, seed, radius, base=13.0, spread=4.0):
        rng = np.random.default_rng(seed)
        self.radius, self.base = radius, base
        lam = np.array([60.0, 25.0, 9.0, 3.5, 1.4, 0.6])
        amp = np.array([2.5, 1.6, 0.9, 0.45, 0.22, 0.1]) * spread / 4.0
        circ = math.tau * radius
        self.k = np.round(circ / lam) * math.tau / circ          # whole cycles around the ring, so it closes
        self.amp, self.ph = amp, rng.uniform(0, math.tau, len(lam))

    def top(self, theta):
        s = np.asarray(theta)[..., None] * self.radius
        return self.base + (self.amp * np.sin(self.k * s + self.ph)).sum(-1)


def backdrop_band(cx, cz, sky: Skyline, y0, y_tex_top, tile_w, step=0.1) -> G.Mesh:
    """Vertical band (normals inward) around (cx, cz) from y0 up to the skyline, one column every `step` m of
    arc. u repeats every tile_w metres; v is 0 at y_tex_top and 1 at y0 (Mitsuba puts v = 0 at the image top)."""
    n = int(math.tau * sky.radius / step)
    th = np.linspace(0.0, math.tau, n + 1)
    x, z, top = cx + sky.radius * np.cos(th), cz + sky.radius * np.sin(th), sky.top(th)
    u = th * sky.radius / tile_w
    u = u * (round(u[-1]) / u[-1])
    p = np.concatenate([np.stack([x, np.full_like(x, y0), z], -1), np.stack([x, top, z], -1)])
    nrm = np.concatenate([np.stack([-np.cos(th), 0 * th, -np.sin(th)], -1)] * 2)
    v = lambda y: (y_tex_top - y) / (y_tex_top - y0)
    uv = np.concatenate([np.stack([u, np.ones_like(u)], -1), np.stack([u, v(top)], -1)])
    return G.Mesh(p, nrm, uv, G._grid_faces(2, n + 1)).oriented()


def backdrop_fringe(rng, cx, cz, sky: Skyline, per_m=3.0, depth=(0.5, 4.0)) -> G.Mesh:
    """Leaf clusters fraying the skyline: irregular 7-sided discs facing the ring centre, scattered from 2.5 m
    below the edge to 1.5 m above it, thinner higher up, a few metres in front of the band. Sky shows through
    the real gaps between them."""
    n = int(math.tau * sky.radius * per_m)
    th = rng.uniform(0, math.tau, n)
    dy = rng.uniform(-2.5, 1.5, n)
    keep = rng.random(n) < np.clip(1.0 - (dy + 2.5) / 4.0, 0.08, 1.0)
    th, dy = th[keep], dy[keep]
    r = sky.radius - rng.uniform(*depth, len(th))
    centre = np.stack([cx + r * np.cos(th), sky.top(th) + dy, cz + r * np.sin(th)], -1)
    inward = np.stack([-np.cos(th), 0 * th, -np.sin(th)], -1)
    tangent = np.stack([-np.sin(th), 0 * th, np.cos(th)], -1)
    size = rng.uniform(0.3, 1.2, len(th))
    k = 7
    ang = np.linspace(0, math.tau, k, endpoint=False)[None, :] + rng.uniform(0, math.tau, (len(th), 1))
    rad = size[:, None] * rng.uniform(0.55, 1.0, (len(th), k))
    ring = (centre[:, None, :] + rad[..., None] * (np.cos(ang)[..., None] * tangent[:, None, :]
                                                   + np.sin(ang)[..., None] * np.array([0.0, 1.0, 0.0])))
    p = np.concatenate([centre[:, None, :], ring], 1).reshape(-1, 3)
    nrm = np.repeat(inward, k + 1, 0)
    base = (np.arange(len(th)) * (k + 1))[:, None]
    f = np.stack([np.zeros(k, int), np.arange(1, k + 1), np.roll(np.arange(1, k + 1), -1)], -1)[None] + base[..., None]
    uv = np.zeros((len(p), 2))
    return G.Mesh(p, nrm, uv, f.reshape(-1, 3)).oriented()


def _periodic(a: np.ndarray, overlap: int) -> np.ndarray:
    """Cross-fade the last `overlap` columns into the first ones and drop them, so the image tiles in u."""
    w = a.shape[1] - overlap
    ramp = np.linspace(0.0, 1.0, overlap)[None, :]
    out = a[:, :w].copy()
    out[:, :overlap] = a[:, w:] * (1 - ramp) + a[:, :overlap] * ramp
    return out


def _srgb(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def backdrop_albedo(seed: int, tile_w: float, y0: float, y_top: float, ppm: int = 32) -> np.ndarray:
    """sRGB albedo tile for `backdrop_band`: a dark self-shadowed canopy above 9 m, deep shade with lit trunks
    below 5 m, blended between. Tiles seamlessly in u."""
    W, H = int(tile_w * ppm), int((y_top - y0) * ppm)
    O = W // 6
    rng = np.random.default_rng(seed)
    y = y_top - (np.arange(H)[:, None] + 0.5) / ppm
    clumps = G.fbm(H, W + O, octaves=5, base=16, seed=seed + 2, aspect=(W + O) / H)
    # Dark on purpose: a forest seen 100 m deep is mostly shade. A vertical band facing the sky catches far
    # more light than a canopy interior, so its albedo sits well below the near foliage (measured: at 3x
    # this it read as a pale grey wall between the trunks).
    canopy = np.stack([0.005 + 0.009 * clumps, 0.009 + 0.014 * clumps, 0.003 + 0.004 * clumps], -1)
    shade = np.array([0.004, 0.005, 0.003])
    trunks = np.zeros((1, W + O))
    for _ in range(int(tile_w / 1.6)):
        c, w = rng.uniform(0, W + O), rng.uniform(0.25, 0.6) * ppm
        trunks = np.maximum(trunks, np.clip(1.0 - np.abs(np.arange(W + O)[None, :] - c) / w, 0, 1) ** 0.5
                            * rng.uniform(0.5, 1.0))
    bark = np.array([0.010, 0.009, 0.007])                        # trunks only slightly lighter than shade
    zone = np.clip((9.0 - y) / 4.0, 0.0, 1.0)[..., None]
    lin = canopy * (1 - zone) + (shade + (bark - shade) * trunks[..., None]) * zone
    out = _periodic(_srgb(lin).transpose(2, 0, 1).reshape(3 * H, W + O), O).reshape(3, H, W).transpose(1, 2, 0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Procedural pieces
# --------------------------------------------------------------------------
def log_axis(rng, length, radius, rings=24):
    """Centre line (rings x 3, local frame: along +x, resting on y = 0, bent sideways in z) and radius per ring
    of the fallen trunk `log` builds from the same rng state. Use it to put things on the trunk's real surface."""
    s = np.linspace(0.0, 1.0, rings)
    sag = 0.04 * length * np.sin(np.pi * s) * rng.uniform(-1, 1)
    r = radius * (1.0 - 0.25 * s) * (1.0 + 0.05 * np.sin(s * 9.0 + rng.uniform(0, 6)))
    return np.stack([s * length, r, sag], -1), r


def log_top(axis, r, u) -> np.ndarray:
    """Local point on top of the trunk at fraction u of its length."""
    x = u * axis[-1, 0]
    return np.array([x, 2.0 * np.interp(x, axis[:, 0], r), np.interp(x, axis[:, 0], axis[:, 2])])


def log(rng, length, radius, sides=18, rings=24) -> G.Mesh:
    """A fallen trunk along +x from the origin, resting on y = 0: slightly bent, tapering, with a lumpy
    section; uv in metres (u around the circumference, v along the trunk) for a scanned bark texture."""
    path, r = log_axis(rng, length, radius, rings)
    s = path[:, 0] / length
    sag = path[:, 2]
    mesh = G.sweep(path, r, sides)
    lump = 1.0 + 0.05 * np.sin(np.arctan2(mesh.n[:, 2], mesh.n[:, 1]) * 5.0)
    axis = np.stack([mesh.p[:, 0], np.interp(mesh.p[:, 0], s * length, r), np.interp(mesh.p[:, 0], s * length, sag)], -1)
    mesh.p = axis + (mesh.p - axis) * lump[:, None]
    mesh.uv = mesh.uv * np.array([math.tau * radius, length])
    return mesh
