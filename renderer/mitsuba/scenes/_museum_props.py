"""Mesh recipes and textures for the museum scene. Pure numpy + Pillow; nothing here imports Mitsuba.

Conventions as in props.py: meters, y up, each recipe in a local frame with its base at y = 0.
Textures are returned with row 0 at the TOP of the image; the scene flips them (np.flipud) where a
Mitsuba `rectangle` reads rows from the bottom.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import procedural as G
from procedural import Mesh, compose, ellipsoid, lathe, rotate, scale, sweep, translate

TAU = 2.0 * math.pi


# ----------------------------------------------------------------------------
# Small mesh helpers
# ----------------------------------------------------------------------------

def rot_between(a, b) -> np.ndarray:
    """4x4 rotation taking direction a onto direction b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    m = np.eye(4)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return m
        return rotate(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]), 180.0)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    m[:3, :3] = np.eye(3) + k + k @ k / (1.0 + c)
    return m


def join(*meshes: Mesh) -> Mesh:
    out = Mesh()
    for m in meshes:
        out += m
    return out


def tube(p0, p1, radius, sides: int = 12) -> Mesh:
    return sweep([p0, p1], radius, sides)


def rounded_box(size, r: float, seg: int = 4) -> Mesh:
    """Box with rounded vertical and top edges: a superellipsoid-ish lathe would be overkill, so this
    places a smaller box plus edge cylinders and corner spheres (closed, opaque use only)."""
    sx, sy, sz = (np.asarray(size, float) / 2)
    out = G.box((2 * sx, 2 * sy - r, 2 * (sz - r)), (0, -r / 2, 0))
    out += G.box((2 * (sx - r), 2 * sy - r, 2 * sz), (0, -r / 2, 0))
    out += G.box((2 * (sx - r), r, 2 * (sz - r)), (0, sy - r / 2, 0))
    for x in (-sx + r, sx - r):
        out += tube((x, sy - r, -sz + r), (x, sy - r, sz - r), r, 4 * seg)
        for z in (-sz + r, sz - r):
            out += tube((x, -sy, z), (x, sy - r, z), r, 4 * seg)
            out += ellipsoid((r, r, r), 2 * seg + 1, 4 * seg).transformed(translate((x, sy - r, z)))
    for z in (-sz + r, sz - r):
        out += tube((-sx + r, sy - r, z), (sx - r, sy - r, z), r, 4 * seg)
    return out


# ----------------------------------------------------------------------------
# Picture frames: a moulding profile run round the canvas with mitred corners
# ----------------------------------------------------------------------------

FRAME_PROFILES = {
    # (outward offset from the canvas edge, depth toward the viewer), walked from the rebate outward.
    "gilt": [(-0.008, 0.0), (-0.008, 0.012), (0.0, 0.016), (0.006, 0.03), (0.016, 0.046), (0.03, 0.05),
             (0.042, 0.044), (0.05, 0.05), (0.062, 0.064), (0.078, 0.07), (0.094, 0.062), (0.104, 0.045),
             (0.11, 0.02), (0.11, 0.0)],
    "black": [(-0.006, 0.0), (-0.006, 0.03), (0.0, 0.038), (0.03, 0.038), (0.034, 0.034), (0.034, 0.0)],
    "oak": [(-0.006, 0.0), (-0.006, 0.026), (0.004, 0.032), (0.05, 0.032), (0.055, 0.026), (0.055, 0.0)],
    "float": [(-0.004, 0.0), (-0.004, 0.012), (0.012, 0.012), (0.012, 0.05), (0.02, 0.05), (0.02, 0.0)],
}


def frame_moulding(w: float, h: float, style: str) -> Mesh:
    """Frame around a w x h canvas centred at the origin in the xy plane, front toward +z."""
    prof = np.asarray(FRAME_PROFILES[style], float)
    o, d = prof[:, 0], prof[:, 1]
    do, dd = np.gradient(o), np.gradient(d)
    n2 = np.stack([-dd, do], 1)
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
    out = Mesh()
    sides = [  # (along axis, outward axis, outward sign, half length, edge position)
        (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), w / 2, h / 2),
        (np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), w / 2, h / 2),
        (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), h / 2, w / 2),
        (np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), h / 2, w / 2),
    ]
    for along, outward, half, edge in sides:
        pts, nrm, uv = [], [], []
        for i in range(len(prof)):
            for s in (-1.0, 1.0):
                pts.append(along * s * (half + o[i]) + outward * (edge + o[i]) + np.array([0, 0, d[i]]))
                nrm.append(outward * n2[i, 0] + np.array([0, 0, n2[i, 1]]))
                uv.append(((s + 1) / 2, i / (len(prof) - 1)))
        n = len(prof)
        f = []
        for i in range(n - 1):
            a, b, c, e = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
            f += [[a, c, b], [b, c, e]]
        out += Mesh(np.array(pts), np.array(nrm), np.array(uv), np.array(f)).oriented()
    return out


def canvas_box(w: float, h: float, depth: float = 0.03) -> dict[str, Mesh]:
    """Stretched canvas: the painted face (uv over [0,1]^2, v = 0 at the bottom) and the sides."""
    face = Mesh(np.array([[-w / 2, -h / 2, depth], [w / 2, -h / 2, depth], [w / 2, h / 2, depth], [-w / 2, h / 2, depth]]),
                np.tile([0, 0, 1.0], (4, 1)), np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]), np.array([[0, 1, 2], [0, 2, 3]]))
    sides = G.box((w, h, depth), (0, 0, depth / 2))
    keep = np.abs(sides.n[:, 2]) < 0.5
    tri = keep[sides.f].all(axis=1)
    return {"face": face, "sides": Mesh(sides.p, sides.n, sides.uv, sides.f[tri])}


# ----------------------------------------------------------------------------
# Lighting hardware
# ----------------------------------------------------------------------------

def spot_head() -> dict[str, Mesh]:
    """Track spotlight can, axis along -y from the pivot at the origin; the lens faces -y."""
    can = lathe([(0.0, -0.20), (0.034, -0.20), (0.042, -0.19), (0.044, -0.17), (0.044, -0.05), (0.036, -0.03),
                 (0.03, -0.0), (0.0, 0.0)], 40)
    fins = Mesh()
    for k in range(6):
        y = -0.06 - 0.016 * k
        fins += lathe([(0.0, y), (0.05, y), (0.05, y - 0.004), (0.0, y - 0.004)], 40)
    lens = lathe([(0.0, -0.2015), (0.03, -0.2015), (0.03, -0.2005), (0.0, -0.2005)], 32)
    return {"can": join(can, fins), "lens": lens}


def spot_mount(track_y: float, pivot) -> Mesh:
    """Adapter on the track, a stem down and a yoke around the pivot."""
    px, py, pz = pivot
    m = G.box((0.06, 0.035, 0.05), (px, track_y - 0.0175, pz))
    m += tube((px, track_y - 0.03, pz), (px, py + 0.06, pz), 0.008, 10)
    m += tube((px - 0.055, py + 0.06, pz), (px + 0.055, py + 0.06, pz), 0.007, 10)
    m += tube((px - 0.055, py + 0.06, pz), (px - 0.055, py, pz), 0.006, 10)
    m += tube((px + 0.055, py + 0.06, pz), (px + 0.055, py, pz), 0.006, 10)
    return m


# ----------------------------------------------------------------------------
# Furniture
# ----------------------------------------------------------------------------

def bench(length: float = 1.8, width: float = 0.48, seat: float = 0.46) -> dict[str, Mesh]:
    """Gallery bench along z: a leather cushion on a brushed-steel frame."""
    cushion = rounded_box((width, 0.09, length), 0.025).transformed(translate((0, seat - 0.045, 0)))
    steel = Mesh()
    for z in (-length / 2 + 0.12, length / 2 - 0.12):
        steel += G.box((width - 0.06, 0.03, 0.05), (0, seat - 0.105, z))
        for x in (-width / 2 + 0.04, width / 2 - 0.04):
            steel += G.box((0.03, seat - 0.12, 0.05), (x, (seat - 0.12) / 2, z))
    steel += G.box((0.04, 0.03, length - 0.24), (0, seat - 0.105, 0))
    return {"cushion": cushion, "steel": steel}


def stanchion() -> Mesh:
    return lathe([(0.0, 0.0), (0.14, 0.0), (0.145, 0.008), (0.12, 0.03), (0.04, 0.05), (0.024, 0.07), (0.022, 0.9),
                  (0.032, 0.92), (0.032, 0.94), (0.026, 0.95), (0.03, 0.97), (0.0, 0.99)], 48)


def rope(p0, p1, sag: float, radius: float = 0.017, n: int = 40) -> Mesh:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)[:, None]
    path = p0 * (1 - t) + p1 * t
    path[:, 1] -= sag * 4 * t[:, 0] * (1 - t[:, 0])
    return sweep(path, radius, 14)


def plinth(w: float, h: float, d: float) -> dict[str, Mesh]:
    """A white plinth with a dark recessed shadow gap at the floor."""
    return {"body": G.box((w, h - 0.03, d), (0, 0.03 + (h - 0.03) / 2, 0)),
            "gap": G.box((w - 0.04, 0.03, d - 0.04), (0, 0.015, 0))}


# ----------------------------------------------------------------------------
# Artefacts
# ----------------------------------------------------------------------------

def amphora(height: float = 0.42) -> dict[str, Mesh]:
    s = height / 0.63
    prof = [(0.0, 0.0), (0.045, 0.0), (0.05, 0.012), (0.034, 0.03), (0.04, 0.055), (0.075, 0.1), (0.11, 0.17),
            (0.135, 0.25), (0.14, 0.31), (0.128, 0.39), (0.1, 0.45), (0.064, 0.49), (0.046, 0.52), (0.042, 0.57),
            (0.048, 0.6), (0.062, 0.612), (0.062, 0.628), (0.05, 0.63), (0.04, 0.615), (0.036, 0.58), (0.0, 0.56)]
    body = lathe([(r * s, y * s) for r, y in prof], 72)
    handles = Mesh()
    for side in (-1.0, 1.0):
        pts = np.array([(0.044, 0.585), (0.09, 0.6), (0.118, 0.57), (0.12, 0.5), (0.105, 0.44)]) * s
        path = np.column_stack([side * pts[:, 0], pts[:, 1], np.zeros(len(pts))])
        handles += sweep(_spline(path, 10), 0.009 * s * 1.4, 10)
    return {"body": body, "handles": handles}


def _spline(pts: np.ndarray, per: int = 8) -> np.ndarray:
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-2])
    return np.array(out)


def jug(height: float = 0.24) -> dict[str, Mesh]:
    s = height / 0.24
    prof = [(0.0, 0.0), (0.04, 0.0), (0.042, 0.006), (0.06, 0.04), (0.075, 0.09), (0.072, 0.13), (0.05, 0.17),
            (0.032, 0.195), (0.03, 0.22), (0.04, 0.236), (0.036, 0.24), (0.026, 0.225), (0.0, 0.2)]
    body = lathe([(r * s, y * s) for r, y in prof], 64)
    path = np.array([(0.03, 0.215, 0), (0.07, 0.235, 0), (0.09, 0.2, 0), (0.08, 0.14, 0), (0.07, 0.115, 0)]) * s
    return {"body": body, "handles": sweep(_spline(path, 10), 0.007 * s, 10)}


def bowl(radius: float = 0.11, height: float = 0.065) -> Mesh:
    r, hh, w = radius, height, 0.005
    return lathe([(0.0, 0.0), (0.4 * r, 0.0), (0.42 * r, 0.01), (0.7 * r, 0.03), (0.92 * r, 0.06 * hh / 0.065),
                  (r, hh), (r - w, hh), (0.88 * r, 0.055 * hh / 0.065), (0.6 * r, 0.03), (0.0, 0.016)], 72)


def coin(radius: float, thick: float = 0.0022) -> Mesh:
    r, t = radius, thick
    return lathe([(0.0, 0.0), (r, 0.0), (r, t), (0.93 * r, t), (0.9 * r, 0.75 * t), (0.0, 0.8 * t)], 40, planar_uv=2 * r)


def coin_tray(rng, n: int = 12, spacing: float = 0.05) -> dict[str, Mesh]:
    """Coins in rows on a 0.34 x 0.2 tray, split by metal."""
    out = {"gold": Mesh(), "silver": Mesh(), "bronze": Mesh()}
    cols = 6
    for k in range(n):
        i, j = k % cols, k // cols
        r = float(rng.uniform(0.009, 0.016))
        metal = ["gold", "silver", "bronze"][int(rng.integers(0, 3))]
        x = (i - (cols - 1) / 2) * spacing + rng.uniform(-0.004, 0.004)
        z = (j - 0.5) * 0.07 + rng.uniform(-0.004, 0.004)
        tilt = rotate((1, 0, 0), float(rng.uniform(-3, 3)))
        out[metal] += coin(r).transformed(compose(translate((x, 0.0, z)), rotate((0, 1, 0), float(rng.uniform(0, 360))), tilt))
    return out


# A jeweller's bust: foot, short stem, chest and shoulders flaring out, then the neck rising from them.
NECK_PROFILE = [(0.0, 0.0), (0.075, 0.0), (0.075, 0.012), (0.03, 0.03), (0.028, 0.07), (0.07, 0.095), (0.105, 0.118),
                (0.112, 0.13), (0.1, 0.142), (0.07, 0.155), (0.045, 0.168), (0.034, 0.185), (0.032, 0.27), (0.028, 0.282),
                (0.0, 0.285)]
_NECK_OUTER = slice(7, 13)      # the shoulder slope and neck, where y increases monotonically


def neck_stand() -> Mesh:
    """Velvet-covered display bust for a necklace."""
    return lathe(NECK_PROFILE, 64)


def neck_radius(y):
    prof = np.array(NECK_PROFILE[_NECK_OUTER])
    return np.interp(y, prof[:, 1], prof[:, 0])


def necklace() -> dict[str, Mesh]:
    """Gold beads resting on the stand's neck, dropping to a pendant gem at the front (+z)."""
    t = np.linspace(0, TAU, 64, endpoint=False)
    drop = np.exp(-((((t - math.pi / 2 + math.pi) % TAU) - math.pi) / 0.8) ** 2)
    y = 0.176 - 0.043 * drop          # the front drapes down to the shoulder rim
    r = neck_radius(y) + 0.0052
    pts = np.column_stack([r * np.cos(t), y, r * np.sin(t)])
    beads = Mesh()
    for k, p in enumerate(pts):
        r = 0.0045 if k % 4 else 0.0062
        beads += ellipsoid((r, r, r), 8, 12).transformed(translate(p))
    front = pts[int(np.argmax(drop))]
    gem = ellipsoid((0.011, 0.014, 0.007), 14, 24).transformed(translate(front + np.array([0, -0.02, 0.006])))
    setting = sweep(np.array([[np.cos(a) * 0.0122, np.sin(a) * 0.0152, 0] for a in np.linspace(0, TAU, 25)])
                    + front + np.array([0, -0.02, 0.005]), 0.0016, 8, closed=False)
    return {"gold": join(beads, setting), "gem": gem}


def ring(radius: float = 0.0095) -> dict[str, Mesh]:
    a = np.linspace(0, TAU, 49)
    band = sweep(np.column_stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)]), 0.0019, 10)
    gem = ellipsoid((0.0045, 0.0035, 0.0045), 10, 16).transformed(translate((0, radius + 0.004, 0)))
    claws = Mesh()
    for k in range(4):
        ang = TAU * k / 4 + math.pi / 4
        claws += tube((0.0042 * math.cos(ang), radius + 0.0015, 0.0042 * math.sin(ang)),
                      (0.0036 * math.cos(ang), radius + 0.0062, 0.0036 * math.sin(ang)), 0.0007, 6)
    return {"gold": join(band, claws), "gem": gem}


def ring_cone() -> Mesh:
    return lathe([(0.0, 0.0), (0.016, 0.0), (0.012, 0.03), (0.007, 0.05), (0.0, 0.052)], 32)


def ammonite(size: float = 0.11, turns: float = 3.6, ribs: int = 34) -> Mesh:
    """Logarithmic coil with ribs, coiled in the xy plane (lying on its side) and centred at the origin."""
    b = 0.19
    th = np.linspace(0.0, turns * TAU, int(220 * turns))
    a = size * 0.5 / math.exp(b * th[-1]) / 1.55
    r = a * np.exp(b * th)
    path = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)])
    rad = 0.55 * r * (1.0 + 0.07 * np.sin(ribs * th / turns * 2.4))
    m = sweep(path, rad, 24)
    return m.transformed(scale((1.0, 1.0, 0.62)))


def rest_on_floor(m: Mesh, matrix=None) -> Mesh:
    """Apply a rotation, then lift the mesh so its lowest point is at y = 0."""
    if matrix is not None:
        m = m.transformed(matrix)
    return m.transformed(translate((0, -m.p[:, 1].min(), 0)))


def stone_slab(w: float, d: float, t: float, rng) -> Mesh:
    m = G.box((w, t, d), (0, t / 2, 0))
    jitter = rng.normal(0, 0.0015, m.p.shape)
    jitter[:, 1] = np.where(m.p[:, 1] > t / 2, jitter[:, 1], 0)
    return Mesh(m.p + jitter, m.n, m.uv, m.f)


def acrylic_stand() -> Mesh:
    return join(G.box((0.06, 0.004, 0.05), (0, 0.002, 0)), G.box((0.004, 0.07, 0.03), (0, 0.035, -0.01)))


def manuscript(width: float = 0.38, depth: float = 0.27, rows: int = 24, cols: int = 60) -> dict[str, Mesh]:
    """An open codex on a cradle: two gently domed pages (uv u across the spread, v = 0 at the far edge),
    a page block, boards, and the cradle wedges. Local frame: spread along x, spine along z, up +y."""
    u = np.linspace(0, 1, cols)
    v = np.linspace(0, 1, rows)
    uu, vv = np.meshgrid(u, v)
    x = (uu - 0.5) * width
    side = np.abs(2 * uu - 1)
    y = 0.045 + 0.02 * np.sin(np.pi * np.clip(side, 0, 1)) ** 0.7 * (1 - 0.6 * side ** 3) - 0.035 * (1 - side) ** 6
    z = (vv - 0.5) * depth
    p = np.stack([x, y, z], -1).reshape(-1, 3)
    f = G._grid_faces(rows, cols)
    uv = np.stack([uu, vv], -1).reshape(-1, 2)
    pages = Mesh(p, G.vertex_normals(p, f), uv, f)
    if pages.n[:, 1].mean() < 0:
        pages = Mesh(pages.p, -pages.n, pages.uv, pages.f[:, ::-1])
    block = Mesh()
    for s in (-1, 1):
        block += G.box((width / 2 - 0.01, 0.036, depth - 0.006), (s * (width / 4 + 0.002), 0.026, 0))
    boards = Mesh()
    for s in (-1, 1):
        boards += G.box((width / 2 + 0.012, 0.008, depth + 0.02), (s * (width / 4 + 0.004), 0.004, 0))
    cradle = Mesh()
    for s in (-1, 1):
        wedge = G.box((width / 2, 0.02, depth * 0.8), (0, 0, 0)).transformed(
            compose(translate((s * width / 4, -0.012, 0)), rotate((0, 0, 1), -s * 6)))
        cradle += wedge
    return {"pages": pages, "block": block, "boards": boards.transformed(translate((0, 0.005, 0))),
            "cradle": cradle.transformed(translate((0, 0.012, 0)))}


# ----------------------------------------------------------------------------
# Sculptures (local frame: base at y = 0, centred on the y axis)
# ----------------------------------------------------------------------------

def bird(height: float = 1.45) -> dict[str, Mesh]:
    """A tall polished-bronze form in the manner of Brancusi's 'Bird in Space', on a stone drum."""
    s = height / 1.45
    prof = [(0.0, 0.0), (0.03, 0.0), (0.028, 0.03), (0.012, 0.09), (0.011, 0.16), (0.016, 0.26), (0.032, 0.42),
            (0.052, 0.62), (0.064, 0.8), (0.066, 0.95), (0.058, 1.1), (0.044, 1.24), (0.028, 1.36), (0.014, 1.43),
            (0.0, 1.45)]
    body = lathe([(r * s, y * s) for r, y in prof], 64).transformed(compose(translate((0, 0.3, 0)), rotate((0, 0, 1), 2.0)))
    drum = lathe([(0.0, 0.0), (0.13, 0.0), (0.13, 0.24), (0.06, 0.26), (0.06, 0.3), (0.0, 0.3)], 48)
    return {"bronze": body, "stone": drum}


def trefoil(size: float = 0.42, radius: float = 0.032) -> Mesh:
    t = np.linspace(0, TAU, 361)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    pts = np.column_stack([x, y, z]) * (size / 6.0)
    pts[:, 1] += size / 2 + radius + 0.01 - pts[:, 1].min() - size / 2
    m = sweep(pts[:-1], radius, 24, closed=True)
    return m


def muse(length: float = 0.3) -> Mesh:
    """An ovoid head lying on its side ('Sleeping Muse'): an egg with a soft brow and nose ridge."""
    m = ellipsoid((0.5, 0.62, 0.5), 48, 96)
    p = m.p.copy()
    p[:, 1] *= 1.0 + 0.12 * np.clip(p[:, 1] / 0.62, 0, 1)
    face = np.exp(-((np.arctan2(p[:, 0], p[:, 2])) / 0.22) ** 2) * (p[:, 2] > 0)
    p[:, 2] += 0.035 * face * np.exp(-((p[:, 1] + 0.05) / 0.18) ** 2)
    p[:, 2] += 0.02 * np.exp(-((np.abs(np.arctan2(p[:, 0], p[:, 2])) - 0.35) / 0.12) ** 2) * np.exp(-((p[:, 1] - 0.18) / 0.06) ** 2)
    m = Mesh(p, G.vertex_normals(p, m.f), m.uv, m.f).oriented()
    k = length / 1.24
    return m.transformed(compose(translate((0, 0.5 * k * 0.92, 0)), rotate((0, 0, 1), 90), rotate((0, 1, 0), 25), scale(k)))


def concretion(size: float = 0.5, seed: int = 3) -> Mesh:
    """A smooth organic form in the manner of Arp: a sphere pushed out into lobes."""
    rng = np.random.default_rng(seed)
    m = ellipsoid((1.0, 1.0, 1.0), 64, 128)
    u = m.p / np.linalg.norm(m.p, axis=1, keepdims=True)
    r = np.ones(len(u))
    for _ in range(5):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        r += rng.uniform(0.18, 0.4) * np.clip(u @ d, 0, 1) ** rng.uniform(2.5, 5.0)
    r -= 0.2 * np.clip(-u[:, 1], 0, 1) ** 3
    p = u * r[:, None]
    p[:, 1] -= p[:, 1].min()
    p *= size / (p[:, 1].max() + 1e-9) * 0.8
    return Mesh(p, G.vertex_normals(p, m.f), m.uv, m.f).oriented()


def urn(height: float = 0.75) -> dict[str, Mesh]:
    s = height / 0.75
    prof = [(0.0, 0.0), (0.13, 0.0), (0.13, 0.04), (0.1, 0.05), (0.06, 0.09), (0.055, 0.14), (0.09, 0.17),
            (0.18, 0.27), (0.21, 0.36), (0.2, 0.45), (0.16, 0.52), (0.12, 0.56), (0.13, 0.6), (0.16, 0.63),
            (0.17, 0.65), (0.15, 0.66), (0.13, 0.66), (0.12, 0.69), (0.07, 0.71), (0.04, 0.73), (0.045, 0.75), (0.0, 0.75)]
    body = lathe([(r * s, y * s) for r, y in prof], 96)
    handles = Mesh()
    for side in (-1.0, 1.0):
        path = np.array([(0.15, 0.6, 0), (0.24, 0.62, 0), (0.27, 0.55, 0), (0.22, 0.48, 0), (0.19, 0.45, 0)]) * s
        path[:, 0] *= side
        handles += sweep(_spline(path, 10), 0.014 * s, 12)
    return {"body": join(body, handles)}


# ----------------------------------------------------------------------------
# Textures: floors, walls, materials
# ----------------------------------------------------------------------------

def limestone(h: int, w: int, tile_px: int, seed: int = 21):
    """Polished limestone slabs with thin grout; returns (albedo, roughness)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    ti, tj = y // tile_px, x // tile_px
    tone = rng.normal(0, 1, (h // tile_px + 1, w // tile_px + 1))[ti, tj]
    warm = rng.normal(0, 1, (h // tile_px + 1, w // tile_px + 1))[ti, tj]
    n1 = G.fbm(h, w, 6, 6, seed=seed)
    n2 = G.fbm(h, w, 3, 120, seed=seed + 1)
    fossils = (G.fbm(h, w, 2, 240, seed=seed + 2) > 0.8).astype(np.float32) * 0.04
    base = np.array([0.66, 0.62, 0.55], np.float32)
    rgb = base * (1 + 0.05 * tone[..., None]) * (0.92 + 0.12 * n1[..., None]) * (0.97 + 0.06 * n2[..., None])
    rgb[..., 0] *= 1 + 0.02 * warm
    rgb -= fossils[..., None]
    gx = np.minimum(x % tile_px, tile_px - 1 - x % tile_px)
    gy = np.minimum(y % tile_px, tile_px - 1 - y % tile_px)
    grout = np.minimum(gx, gy) < 1.5
    rgb[grout] = [0.42, 0.40, 0.36]
    # Satin polish (roughness ~0.3, alpha ~0.09): a near-mirror floor turns every track spot into glossy caustics.
    rough = np.clip(0.27 + 0.08 * n2 + 0.05 * (1 - n1) + 0.5 * grout, 0.2, 0.9)
    return np.clip(rgb, 0, 1), rough


def fabric(h: int, w: int, colour, seed: int = 31) -> np.ndarray:
    """Fine linen weave for case decks."""
    y, x = np.mgrid[0:h, 0:w]
    weave = 0.5 + 0.25 * np.sin(x * 1.9) * np.sin(y * 0.35) + 0.25 * np.sin(y * 1.9) * np.sin(x * 0.35)
    slub = G.fbm(h, w, 3, 40, seed=seed)
    return np.clip(np.asarray(colour, np.float32) * (0.85 + 0.2 * weave[..., None]) * (0.92 + 0.12 * slub[..., None]), 0, 1)


def fossil_stone(h: int, w: int, seed: int = 41) -> np.ndarray:
    n = G.fbm(h, w, 6, 4, seed=seed)
    fine = G.fbm(h, w, 3, 90, seed=seed + 1)
    base = np.array([0.46, 0.38, 0.28], np.float32)
    return np.clip(base * (0.75 + 0.45 * n[..., None]) * (0.9 + 0.2 * fine[..., None]), 0, 1)


def patina_mask(h: int, w: int, seed: int = 51) -> np.ndarray:
    n = G.fbm(h, w, 6, 5, seed=seed)
    return np.clip((n - 0.45) * 3.0, 0, 1)


def bronze_albedo(h: int, w: int, seed: int = 52) -> np.ndarray:
    n = G.fbm(h, w, 5, 8, seed=seed)
    return np.clip(np.array([0.58, 0.40, 0.22], np.float32) * (0.8 + 0.3 * n[..., None]), 0, 1)


def black_figure(h: int, w: int, seed: int = 61) -> np.ndarray:
    """Attic black-figure decoration in lathe uv (u around, v = 0 at the foot). Mitsuba reads row 0 at v = 0,
    so row 0 here is the foot and the PNG looks upside down."""
    rng = np.random.default_rng(seed)
    clay = np.array([0.70, 0.36, 0.17], np.float32)
    glaze = np.array([0.035, 0.03, 0.028], np.float32)
    v = (np.arange(h) + 0.5)[:, None] / h
    u = (np.arange(w) + 0.5)[None, :] / w
    black = np.zeros((h, w), bool)
    black |= v < 0.07
    black |= (v > 0.6) & (v < 0.62)
    black |= (v > 0.52) & (v < 0.6) & (np.abs(((u * 40) % 1.0) - 0.5) * 2 < 0.55 * (0.6 - v) / 0.08)   # tongues
    black |= v > 0.72
    # meander band
    mv = (v - 0.47) / 0.04
    mu = (u * 48) % 1.0
    band = (mv > 0) & (mv < 1)
    key = band & ((mv < 0.18) | (mv > 0.82) | ((mu < 0.18) & (mv < 0.7)) | ((mu > 0.5) & (mu < 0.68) & (mv > 0.3)))
    black |= key
    # figure frieze: striding silhouettes
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    fy0, fy1 = int((1 - 0.44) * h), int((1 - 0.2) * h)   # drawn upright, flipped below
    for k in range(6):
        cx = (k + 0.5) / 6 * w + rng.uniform(-0.02, 0.02) * w
        fh = fy1 - fy0
        head = fh * 0.09
        d.ellipse([cx - head, fy0 + fh * 0.05, cx + head, fy0 + fh * 0.05 + 2 * head], fill=255)
        d.polygon([(cx - fh * 0.07, fy0 + fh * 0.25), (cx + fh * 0.08, fy0 + fh * 0.25), (cx + fh * 0.05, fy0 + fh * 0.62),
                   (cx - fh * 0.05, fy0 + fh * 0.62)], fill=255)
        stride = rng.uniform(0.08, 0.16) * fh
        d.line([(cx, fy0 + fh * 0.6), (cx - stride, fy1)], fill=255, width=max(3, int(fh * 0.05)))
        d.line([(cx, fy0 + fh * 0.6), (cx + stride, fy1)], fill=255, width=max(3, int(fh * 0.05)))
        d.line([(cx, fy0 + fh * 0.3), (cx + rng.choice([-1, 1]) * fh * 0.22, fy0 + fh * 0.15)], fill=255, width=max(2, int(fh * 0.035)))
        if rng.random() < 0.6:
            d.line([(cx + fh * 0.1, fy0), (cx + fh * 0.1, fy1)], fill=255, width=max(2, int(fh * 0.015)))   # spear
    figs = np.flipud(np.asarray(img, np.float32) / 255.0 > 0.5)
    black |= figs
    wear = G.fbm(h, w, 4, 12, seed=seed + 1)
    rgb = np.where(black[..., None], glaze * (0.8 + 0.5 * wear[..., None]), clay * (0.88 + 0.2 * wear[..., None]))
    chips = (G.fbm(h, w, 2, 160, seed=seed + 2) > 0.86) & black
    rgb[chips] = clay * 0.8
    return np.clip(rgb, 0, 1)


def coin_face(size: int = 256, seed: int = 71) -> np.ndarray:
    """Albedo with a profile head and a lettered rim, read as relief once on a metallic BSDF."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 150)
    d = ImageDraw.Draw(img)
    c = size / 2
    d.ellipse([4, 4, size - 4, size - 4], outline=230, width=max(2, size // 40))
    d.ellipse([c - size * 0.2, c - size * 0.26, c + size * 0.14, c + size * 0.16], fill=205)
    d.polygon([(c + 0.12 * size, c - 0.02 * size), (c + 0.2 * size, c + 0.04 * size), (c + 0.12 * size, c + 0.07 * size)], fill=205)
    d.rectangle([c - 0.1 * size, c + 0.12 * size, c + 0.06 * size, c + 0.3 * size], fill=195)
    for k in range(26):
        a = TAU * k / 26 + 0.3
        r = size * 0.4
        x, y = c + r * math.cos(a), c + r * math.sin(a)
        d.rectangle([x - 2, y - 4, x + 2, y + 4], fill=int(rng.uniform(200, 235)))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    g = np.asarray(img, np.float32) / 255.0
    wear = G.fbm(size, size, 3, 10, seed=seed)
    return np.clip(g * (0.85 + 0.3 * wear), 0, 1)


def deck_cloth(h: int, w: int, colour, seed: int = 81) -> np.ndarray:
    return fabric(h, w, colour, seed)


# ----------------------------------------------------------------------------
# Paintings
# ----------------------------------------------------------------------------

def _grid(h: int, w: int):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x / w, y / h


def _lerp(a, b, t):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    t = np.asarray(t, np.float32)
    return a * (1 - t[..., None]) + b * t[..., None]


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _ridge(w: int, rng, base: float = 3.0, octaves: int = 5) -> np.ndarray:
    x = np.linspace(0, 1, w)
    out, a = np.zeros(w), 1.0
    for o in range(octaves):
        f = base * 2 ** o
        out += a * np.sin(TAU * f * x + rng.uniform(0, TAU)) * 0.5 + a * 0.5 * np.sin(TAU * f * 1.37 * x + rng.uniform(0, TAU))
        a *= 0.5
    return (out - out.min()) / max(out.max() - out.min(), 1e-6)


def brushwork(h: int, w: int, seed: int) -> np.ndarray:
    """Painterly streaks: horizontal and vertical stroke noise blended by a slow mask, centred at 0."""
    rng = np.random.default_rng(seed)
    hs = np.asarray(Image.fromarray(rng.random((max(2, h // 5), max(2, w // 40))).astype(np.float32), mode="F")
                    .resize((w, h), Image.BICUBIC))
    vs = np.asarray(Image.fromarray(rng.random((max(2, h // 40), max(2, w // 5))).astype(np.float32), mode="F")
                    .resize((w, h), Image.BICUBIC))
    mix = G.fbm(h, w, 3, 3, seed=seed + 1)
    fine = G.fbm(h, w, 2, 150, seed=seed + 2)
    return (mix * hs + (1 - mix) * vs - 0.5) * 0.8 + (fine - 0.5) * 0.3


def age(rgb: np.ndarray, seed: int, varnish: float = 1.0, cracks: float = 1.0) -> np.ndarray:
    """Yellowed varnish, craquelure and a darker rebate edge, for older pictures."""
    h, w = rgb.shape[:2]
    out = rgb * np.array([1.0, 1.0 - 0.05 * varnish, 1.0 - 0.2 * varnish], np.float32)
    if cracks > 0:
        n = G.fbm(h, w, 4, 20, seed=seed + 7)
        m = G.fbm(h, w, 4, 47, seed=seed + 8)
        crack = np.exp(-((n - 0.5) / 0.006) ** 2) + 0.6 * np.exp(-((m - 0.5) / 0.005) ** 2)
        out = out * (1 - 0.35 * cracks * np.clip(crack, 0, 1)[..., None])
    x, y = _grid(h, w)
    edge = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    out *= (0.82 + 0.18 * _smooth(0.0, 0.03, edge))[..., None]
    return np.clip(out, 0, 1)


def _sky(h, w, hz, rng, warm: float):
    x, y = _grid(h, w)
    zen = np.array([0.24, 0.38, 0.62]) * (1 - warm) + np.array([0.42, 0.42, 0.55]) * warm
    hor = np.array([0.86, 0.82, 0.70]) * (1 - warm) + np.array([0.96, 0.74, 0.46]) * warm
    sky = _lerp(zen, hor, np.clip(y / hz, 0, 1) ** 0.9)
    cl = G.fbm(h, w, 6, 3, seed=int(rng.integers(1 << 30)), aspect=2.5 * w / h)
    c = _smooth(0.5, 0.72, cl) * np.clip(1.2 - y / hz, 0, 1)
    under = _smooth(0.6, 0.85, G.fbm(h, w, 4, 3, seed=int(rng.integers(1 << 30)), aspect=2.5 * w / h))
    cloud_col = _lerp(np.array([0.95, 0.93, 0.88]) * (1 - 0.15 * warm) + np.array([0.15, 0.05, -0.05]) * warm,
                      np.array([0.55, 0.55, 0.6]), under * 0.7)
    return sky * (1 - c[..., None]) + cloud_col * c[..., None]


def landscape(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = _grid(h, w)
    hz = float(rng.uniform(0.42, 0.58))
    warm = float(rng.uniform(0, 0.8))
    img = _sky(h, w, hz, rng, warm)
    haze = _lerp([0.86, 0.82, 0.7], [0.96, 0.74, 0.46], np.full((1, 1), warm))[0, 0]
    layers = [(hz - 0.06, 0.12, (0.42, 0.48, 0.56), 0.75), (hz - 0.01, 0.10, (0.3, 0.38, 0.32), 0.45),
              (hz + 0.06, 0.08, (0.24, 0.32, 0.16), 0.2), (hz + 0.16, 0.06, (0.3, 0.3, 0.12), 0.05)]
    ridges = []
    for k, (base_y, amp, col, mist) in enumerate(layers):
        ridge = base_y - amp * _ridge(w, rng, base=2 + k, octaves=5)
        ridges.append(ridge)
        mask = y > ridge[None, :]
        tex = G.fbm(h, w, 5, 6 + 4 * k, seed=int(rng.integers(1 << 30)))
        col = np.asarray(col, np.float32) * (0.7 + 0.5 * tex[..., None])
        col = col * (1 - mist) + haze * mist
        shade = np.clip((y - ridge[None, :]) / 0.15, 0, 1)
        col = col * (1 - 0.25 * shade[..., None])
        img = np.where(mask[..., None], col, img)
    # A river winding down from the horizon.
    yc = np.clip((y - hz) / (1 - hz), 0, 1)
    xc = 0.5 + 0.18 * np.sin(yc * 5 + rng.uniform(0, TAU)) * (0.3 + yc)
    width = 0.01 + 0.12 * yc ** 1.6
    river = (np.abs(x - xc) < width) & (y > hz + 0.02)
    sky_ref = _lerp([0.5, 0.6, 0.72], [0.8, 0.76, 0.66], yc)
    img = np.where(river[..., None], sky_ref * (0.85 + 0.15 * G.fbm(h, w, 3, 30, seed=seed + 9)[..., None]), img)
    # Trees: clumps of foliage along the middle ground.
    n_trees = int(rng.integers(5, 11))
    layer = Image.new("L", (w, h), 0)
    dd = ImageDraw.Draw(layer)
    trunks = Image.new("L", (w, h), 0)
    td = ImageDraw.Draw(trunks)
    for _ in range(n_trees):
        cx = float(rng.uniform(0.05, 0.95))
        gy = float(ridges[2][int(cx * (w - 1))]) + rng.uniform(0.0, 0.08)
        th = float(rng.uniform(0.12, 0.3)) * (0.6 + gy)
        tw = th * rng.uniform(0.35, 0.6)
        td.line([(cx * w, gy * h), (cx * w, (gy - th * 0.45) * h)], fill=255, width=max(2, int(w * 0.006)))
        for _ in range(9):
            bx = cx + rng.normal(0, tw * 0.28) * h / w
            by = gy - th * rng.uniform(0.45, 1.0)
            r = tw * rng.uniform(0.18, 0.32)
            dd.ellipse([(bx - r * h / w) * w, (by - r) * h, (bx + r * h / w) * w, (by + r) * h], fill=255)
    fol = np.asarray(layer.filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255.0
    edge_noise = G.fbm(h, w, 3, 60, seed=seed + 11)
    fol = _smooth(0.35, 0.6, fol * (0.75 + 0.5 * edge_noise))
    lit = np.clip(1.0 - 1.5 * (G.fbm(h, w, 4, 20, seed=seed + 12) - 0.3), 0.3, 1.2)
    fcol = np.array([0.12, 0.17, 0.07], np.float32) * lit[..., None] * (1 + 0.4 * warm)
    tr = np.asarray(trunks, np.float32)[..., None] / 255.0
    img = img * (1 - tr) + np.array([0.12, 0.08, 0.05]) * tr
    img = img * (1 - fol[..., None]) + fcol * fol[..., None]
    img *= (1 + 0.12 * brushwork(h, w, seed + 13))[..., None]
    return age(np.clip(img, 0, 1), seed, varnish=0.8, cracks=0.8)


def seascape(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = _grid(h, w)
    hz = float(rng.uniform(0.5, 0.62))
    warm = float(rng.uniform(0.2, 0.9))
    img = _sky(h, w, hz, rng, warm)
    sun_x = float(rng.uniform(0.25, 0.75))
    glow = np.exp(-((x - sun_x) ** 2 * 6 + (y - hz + 0.05) ** 2 * 30))
    img += (np.array([0.5, 0.35, 0.15]) * warm * glow[..., None])
    sea_t = np.clip((y - hz) / (1 - hz), 0, 1)
    sea = _lerp([0.32, 0.38, 0.4], [0.08, 0.14, 0.15], sea_t)
    waves = np.asarray(Image.fromarray(rng.random((h // 3, max(2, w // 50))).astype(np.float32), mode="F").resize((w, h), Image.BICUBIC))
    crest = _smooth(0.72, 0.9, waves) * (0.2 + sea_t)
    sea = sea * (0.85 + 0.3 * waves[..., None]) + crest[..., None] * np.array([0.6, 0.62, 0.6])
    path = np.exp(-((x - sun_x) / (0.04 + 0.2 * sea_t)) ** 2) * warm
    sea += path[..., None] * np.array([0.5, 0.38, 0.2]) * (0.4 + 0.6 * waves[..., None])
    img = np.where((y > hz)[..., None], sea, img)
    # A sailing ship near the horizon.
    sh = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(sh)
    sx = float(rng.uniform(0.15, 0.85)) * w
    base = hz * h + 0.01 * h
    L = 0.16 * w
    sd.polygon([(sx - L / 2, base - 0.025 * h), (sx + L / 2, base - 0.03 * h), (sx + L * 0.4, base + 0.01 * h), (sx - L * 0.42, base + 0.012 * h)], fill=255)
    sails = Image.new("L", (w, h), 0)
    sa = ImageDraw.Draw(sails)
    for k, mx in enumerate((-0.28, 0.0, 0.26)):
        mh = (0.2 - 0.03 * abs(k - 1)) * h
        px = sx + mx * L
        sd.line([(px, base - 0.025 * h), (px, base - 0.025 * h - mh)], fill=255, width=max(2, w // 300))
        for j in range(3):
            top = base - 0.025 * h - mh * (0.95 - j * 0.3)
            sa.polygon([(px - 0.07 * L, top), (px + 0.07 * L, top), (px + 0.08 * L, top + mh * 0.24), (px - 0.08 * L, top + mh * 0.24)], fill=255)
    hull = np.asarray(sh, np.float32)[..., None] / 255.0
    sail = np.asarray(sails.filter(ImageFilter.GaussianBlur(0.8)), np.float32)[..., None] / 255.0
    img = img * (1 - sail) + np.array([0.82, 0.78, 0.66]) * sail
    img = img * (1 - hull) + np.array([0.1, 0.07, 0.05]) * hull
    img *= (1 + 0.14 * brushwork(h, w, seed + 13))[..., None]
    return age(np.clip(img, 0, 1), seed, varnish=0.7, cracks=0.7)


def _shaded_disc(x, y, cx, cy, rx, ry, light=(-0.5, -0.6, 0.62)):
    nx, ny = (x - cx) / rx, (y - cy) / ry
    r2 = nx ** 2 + ny ** 2
    inside = r2 < 1
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    l = np.asarray(light) / np.linalg.norm(light)
    lam = np.clip(nx * l[0] + ny * l[1] + nz * l[2], 0, 1)
    spec = np.clip(nx * l[0] + ny * l[1] + nz * l[2], 0, 1) ** 40
    edge = _smooth(1.0, 0.92, np.sqrt(r2))
    return inside, lam, spec, edge


def portrait(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = _grid(h, w)
    a = w / h
    xs = (x - 0.5) * a + 0.5
    cx, cy = 0.5 + rng.uniform(-0.06, 0.06), 0.36 + rng.uniform(-0.03, 0.03)
    bg = _lerp([0.16, 0.12, 0.08], [0.04, 0.03, 0.02], np.clip(np.hypot(xs - cx, y - cy) * 1.6, 0, 1))
    img = bg * (1 + 0.15 * brushwork(h, w, seed)[..., None])
    cloth = [np.array(c) for c in ((0.05, 0.05, 0.06), (0.25, 0.05, 0.05), (0.06, 0.12, 0.08), (0.12, 0.1, 0.18))][int(rng.integers(0, 4))]
    body_in, lam, _, edge = _shaded_disc(xs, y, cx + 0.02, 1.02, 0.42, 0.42)
    img = np.where(body_in[..., None], cloth * (0.5 + 0.9 * lam[..., None]), img)
    collar_in = (np.abs(np.hypot((xs - cx) / 0.17, (y - 0.66) / 0.07) - 1) < 0.25) & (y > 0.6)
    img = np.where(collar_in[..., None], np.array([0.78, 0.76, 0.7]) * (0.6 + 0.5 * lam[..., None]), img)
    skin = np.array([0.82, 0.6, 0.47]) * rng.uniform(0.8, 1.05)
    neck = (np.abs(xs - cx) < 0.06) & (y > cy + 0.1) & (y < 0.66)
    img = np.where(neck[..., None], skin * 0.55, img)
    head_in, lam, spec, edge = _shaded_disc(xs, y, cx, cy, 0.125, 0.165)
    face = skin * (0.25 + 0.95 * lam[..., None]) + spec[..., None] * 0.15
    blush = np.exp(-(((xs - cx - 0.05) / 0.04) ** 2 + ((y - cy - 0.04) / 0.03) ** 2))
    face = face + blush[..., None] * np.array([0.08, -0.02, -0.02])
    for ex in (-0.045, 0.045):
        eye = np.exp(-(((xs - cx - ex) / 0.018) ** 2 + ((y - cy + 0.01) / 0.008) ** 2))
        face = face * (1 - 0.6 * eye[..., None])
    mouth = np.exp(-(((xs - cx) / 0.03) ** 2 + ((y - cy - 0.085) / 0.006) ** 2))
    face = face * (1 - 0.35 * mouth[..., None]) + mouth[..., None] * np.array([0.08, 0.0, 0.0])
    nose = np.exp(-(((xs - cx + 0.012) / 0.01) ** 2 + ((y - cy - 0.035) / 0.03) ** 2))
    face = face * (1 - 0.25 * nose[..., None])
    img = np.where(head_in[..., None], face, img)
    hair_col = np.array([(0.08, 0.05, 0.03), (0.2, 0.12, 0.06), (0.03, 0.03, 0.03), (0.4, 0.34, 0.26)][int(rng.integers(0, 4))])
    hair_in, hl, _, _ = _shaded_disc(xs, y, cx + 0.01, cy - 0.04, 0.145, 0.17)
    hair = hair_in & ~(head_in & (y > cy - 0.09)) & (y < cy + 0.06)
    hat = rng.random() < 0.45
    if hat:
        brim = (np.abs((xs - cx) / 0.22) ** 2 + ((y - cy + 0.13) / 0.035) ** 2 < 1)
        crown = (np.abs(xs - cx) < 0.12) & (y > cy - 0.28) & (y < cy - 0.12)
        hair = hair | brim | crown
        hair_col = np.array([0.03, 0.03, 0.03])
    img = np.where(hair[..., None], hair_col * (0.5 + 0.8 * hl[..., None]), img)
    img = np.clip(np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(h / 700)),
                             np.float32) / 255.0, 0, 1)
    img *= (1 + 0.1 * brushwork(h, w, seed + 3))[..., None]
    return age(np.clip(img, 0, 1), seed, varnish=1.0, cracks=1.0)


def still_life(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = _grid(h, w)
    a = w / h
    xs = x * a
    img = _lerp([0.12, 0.1, 0.06], [0.03, 0.025, 0.02], np.clip(np.hypot(x - 0.3, y - 0.3), 0, 1))
    ty = float(rng.uniform(0.6, 0.7))
    table = y > ty
    wood = _lerp([0.28, 0.16, 0.08], [0.12, 0.07, 0.03], np.clip((y - ty) * 3, 0, 1)) * (0.8 + 0.3 * G.fbm(h, w, 4, 8, seed=seed)[..., None])
    img = np.where(table[..., None], wood, img)
    edge = (y > ty) & (y < ty + 0.012)
    img = np.where(edge[..., None], np.array([0.42, 0.28, 0.15]), img)
    # cloth drape on the left
    cl = (xs < a * rng.uniform(0.35, 0.5) + 0.06 * np.sin(y * 18)) & (y > ty - 0.05)
    folds = 0.75 + 0.25 * np.sin(xs * 40 + 6 * G.fbm(h, w, 3, 4, seed=seed + 1))
    img = np.where(cl[..., None], np.array([0.85, 0.83, 0.78]) * folds[..., None] * _lerp([1, 1, 1], [0.6, 0.6, 0.62], np.clip((y - ty) * 2, 0, 1)), img)
    # a pewter jug
    jx = a * rng.uniform(0.25, 0.4)
    jh = 0.32
    yy = np.clip((ty - y) / jh, 0, 1)
    prof = 0.07 + 0.06 * np.sin(np.pi * np.clip(yy * 1.2, 0, 1)) - 0.03 * (yy > 0.8)
    jug_in = (np.abs(xs - jx) < prof) & (y < ty) & (y > ty - jh)
    nx = (xs - jx) / np.maximum(prof, 1e-3)
    lam = np.clip(0.4 - 0.6 * nx + 0.3 * np.sqrt(np.clip(1 - nx ** 2, 0, 1)), 0.05, 1)
    sp = np.exp(-((nx + 0.45) / 0.08) ** 2)
    img = np.where(jug_in[..., None], np.array([0.42, 0.42, 0.4]) * lam[..., None] + sp[..., None] * 0.6, img)
    # fruit
    fruits = []
    for _ in range(int(rng.integers(4, 8))):
        kind = int(rng.integers(0, 3))
        fx = a * rng.uniform(0.35, 0.9)
        r = rng.uniform(0.045, 0.07)
        fy = ty - r * rng.uniform(0.6, 0.9)
        fruits.append((fy, fx, r, kind))
    for fy, fx, r, kind in sorted(fruits):
        col = [np.array([0.62, 0.08, 0.05]), np.array([0.85, 0.7, 0.12]), np.array([0.3, 0.42, 0.1])][kind]
        inside, lam, spec, e = _shaded_disc(xs, y, fx, fy, r * (1.25 if kind == 1 else 1.0), r)
        shadow = np.exp(-(((xs - fx - 0.03) / (r * 1.5)) ** 2 + ((y - ty - 0.01) / 0.015) ** 2)) * table
        img = img * (1 - 0.5 * shadow[..., None])
        img = np.where(inside[..., None], col * (0.15 + 0.9 * lam[..., None]) + spec[..., None] * 0.5, img)
    for _ in range(int(rng.integers(10, 18))):
        gx, gy, gr = a * rng.uniform(0.55, 0.75), ty - rng.uniform(0.02, 0.12), 0.022
        inside, lam, spec, e = _shaded_disc(xs, y, gx, gy, gr, gr)
        img = np.where(inside[..., None], np.array([0.22, 0.06, 0.2]) * (0.15 + lam[..., None]) + spec[..., None] * 0.4, img)
    img *= (1 + 0.12 * brushwork(h, w, seed + 3))[..., None]
    return age(np.clip(img, 0, 1), seed, varnish=0.9, cracks=0.9)


def colour_field(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = _grid(h, w)
    pal = [((0.55, 0.08, 0.06), (0.85, 0.35, 0.08), (0.2, 0.05, 0.08)), ((0.1, 0.18, 0.3), (0.06, 0.08, 0.12), (0.45, 0.5, 0.55)),
           ((0.7, 0.55, 0.15), (0.85, 0.75, 0.4), (0.45, 0.18, 0.08)), ((0.22, 0.3, 0.2), (0.08, 0.1, 0.08), (0.6, 0.62, 0.5))][int(rng.integers(0, 4))]
    img = np.tile(np.asarray(pal[0], np.float32), (h, w, 1))
    bands = sorted(rng.uniform(0.08, 0.92, 4))
    for k, (y0, y1) in enumerate(((0.07, bands[1]), (bands[2], 0.93))):
        n = G.fbm(h, w, 4, 6, seed=seed + k)
        d = np.minimum.reduce([x - 0.07, 0.93 - x, y - y0, y1 - y]) + 0.02 * (n - 0.5)
        m = _smooth(-0.005, 0.02, d)
        img = img * (1 - m[..., None]) + np.asarray(pal[k + 1], np.float32) * (0.9 + 0.2 * n[..., None]) * m[..., None]
    img *= (1 + 0.08 * brushwork(h, w, seed + 5))[..., None]
    return np.clip(img, 0, 1)


def geometric(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 0.9, np.float32)
    xs = np.sort(rng.uniform(0.1, 0.9, int(rng.integers(2, 5))))
    ys = np.sort(rng.uniform(0.1, 0.9, int(rng.integers(2, 5))))
    xb, yb = np.concatenate([[0], xs, [1]]), np.concatenate([[0], ys, [1]])
    cols = [(0.78, 0.1, 0.08), (0.08, 0.2, 0.55), (0.92, 0.78, 0.1), (0.9, 0.9, 0.88), (0.9, 0.9, 0.88)]
    for i in range(len(xb) - 1):
        for j in range(len(yb) - 1):
            if rng.random() < 0.35:
                c = cols[int(rng.integers(0, 3))]
                img[int(yb[j] * h):int(yb[j + 1] * h), int(xb[i] * w):int(xb[i + 1] * w)] = c
    t = max(3, w // 60)
    for xv in xs:
        img[:, int(xv * w) - t // 2:int(xv * w) + t // 2] = 0.03
    for yv in ys:
        img[int(yv * h) - t // 2:int(yv * h) + t // 2, :] = 0.03
    img *= (1 + 0.05 * brushwork(h, w, seed + 5))[..., None]
    return np.clip(img, 0, 1)


def gestural(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pal = [(0.85, 0.2, 0.1), (0.1, 0.12, 0.2), (0.95, 0.75, 0.15), (0.2, 0.45, 0.6), (0.05, 0.05, 0.05)]
    img = Image.new("RGB", (w, h), (232, 226, 210))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(18, 34))):
        c = tuple(int(255 * v) for v in pal[int(rng.integers(0, len(pal)))])
        pts = np.cumsum(rng.normal(0, 0.12, (5, 2)), 0) + rng.uniform(0.1, 0.9, 2)
        pts = _spline(np.column_stack([pts, np.zeros(5)]), 12)[:, :2]
        d.line([(float(px) * w, float(py) * h) for px, py in pts], fill=c, width=int(rng.uniform(0.008, 0.03) * w), joint="curve")
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(w / 600)), np.float32) / 255.0
    arr *= (1 + 0.1 * brushwork(h, w, seed + 5))[..., None]
    return np.clip(arr, 0, 1)


PAINTERS = {"landscape": landscape, "seascape": seascape, "portrait": portrait, "still_life": still_life,
            "colour_field": colour_field, "geometric": geometric, "gestural": gestural,
            "blobs": lambda h, w, s: G.abstract_art(h, w, [(0.86, 0.8, 0.66), (0.7, 0.3, 0.15), (0.12, 0.24, 0.34), (0.88, 0.62, 0.2), (0.3, 0.42, 0.3)], seed=s)}
OLD = {"landscape", "seascape", "portrait", "still_life"}


# ----------------------------------------------------------------------------
# Lettering: stroke glyphs (no system fonts, so every machine draws the same pixels)
# ----------------------------------------------------------------------------

def _arc(cx, cy, rx, ry, a0, a1, n=12):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


def _glyphs():
    g = {}
    bowl = lambda c: _arc(c, 0.5, 0.38, 0.5, 0, 360, 18)
    g["o"] = (0.95, [bowl(0.5)])
    g["a"] = (0.95, [bowl(0.42), [(0.8, 1.0), (0.8, 0.0)]])
    g["d"] = (0.95, [bowl(0.42), [(0.8, 1.6), (0.8, 0.0)]])
    g["b"] = (0.95, [[(0.15, 1.6), (0.15, 0.0)], bowl(0.55)])
    g["p"] = (0.95, [[(0.15, 1.0), (0.15, -0.55)], bowl(0.55)])
    g["g"] = (0.95, [bowl(0.42), [(0.8, 1.0), (0.8, -0.2)] + _arc(0.45, -0.2, 0.35, 0.35, 0, -150, 7)])
    g["c"] = (0.85, [_arc(0.55, 0.5, 0.4, 0.5, 40, 320, 14)])
    g["e"] = (0.9, [_arc(0.5, 0.5, 0.4, 0.5, 15, 330, 16), [(0.1, 0.5), (0.9, 0.5)]])
    g["n"] = (0.9, [[(0.15, 1.0), (0.15, 0.0)], _arc(0.5, 0.58, 0.35, 0.42, 180, 0, 9) + [(0.85, 0.0)]])
    g["h"] = (0.9, [[(0.15, 1.6), (0.15, 0.0)], _arc(0.5, 0.58, 0.35, 0.42, 180, 0, 9) + [(0.85, 0.0)]])
    g["m"] = (1.45, [[(0.1, 1.0), (0.1, 0.0)], _arc(0.32, 0.6, 0.22, 0.4, 180, 0, 7) + [(0.54, 0.0)],
                     _arc(0.76, 0.6, 0.22, 0.4, 180, 0, 7) + [(0.98, 0.0)]])
    g["r"] = (0.7, [[(0.2, 1.0), (0.2, 0.0)], _arc(0.55, 0.62, 0.35, 0.38, 170, 40, 7)])
    g["u"] = (0.9, [[(0.15, 1.0), (0.15, 0.45)] + _arc(0.5, 0.45, 0.35, 0.45, 180, 360, 9) + [(0.85, 1.0)], [(0.85, 1.0), (0.85, 0.0)]])
    g["i"] = (0.45, [[(0.5, 0.78), (0.5, 0.0)], [(0.5, 1.22), (0.5, 1.3)]])
    g["l"] = (0.45, [[(0.5, 1.6), (0.5, 0.0)]])
    g["t"] = (0.6, [[(0.5, 1.35), (0.5, 0.15)] + _arc(0.75, 0.15, 0.25, 0.15, 180, 270, 5), [(0.2, 1.0), (0.85, 1.0)]])
    g["f"] = (0.6, [_arc(0.8, 1.35, 0.3, 0.25, 0, 180, 7) + [(0.5, 0.0)], [(0.2, 1.0), (0.85, 1.0)]])
    g["s"] = (0.8, [_arc(0.5, 0.76, 0.32, 0.24, 30, 270, 9) + _arc(0.5, 0.27, 0.34, 0.27, 90, -150, 9)])
    g["v"] = (0.9, [[(0.08, 1.0), (0.5, 0.0), (0.92, 1.0)]])
    g["w"] = (1.3, [[(0.05, 1.0), (0.28, 0.0), (0.5, 0.8), (0.72, 0.0), (0.95, 1.0)]])
    g["y"] = (0.9, [[(0.08, 1.0), (0.5, 0.05)], [(0.92, 1.0), (0.3, -0.55)]])
    g["k"] = (0.85, [[(0.15, 1.6), (0.15, 0.0)], [(0.85, 1.0), (0.15, 0.4)], [(0.35, 0.58), (0.85, 0.0)]])
    for ch, (adv, strokes) in {
        "T": (1.0, [[(0.05, 1.5), (0.95, 1.5)], [(0.5, 1.5), (0.5, 0.0)]]),
        "H": (1.1, [[(0.1, 1.5), (0.1, 0.0)], [(0.9, 1.5), (0.9, 0.0)], [(0.1, 0.75), (0.9, 0.75)]]),
        "E": (0.9, [[(0.85, 1.5), (0.1, 1.5), (0.1, 0.0), (0.85, 0.0)], [(0.1, 0.75), (0.7, 0.75)]]),
        "A": (1.1, [[(0.05, 0.0), (0.5, 1.5), (0.95, 0.0)], [(0.25, 0.55), (0.75, 0.55)]]),
        "S": (0.9, [_arc(0.5, 1.12, 0.38, 0.38, 20, 270, 10) + _arc(0.5, 0.4, 0.42, 0.4, 90, -160, 10)]),
        "R": (1.0, [[(0.1, 0.0), (0.1, 1.5)] + _arc(0.5, 1.12, 0.4, 0.38, 90, -90, 9) + [(0.1, 0.75)], [(0.45, 0.75), (0.9, 0.0)]]),
        "N": (1.1, [[(0.1, 0.0), (0.1, 1.5), (0.9, 0.0), (0.9, 1.5)]]),
        "C": (1.0, [_arc(0.55, 0.75, 0.45, 0.75, 40, 320, 16)]),
        "O": (1.05, [_arc(0.52, 0.75, 0.45, 0.75, 0, 360, 20)]),
        "I": (0.45, [[(0.5, 1.5), (0.5, 0.0)]]),
        "L": (0.85, [[(0.12, 1.5), (0.12, 0.0), (0.85, 0.0)]]),
        "M": (1.3, [[(0.08, 0.0), (0.08, 1.5), (0.5, 0.5), (0.92, 1.5), (0.92, 0.0)]]),
        "D": (1.05, [[(0.1, 0.0), (0.1, 1.5)] + _arc(0.45, 0.75, 0.45, 0.75, 90, -90, 12) + [(0.1, 0.0)]]),
        "U": (1.05, [[(0.12, 1.5), (0.12, 0.5)] + _arc(0.52, 0.5, 0.4, 0.5, 180, 360, 10) + [(0.92, 1.5)]]),
        "V": (1.05, [[(0.05, 1.5), (0.52, 0.0), (0.98, 1.5)]]),
        "Q": (1.05, [_arc(0.52, 0.75, 0.45, 0.75, 0, 360, 20), [(0.6, 0.3), (0.98, -0.1)]]),
        "0": (0.85, [_arc(0.5, 0.75, 0.4, 0.75, 0, 360, 18)]),
        "1": (0.6, [[(0.25, 1.2), (0.55, 1.5), (0.55, 0.0)]]),
        "2": (0.85, [_arc(0.5, 1.1, 0.38, 0.4, 160, -30, 9) + [(0.1, 0.0), (0.9, 0.0)]]),
        "3": (0.85, [_arc(0.5, 1.15, 0.36, 0.35, 150, -90, 9) + _arc(0.5, 0.4, 0.4, 0.4, 90, -150, 9)]),
        "4": (0.9, [[(0.7, 0.0), (0.7, 1.5), (0.05, 0.5), (0.95, 0.5)]]),
        "5": (0.85, [[(0.85, 1.5), (0.2, 1.5), (0.15, 0.85)] + _arc(0.5, 0.45, 0.4, 0.45, 110, -150, 10)]),
        "7": (0.85, [[(0.1, 1.5), (0.9, 1.5), (0.35, 0.0)]]),
        "8": (0.85, [_arc(0.5, 1.1, 0.33, 0.4, 0, 360, 12), _arc(0.5, 0.38, 0.4, 0.38, 0, 360, 12)]),
        "9": (0.85, [_arc(0.5, 1.05, 0.38, 0.45, 0, 360, 12), [(0.88, 1.05), (0.8, 0.0)]]),
        "-": (0.5, [[(0.15, 0.55), (0.85, 0.55)]]),
        ",": (0.35, [[(0.55, 0.05), (0.4, -0.25)]]),
        ".": (0.35, [[(0.5, 0.0), (0.5, 0.06)]]),
    }.items():
        g[ch] = (adv, strokes)
    return g


GLYPHS = _glyphs()
LOWER = "etaoinshrdlcumwfgypbvk"
LOWER_P = np.array([12.7, 9.1, 8.2, 7.5, 7.0, 6.7, 6.3, 6.1, 6.0, 4.3, 4.0, 2.8, 2.8, 2.4, 2.4, 2.2, 2.0, 2.0, 1.9, 1.5, 1.0, 0.8])
LOWER_P = LOWER_P / LOWER_P.sum()
CAPS = "TEHASRNCOILMDUV"
SS = 2


class Ink:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.mask = Image.new("L", (w * SS, h * SS), 0)
        self.draw = ImageDraw.Draw(self.mask)

    def line(self, pts, width_px):
        self.draw.line([(x * SS, y * SS) for x, y in pts], fill=255, width=max(1, int(round(width_px * SS))), joint="curve")

    def coverage(self) -> np.ndarray:
        return np.asarray(self.mask.reduce(SS), np.float32) / 255.0


def word_width(word: str, xh: float, track: float = 0.12) -> float:
    return sum(GLYPHS.get(c, (0.5, []))[0] * xh * (1 + track) for c in word)


def print_word(ink: Ink, x: float, y_base: float, word: str, xh: float, width: float, track: float = 0.12) -> float:
    for ch in word:
        if ch not in GLYPHS:
            x += xh * 0.5
            continue
        adv, strokes = GLYPHS[ch]
        for poly in strokes:
            ink.line([(x + gx * xh * adv * 0.9, y_base - gy * xh) for gx, gy in poly], width)
        x += adv * xh * (1 + track)
    return x


def random_word(rng, lo=2, hi=9) -> str:
    return "".join(rng.choice(list(LOWER), size=int(rng.integers(lo, hi + 1)), p=LOWER_P))


def title_words(rng, n: int) -> str:
    return " ".join(str(rng.choice(list(CAPS))) + random_word(rng, 2, 7) for _ in range(n))


def text_block(ink: Ink, box, pitch, xh, width, rng, ragged=True) -> float:
    x0, y0, x1, y1 = box
    y = y0 + pitch
    while y < y1:
        x = x0
        limit = x1 - (rng.uniform(0.0, 0.08) * (x1 - x0) if ragged else 0)
        while True:
            wd = random_word(rng)
            ww = word_width(wd, xh)
            if x + ww > limit:
                break
            print_word(ink, x, y, wd, xh, width)
            x += ww + xh * 0.55
        y += pitch
    return y


def label_texture(seed: int, h: int = 400, w: int = 640, brass: bool = False) -> np.ndarray:
    """A wall label: title, artist and date line, then a short paragraph."""
    rng = np.random.default_rng(seed)
    ink = Ink(w, h)
    m = 0.08 * w
    print_word(ink, m, 0.2 * h, title_words(rng, int(rng.integers(1, 3))), 0.06 * h, 0.012 * h)
    print_word(ink, m, 0.33 * h, title_words(rng, 2) + " " + str(int(rng.integers(1480, 1980))), 0.035 * h, 0.006 * h)
    text_block(ink, (m, 0.38 * h, w - m, 0.9 * h), 0.075 * h, 0.032 * h, 0.0055 * h, rng)
    cov = ink.coverage()[..., None]
    if brass:
        base = np.array([0.78, 0.6, 0.32], np.float32) * (0.92 + 0.08 * G.fbm(h, w, 3, 8, seed=seed)[..., None])
        return np.clip(base * (1 - 0.85 * cov), 0, 1)
    base = np.full((h, w, 3), 0.92, np.float32)
    return np.clip(base * (1 - cov) + np.array([0.05, 0.05, 0.06]) * cov, 0, 1)


def text_panel(seed: int, h: int = 1400, w: int = 1000, wall=(0.9, 0.89, 0.86)) -> np.ndarray:
    """An introductory wall text: a big title over two columns of body copy."""
    rng = np.random.default_rng(seed)
    big, body = Ink(w, h), Ink(w, h)
    print_word(big, 0.08 * w, 0.14 * h, "ANTIQUITIES", 0.055 * h, 0.012 * h, track=0.25)
    print_word(big, 0.08 * w, 0.21 * h, title_words(rng, 3), 0.022 * h, 0.004 * h)
    col = (w - 0.24 * w) / 2
    for c in range(2):
        x0 = 0.08 * w + c * (col + 0.08 * w)
        text_block(body, (x0, 0.26 * h, x0 + col, 0.92 * h), 0.032 * h, 0.012 * h, 0.0024 * h, rng)
    cov = np.clip(big.coverage() + body.coverage(), 0, 1)[..., None]
    base = np.asarray(wall, np.float32) * (0.97 + 0.03 * G.fbm(h, w, 3, 8, seed=seed)[..., None])
    return np.clip(base * (1 - cov) + np.array([0.06, 0.06, 0.07]) * cov, 0, 1)


def manuscript_spread(h: int, w: int, seed: int) -> np.ndarray:
    """Two parchment pages with two text columns each, red rubrics and an illuminated initial.
    Row 0 is the far edge (v = 0 in the page mesh's uv)."""
    rng = np.random.default_rng(seed)
    base = np.array([0.86, 0.78, 0.6], np.float32)
    stain = G.fbm(h, w, 5, 4, seed=seed)
    fibre = G.fbm(h, w, 2, 200, seed=seed + 1)
    img = base * (0.85 + 0.2 * stain[..., None]) * (0.96 + 0.06 * fibre[..., None])
    gutter = np.exp(-(((np.arange(w) - w / 2) / (0.03 * w)) ** 2))[None, :, None]
    img *= 1 - 0.35 * gutter
    black, red = Ink(w, h), Ink(w, h)
    pw = w / 2
    for page in range(2):
        x0 = page * pw + 0.12 * pw
        cw = (pw - 0.26 * pw) / 2
        for c in range(2):
            cx0 = x0 + c * (cw + 0.04 * pw)
            y = 0.1 * h
            while y < 0.86 * h:
                ink = red if rng.random() < 0.08 else black
                x = cx0
                while True:
                    wd = random_word(rng, 2, 7)
                    ww = word_width(wd, 0.012 * h)
                    if x + ww > cx0 + cw:
                        break
                    print_word(ink, x, y, wd, 0.012 * h, 0.0028 * h)
                    x += ww + 0.007 * h
                y += 0.026 * h
    img = img * (1 - 0.88 * black.coverage()[..., None]) + 0 * img
    rc = red.coverage()[..., None]
    img = img * (1 - rc) + np.array([0.6, 0.1, 0.06]) * rc
    # illuminated initial on the left page
    ix, iy, s = int(0.12 * pw), int(0.1 * h), int(0.1 * h)
    img[iy:iy + s, ix:ix + s] = [0.12, 0.2, 0.5]
    cov = Ink(w, h)
    print_word(cov, ix + 0.15 * s, iy + 0.85 * s, "D", 0.48 * s, 0.09 * s)
    gold = cov.coverage()[..., None]
    img = img * (1 - gold) + np.array([0.85, 0.65, 0.25]) * gold
    img[iy:iy + s, ix:ix + 3] = img[iy:iy + s, ix + s - 3:ix + s] = [0.8, 0.6, 0.2]
    return np.clip(img, 0, 1)
