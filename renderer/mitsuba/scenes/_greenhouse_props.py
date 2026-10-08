"""Geometry, textures and plant builders for scenes/greenhouse.py.

Everything is numpy meshes (procedural.Mesh) in meters, y up. Plant builders append
into an `Groups`, which merges every leaf (or stem, or petal of one colour) of the
whole scene into one mesh per material, so a scene with ~40k leaves is ~15 shapes.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import procedural as G
from props import arc, bezier

TAU = 2 * math.pi
UP = np.array([0.0, 1.0, 0.0])

# --------------------------------------------------------------------------
# Cross-section of the nave: dwarf wall, vertical glazing, then a circular arch.
# --------------------------------------------------------------------------
HALF_W, Y_WALL, Y_EAVE, ARC_R, ARC_CY = 4.0, 0.9, 3.0, 4.1, 2.1   # (4, 3.0) lies on the circle


def _profile_dense(n: int = 600) -> np.ndarray:
    """Polyline of the glazed cross-section, left wall foot over the arch to the right wall foot."""
    a0 = math.atan2(Y_EAVE - ARC_CY, HALF_W)
    left = np.array([[-HALF_W, y] for y in np.linspace(Y_WALL, Y_EAVE, n // 6)])
    ang = np.linspace(math.pi - a0, a0, n)
    arc_pts = np.stack([ARC_R * np.cos(ang), ARC_CY + ARC_R * np.sin(ang)], 1)
    right = np.array([[HALF_W, y] for y in np.linspace(Y_EAVE, Y_WALL, n // 6)])
    return np.vstack([left, arc_pts[1:-1], right])


_DENSE = _profile_dense()
_CUM = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(_DENSE, axis=0), axis=1))])
PROFILE_LEN = float(_CUM[-1])


def profile_at(s) -> np.ndarray:
    """(x, y) at arclength s along the cross-section."""
    s = np.clip(np.asarray(s, float), 0, PROFILE_LEN)
    return np.stack([np.interp(s, _CUM, _DENSE[:, 0]), np.interp(s, _CUM, _DENSE[:, 1])], -1)


def inward_normal_at(s) -> np.ndarray:
    ds = 0.02
    s = np.clip(np.asarray(s, float), ds, PROFILE_LEN - ds)      # keep both sample points distinct at the ends
    t = profile_at(s + ds) - profile_at(s - ds)
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    return np.stack([t[..., 1], -t[..., 0]], -1)


def roof_y(x) -> np.ndarray:
    x = np.clip(np.asarray(x, float), -HALF_W, HALF_W)
    return np.where(np.abs(x) >= HALF_W - 1e-6, Y_EAVE, ARC_CY + np.sqrt(np.maximum(ARC_R ** 2 - x ** 2, 0)))


# --------------------------------------------------------------------------
# Mesh accumulation
# --------------------------------------------------------------------------
class Acc:
    """Collects many small meshes and concatenates once (Mesh += is quadratic)."""

    def __init__(self):
        self.p, self.n, self.uv, self.f, self.off = [], [], [], [], 0

    def add(self, mesh: G.Mesh, matrix=None) -> None:
        if matrix is not None:
            mesh = mesh.transformed(matrix)
        self.p.append(mesh.p), self.n.append(mesh.n), self.uv.append(mesh.uv), self.f.append(mesh.f + self.off)
        self.off += len(mesh.p)

    def mesh(self) -> G.Mesh:
        if not self.p:
            return G.Mesh()
        return G.Mesh(np.vstack(self.p), np.vstack(self.n), np.vstack(self.uv), np.vstack(self.f))


class Groups(defaultdict):
    def __init__(self):
        super().__init__(Acc)

    def meshes(self) -> dict[str, G.Mesh]:
        return {k: v.mesh() for k, v in self.items() if v.off}


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def frame(origin, forward, up=UP) -> np.ndarray:
    """Local +z -> forward, +y -> up (as far as perpendicular), +x -> right-handed."""
    f = unit(forward)
    r = np.cross(up, f)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r = unit(r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, np.cross(f, r), f, origin
    return m


# --------------------------------------------------------------------------
# Leaf atlas: K colour variants side by side; each leaf picks one through its uv.
# --------------------------------------------------------------------------
LEAF_COLOURS = [(0.06, 0.27, 0.07), (0.08, 0.33, 0.09), (0.05, 0.22, 0.06), (0.13, 0.36, 0.08),
                (0.04, 0.17, 0.07), (0.07, 0.28, 0.17), (0.30, 0.36, 0.07), (0.24, 0.08, 0.07),
                (0.30, 0.19, 0.07), (0.20, 0.17, 0.08)]                     # last two: dying, brown
LEAF_WEIGHTS = np.array([0.20, 0.18, 0.13, 0.12, 0.10, 0.09, 0.06, 0.04, 0.05, 0.03])
N_VARIANTS = len(LEAF_COLOURS)


def leaf_atlas(h: int = 512, cell_w: int = 128) -> np.ndarray:
    """RGB atlas, v along the leaf (base to tip), u across it. Midrib, lateral veins, dark edge, tip yellowing."""
    out = np.zeros((h, cell_w * N_VARIANTS, 3), np.float32)
    v = np.linspace(0, 1, h)[:, None]
    u = np.linspace(-1, 1, cell_w)[None, :]
    for k, base in enumerate(LEAF_COLOURS):
        noise = G.fbm(h, cell_w, octaves=4, base=6, seed=100 + k)
        fine = G.fbm(h, cell_w, octaves=3, base=40, seed=200 + k)
        col = np.array(base, np.float32)
        rib = np.exp(-(u / 0.05) ** 2)
        lateral = 0.5 + 0.5 * np.cos(TAU * (v * 9.0 - np.abs(u) * 2.2))
        veins = np.clip(0.5 * rib + 0.22 * (lateral ** 6) * (1 - np.abs(u)), 0, 1)
        edge = np.abs(u) ** 3
        tone = 0.80 + 0.35 * noise + 0.12 * fine
        img = col * tone[..., None] * (1 - 0.28 * edge[..., None])
        tip = (v ** 3)[..., None]                                   # older tips go yellow-brown
        img = img * (1 - 0.35 * tip) + np.array([0.30, 0.26, 0.06]) * (0.35 * tip) * (0.6 + 0.8 * noise[..., None])
        img = img * (1 - 0.4 * veins[..., None]) + (col * 1.9 + 0.05) * (0.4 * veins[..., None])
        out[:, k * cell_w:(k + 1) * cell_w] = img
    return np.clip(out, 0, 1)


def make_leaf(rng, length, width, curl=0.3, fold=0.3, rows=4, cols=3, variant=None) -> G.Mesh:
    m = G.leaf(length, width, curl=curl, fold=fold, rows=rows, cols=cols)
    k = int(rng.choice(N_VARIANTS, p=LEAF_WEIGHTS / LEAF_WEIGHTS.sum())) if variant is None else variant
    m.uv = np.stack([(k + 0.05 + 0.9 * m.uv[:, 0]) / N_VARIANTS, 0.01 + 0.98 * m.uv[:, 1]], 1)
    return m


# --------------------------------------------------------------------------
# Textures
# --------------------------------------------------------------------------
BRICK_TILE_M = (0.225 * 9, 0.075 * 26)      # width, height of the brick texture tile in meters


def brick_texture(h: int = 1024, w: int = 1024) -> np.ndarray:
    """Running-bond brick: 9 bricks across by 26 courses, tile size BRICK_TILE_M."""
    rng = np.random.default_rng(31)
    pitch_x, pitch_y, joint = 0.225, 0.075, 0.011
    tile_w, tile_h = pitch_x * 9, pitch_y * 26
    x = (np.arange(w) + 0.5)[None, :] / w * tile_w
    y = (np.arange(h) + 0.5)[:, None] / h * tile_h
    course = np.floor(y / pitch_y).astype(int)
    xo = x + (course % 2) * pitch_x / 2
    brick = np.floor(xo / pitch_x).astype(int) % 9
    lx, ly = xo % pitch_x, y % pitch_y
    tone = rng.normal(0, 1, (26, 9)).astype(np.float32)[course % 26, brick]
    warm = rng.random((26, 9)).astype(np.float32)[course % 26, brick]
    base = np.array([0.43, 0.17, 0.10]) * (1 - warm[..., None]) + np.array([0.52, 0.25, 0.14]) * warm[..., None]
    n = G.fbm(h, w, octaves=5, base=24, seed=33)[..., None]
    rgb = base * (0.88 + 0.12 * tone[..., None]) * (0.75 + 0.5 * n)
    mortar = (lx < joint) | (ly < joint)
    rgb = np.where(mortar[..., None], np.array([0.46, 0.44, 0.38]) * (0.8 + 0.4 * n), rgb)
    return np.clip(rgb, 0, 1).astype(np.float32)


def quarry_tiles(h: int = 1024, w: int = 1024):
    """Terracotta/cream/black 0.25 m quarry tiles with grout; 8 x 8 tiles per 2 m tile."""
    rng = np.random.default_rng(41)
    n = 8
    u, v = (np.arange(w) + 0.5)[None, :] / w * n, (np.arange(h) + 0.5)[:, None] / h * n
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    pick = rng.random((n, n))[iv, iu]
    jitter = rng.normal(0, 1, (n, n)).astype(np.float32)[iv, iu]
    cols = np.array([[0.50, 0.25, 0.15], [0.45, 0.20, 0.12], [0.62, 0.56, 0.42], [0.12, 0.10, 0.09]])
    idx = np.where(pick < 0.42, 0, np.where(pick < 0.80, 1, np.where(pick < 0.95, 2, 3)))
    # a cream/black diamond border pattern would need layout; a random mix reads as old quarry tile
    rgb = cols[idx] * (0.9 + 0.08 * jitter[..., None])
    rgb = rgb * (0.7 + 0.6 * G.fbm(h, w, octaves=5, base=20, seed=43)[..., None])
    grout = np.minimum(np.minimum(u % 1, 1 - u % 1), np.minimum(v % 1, 1 - v % 1)) * (w / n) < 2.5
    rgb = np.where(grout[..., None], np.array([0.30, 0.28, 0.25]), rgb)
    return np.clip(rgb, 0, 1).astype(np.float32)


def soil_texture(h: int = 512, w: int = 512) -> np.ndarray:
    n1 = G.fbm(h, w, octaves=5, base=10, seed=51)
    n2 = G.fbm(h, w, octaves=3, base=80, seed=52)
    rgb = np.array([0.10, 0.065, 0.04]) * (0.55 + 0.9 * n1[..., None]) * (0.8 + 0.4 * n2[..., None])
    pebbles = (n2 > 0.86)[..., None]
    return np.clip(np.where(pebbles, np.array([0.3, 0.27, 0.22]), rgb), 0, 1).astype(np.float32)


def stone_texture(h: int = 512, w: int = 512) -> np.ndarray:
    n1 = G.fbm(h, w, octaves=6, base=6, seed=61)
    n2 = G.fbm(h, w, octaves=3, base=60, seed=62)
    cracks = np.exp(-((G.fbm(h, w, octaves=2, base=5, seed=63) - 0.5) / 0.006) ** 2)
    rgb = np.array([0.46, 0.44, 0.39]) * (0.7 + 0.5 * n1[..., None]) * (0.9 + 0.2 * n2[..., None])
    rgb = rgb * (1 - 0.18 * cracks[..., None])
    return np.clip(rgb, 0, 1).astype(np.float32)


def lawn_texture(h: int = 1024, w: int = 1024) -> np.ndarray:
    n1 = G.fbm(h, w, octaves=5, base=8, seed=71)
    n2 = G.fbm(h, w, octaves=4, base=120, seed=72)
    rgb = np.array([0.10, 0.27, 0.06]) * (0.6 + 0.8 * n1[..., None]) * (0.7 + 0.6 * n2[..., None])
    return np.clip(rgb, 0, 1).astype(np.float32)


def trunk_texture(h: int = 512, w: int = 256) -> np.ndarray:
    v = np.linspace(0, 1, h)[:, None]
    rings = 0.5 + 0.5 * np.cos(TAU * v * 28)
    n = G.fbm(h, w, octaves=5, base=8, seed=81, aspect=0.3)
    rgb = np.array([0.30, 0.22, 0.14]) * (0.5 + 0.9 * n[..., None]) * (0.6 + 0.5 * rings[..., None])
    return np.clip(rgb, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
def tube(points, radius, sides=8) -> G.Mesh:
    return G.sweep(np.asarray(points, float), radius, sides)


def iron_frame(z_near: float, z_far: float, rib_step: float = 1.0, n_bars: int = 16) -> G.Mesh:
    """Arched ribs every `rib_step`, longitudinal glazing bars, ridge and eave beams, wall plates."""
    acc = Acc()
    zs = np.arange(z_near, z_far - 1e-6, -rib_step)
    if abs(zs[-1] - z_far) > 1e-6:
        zs = np.append(zs, z_far)
    s_rib = np.linspace(0, PROFILE_LEN, 80)
    rib_xy = profile_at(s_rib)
    for z in zs:
        acc.add(tube(np.column_stack([rib_xy[:, 0], rib_xy[:, 1], np.full(len(rib_xy), z)]), 0.034, 8))
    bars = np.linspace(0, PROFILE_LEN, n_bars + 1)
    for s in bars[1:-1]:
        x, y = profile_at(s)
        acc.add(tube([(x, y, z_near), (x, y, z_far)], 0.017, 6))
    for s, r in ((0.0, 0.05), (2.1, 0.045), (PROFILE_LEN - 2.1, 0.045), (PROFILE_LEN, 0.05), (PROFILE_LEN / 2, 0.06)):
        x, y = profile_at(s)
        acc.add(tube([(x, y, z_near), (x, y, z_far)], r, 8))
    return acc.mesh()


def roof_panes(rng, z_near: float, z_far: float, rib_step: float = 1.0, n_bars: int = 16):
    """Whitewashed (shading-painted) panes between ribs and bars. Returns (mesh, number of clear panes).

    Clear glass is modelled as an open cell (see scenes/greenhouse.py for why). Shading paint goes on the
    upper roof in patchy runs along z, as it is brushed on in summer; the vertical walls stay clear.
    """
    zs = np.arange(z_near, z_far - 1e-6, -rib_step)
    if abs(zs[-1] - z_far) > 1e-6:
        zs = np.append(zs, z_far)
    ss = np.linspace(0, PROFILE_LEN, n_bars + 1)
    xy = profile_at(ss)
    p, n, uv, f = [], [], [], []
    clear = 0
    for j in range(n_bars):
        upper = 4 <= j <= n_bars - 5
        p_shade = 0.0 if j < 3 or j > n_bars - 4 else (0.3 if upper else 0.08)
        run = rng.random() < p_shade                    # paint goes on in runs of a few panes
        for i in range(len(zs) - 1):
            if rng.random() < 0.3:
                run = rng.random() < p_shade
            if not run:
                clear += 1
                continue
            a, b = xy[j], xy[j + 1]
            z0, z1 = zs[i], zs[i + 1]
            quad = np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]])
            nrm = unit(np.cross(quad[1] - quad[0], quad[3] - quad[0]))
            base = len(p) * 4
            p.append(quad), n.append(np.tile(nrm, (4, 1))), uv.append([[0, 0], [1, 0], [1, 1], [0, 1]])
            f.append([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return G.Mesh(np.vstack(p), np.vstack(n), np.vstack(uv).astype(float), np.vstack(f)), clear


DOOR_HALF_W, DOOR_H = 1.0, 2.6
END_YS = [Y_WALL, 1.75, DOOR_H, 3.4, 4.2, 5.0, 5.8, 6.5]


def end_wall(z: float, door: bool) -> tuple[G.Mesh, G.Mesh, G.Mesh]:
    """Glazed gable end at z. Returns (iron, panes, brick). With `door`, a 2 m opening up to 2.6 m."""
    iron = Acc()
    xs = np.linspace(-HALF_W, HALF_W, 9)
    for x in xs:
        top = float(roof_y(x))
        pts = [(x, y, z) for y in END_YS if y < top - 0.05] + [(x, top, z)]
        if door and abs(x) < DOOR_HALF_W - 1e-6:
            pts = [q for q in pts if q[1] >= DOOR_H - 1e-6]
        if len(pts) > 1:
            iron.add(tube(pts, 0.017, 6))
    for y in END_YS:
        spans = [(-HALF_W, -DOOR_HALF_W), (DOOR_HALF_W, HALF_W)] if door and y < DOOR_H - 1e-6 else [(-HALF_W, HALF_W)]
        for x0, x1 in spans:
            row = [(x, y, z) for x in np.linspace(x0, x1, 5) if y < float(roof_y(x)) - 0.02]
            if len(row) > 1:
                iron.add(tube(row, 0.017, 6))
    p, n, uv, f = [], [], [], []
    for xi in range(len(xs) - 1):
        x0, x1 = xs[xi], xs[xi + 1]
        for yi, y0 in enumerate(END_YS):
            y1 = END_YS[yi + 1] if yi + 1 < len(END_YS) else 8.0
            if y0 < 4.2 - 1e-6:                        # clear glass (an open cell) below the gable top
                continue
            c0, c1 = min(y1, float(roof_y(x0))), min(y1, float(roof_y(x1)))
            if c0 <= y0 + 0.02 or c1 <= y0 + 0.02:
                continue
            quad = np.array([[x0, y0, z], [x1, y0, z], [x1, c1, z], [x0, c0, z]])
            base = len(p) * 4
            p.append(quad), n.append(np.tile([0.0, 0.0, 1.0], (4, 1))), uv.append([[0, 0], [1, 0], [1, 1], [0, 1]])
            f.append([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    panes = G.Mesh(np.vstack(p), np.vstack(n), np.vstack(uv).astype(float), np.vstack(f))
    if door:
        iron.add(tube([(-DOOR_HALF_W, 0, z), (-DOOR_HALF_W, DOOR_H, z), (DOOR_HALF_W, DOOR_H, z),
                       (DOOR_HALF_W, 0, z)], 0.045, 8))
    brick = Acc()
    spans = [(-HALF_W, -DOOR_HALF_W), (DOOR_HALF_W, HALF_W)] if door else [(-HALF_W, HALF_W)]
    for x0, x1 in spans:
        brick.add(brick_slab((x0, z), (x1, z), 0.0, Y_WALL, 0.3))
    return iron.mesh(), panes, brick.mesh()


def brick_slab(p0, p1, y0, y1, thick) -> G.Mesh:
    """Vertical wall slab between two (x, z) points. UVs are in meters along the wall and up it, so
    courses run horizontally on every face (G.box would turn them 90 degrees on x-facing faces)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    d = d / length
    nrm = np.array([-d[1], d[0]])                      # horizontal, perpendicular
    h = thick / 2
    out = Acc()

    def quad(a, b, c, dd, normal, uvs):
        out.add(G.Mesh(np.array([a, b, c, dd]), np.tile(normal, (4, 1)), np.array(uvs, float),
                       np.array([[0, 1, 2], [0, 2, 3]])))

    def P(pt, s, y):
        return [pt[0] + nrm[0] * s, y, pt[1] + nrm[1] * s]

    for s, sign in ((h, 1.0), (-h, -1.0)):             # the two long faces
        nn = [sign * nrm[0], 0.0, sign * nrm[1]]
        quad(P(p0, s, y0), P(p1, s, y0), P(p1, s, y1), P(p0, s, y1), nn,
             [[0, y0], [length, y0], [length, y1], [0, y1]])
    quad(P(p0, -h, y1), P(p1, -h, y1), P(p1, h, y1), P(p0, h, y1), [0, 1, 0],
         [[0, 0], [length, 0], [length, thick], [0, thick]])
    for pt, sign in ((p0, -1.0), (p1, 1.0)):           # the end caps
        nn = [sign * d[0], 0.0, sign * d[1]]
        quad(P(pt, -h, y0), P(pt, h, y0), P(pt, h, y1), P(pt, -h, y1), nn,
             [[0, y0], [thick, y0], [thick, y1], [0, y1]])
    return out.mesh().oriented()


# --------------------------------------------------------------------------
# Furniture and containers
# --------------------------------------------------------------------------
def pot(radius: float, height: float) -> G.Mesh:
    r, h = radius, height
    prof = [(0.0, 0.0), (r * 0.7, 0.0), (r * 0.74, 0.012), (r, h - 0.03), (r + 0.014, h - 0.025), (r + 0.014, h),
            (r - 0.008, h), (r - 0.012, h - 0.04), (0.0, h - 0.04)]
    return G.lathe(prof, 48)


def pot_soil(radius: float, height: float) -> G.Mesh:
    return G.lathe([(radius - 0.01, height - 0.045), (0.0, height - 0.035)], 24)


def urn(height: float = 1.1) -> G.Mesh:
    """Stone urn on a plinth for the centre of the pond."""
    prof = [(0.0, 0.0), (0.42, 0.0), (0.42, 0.12), (0.34, 0.14), (0.34, 0.28), (0.22, 0.32), (0.16, 0.42),
            (0.2, 0.55), (0.3, 0.7), (0.34, 0.85), (0.30, 0.95), (0.34, 1.0), (0.30, 1.04), (0.26, 0.98),
            (0.0, 0.98)]
    return G.lathe(prof, 56)


def bench() -> dict[str, G.Mesh]:
    """Slatted park bench with cast-iron ends, 1.6 m long, seat front toward +z, base at y = 0."""
    wood, iron = Acc(), Acc()
    L = 1.6
    for k in range(5):                                      # seat slats
        wood.add(G.box((L, 0.03, 0.065), (0, 0.45, -0.17 + k * 0.085)))
    for k in range(3):                                      # back slats, leaning back
        m = G.compose(G.translate((0, 0.72 + k * 0.1, -0.24 - k * 0.012)), G.rotate((1, 0, 0), -12))
        wood.add(G.box((L, 0.08, 0.025)), m)
    for sx in (-L / 2 + 0.06, L / 2 - 0.06):
        iron.add(tube([(sx, 0.0, 0.18), (sx, 0.43, 0.14), (sx, 0.44, -0.2)], 0.014, 8))
        iron.add(tube([(sx, 0.0, -0.24), (sx, 0.44, -0.2), (sx, 0.98, -0.3)], 0.014, 8))
        iron.add(tube([(sx, 0.62, -0.02), (sx, 0.62, -0.22)], 0.012, 8))
    return {"wood": wood.mesh(), "iron": iron.mesh()}


def pond(centre, rim_in=1.45, rim_out=1.78, rim_h=0.46) -> dict[str, G.Mesh]:
    """Stone rim: a lathe profile with a rounded coping. The profile runs outer foot, up, over, inner foot,
    because lathe takes its normals from the travel direction (a reversed profile renders black)."""
    mid = 0.5 * (rim_in + rim_out)
    coping = arc(0.06, 0, 180, (mid, rim_h), 7)
    rim = G.lathe([(rim_out, 0.0), (rim_out, rim_h)] + [(x, y) for x, y in coping] + [(rim_in, rim_h), (rim_in, 0.0)], 72)
    return {"rim": rim}


def sky_gradient(h: int = 128, w: int = 256) -> np.ndarray:
    """Lat-long clear-sky radiance (linear RGB) for an `envmap`, row 0 at the zenith, scale 1.

    L = A cos(t) + B (1 - cos(t)) above the horizon, bluer and dimmer at the zenith (B = 2A). The mean
    cosine-weighted radiance E/pi is 0.079, which is what Mitsuba's `sunsky` sky term gives a horizontal
    surface at scale 1 (measured: a 0.5-albedo floor reads 0.0395). Below the horizon: 30% of the horizon.
    """
    theta = (np.arange(h) + 0.5) / h * math.pi
    c = np.cos(theta)[:, None]
    zen, hor = np.array([0.767, 1.015, 1.547]), np.array([1.03, 1.0, 0.926])
    A, B = 0.0593, 0.1185
    sky = np.where(c[..., None] > 0, A * c[..., None] * zen + B * (1 - c[..., None]) * hor, 0.3 * B * hor)
    return np.broadcast_to(sky, (h, w, 3)).astype(np.float32).copy()


def stone_edge(x0, x1, z0, z1, h=0.26, t=0.12) -> G.Mesh:
    """Raised bed kerb: four boxes."""
    acc = Acc()
    acc.add(G.box((x1 - x0, h, t), ((x0 + x1) / 2, h / 2, z0 + t / 2)))
    acc.add(G.box((x1 - x0, h, t), ((x0 + x1) / 2, h / 2, z1 - t / 2)))
    acc.add(G.box((t, h, z1 - z0 - 2 * t), (x0 + t / 2, h / 2, (z0 + z1) / 2)))
    acc.add(G.box((t, h, z1 - z0 - 2 * t), (x1 - t / 2, h / 2, (z0 + z1) / 2)))
    return acc.mesh()


# --------------------------------------------------------------------------
# Plants. Every builder appends to `grp` (a Groups) in world space.
# --------------------------------------------------------------------------
def _pinnate(grp, rng, path, n_leaflets, leaflet_len, leaflet_w, spread_deg, droop, key="leaf", rows=4, cols=3):
    """Leaflets along both sides of a rachis `path` (N x 3), longest in the middle, shortening to the tip."""
    tang = np.gradient(path, axis=0)
    for k in range(n_leaflets):
        t = 0.10 + 0.88 * (k + rng.uniform(0, 0.5)) / n_leaflets
        i = min(int(t * (len(path) - 1)), len(path) - 1)
        T = unit(tang[i])
        S = unit(np.cross(T, UP))
        taper = 0.35 + 0.65 * math.sin(math.pi * min(t * 1.05, 1.0)) ** 0.7
        for sign in (1.0, -1.0):
            a = math.radians(spread_deg + rng.uniform(-8, 8))
            d = unit(T * math.cos(a) + sign * S * math.sin(a) + np.array([0, -droop + rng.uniform(-0.1, 0.1), 0]))
            lf = make_leaf(rng, leaflet_len * taper * rng.uniform(0.85, 1.15), leaflet_w * rng.uniform(0.9, 1.15),
                           curl=rng.uniform(0.1, 0.3), fold=rng.uniform(0.1, 0.3), rows=rows, cols=cols)
            grp[key].add(lf, frame(path[i], d))


def palm(grp, rng, base, height, n_fronds=13, frond_len=2.0):
    base = np.asarray(base, float)
    lean = rng.uniform(-0.3, 0.3, 2)
    t = np.linspace(0, 1, 14)
    trunk = np.column_stack([base[0] + lean[0] * t ** 2 * 0.5, base[1] + height * t, base[2] + lean[1] * t ** 2 * 0.5])
    grp["trunk"].add(G.sweep(trunk, np.linspace(0.14, 0.085, len(trunk)), 12))
    crown = trunk[-1]
    for i in range(n_fronds):
        yaw = TAU * (i + rng.uniform(-0.3, 0.3)) / n_fronds
        el = rng.uniform(0.05, 0.85)
        d0 = np.array([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        L = frond_len * rng.uniform(0.8, 1.15)
        path = bezier([crown, crown + d0 * L * 0.45 + [0, 0.3 * L * math.sin(el + 0.4), 0],
                       crown + d0 * L * 0.95 + [0, -0.3 * L, 0]], 22)
        grp["stem"].add(G.sweep(path, np.linspace(0.016, 0.003, len(path)), 6))
        _pinnate(grp, rng, path, 26, 0.55 * L / 2, 0.045, 52, 0.45)


def fern(grp, rng, base, n_fronds=16, length=0.9):
    base = np.asarray(base, float)
    for i in range(n_fronds):
        yaw = TAU * (i + rng.uniform(-0.3, 0.3)) / n_fronds
        el = rng.uniform(0.45, 1.0)
        d0 = np.array([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        L = length * rng.uniform(0.7, 1.15)
        path = bezier([base, base + d0 * L * 0.5 + [0, 0.22 * L, 0], base + d0 * L * 0.9 + [0, -0.15 * L, 0]], 14)
        grp["stem"].add(G.sweep(path, np.linspace(0.005, 0.0015, len(path)), 5))
        _pinnate(grp, rng, path, 18, 0.14 * L / 0.9, 0.032, 70, 0.25, rows=3, cols=3)


def tropical(grp, rng, base, n_leaves=9, height=1.5, blade=(1.1, 0.5)):
    """Banana / elephant-ear: petioles leaning out, one broad arching blade on each."""
    base = np.asarray(base, float)
    for i in range(n_leaves):
        yaw = TAU * (i + rng.uniform(-0.35, 0.35)) / n_leaves
        lean = rng.uniform(0.15, 0.6)
        h = height * rng.uniform(0.6, 1.0)
        top = base + np.array([math.cos(yaw) * lean * h, h, math.sin(yaw) * lean * h])
        path = bezier([base, base + [0.1 * math.cos(yaw), 0.6 * h, 0.1 * math.sin(yaw)], top], 12)
        grp["stem"].add(G.sweep(path, np.linspace(0.028, 0.012, len(path)), 8))
        size = rng.uniform(0.75, 1.15)
        d = unit([math.cos(yaw), 0.45, math.sin(yaw)])
        lf = make_leaf(rng, blade[0] * size, blade[1] * size, curl=rng.uniform(0.15, 0.4), fold=rng.uniform(0.35, 0.6),
                       rows=9, cols=7)
        grp["leaf"].add(lf, frame(top, d))


def hosta(grp, rng, base, n=9, length=0.38, width=0.24):
    base = np.asarray(base, float)
    for i in range(n):
        yaw = TAU * (i + rng.uniform(-0.3, 0.3)) / n
        el = rng.uniform(0.35, 0.8)
        d = unit([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        lf = make_leaf(rng, length * rng.uniform(0.8, 1.2), width * rng.uniform(0.85, 1.15), curl=rng.uniform(0.3, 0.6),
                       fold=0.35, rows=6, cols=5)
        grp["leaf"].add(lf, frame(base + [0, 0.02, 0], d))


def grass_tuft(grp, rng, base, n=26, length=0.32):
    base = np.asarray(base, float)
    for i in range(n):
        yaw = rng.uniform(0, TAU)
        el = rng.uniform(0.9, 1.45)
        d = unit([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        m = G.leaf(length * rng.uniform(0.6, 1.2), 0.016, curl=rng.uniform(0.2, 0.7), fold=0.0, rows=4, cols=2)
        grp["grass"].add(m, frame(base + rng.normal(0, 0.03, 3) * [1, 0, 1], d))


FLOWER_KEYS = ["flower_pink", "flower_orange", "flower_yellow", "flower_white", "flower_red"]


def bloom(grp, rng, key, centre, normal, size=0.05):
    for k in range(5):
        a = TAU * k / 5 + rng.uniform(-0.2, 0.2)
        d = unit(np.asarray(normal, float) * 0.55 + np.array([math.cos(a), 0.0, math.sin(a)]) * 0.8)
        m = G.leaf(size, size * 0.8, curl=-0.3, fold=0.1, rows=3, cols=3)
        grp[key].add(m, frame(centre, d, normal))


def shrub(grp, rng, base, radius=0.55, height=0.9, n_stems=34, flowers=True, key=None):
    base = np.asarray(base, float)
    key = key or str(rng.choice(FLOWER_KEYS))
    for i in range(n_stems):
        yaw = rng.uniform(0, TAU)
        reach = radius * math.sqrt(rng.uniform(0.05, 1.0))
        h = height * rng.uniform(0.5, 1.0) * (1.0 - 0.35 * reach / radius)
        top = base + np.array([math.cos(yaw) * reach, h, math.sin(yaw) * reach])
        path = bezier([base, base + [0.2 * (top[0] - base[0]), 0.55 * h, 0.2 * (top[2] - base[2])], top], 10)
        grp["stem"].add(G.sweep(path, np.linspace(0.006, 0.002, len(path)), 5))
        for k in range(6):
            idx = 3 + k
            if idx >= len(path):
                break
            d = unit([rng.normal(), rng.uniform(0.0, 0.9), rng.normal()])
            grp["leaf"].add(make_leaf(rng, 0.09, 0.05, curl=0.3, fold=0.3, rows=4, cols=3), frame(path[idx], d))
        if flowers and rng.random() < 0.8:
            bloom(grp, rng, key, top, [0, 1, 0], 0.035)


def vine(grp, rng, path, leaf_len=0.085, every=0.09, sprout=0.35):
    path = np.asarray(path, float)
    grp["stem"].add(G.sweep(path, 0.007, 5))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    for s in np.arange(0.0, cum[-1], every):
        i = min(int(np.searchsorted(cum, s)), len(path) - 1)
        d = unit([rng.normal(), rng.uniform(-0.8, 0.3), rng.normal()])
        grp["leaf"].add(make_leaf(rng, leaf_len * rng.uniform(0.7, 1.3), leaf_len * 0.8, curl=0.3, fold=0.25), frame(path[i], d))
        if rng.random() < sprout:
            tip = path[i] + unit([rng.normal(), rng.uniform(-1, -0.2), rng.normal()]) * rng.uniform(0.08, 0.2)
            grp["stem"].add(G.sweep(np.stack([path[i], tip]), 0.003, 4))
            grp["leaf"].add(make_leaf(rng, leaf_len * 0.8, leaf_len * 0.65), frame(tip, unit([rng.normal(), -0.5, rng.normal()])))


def climber_on_rib(grp, rng, z, s0, s1, offset=0.07, wobble=0.04, **kw):
    """A vine following the rib at depth z between arclengths s0..s1, kept just inside it."""
    s0, s1 = max(s0, 0.05), min(s1, PROFILE_LEN - 0.05)
    if s1 - s0 < 1.0:
        return
    ss = np.linspace(s0, s1, 60)
    xy = profile_at(ss) + inward_normal_at(ss) * offset
    xy = xy + inward_normal_at(ss) * (wobble * np.sin(ss * rng.uniform(2, 5) + rng.uniform(0, 6)))[:, None]
    zz = z + wobble * np.sin(ss * rng.uniform(3, 7) + rng.uniform(0, 6))
    vine(grp, rng, np.column_stack([xy[:, 0], xy[:, 1], zz]), **kw)


def hanging_basket(grp, rng, anchor, drop=1.7, radius=0.2, trail=0.9, fill=True):
    """Moss-lined wire basket on three chains, with flowers and trailing vines. Returns the basket centre."""
    anchor = np.asarray(anchor, float)
    centre = anchor - [0, drop, 0]
    for k in range(3):
        a = TAU * k / 3
        rim = centre + [radius * math.cos(a), 0.05, radius * math.sin(a)]
        grp["chain"].add(G.sweep(np.stack([rim, anchor]), 0.004, 4))
    prof = [(0.0, -0.1), (0.06, -0.095), (0.14, -0.06), (radius, 0.0), (radius + 0.015, 0.06), (radius - 0.01, 0.07),
            (0.0, 0.04)]
    grp["basket"].add(G.lathe(prof, 24), G.translate(centre))
    liner = [(0.0, -0.085), (0.13, -0.05), (radius - 0.012, 0.01), (radius - 0.02, 0.06), (0.0, 0.05)]
    grp["moss"].add(G.lathe(liner, 24), G.translate(centre))
    for k in range(5 if fill else 3):
        a = rng.uniform(0, TAU)
        start = centre + [0.7 * radius * math.cos(a), 0.06, 0.7 * radius * math.sin(a)]
        end = start + [0.15 * math.cos(a), -trail * rng.uniform(0.4, 1.0), 0.15 * math.sin(a)]
        vine(grp, rng, bezier([start, start + [0.18 * math.cos(a), 0.0, 0.18 * math.sin(a)], end], 16), leaf_len=0.06, every=0.07)
    if not fill:
        return centre
    key = str(rng.choice(FLOWER_KEYS))
    for k in range(16):
        a, r = rng.uniform(0, TAU), radius * math.sqrt(rng.uniform(0, 0.85))
        bloom(grp, rng, key, centre + [r * math.cos(a), 0.08 + rng.uniform(0, 0.06), r * math.sin(a)], [0, 1, 0], 0.035)
        grp["leaf"].add(make_leaf(rng, 0.1, 0.06), frame(centre + [r * math.cos(a) * 0.8, 0.07, r * math.sin(a) * 0.8],
                                                         unit([rng.normal(), 0.6, rng.normal()])))
    return centre


def agave(grp, rng, base, n=16, length=0.55):
    base = np.asarray(base, float)
    for i in range(n):
        yaw = TAU * i / n + rng.uniform(-0.2, 0.2)
        el = rng.uniform(0.3, 1.2) if i > 4 else rng.uniform(1.1, 1.45)
        d = unit([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        m = G.leaf(length * rng.uniform(0.75, 1.1), 0.075, curl=rng.uniform(0.1, 0.3), fold=0.5, rows=6, cols=4)
        grp["succulent"].add(m, frame(base + [0, 0.03, 0], d))


def tree(grp, rng, base, height=6.0, crown_r=2.2, n_leaves=1100, leaf=(0.26, 0.15)):
    """Garden tree outside the door: trunk, a few limbs, and a dome of leaves."""
    base = np.asarray(base, float)
    top = base + [0, height * 0.55, 0]
    grp["trunk"].add(G.sweep(np.stack([base, top]), np.linspace(0.28, 0.16, 2), 10))
    centre = base + [0, height * 0.78, 0]
    for k in range(5):
        a = TAU * k / 5 + rng.uniform(-0.3, 0.3)
        end = centre + [crown_r * 0.7 * math.cos(a), crown_r * 0.2, crown_r * 0.7 * math.sin(a)]
        grp["trunk"].add(G.sweep(bezier([top, top + [0, 0.5, 0], end], 8), np.linspace(0.12, 0.04, 8), 7))
    for i in range(n_leaves):
        v = unit(rng.normal(size=3))
        v[1] = abs(v[1]) * 0.8
        p = centre + v * crown_r * rng.uniform(0.55, 1.0) ** 0.5 * np.array([1.0, 0.75, 1.0])
        grp["leaf"].add(make_leaf(rng, leaf[0], leaf[1], curl=0.3, fold=0.3), frame(p, unit(rng.normal(size=3))))


def lily_pad(grp, rng, centre, radius):
    """Flat disc with a notch, floating at the water line."""
    n = 18
    a = np.linspace(0.12, TAU - 0.12, n) + rng.uniform(0, TAU)
    rim = np.stack([radius * np.cos(a), rng.uniform(0.002, 0.012, n) + 0.006 * np.sin(2 * a), radius * np.sin(a)], 1)
    p = np.vstack([[0, 0, 0], rim])
    f = np.array([[0, i + 1, i + 2] for i in range(n - 1)])
    nrm = np.tile([0.0, 1.0, 0.0], (len(p), 1))
    uv = np.column_stack([0.5 + p[:, 0] / (2 * radius), 0.5 + p[:, 2] / (2 * radius)])
    m = G.Mesh(p, nrm, uv, f)
    grp["pad"].add(m, G.translate(centre))


# --------------------------------------------------------------------------
# Scanned (Poly Haven) assets: many plant downloads are a sheet of several variants side by side.
# These helpers read the converted PLYs, find the variants by gaps along x, and cut one out.
# --------------------------------------------------------------------------
def read_ply(path) -> G.Mesh:
    """Read the binary PLY that procedural.Mesh.write_ply (and web_assets' converter) writes."""
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fh.readline()
        lines = header.decode("ascii").splitlines()
        nv = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
        nf = int(next(l for l in lines if l.startswith("element face")).split()[-1])
        v = np.frombuffer(fh.read(nv * 32), dtype="<f4").reshape(nv, 8).astype(np.float64)
        faces = np.frombuffer(fh.read(nf * 13), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    return G.Mesh(v[:, :3], v[:, 3:6], v[:, 6:8], faces["i"].astype(np.int64))


def variant_ranges(meshes, axis: int = 0, gap: float = 0.04, bins: int = 2000) -> list[tuple[float, float]]:
    """Intervals along `axis` occupied by geometry, split wherever an empty gap wider than `gap` appears."""
    xs = np.concatenate([m.p[:, axis] for m in meshes if len(m.p)])
    lo, hi = xs.min(), xs.max()
    hist, edges = np.histogram(xs, bins=bins, range=(lo, hi))
    width = (hi - lo) / bins
    ranges, start, empty = [], None, 0
    for i, c in enumerate(hist):
        if c:
            if start is None:
                start = edges[i]
            elif empty * width > gap:
                ranges.append((start, edges[i - empty]))
                start = edges[i]
            empty, last = 0, edges[i + 1]
        else:
            empty += 1
    ranges.append((start, last))
    return [(float(a), float(b)) for a, b in ranges]


def cut_variant(meshes: dict, xr, axis: int = 0) -> dict:
    """Faces whose centroid lies in xr, re-centred: footprint centre at the origin, lowest point at y = 0."""
    out = {}
    for name, m in meshes.items():
        c = m.p[m.f].mean(axis=1)[:, axis]
        keep = (c >= xr[0]) & (c <= xr[1])
        if not keep.any():
            continue
        f = m.f[keep]
        used, inv = np.unique(f, return_inverse=True)
        out[name] = G.Mesh(m.p[used], m.n[used], m.uv[used], inv.reshape(-1, 3))
    if not out:
        return out
    allp = np.vstack([m.p for m in out.values()])
    lo, hi = allp.min(0), allp.max(0)
    shift = -np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return {k: G.Mesh(m.p + shift, m.n, m.uv, m.f) for k, m in out.items()}


def split_variants(meshes: dict, gap: float = 0.04, min_tris: int = 400) -> list[dict]:
    """Split a sheet of scanned plants into separate plants: by gaps along x, then along z within each.
    Each variant is re-centred (footprint centre at the origin, base at y = 0). Debris under min_tris is dropped."""
    out = []
    for xr in variant_ranges(list(meshes.values()), axis=0, gap=gap):
        col = cut_variant(meshes, xr, axis=0)
        if not col:
            continue
        for zr in variant_ranges(list(col.values()), axis=2, gap=gap):
            v = cut_variant(col, zr, axis=2)
            if v and sum(len(m.f) for m in v.values()) >= min_tris:
                out.append(v)
    return out


def hedge(grp, rng, centre, size, density: float = 150.0):
    """Clipped hedge: a dark core box covered by a shell of small leaves (atlas leaves, outward-leaning)."""
    c, s = np.asarray(centre, float), np.asarray(size, float)       # centre of the base, (width x, height, length z)
    grp["hedge_core"].add(G.box(s - 0.08, c + [0, s[1] / 2, 0]))
    faces = [((0, 1), 2, s[1] * s[2]), ((0, -1), 2, s[1] * s[2]), ((2, 1), 0, s[0] * s[1]), ((2, -1), 0, s[0] * s[1]),
             ((1, 1), 0, s[0] * s[2])]
    for (axis, sign), _, area in faces:
        for _ in range(int(area * density)):
            p = c + [rng.uniform(-s[0] / 2, s[0] / 2), rng.uniform(0.05, s[1]), rng.uniform(-s[2] / 2, s[2] / 2)]
            p[axis] = c[axis] + (s[1] if axis == 1 else sign * s[axis] / 2) + rng.normal(0, 0.03)
            out = np.zeros(3)
            out[axis] = sign
            d = unit(out * 0.8 + rng.normal(0, 0.6, 3))
            grp["leaf"].add(make_leaf(rng, 0.075, 0.045, curl=0.2, fold=0.2, rows=3, cols=3), frame(p, d))


# --------------------------------------------------------------------------
# Wear and working clutter
# --------------------------------------------------------------------------
def pane_grime(h: int = 256, w: int = 256) -> np.ndarray:
    """Shading-paint pane albedo: algae and dirt collecting along the bars, run-off streaks down the pane."""
    u = np.linspace(0, 1, w)[None, :]
    v = np.linspace(0, 1, h)[:, None]
    edge = np.exp(-np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)) / 0.05)
    streaks = G.fbm(h, w, octaves=4, base=3, seed=91, aspect=8.0) ** 3
    blotch = G.fbm(h, w, octaves=5, base=5, seed=92)
    base = np.array([0.66, 0.70, 0.62])
    algae = np.array([0.22, 0.30, 0.12])
    mix = np.clip(0.75 * edge + 0.35 * streaks + 0.2 * (blotch - 0.5), 0, 1)[..., None]
    return np.clip(base * (1 - mix) + algae * mix, 0, 1).astype(np.float32)


def label(grp, rng, at, yaw):
    """White plastic plant label stuck in soil, slightly tilted."""
    m = G.box((0.016, 0.075, 0.002))
    grp["label"].add(m, G.compose(G.translate(at), G.rotate((0, 1, 0), yaw), G.rotate((1, 0, 0), rng.uniform(-12, 12)),
                                  G.translate((0, 0.03, 0))))


def coiled_hose(grp, centre, turns=4.5, r0=0.16, r1=0.34, tube=0.011):
    a = np.linspace(0, TAU * turns, 260)
    r = np.linspace(r0, r1, len(a))
    path = np.column_stack([centre[0] + r * np.cos(a), centre[1] + tube + 0.004 * np.sin(a * 3), centre[2] + r * np.sin(a)])
    tail = bezier([path[-1], path[-1] + [0.4, 0, 0.1], path[-1] + [0.9, 0, -0.3]], 30)
    grp["hose"].add(G.sweep(np.vstack([path, tail[1:]]), tube, 8))


def blob(rng, centre, radius, n=24, y=0.0015) -> G.Mesh:
    """Irregular flat patch (puddle, soil spill) lying on the floor; uv in meters."""
    a = np.linspace(0, TAU, n, endpoint=False)
    r = radius * (0.7 + 0.3 * G.fbm(1, n, octaves=2, base=3, seed=int(rng.integers(1 << 30)))[0]) * rng.uniform(0.8, 1.2, n)
    rim = np.stack([centre[0] + r * np.cos(a), np.full(n, y), centre[2] + r * np.sin(a)], 1)
    p = np.vstack([[centre[0], y, centre[2]], rim])
    f = np.array([[0, 1 + (i + 1) % n, 1 + i] for i in range(n)])
    return G.Mesh(p, np.tile([0.0, 1.0, 0.0], (len(p), 1)), p[:, [0, 2]].copy(), f).oriented()
