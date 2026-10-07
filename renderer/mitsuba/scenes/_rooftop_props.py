"""Geometry, textures and plant builders for scenes/rooftop.py.

Pure numpy + Pillow (no Mitsuba), meters, y up. Many small meshes are merged per material through
`Groups`, so a city of thousands of buildings and terraces full of leaves stay a few dozen shapes.

Textures are generated in LINEAR colour and written sRGB-encoded (`save_albedo`), because Mitsuba
decodes 8-bit PNG albedo as sRGB. Roughness and other data maps are written as-is and loaded raw.
Facade maps are stored bottom-up (row 0 = ground level), matching uv v = height in meters.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import procedural as G
from props import arc, bezier

TAU = 2 * math.pi
UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------
# Mesh helpers
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
        return {k: v.mesh() for k, v in sorted(self.items()) if v.off}


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


def merge(meshes) -> G.Mesh:
    acc = Acc()
    for m in meshes:
        acc.add(m)
    return acc.mesh()


def quad(p0, p1, p2, p3, normal, uv) -> G.Mesh:
    p = np.array([p0, p1, p2, p3], float)
    n = np.tile(unit(normal), (4, 1))
    return G.Mesh(p, n, np.asarray(uv, float), np.array([[0, 1, 2], [0, 2, 3]])).oriented()


def box(size, centre) -> G.Mesh:
    return G.box(size, centre)


def tube(points, radius, sides=8) -> G.Mesh:
    return G.sweep(np.asarray(points, float), radius, sides)


def cylinder(radius, height, segments=32, y0=0.0) -> G.Mesh:
    return G.lathe([(0.0, y0), (radius, y0), (radius, y0 + height), (0.0, y0 + height)], segments)


def superellipsoid(radii, e: float = 0.25, rows: int = 12, cols: int = 20) -> G.Mesh:
    """Rounded box: exponent 1 is an ellipsoid, small exponents approach a box."""
    a, b, c = radii
    th = np.linspace(-np.pi / 2, np.pi / 2, rows)[:, None]
    ph = np.linspace(0.0, TAU, cols + 1)[None, :]
    sp = lambda u, k: np.sign(u) * np.abs(u) ** k
    cl, sl = sp(np.cos(th), e), sp(np.sin(th), e)
    p = np.stack(np.broadcast_arrays(a * cl * sp(np.cos(ph), e), b * sl, c * cl * sp(np.sin(ph), e)), -1).reshape(-1, 3)
    uv = np.stack(np.broadcast_arrays(ph / TAU, (th + np.pi / 2) / np.pi), -1).reshape(-1, 2)
    f = G._grid_faces(rows, cols + 1)
    return G.Mesh(p, G.vertex_normals(p, f), uv, f).oriented()


# --------------------------------------------------------------------------
# Texture output
# --------------------------------------------------------------------------
def srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def save_albedo(path: Path, lin: np.ndarray) -> None:
    """Linear albedo -> sRGB-encoded 8-bit PNG (Mitsuba decodes it back to linear). Rows bottom-up."""
    G.save_rgb(path, srgb(np.asarray(lin, np.float32)))


def save_data(path: Path, gray: np.ndarray) -> None:
    """Data map (roughness, masks) written linearly; load it with raw=True."""
    G.save_rgb(path, np.repeat(np.asarray(gray, np.float32)[..., None], 3, -1))


def _grid(tw: float, th: float, ppm: float):
    w, h = int(round(tw * ppm)), int(round(th * ppm))
    u = (np.arange(w) + 0.5) / ppm
    v = (np.arange(h) + 0.5) / ppm
    return np.meshgrid(u, v)          # U, V in meters, row 0 = v small = bottom


def _rect(U, V, x0, x1, y0, y1):
    return (U >= x0) & (U < x1) & (V >= y0) & (V < y1)


# --------------------------------------------------------------------------
# Facade textures. Each returns (albedo HxWx3 linear, roughness HxW, (tile_w, tile_h) in meters).
# --------------------------------------------------------------------------
# Facade kinds. Real kinds put a scanned Poly Haven wall texture (id, tint) everywhere except the window layer,
# which stays procedural (glass, frames, sills, shutters, interiors). `layout` picks the window pattern.
FACADES = {
    "glass":      {"layout": "glass", "tex": None},
    "dark":       {"layout": "dark", "tex": None},
    "brick_a":    {"layout": "brick", "tex": ("exterior_wall_cladding_02", None)},
    "brick_b":    {"layout": "brick", "tex": ("patterned_brick_wall_03", None)},
    "brick_c":    {"layout": "brick", "tex": ("red_brick_plaster_patch_02", None)},
    "brick_d":    {"layout": "brick", "tex": ("exterior_wall_cladding_02", (0.78, 0.72, 0.66))},
    "stone":      {"layout": "stone", "tex": ("rough_block_wall", None)},
    "render_a":   {"layout": "render0", "tex": ("beige_wall_001", None)},
    "render_b":   {"layout": "render1", "tex": ("plastered_wall_02", (1.0, 0.86, 0.70))},
    "render_c":   {"layout": "render2", "tex": ("blue_plaster_weathered", None)},
    "render_d":   {"layout": "render1", "tex": ("beige_wall_001", (1.0, 0.80, 0.66))},
    "concrete_a": {"layout": "concrete", "tex": ("concrete_tile_facade", None)},
    "concrete_b": {"layout": "concrete", "tex": ("concrete_slab_wall_02", None)},
    "tiles":      {"layout": "concrete", "tex": ("rectangular_facade_tiles", None)},
}
FACADE_KINDS = list(FACADES)
RENDER_COLOURS = [(0.60, 0.40, 0.21), (0.66, 0.60, 0.49), (0.52, 0.26, 0.16)]
WARM, COOL = np.array([1.0, 0.64, 0.30]), np.array([0.80, 0.88, 1.0])


def _windows(U, V, bay, floor, wx, wy0, wy1, rng):
    """Masks for a regular window grid: glass, frame (5 cm), sill; plus a per-window interior tone."""
    ub, vb = U % bay, V % floor
    cx = bay / 2
    glass = (np.abs(ub - cx) < wx / 2) & (vb >= wy0) & (vb < wy1)
    frame_ = (np.abs(ub - cx) < wx / 2 + 0.06) & (vb >= wy0 - 0.06) & (vb < wy1 + 0.06) & ~glass
    sill = (np.abs(ub - cx) < wx / 2 + 0.12) & (vb >= wy0 - 0.12) & (vb < wy0 - 0.04)
    i, j = (U // bay).astype(int), (V // floor).astype(int)
    cells = rng.random((int(i.max()) + 2, int(j.max()) + 2))
    return glass, frame_, sill, cells[i, j]


def _interior(tone, glass_tint):
    """What a window shows: dark room, blinds or curtains, with the glazing's own tint."""
    return np.where(tone[..., None] < 0.55, np.array(glass_tint) * (0.6 + 0.8 * tone[..., None]),
                    np.where(tone[..., None] < 0.8, np.array([0.30, 0.28, 0.24]), np.array([0.42, 0.36, 0.28])))


def _lit(sel, tone, colour, U, V, bay, floor):
    """Unit-radiance map of lit windows: `sel` picks pixels, per-window level from `tone`."""
    level = 0.45 + 0.55 * ((tone * 7.31) % 1.0)
    return np.where(sel[..., None], colour * level[..., None], 0.0).astype(np.float32)


def facade_texture(kind: str, ppm: float = 42.0):
    """(window-layer albedo, roughness, window mask, lit-window radiance map, (tile_w, tile_h) m) for a kind."""
    layout = FACADES[kind]["layout"]
    rng = np.random.default_rng(FACADE_KINDS.index(kind) * 17 + 3)
    if layout == "glass":
        tw, th = 6.0, 7.2
        U, V = _grid(tw, th, ppm)
        n = G.fbm(*U.shape, octaves=3, base=6, seed=31)
        mull = ((U % 1.5) < 0.07) | ((V % 3.6) < 0.08)
        span = (V % 3.6) < 0.95
        panel = np.floor(U / 1.5) + 7 * np.floor(V / 3.6)
        pv = (np.sin(panel * 12.9898) * 43758.5453) % 1.0           # per-pane tint jitter
        glass = np.array([0.035, 0.050, 0.065]) * (0.8 + 0.5 * pv[..., None])
        alb = np.where(span[..., None], np.array([0.055, 0.065, 0.075]), glass)
        alb = np.where(mull[..., None], np.array([0.42, 0.43, 0.44]), alb) * (0.92 + 0.16 * n[..., None])
        rough = np.where(mull, 0.35, np.where(span, 0.25, 0.04))
        mask = np.ones_like(U)
        fl = (np.sin(np.floor(V / 3.6) * 17.17 + 3.0) * 7531.3) % 1.0             # whole office floors are lit
        lit = _lit(~mull & ~span & (fl < 0.10), pv, COOL, U, V, 1.5, 3.6)
    elif layout == "brick":
        tw, th = 11.2, 9.6
        U, V = _grid(tw, th, ppm)
        glass, fr, sill, tone = _windows(U, V, 2.8, 3.2, 1.25, 0.9, 2.65, rng)
        alb = np.zeros(U.shape + (3,))
        alb = np.where(glass[..., None], _interior(tone, (0.04, 0.045, 0.05)), alb)
        alb = np.where(fr[..., None], np.array([0.72, 0.71, 0.68]), alb)
        alb = np.where(sill[..., None], np.array([0.55, 0.52, 0.47]), alb)
        rough = np.where(glass, 0.05, 0.6)
        mask = (glass | fr | sill).astype(float)
        lit = _lit(glass & (tone > 0.08) & (tone < 0.15), tone, WARM, U, V, 2.8, 3.2)
    elif layout == "stone":
        tw, th = 12.0, 10.8
        U, V = _grid(tw, th, ppm)
        glass, fr, sill, tone = _windows(U, V, 3.0, 3.6, 1.3, 0.85, 3.0, rng)
        alb = np.zeros(U.shape + (3,))
        alb = np.where(glass[..., None], _interior(tone, (0.035, 0.04, 0.045)), alb)
        alb = np.where(fr[..., None], np.array([0.16, 0.15, 0.13]), alb)
        alb = np.where(sill[..., None], np.array([0.62, 0.57, 0.48]), alb)
        rough = np.where(glass, 0.05, 0.6)
        mask = (glass | fr | sill).astype(float)
        lit = _lit(glass & (tone > 0.1) & (tone < 0.17), tone, WARM, U, V, 3.0, 3.6)
    elif layout.startswith("render"):
        k = int(layout[-1])
        tw, th = 10.4, 9.3
        U, V = _grid(tw, th, ppm)
        glass, fr, sill, tone = _windows(U, V, 2.6, 3.1, 1.05, 0.95, 2.55, rng)
        shut_col = np.array([(0.07, 0.17, 0.09), (0.20, 0.11, 0.05), (0.10, 0.13, 0.18)][k])
        ub = U % 2.6
        shutter = (((np.abs(ub - 1.3 + 0.82) < 0.27) | (np.abs(ub - 1.3 - 0.82) < 0.27))
                   & ((V % 3.1) >= 0.95) & ((V % 3.1) < 2.55))
        closed = glass & (tone > 0.8)
        louvre = 0.75 + 0.25 * (((V * 18.0) % 1.0) > 0.5)
        alb = np.zeros(U.shape + (3,))
        alb = np.where(glass[..., None], _interior(tone, (0.04, 0.042, 0.045)), alb)
        alb = np.where((shutter | closed)[..., None], shut_col * louvre[..., None], alb)
        alb = np.where(fr[..., None], np.array([0.70, 0.68, 0.64]), alb)
        alb = np.where(sill[..., None], np.array([0.62, 0.58, 0.52]), alb)
        rough = np.where(glass & ~closed, 0.05, 0.75)
        mask = (glass | fr | sill | shutter).astype(float)
        lit = _lit(glass & ~closed & (tone > 0.1) & (tone < 0.17), tone, WARM, U, V, 2.6, 3.1)
    elif layout == "concrete":
        tw, th = 7.2, 6.8
        U, V = _grid(tw, th, ppm)
        ribbon = ((V % 3.4) > 1.0) & ((V % 3.4) < 2.55)
        mull = (U % 1.2) < 0.06
        tone = (np.sin(np.floor(U / 1.2) * 3.1 + np.floor(V / 3.4) * 7.7) * 4378.5) % 1.0
        alb = np.where(ribbon[..., None], _interior(tone, (0.035, 0.045, 0.05)), 0.0)
        alb = np.where((ribbon & mull)[..., None], np.array([0.12, 0.12, 0.12]), alb)
        rough = np.where(ribbon & ~mull, 0.05, 0.6)
        mask = ribbon.astype(float)
        lit = _lit(ribbon & ~mull & (tone > 0.1) & (tone < 0.17), tone, COOL * 0.5 + WARM * 0.5, U, V, 1.2, 3.4)
    elif layout == "dark":
        tw, th = 6.0, 7.2
        U, V = _grid(tw, th, ppm)
        n = G.fbm(*U.shape, octaves=3, base=8, seed=37)
        alb = np.array([0.07, 0.07, 0.075]) * (0.9 + 0.2 * n[..., None])
        strip = ((U % 1.5) > 0.45) & ((U % 1.5) < 1.35) & ((V % 3.6) > 0.5)
        fin = (U % 1.5) < 0.12
        alb = np.where(strip[..., None], np.array([0.03, 0.04, 0.05]), alb)
        alb = np.where(fin[..., None], np.array([0.20, 0.19, 0.17]), alb)
        rough = np.where(strip, 0.05, np.where(fin, 0.45, 0.4))
        mask = np.ones_like(U)
        tone = (np.sin(np.floor(U / 1.5) * 5.3 + np.floor(V / 3.6) * 2.9) * 9157.1) % 1.0
        lit = _lit(strip & (tone < 0.06), tone * 3, COOL, U, V, 1.5, 3.6)
    else:
        raise ValueError(kind)
    return (alb.astype(np.float32), rough.astype(np.float32), mask.astype(np.float32), lit, (tw, th))


def roof_texture(ppm: float = 20.0):
    U, V = _grid(10.0, 10.0, ppm)
    n = G.fbm(*U.shape, octaves=5, base=6, seed=41)
    fine = G.fbm(*U.shape, octaves=2, base=60, seed=42)
    alb = np.array([0.26, 0.25, 0.23]) * (0.7 + 0.4 * n[..., None] + 0.15 * fine[..., None])
    return alb.astype(np.float32)


STREET_PITCH, STREET_W = 110.0, 22.0


def street_tile(ppm: float = 12.0):
    """One city block period (110 m): block interior pavement in the middle, 22 m streets on the edges
    with kerbs, sidewalks, dashed centre lines, lane lines and zebra crossings at the intersections."""
    P = STREET_PITCH
    U, V = _grid(P, P, ppm)
    n = G.fbm(*U.shape, octaves=5, base=12, seed=51)
    half = STREET_W / 2
    du = np.minimum(U, P - U)                 # distance to the street centre line along u
    dv = np.minimum(V, P - V)
    road_u, road_v = du < half - 3.0, dv < half - 3.0      # carriageway (16 m); 3 m sidewalks
    side_u, side_v = (du < half) & ~road_u, (dv < half) & ~road_v
    asphalt = np.array([0.075, 0.075, 0.08]) * (0.75 + 0.5 * n[..., None])
    pave = np.array([0.33, 0.31, 0.29]) * (0.85 + 0.25 * n[..., None])
    court = np.array([0.24, 0.23, 0.21]) * (0.8 + 0.4 * n[..., None])
    alb = np.where((road_u | road_v)[..., None], asphalt, np.where((side_u | side_v)[..., None], pave, court))
    white = np.array([0.62, 0.62, 0.60])
    centre_u = road_u & ~road_v & (du < 0.08) & ((V % 6.0) < 3.0)
    centre_v = road_v & ~road_u & (dv < 0.08) & ((U % 6.0) < 3.0)
    lane_u = road_u & ~road_v & (np.abs(du - 4.0) < 0.06) & ((V % 9.0) < 2.0)
    lane_v = road_v & ~road_u & (np.abs(dv - 4.0) < 0.06) & ((U % 9.0) < 2.0)
    zebra_u = road_u & (dv > half - 0.2) & (dv < half + 3.8) & ((U % 1.0) < 0.5)   # crossing across the u-street
    zebra_v = road_v & (du > half - 0.2) & (du < half + 3.8) & ((V % 1.0) < 0.5)
    marks = centre_u | centre_v | lane_u | lane_v | zebra_u | zebra_v
    alb = np.where(marks[..., None], white * (0.8 + 0.3 * n[..., None]), alb)
    kerb = ((np.abs(du - (half - 3.0)) < 0.15) & ~road_v) | ((np.abs(dv - (half - 3.0)) < 0.15) & ~road_u)
    alb = np.where(kerb[..., None], np.array([0.45, 0.44, 0.42]), alb)
    return alb.astype(np.float32)


def deck_texture(ppm: float = 300.0):
    """Hardwood decking boards (14 cm, 5 mm gaps) running along x, staggered ends, 2.8 m tile."""
    U, V = _grid(2.8, 1.4, ppm)
    board = np.floor(V / 0.14)
    bv = (np.sin(board * 12.345) * 4321.1) % 1.0
    shift = bv * 2.8
    ends = ((U + shift) % 2.8) < 0.006
    gap = (V % 0.14) < 0.006
    grain = G.fbm(U.shape[0], U.shape[1], octaves=4, base=4, seed=52, aspect=8.0)
    streak = 0.5 + 0.5 * np.sin(V * 380.0 + grain * 9.0)
    seg = np.floor((U + shift) / 2.8)
    sv = (np.sin(seg * 7.7 + board * 3.3) * 999.1) % 1.0
    base = np.array([0.30, 0.17, 0.09]) * (0.72 + 0.35 * sv[..., None] + 0.12 * streak[..., None])
    weather = np.array([0.36, 0.31, 0.26])                              # silvering on some boards
    base = base * (1 - 0.35 * bv[..., None]) + weather * 0.35 * bv[..., None]
    alb = np.where((gap | ends)[..., None], np.array([0.02, 0.015, 0.01]), base)
    rough = np.where(gap | ends, 0.9, 0.55 + 0.2 * grain)
    return alb.astype(np.float32), rough.astype(np.float32)


def paver_texture(ppm: float = 200.0):
    U, V = _grid(1.2, 1.2, ppm)
    joint = ((U % 0.6) < 0.006) | ((V % 0.6) < 0.006)
    tid = np.floor(U / 0.6) + 3 * np.floor(V / 0.6)
    tv = (np.sin(tid * 91.7) * 777.7) % 1.0
    n = G.fbm(*U.shape, octaves=4, base=6, seed=53)
    alb = np.array([0.40, 0.38, 0.35]) * (0.85 + 0.12 * tv[..., None] + 0.15 * n[..., None])
    alb = np.where(joint[..., None], np.array([0.12, 0.11, 0.10]), alb)
    return alb.astype(np.float32)


def corten_texture(ppm: float = 300.0):
    U, V = _grid(1.0, 1.0, ppm)
    n = G.fbm(*U.shape, octaves=5, base=5, seed=54)
    fine = G.fbm(*U.shape, octaves=3, base=50, seed=55)
    alb = np.array([0.24, 0.085, 0.03]) * (0.65 + 0.5 * n[..., None] + 0.2 * fine[..., None])
    streaks = np.clip(G.fbm(*U.shape, octaves=3, base=4, seed=56, aspect=0.15) - 0.55, 0, 1)
    alb = alb * (1 - 0.6 * streaks[..., None])
    return alb.astype(np.float32)


def plaster_texture(colour, seed: int, ppm: float = 150.0):
    U, V = _grid(2.0, 2.0, ppm)
    n = G.fbm(*U.shape, octaves=5, base=5, seed=seed)
    fine = G.fbm(*U.shape, octaves=2, base=80, seed=seed + 1)
    return (np.array(colour) * (0.88 + 0.18 * n[..., None] + 0.06 * fine[..., None])).astype(np.float32)


def fabric_texture(colour, seed: int, ppm: float = 400.0):
    U, V = _grid(0.5, 0.5, ppm)
    weave = 0.9 + 0.1 * (np.sin(U * 900) * np.sin(V * 900))
    n = G.fbm(*U.shape, octaves=3, base=6, seed=seed)
    return (np.array(colour) * weave[..., None] * (0.92 + 0.12 * n[..., None])).astype(np.float32)


def marble_texture():
    return np.clip(G.marble(512, 512, seed=57) * 0.85, 0, 1).astype(np.float32)


# leaf atlas: olive (dark sage top), lavender foliage (grey-green), vine (fresh green), grass (green-straw)
LEAF_COLOURS = [(0.11, 0.14, 0.08), (0.15, 0.18, 0.11), (0.21, 0.24, 0.17), (0.17, 0.21, 0.15),
                (0.08, 0.20, 0.05), (0.12, 0.26, 0.06), (0.30, 0.30, 0.12), (0.18, 0.25, 0.08)]
OLIVE, LAV, VINE, GRASS = (0, 1, 2), (3,), (4, 5), (6, 7)
N_VARIANTS = len(LEAF_COLOURS)


def leaf_atlas(h: int = 256, cell_w: int = 64) -> np.ndarray:
    out = np.zeros((h, cell_w * N_VARIANTS, 3), np.float32)
    v = np.linspace(0, 1, h)[:, None]
    u = np.linspace(-1, 1, cell_w)[None, :]
    for k, base in enumerate(LEAF_COLOURS):
        noise = G.fbm(h, cell_w, octaves=4, base=6, seed=300 + k)
        col = np.array(base, np.float32)
        rib = np.exp(-(u / 0.07) ** 2)
        img = col * (0.82 + 0.3 * noise[..., None]) * (1 - 0.25 * (np.abs(u) ** 3)[..., None])
        img = img * (1 - 0.25 * rib[..., None]) + (col * 1.6 + 0.04) * 0.25 * rib[..., None]
        img = img * (1 - 0.2 * (v ** 3)[..., None])
        out[:, k * cell_w:(k + 1) * cell_w] = img
    return np.clip(out, 0, 1)


def make_leaf(rng, length, width, variants, curl=0.25, fold=0.25, rows=3, cols=3) -> G.Mesh:
    m = G.leaf(length, width, curl=curl, fold=fold, rows=rows, cols=cols)
    k = int(variants[int(rng.integers(len(variants)))])
    m.uv = np.stack([(k + 0.05 + 0.9 * m.uv[:, 0]) / N_VARIANTS, 0.01 + 0.98 * m.uv[:, 1]], 1)
    return m


# --------------------------------------------------------------------------
# Far haze gradients (aerial perspective) for the distance-mapped ground and hills
# --------------------------------------------------------------------------
HAZE_LENGTH = 2600.0          # m: transmittance exp(-d / L) of the golden-hour haze
GROUND_R0, GROUND_R1 = 600.0, 16000.0


def transmittance(d) -> np.ndarray:
    return np.exp(-np.asarray(d, float) / HAZE_LENGTH)


def ground_v_to_distance(v):
    return GROUND_R0 * (GROUND_R1 / GROUND_R0) ** np.asarray(v, float)


def far_ground_mesh(rings: int = 48, sectors: int = 160) -> G.Mesh:
    """Polar ground grid from 600 m to 16 km, y = 0; uv = (angle, log-distance)."""
    v = np.linspace(0, 1, rings)
    r = ground_v_to_distance(v)
    a = np.linspace(0, TAU, sectors + 1)
    R, A = np.meshgrid(r, a, indexing="ij")
    p = np.stack([R * np.sin(A), np.zeros_like(R), -R * np.cos(A)], -1).reshape(-1, 3)
    n = np.tile([0.0, 1.0, 0.0], (len(p), 1))
    uv = np.stack(np.broadcast_arrays(a[None, :] / TAU * 40.0, v[:, None]), -1).reshape(-1, 2)
    return G.Mesh(p, n, uv, G._grid_faces(rings, sectors + 1)).oriented()


def far_ground_albedo(h: int = 256, w: int = 256):
    """u: angle (tiled 40x round the horizon), v: log distance. City grey near, fields further out."""
    v = (np.arange(h) + 0.5) / h
    d = ground_v_to_distance(v)[:, None]
    n = G.fbm(h, w, octaves=5, base=6, seed=61)
    patches = G.fbm(h, w, octaves=3, base=10, seed=62)
    city = np.array([0.13, 0.125, 0.12])
    fields = np.where(patches[..., None] > 0.55, np.array([0.12, 0.13, 0.06]), np.array([0.20, 0.17, 0.10]))
    t = np.clip((d - 2900.0) / 900.0, 0, 1)[..., None]
    alb = (city * (1 - t) + fields * t) * (0.8 + 0.4 * n[..., None])
    return alb.astype(np.float32)


def hills_mesh(seed: int = 7, cols: int = 280, rows: int = 36) -> G.Mesh:
    """Ridges between 3.4 and 10 km away, 100-650 m high; uv = (x tile, log distance like the ground)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-15000, 15000, cols)
    zs = -np.linspace(3400, 10000, rows)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    ph = rng.uniform(0, TAU, 6)

    def ridge(x, amp, wl, k):
        return amp * (0.55 + 0.25 * np.sin(x / wl + ph[k]) + 0.2 * np.sin(x / (wl * 0.37) + ph[k + 1]))

    depth = (-Z - 3400.0) / 6600.0
    front = ridge(X, 160.0, 900.0, 0) * np.clip(np.sin(np.pi * np.clip(depth / 0.35, 0, 1)), 0, 1)
    back = ridge(X, 520.0, 2400.0, 2) * np.clip((depth - 0.2) / 0.5, 0, 1) ** 0.7
    far = ridge(X, 380.0, 3800.0, 4) * np.clip((depth - 0.6) / 0.4, 0, 1)
    Y = np.maximum(np.maximum(front, back), far) * np.clip(depth / 0.06, 0, 1)
    p = np.stack([X, Y, Z], -1).reshape(-1, 3)
    f = G._grid_faces(cols, rows)
    d = np.sqrt(X ** 2 + Z ** 2)
    v = np.log(np.clip(d, GROUND_R0, GROUND_R1) / GROUND_R0) / np.log(GROUND_R1 / GROUND_R0)
    uv = np.stack([X / 1500.0, v], -1).reshape(-1, 2)
    m = G.Mesh(p, G.vertex_normals(p, f), uv, f)
    if m.n[:, 1].mean() < 0:
        m.n = -m.n
    return m.oriented()


def hills_albedo(h: int = 256, w: int = 256):
    n = G.fbm(h, w, octaves=5, base=5, seed=63)
    forest = G.fbm(h, w, octaves=3, base=8, seed=64)
    alb = np.where(forest[..., None] > 0.5, np.array([0.05, 0.08, 0.04]), np.array([0.14, 0.13, 0.08]))
    return (alb * (0.8 + 0.4 * n[..., None])).astype(np.float32)


def haze_emission(radiance, h: int = 256, w: int = 4) -> np.ndarray:
    """(1 - T(d)) x horizon radiance along v (log distance), as a linear float image (bottom-up rows)."""
    v = (np.arange(h) + 0.5) / h
    k = (1.0 - transmittance(ground_v_to_distance(v)))[:, None, None]
    return np.broadcast_to(k * np.asarray(radiance, np.float32), (h, w, 3)).astype(np.float32).copy()


def haze_weight(h: int = 256, w: int = 4) -> np.ndarray:
    """1 - T(d) along v, for blendbsdf (weight 1 -> black)."""
    v = (np.arange(h) + 0.5) / h
    k = (1.0 - transmittance(ground_v_to_distance(v)))[:, None]
    return np.broadcast_to(k, (h, w)).astype(np.float32).copy()


# --------------------------------------------------------------------------
# City
# --------------------------------------------------------------------------
TIER_EDGES = (450.0, 850.0, 1350.0, 2000.0)          # distance tiers; tier 0 has no haze


def tier_of(d: float) -> int:
    return int(np.searchsorted(TIER_EDGES, d))


def tier_distance(t: int) -> float:
    edges = (0.0,) + TIER_EDGES + (3000.0,)
    return 0.5 * (edges[t] + edges[t + 1])


CBD = np.array([-120.0, -960.0])
RIVER = (-1420.0, -1336.0)        # water z range
PARK = (-1474.0, -1276.0)          # river + banks: no blocks in here
LANDMARK_A = (-110.0, -880.0)      # tall glass tower with a spire
LANDMARK_B = (440.0, -1870.0)      # stepped stone tower beyond the river


def _kind_for(rng, d, cbd: bool, tall: bool) -> str:
    if cbd or tall:
        return str(rng.choice(["glass", "glass", "glass", "dark", "dark", "concrete_a", "concrete_b", "tiles", "stone"]))
    kinds = ["glass", "dark", "brick_a", "brick_b", "brick_c", "brick_d", "stone", "render_a", "render_b", "render_c",
             "render_d", "concrete_a", "concrete_b", "tiles"]
    w = np.array([0.05, 0.03, 0.10, 0.09, 0.08, 0.08, 0.10, 0.09, 0.08, 0.06, 0.07, 0.07, 0.05, 0.05])
    return str(rng.choice(kinds, p=w / w.sum()))


def city_layout(rng) -> list[dict]:
    """Building specs on a 110 m block grid (22 m streets). Our block (centre 0, 0) is built by hand."""
    specs = []
    P, half = STREET_PITCH, (STREET_PITCH - STREET_W) / 2
    for gz in range(-27, 12):
        cz = gz * P
        if PARK[0] < cz < PARK[1]:
            continue
        for gx in range(-31, 32):
            cx = gx * P
            if gx == 0 and gz == 0:
                continue
            if cz > -600 and abs(cx) > 1600:
                continue
            if cz > 300 and abs(cx) > 1000:
                continue
            d = math.hypot(cx, cz)
            if d > 3300:
                continue
            if cz < -1474 and rng.random() < 0.18:      # sparser beyond the river: some open blocks
                continue
            # split the 88 m block into lots
            nx, nz = int(rng.choice([1, 2, 2, 3])), int(rng.choice([1, 2, 2, 3]))
            xs = np.sort(np.concatenate([[-half, half], rng.uniform(-half + 15, half - 15, nx - 1)]))
            zs = np.sort(np.concatenate([[-half, half], rng.uniform(-half + 15, half - 15, nz - 1)]))
            cbd = np.linalg.norm(np.array([cx, cz]) - CBD) < 470
            for i in range(nx):
                for j in range(nz):
                    if rng.random() < 0.06:                 # courtyard / empty lot
                        continue
                    m = rng.uniform(0.5, 3.0, 4)
                    x0, x1 = cx + xs[i] + m[0], cx + xs[i + 1] - m[1]
                    z0, z1 = cz + zs[j] + m[2], cz + zs[j + 1] - m[3]
                    if x1 - x0 < 8 or z1 - z0 < 8:
                        continue
                    tall = rng.random() < (0.05 if d > 600 else 0.0)
                    if cbd:
                        h = float(np.clip(rng.lognormal(math.log(85), 0.45), 30, 230))
                    elif tall:
                        h = rng.uniform(55, 110)
                    elif cz < PARK[0]:
                        h = rng.uniform(9, 34)
                    else:
                        h = rng.uniform(12, 40)
                    # keep the roofscape in front of the terrace below eye level: the vista stays open
                    if -420 < cz < -30 and abs(cx) < 260:
                        h = min(h, rng.uniform(16, 36))
                    kind = _kind_for(rng, d, cbd, tall or h > 60)
                    specs.append({"box": (x0, x1, z0, z1), "h": h, "kind": kind, "d": d,
                                  "setback": h > 50 and rng.random() < 0.6, "u0": float(rng.uniform(0, 40))})
    # landmarks
    ax, az = LANDMARK_A
    specs.append({"box": (ax - 21, ax + 21, az - 21, az + 21), "h": 262.0, "kind": "glass", "d": math.hypot(ax, az),
                  "setback": False, "u0": 3.0, "spire": 48.0})
    bx, bz = LANDMARK_B
    specs.append({"box": (bx - 26, bx + 26, bz - 26, bz + 26), "h": 120.0, "kind": "stone", "d": math.hypot(bx, bz),
                  "setback": False, "u0": 1.0, "stepped": (155.0, 186.0)})
    # drop ordinary buildings overlapping the landmarks
    def clear(s):
        x0, x1, z0, z1 = s["box"]
        for (lx, lz) in (LANDMARK_A, LANDMARK_B):
            if x0 < lx + 30 and x1 > lx - 30 and z0 < lz + 30 and z1 > lz - 30 and "spire" not in s and "stepped" not in s:
                return False
        return True
    return [s for s in specs if clear(s)]


def walls(x0, x1, z0, z1, y0, y1, u0=0.0) -> G.Mesh:
    """Four outward-facing wall quads; uv = (meters along the perimeter + u0, meters above ground)."""
    corners = [(x0, z1), (x1, z1), (x1, z0), (x0, z0)]         # counter-clockwise seen from above
    normals = [(0, 0, 1), (1, 0, 0), (0, 0, -1), (-1, 0, 0)]
    out, u = Acc(), u0
    for k in range(4):
        (ax, az), (bx, bz) = corners[k], corners[(k + 1) % 4]
        L = math.hypot(bx - ax, bz - az)
        out.add(quad((ax, y0, az), (bx, y0, bz), (bx, y1, bz), (ax, y1, az), normals[k],
                     [(u, y0), (u + L, y0), (u + L, y1), (u, y1)]))
        u += L
    return out.mesh()


def roof_cap(x0, x1, z0, z1, y) -> G.Mesh:
    return quad((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), (0, 1, 0), [(x0, z0), (x1, z0), (x1, z1), (x0, z1)])


def parapet(x0, x1, z0, z1, y, h=0.9, t=0.25) -> G.Mesh:
    acc = Acc()
    acc.add(box((x1 - x0, h, t), ((x0 + x1) / 2, y + h / 2, z0 + t / 2)))
    acc.add(box((x1 - x0, h, t), ((x0 + x1) / 2, y + h / 2, z1 - t / 2)))
    acc.add(box((t, h, z1 - z0 - 2 * t), (x0 + t / 2, y + h / 2, (z0 + z1) / 2)))
    acc.add(box((t, h, z1 - z0 - 2 * t), (x1 - t / 2, y + h / 2, (z0 + z1) / 2)))
    return acc.mesh()


def water_tank(centre, y, rng) -> tuple[G.Mesh, G.Mesh]:
    """NYC-style timber tank on a steel stand: (timber, steel)."""
    r, h, leg = rng.uniform(1.4, 2.0), rng.uniform(3.0, 4.0), rng.uniform(2.0, 3.0)
    cx, cz = centre
    tank = G.lathe([(0.0, 0.0), (r, 0.0), (r, h), (r * 1.05, h), (0.0, h + r * 0.55)], 24).transformed(
        G.translate((cx, y + leg, cz)))
    steel = Acc()
    for a in np.radians([45, 135, 225, 315]):
        steel.add(tube([(cx + 0.8 * r * math.cos(a), y, cz + 0.8 * r * math.sin(a)),
                        (cx + 0.8 * r * math.cos(a), y + leg, cz + 0.8 * r * math.sin(a))], 0.09, 6))
    steel.add(box((2 * r, 0.25, 2 * r), (cx, y + leg - 0.12, cz)))
    return tank, steel.mesh()


def city_groups(rng, specs, detail_radius: float = 500.0) -> dict[str, G.Mesh]:
    """Merge the city per material: walls_<kind>_t<tier>, roof_t<tier>, roofstuff, tank, steel, cars, trees."""
    grp = Groups()
    for s in specs:
        x0, x1, z0, z1 = s["box"]
        h, t, k = s["h"], tier_of(s["d"]), s["kind"]
        wkey = f"walls_{k}_t{t}"
        if s.get("stepped"):
            y = 0.0
            tiers = [(0.0, h, 0.0), (h, s["stepped"][0], 6.0), (s["stepped"][0], s["stepped"][1], 13.0)]
            for (ya, yb, inset) in tiers:
                grp[wkey].add(walls(x0 + inset, x1 - inset, z0 + inset, z1 - inset, ya, yb, s["u0"]))
                grp[f"roof_t{t}"].add(roof_cap(x0 + inset, x1 - inset, z0 + inset, z1 - inset, yb))
            top = s["stepped"][1]
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            grp[f"spire_t{t}"].add(G.lathe([(0, top), (7.0, top), (5.5, top + 10), (2.0, top + 22), (0.3, top + 34), (0, top + 36)], 24),
                                   G.translate((cx, 0, cz)))
            continue
        if s["setback"]:
            yb = h * rng.uniform(0.55, 0.8)
            inset = min(x1 - x0, z1 - z0) * rng.uniform(0.12, 0.22)
            grp[wkey].add(walls(x0, x1, z0, z1, 0.0, yb, s["u0"]))
            grp[f"roof_t{t}"].add(roof_cap(x0, x1, z0, z1, yb))
            grp[wkey].add(walls(x0 + inset, x1 - inset, z0 + inset, z1 - inset, yb, h, s["u0"]))
            rx0, rx1, rz0, rz1 = x0 + inset, x1 - inset, z0 + inset, z1 - inset
        else:
            grp[wkey].add(walls(x0, x1, z0, z1, 0.0, h, s["u0"]))
            rx0, rx1, rz0, rz1 = x0, x1, z0, z1
        grp[f"roof_t{t}"].add(roof_cap(rx0, rx1, rz0, rz1, h))
        if s.get("spire"):
            cx, cz = (rx0 + rx1) / 2, (rz0 + rz1) / 2
            grp[f"spire_t{t}"].add(box((14.0, 9.0, 14.0), (cx, h + 4.5, cz)))
            grp[f"spire_t{t}"].add(G.lathe([(0, h + 9), (1.6, h + 9), (0.9, h + 9 + s["spire"] * 0.6),
                                            (0.25, h + 9 + s["spire"]), (0, h + 9.5 + s["spire"])], 16), G.translate((cx, 0, cz)))
        if s["d"] < detail_radius:
            grp["roofstuff"].add(parapet(rx0, rx1, rz0, rz1, h, rng.uniform(0.6, 1.1)))
            for _ in range(int(rng.integers(1, 4))):
                w, dd, hh = rng.uniform(1.5, 4.0), rng.uniform(1.2, 3.0), rng.uniform(1.0, 2.2)
                cx, cz = rng.uniform(rx0 + w, rx1 - w), rng.uniform(rz0 + dd, rz1 - dd)
                grp["roofstuff"].add(box((w, hh, dd), (cx, h + hh / 2, cz)))
            if rng.random() < 0.35:
                cx, cz = rng.uniform(rx0 + 3, rx1 - 3), rng.uniform(rz0 + 3, rz1 - 3)
                tank, steel = water_tank((cx, cz), h, rng)
                grp["tank"].add(tank)
                grp["steel"].add(steel)
            if rng.random() < 0.3:
                cx, cz = rng.uniform(rx0 + 2, rx1 - 2), rng.uniform(rz0 + 2, rz1 - 2)
                grp["roofstuff"].add(box((3.0, 2.8, 3.4), (cx, h + 1.4, cz)))
    return grp


def _at_crossing(s: float) -> bool:
    """True near a cross street: street centre lines sit at k * 110 + 55."""
    m = (s - STREET_PITCH / 2) % STREET_PITCH
    return min(m, STREET_PITCH - m) < STREET_W / 2 + 2.0


def street_life(rng, radius: float = 420.0) -> Groups:
    """Street trees on the sidewalks and parked/moving cars on near streets."""
    grp = Groups()
    P = STREET_PITCH
    n_lines = int(radius // P) + 1
    for axis in (0, 1):
        for k in range(-n_lines, n_lines + 1):
            c = k * P + P / 2                                   # street centre line (between blocks)
            for s in np.arange(-radius, radius, 14.0):
                if rng.random() < 0.35:
                    continue
                for side in (-1, 1):
                    off = side * (STREET_W / 2 - 1.5)
                    x, z = (c + off, s) if axis == 0 else (s, c + off)
                    if math.hypot(x, z) > radius or (abs(x) < 46 and abs(z) < 46):
                        continue
                    if _at_crossing(s):
                        continue
                    h = rng.uniform(5.0, 8.0)
                    grp["trunk"].add(tube([(x, 0, z), (x, h * 0.45, z)], 0.18, 6))
                    for q in range(2):
                        r = rng.uniform(2.0, 3.2)
                        off3 = rng.normal(0, 0.7, 3) * [1, 0.3, 1]
                        grp["canopy"].add(G.ellipsoid((r, r * 0.8, r), 7, 10), G.translate((x + off3[0], h * 0.7 + off3[1], z + off3[2])))
            for s in np.arange(-radius, radius, 9.0):
                if rng.random() < 0.72:
                    continue
                lane = rng.choice([-5.5, -2.0, 2.0, 5.5])
                x, z = (c + lane, s) if axis == 0 else (s, c + lane)
                if math.hypot(x, z) > radius or (abs(x) < 46 and abs(z) < 46):
                    continue
                if _at_crossing(s):
                    continue
                L, W = rng.uniform(4.0, 4.9), rng.uniform(1.75, 1.9)
                yaw = 0.0 if axis == 0 else 90.0
                col = f"car_{int(rng.integers(6))}"
                m = G.compose(G.translate((x, 0, z)), G.rotate((0, 1, 0), yaw))
                grp[col].add(superellipsoid((W / 2, 0.42, L / 2), 0.2, 6, 10), m @ G.translate((0, 0.62, 0)))
                grp["car_glass"].add(superellipsoid((W / 2 * 0.86, 0.36, L / 2 * 0.5), 0.35, 6, 10), m @ G.translate((0, 1.05, -0.15 * L / 2)))
                for wx in (-W / 2 + 0.1, W / 2 - 0.1):
                    for wz in (-L / 2 + 0.75, L / 2 - 0.75):
                        grp["tyre"].add(G.lathe([(0, -0.12), (0.33, -0.12), (0.34, 0.0), (0.33, 0.12), (0, 0.12)], 8),
                                        m @ G.translate((wx, 0.33, wz)) @ G.rotate((0, 0, 1), 90))
    return grp


def river_park(rng) -> Groups:
    grp = Groups()
    x0, x1 = -16000.0, 16000.0
    grp["water"].add(quad((x0, 0.4, RIVER[0]), (x1, 0.4, RIVER[0]), (x1, 0.4, RIVER[1]), (x0, 0.4, RIVER[1]), (0, 1, 0),
                          [(x0, RIVER[0]), (x1, RIVER[0]), (x1, RIVER[1]), (x0, RIVER[1])]))
    for (za, zb) in ((PARK[0], RIVER[0]), (RIVER[1], PARK[1])):
        grp["park"].add(quad((x0, 0.15, za), (x1, 0.15, za), (x1, 0.15, zb), (x0, 0.15, zb), (0, 1, 0),
                             [(x0, za), (x1, za), (x1, zb), (x0, zb)]))
        for x in np.arange(-3300.0, 3300.0, 11.0):
            if rng.random() < 0.45:
                continue
            z = rng.uniform(min(za, zb) + 4, max(za, zb) - 4)
            r = rng.uniform(3.0, 6.0)
            grp["park_tree"].add(G.ellipsoid((r, r * 0.85, r), 6, 9), G.translate((x, r * 1.3, z)))
    for bx in (0.0, 880.0, -1100.0):                       # bridges: deck, piers, parapets
        zc, L = (RIVER[0] + RIVER[1]) / 2, PARK[1] - PARK[0] + 8
        grp["bridge"].add(box((18.0, 1.6, L), (bx, 8.0, zc)))
        grp["bridge"].add(box((18.6, 1.0, L), (bx, 9.2, zc)))
        for pz in np.linspace(RIVER[0] + 12, RIVER[1] - 12, 3):
            grp["bridge"].add(box((14.0, 8.0, 4.0), (bx, 4.0, pz)))
    return grp


# --------------------------------------------------------------------------
# Terrace furniture
# --------------------------------------------------------------------------
def railing(p0, p1, y, height=1.05, post_every=1.6) -> dict[str, G.Mesh]:
    """Square steel posts, a timber handrail and 6 stainless cables between two points at deck height y."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    d = (p1 - p0) / L
    n = max(int(math.ceil(L / post_every)), 1)
    posts, cables, rail = Acc(), Acc(), Acc()
    yaw = math.degrees(math.atan2(d[0], d[2]))
    for i in range(n + 1):
        c = p0 + d * L * i / n
        posts.add(box((0.05, height, 0.05), (c[0], y + height / 2, c[2])))
    for k in range(6):
        yy = y + 0.12 + k * (height - 0.25) / 5
        cables.add(tube([(p0[0], yy, p0[2]), (p1[0], yy, p1[2])], 0.0032, 6))
    rail.add(box((0.11, 0.05, L + 0.06), (0, 0, 0)), G.compose(G.translate(((p0 + p1) / 2)[[0, 1, 2]] * [1, 0, 1] + [0, y + height + 0.025, 0]),
                                                                    G.rotate((0, 1, 0), yaw)))
    return {"post": posts.mesh(), "cable": cables.mesh(), "handrail": rail.mesh()}


def sofa(width=2.0, depth=0.85) -> dict[str, G.Mesh]:
    """Outdoor sofa facing +z: teak frame, thick cushions."""
    frame, cush = Acc(), Acc()
    frame.add(box((width, 0.12, depth), (0, 0.18, 0)))
    for sx in (-1, 1):
        frame.add(box((0.12, 0.62, depth), (sx * (width / 2 - 0.06), 0.31, 0)))
        for sz in (-1, 1):
            frame.add(box((0.06, 0.12, 0.06), (sx * (width / 2 - 0.06), 0.06, sz * (depth / 2 - 0.05))))
    frame.add(box((width - 0.24, 0.5, 0.08), (0, 0.5, -depth / 2 + 0.04)))
    seats = 2 if width < 2.4 else 3
    sw = (width - 0.26) / seats
    for i in range(seats):
        x = -width / 2 + 0.13 + sw * (i + 0.5)
        cush.add(superellipsoid((sw / 2 - 0.01, 0.075, depth / 2 - 0.08), 0.2, 10, 16), G.translate((x, 0.32, 0.03)))
        cush.add(superellipsoid((sw / 2 - 0.02, 0.24, 0.085), 0.25, 10, 16),
                 G.compose(G.translate((x, 0.62, -depth / 2 + 0.17)), G.rotate((1, 0, 0), -12)))
    return {"teak": frame.mesh(), "cushion": cush.mesh()}


def coffee_table(w=1.1, d=0.6, h=0.4) -> G.Mesh:
    acc = Acc()
    for i in range(9):
        acc.add(box((w, 0.025, d / 9 - 0.008), (0, h - 0.0125, -d / 2 + d / 9 * (i + 0.5))))
    for sx in (-1, 1):
        for sz in (-1, 1):
            acc.add(box((0.05, h - 0.03, 0.05), (sx * (w / 2 - 0.05), (h - 0.03) / 2, sz * (d / 2 - 0.05))))
    return acc.mesh()


def bar_stool() -> dict[str, G.Mesh]:
    seat = G.lathe([(0, 0), (0.19, 0), (0.2, 0.02), (0.19, 0.05), (0, 0.055)], 32).transformed(G.translate((0, 0.74, 0)))
    legs = Acc()
    for a in np.radians([45, 135, 225, 315]):
        legs.add(tube([(0.07 * math.cos(a), 0.74, 0.07 * math.sin(a)), (0.2 * math.cos(a), 0.0, 0.2 * math.sin(a))], 0.013, 6))
    legs.add(G.sweep([(0.17 * math.cos(a), 0.28, 0.17 * math.sin(a)) for a in np.linspace(0, TAU, 33)[:-1]], 0.009, 6, closed=True))
    return {"seat": seat, "metal": legs.mesh()}


def umbrella(radius=1.45, height=2.45, ribs=8) -> dict[str, G.Mesh]:
    """Open market parasol: pole, a shallow octagonal canvas cone, ribs and a heavy base."""
    pole = tube([(0, 0.05, 0), (0, height + 0.25, 0)], 0.022, 10)
    pts = []
    canvas = Acc()
    for i in range(ribs):
        a0, a1 = TAU * i / ribs, TAU * (i + 1) / ribs
        c0 = (radius * math.cos(a0), height - 0.28, radius * math.sin(a0))
        c1 = (radius * math.cos(a1), height - 0.28, radius * math.sin(a1))
        apex = (0, height + 0.12, 0)
        canvas.add(G.Mesh(np.array([apex, c0, c1]), np.tile([0, 1, 0], (3, 1)), np.array([[0.5, 1], [0, 0], [1, 0]]),
                          np.array([[0, 2, 1]])))
        # valance flap
        canvas.add(quad((c0[0], c0[1], c0[2]), (c1[0], c1[1], c1[2]), (c1[0], c1[1] - 0.16, c1[2]), (c0[0], c0[1] - 0.16, c0[2]),
                        (math.cos((a0 + a1) / 2), 0, math.sin((a0 + a1) / 2)), [(0, 1), (1, 1), (1, 0), (0, 0)]))
        pts.append(c0)
    m = canvas.mesh()
    ribs_m = Acc()
    for c in pts:
        ribs_m.add(tube([(0, height + 0.08, 0), c], 0.008, 5))
        ribs_m.add(tube([(0, height - 0.55, 0), (c[0] * 0.5, height - 0.12, c[2] * 0.5)], 0.006, 5))
    base = G.lathe([(0, 0), (0.32, 0), (0.32, 0.05), (0.05, 0.09), (0, 0.09)], 32)
    return {"canvas": m, "pole": merge([pole, ribs_m.mesh()]), "base": base}


def bottle_rows(rng, x, z0, z1, y_shelves, n_per=14) -> dict[str, G.Mesh]:
    """Bottles on back-bar shelves along z (facing -x)."""
    from props import bottle
    b = bottle()
    out = Groups()
    for y in y_shelves:
        for z in np.linspace(z0 + 0.08, z1 - 0.08, n_per):
            if rng.random() < 0.12:
                continue
            k = int(rng.integers(5))
            s = rng.uniform(0.85, 1.12)
            out[f"bottle_{k}"].add(b, G.compose(G.translate((x + rng.uniform(-0.04, 0.04), y, z + rng.uniform(-0.02, 0.02))),
                                                G.scale((1.0, s, 1.0))))
    return out.meshes()


def festoon(p0, p1, sag, spacing=0.55, bulb_r=0.032) -> tuple[G.Mesh, G.Mesh, list]:
    """A catenary-ish cable between two points with pendant globe bulbs: (cable, bulbs, bulb centres)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    n = max(int(L / spacing), 2)
    t = np.linspace(0, 1, 4 * n + 1)
    path = p0[None] + (p1 - p0)[None] * t[:, None] - np.array([0, 1, 0]) * (4 * sag * t * (1 - t))[:, None]
    cable = tube(path, 0.0035, 5)
    bulbs, centres, sockets = Acc(), [], Acc()
    for k in range(1, n):
        q = path[4 * k]
        c = q - [0, 0.075, 0]
        sockets.add(tube([q, q - [0, 0.04, 0]], 0.009, 6))
        bulbs.add(G.ellipsoid((bulb_r, bulb_r * 1.15, bulb_r), 8, 12), G.translate(c))
        centres.append(c.tolist())
    return merge([cable, sockets.mesh()]), bulbs.mesh(), centres


# --------------------------------------------------------------------------
# Plants (append to `grp` in world space)
# --------------------------------------------------------------------------
def _leaves_along(grp, rng, path, every, size, variants, spread=0.6):
    """Opposite pairs of narrow leaves along a twig polyline, angled outward and toward the tip."""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    for s in np.arange(every * 0.5, cum[-1], every):
        i = min(int(np.searchsorted(cum, s)), len(path) - 1)
        tang = unit(path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)])
        side = unit(np.cross(tang, UP + rng.normal(0, 0.3, 3)))
        for sgn in (1.0, -1.0):
            d = unit(tang * 0.6 + sgn * side * spread + rng.normal(0, 0.25, 3))
            L = size[0] * rng.uniform(0.8, 1.2)
            lf = make_leaf(rng, L, size[1] * rng.uniform(0.85, 1.15), variants, curl=0.12, fold=0.2, rows=3, cols=2)
            grp["leaf"].add(lf, frame(path[i], d))


def olive_tree(grp, rng, base, height=2.6):
    """Gnarled olive: twisting trunk, 3-4 limbs, many arching twigs, and a dense crown of narrow
    silver-green leaves set in pairs along the twigs (about 6k leaves)."""
    base = np.asarray(base, float)
    t = np.linspace(0, 1, 12)
    lean = rng.uniform(-0.18, 0.18, 2)
    trunk = np.column_stack([base[0] + 0.04 * np.sin(t * 6 + rng.uniform(0, 6)) + lean[0] * t ** 2,
                             base[1] + height * 0.4 * t,
                             base[2] + 0.04 * np.cos(t * 5 + rng.uniform(0, 6)) + lean[1] * t ** 2])
    grp["bark"].add(G.sweep(trunk, np.linspace(0.12, 0.08, len(trunk)) * (1 + 0.18 * np.sin(t * 19)), 10))
    top = trunk[-1]
    crown_c = top + [0, height * 0.42, 0]
    crown_r = np.array([0.85, 0.62, 0.85]) * (height / 2.6)
    n_limbs = int(rng.integers(3, 5))
    for i in range(n_limbs):
        a = TAU * (i + rng.uniform(-0.25, 0.25)) / n_limbs
        end = crown_c + [0.45 * crown_r[0] * math.cos(a), rng.uniform(-0.1, 0.25), 0.45 * crown_r[2] * math.sin(a)]
        limb = bezier([top, top + [0.15 * math.cos(a), 0.35, 0.15 * math.sin(a)], end], 10)
        grp["bark"].add(G.sweep(limb, np.linspace(0.065, 0.025, len(limb)), 7))
        for j in range(20):                                    # arching twigs to points on the crown surface
            src = limb[int(rng.integers(4, 10))]
            v = unit(rng.normal(size=3) * [1, 0.7, 1] + [math.cos(a) * 0.8, 0.35, math.sin(a) * 0.8])
            dst = crown_c + v * crown_r * rng.uniform(0.75, 1.05)
            mid = (src + dst) / 2 + [0, rng.uniform(0.05, 0.15), 0]
            twig = bezier([src, mid, dst], 9)
            grp["bark"].add(G.sweep(twig, np.linspace(0.014, 0.004, len(twig)), 5))
            _leaves_along(grp, rng, twig[3:], 0.024, (0.085, 0.017), OLIVE)
            for k in range(5):                                 # side shoots that fill the crown
                s0 = twig[int(rng.integers(3, 8))]
                d2 = unit(rng.normal(size=3) * [1, 0.6, 1] + v)
                shoot = np.stack([s0, s0 + d2 * 0.16 + [0, 0.03, 0], s0 + d2 * 0.32 - [0, 0.05, 0]])
                grp["bark"].add(G.sweep(shoot, [0.005, 0.004, 0.002], 4))
                _leaves_along(grp, rng, shoot, 0.024, (0.08, 0.016), OLIVE)
    return float(crown_c[1])


def lavender(grp, rng, base, n=60, height=0.5):
    base = np.asarray(base, float)
    for i in range(n):
        yaw, el = rng.uniform(0, TAU), rng.uniform(1.05, 1.45)
        d = unit([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        h = height * rng.uniform(0.7, 1.15)
        tip = base + d * h + rng.normal(0, 0.02, 3)
        stem = bezier([base + rng.normal(0, 0.03, 3) * [1, 0, 1], base + d * h * 0.5, tip], 6)
        grp["stem"].add(G.sweep(stem, 0.0018, 4))
        for k in range(9):                                    # flower spike: stacked purple whorls
            c = tip - d * (0.013 * k) + rng.normal(0, 0.002, 3)
            grp["lavender_flower"].add(G.ellipsoid((0.008, 0.011, 0.008), 3, 4), G.translate(c))
    for _ in range(90):                                       # grey foliage at the base
        p = base + rng.normal(0, 1, 3) * [0.09, 0.0, 0.09] + [0, rng.uniform(0.02, 0.14), 0]
        lf = make_leaf(rng, rng.uniform(0.05, 0.08), 0.008, LAV, curl=0.2, fold=0.1, rows=3, cols=2)
        grp["leaf"].add(lf, frame(p, unit([rng.normal(), rng.uniform(0.3, 1.0), rng.normal()])))


def grass(grp, rng, base, n=80, length=0.8):
    base = np.asarray(base, float)
    for i in range(n):
        yaw, el = rng.uniform(0, TAU), rng.uniform(0.95, 1.5)
        d = unit([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        m = make_leaf(rng, length * rng.uniform(0.6, 1.2), 0.016, GRASS, curl=rng.uniform(0.3, 0.8), fold=0.0, rows=5, cols=2)
        grp["leaf"].add(m, frame(base + rng.normal(0, 0.035, 3) * [1, 0, 1], d))
    for i in range(int(n * 0.3)):                             # seed plumes
        yaw = rng.uniform(0, TAU)
        tip = base + [0.15 * math.cos(yaw), length * rng.uniform(0.9, 1.25), 0.15 * math.sin(yaw)]
        grp["stem"].add(G.sweep(np.stack([base, tip]), 0.0015, 4))
        grp["plume"].add(G.ellipsoid((0.012, 0.07, 0.012), 4, 6), G.translate(tip))


def vine(grp, rng, path, leaf_len=0.09, every=0.045):
    path = np.asarray(path, float)
    grp["stem"].add(G.sweep(path, 0.006, 5))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    for s in np.arange(0.0, cum[-1], every):
        i = min(int(np.searchsorted(cum, s)), len(path) - 1)
        p = path[i] + rng.normal(0, 0.03, 3)
        d = unit([rng.normal(), rng.uniform(-0.9, 0.4), rng.normal()])
        grp["leaf"].add(make_leaf(rng, leaf_len * rng.uniform(0.7, 1.3), leaf_len * 0.85, VINE, curl=0.3, fold=0.3), frame(p, d))
