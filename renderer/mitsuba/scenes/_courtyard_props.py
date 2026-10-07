"""Geometry and plant builders for scenes/courtyard.py.

Numpy meshes (procedural.Mesh), meters, y up. Walls are built in a WALL-LOCAL frame: u along +x from the
wall's start, v up, the courtyard face at z = 0 facing +z and the wall body toward -z; the scene places
each wall with one matrix. Openings are real holes with reveals (jambs, sills, arch intrados), so arches
and windows have depth and silhouette edges. Plant builders append to a `Groups` (one merged mesh per
material) like the greenhouse does, so tens of thousands of leaves stay a handful of shapes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import procedural as G
from props import arc, bezier

TAU = 2 * math.pi
UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------
# Mesh accumulation and placement
# --------------------------------------------------------------------------
class Acc:
    """Collects many small meshes and concatenates once (Mesh += is quadratic)."""

    def __init__(self):
        self.p, self.n, self.uv, self.f, self.off = [], [], [], [], 0

    def add(self, mesh: G.Mesh, matrix=None) -> None:
        if matrix is not None:
            mesh = mesh.transformed(matrix)
        self.add_raw(mesh.p, mesh.n, mesh.uv, mesh.f)

    def add_raw(self, p, n, uv, f) -> None:
        self.p.append(p), self.n.append(n), self.uv.append(uv), self.f.append(f + self.off)
        self.off += len(p)

    def place(self, mesh: G.Mesh, rot: np.ndarray, origin, s: float) -> None:
        """Fast rigid placement with uniform scale (no inverse, no determinant): rot is 3x3 orthonormal."""
        self.add_raw(mesh.p @ (rot * s).T + origin, mesh.n @ rot.T, mesh.uv, mesh.f)

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


def basis(forward, up=UP) -> np.ndarray:
    """3x3 rotation: local +z -> forward, +y -> up (made perpendicular), +x -> right-handed."""
    f = unit(forward)
    r = np.cross(up, f)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross([1.0, 0.0, 0.0] if abs(f[0]) < 0.9 else [0.0, 0.0, 1.0], f)
    r = unit(r)
    return np.column_stack([r, np.cross(f, r), f])


def frame(origin, forward, up=UP) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = basis(forward, up)
    m[:3, 3] = origin
    return m


def quad(p0, p1, p2, p3, uv=None) -> G.Mesh:
    """Planar quad, counter-clockwise seen from the side it faces. uv defaults to edge lengths in meters."""
    p = np.array([p0, p1, p2, p3], float)
    n = unit(np.cross(p[1] - p[0], p[3] - p[0]))
    if uv is None:
        a, b = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0])
        uv = [(0, 0), (a, 0), (a, b), (0, b)]
    return G.Mesh(p, np.tile(n, (4, 1)), np.asarray(uv, float), np.array([[0, 1, 2], [0, 2, 3]]))


def grid_faces(rows: int, cols: int) -> np.ndarray:
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    i0 = (r * cols + c).ravel()
    return np.concatenate([np.stack([i0, i0 + cols, i0 + 1], 1), np.stack([i0 + 1, i0 + cols, i0 + cols + 1], 1)])


def box(size, centre) -> G.Mesh:
    return G.box(size, centre)


def tube(points, radius, sides=8) -> G.Mesh:
    return G.sweep(np.asarray(points, float), radius, sides)


def extrude(poly, z0: float, z1: float) -> G.Mesh:
    """Convex polygon (N x 2, (u, v), counter-clockwise seen from +z) extruded between z0 < z1.
    Front face at z1 facing +z, back at z0, sides; uv in meters (u, v)."""
    poly = np.asarray(poly, float)
    n = len(poly)
    acc = Acc()
    c = poly.mean(0)
    for z, sign in ((z1, 1.0), (z0, -1.0)):
        p = np.vstack([[c[0], c[1], z], np.column_stack([poly, np.full(n, z)])])
        f = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
        if sign < 0:
            f = f[:, ::-1]
        acc.add_raw(p, np.tile([0.0, 0.0, sign], (n + 1, 1)), p[:, :2].copy(), f)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        acc.add(quad([a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]))
    return acc.mesh().oriented()


# --------------------------------------------------------------------------
# Walls with real openings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Opening:
    u0: float
    u1: float
    v0: float
    v1: float                 # springline (arch) or head (rectangular)
    arch: bool = False
    kind: str = "window"      # window | door | arcade | balcony | passage

    @property
    def c(self) -> float:
        return 0.5 * (self.u0 + self.u1)

    @property
    def r(self) -> float:
        return 0.5 * (self.u1 - self.u0)

    @property
    def apex(self) -> float:
        return self.v1 + (self.r if self.arch else 0.0)

    def top(self, u: float) -> float:
        if not self.arch:
            return self.v1
        d = min(abs(u - self.c), self.r)
        return self.v1 + math.sqrt(max(self.r * self.r - d * d, 0.0))

    def tex(self):
        return (self.u0, self.u1, self.v0, self.v1, self.arch)


def wall(L: float, H: float, T: float, openings, seg: int = 24) -> dict[str, G.Mesh]:
    """Wall slab [0, L] x [0, H] x [-T, 0] with holes. Parts: face (z = 0, uv meters), back (z = -T,
    uv mirrored so text reads right from behind), reveal (jambs, sills, heads, arch intrados), cap."""
    cuts = {0.0, L}
    for o in openings:
        cuts.update((o.u0, o.u1))
        if o.arch:
            cuts.update(np.linspace(o.u0, o.u1, seg + 1).tolist())
    cuts = sorted(c for c in cuts if -1e-9 <= c <= L + 1e-9)
    front, back, rev, cap = Acc(), Acc(), Acc(), Acc()

    def face(a, b, ya, yb, ta, tb):
        front.add(quad([a, ya, 0], [b, yb, 0], [b, tb, 0], [a, ta, 0], [(a, ya), (b, yb), (b, tb), (a, ta)]))
        back.add(quad([b, yb, -T], [a, ya, -T], [a, ta, -T], [b, tb, -T],
                      [(L - b, yb), (L - a, ya), (L - a, ta), (L - b, tb)]))

    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-6:
            continue
        mid = 0.5 * (a + b)
        cover = sorted((o for o in openings if o.u0 - 1e-9 <= mid <= o.u1 + 1e-9), key=lambda o: o.v0)
        ya = yb = 0.0
        for o in cover:
            if o.v0 > max(ya, yb) + 1e-6:
                face(a, b, ya, yb, o.v0, o.v0)
            ya, yb = o.top(a), o.top(b)
        if H > max(ya, yb) + 1e-6:
            face(a, b, ya, yb, H, H)

    for o in openings:
        rev.add(quad([o.u0, o.v0, 0], [o.u0, o.v0, -T], [o.u0, o.v1, -T], [o.u0, o.v1, 0],
                     [(0, o.v0), (T, o.v0), (T, o.v1), (0, o.v1)]))
        rev.add(quad([o.u1, o.v0, -T], [o.u1, o.v0, 0], [o.u1, o.v1, 0], [o.u1, o.v1, -T],
                     [(T, o.v0), (0, o.v0), (0, o.v1), (T, o.v1)]))
        if o.v0 > 1e-3:
            rev.add(quad([o.u0, o.v0, 0], [o.u1, o.v0, 0], [o.u1, o.v0, -T], [o.u0, o.v0, -T]))
        if not o.arch:
            rev.add(quad([o.u0, o.v1, -T], [o.u1, o.v1, -T], [o.u1, o.v1, 0], [o.u0, o.v1, 0]))
        else:
            ang = np.linspace(math.pi, 0.0, seg + 1)
            q = np.column_stack([o.c + o.r * np.cos(ang), o.v1 + o.r * np.sin(ang)])
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
            for k in range(seg):
                (u_a, v_a), (u_b, v_b) = q[k], q[k + 1]
                rev.add(quad([u_a, v_a, -T], [u_b, v_b, -T], [u_b, v_b, 0], [u_a, v_a, 0],
                             [(s[k], T), (s[k + 1], T), (s[k + 1], 0), (s[k], 0)]))
    cap.add(quad([0, H, 0], [L, H, 0], [L, H, -T], [0, H, -T]))
    cap.add(quad([0, 0, -T], [0, 0, 0], [0, H, 0], [0, H, -T]))
    cap.add(quad([L, 0, 0], [L, 0, -T], [L, H, -T], [L, H, 0]))
    return {"face": front.mesh(), "back": back.mesh(), "reveal": rev.mesh(), "cap": cap.mesh()}


# --------------------------------------------------------------------------
# Openings: windows, doors, shutters, surrounds, sills, ledges
# --------------------------------------------------------------------------
def arch_points(o: Opening, radius: float, n: int = 24, a0: float = math.pi, a1: float = 0.0) -> np.ndarray:
    ang = np.linspace(a0, a1, n)
    return np.column_stack([o.c + radius * np.cos(ang), o.v1 + radius * np.sin(ang)])


def window(o: Opening, depth: float = 0.17, fw: float = 0.06, panes: int = 3) -> dict[str, G.Mesh]:
    """Casement window set back `depth` into the reveal: frame, glazing bars and one dark pane surface.
    Arched openings get a semicircular fanlight above a transom at the springline."""
    frame_acc, glass = Acc(), Acc()
    z = -depth
    w, h = o.u1 - o.u0, o.v1 - o.v0
    for u in (o.u0 + fw / 2, o.u1 - fw / 2):
        frame_acc.add(box((fw, h, 0.07), (u, o.v0 + h / 2, z)))
    frame_acc.add(box((w, fw * 1.3, 0.08), (o.c, o.v0 + fw * 0.65, z)))
    frame_acc.add(box((w, fw, 0.07), (o.c, o.v1 - fw / 2, z)))
    frame_acc.add(box((0.045, h - 2 * fw, 0.06), (o.c, o.v0 + h / 2, z)))
    for k in range(1, panes):
        frame_acc.add(box((w - 2 * fw, 0.028, 0.045), (o.c, o.v0 + fw + (h - 2 * fw) * k / panes, z)))
    glass.add(quad([o.u0, o.v0, z - 0.01], [o.u1, o.v0, z - 0.01], [o.u1, o.v1, z - 0.01], [o.u0, o.v1, z - 0.01]))
    if o.arch:
        pts = arch_points(o, o.r - fw / 2, 25)
        frame_acc.add(tube(np.column_stack([pts, np.full(len(pts), z)]), fw / 2, 6))
        for a in (math.radians(45), math.radians(90), math.radians(135)):
            frame_acc.add(tube([(o.c, o.v1, z), (o.c + (o.r - fw) * math.cos(a), o.v1 + (o.r - fw) * math.sin(a), z)],
                               0.013, 5))
        rim = arch_points(o, o.r, 25)
        p = np.vstack([[o.c, o.v1, z - 0.01], np.column_stack([rim, np.full(len(rim), z - 0.01)])])
        f = np.array([[0, i + 2, i + 1] for i in range(len(rim) - 1)])
        glass.add_raw(p, np.tile([0.0, 0.0, 1.0], (len(p), 1)), p[:, :2].copy(), f)
    return {"frame": frame_acc.mesh(), "glass": glass.mesh().oriented()}


def door(o: Opening, depth: float = 0.2, thick: float = 0.07) -> dict[str, G.Mesh]:
    """Two plank leaves (arched head if the opening is arched), a ring pull, an iron strap per leaf."""
    leaves, iron = Acc(), Acc()
    z1 = -depth
    z0 = z1 - thick
    for side in (-1, 1):
        if o.arch:
            if side < 0:
                q = arch_points(o, o.r, 12, math.pi / 2, math.pi)
                poly = [(o.u0, o.v0), (o.c, o.v0), (o.c, o.apex)] + [tuple(p) for p in q[1:]]
            else:
                q = arch_points(o, o.r, 12, 0.0, math.pi / 2)
                poly = [(o.c, o.v0), (o.u1, o.v0)] + [tuple(p) for p in q[:-1]] + [(o.c, o.apex)]
        else:
            poly = ([(o.u0, o.v0), (o.c, o.v0), (o.c, o.v1), (o.u0, o.v1)] if side < 0
                    else [(o.c, o.v0), (o.u1, o.v0), (o.u1, o.v1), (o.c, o.v1)])
        leaves.add(extrude(np.array(poly) + [[side * 0.004, 0.0]] * len(poly), z0, z1))
        cx = o.c + side * 0.18
        iron.add(tube(np.array([(cx + 0.045 * math.cos(a), o.v0 + 1.05 + 0.045 * math.sin(a), z1 + 0.012)
                                for a in np.linspace(0, TAU, 17)]), 0.006, 5))
        for vy in (o.v0 + 0.35, o.v0 + (o.v1 - o.v0) * 0.8):
            u_a, u_b = (o.u0 + 0.05, o.c - 0.05) if side < 0 else (o.c + 0.05, o.u1 - 0.05)
            iron.add(box((u_b - u_a, 0.05, 0.012), ((u_a + u_b) / 2, vy, z1 + 0.006)))
    return {"door": leaves.mesh(), "iron": iron.mesh()}


def shutter_leaf(W: float, Hs: float, t: float = 0.035) -> G.Mesh:
    """Louvred shutter leaf in hinge-local coords: x in [0, W], y in [0, Hs], z in [-t, 0]."""
    acc = Acc()
    sw = 0.055
    for x in (sw / 2, W - sw / 2):
        acc.add(box((sw, Hs, t), (x, Hs / 2, -t / 2)))
    for y, hh in ((0.045, 0.09), (Hs - 0.035, 0.07), (Hs * 0.5, 0.06)):
        acc.add(box((W - 2 * sw, hh, t * 0.9), (W / 2, y, -t / 2)))
    slat = G.box((W - 2 * sw, 0.052, 0.007))
    rot = G.rotate((1, 0, 0), 38.0)
    for lo, hi in ((0.09, Hs * 0.5 - 0.03), (Hs * 0.5 + 0.03, Hs - 0.07)):
        for y in np.arange(lo + 0.025, hi - 0.02, 0.042):
            acc.add(slat, G.compose(G.translate((W / 2, y, -t / 2)), rot))
    return acc.mesh()


def shutters(o: Opening, state: str, angle: float):
    """Placement matrices (wall-local) for the two leaves of a window. state: open | ajar | closed."""
    W = o.r - 0.004
    if state == "closed":
        zh, a = -0.045, 0.0
    else:
        zh, a = 0.006, angle
    left = G.compose(G.translate((o.u0, o.v0, zh)), G.rotate((0, 1, 0), -a))
    right = G.compose(G.translate((o.u1, o.v0, zh)), G.rotate((0, 1, 0), a), G.scale((-1, 1, 1)))
    return [left, right], W


def shutter_footprint(o: Opening, state: str, angle: float):
    """Area of wall face covered by open shutters, for climbers to avoid: list of (u0, u1, v0, v1)."""
    if state == "closed":
        return []
    reach = o.r * abs(math.cos(math.radians(angle)))
    if angle < 150:
        reach = o.r * 0.35
    return [(o.u0 - reach - 0.04, o.u0, o.v0 - 0.05, o.v1 + 0.05), (o.u1, o.u1 + reach + 0.04, o.v0 - 0.05, o.v1 + 0.05)]


def surround(o: Opening, bw: float = 0.16, proud: float = 0.03, seg: int = 24) -> G.Mesh:
    """Raised stone band around an opening: jambs plus a lintel (rectangular) or voussoirs (arched)."""
    acc = Acc()
    v_top = o.v1 if o.arch else o.v1 + bw
    for u in (o.u0 - bw / 2, o.u1 + bw / 2):
        acc.add(box((bw, v_top - o.v0, proud), (u, (o.v0 + v_top) / 2, proud / 2)))
    if not o.arch:
        acc.add(box((o.u1 - o.u0 + 2 * bw + 0.06, bw * 1.15, proud + 0.01), (o.c, o.v1 + bw * 0.575, (proud + 0.01) / 2)))
        return acc.mesh()
    inner = arch_points(o, o.r, seg + 1)
    outer = arch_points(o, o.r + bw, seg + 1)
    for k in range(seg):
        poly = np.array([inner[k + 1], inner[k], outer[k], outer[k + 1]])
        acc.add(extrude(poly[::-1] if _signed_area(poly) < 0 else poly, 0.0, proud))
    return acc.mesh()


def _signed_area(poly) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def sill(o: Opening):
    """Stone sill under a window. Returns (mesh, snow box (size, centre))."""
    w = o.u1 - o.u0 + 0.2
    return box((w, 0.08, 0.17), (o.c, o.v0 - 0.04, 0.035)), ((w - 0.02, 0.035, 0.15), (o.c, o.v0 + 0.0175, 0.04))


def ledge(L: float, y: float, h: float = 0.16, d: float = 0.09):
    return box((L, h, d), (L / 2, y + h / 2, d / 2)), ((L, 0.035, d - 0.005), (L / 2, y + h + 0.0175, d / 2))


def cornice(L: float, H: float) -> G.Mesh:
    acc = Acc()
    acc.add(box((L, 0.14, 0.17), (L / 2, H - 0.40, 0.085)))
    acc.add(box((L, 0.10, 0.25), (L / 2, H - 0.28, 0.125)))
    acc.add(box((L, 0.20, 0.36), (L / 2, H - 0.13, 0.18)))
    return acc.mesh()


# --------------------------------------------------------------------------
# Roofs, chimneys, gutters
# --------------------------------------------------------------------------
def roof(A: float, D: float, y_e: float, slope_deg: float = 28.0, widen: bool = True,
         pitch_u: float = 0.21, course: float = 0.36) -> G.Mesh:
    """Barrel-tile roof plane in roof-local coords: eave along x at z = 0, rising toward -z for depth D.
    Plan is a trapezoid: |u| <= A + w (widen: valleys at inner corners) or A - w (hips). uv in meters."""
    span = A + D if widen else A
    us = np.linspace(-span, span, int(2 * span / (pitch_u / 4)) + 1)
    ws = np.arange(0.0, D + 1e-6, course / 2)
    U, W = np.meshgrid(us, ws)
    tan = math.tan(math.radians(slope_deg))
    Y = y_e + W * tan + 0.034 * np.abs(np.sin(np.pi * U / pitch_u)) ** 0.7 + 0.014 * ((W % course) / course)
    P = np.stack([U, Y, -W], -1).reshape(-1, 3)
    f = grid_faces(len(ws), len(us))[:, ::-1]
    cu, cw = U.reshape(-1)[f].mean(1), W.reshape(-1)[f].mean(1)
    lim = A + cw if widen else A - cw
    f = f[np.abs(cu) <= lim]
    uv = np.stack([U + span, W], -1).reshape(-1, 2)
    return G.Mesh(P, G.vertex_normals(P, f), uv, f)


def eave_trim(A: float, y_e: float) -> dict[str, G.Mesh]:
    acc = Acc()
    acc.add(box((2 * A, 0.18, 0.05), (0.0, y_e - 0.06, 0.02)))
    gutter = tube([(-A, y_e - 0.07, 0.09), (A, y_e - 0.07, 0.09)], 0.055, 10)
    return {"fascia": acc.mesh(), "gutter": gutter}


def chimney(h: float = 2.2) -> G.Mesh:
    acc = Acc()
    acc.add(box((0.75, h, 0.6), (0.0, h / 2, 0.0)))
    acc.add(box((0.9, 0.12, 0.75), (0.0, h + 0.06, 0.0)))
    for x in (-0.18, 0.18):
        acc.add(G.sweep(np.array([(x, h + 0.1, 0.0), (x, h + 0.42, 0.0)]), 0.08, 10))
    return acc.mesh()


# --------------------------------------------------------------------------
# Ground: flagstones with real joints
# --------------------------------------------------------------------------
def flagstones(rng, x0, x1, z0, z1, cell=0.3, gap=0.012, top=0.035):
    """Random coursed flagstones. Returns (rects in world (x0, x1, z0, z1), mesh with world-normalised uv
    for the paving texture: u = (x + 8) / 16, v = (z + 9) / 16)."""
    nx, nz = int(round((x1 - x0) / cell)), int(round((z1 - z0) / cell))
    cx, cz = (x1 - x0) / nx, (z1 - z0) / nz
    used = np.zeros((nz, nx), bool)
    sizes = [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4), (2, 1), (1, 2), (3, 1), (1, 1)]
    wts = np.array([0.18, 0.14, 0.14, 0.10, 0.08, 0.08, 0.09, 0.09, 0.05, 0.05])
    stones = []
    for j in range(nz):
        for i in range(nx):
            if used[j, i]:
                continue
            sw, sh = 1, 1
            for _ in range(6):
                a, b = sizes[rng.choice(len(sizes), p=wts / wts.sum())]
                if i + a <= nx and j + b <= nz and not used[j:j + b, i:i + a].any():
                    sw, sh = a, b
                    break
            used[j:j + sh, i:i + sw] = True
            stones.append((x0 + i * cx, x0 + (i + sw) * cx, z0 + j * cz, z0 + (j + sh) * cz))
    acc = Acc()
    g = gap / 2
    for (a0, a1, b0, b1) in stones:
        y = top + rng.normal(0, 0.003, 4).clip(-0.006, 0.006)
        xs, zs = (a0 + g, a1 - g), (b0 + g, b1 - g)
        pts = [(xs[0], y[0], zs[1]), (xs[1], y[1], zs[1]), (xs[1], y[2], zs[0]), (xs[0], y[3], zs[0])]
        uv = [((p[0] + 8.0) / 16.0, (p[2] + 9.0) / 16.0) for p in pts]
        acc.add_raw(np.array(pts), np.tile([0.0, 1.0, 0.0], (4, 1)), np.array(uv), np.array([[0, 1, 2], [0, 2, 3]]))
        for (p, q) in ((pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])):
            acc.add(quad([p[0], 0.0, p[2]], [q[0], 0.0, q[2]], [q[0], q[1], q[2]], [p[0], p[1], p[2]],
                         [(0.5, 0.5)] * 4))
    return stones, acc.mesh().oriented()


def floor_quad(x0, x1, z0, z1, y, up=True) -> G.Mesh:
    if up:
        return quad([x0, y, z1], [x1, y, z1], [x1, y, z0], [x0, y, z0], [(x0, z1), (x1, z1), (x1, z0), (x0, z0)])
    return quad([x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1], [(x0, z0), (x1, z0), (x1, z1), (x0, z1)])


# --------------------------------------------------------------------------
# Fountain, kerbs, pots, bench
# --------------------------------------------------------------------------
def fountain() -> dict[str, G.Mesh]:
    """Round stone basin (outer r 1.55, water at 0.44) with a column and an upper bowl (water at 1.24).
    Everything under the water is a separate 'lining' part: shadow rays cannot pass a dielectric, so a
    submerged surface sees the sun only through lucky refracted paths (caustic fireflies that then light
    the column and walls). A dark, algae-stained lining keeps that energy, and the noise, tiny."""
    cop = arc(0.075, 0, 180, (1.475, 0.475), 9)
    rim = G.lathe([(1.55, 0.0), (1.55, 0.40), (1.56, 0.45)] + [(x, y) for x, y in cop] + [(1.40, 0.47)], 96)
    lining = G.lathe([(1.40, 0.47), (1.37, 0.43), (1.36, 0.14), (1.30, 0.12), (0.0, 0.12)], 96)
    col = G.lathe([(0.44, 0.12), (0.44, 0.30), (0.36, 0.34), (0.22, 0.42), (0.17, 0.52), (0.15, 0.95), (0.20, 0.99),
                   (0.36, 1.06), (0.52, 1.14), (0.60, 1.21), (0.61, 1.27), (0.58, 1.29)], 64)
    bowl = G.lathe([(0.58, 1.29), (0.53, 1.26), (0.48, 1.20), (0.12, 1.17)], 64)
    finial = G.lathe([(0.12, 1.17), (0.075, 1.24), (0.065, 1.31), (0.10, 1.34), (0.105, 1.36), (0.07, 1.37)]
                     + [(x, y) for x, y in arc(0.085, -60, 90, (0.0, 1.45), 9)], 48)
    return {"stone": _merge(rim, col, finial), "lining": _merge(lining, bowl),
            "water_low": ripple_disc(1.375, 0.44, 7), "water_high": ripple_disc(0.49, 1.235, 9)}


def ripple_disc(radius: float, y: float, seed: int, rings: int = 28, segs: int = 96) -> G.Mesh:
    rng = np.random.default_rng(seed)
    r = np.linspace(0.0, radius, rings)[:, None]
    a = np.linspace(0.0, TAU, segs + 1)[None, :]
    h = np.zeros(np.broadcast(r, a).shape)
    for _ in range(5):
        cx, cz, k, ph = rng.uniform(-radius, radius), rng.uniform(-radius, radius), rng.uniform(20, 45), rng.uniform(0, TAU)
        d = np.hypot(r * np.cos(a) - cx, r * np.sin(a) - cz)
        h += 0.0016 * np.sin(k * d + ph) * np.exp(-d / (radius * 0.8))
    h *= np.clip((radius - r) / 0.05, 0, 1)
    P = np.stack(np.broadcast_arrays(r * np.cos(a), y + h, r * np.sin(a)), -1).reshape(-1, 3)
    f = grid_faces(rings, segs + 1)
    uv = np.stack(np.broadcast_arrays(0.5 + r * np.cos(a) / (2 * radius), 0.5 + r * np.sin(a) / (2 * radius)), -1).reshape(-1, 2)
    n = G.vertex_normals(P, f)
    if n[:, 1].mean() < 0:
        f = f[:, ::-1]
        n = -n
    n[np.linalg.norm(n, axis=1) < 0.5] = [0.0, 1.0, 0.0]            # the collapsed centre ring
    return G.Mesh(P, n, uv, f)


def kerb_ring(r_in: float, r_out: float, h: float) -> G.Mesh:
    mid = 0.5 * (r_in + r_out)
    cop = arc(0.5 * (r_out - r_in), 0, 180, (mid, h - 0.5 * (r_out - r_in)), 7)
    return G.lathe([(r_out, 0.0), (r_out, h - 0.5 * (r_out - r_in))] + [(x, y) for x, y in cop] + [(r_in, 0.0)], 96)


def pot(radius: float, height: float, lip: float = 0.02) -> G.Mesh:
    r, h = radius, height
    return G.lathe([(0.0, 0.0), (r * 0.68, 0.0), (r * 0.72, 0.015), (r, h - 0.05), (r + lip, h - 0.045),
                    (r + lip, h), (r - 0.012, h), (r - 0.018, h - 0.05), (0.0, h - 0.05)], 48)


def disc(radius: float, y: float, segs: int = 32) -> G.Mesh:
    return G.lathe([(radius, y), (0.0, y + 0.004)], segs)


def trough(L: float, w: float = 0.2, h: float = 0.18) -> G.Mesh:
    acc = Acc()
    acc.add(box((L, 0.02, w), (0, 0.01, 0)))
    for s in (-1, 1):
        acc.add(box((L, h, 0.025), (0, h / 2, s * (w / 2 - 0.0125))))
        acc.add(box((0.025, h, w), (s * (L / 2 - 0.0125), h / 2, 0)))
    return acc.mesh()


def bench() -> dict[str, G.Mesh]:
    """Slatted bench with cast-iron ends, 1.6 m long, seat front toward +z, base at y = 0."""
    wood, iron = Acc(), Acc()
    L = 1.6
    for k in range(5):
        wood.add(G.box((L, 0.03, 0.065), (0, 0.45, -0.17 + k * 0.085)))
    for k in range(3):
        wood.add(G.box((L, 0.08, 0.025)), G.compose(G.translate((0, 0.72 + k * 0.1, -0.24 - k * 0.012)),
                                                     G.rotate((1, 0, 0), -12)))
    for sx in (-L / 2 + 0.06, L / 2 - 0.06):
        iron.add(tube([(sx, 0.0, 0.18), (sx, 0.43, 0.14), (sx, 0.44, -0.2)], 0.014, 8))
        iron.add(tube([(sx, 0.0, -0.24), (sx, 0.44, -0.2), (sx, 0.98, -0.3)], 0.014, 8))
        iron.add(tube([(sx, 0.62, -0.02), (sx, 0.62, -0.22)], 0.012, 8))
    return {"wood": wood.mesh(), "iron": iron.mesh()}


# --------------------------------------------------------------------------
# Bicycle, lantern, balcony, laundry
# --------------------------------------------------------------------------
def bicycle() -> dict[str, G.Mesh]:
    """City bike, wheels along z (front +z), ground at y = 0, frame plane x = 0."""
    paint, chrome, rubber, leather, dark = Acc(), Acc(), Acc(), Acc(), Acc()
    R = 0.34
    F, B = np.array([0.0, R, 0.53]), np.array([0.0, R, -0.53])
    for c in (F, B):
        a = np.linspace(0, TAU, 73)
        ring = lambda rr: np.column_stack([np.zeros_like(a), c[1] + rr * np.cos(a), c[2] + rr * np.sin(a)])
        rubber.add(G.sweep(ring(R - 0.018), 0.018, 10, closed=True))
        chrome.add(G.sweep(ring(R - 0.04), 0.008, 6, closed=True))
        chrome.add(tube([c + [-0.05, 0, 0], c + [0.05, 0, 0]], 0.018, 10))
        for k in range(28):
            ang = TAU * k / 28
            hub = c + np.array([0.03 * (1 if k % 2 else -1), 0.022 * math.cos(ang), 0.022 * math.sin(ang)])
            ang2 = ang + (0.35 if k % 2 else -0.35)
            rim = c + np.array([0.0, (R - 0.045) * math.cos(ang2), (R - 0.045) * math.sin(ang2)])
            chrome.add(G.sweep(np.stack([hub, rim]), 0.0012, 3, caps=False))
    BB = np.array([0.0, 0.27, -0.06])
    S = np.array([0.0, 0.84, -0.20])
    HT, HB = np.array([0.0, 0.86, 0.36]), np.array([0.0, 0.70, 0.41])
    for p, q, r in ((BB, S, 0.016), (S, HT, 0.014), (BB, HB, 0.018), (HB, HT, 0.02)):
        paint.add(tube([p, q], r, 10))
    for s in (-1, 1):
        off = np.array([s * 0.05, 0.0, 0.0])
        paint.add(tube([BB + off * 0.6, B + off], 0.01, 8))
        paint.add(tube([S + off * 0.3, B + off], 0.009, 8))
        paint.add(tube(bezier([HB + off * 0.5, HB + off + [0, -0.2, 0.06], F + off + [0, 0.02, -0.02]], 8), 0.011, 8))
        # rack over the rear wheel
        paint.add(tube([B + off * 1.3 + [0, 0.02, 0], B + off * 1.3 + [0, 0.42, -0.02], B + off * 1.3 + [0, 0.42, 0.33],
                        S + off * 1.3 + [0, -0.08, 0.02]], 0.006, 6))
    paint.add(tube([B + [-0.065, 0.42, -0.02], B + [0.065, 0.42, -0.02]], 0.006, 6))
    stem_top = HT + [0, 0.12, -0.04]
    chrome.add(tube([HT, stem_top], 0.013, 8))
    bar = bezier([[-0.27, 0.98, 0.20], [-0.22, 0.99, 0.34], [0.0, stem_top[1], stem_top[2]],
                  [0.22, 0.99, 0.34], [0.27, 0.98, 0.20]], 24)
    chrome.add(tube(bar, 0.011, 8))
    for s in (-1, 1):
        rubber.add(tube([(s * 0.27, 0.98, 0.20), (s * 0.255, 0.985, 0.31)], 0.016, 8))
    chrome.add(tube([S, S + [0, 0.13, -0.035]], 0.0135, 8))
    leather.add(G.ellipsoid((0.085, 0.035, 0.13), 10, 20), G.translate(S + [0, 0.17, -0.06]))
    ring_ = G.lathe([(0.105, -0.003), (0.105, 0.003), (0.0, 0.003)], 40)
    dark.add(ring_, G.compose(G.translate(BB + [0.06, 0, 0]), G.rotate((0, 0, 1), 90)))
    for s, ang in ((1, 30.0), (-1, 210.0)):
        d = np.array([0.0, math.sin(math.radians(ang)), math.cos(math.radians(ang))]) * 0.17
        chrome.add(tube([BB + [s * 0.075, 0, 0], BB + [s * 0.075, 0, 0] + d], 0.009, 6))
        dark.add(G.box((0.1, 0.02, 0.05), BB + [s * 0.12, 0, 0] + d))
    chain = np.array([BB + [0.06, 0.105 * math.sin(t), 0.105 * math.cos(t)] for t in np.linspace(1.4, -1.4, 12)]
                     + [B + [0.06, 0.04 * math.sin(s), -0.04 * math.cos(s)] for s in np.linspace(-1.5, 1.5, 8)])
    dark.add(G.sweep(chain, 0.0035, 4, closed=True))
    for c, a0, a1 in ((F, -25, 140), (B, 40, 215)):
        a = np.radians(np.linspace(a0, a1, 24))
        mud = np.column_stack([np.zeros_like(a), c[1] + (R + 0.02) * np.sin(a), c[2] + (R + 0.02) * np.cos(a)])
        paint.add(G.sweep(mud, 0.022, 6, caps=False))
    return {"paint": paint.mesh(), "chrome": chrome.mesh(), "rubber": rubber.mesh(), "leather": leather.mesh(),
            "dark": dark.mesh()}


def lantern_bracket() -> dict[str, G.Mesh]:
    """Wall lantern: scrolled iron bracket out from the wall (+z) with a lantern hanging from it."""
    iron, glass = Acc(), Acc()
    arm = bezier([(0, 0, 0.0), (0, 0.12, 0.25), (0, 0.06, 0.55)], 20)
    iron.add(tube(arm, 0.014, 6))
    t = np.linspace(0, 2.2 * math.pi, 40)
    scroll = np.column_stack([np.zeros_like(t), -0.12 + 0.09 * np.exp(-t / 5) * np.sin(t),
                              0.12 + 0.09 * np.exp(-t / 5) * np.cos(t)])
    iron.add(tube(np.vstack([[0, -0.25, 0.0], [0, -0.15, 0.06], scroll]), 0.009, 5))
    iron.add(G.box((0.12, 0.3, 0.02), (0, -0.05, 0.01)))
    top = np.array([0.0, 0.0, 0.55])
    iron.add(tube([top + [0, 0.06, 0], top + [0, -0.08, 0]], 0.006, 5))
    c = top + [0, -0.30, 0]
    hw, hh = 0.11, 0.17
    for sx in (-1, 1):
        for sz in (-1, 1):
            iron.add(tube([c + [sx * hw, -hh, sz * hw], c + [sx * hw, hh, sz * hw]], 0.008, 5))
    for y in (-hh, hh):
        iron.add(G.box((2 * hw + 0.03, 0.02, 2 * hw + 0.03), c + [0, y, 0]))
    iron.add(G.lathe([(0.17, 0.0), (0.02, 0.1), (0.0, 0.1)], 4), G.compose(G.translate(c + [0, hh + 0.01, 0]), G.rotate((0, 1, 0), 45)))
    for nrm in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)):
        n = np.array(nrm, float)
        side = np.cross(UP, n)                       # quad normal = side x up = n
        p0 = c + n * (hw - 0.004)
        glass.add(quad(p0 - side * hw - [0, hh, 0], p0 + side * hw - [0, hh, 0], p0 + side * hw + [0, hh, 0],
                       p0 - side * hw + [0, hh, 0]))
    return {"iron": iron.mesh(), "glass": glass.mesh()}


def balcony(width: float = 1.9, depth: float = 0.75, rail_h: float = 0.95) -> dict[str, G.Mesh]:
    """Stone slab on two consoles with an iron railing, in wall-local coords: top of the slab at y = 0."""
    stone, iron = Acc(), Acc()
    stone.add(box((width, 0.12, depth), (0.0, -0.06, depth / 2)))
    for x in (-width / 2 + 0.2, width / 2 - 0.2):
        stone.add(extrude(np.array([(0.0, -0.12), (0.0, -0.45), (0.55, -0.12)]), -0.06, 0.06),
                  G.compose(G.translate((x, 0, 0)), G.rotate((0, 1, 0), -90)))
    pts = [(-width / 2 + 0.03, depth - 0.03), (width / 2 - 0.03, depth - 0.03)]
    corners = [(-width / 2 + 0.03, 0.02), pts[0], pts[1], (width / 2 - 0.03, 0.02)]
    for y in (0.08, rail_h):
        iron.add(tube([(x, y, z) for x, z in corners], 0.012 if y > 0.5 else 0.009, 6))
    for (x0, z0), (x1, z1) in zip(corners[:-1], corners[1:]):
        n = max(int(math.hypot(x1 - x0, z1 - z0) / 0.11), 1)
        for t in np.linspace(0, 1, n + 1)[:-1]:
            x, z = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
            iron.add(G.sweep(np.array([(x, 0.08, z), (x, rail_h, z)]), 0.007, 4, caps=False))
    return {"stone": stone.mesh(), "iron": iron.mesh()}


def laundry(rng, a, b, n_items: int = 5) -> dict[str, G.Mesh]:
    """A sagging line between points a and b with clothes pegged on it. Parts: rope, cloth_<k>."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid = (a + b) / 2 - [0, 0.22, 0]
    line = bezier([a, mid, b], 30)
    out = {"rope": tube(line, 0.004, 5)}
    span = np.linalg.norm(b - a)
    along = unit(b - a)
    side_n = unit(np.cross(along, UP))
    acc = defaultdict(Acc)
    t = 0.12
    while t < 0.88 and len(acc) < n_items * 2:
        w = rng.uniform(0.35, 0.7) / span
        if t + w > 0.92:
            break
        k = int(rng.integers(0, 5))
        hgt = rng.uniform(0.45, 0.9)
        i0, i1 = int(t * 29), int(min(t + w, 1.0) * 29)
        top = line[i0:i1 + 1]
        cols = len(top)
        rows = 10
        P = []
        for r in range(rows):
            s = r / (rows - 1)
            for ci, p in enumerate(top):
                fold = 0.025 * math.sin(ci * 1.7 + k) * s + 0.02 * s * s
                P.append(p - [0, s * hgt, 0] + side_n * fold)
        P = np.array(P)
        f = grid_faces(rows, cols)
        uv = np.array([(ci / max(cols - 1, 1), r / (rows - 1)) for r in range(rows) for ci in range(cols)])
        acc[f"cloth_{k}"].add_raw(P, G.vertex_normals(P, f), uv, f)
        t += w + rng.uniform(0.04, 0.12)
    out.update({k: v.mesh() for k, v in acc.items()})
    return out


# --------------------------------------------------------------------------
# Leaves
# --------------------------------------------------------------------------
_LOBES = {
    # (angle from the tip direction, radius) for one half; mirrored for the other half
    "ivy": [(0.0, 1.0), (0.5, 0.60), (0.95, 0.80), (1.42, 0.50), (1.95, 0.62), (2.5, 0.42), (math.pi, 0.30)],
    "creeper": [(0.0, 1.0), (0.32, 0.72), (0.62, 0.52), (1.05, 0.90), (1.5, 0.55), (1.9, 0.52), (2.5, 0.38),
                (math.pi, 0.26)],
    "round": [(0.0, 0.92), (0.8, 0.96), (1.6, 0.98), (2.4, 0.9), (2.9, 0.62), (math.pi, 0.45)],
}


def lobed_leaf(rng, shape: str, n_variants: int, variant: int, serrate: float = 0.0, cup: float = 0.14,
               subdiv: int = 0) -> G.Mesh:
    """Unit-length palmate leaf: petiole notch at the origin, tip toward +z, upper surface +y. The outline
    runs through the lobe tips and sinuses themselves (plus `subdiv` points between, where `serrate` adds
    teeth), so an ivy leaf is 12 triangles. uv: planar over the blade, squeezed into atlas cell `variant`."""
    ctrl = np.array(_LOBES[shape])
    ca, cr = ctrl[:, 0], ctrl[:, 1] * rng.uniform(0.9, 1.1, len(ctrl))
    cr[0] = ctrl[0, 1]
    half = [ca[0]]
    for a0, a1 in zip(ca[:-1], ca[1:]):
        half += [a0 + (a1 - a0) * (k + 1) / (subdiv + 1) for k in range(subdiv)] + [a1]
    half = np.array(half)
    phi = np.concatenate([-half[::-1][:-1], half[1:-1]])            # -pi .. pi, no duplicate at +-pi or 0
    n = len(phi)
    r = np.interp(np.abs(phi), ca, cr)
    if serrate and subdiv:
        tooth = np.array([0.0 if np.isclose(np.abs(p), ca).any() else 1.0 for p in phi])
        r = r * (1 + serrate * tooth)
    C = np.array([0.0, 0.0, 0.42])
    R0 = 0.58
    x = C[0] + R0 * r * np.sin(phi)
    z = C[2] + R0 * r * np.cos(phi)
    z0 = z[np.argmin(np.abs(np.abs(phi) - math.pi))]
    rho = np.hypot(x - C[0], z - C[2]) / R0
    y = -cup * rho ** 2 - 0.06 * ((z - z0) / 1.0) ** 2
    rim = np.column_stack([x, y, z - z0])
    centre = np.array([[0.0, 0.0, C[2] - z0 + 0.01]])
    p = np.vstack([centre, rim])
    f = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
    u = 0.5 + p[:, 0] / (2 * max(np.abs(x).max(), 1e-6))
    v = (p[:, 2] - p[:, 2].min()) / max(p[:, 2].max() - p[:, 2].min(), 1e-6)
    uv = np.column_stack([(variant + 0.03 + 0.94 * u) / n_variants, 0.02 + 0.96 * v])
    nrm = G.vertex_normals(p, f)
    if nrm[0, 1] < 0:
        f = f[:, ::-1]
        nrm = G.vertex_normals(p, f)
    return G.Mesh(p, nrm, uv, f)


def blade_leaf(rng, n_variants: int, variant: int, width: float = 0.55, rows: int = 5, cols: int = 3) -> G.Mesh:
    m = G.leaf(1.0, width, curl=rng.uniform(0.1, 0.35), fold=rng.uniform(0.15, 0.35), rows=rows, cols=cols)
    m.uv = np.column_stack([(variant + 0.03 + 0.94 * m.uv[:, 0]) / n_variants, 0.02 + 0.96 * m.uv[:, 1]])
    return m


class LeafLibrary:
    """A few dozen pre-built unit leaves; placing one is a matrix product, not a rebuild."""

    def __init__(self, rng, maker, n_shapes: int, n_variants: int):
        self.items = [maker(rng, k % n_variants) for k in range(n_shapes * n_variants)]

    def pick(self, rng) -> G.Mesh:
        return self.items[int(rng.integers(len(self.items)))]


# --------------------------------------------------------------------------
# Climbers on a wall (wall-local coordinates)
# --------------------------------------------------------------------------
def climber(rng, region, top_fn, obstacles, n_roots: int, lib: LeafLibrary, size=(0.05, 0.085),
            every=0.04, step=0.05, branch_p=0.045, max_len=11.0, max_stems=500, roots=None,
            hang=0.6, stem_r=0.006, budget: float = 600.0) -> dict[str, G.Mesh]:
    """Grow stems up a wall face from the ground (or from `roots`), turning around openings (`obstacles`,
    (u0, u1, v0, v1)), branching sideways, stopping at top_fn(u). Growth is breadth-first with a total
    stem-length `budget` (m), so every root gets its main stem before any lateral, and the leaf count is
    bounded (about budget / every). Leaves cling to the face (+z), blades hanging down and outward.
    Returns {'stem', 'leaf'} in wall-local coords."""
    from collections import deque
    u_lo, u_hi = sorted(region)

    def blocked(u, v):
        for (a0, a1, b0, b1) in obstacles:
            if a0 < u < a1 and b0 < v < b1:
                return True
        return False

    starts = roots if roots is not None else [float(u) for u in np.sort(rng.uniform(u_lo, u_hi, n_roots))]
    queue = deque((u, 0.03, rng.normal(0, 0.15), 0.0, 0) for u in starts if not blocked(u, 0.05))
    stems = []
    total = 0.0
    while queue and len(stems) < max_stems and total < budget:
        u, v, ang, target, depth = queue.popleft()
        pts = [(u, v)]
        length, tries = 0.0, 0
        while length < max_len * (0.6 ** depth) and tries < 400:
            tries += 1
            ang = target + 0.8 * (ang - target) + rng.normal(0, 0.3)
            nu, nv = u + step * math.sin(ang), v + step * math.cos(ang)
            if not (u_lo <= nu <= u_hi):
                ang = -ang
                continue
            if nv > top_fn(nu) or nv < 0.01:
                break
            if blocked(nu, nv):
                moved = False
                for d in (0.9, -0.9, 1.5, -1.5, 2.2, -2.2):
                    a2 = ang + d
                    mu, mv = u + step * math.sin(a2), v + step * math.cos(a2)
                    if u_lo <= mu <= u_hi and 0.01 < mv < top_fn(mu) and not blocked(mu, mv):
                        ang, nu, nv, moved = a2, mu, mv, True
                        break
                if not moved:
                    break
            u, v = nu, nv
            pts.append((u, v))
            length += step
            if depth < 3 and rng.random() < branch_p * 0.6 ** depth:
                side = float(rng.choice([-1.0, 1.0]))
                queue.append((u, v, side * rng.uniform(0.8, 1.4), side * rng.uniform(0.5, 1.0), depth + 1))
        if len(pts) >= 4:
            stems.append((np.array(pts), depth))
            total += length
    stem_acc, leaf_acc = Acc(), Acc()
    for pts, depth in stems:
        w = 0.012 + 0.006 * np.sin(np.arange(len(pts)) * 0.7 + rng.uniform(0, 6))
        path = np.column_stack([pts, w])
        coarse = path[::3] if len(path) > 6 else path
        rad = np.linspace(stem_r * (1.6 if depth == 0 else 1.0), stem_r * 0.5, len(coarse))
        stem_acc.add(G.sweep(coarse, rad, 4, caps=False))
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = rng.uniform(0, every)
        while s < cum[-1]:
            i = min(int(np.searchsorted(cum, s)), len(path) - 1)
            p = path[i] + [rng.normal(0, 0.012), rng.normal(0, 0.012), rng.uniform(0.012, 0.045)]
            up = unit([rng.normal(0, 0.3), rng.normal(0.15, 0.3), 1.0])
            fwd = unit([rng.normal(0, 0.6), -hang + rng.normal(0, 0.35), 0.4 + rng.uniform(0, 0.4)])
            scale = rng.uniform(*size) * (1.15 if depth == 0 and p[1] < 1.5 else 1.0)
            leaf_acc.place(lib.pick(rng), basis(fwd, up), p, scale)
            s += every * rng.uniform(0.6, 1.4)
    return {"stem": stem_acc.mesh(), "leaf": leaf_acc.mesh()}


def smooth_top(rng, lo: float, hi: float, base: float, amp: float, period: float = 2.5):
    """A wavy upper limit for climber coverage along u."""
    lo, hi = sorted((lo, hi))
    k = max(int((hi - lo) / period) + 3, 3)
    xs = np.linspace(lo - period, hi + period, k)
    ys = base + amp * rng.uniform(-1, 1, k)
    return lambda u: float(np.interp(u, xs, ys))


# --------------------------------------------------------------------------
# Tree, citrus, geraniums, grass (world coordinates)
# --------------------------------------------------------------------------
def tree(grp, rng, base, lib: LeafLibrary, trunk_h=2.7, crown_r=3.2, n_leaves=11000, bias=(0.0, 0.0),
         low_limb=None, leaf_size=(0.09, 0.14), keep_out=None) -> None:
    """Broadleaf tree: flared trunk, 5 limbs, secondaries, twigs, leaves clustered on twigs. `bias` (x, z)
    leans the crown, `low_limb` (x, z) adds a long low limb toward that direction, `keep_out(p)` shortens
    anything that would pierce a wall."""
    base = np.asarray(base, float)
    lean = np.array([bias[0], 0.0, bias[1]]) * 0.25
    trunk = np.array([base + [0, 0, 0], base + [0, trunk_h * 0.4, 0] + lean * 0.3, base + [0, trunk_h, 0] + lean])
    path = bezier(trunk, 16)
    rad = np.interp(np.linspace(0, 1, 16), [0, 0.08, 0.25, 1.0], [0.42, 0.30, 0.25, 0.21])
    grp["bark"].add(G.sweep(path, rad, 14))
    fork = path[-1]
    twigs = []

    def clip(p):
        return keep_out(p) if keep_out is not None else p

    def branch(start, d, length, r0, r1, level, n_child, droop=False):
        end = clip(start + d * length)
        ctrl = [start, start + d * length * 0.45 + [0, length * 0.12, 0], end]
        pts = bezier(ctrl, max(5, int(length * 6)))
        grp["bark"].add(G.sweep(pts, np.linspace(r0, r1, len(pts)), 10 if level == 0 else 6 if level == 1 else 4))
        if level == 2:
            twigs.append(pts)
            return
        for k in range(n_child):
            t = rng.uniform(0.3, 0.95)
            i = min(int(t * (len(pts) - 1)), len(pts) - 2)
            dd = unit(pts[i + 1] - pts[i])
            side = unit(np.cross(dd, UP) * rng.choice([-1, 1]) + rng.normal(0, 0.3, 3))
            nd = unit(dd * 0.6 + side * 0.65 + [0, rng.uniform(0.15, 0.7), 0])
            if level == 1:
                nd = unit(nd + [0, -0.25, 0])
            if droop:                                   # a low limb's shoots hang into view instead of rising
                nd = unit(nd * [1, 0, 1] + [0, -rng.uniform(0.4, 0.9), 0])
            branch(pts[i], nd, length * rng.uniform(0.35, 0.55) if level == 0 else rng.uniform(0.35, 0.75),
                   r0 * 0.35, r1 * 0.5, level + 1, 9 if level == 0 else 0, droop)

    n_limbs = 5
    for k in range(n_limbs):
        yaw = TAU * (k + rng.uniform(-0.25, 0.25)) / n_limbs
        el = math.radians(rng.uniform(48, 70))
        d = unit([math.cos(yaw) * math.cos(el) + bias[0] * 0.45, math.sin(el), math.sin(yaw) * math.cos(el) + bias[1] * 0.45])
        branch(fork, d, crown_r * rng.uniform(0.75, 1.0), 0.17, 0.06, 0, 4)
    if low_limb is not None:
        d = unit([low_limb[0], 0.18, low_limb[1]])
        start = path[int(len(path) * 0.82)]
        branch(start, d, crown_r * 1.05, 0.12, 0.05, 0, 6, droop=True)
    per = max(n_leaves // max(len(twigs), 1), 1)
    for pts in twigs:
        dd = unit(pts[-1] - pts[0])
        for _ in range(per):
            t = rng.uniform(0.15, 1.0)
            p = pts[min(int(t * (len(pts) - 1)), len(pts) - 1)] + rng.normal(0, 0.11, 3)
            p = clip(p)
            up = unit(UP * 0.8 + rng.normal(0, 0.45, 3))
            fwd = unit(dd * 0.5 + rng.normal(0, 0.7, 3) + [0, -0.3, 0])
            grp["tree_leaf"].place(lib.pick(rng), basis(fwd, up), p, rng.uniform(*leaf_size))


def citrus(grp, rng, base, lib: LeafLibrary, height: float = 1.0, crown=(0.48, 0.42)) -> list[float]:
    """Lemon tree for a big pot: trunk from `base` (soil level), a dense round crown, lemons. Returns
    a point on the crown surface (for a focus target)."""
    base = np.asarray(base, float)
    top = base + [rng.normal(0, 0.03), height, rng.normal(0, 0.03)]
    grp["bark"].add(G.sweep(np.stack([base, top]), np.array([0.035, 0.022]), 8))
    c = top + [0, crown[1] * 0.7, 0]
    for k in range(4):
        d = unit([rng.normal(), 0.8, rng.normal()])
        grp["bark"].add(G.sweep(np.stack([top, top + d * crown[0] * 0.7]), np.array([0.015, 0.006]), 5))
    for _ in range(650):
        v = unit(rng.normal(size=3))
        p = c + v * np.array([crown[0], crown[1], crown[0]]) * rng.uniform(0.55, 1.0) ** 0.5
        up = unit(v * 0.6 + UP * 0.6 + rng.normal(0, 0.3, 3))
        grp["citrus_leaf"].place(lib.pick(rng), basis(unit(v + rng.normal(0, 0.6, 3)), up), p, rng.uniform(0.065, 0.095))
    lemon = G.ellipsoid((0.038, 0.034, 0.05), 10, 16)
    for _ in range(int(rng.integers(10, 18))):
        v = unit(rng.normal(size=3) * [1, 0.7, 1])
        p = c + v * np.array([crown[0], crown[1], crown[0]]) * 0.95
        grp["lemon"].add(lemon, frame(p, unit(rng.normal(size=3))))
    return (c + np.array([0.0, 0.0, crown[0]])).tolist()


def geranium(grp, rng, base, lib: LeafLibrary, flower_key: str, n_leaves: int = 12, n_umbels: int = 5,
             spread: float = 0.16) -> None:
    base = np.asarray(base, float)
    for _ in range(n_leaves):
        yaw = rng.uniform(0, TAU)
        reach = spread * rng.uniform(0.3, 1.0)
        tip = base + [math.cos(yaw) * reach, rng.uniform(0.06, 0.16), math.sin(yaw) * reach]
        grp["stem"].add(G.sweep(bezier([base, base + [0, 0.08, 0], tip], 6), 0.004, 4))
        grp["geranium_leaf"].place(lib.pick(rng), basis(unit([math.cos(yaw), 0.2, math.sin(yaw)]),
                                                        unit(UP + rng.normal(0, 0.25, 3))), tip, rng.uniform(0.06, 0.09))
    petal = G.leaf(0.022, 0.02, curl=-0.2, fold=0.05, rows=3, cols=3)
    for _ in range(n_umbels):
        yaw = rng.uniform(0, TAU)
        reach = spread * rng.uniform(0.0, 0.7)
        top = base + [math.cos(yaw) * reach, rng.uniform(0.2, 0.32), math.sin(yaw) * reach]
        grp["stem"].add(G.sweep(bezier([base + [0, 0.05, 0], base + [0, 0.16, 0], top], 6), 0.003, 4))
        for _ in range(14):
            v = unit(rng.normal(size=3) * [1, 0.5, 1] + [0, 0.6, 0])
            fc = top + v * 0.035
            for k in range(5):
                a = TAU * k / 5 + rng.uniform(-0.2, 0.2)
                side = basis(v)
                d = unit(side[:, 0] * math.cos(a) + side[:, 1] * math.sin(a) + v * 0.3)
                grp[flower_key].add(petal, frame(fc, d, v))


def grass_tuft(grp, rng, base, n=10, length=0.1, key="grass") -> None:
    base = np.asarray(base, float)
    for _ in range(n):
        yaw = rng.uniform(0, TAU)
        el = rng.uniform(0.9, 1.45)
        d = unit([math.cos(yaw) * math.cos(el), math.sin(el), math.sin(yaw) * math.cos(el)])
        m = G.leaf(length * rng.uniform(0.5, 1.2), 0.007, curl=rng.uniform(0.2, 0.7), fold=0.0, rows=4, cols=2)
        grp[key].add(m, frame(base + rng.normal(0, 0.01, 3) * [1, 0, 1], d))


def fallen_leaves(grp, rng, centre, radius, n, lib: LeafLibrary, y: float = 0.047, key="fallen_leaf") -> None:
    for _ in range(n):
        a, r = rng.uniform(0, TAU), radius * math.sqrt(rng.uniform(0, 1))
        p = np.asarray(centre, float) + [r * math.cos(a), 0.0, r * math.sin(a)]
        p[1] = y
        yaw = rng.uniform(0, TAU)
        grp[key].place(lib.pick(rng), basis([math.cos(yaw), -0.05, math.sin(yaw)], UP), p, rng.uniform(0.08, 0.13))


def _merge(*meshes) -> G.Mesh:
    acc = Acc()
    for m in meshes:
        acc.add(m)
    return acc.mesh()
