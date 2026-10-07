"""Mesh recipes for the workshop scene: hand tools, vise, tyres, drums, shelving, a motorcycle, chains, hoses.

Same conventions as props.py: meters, y up, local frame with the object resting at y = 0. Each recipe returns
a Mesh or a dict of Meshes keyed by material name. Pure numpy + Pillow; nothing here imports Mitsuba
(and no OpenCV: the cluster environment does not have it).

Flat tools (wrenches, pliers, saws...) are drawn as silhouettes in a raster mask, turned into polygons with
holes, triangulated and extruded; that gives real outlines, real holes and thin edges, which is the point.
Tools lie in the xy plane, centred on z = 0, so they can be hung on a wall by rotating about x.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

import procedural as G
from procedural import Mesh, box, compose, ellipsoid, lathe, rotate, scale, sweep, translate

TAU = 2 * np.pi


# ----------------------------------------------------------------------------
# Small helpers (adapted from the desk's props)
# ----------------------------------------------------------------------------

def rot_between(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    m = np.eye(4)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return m
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return rotate(axis, 180.0)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    m[:3, :3] = np.eye(3) + k + k @ k * (1.0 / (1.0 + c))
    return m


def join(*meshes: Mesh) -> Mesh:
    out = Mesh()
    for m in meshes:
        out += m
    return out


def tube(p0, p1, radius, sides: int = 12, caps: bool = True) -> Mesh:
    return sweep([p0, p1], radius, sides, caps)


def put(mesh: Mesh, *mats) -> Mesh:
    return mesh.transformed(compose(*mats))


# ----------------------------------------------------------------------------
# Polygon triangulation (ear clipping with hole bridging) and extrusion
# ----------------------------------------------------------------------------

def _area(poly) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _seg_hit(p, q, a, b) -> bool:
    """Proper intersection of segments pq and ab (shared endpoints do not count)."""
    def orient(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    d1, d2, d3, d4 = orient(p, q, a), orient(p, q, b), orient(a, b, p), orient(a, b, q)
    return (d1 * d2 < -1e-18) and (d3 * d4 < -1e-18)


def _bridge(outer: np.ndarray, holes: list[np.ndarray]):
    """Merge holes into the outer ring with zero-width bridges. Returns an (n, 2) ring that may repeat points."""
    ring = [tuple(p) for p in outer]
    for hole in sorted(holes, key=lambda h: -h[:, 0].max()):
        h = [tuple(p) for p in hole]
        best = None
        edges = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        edges += [(h[i], h[(i + 1) % len(h)]) for i in range(len(h))]
        for hi, hp in enumerate(h):
            for ri, rp in enumerate(ring):
                d = (hp[0] - rp[0]) ** 2 + (hp[1] - rp[1]) ** 2
                if best is not None and d >= best[0]:
                    continue
                if any(_seg_hit(hp, rp, a, b) for a, b in edges):
                    continue
                best = (d, ri, hi)
        if best is None:
            continue
        _, ri, hi = best
        ring = ring[:ri + 1] + h[hi:] + h[:hi + 1] + ring[ri:]
    return np.array(ring)


def triangulate(outer: np.ndarray, holes=()) -> tuple[np.ndarray, np.ndarray]:
    """Ear clipping. Returns (points, triangles) for the bridged ring; outer CCW, holes CW."""
    pts = _bridge(outer, list(holes)) if len(holes) else np.asarray(outer)
    idx = list(range(len(pts)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 20 * len(pts) + 100:
        guard += 1
        n = len(idx)
        clipped = False
        for i in range(n):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % n]
            pa, pb, pc = pts[a], pts[b], pts[c]
            cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if cross <= 1e-14:
                continue
            ok = True
            for j in idx:
                if j in (a, b, c):
                    continue
                q = pts[j]
                if (q == pa).all() or (q == pb).all() or (q == pc).all():
                    continue
                d1 = (pb[0] - pa[0]) * (q[1] - pa[1]) - (pb[1] - pa[1]) * (q[0] - pa[0])
                d2 = (pc[0] - pb[0]) * (q[1] - pb[1]) - (pc[1] - pb[1]) * (q[0] - pb[0])
                d3 = (pa[0] - pc[0]) * (q[1] - pc[1]) - (pa[1] - pc[1]) * (q[0] - pc[0])
                if d1 >= -1e-14 and d2 >= -1e-14 and d3 >= -1e-14:
                    ok = False
                    break
            if ok:
                tris.append((a, b, c))
                del idx[i]
                clipped = True
                break
        if not clipped:                      # numerically degenerate: drop the flattest vertex
            k = min(range(n), key=lambda i: abs((pts[idx[i]][0] - pts[idx[i - 1]][0]) * (pts[idx[(i + 1) % n]][1] - pts[idx[i - 1]][1])
                                                - (pts[idx[i]][1] - pts[idx[i - 1]][1]) * (pts[idx[(i + 1) % n]][0] - pts[idx[i - 1]][0])))
            del idx[k]
    if len(idx) == 3:
        tris.append(tuple(idx))
    return pts, np.array(tris, dtype=np.int64)


def extrude(outer: np.ndarray, holes, thickness: float, smooth_deg: float = 38.0, uv_scale: float = 1.0) -> Mesh:
    """Extrude a polygon with holes (xy, metres) along z, centred on z = 0. Sides are smoothed across gentle corners."""
    outer = np.asarray(outer, float)
    if _area(outer) < 0:
        outer = outer[::-1]
    holes = [np.asarray(h, float) if _area(h) < 0 else np.asarray(h, float)[::-1] for h in holes]
    pts, tris = triangulate(outer, holes)
    t = thickness / 2
    mesh = Mesh()
    for z, nz, flip in ((t, 1.0, False), (-t, -1.0, True)):
        p3 = np.column_stack([pts, np.full(len(pts), z)])
        f = tris[:, ::-1] if flip else tris
        mesh += Mesh(p3, np.tile([0, 0, nz], (len(pts), 1)), pts * uv_scale, f)
    cos_lim = np.cos(np.radians(smooth_deg))
    for ring in [outer] + holes:
        n = len(ring)
        e = np.roll(ring, -1, axis=0) - ring
        ln = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        en = np.column_stack([e[:, 1], -e[:, 0]]) / ln[:, None]      # material is on the left: outward is to the right
        s = np.concatenate([[0.0], np.cumsum(ln)])
        for i in range(n):
            j = (i + 1) % n
            prev, nxt = en[i - 1], en[(j) % n]
            n_i = (en[i] + prev) if float(np.dot(en[i], prev)) > cos_lim else en[i]
            n_j = (en[i] + nxt) if float(np.dot(en[i], nxt)) > cos_lim else en[i]
            n_i = n_i / max(np.linalg.norm(n_i), 1e-12)
            n_j = n_j / max(np.linalg.norm(n_j), 1e-12)
            p = np.array([[*ring[i], -t], [*ring[j], -t], [*ring[j], t], [*ring[i], t]])
            nn = np.array([[*n_i, 0], [*n_j, 0], [*n_j, 0], [*n_i, 0]])
            uv = np.array([[s[i], 0], [s[i + 1], 0], [s[i + 1], thickness], [s[i], thickness]]) * uv_scale
            mesh += Mesh(p, nn, uv, np.array([[0, 1, 2], [0, 2, 3]]))
    return mesh.oriented()


def _trace_loops(mask: np.ndarray) -> list[np.ndarray]:
    """Boundary loops of a binary mask along pixel edges, image coords (x right, y down).

    Outer boundaries and hole boundaries come out with opposite orientation. Where two regions touch only at a
    corner, the walk takes the sharpest right turn, which keeps them separate.
    """
    m = np.pad(mask.astype(bool), 1)
    inside = m[1:-1, 1:-1]
    i, j = np.nonzero(inside & ~m[:-2, 1:-1])               # top edges: (j, i) -> (j+1, i)
    edges = [np.stack([j, i, j + 1, i], 1)]
    i, j = np.nonzero(inside & ~m[2:, 1:-1])                # bottom: (j+1, i+1) -> (j, i+1)
    edges.append(np.stack([j + 1, i + 1, j, i + 1], 1))
    i, j = np.nonzero(inside & ~m[1:-1, :-2])               # left: (j, i+1) -> (j, i)
    edges.append(np.stack([j, i + 1, j, i], 1))
    i, j = np.nonzero(inside & ~m[1:-1, 2:])                # right: (j+1, i) -> (j+1, i+1)
    edges.append(np.stack([j + 1, i, j + 1, i + 1], 1))
    e = np.concatenate(edges)
    nxt: dict[tuple, list] = {}
    for x0, y0, x1, y1 in e.tolist():
        nxt.setdefault((x0, y0), []).append((x1, y1))
    loops = []
    while nxt:
        start = next(iter(nxt))
        loop = [start]
        prev, cur = None, start
        while True:
            outs = nxt.get(cur)
            if not outs:
                break
            if len(outs) == 1 or prev is None:
                nb = outs.pop(0)
            else:
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                right = (-dy, dx)                               # in y-down coordinates
                nb = max(outs, key=lambda q: (q[0] - cur[0]) * right[0] + (q[1] - cur[1]) * right[1])
                outs.remove(nb)
            if not outs:
                del nxt[cur]
            prev, cur = cur, nb
            if cur == start:
                break
            loop.append(cur)
        if len(loop) >= 4:
            loops.append(np.array(loop, float))
    return loops


def _simplify(loop: np.ndarray, eps: float) -> np.ndarray:
    """Douglas-Peucker on a closed loop (iterative). Collinear runs are dropped first."""
    d1 = np.roll(loop, -1, 0) - loop
    d0 = loop - np.roll(loop, 1, 0)
    turn = np.abs(d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]) > 1e-9
    pts = loop[turn] if turn.sum() >= 3 else loop
    n = len(pts)
    if n < 4:
        return pts
    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    keep = np.zeros(n, bool)
    keep[0] = keep[far] = True
    stack = [(0, far), (far, n)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        a, b = pts[s], pts[e % n]
        seg = pts[s + 1:e]
        ab = b - a
        ln = np.linalg.norm(ab)
        if ln < 1e-12:
            dist = np.linalg.norm(seg - a, axis=1)
        else:
            dist = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / ln
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[s + 1 + k] = True
            stack += [(s, s + 1 + k), (s + 1 + k, e)]
    return pts[keep]


def _inside(pt, poly: np.ndarray) -> bool:
    x, y = pt
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    hit = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-30, yj - yi) + xi)
    return bool(np.count_nonzero(hit) % 2)


class Canvas:
    """A raster silhouette in metres; x right, y up, origin at the left end. Draw 255, erase 0."""

    def __init__(self, width: float, height: float, ppm: float = 4000.0, origin=(0.0, 0.0)):
        self.ppm, self.origin = ppm, np.asarray(origin, float)
        self.h, self.w = int(height * ppm), int(width * ppm)
        self.img = Image.new("L", (self.w, self.h), 0)
        self.draw = ImageDraw.Draw(self.img)

    def px(self, p) -> tuple[float, float]:
        return (p[0] - self.origin[0]) * self.ppm, self.h - (p[1] - self.origin[1]) * self.ppm

    def poly(self, pts, v=255):
        self.draw.polygon([self.px(p) for p in pts], fill=v)

    def circle(self, c, r, v=255):
        x, y = self.px(c)
        rr = max(0.5, r * self.ppm)
        self.draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=v)

    def line(self, p0, p1, width, v=255):
        """A thick segment with round caps."""
        self.draw.line([self.px(p0), self.px(p1)], fill=v, width=max(1, int(round(width * self.ppm))))
        self.circle(p0, width / 2, v)
        self.circle(p1, width / 2, v)

    def ellipse(self, c, rx, ry, angle, v=255):
        a = np.radians(angle)
        t = np.linspace(0, TAU, 48, endpoint=False)
        pts = np.stack([rx * np.cos(t), ry * np.sin(t)], 1) @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        self.poly([(c[0] + x, c[1] + y) for x, y in pts], v)

    def rect(self, p0, p1, v=255):
        (x0, y0), (x1, y1) = self.px(p0), self.px(p1)
        self.draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=v)

    def polygons(self, eps_px: float = 0.9):
        """(outer, [holes]) groups in metres, outer CCW and holes CW in xy."""
        loops = _trace_loops(np.asarray(self.img) > 127)
        # The tracer walks outer boundaries clockwise once y points up, so reverse: outer CCW, holes CW.
        conv = lambda c: np.column_stack([c[::-1, 0] / self.ppm + self.origin[0], (self.h - c[::-1, 1]) / self.ppm + self.origin[1]])
        polys = [conv(_simplify(lp, eps_px)) for lp in loops]
        polys = [p for p in polys if len(p) >= 3 and abs(_area(p)) > 1e-9]
        outers = [p for p in polys if _area(p) > 0]
        holes = [p for p in polys if _area(p) < 0]
        groups = [(o, []) for o in outers]
        for h in holes:
            owners = [g for g in groups if _inside(h.mean(0), g[0])]
            if owners:
                min(owners, key=lambda g: abs(_area(g[0])))[1].append(h)
        return groups

    def mesh(self, thickness: float, **kw) -> Mesh:
        out = Mesh()
        for outer, holes in self.polygons():
            out += extrude(outer, holes, thickness, **kw)
        return out


# ----------------------------------------------------------------------------
# Hand tools (flat, in the xy plane; length along +x from the origin)
# ----------------------------------------------------------------------------

def _regular(c, r, n=6, rot=0.0):
    a = np.radians(rot) + np.arange(n) * TAU / n
    return [(c[0] + r * np.cos(t), c[1] + r * np.sin(t)) for t in a]


def _open_end(cv: Canvas, c, r, slot, angle):
    """Jaw head with a U slot cut along `angle` (degrees)."""
    cv.circle(c, r)
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    n = np.array([-d[1], d[0]])
    p0, p1 = np.asarray(c) + d * 0.0, np.asarray(c) + d * (r * 1.5)
    cv.poly([p0 + n * slot / 2, p1 + n * slot / 2, p1 - n * slot / 2, p0 - n * slot / 2], 0)
    cv.circle(np.asarray(c), slot / 2, 0)


def combination_wrench(size: float = 0.013, length: float = 0.18, thick: float = 0.0045) -> dict[str, Mesh]:
    """Ring end on the left, open end on the right, a hang hole in the open end's neck."""
    rr, ro = size * 1.55, size * 1.35
    cv = Canvas(length + 2 * rr, 2 * rr + 0.004, origin=(-rr - 0.002, -rr - 0.002))
    cv.circle((0, 0), rr)
    cv.poly([(0, rr * 0.55), (length, ro * 0.5), (length, -ro * 0.5), (0, -rr * 0.55)])
    cv.circle((length, 0), ro)
    _open_end_slot = size * 1.02
    d = np.array([np.cos(np.radians(15)), np.sin(np.radians(15))])
    nrm = np.array([-d[1], d[0]])
    c = np.array([length, 0.0])
    cv.poly([c + nrm * _open_end_slot / 2, c + d * ro * 1.6 + nrm * _open_end_slot / 2, c + d * ro * 1.6 - nrm * _open_end_slot / 2, c - nrm * _open_end_slot / 2])
    cv.poly(_regular((0, 0), size * 0.62, 12, 0), 0)                # 12-point ring
    cv.circle((length - ro * 1.9, 0), 0.0035, 0)                    # hang hole
    return {"steel": cv.mesh(thick, uv_scale=40.0)}


def open_end_wrench(size: float = 0.012, length: float = 0.15, thick: float = 0.004) -> dict[str, Mesh]:
    ro = size * 1.4
    cv = Canvas(length + 2 * ro, 2 * ro + 0.004, origin=(-ro - 0.002, -ro - 0.002))
    cv.circle((0, 0), ro)
    cv.poly([(0, ro * 0.45), (length, ro * 0.45), (length, -ro * 0.45), (0, -ro * 0.45)])
    cv.circle((length, 0), ro)
    slot = size * 1.02
    for c, ang in (((0.0, 0.0), 195.0), ((length, 0.0), -15.0)):
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        nrm = np.array([-d[1], d[0]])
        c = np.array(c)
        cv.poly([c + nrm * slot / 2, c + d * ro * 1.6 + nrm * slot / 2, c + d * ro * 1.6 - nrm * slot / 2, c - nrm * slot / 2], 0)
    cv.circle((length / 2, 0), 0.0033, 0)
    return {"steel": cv.mesh(thick, uv_scale=40.0)}


def adjustable_spanner(length: float = 0.2) -> dict[str, Mesh]:
    cv = Canvas(length + 0.06, 0.07, origin=(-0.03, -0.035))
    cv.poly([(0, 0.011), (length, 0.007), (length, -0.007), (0, -0.011)])
    cv.circle((length - 0.002, 0.0), 0.0075)
    cv.rect((-0.03, 0.0), (0.0, 0.032))                                         # fixed jaw upward
    cv.poly([(-0.032, 0.03), (0.012, 0.03), (0.012, 0.012), (-0.032, 0.0)])
    cv.rect((-0.0, -0.0), (0.012, 0.02), 255)
    cv.rect((-0.02, 0.0075), (0.0, 0.026), 0)                                    # jaw opening
    cv.circle((0.014, 0.0), 0.0045, 0)
    cv.circle((length - 0.008, 0.0), 0.0033, 0)
    return {"steel": cv.mesh(0.007, uv_scale=40.0)}


def pliers(length: float = 0.2) -> dict[str, Mesh]:
    """Slip-joint pliers: jaws on the left, crossed arms, rubber grips on the right."""
    w = 0.05
    cv = Canvas(length + 0.02, w, origin=(-0.01, -w / 2))
    cv.poly([(0.0, 0.003), (0.02, 0.011), (0.052, 0.014), (0.07, 0.003), (0.07, -0.003), (0.052, -0.014), (0.02, -0.011), (0.0, -0.003)])
    cv.line((0.05, 0.012), (length, -0.017), 0.011)
    cv.line((0.05, -0.012), (length, 0.017), 0.011)
    cv.circle((0.05, 0.0), 0.011)
    cv.line((0.0, 0.0), (0.012, 0.0), 0.0016, 0)                                 # the gap between the jaws
    cv.circle((0.05, 0.0), 0.0035, 0)                                             # pivot
    cv.circle((0.085, 0.0), 0.0045, 0)
    steel = cv.mesh(0.0055, uv_scale=40.0)
    g = Canvas(length + 0.02, w, origin=(-0.01, -w / 2))
    g.line((0.115, -0.0075), (length, -0.0205), 0.0145)
    g.line((0.115, 0.0075), (length, 0.0205), 0.0145)
    return {"steel": steel, "grip": g.mesh(0.0085, uv_scale=40.0)}


def hammer(length: float = 0.33) -> dict[str, Mesh]:
    """Claw hammer lying in the xy plane: handle from the origin along +x, head across y at x = length."""
    cv = Canvas(0.05, 0.14, origin=(length - 0.025, -0.07))
    L = length
    cv.poly([(L - 0.017, 0.036), (L + 0.019, 0.036), (L + 0.017, 0.0), (L + 0.017, -0.012), (L + 0.011, -0.04), (L + 0.0045, -0.062),
             (L - 0.002, -0.066), (L - 0.0035, -0.04), (L - 0.017, -0.012), (L - 0.017, 0.0)])
    head = cv.mesh(0.026, uv_scale=40.0)
    handle = sweep([(0.0, 0.0, 0.0), (L * 0.5, 0.0, 0.0), (L * 0.98, 0.0, 0.0)], [0.0105, 0.0125, 0.0155], 14)
    return {"steel": head, "wood": handle.transformed(scale((1.0, 1.0, 0.8)))}


def handsaw(length: float = 0.5) -> dict[str, Mesh]:
    """Plate with real teeth along the bottom edge, wooden handle with a grip hole."""
    blade = Canvas(length + 0.02, 0.12, origin=(0.0, -0.06))
    top = [(0.0, 0.052), (length, 0.018)]
    bot_y0, bot_y1 = -0.045, -0.012
    teeth = [(length, 0.018), (length, bot_y1)]
    n = int(length / 0.0085)
    for i in range(n, -1, -1):
        x = length * i / n
        y = bot_y0 + (bot_y1 - bot_y0) * i / n
        teeth += [(x, y - 0.0055), (x - 0.0042, y)]
    pts = top + teeth + [(0.0, bot_y0)]
    blade.poly(pts)
    for k in range(3):
        blade.circle((0.012, -0.012 + 0.02 * k - 0.0), 0.0022, 0)
    plate = blade.mesh(0.0009, uv_scale=30.0)
    hnd = Canvas(0.14, 0.15, origin=(-0.14, -0.1))
    hnd.poly([(0.0, 0.05), (-0.03, 0.058), (-0.13, 0.05), (-0.14, 0.0), (-0.13, -0.075), (-0.06, -0.09), (0.0, -0.05)])
    hnd.poly([(-0.025, 0.015), (-0.115, 0.03), (-0.12, -0.04), (-0.04, -0.05)], 0)
    wood = hnd.mesh(0.022, uv_scale=10.0)
    wood = wood.transformed(translate((0.002, 0.0, 0.0)))
    return {"steel": plate, "wood": wood}


def hacksaw() -> dict[str, Mesh]:
    cv = Canvas(0.36, 0.14, origin=(0.0, -0.05))
    cv.poly([(0.0, 0.07), (0.3, 0.07), (0.3, 0.062), (0.01, 0.062)])
    cv.poly([(0.292, 0.07), (0.3, 0.07), (0.3, -0.005), (0.292, -0.005)])
    cv.poly([(0.0, 0.07), (0.008, 0.07), (0.008, 0.0), (0.0, 0.0)])
    cv.rect((0.0, 0.0), (0.05, 0.02))
    frame = cv.mesh(0.012, uv_scale=30.0)
    bl = Canvas(0.3, 0.02, origin=(0.0, 0.0))
    bl.rect((0.008, 0.0), (0.292, 0.012))
    blade = bl.mesh(0.0007, uv_scale=30.0).transformed(translate((0.0, 0.0, 0.0)))
    hd = Canvas(0.12, 0.09, origin=(-0.12, -0.03))
    hd.poly([(0.0, 0.06), (-0.12, 0.056), (-0.13, 0.0), (-0.1, -0.03), (0.0, -0.01)])
    hd.poly([(-0.02, 0.03), (-0.1, 0.034), (-0.1, 0.006), (-0.02, 0.006)], 0)
    grip = hd.mesh(0.026, uv_scale=10.0)
    return {"steel": join(frame, blade.transformed(translate((0.0, 0.0, 0.0)))), "grip": grip.transformed(translate((0.002, 0.0, 0.0)))}


def screwdriver(shaft: float = 0.1, flat: bool = True, handle_r: float = 0.0105) -> dict[str, Mesh]:
    """Along +x from the tip at the origin."""
    handle = lathe([(0, 0), (handle_r * 0.55, 0.0), (handle_r * 0.8, 0.012), (handle_r, 0.04), (handle_r, 0.085), (handle_r * 0.78, 0.098),
                    (handle_r * 0.6, 0.105), (0, 0.105)], 28)
    handle = handle.transformed(compose(translate((shaft, 0, 0)), rot_between((0, 1, 0), (1, 0, 0))))
    ridges = Mesh()
    for k in range(8):
        a = k * TAU / 8
        ridges += tube((shaft + 0.03, handle_r * np.cos(a), handle_r * np.sin(a)), (shaft + 0.085, handle_r * np.cos(a), handle_r * np.sin(a)), 0.0012, 6)
    stem = tube((0.0, 0.0, 0.0), (shaft, 0.0, 0.0), 0.0033, 10)
    tip = box((0.012, 0.0072, 0.0012), (0.006, 0.0, 0.0)) if flat else tube((0.0, 0.0, 0.0), (0.012, 0.0, 0.0), 0.003, 6)
    return {"steel": join(stem, tip), "handle": join(handle, ridges)}


def hook_pin(length: float = 0.03, r: float = 0.0022) -> Mesh:
    """Pegboard hook: out of the board, then up (hangs in the plane xy, pointing out along +z)."""
    path = [(0, 0, 0), (0, 0, length), (0, 0.006, length + 0.006), (0, 0.016, length + 0.006)]
    return sweep(path, r, 8)


# ----------------------------------------------------------------------------
# Bench: vise, grinder, small parts
# ----------------------------------------------------------------------------

def bench_vise() -> dict[str, Mesh]:
    """Bolted to the bench edge, jaws facing +z. Origin on the bench top at the jaw centre."""
    body = box((0.2, 0.09, 0.13), (0.0, 0.045, -0.085))
    base = box((0.18, 0.025, 0.2), (0.0, 0.0125, -0.1))
    swivel = lathe([(0.0, 0.0), (0.06, 0.0), (0.06, 0.02), (0.0, 0.02)], 24).transformed(translate((0.0, 0.025, -0.1)))
    jaw_fixed = box((0.2, 0.05, 0.03), (0.0, 0.1, -0.01))
    jaw_mov = box((0.2, 0.05, 0.03), (0.0, 0.1, 0.07))
    slide = join(tube((-0.06, 0.07, -0.02), (-0.06, 0.07, 0.07), 0.01, 12), tube((0.06, 0.07, -0.02), (0.06, 0.07, 0.07), 0.01, 12))
    screw = tube((0.0, 0.05, 0.08), (0.0, 0.05, 0.27), 0.012, 14)
    handle = join(tube((-0.1, 0.05, 0.27), (0.1, 0.05, 0.27), 0.006, 10),
                  ellipsoid((0.014, 0.014, 0.014), 10, 16).transformed(translate((-0.1, 0.05, 0.27))),
                  ellipsoid((0.014, 0.014, 0.014), 10, 16).transformed(translate((0.1, 0.05, 0.27))))
    hand = tube((0.0, 0.05, 0.27), (0.0, 0.05, 0.27), 0.0, 3)
    return {"paint": join(body, base, swivel, jaw_fixed.transformed(translate((0, 0, 0)))), "jaw": join(jaw_fixed, jaw_mov),
            "steel": join(slide, screw, handle)}


def bench_grinder() -> dict[str, Mesh]:
    motor = lathe([(0.0, -0.1), (0.055, -0.1), (0.07, -0.06), (0.07, 0.06), (0.055, 0.1), (0.0, 0.1)], 32).transformed(
        compose(translate((0.0, 0.1, 0.0)), rot_between((0, 1, 0), (1, 0, 0))))
    base = box((0.3, 0.03, 0.16), (0.0, 0.015, 0.0))
    stand = box((0.1, 0.06, 0.1), (0.0, 0.06, 0.0))
    wheels = Mesh()
    guards = Mesh()
    for s in (-1, 1):
        wheel = lathe([(0.0, -0.01), (0.075, -0.01), (0.075, 0.01), (0.0, 0.01)], 36).transformed(
            compose(translate((s * 0.13, 0.1, 0.0)), rot_between((0, 1, 0), (1, 0, 0))))
        wheels += wheel
        guards += lathe([(0.08, -0.015), (0.087, -0.015), (0.087, 0.015), (0.08, 0.015)], 36).transformed(
            compose(translate((s * 0.13, 0.1, 0.0)), rot_between((0, 1, 0), (1, 0, 0))))
        guards += box((0.03, 0.09, 0.1), (s * 0.13, 0.15, -0.03))
    rest = join(box((0.05, 0.004, 0.05), (-0.13, 0.065, 0.085)), box((0.05, 0.004, 0.05), (0.13, 0.065, 0.085)))
    return {"paint": join(motor, base, stand, guards), "wheel": wheels, "steel": rest}


def parts_cabinet(cols: int = 5, rows: int = 8) -> dict[str, Mesh]:
    """Small-drawer organiser: frame plus a grid of drawers with a pull and a label holder each."""
    w, h, d = 0.36, 0.32, 0.17
    cw, ch = (w - 0.012) / cols, (h - 0.012) / rows
    frame = join(box((w, 0.006, d), (0, 0.003, 0)), box((w, 0.006, d), (0, h - 0.003, 0)),
                 box((0.006, h, d), (-w / 2 + 0.003, h / 2, 0)), box((0.006, h, d), (w / 2 - 0.003, h / 2, 0)),
                 box((w, h, 0.004), (0, h / 2, -d / 2 + 0.002)))
    for i in range(1, cols):
        frame += box((0.003, h, d - 0.01), (-w / 2 + 0.006 + i * cw, h / 2, 0.0))
    for j in range(1, rows):
        frame += box((w, 0.003, d - 0.01), (0.0, 0.006 + j * ch, 0.0))
    fronts, pulls = Mesh(), Mesh()
    for i in range(cols):
        for j in range(rows):
            cx, cy = -w / 2 + 0.006 + (i + 0.5) * cw, 0.006 + (j + 0.5) * ch
            fronts += box((cw - 0.004, ch - 0.004, 0.006), (cx, cy, d / 2 - 0.003))
            pulls += box((cw * 0.4, 0.0035, 0.008), (cx, cy - ch * 0.12, d / 2 + 0.004))
            pulls += box((cw * 0.5, ch * 0.28, 0.0015), (cx, cy + ch * 0.15, d / 2 + 0.0008))
    return {"frame": frame, "fronts": fronts, "pulls": pulls}


# ----------------------------------------------------------------------------
# Tyres, wheels, drums, cans
# ----------------------------------------------------------------------------

def tyre(outer_r: float = 0.31, width: float = 0.2, rim_r: float = 0.2, grooves: int = 4, segments: int = 72) -> dict[str, Mesh]:
    """Tyre on the y axis, y in [0, width] (stack-ready, rests on y = 0): sidewall bulge, grooved tread in the profile."""
    m = min(0.065, 0.32 * width)
    n = grooves * 2 + 1
    xs = np.linspace(m, width - m, n + 1)
    treads = []
    for k in range(n):
        r = outer_r if k % 2 == 0 else outer_r - 0.008
        treads += [(r, xs[k]), (r, xs[k + 1])]
    side = outer_r - rim_r
    lower = [(rim_r, 0.0), (rim_r + 0.012, 0.0), (outer_r - 0.5 * side, 0.05 * width), (outer_r - 0.2 * side, 0.17 * width),
             (outer_r - 0.04 * side, m)]
    upper = [(outer_r - 0.04 * side, width - m), (outer_r - 0.2 * side, width - 0.17 * width), (outer_r - 0.5 * side, width - 0.05 * width),
             (rim_r + 0.012, width), (rim_r, width)]
    return {"rubber": lathe(lower + treads + upper + [(rim_r, width), (rim_r, 0.0)], segments)}


def wheel_rim(rim_r: float = 0.2, width: float = 0.17, spokes: int = 18, hub_r: float = 0.04) -> dict[str, Mesh]:
    """Wire-spoke rim on the y axis, centred at the origin; spokes are thin tubes (fine, dense edges)."""
    hw = width / 2
    rim = lathe([(rim_r - 0.012, -hw), (rim_r + 0.003, -hw * 0.9), (rim_r + 0.003, hw * 0.9), (rim_r - 0.012, hw),
                 (rim_r - 0.02, hw * 0.4), (rim_r - 0.02, -hw * 0.4), (rim_r - 0.012, -hw)], 64)
    hub = lathe([(0.0, -hw * 0.9), (hub_r, -hw * 0.9), (hub_r * 0.7, -hw * 0.5), (hub_r * 0.7, hw * 0.5), (hub_r, hw * 0.9), (0.0, hw * 0.9)], 24)
    sp = Mesh()
    for k in range(spokes):
        a = k * TAU / spokes
        side = -1 if k % 2 == 0 else 1
        flange = (hub_r * 0.95 * np.cos(a + 0.18 * side), side * hw * 0.85, hub_r * 0.95 * np.sin(a + 0.18 * side))
        rimpt = ((rim_r - 0.014) * np.cos(a), 0.0, (rim_r - 0.014) * np.sin(a))
        sp += tube(flange, rimpt, 0.0016, 6, False)
    return {"rim": rim, "hub": hub, "spokes": sp}


def oil_drum(radius: float = 0.29, height: float = 0.88) -> dict[str, Mesh]:
    prof = [(0.0, 0.0), (radius - 0.02, 0.0), (radius, 0.015), (radius, 0.05), (radius - 0.008, 0.06), (radius, 0.07)]
    for y in (0.3, 0.58):
        prof += [(radius, y - 0.03), (radius - 0.012, y - 0.016), (radius - 0.012, y + 0.016), (radius, y + 0.03)]
    prof += [(radius, height - 0.07), (radius - 0.008, height - 0.06), (radius, height - 0.05), (radius, height - 0.015),
             (radius - 0.02, height), (radius - 0.04, height - 0.01), (radius - 0.05, height - 0.03), (radius - 0.14, height - 0.03),
             (radius - 0.15, height - 0.02), (0.0, height - 0.02)]
    body = lathe(prof, 64)
    bung = lathe([(0.0, 0.0), (0.024, 0.0), (0.026, 0.012), (0.0, 0.014)], 20).transformed(translate((0.14, height - 0.02, 0.0)))
    bung += lathe([(0.0, 0.0), (0.018, 0.0), (0.02, 0.01), (0.0, 0.012)], 20).transformed(translate((-0.1, height - 0.02, 0.08)))
    return {"body": body, "bung": bung}


def jerry_can(w: float = 0.34, h: float = 0.46, d: float = 0.16) -> dict[str, Mesh]:
    """Pressed-steel fuel can: box body with a recessed X, a handle, a spout."""
    body = box((w, h, d), (0.0, h / 2, 0.0))
    handles = join(
        tube((-0.07, h, -0.0), (-0.07, h + 0.07, 0.0), 0.012, 10),
        tube((0.09, h, 0.0), (0.09, h + 0.07, 0.0), 0.012, 10),
        tube((-0.07, h + 0.07, 0.0), (0.09, h + 0.07, 0.0), 0.012, 10))
    spout = lathe([(0.0, 0.0), (0.025, 0.0), (0.025, 0.06), (0.032, 0.07), (0.032, 0.085), (0.0, 0.085)], 20).transformed(
        compose(translate((-0.13, h, 0.0)), rotate((0, 0, 1), 28)))
    ribs = Mesh()
    for k in range(2):
        a = np.array([(-w / 2 + 0.03, 0.04, d / 2 + 0.002), (w / 2 - 0.03, h - 0.06, d / 2 + 0.002)])
        if k:
            a = np.array([(-w / 2 + 0.03, h - 0.06, d / 2 + 0.002), (w / 2 - 0.03, 0.04, d / 2 + 0.002)])
        ribs += tube(a[0], a[1], 0.005, 6, False)
    ribs += box((w - 0.05, 0.012, 0.004), (0.0, h * 0.5, d / 2 + 0.002))
    return {"body": join(body, handles, ribs), "cap": spout}


def paint_can(radius: float = 0.075, height: float = 0.16) -> dict[str, Mesh]:
    body = lathe([(0.0, 0.0), (radius, 0.0), (radius + 0.003, 0.004), (radius, 0.008), (radius, height - 0.012), (radius + 0.003, height - 0.008),
                  (radius + 0.003, height), (radius - 0.01, height), (radius - 0.012, height - 0.006), (0.0, height - 0.006)], 36)
    handle = sweep([(radius + 0.004, height - 0.025, 0.0), (radius + 0.03, height + 0.035, 0.0), (0.0, height + 0.065, 0.0),
                    (-radius - 0.03, height + 0.035, 0.0), (-radius - 0.004, height - 0.025, 0.0)], 0.0016, 6)
    return {"can": body, "handle": handle}


def oil_bottle() -> dict[str, Mesh]:
    body = lathe([(0.0, 0.0), (0.04, 0.0), (0.045, 0.01), (0.045, 0.2), (0.03, 0.24), (0.015, 0.25), (0.015, 0.28), (0.0, 0.28)], 28)
    return {"body": body}


# ----------------------------------------------------------------------------
# Shelving, bins, boxes, tool chest, compressor
# ----------------------------------------------------------------------------

def shelving_bay(width: float = 1.0, depth: float = 0.45, height: float = 2.2, levels=(0.18, 0.62, 1.06, 1.5, 1.94)) -> dict[str, Mesh]:
    """Boltless steel shelving: angle-iron corner posts (L section), box-profile shelves with a front lip, cross braces."""
    post_w = 0.04
    posts = Mesh()
    for sx in (-1, 1):
        for sz in (-1, 1):
            cx, cz = sx * (width / 2 - post_w / 2), sz * (depth / 2 - post_w / 2)
            posts += box((post_w, height, 0.004), (cx, height / 2, cz + sz * (post_w / 2 - 0.002)))
            posts += box((0.004, height, post_w), (cx + sx * (post_w / 2 - 0.002), height / 2, cz))
    shelves = Mesh()
    for y in levels:
        shelves += box((width - 0.04, 0.016, depth - 0.02), (0.0, y, 0.0))
        shelves += box((width - 0.04, 0.035, 0.012), (0.0, y - 0.012, depth / 2 - 0.012))
        shelves += box((width - 0.04, 0.035, 0.012), (0.0, y - 0.012, -depth / 2 + 0.012))
    brace = Mesh()
    for sx in (-1, 1):
        brace += tube((sx * (width / 2 - 0.02), 0.2, -depth / 2 + 0.01), (sx * (width / 2 - 0.02), height - 0.3, depth / 2 - 0.01), 0.004, 6)
    brace += tube((-(width / 2 - 0.03), 0.3, -depth / 2 + 0.01), (width / 2 - 0.03, height - 0.3, -depth / 2 + 0.01), 0.004, 6)
    return {"post": join(posts, brace), "shelf": shelves}


def storage_bin(w: float = 0.3, h: float = 0.18, d: float = 0.4, wall: float = 0.003) -> dict[str, Mesh]:
    """Open-fronted plastic parts bin: floor, back, two sides, and a sloped front lip."""
    body = join(box((w, wall, d), (0, wall / 2, 0)), box((w, h, wall), (0, h / 2, -d / 2 + wall / 2)),
                box((wall, h, d), (-w / 2 + wall / 2, h / 2, 0)), box((wall, h, d), (w / 2 - wall / 2, h / 2, 0)),
                box((w, h * 0.45, wall), (0, h * 0.225, d / 2 - wall / 2)))
    label = box((w * 0.5, 0.045, 0.002), (0, h * 0.19, d / 2 + 0.001))
    return {"bin": body, "label": label}


def cardboard_box(w: float, h: float, d: float, flaps: bool = True) -> dict[str, Mesh]:
    body = box((w, h, d), (0, h / 2, 0))
    top = Mesh()
    if flaps:
        top += box((w / 2 - 0.004, 0.003, d), (-w / 4, h + 0.0015, 0.0)).transformed(compose(translate((-w / 2, h, 0)), rotate((0, 0, 1), 8), translate((w / 2, -h, 0))))
    return {"box": body, "flap": top}


def tool_chest(w: float = 0.7, h: float = 1.0, d: float = 0.45, drawers: int = 6) -> dict[str, Mesh]:
    """Rolling tool chest: cabinet, drawer fronts with handles, top tray, four casters."""
    base_y = 0.1
    body = box((w, h - base_y, d), (0, base_y + (h - base_y) / 2, 0))
    top = box((w + 0.02, 0.03, d + 0.02), (0, h + 0.015, 0))
    fronts, handles, caster = Mesh(), Mesh(), Mesh()
    gap = 0.006
    dh = (h - base_y - 0.04 - gap * (drawers + 1)) / drawers
    for k in range(drawers):
        y = base_y + 0.02 + gap + k * (dh + gap) + dh / 2
        fronts += box((w - 0.04, dh, 0.012), (0, y, d / 2 + 0.004))
        handles += tube((-w * 0.32, y + dh * 0.12, d / 2 + 0.03), (w * 0.32, y + dh * 0.12, d / 2 + 0.03), 0.007, 10)
        handles += tube((-w * 0.32, y + dh * 0.12, d / 2 + 0.03), (-w * 0.32, y + dh * 0.12, d / 2 + 0.01), 0.005, 8)
        handles += tube((w * 0.32, y + dh * 0.12, d / 2 + 0.03), (w * 0.32, y + dh * 0.12, d / 2 + 0.01), 0.005, 8)
    for sx in (-1, 1):
        for sz in (-1, 1):
            c = (sx * (w / 2 - 0.06), 0.0, sz * (d / 2 - 0.06))
            caster += lathe([(0.0, 0.0), (0.045, 0.0), (0.05, 0.02), (0.05, 0.05), (0.045, 0.07), (0.0, 0.07)], 20).transformed(
                compose(translate((c[0], 0.05, c[2])), rot_between((0, 1, 0), (1, 0, 0)), scale((1.0, 0.5, 1.0)))
            )
            caster += box((0.04, 0.04, 0.06), (c[0], 0.085, c[2]))
    return {"body": join(body, top), "front": fronts, "handle": handles, "caster": caster}


def air_compressor() -> dict[str, Mesh]:
    """Horizontal red tank on a frame with two rubber wheels, a motor and pump on top, a gauge."""
    tank = lathe([(0.0, -0.5), (0.1, -0.5), (0.2, -0.46), (0.23, -0.4), (0.23, 0.4), (0.2, 0.46), (0.1, 0.5), (0.0, 0.5)], 40).transformed(
        compose(translate((0.0, 0.42, 0.0)), rot_between((0, 1, 0), (1, 0, 0))))
    legs = join(*[tube((sx * 0.35, 0.2, sz * 0.13), (sx * 0.35, 0.0, sz * 0.2), 0.012, 8) for sx in (-1, 1) for sz in (-1, 1)])
    motor = lathe([(0.0, -0.13), (0.09, -0.13), (0.1, -0.1), (0.1, 0.1), (0.09, 0.13), (0.0, 0.13)], 28).transformed(
        compose(translate((-0.05, 0.75, 0.0)), rot_between((0, 1, 0), (0, 0, 1))))
    pump = join(lathe([(0.0, 0.0), (0.05, 0.0), (0.05, 0.12), (0.0, 0.12)], 20).transformed(translate((0.18, 0.64, 0.0))),
                *[box((0.14, 0.004, 0.14), (0.18, 0.7 + 0.018 * k, 0.0)) for k in range(5)])
    gauge = lathe([(0.0, 0.0), (0.045, 0.0), (0.05, 0.01), (0.05, 0.03), (0.0, 0.03)], 24).transformed(
        compose(translate((0.0, 0.5, 0.235)), rot_between((0, 1, 0), (0, 0, 1))))
    wheels = Mesh()
    for sz in (-1, 1):
        wheels += lathe([(0.0, -0.025), (0.12, -0.025), (0.13, -0.015), (0.13, 0.015), (0.12, 0.025), (0.0, 0.025)], 32).transformed(
            compose(translate((0.0, 0.13, sz * 0.22)), rot_between((0, 1, 0), (0, 0, 1))))
    hoses = sweep([(0.0, 0.62, 0.1), (0.1, 0.72, 0.1), (0.2, 0.72, 0.1), (0.2, 0.8, 0.02)], 0.007, 8)
    handle = join(tube((-0.45, 0.2, 0.0), (-0.45, 0.8, 0.0), 0.012, 8), tube((-0.45, 0.8, -0.18), (-0.45, 0.8, 0.18), 0.012, 8))
    return {"tank": tank, "frame": join(legs, handle), "motor": join(motor, pump), "gauge": gauge, "wheel": wheels, "hose": hoses}


# ----------------------------------------------------------------------------
# Chains, hoses, cables, I-beams, lights
# ----------------------------------------------------------------------------

def chain(length: float = 1.2, link_len: float = 0.045, link_w: float = 0.024, wire: float = 0.0045) -> Mesh:
    """Hanging chain along -y from the origin; links alternate between the xy and zy planes."""
    out = Mesh()
    pitch = link_len - 2 * wire
    n = int(length / pitch)
    ang = np.linspace(0, TAU, 20, endpoint=False)
    stad = []
    half_s = link_len / 2 - link_w / 2
    for a in ang:
        s = np.cos(a)
        t = np.sin(a)
        stad.append((link_w / 2 * s, (link_w / 2 * t) + (half_s if t >= 0 else -half_s)))
    stad = np.array(stad)
    base = sweep(np.column_stack([stad[:, 0], stad[:, 1], np.zeros(len(stad))]), wire, 6, closed=True)
    for k in range(n):
        m = base.transformed(compose(translate((0.0, -k * pitch - link_len / 2, 0.0)), rotate((0, 1, 0), 90.0 * (k % 2))))
        out += m
    return out


def helix_hose(radius: float = 0.16, turns: int = 6, pitch: float = 0.04, hose_r: float = 0.011, tail: float = 0.4) -> Mesh:
    """A coiled air hose lying on a hook (axis along z) with a free tail."""
    t = np.linspace(0, turns * TAU, turns * 28)
    path = np.column_stack([radius * np.cos(t), radius * np.sin(t), pitch * t / TAU])
    tail_pts = [path[-1] + np.array([0.0, -0.05 * k, 0.01 * k]) for k in range(1, 8)]
    return sweep(np.vstack([path, tail_pts]), hose_r, 10)


def drooping_cable(p0, p1, sag: float, radius: float = 0.006, n: int = 24) -> Mesh:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)[:, None]
    path = p0 + (p1 - p0) * t
    path[:, 1] -= sag * 4 * t[:, 0] * (1 - t[:, 0])
    return sweep(path, radius, 8)


def i_beam(length: float, depth: float = 0.2, flange: float = 0.1, web: float = 0.008, fl_t: float = 0.012) -> Mesh:
    """Along x, centred at the origin, hung with the top flange at y = 0 going down."""
    return join(box((length, fl_t, flange), (0, -fl_t / 2, 0)), box((length, fl_t, flange), (0, -depth + fl_t / 2, 0)),
                box((length, depth - 2 * fl_t, web), (0, -depth / 2, 0)))


def tube_light(length: float = 1.5) -> dict[str, Mesh]:
    """A batten fitting hung on two short chains: an open-bottom steel channel and two end caps; tubes are emitters in the scene."""
    chan = join(box((length, 0.006, 0.12), (0, 0.003, 0.0)), box((length, 0.04, 0.004), (0, -0.017, 0.058)),
                box((length, 0.04, 0.004), (0, -0.017, -0.058)))
    caps = join(box((0.012, 0.045, 0.125), (-length / 2 + 0.006, -0.015, 0.0)), box((0.012, 0.045, 0.125), (length / 2 - 0.006, -0.015, 0.0)))
    return {"housing": join(chan, caps)}


def caged_work_light() -> dict[str, Mesh]:
    """Trouble light: handle, cage of meridian wires plus rings, a hook. The bulb is an emitter in the scene."""
    cage = Mesh()
    r, h = 0.055, 0.2
    for k in range(8):
        a = k * TAU / 8
        pts = [(r * 0.55 * np.cos(a), 0.0, r * 0.55 * np.sin(a))]
        for s in np.linspace(0.0, 1.0, 9)[1:]:
            rad = r * (0.55 + 0.45 * np.sin(s * np.pi * 0.5 + 0.2)) if s < 0.85 else r * 0.5
            pts.append((rad * np.cos(a), -s * h, rad * np.sin(a)))
        cage += sweep(pts, 0.0011, 5, False)
    for y, rr in ((-0.05, r * 0.95), (-0.11, r), (-0.17, r * 0.8)):
        cage += sweep([(rr * np.cos(t), y, rr * np.sin(t)) for t in np.linspace(0, TAU, 28, endpoint=False)], 0.0012, 5, closed=True)
    top = lathe([(0.0, 0.05), (0.03, 0.05), (0.034, 0.0), (0.04, -0.02), (0.0, -0.02)], 20)
    hook = sweep([(0, 0.05, 0), (0, 0.1, 0), (0.02, 0.13, 0), (0.04, 0.12, 0)], 0.004, 8)
    return {"cage": cage, "body": join(top, hook)}


def spring(p0, p1, radius: float = 0.03, turns: int = 9, wire: float = 0.0035) -> Mesh:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    ln = np.linalg.norm(axis)
    s = np.linspace(0, 1, turns * 16)
    path = np.column_stack([radius * np.cos(s * turns * TAU), s * ln, radius * np.sin(s * turns * TAU)])
    return sweep(path, wire, 6).transformed(compose(translate(p0), rot_between((0, 1, 0), axis)))


def rag(w: float = 0.28, d: float = 0.2, seed: int = 0, rows: int = 24, cols: int = 32) -> Mesh:
    """A crumpled cloth lying on a flat surface: a grid with a few sine folds, edges pinned down."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, cols), np.linspace(-0.5, 0.5, rows))
    hgt = np.zeros_like(u)
    for _ in range(4):
        k = rng.uniform(5, 14)
        a = rng.uniform(0, np.pi)
        hgt += rng.uniform(0.004, 0.012) * np.sin(k * (u * np.cos(a) + v * np.sin(a)) + rng.uniform(0, TAU))
    env = np.clip(1.6 * (1 - (2 * u) ** 2) * (1 - (2 * v) ** 2), 0, 1)
    y = 0.004 + np.abs(hgt) * env + 0.0008
    p = np.stack([u * w, y, v * d], -1).reshape(-1, 3)
    f = G._grid_faces(rows, cols)
    n = G.vertex_normals(p, f)
    return Mesh(p, n, np.stack([u.ravel() + 0.5, v.ravel() + 0.5], 1), f).oriented()


def drill_press() -> dict[str, Mesh]:
    base = join(box((0.36, 0.03, 0.24), (0, 0.015, 0)), lathe([(0.0, 0.03), (0.07, 0.03), (0.06, 0.06), (0.0, 0.06)], 24))
    column = tube((0.0, 0.06, -0.02), (0.0, 1.3, -0.02), 0.04, 20)
    table = join(box((0.3, 0.028, 0.3), (0, 0.62, 0.1)), lathe([(0.0, 0.0), (0.055, 0.0), (0.055, 0.1), (0.0, 0.1)], 20).transformed(translate((0, 0.52, -0.02))))
    head = join(box((0.2, 0.2, 0.26), (0.0, 1.3, 0.08)), box((0.26, 0.08, 0.14), (0.0, 1.46, 0.0)))
    belt_cover = box((0.1, 0.1, 0.22), (0.0, 1.55, 0.0))
    motor = lathe([(0.0, -0.15), (0.075, -0.15), (0.085, -0.12), (0.085, 0.12), (0.075, 0.15), (0.0, 0.15)], 28).transformed(
        compose(translate((0.0, 1.62, -0.18)), rot_between((0, 1, 0), (0, 0, 1))))
    quill = lathe([(0.0, 0.0), (0.03, 0.0), (0.03, 0.12), (0.0, 0.12)], 20).transformed(translate((0.0, 1.1, 0.14)))
    chuck = lathe([(0.0, 0.0), (0.022, 0.0), (0.03, 0.02), (0.03, 0.06), (0.018, 0.075), (0.0, 0.075)], 24).transformed(translate((0.0, 1.03, 0.14)))
    bit = tube((0.0, 0.85, 0.14), (0.0, 1.03, 0.14), 0.005, 8)
    lever = join(*[tube((0.04, 1.22, 0.2), (0.04 + 0.12 * np.cos(a), 1.22 + 0.12 * np.sin(a), 0.2), 0.006, 8) for a in (0.3, 2.4, 4.5)],
                 *[ellipsoid((0.016, 0.016, 0.016), 8, 12).transformed(translate((0.04 + 0.12 * np.cos(a), 1.22 + 0.12 * np.sin(a), 0.2))) for a in (0.3, 2.4, 4.5)])
    return {"paint": join(base, column, table, head, belt_cover, motor), "steel": join(quill, chuck, bit), "knob": lever}


def moto_wheel(rim_r: float = 0.2, outer_r: float = 0.31, width: float = 0.12) -> dict[str, Mesh]:
    """A spoked motorcycle wheel with its axis along z, centred at the origin; disc brake on +z."""
    to_z = rotate((1, 0, 0), 90.0)
    ty = tyre(outer_r, width, rim_r, grooves=2, segments=72)["rubber"].transformed(compose(to_z, translate((0, -width / 2, 0))))
    w = wheel_rim(rim_r, width * 0.78, spokes=36, hub_r=0.032)
    rim = w["rim"].transformed(to_z)
    hub = w["hub"].transformed(to_z)
    spokes = w["spokes"].transformed(to_z)
    disc = lathe([(0.085, 0.0), (0.145, 0.0), (0.145, 0.004), (0.085, 0.004)], 48).transformed(compose(translate((0, 0, 0.04)), to_z))
    return {"rubber": ty, "rim": rim, "hub": join(hub, disc), "spokes": spokes}


def motorcycle() -> dict[str, Mesh]:
    """Naked roadster, x forward (front wheel at x = 1.38), y up with the tyres on y = 0, z across; symmetric in z."""
    parts = {k: Mesh() for k in ("rubber", "rim", "spokes", "chrome", "frame", "engine", "fins", "tank", "seat", "black", "glass", "hub")}
    for cx in (0.0, 1.38):
        wl = moto_wheel()
        for k in ("rubber", "rim", "spokes"):
            parts[k] += wl[k].transformed(translate((cx, 0.31, 0.0)))
        parts["hub"] += wl["hub"].transformed(translate((cx, 0.31, 0.0)))
    tb = lambda a, b, r=0.0125, sides=10, mat="frame": parts[mat].__iadd__(tube(a, b, r, sides))
    for s in (-1, 1):
        tb((1.12, 0.96, 0.03 * s), (0.80, 0.34, 0.1 * s))
        tb((0.80, 0.34, 0.1 * s), (0.46, 0.28, 0.1 * s))
        tb((0.46, 0.28, 0.1 * s), (0.36, 0.52, 0.105 * s))
        tb((0.36, 0.52, 0.105 * s), (0.30, 0.82, 0.07 * s))
        tb((0.55, 0.9, 0.08 * s), (-0.16, 0.82, 0.1 * s))
        tb((0.34, 0.8, 0.11 * s), (0.0, 0.31, 0.115 * s), 0.011)
        tb((0.36, 0.40, 0.125 * s), (0.0, 0.31, 0.12 * s), 0.02)                           # swingarm
        tb((1.22, 1.03, 0.095 * s), (1.37, 0.34, 0.095 * s), 0.0105, 10, "chrome")          # fork leg
        tb((1.18, 1.1, 0.095 * s), (1.2, 0.74, 0.095 * s), 0.0165, 12, "black")             # stanchion
        tb((0.12, 0.36, 0.17 * s), (0.16, 0.44, 0.17 * s), 0.01)                            # footpeg hanger
        tb((0.16, 0.44, 0.17 * s), (0.16, 0.44, 0.24 * s), 0.008, 8, "chrome")
        tb((1.39, 0.31, 0.06 * s), (1.39, 0.31, 0.11 * s), 0.014, 10, "chrome")             # axle ends
        tb((1.12, 0.98, 0.04 * s), (0.30, 0.86, 0.04 * s))
        parts["frame"] += spring((0.3, 0.84, 0.15 * s), (0.06, 0.4, 0.15 * s), 0.028, 10, 0.0032)
        tb((0.3, 0.84, 0.15 * s), (0.06, 0.4, 0.15 * s), 0.012, 10, "black")
    tb((1.12, 0.96, 0.0), (1.14, 1.04, 0.0), 0.03, 14)                                         # steering head
    tb((1.17, 1.07, -0.095), (1.17, 1.07, 0.095), 0.014, 10, "chrome")                         # top yoke
    parts["frame"] += tube((1.14, 0.95, -0.09), (1.14, 0.95, 0.09), 0.012, 8)
    # handlebar: tube with grips, levers, mirrors
    bar = sweep([(1.06, 1.14, -0.4), (1.09, 1.16, -0.28), (1.12, 1.15, -0.1), (1.12, 1.15, 0.1), (1.09, 1.16, 0.28), (1.06, 1.14, 0.4)], 0.0105, 10)
    parts["chrome"] += bar
    for s in (-1, 1):
        parts["black"] += tube((1.06, 1.14, 0.3 * s), (1.055, 1.14, 0.42 * s), 0.0165, 12)
        parts["chrome"] += tube((1.09, 1.145, 0.26 * s), (1.2, 1.12, 0.3 * s), 0.004, 6)
        parts["chrome"] += tube((1.06, 1.17, 0.36 * s), (1.0, 1.35, 0.4 * s), 0.004, 6)
        parts["black"] += ellipsoid((0.035, 0.022, 0.012), 10, 16).transformed(translate((0.98, 1.38, 0.41 * s)))
        parts["glass"] += ellipsoid((0.028, 0.016, 0.003), 8, 12).transformed(translate((0.975, 1.38, 0.41 * s)))
    # engine: crankcase, finned barrel, head, carburettors, airbox, exhaust
    parts["engine"] += box((0.3, 0.2, 0.2), (0.66, 0.36, 0.0))
    parts["engine"] += box((0.16, 0.1, 0.18), (0.5, 0.22, 0.0))
    barrel = lathe([(0.0, 0.0), (0.052, 0.0), (0.052, 0.17), (0.0, 0.17)], 24)
    barrel = barrel.transformed(compose(translate((0.78, 0.4, 0.0)), rotate((0, 0, 1), -18)))
    parts["engine"] += barrel
    for k in range(9):
        parts["fins"] += lathe([(0.0, 0.0), (0.082, 0.0), (0.082, 0.004), (0.0, 0.004)], 28).transformed(
            compose(translate((0.78, 0.4, 0.0)), rotate((0, 0, 1), -18), translate((0.0, 0.02 * k, 0.0))))
    parts["engine"] += box((0.14, 0.08, 0.16), (0.89, 0.67, 0.0)).transformed(compose(translate((0, 0, 0))))
    for s in (-1, 1):
        parts["engine"] += tube((0.89, 0.62, 0.06 * s), (0.64, 0.64, 0.06 * s), 0.024, 10)
        parts["chrome"] += tube((0.62, 0.64, 0.06 * s), (0.62, 0.66, 0.06 * s), 0.03, 10)
    parts["black"] += box((0.22, 0.13, 0.2), (0.5, 0.66, 0.0))
    ex = sweep([(0.9, 0.55, 0.06), (0.97, 0.38, 0.12), (0.88, 0.26, 0.19), (0.6, 0.2, 0.2), (0.2, 0.23, 0.2), (0.12, 0.28, 0.2)], 0.02, 12)
    parts["chrome"] += ex
    mu = lathe([(0.0, 0.0), (0.035, 0.0), (0.048, 0.06), (0.05, 0.3), (0.04, 0.34), (0.0, 0.35)], 24).transformed(
        compose(translate((0.12, 0.28, 0.2)), rot_between((0, 1, 0), (-0.99, 0.1, 0))))
    parts["chrome"] += mu
    ex2 = sweep([(0.9, 0.56, -0.06), (0.97, 0.36, -0.12), (0.9, 0.24, -0.19), (0.6, 0.19, -0.2), (0.2, 0.22, -0.2), (0.12, 0.27, -0.2)], 0.02, 12)
    parts["chrome"] += ex2
    parts["chrome"] += mu.transformed(compose(scale((1.0, 1.0, -1.0))))
    # tank, seat, tail, headlight, rear light
    parts["tank"] += ellipsoid((0.21, 0.12, 0.155), 26, 40).transformed(translate((0.78, 1.02, 0.0)))
    parts["tank"] += ellipsoid((0.2, 0.075, 0.1), 20, 30).transformed(translate((0.18, 0.99, 0.0)))
    parts["seat"] += ellipsoid((0.33, 0.045, 0.135), 18, 32).transformed(translate((0.34, 0.92, 0.0)))
    parts["tank"] += ellipsoid((0.14, 0.04, 0.09), 14, 24).transformed(compose(translate((-0.13, 0.94, 0.0)), rotate((0, 0, 1), 10)))
    parts["tank"] += ellipsoid((0.16, 0.012, 0.085), 8, 20).transformed(compose(translate((-0.04, 0.5, 0.0)), rotate((0, 0, 1), 12)))   # rear fender
    hl = lathe([(0.0, 0.0), (0.075, 0.0), (0.08, 0.02), (0.075, 0.07), (0.05, 0.1), (0.0, 0.11)], 36).transformed(
        compose(translate((1.2, 1.04, 0.0)), rot_between((0, 1, 0), (1, 0, 0))))
    parts["chrome"] += hl
    parts["glass"] += ellipsoid((0.01, 0.062, 0.062), 12, 20).transformed(translate((1.31, 1.04, 0.0)))
    parts["black"] += box((0.02, 0.03, 0.06), (-0.27, 0.93, 0.0))
    # chain
    rear_s, front_s = (0.0, 0.31), (0.5, 0.33)
    rs, fs = 0.075, 0.04
    ch = []
    for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 14):
        ch.append((rear_s[0] + rs * np.cos(a), rear_s[1] + rs * np.sin(a), 0.07))
    for a in np.linspace(-np.pi / 2, np.pi / 2, 12):
        ch.append((front_s[0] + fs * np.cos(a), front_s[1] + fs * np.sin(a), 0.07))
    parts["black"] += sweep([(x, y, z) for x, y, z in ch], 0.0055, 6, closed=True)
    parts["chrome"] += lathe([(0.0, 0.0), (rs, 0.0), (rs, 0.008), (0.0, 0.008)], 40).transformed(
        compose(translate((rear_s[0], rear_s[1], 0.065)), rotate((1, 0, 0), 90)))
    return parts


def stool(height: float = 0.5, radius: float = 0.17) -> dict[str, Mesh]:
    seat = lathe([(0.0, height), (radius, height), (radius + 0.008, height - 0.012), (radius, height - 0.035), (0.0, height - 0.035)], 36)
    legs = Mesh()
    for k in range(3):
        a = k * TAU / 3 + 0.3
        legs += tube((0.1 * np.cos(a), height - 0.035, 0.1 * np.sin(a)), (0.2 * np.cos(a), 0.0, 0.2 * np.sin(a)), 0.012, 8)
    ring = sweep([(0.16 * np.cos(t), 0.2, 0.16 * np.sin(t)) for t in np.linspace(0, TAU, 24, endpoint=False)], 0.007, 6, closed=True)
    return {"seat": seat, "legs": join(legs, ring)}


def bucket(radius: float = 0.14, height: float = 0.3) -> dict[str, Mesh]:
    body = lathe([(0.0, 0.0), (radius * 0.8, 0.0), (radius * 0.82, 0.01), (radius, height), (radius * 1.02, height + 0.006), (radius * 0.98, height + 0.01),
                  (radius * 0.94, height - 0.004), (radius * 0.78, 0.016), (0.0, 0.016)], 36)
    handle = sweep([(radius * 1.0 * np.cos(a), height - 0.01 + 0.17 * np.sin(a) ** 2 * 0 + 0.17 * np.sin(a), radius * 1.0 * 0 + 0.0) for a in np.linspace(0, np.pi, 18)], 0.0035, 6)
    return {"body": body, "handle": handle}


def broom(length: float = 1.5, seed: int = 0) -> dict[str, Mesh]:
    """Standing on its bristles at the origin, leaning is done by the caller. Bristles are ~100 thin tubes."""
    rng = np.random.default_rng(seed)
    head = box((0.3, 0.05, 0.035), (0.0, 0.17, 0.0))
    handle = tube((0.0, 0.17, 0.0), (0.0, length, 0.0), 0.012, 12)
    bristles = Mesh()
    for _ in range(110):
        x, z = rng.uniform(-0.145, 0.145), rng.uniform(-0.015, 0.015)
        flare = rng.uniform(-0.02, 0.02)
        bristles += tube((x, 0.15, z), (x * 1.05 + flare, 0.0, z + rng.uniform(-0.008, 0.008)), 0.0011, 4, False)
    return {"head": head, "handle": handle, "bristles": bristles}
