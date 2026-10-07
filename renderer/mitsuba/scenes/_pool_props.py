"""Mesh recipes for the pool scene: the rippled water surface, the sloped pool floor, lane ropes, starting
blocks, ladders, a lifeguard chair, benches, backstroke flags, stadium seats, pendants, kickboards, noodles, trees.

Meters, y up. Recipes return a Mesh or a dict of Meshes keyed by material role. Pure numpy.
"""

from __future__ import annotations

import numpy as np

import procedural as G
from procedural import Mesh, TAU, box, compose, lathe, rotate, scale, sweep, translate, vertex_normals


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def merge(meshes) -> Mesh:
    """Concatenate many meshes in one pass (Mesh += is quadratic for thousands of parts)."""
    meshes = [m for m in meshes if len(m.f)]
    if not meshes:
        return Mesh()
    offs = np.cumsum([0] + [len(m.p) for m in meshes[:-1]])
    return Mesh(np.vstack([m.p for m in meshes]), np.vstack([m.n for m in meshes]),
                np.vstack([m.uv for m in meshes]), np.vstack([m.f + o for m, o in zip(meshes, offs)]))


def tube(p0, p1, radius, sides: int = 10, caps: bool = True) -> Mesh:
    return sweep([p0, p1], radius, sides, caps)


def quad(p0, p1, p3, normal, uv_unit: float = 1.0) -> Mesh:
    """Planar quad p0 -> p1 (u) and p0 -> p3 (v); uv in meters / uv_unit; faces wound to `normal`."""
    p0, p1, p3 = (np.asarray(v, float) for v in (p0, p1, p3))
    p2 = p1 + (p3 - p0)
    lu, lv = np.linalg.norm(p1 - p0), np.linalg.norm(p3 - p0)
    p = np.array([p0, p1, p2, p3])
    uv = np.array([[0, 0], [lu, 0], [lu, lv], [0, lv]]) / uv_unit
    n = np.tile(np.asarray(normal, float) / np.linalg.norm(normal), (4, 1))
    return Mesh(p, n, uv, np.array([[0, 1, 2], [0, 2, 3]])).oriented()


def superellipsoid(radii, e_lat: float = 0.3, e_lon: float = 0.3, rows: int = 14, cols: int = 24) -> Mesh:
    a, b, c = radii
    th = np.linspace(-np.pi / 2, np.pi / 2, rows)[:, None]
    ph = np.linspace(0.0, TAU, cols + 1)[None, :]
    sp = lambda u, e: np.sign(u) * np.abs(u) ** e
    cl, sl = sp(np.cos(th), e_lat), sp(np.sin(th), e_lat)
    p = np.stack(np.broadcast_arrays(a * cl * sp(np.cos(ph), e_lon), b * sl, c * cl * sp(np.sin(ph), e_lon)), -1).reshape(-1, 3)
    uv = np.stack(np.broadcast_arrays(ph / TAU, (th + np.pi / 2) / np.pi), -1).reshape(-1, 2)
    f = G._grid_faces(rows, cols + 1)
    return Mesh(p, vertex_normals(p, f), uv, f).oriented()


# ----------------------------------------------------------------------------
# Water and the pool tank
# ----------------------------------------------------------------------------

def water_surface(x0, x1, z0, z1, y, seed: int, spacing: float = 0.04, slope_rms: float = 0.035, waves: int = 32) -> Mesh:
    """A rippled height field over [x0, x1] x [z1, z0] (z0 > z1) at mean height y; normals point up (air side).

    A sum of `waves` sinusoids with wavelengths 6 cm to 2 m and a k^-0.75 amplitude falloff, scaled so the
    RMS surface slope is `slope_rms` (a calm indoor pool a few minutes after swimmers). The seed picks
    directions and phases.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(x0, x1, int(round((x1 - x0) / spacing)) + 1)
    zs = np.linspace(z0, z1, int(round((z0 - z1) / spacing)) + 1)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    H, Hx, Hz = (np.zeros_like(X) for _ in range(3))
    lam = np.exp(rng.uniform(np.log(0.06), np.log(2.0), waves))
    k = TAU / lam
    amp = k ** -0.75
    ang = rng.uniform(0, TAU, waves)
    phase = rng.uniform(0, TAU, waves)
    for i in range(waves):
        dx, dz = np.cos(ang[i]), np.sin(ang[i])
        arg = k[i] * (dx * X + dz * Z) + phase[i]
        s, c = np.sin(arg), np.cos(arg)
        H += amp[i] * s
        Hx += amp[i] * k[i] * dx * c
        Hz += amp[i] * k[i] * dz * c
    rms = np.sqrt(np.mean(Hx ** 2 + Hz ** 2))
    g = slope_rms / rms
    H, Hx, Hz = H * g, Hx * g, Hz * g
    p = np.stack([X, y + H, Z], -1).reshape(-1, 3)
    n = np.stack([-Hx, np.ones_like(H), -Hz], -1).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    uv = np.stack([X, Z], -1).reshape(-1, 2)
    return Mesh(p, n, uv, G._grid_faces(len(xs), len(zs))).oriented()


def pool_floor(x0, x1, z0, z1, y0, y1, step: float = 0.5) -> Mesh:
    """Floor sloping linearly from y0 at z0 to y1 at z1; uv in meters (x, -z); normal up."""
    xs = np.linspace(x0, x1, int(round((x1 - x0) / step)) + 1)
    zs = np.linspace(z0, z1, int(round(abs(z0 - z1) / step)) + 1)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    Y = y0 + (y1 - y0) * (Z - z0) / (z1 - z0)
    p = np.stack([X, Y, Z], -1).reshape(-1, 3)
    slope = (y1 - y0) / (z1 - z0)
    n = np.tile(np.array([0.0, 1.0, -slope]) / np.hypot(1.0, slope), (len(p), 1))
    uv = np.stack([X, -Z], -1).reshape(-1, 2)
    return Mesh(p, n, uv, G._grid_faces(len(xs), len(zs))).oriented()


# ----------------------------------------------------------------------------
# Lane ropes and backstroke flags
# ----------------------------------------------------------------------------

FLOAT_LEN = 0.11


def lane_float() -> Mesh:
    """One 'wave-eater' float: a 13 cm toothed disc stack, axis along z, centred at the origin."""
    prof = [(0.0, -0.055), (0.028, -0.055), (0.058, -0.046), (0.044, -0.032), (0.064, -0.017), (0.044, -0.004),
            (0.064, 0.010), (0.044, 0.024), (0.060, 0.040), (0.030, 0.055), (0.0, 0.055)]
    return lathe(prof, 16).transformed(rotate((1, 0, 0), 90.0))


def lane_rope(x: float, z0: float, z1: float, y: float, colours: list, rng) -> dict[int, Mesh]:
    """Floats edge to edge from z0 down to z1 at height y; colours[i] is the colour index of float i."""
    unit = lane_float()
    n = int((z0 - z1) / FLOAT_LEN)
    out: dict[int, list] = {}
    wob = rng.uniform(0, TAU)
    for i in range(n):
        z = z0 - (i + 0.5) * (z0 - z1) / n
        dx = 0.012 * np.sin(wob + z * 0.9) + 0.006 * np.sin(wob * 2 + z * 3.1)
        m = compose(translate((x + dx, y + rng.normal(0, 0.003), z)), rotate((0, 0, 1), rng.uniform(0, 360)))
        out.setdefault(colours[i], []).append(unit.transformed(m))
    return {c: merge(ms) for c, ms in out.items()}


def lane_colours(n: int, scheme: tuple[int, int, int], end_m: float = 5.0) -> list:
    """Colour index per float: the end colour within end_m of each wall, alternating colour pairs between."""
    end_c, a, b = scheme
    per_end = int(end_m / FLOAT_LEN)
    cols = []
    for i in range(n):
        if i < per_end or i >= n - per_end:
            cols.append(end_c)
        else:
            cols.append(a if (i // 6) % 2 == 0 else b)
    return cols


def flag_line(x0: float, x1: float, z: float, y: float, sag: float, colours: list, rng, pitch: float = 0.30) -> dict:
    """Triangular pennants on a sagging cord across x at height y: {colour index: Mesh, 'cord': Mesh}."""
    n = int((x1 - x0) / pitch)
    xs = np.linspace(x0, x1, 41)
    cord_y = y - sag * (1 - ((xs - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2)
    cord = sweep(np.stack([xs, cord_y, np.full_like(xs, z)], -1), 0.004, 5)
    out: dict = {"cord": [cord]}
    for i in range(n):
        xc = x0 + (i + 0.5) * pitch
        yc = y - sag * (1 - ((xc - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2)
        w, h = 0.21, 0.30
        p = np.array([[-w / 2, 0, 0], [w / 2, 0, 0], [0, -h, 0.0]])
        tw = rotate((0, 1, 0), rng.normal(0, 12.0)) @ rotate((1, 0, 0), rng.normal(0, 5.0))
        p = p @ tw[:3, :3].T + np.array([xc, yc, z])
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        nrm /= np.linalg.norm(nrm)
        m = Mesh(p, np.tile(nrm, (3, 1)), np.array([[0, 1], [1, 1], [0.5, 0]]), np.array([[0, 1, 2]]))
        out.setdefault(colours[i % len(colours)], []).append(m)
    return {c: merge(ms) for c, ms in out.items()}


# ----------------------------------------------------------------------------
# Deck furniture
# ----------------------------------------------------------------------------

def starting_block() -> dict[str, Mesh]:
    """A competition starting block standing on the deck behind the pool edge.

    Local frame: the pool edge is the line z = 0, the pool is toward -z, the deck at y = 0.
    """
    tilt = rotate((1, 0, 0), -8.0)                     # top falls toward the pool
    body = [box((0.52, 0.07, 0.60), (0.0, 0.0, 0.0)).transformed(compose(translate((0.0, 0.66, 0.06)), tilt))]
    mat = [box((0.48, 0.012, 0.54), (0.0, 0.0, 0.0)).transformed(compose(translate((0.0, 0.70, 0.06)), tilt))]
    kick = [box((0.46, 0.13, 0.03), (0.0, 0.0, 0.0)).transformed(
        compose(translate((0.0, 0.74, 0.27)), tilt, rotate((1, 0, 0), -35.0)))]
    steel = []
    for sx in (-0.19, 0.19):
        steel.append(box((0.06, 0.62, 0.06), (sx, 0.31, 0.30)))
        steel.append(box((0.06, 0.04, 0.48), (sx, 0.02, 0.20)))
        steel.append(tube((sx, 0.60, 0.29), (sx, 0.64, -0.10), 0.02, 8))
    steel.append(box((0.44, 0.05, 0.05), (0.0, 0.12, 0.30)))
    handle = [tube((-0.24, 0.62, -0.03), (0.24, 0.62, -0.03), 0.014, 8),
              tube((-0.27, 0.44, 0.0), (-0.27, 0.66, 0.0), 0.014, 8), tube((0.27, 0.44, 0.0), (0.27, 0.66, 0.0), 0.014, 8),
              tube((-0.27, 0.66, 0.0), (-0.24, 0.62, -0.03), 0.014, 8), tube((0.27, 0.66, 0.0), (0.24, 0.62, -0.03), 0.014, 8)]
    return {"body": merge(body + kick), "mat": merge(mat), "steel": merge(steel), "handle": merge(handle)}


def ladder(depth: float = 1.2) -> dict[str, Mesh]:
    """A stainless pool ladder. Local frame: pool wall in the plane z = 0, pool toward -z, deck at y = 0."""
    rails = []
    for sx in (-0.24, 0.24):
        arc = [(sx, 0.0, 0.36), (sx, 0.55, 0.36)]
        for a in np.linspace(0, 180, 13):
            r = np.radians(a)
            arc.append((sx, 0.62 + 0.20 * np.sin(r), 0.20 + 0.16 * np.cos(r)))
        arc += [(sx, 0.55, 0.04), (sx, 0.25, -0.10), (sx, -0.20, -0.13), (sx, -depth, -0.13), (sx, -depth - 0.05, -0.01)]
        rails.append(sweep(np.array(arc), 0.021, 12))
    treads = []
    for k, d in enumerate((0.38, 0.68, 0.98)):
        if d < depth:
            treads.append(box((0.46, 0.025, 0.09), (0.0, -d, -0.075)))
            treads.append(box((0.46, 0.015, 0.012), (0.0, -d + 0.02, -0.11)))
    return {"steel": merge(rails + treads)}


def lifeguard_chair() -> dict[str, Mesh]:
    """A tall lifeguard chair facing +z: splayed legs, rungs, seat at 1.55 m, backrest."""
    frame = []
    legs = [((-0.42, 0.0, -0.45), (-0.30, 1.50, -0.25)), ((0.42, 0.0, -0.45), (0.30, 1.50, -0.25)),
            ((-0.42, 0.0, 0.55), (-0.30, 1.50, 0.25)), ((0.42, 0.0, 0.55), (0.30, 1.50, 0.25))]
    for a, b in legs:
        frame.append(tube(a, b, 0.028, 10))
    for t in np.linspace(0.12, 0.85, 6):
        for (a, b), (c, d) in ((legs[2], legs[3]), (legs[0], legs[2]), (legs[1], legs[3])):
            p = np.asarray(a) * (1 - t) + np.asarray(b) * t
            q = np.asarray(c) * (1 - t) + np.asarray(d) * t
            frame.append(tube(p, q, 0.018, 8))
    frame.append(tube((-0.30, 1.50, -0.25), (-0.30, 2.25, -0.32), 0.022, 8))
    frame.append(tube((0.30, 1.50, -0.25), (0.30, 2.25, -0.32), 0.022, 8))
    frame.append(tube((-0.30, 1.50, 0.25), (-0.30, 1.85, 0.25), 0.022, 8))
    frame.append(tube((0.30, 1.50, 0.25), (0.30, 1.85, 0.25), 0.022, 8))
    frame.append(tube((-0.30, 1.85, 0.25), (-0.30, 1.85, -0.25), 0.02, 8))
    frame.append(tube((0.30, 1.85, 0.25), (0.30, 1.85, -0.25), 0.02, 8))
    seat = [box((0.66, 0.05, 0.56), (0.0, 1.53, 0.0)),
            box((0.62, 0.42, 0.04), (0.0, 0.0, 0.0)).transformed(compose(translate((0.0, 1.95, -0.29)), rotate((1, 0, 0), -10.0)))]
    return {"frame": merge(frame), "seat": merge(seat)}


def bench(length: float = 2.0) -> dict[str, Mesh]:
    """A slatted deck bench along x, front toward +z."""
    wood = [box((length, 0.035, 0.11), (0.0, 0.44, z)) for z in (-0.15, 0.0, 0.15)]
    wood += [box((length, 0.10, 0.03), (0.0, 0.0, 0.0)).transformed(compose(translate((0.0, 0.62, -0.23)), rotate((1, 0, 0), -12.0)))
             for _ in range(1)]
    wood += [box((length, 0.10, 0.03), (0.0, 0.0, 0.0)).transformed(compose(translate((0.0, 0.76, -0.26)), rotate((1, 0, 0), -12.0)))]
    steel = []
    for x in (-length / 2 + 0.15, length / 2 - 0.15):
        steel.append(box((0.04, 0.42, 0.04), (x, 0.21, 0.18)))
        steel.append(box((0.04, 0.80, 0.04), (x, 0.40, -0.22)).transformed(compose(translate((0, 0, 0)), rotate((1, 0, 0), 0.0))))
        steel.append(box((0.04, 0.04, 0.46), (x, 0.42, -0.02)))
    return {"wood": merge(wood), "steel": merge(steel)}


def stadium_seat() -> dict[str, Mesh]:
    """A moulded plastic spectator seat facing +z, seat surface at y = 0.42 above its tier."""
    pan = superellipsoid((0.21, 0.035, 0.20), 0.25, 0.35, 10, 18).transformed(translate((0.0, 0.42, 0.02)))
    back = superellipsoid((0.21, 0.19, 0.03), 0.3, 0.35, 10, 18).transformed(
        compose(translate((0.0, 0.62, -0.17)), rotate((1, 0, 0), -12.0)))
    post = merge([box((0.05, 0.40, 0.05), (0.0, 0.20, -0.05)), box((0.30, 0.03, 0.08), (0.0, 0.015, -0.05))])
    return {"shell": merge([pan, back]), "post": post}


def pendant(radius: float = 0.45, height: float = 0.22) -> Mesh:
    """A high-bay pendant housing open at the bottom (y = 0); the emitter disk sits just inside it."""
    prof = [(radius - 0.02, 0.0), (radius, 0.0), (radius, height * 0.7), (radius * 0.55, height), (0.06, height + 0.05),
            (0.06, height + 0.12), (0.0, height + 0.12)]
    return lathe(prof, 36)


def kickboard() -> Mesh:
    return superellipsoid((0.22, 0.016, 0.145), 0.15, 0.35, 8, 20)


def noodle(rng, length: float = 1.5, radius: float = 0.035) -> Mesh:
    bend = rng.normal(0, 0.05, 2)
    t = np.linspace(0, 1, 16)
    path = np.stack([bend[0] * np.sin(np.pi * t), t * length, bend[1] * np.sin(np.pi * t)], -1)
    return sweep(path, radius, 10)


def rack(width: float = 1.2, depth: float = 0.45, height: float = 1.25, shelves=(0.15, 0.6, 1.05)) -> Mesh:
    """A wire storage rack: four posts, a frame and slats per shelf."""
    parts = []
    for sx in (-width / 2, width / 2):
        for sz in (-depth / 2, depth / 2):
            parts.append(tube((sx, 0.0, sz), (sx, height, sz), 0.012, 6))
    for y in shelves:
        for sz in (-depth / 2, depth / 2):
            parts.append(tube((-width / 2, y, sz), (width / 2, y, sz), 0.009, 5))
        for sx in np.linspace(-width / 2, width / 2, 9):
            parts.append(tube((sx, y, -depth / 2), (sx, y, depth / 2), 0.005, 4))
    return merge(parts)


def tree(seed: int, height: float = 7.0, crown: float = 2.6, leaves: int = 2400) -> dict[str, Mesh]:
    """A deciduous tree: a trunk forking into a few limbs, and an irregular crown of leaf blades."""
    rng = np.random.default_rng(seed)
    trunk = [sweep([(0, 0, 0), (0.05, height * 0.25, 0.0), (0.0, height * 0.45, 0.05)], [0.22, 0.17, 0.14], 10)]
    centres = []
    for k in range(4):
        a = rng.uniform(0, TAU)
        tip = np.array([np.cos(a) * crown * 0.5, height * rng.uniform(0.62, 0.82), np.sin(a) * crown * 0.5])
        base = np.array([0.0, height * 0.45, 0.05])
        trunk.append(sweep([base, base * 0.5 + tip * 0.5 + np.array([0, 0.3, 0]), tip], [0.12, 0.08, 0.04], 8))
        centres.append(tip)
    leaf = G.leaf(0.13, 0.07, curl=0.25, fold=0.2, rows=4, cols=3)
    blades = []
    for i in range(leaves):
        c = centres[i % len(centres)]
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        p = c + d * np.array([crown * 0.62, crown * 0.48, crown * 0.62]) * rng.uniform(0.3, 1.0) ** 0.5
        m = compose(translate(p), rotate(rng.normal(size=3), rng.uniform(0, 360)), scale(rng.uniform(0.8, 1.4)))
        blades.append(leaf.transformed(m))
    return {"trunk": merge(trunk), "leaf": merge(blades)}


def treeline(radius: float, seed: int, segments: int = 720, base: float = 9.0, var: float = 7.0) -> Mesh:
    """A ring of distant woodland: a vertical band at `radius` around the origin whose top follows a noisy
    canopy silhouette (base +- var metres); normals face the centre."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0.0, TAU, segments + 1)
    h = np.zeros_like(th)
    for k, a in ((3, 0.35), (11, 0.3), (37, 0.2), (113, 0.15)):
        ph = rng.uniform(0, TAU, 2)
        h += a * (np.sin(k * th + ph[0]) + 0.5 * np.sin(2.3 * k * th + ph[1]))
    h = base + var * h / np.abs(h).max()
    x, z = radius * np.cos(th), radius * np.sin(th)
    p = np.concatenate([np.stack([x, np.full_like(x, -0.3), z], -1), np.stack([x, h, z], -1)])
    n = np.concatenate([np.stack([-np.cos(th), 0 * th, -np.sin(th)], -1)] * 2)
    uv = np.concatenate([np.stack([th / TAU, np.zeros_like(th)], -1), np.stack([th / TAU, np.ones_like(th)], -1)])
    f = G._grid_faces(2, segments + 1)
    return Mesh(p, n, uv, f).oriented()


def hedge(length: float, seed: int, height: float = 1.2, depth: float = 0.9, leaves: int = 5000) -> Mesh:
    """Leaves filling a box along x (centred), for a clipped hedge."""
    rng = np.random.default_rng(seed)
    leaf = G.leaf(0.09, 0.05, curl=0.2, fold=0.2, rows=4, cols=3)
    blades = []
    for _ in range(leaves):
        p = np.array([rng.uniform(-length / 2, length / 2), rng.uniform(0.05, height), rng.uniform(-depth / 2, depth / 2)])
        blades.append(leaf.transformed(compose(translate(p), rotate(rng.normal(size=3), rng.uniform(0, 360)))))
    return merge(blades)
