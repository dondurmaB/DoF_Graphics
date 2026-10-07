"""Mesh recipes for the terminal scene: roof trusses, vault shell, columns, balustrades, benches,
luggage, trolleys, potted trees, an escalator.

Same conventions as props.py: meters, y up, a local frame with the base at y = 0. Each recipe returns a
Mesh or a dict of Meshes keyed by material. Pure numpy; nothing here imports Mitsuba except `save_exr`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

import procedural as G
from procedural import Mesh, TAU, box, compose, lathe, rotate, scale, sweep, translate, vertex_normals


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def merge(meshes) -> Mesh:
    """Concatenate many meshes in one pass (Mesh += is quadratic for thousands of parts)."""
    meshes = list(meshes)
    if not meshes:
        return Mesh()
    offs = np.cumsum([0] + [len(m.p) for m in meshes[:-1]])
    return Mesh(np.vstack([m.p for m in meshes]), np.vstack([m.n for m in meshes]),
                np.vstack([m.uv for m in meshes]), np.vstack([m.f + o for m, o in zip(meshes, offs)]))


def tube(p0, p1, radius, sides: int = 8, caps: bool = True) -> Mesh:
    return sweep([p0, p1], radius, sides, caps)


def save_exr(path: Path, arr: np.ndarray) -> None:
    """Write a float RGB image atomically; the extension picks the format."""
    import mitsuba as mi

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".exr")
    os.close(fd)
    mi.Bitmap(np.ascontiguousarray(arr, np.float32)).write(tmp)
    os.replace(tmp, path)


def superellipsoid(radii, e_lat: float = 0.3, e_lon: float = 0.3, rows: int = 16, cols: int = 28) -> Mesh:
    """Rounded-box-like solid: exponent 1 is an ellipsoid, small exponents approach a box."""
    a, b, c = radii
    th = np.linspace(-np.pi / 2, np.pi / 2, rows)[:, None]
    ph = np.linspace(0.0, TAU, cols + 1)[None, :]
    sp = lambda u, e: np.sign(u) * np.abs(u) ** e
    cl, sl = sp(np.cos(th), e_lat), sp(np.sin(th), e_lat)
    p = np.stack(np.broadcast_arrays(a * cl * sp(np.cos(ph), e_lon), b * sl, c * cl * sp(np.sin(ph), e_lon)), -1)
    p = p.reshape(-1, 3)
    uv = np.stack(np.broadcast_arrays(ph / TAU, (th + np.pi / 2) / np.pi), -1).reshape(-1, 2)
    f = G._grid_faces(rows, cols + 1)
    return Mesh(p, vertex_normals(p, f), uv, f).oriented()


def arc_xy(radius: float, yc: float, a0: float, a1: float, n: int, z: float = 0.0) -> np.ndarray:
    """Points (radius sin a, yc + radius cos a, z) for a from a0 to a1 degrees, measured from straight up."""
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([radius * np.sin(a), yc + radius * np.cos(a), np.full(n, z)], -1)


# ----------------------------------------------------------------------------
# Roof: trusses and the vault shell
# ----------------------------------------------------------------------------

def rib(r_in: float, yc: float, half_deg: float, depth: float = 0.55, panels: int = 36) -> Mesh:
    """A Warren-truss rib in the plane z = 0: two chords joined by posts and diagonals."""
    n = panels + 1
    inner = arc_xy(r_in, yc, -half_deg, half_deg, n)
    outer = arc_xy(r_in + depth, yc, -half_deg, half_deg, n)
    parts = [sweep(arc_xy(r_in, yc, -half_deg, half_deg, 56), 0.075, 8), sweep(arc_xy(r_in + depth, yc, -half_deg, half_deg, 56), 0.075, 8)]
    for i in range(n):
        parts.append(tube(inner[i], outer[i], 0.032, 6))
        if i + 1 < n:
            a, b = (inner[i], outer[i + 1]) if i % 2 == 0 else (outer[i], inner[i + 1])
            parts.append(tube(a, b, 0.032, 6))
    return merge(parts)


def vault_shell(radius: float, yc: float, a0: float, a1: float, z0: float, z1: float, rows: int = 20) -> Mesh:
    """The inner face of a cylindrical shell between angles a0..a1 (degrees from straight up), z0..z1."""
    a = np.radians(np.linspace(a0, a1, rows))
    z = np.array([z0, 0.5 * (z0 + z1), z1])
    p = np.stack(np.broadcast_arrays((radius * np.sin(a))[:, None], (yc + radius * np.cos(a))[:, None], z[None, :]), -1)
    inward = np.stack([-np.sin(a), -np.cos(a), np.zeros_like(a)], -1)
    n = np.broadcast_to(inward[:, None, :], p.shape)
    uv = np.stack(np.broadcast_arrays((a * radius)[:, None], z[None, :]), -1)
    return Mesh(p.reshape(-1, 3), n.reshape(-1, 3), uv.reshape(-1, 2), G._grid_faces(rows, 3)).oriented()


def column(height: float = 9.0) -> Mesh:
    """A round painted-steel column with a plinth and a flared capital, base at y = 0."""
    h = height
    prof = [(0.0, 0.0), (0.42, 0.0), (0.42, 0.10), (0.34, 0.18), (0.29, 0.30), (0.29, h - 0.55), (0.34, h - 0.38),
            (0.46, h - 0.14), (0.50, h - 0.06), (0.50, h), (0.0, h)]
    return lathe(prof, 36)


def balustrade_bars(length: float, height: float = 1.0, pitch: float = 0.13) -> Mesh:
    """Vertical bars along +x from the origin, y from 0.06 to height."""
    n = int(length / pitch) + 1
    return merge(tube((i * pitch, 0.06, 0.0), (i * pitch, height, 0.0), 0.012, 5) for i in range(n))


# ----------------------------------------------------------------------------
# Furniture and luggage
# ----------------------------------------------------------------------------

def bench(seats: int = 4) -> dict[str, Mesh]:
    """A station bench along x (front toward +z): perforated-looking seat pans on a steel beam."""
    pitch, L = 0.55, 0.55 * seats
    seat, frame = [], []
    for i in range(seats):
        x = (i + 0.5) * pitch - L / 2
        seat.append(box((0.50, 0.05, 0.46), (x, 0.46, 0.0)))
        seat.append(box((0.50, 0.30, 0.04), (x, 0.74, -0.22)).transformed(rotate((1, 0, 0), 8.0)))
    frame.append(box((L + 0.04, 0.07, 0.10), (0.0, 0.34, -0.12)))
    for x in (-L / 2 + 0.05, L / 2 - 0.05):
        frame.append(tube((x, 0.0, 0.18), (x, 0.34, -0.12), 0.025, 8))
        frame.append(tube((x, 0.0, -0.40), (x, 0.34, -0.12), 0.025, 8))
        frame.append(box((0.05, 0.03, 0.62), (x, 0.0, -0.11)))
    return {"seat": merge(seat), "frame": merge(frame)}


def suitcase(w: float = 0.42, h: float = 0.66, d: float = 0.26) -> dict[str, Mesh]:
    """An upright hard-shell suitcase standing on its wheels; handle toward -z, telescopic handle at the back."""
    body = superellipsoid((w / 2, h / 2, d / 2), 0.22, 0.22, 14, 24).transformed(translate((0, 0.045 + h / 2, 0)))
    trim = [tube((-w * 0.14, h + 0.045, 0.0), (-w * 0.14, h + 0.085, 0.0), 0.012, 6),
            tube((w * 0.14, h + 0.045, 0.0), (w * 0.14, h + 0.085, 0.0), 0.012, 6),
            tube((-w * 0.14, h + 0.085, 0.0), (w * 0.14, h + 0.085, 0.0), 0.015, 6)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            trim.append(lathe([(0.0, 0.0), (0.028, 0.0), (0.028, 0.045), (0.0, 0.045)], 12)
                        .transformed(translate((sx * (w / 2 - 0.05), 0.0, sz * (d / 2 - 0.05)))))
        trim.append(tube((sx * 0.10, h * 0.45, -d / 2 - 0.012), (sx * 0.10, h + 0.04, -d / 2 - 0.012), 0.009, 6))
    trim.append(tube((-0.10, h + 0.03, -d / 2 - 0.012), (0.10, h + 0.03, -d / 2 - 0.012), 0.014, 6))
    # two vertical seams, so the shell reads as a hard case rather than a lump
    for sx in (-0.18 * w, 0.18 * w):
        trim.append(box((0.008, h * 0.88, d + 0.004), (sx, 0.045 + h / 2, 0.0)))
    return {"shell": body, "trim": merge(trim)}


def duffel(length: float = 0.62, rad: float = 0.17) -> dict[str, Mesh]:
    body = superellipsoid((length / 2, rad, rad), 0.55, 0.9, 12, 24).transformed(translate((0, rad, 0)))
    strap = [tube((x, rad * 2 - 0.01, -rad * 0.6), (x, rad * 2 + 0.07, 0.0), 0.012, 6) for x in (-0.13, 0.13)]
    strap.append(tube((-0.13, rad * 2 + 0.07, 0.0), (0.13, rad * 2 + 0.07, 0.0), 0.012, 6))
    return {"shell": body, "trim": merge(strap)}


def backpack() -> dict[str, Mesh]:
    body = superellipsoid((0.16, 0.26, 0.11), 0.4, 0.5, 12, 20).transformed(translate((0, 0.26, 0)))
    straps = [tube((x, 0.42, -0.105), (x * 0.9, 0.08, -0.10), 0.014, 6) for x in (-0.07, 0.07)]
    return {"shell": body, "trim": merge(straps)}


def trolley() -> dict[str, Mesh]:
    """An airport luggage cart, about 0.58 wide x 0.95 long, handle at +z."""
    f = []
    w, L, hb = 0.29, 0.95, 0.20
    for sx in (-1, 1):
        f.append(tube((sx * w, hb, -L / 2), (sx * w, hb, L / 2), 0.012, 6))
        f.append(tube((sx * w, hb, L / 2), (sx * w, 0.92, L / 2 + 0.05), 0.012, 6))
        f.append(tube((sx * w, hb, -L / 2), (sx * w, 0.60, -L / 2), 0.012, 6))
    f.append(tube((-w, 0.92, L / 2 + 0.05), (w, 0.92, L / 2 + 0.05), 0.016, 8))
    f.append(tube((-w, 0.60, -L / 2), (w, 0.60, -L / 2), 0.012, 6))
    for k in range(1, 6):
        f.append(tube((-w, hb, -L / 2 + k * L / 6), (w, hb, -L / 2 + k * L / 6), 0.008, 5))
    for k in range(1, 4):
        f.append(tube((-w, hb + k * 0.1, -L / 2), (w, hb + k * 0.1, -L / 2), 0.007, 5))
    wheels = [lathe([(0.0, -0.05), (0.05, -0.05), (0.05, 0.05), (0.0, 0.05)], 14).transformed(
        compose(translate((sx * w, 0.08, sz * (L / 2 - 0.02))), rotate((0, 0, 1), 90))) for sx in (-1, 1) for sz in (-1, 1)]
    return {"frame": merge(f), "wheel": merge(wheels)}


def potted_tree(seed: int, height: float = 3.2, leaves: int = 700) -> dict[str, Mesh]:
    """A ficus-like indoor tree in a tall pot: trunk, leafy crown of `leaves` blades."""
    rng = np.random.default_rng(seed)
    pot = lathe([(0.0, 0.0), (0.30, 0.0), (0.36, 0.05), (0.44, 0.78), (0.46, 0.82), (0.43, 0.84), (0.0, 0.84)], 28)
    trunk, crown_c = [], np.array([0.0, height - 1.0, 0.0])
    for k in range(3):
        a = rng.uniform(0, TAU)
        tip = crown_c + np.array([np.cos(a) * 0.35, rng.uniform(-0.3, 0.4), np.sin(a) * 0.35])
        path = [np.array([np.cos(a) * 0.05, 0.8, np.sin(a) * 0.05]),
                np.array([np.cos(a) * 0.10, 1.3, np.sin(a) * 0.10]), tip * 0.55 + np.array([0, 0.5, 0]), tip]
        trunk.append(sweep(path, [0.06, 0.05, 0.035, 0.02], 8))
    leaf = G.leaf(0.26, 0.11, curl=0.3, fold=0.25, rows=5, cols=3)
    blades = []
    for _ in range(leaves):
        d = rng.normal(size=3) * np.array([1.0, 0.8, 1.0])
        d /= np.linalg.norm(d)
        p = crown_c + d * np.array([1.05, 0.95, 1.05]) * rng.uniform(0.35, 1.0)
        axis = rng.normal(size=3)
        m = compose(translate(p), rotate(axis, rng.uniform(0, 360)), scale(rng.uniform(0.8, 1.35)))
        blades.append(leaf.transformed(m))
    return {"pot": pot, "trunk": merge(trunk), "leaf": merge(blades)}


def escalator(rise: float = 4.8, angle_deg: float = 30.0, width: float = 1.1) -> dict[str, Mesh]:
    """An escalator climbing toward +z; base of the lower end at the origin. uv of `steps` spans the slope."""
    a = np.radians(angle_deg)
    run = rise / np.tan(a)
    L = np.hypot(run, rise)
    tilt = rotate((1, 0, 0), -angle_deg)
    steps = box((width, 0.05, L), (0.0, 0.0, L / 2)).transformed(compose(translate((0, 0.18, 0)), tilt))
    skirts, rails, rub = [], [], []
    for sx in (-1, 1):
        x = sx * (width / 2 + 0.09)
        skirts.append(box((0.18, 0.95, L + 0.2), (0, 0.0, L / 2)).transformed(compose(translate((x, 0.45, 0)), tilt)))
        p0 = np.array([x, 1.10, 0.0])
        p1 = np.array([x, 1.10 + rise, run])
        rub.append(tube(p0, p1, 0.035, 8))
        for t in np.linspace(0.05, 0.95, 9):
            rails.append(tube(p0 * (1 - t) + p1 * t - [0, 0.25, 0], p0 * (1 - t) + p1 * t - [0, 0.55, 0], 0.018, 6))
    plates = [box((width + 0.4, 0.04, 0.9), (0.0, 0.0, 0.45)).transformed(translate((0, 0.17, 0))),
              box((width + 0.4, 0.04, 0.9), (0.0, 0.0, 0.45)).transformed(translate((0, rise + 0.17, run - 0.45)))]
    return {"steps": steps, "metal": merge(skirts + rails + plates), "rubber": merge(rub)}
