"""Procedural meshes and textures for the Mitsuba cafe scene.

Pure numpy + Pillow: nothing here imports Mitsuba, so assets can be generated
and inspected without a renderer. Every generator is deterministic, which keeps
the scene identical between a laptop preview and an HPC render.

Conventions: meters, y up. Lathe profiles are (radius, height) polylines walked
from the bottom centre outward, up the outside, over the rim and down the
inside; with that ordering the analytic normal (dy, -dr) points out of the
material on every segment.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

TAU = 2.0 * np.pi


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class Mesh:
    """Triangle soup with per-vertex normals and texture coordinates."""

    def __init__(self, p=None, n=None, uv=None, f=None):
        self.p = np.zeros((0, 3)) if p is None else np.asarray(p, dtype=np.float64)
        self.n = np.zeros((0, 3)) if n is None else np.asarray(n, dtype=np.float64)
        self.uv = np.zeros((0, 2)) if uv is None else np.asarray(uv, dtype=np.float64)
        self.f = np.zeros((0, 3), dtype=np.int64) if f is None else np.asarray(f, dtype=np.int64)

    def __iadd__(self, other: "Mesh") -> "Mesh":
        offset = len(self.p)
        self.p = np.vstack([self.p, other.p])
        self.n = np.vstack([self.n, other.n])
        self.uv = np.vstack([self.uv, other.uv])
        self.f = np.vstack([self.f, other.f + offset])
        return self

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        m = np.asarray(matrix, dtype=np.float64)
        p = self.p @ m[:3, :3].T + m[:3, 3]
        n = _normalize(self.n @ np.linalg.inv(m[:3, :3]))
        f = self.f if np.linalg.det(m[:3, :3]) > 0 else self.f[:, ::-1]
        return Mesh(p, n, self.uv.copy(), f.copy())

    def oriented(self) -> "Mesh":
        """Flip any triangle whose winding disagrees with its vertex normals.

        Mitsuba derives inside/outside from winding, so a dielectric whose
        winding and shading normals disagree would refract the wrong way.
        """
        a, b, c = (self.p[self.f[:, i]] for i in range(3))
        face_n = np.cross(b - a, c - a)
        vert_n = self.n[self.f].sum(axis=1)
        flip = np.einsum("ij,ij->i", face_n, vert_n) < 0
        f = self.f.copy()
        f[flip] = f[flip][:, ::-1]
        return Mesh(self.p, self.n, self.uv, f)

    def write_ply(self, path: Path) -> None:
        """Binary little-endian PLY with normals and uv, written atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        verts = np.hstack([self.p, self.n, self.uv]).astype("<f4")
        faces = np.empty(len(self.f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
        faces["n"] = 3
        faces["i"] = self.f
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(verts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property float u\nproperty float v\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\nend_header\n"
        )
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".ply.tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(verts.tobytes())
            fh.write(faces.tobytes())
        os.replace(tmp, path)


def vertex_normals(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = (p[f[:, i]] for i in range(3))
    face_n = np.cross(b - a, c - a)  # area weighted
    n = np.zeros_like(p)
    for i in range(3):
        np.add.at(n, f[:, i], face_n)
    return _normalize(n)


def _grid_faces(rows: int, cols: int) -> np.ndarray:
    """Two triangles per quad of a (rows x cols) vertex grid."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    i0 = (r * cols + c).ravel()
    i1, i2, i3 = i0 + 1, i0 + cols, i0 + cols + 1
    return np.concatenate([np.stack([i0, i2, i1], 1), np.stack([i1, i2, i3], 1)])


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------

def translate(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def scale(s) -> np.ndarray:
    return np.diag([*np.broadcast_to(np.asarray(s, dtype=float), (3,)), 1.0])


def rotate(axis, degrees: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, dtype=float))
    x, y, z = axis
    a = np.radians(degrees)
    c, s, t = np.cos(a), np.sin(a), 1.0 - np.cos(a)
    m = np.eye(4)
    m[:3, :3] = [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
    return m


def compose(*matrices) -> np.ndarray:
    out = np.eye(4)
    for m in matrices:
        out = out @ m
    return out


# --------------------------------------------------------------------------
# Primitive generators
# --------------------------------------------------------------------------

def lathe(profile, segments: int = 72, planar_uv: float | None = None) -> Mesh:
    """Revolve a (radius, height) polyline around the y axis.

    `planar_uv`, when given, maps uv from the xz position over that diameter
    instead of (angle, arc length) - used for table tops so wood grain runs
    straight across rather than radially.
    """
    prof = np.asarray(profile, dtype=np.float64)
    r, y = prof[:, 0], prof[:, 1]
    dr, dy = np.gradient(r), np.gradient(y)
    n2 = _normalize(np.stack([dy, -dr], 1))
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(r), np.diff(y)))])
    v = arc / max(arc[-1], 1e-9)

    theta = np.linspace(0.0, TAU, segments + 1)
    ct, st = np.cos(theta)[None, :], np.sin(theta)[None, :]
    grid = (len(r), len(theta))
    p = np.stack([r[:, None] * ct, np.broadcast_to(y[:, None], grid), r[:, None] * st], -1)
    n = np.stack([n2[:, :1] * ct, np.broadcast_to(n2[:, 1:], grid), n2[:, :1] * st], -1)
    if planar_uv:
        uv = np.stack([p[..., 0] / planar_uv + 0.5, p[..., 2] / planar_uv + 0.5], -1)
    else:
        uv = np.stack(np.broadcast_arrays(theta[None, :] / TAU, v[:, None]), -1)
    mesh = Mesh(p.reshape(-1, 3), n.reshape(-1, 3), uv.reshape(-1, 2),
                _grid_faces(len(r), len(theta)))
    return mesh.oriented()


def lathe_parts(profiles, segments: int = 72, planar_uv: float | None = None) -> Mesh:
    """Several polylines revolved into one mesh; each break is a hard crease."""
    out = Mesh()
    for prof in profiles:
        out += lathe(prof, segments, planar_uv)
    return out


def _frames(path: np.ndarray):
    """Parallel-transport frames along a polyline: no twist, no flips."""
    t = _normalize(np.gradient(path, axis=0))
    ref = np.array([0.0, 1.0, 0.0]) if abs(t[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normals = [_normalize(np.cross(t[0], ref))]
    for i in range(1, len(path)):
        prev = normals[-1]
        cand = prev - t[i] * np.dot(prev, t[i])
        normals.append(_normalize(cand) if np.linalg.norm(cand) > 1e-8 else prev)
    nrm = np.array(normals)
    return t, nrm, np.cross(t, nrm)


def sweep(path, radius, sides: int = 20, caps: bool = True, closed: bool = False) -> Mesh:
    """A tube of (optionally varying) circular section along a 3D polyline."""
    path = np.asarray(path, dtype=np.float64)
    if closed:
        path = np.vstack([path, path[:1]])
    rad = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(path),))
    t, nrm, bin_ = _frames(path)
    a = np.linspace(0.0, TAU, sides + 1)
    ring = np.cos(a)[None, :, None] * nrm[:, None, :] + np.sin(a)[None, :, None] * bin_[:, None, :]
    p = path[:, None, :] + rad[:, None, None] * ring
    uv = np.stack(np.broadcast_arrays(a[None, :] / TAU, np.linspace(0, 1, len(path))[:, None]), -1)
    mesh = Mesh(p.reshape(-1, 3), ring.reshape(-1, 3), uv.reshape(-1, 2),
                _grid_faces(len(path), sides + 1))
    if caps and not closed:
        for end, sign in ((0, -1.0), (-1, 1.0)):
            centre = path[end]
            rim = p[end]
            cp = np.vstack([centre, rim])
            cn = np.tile(sign * t[end], (len(cp), 1))
            cuv = np.vstack([[0.5, 0.5], 0.5 + 0.5 * np.stack([np.cos(a), np.sin(a)], 1)])
            cf = np.stack([np.zeros(sides, int), np.arange(1, sides + 1), np.arange(2, sides + 2)], 1)
            mesh += Mesh(cp, cn, cuv, cf)
    return mesh.oriented()


def box(size, centre=(0.0, 0.0, 0.0)) -> Mesh:
    """Axis-aligned box with per-face normals and uv in meters (tileable)."""
    sx, sy, sz = np.asarray(size, dtype=float) / 2.0
    out = Mesh()
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u_ax, v_ax = [i for i in range(3) if i != axis]
            half = np.array([sx, sy, sz])
            corners = []
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                c = np.zeros(3)
                c[axis] = sign * half[axis]
                c[u_ax] = du * half[u_ax]
                c[v_ax] = dv * half[v_ax]
                corners.append(c)
            corners = np.array(corners) + centre
            n = np.zeros(3)
            n[axis] = sign
            uv = np.array([[0, 0], [2 * half[u_ax], 0], [2 * half[u_ax], 2 * half[v_ax]], [0, 2 * half[v_ax]]])
            out += Mesh(corners, np.tile(n, (4, 1)), uv, np.array([[0, 1, 2], [0, 2, 3]]))
    return out.oriented()


def ellipsoid(radii, rows: int = 24, cols: int = 48) -> Mesh:
    rx, ry, rz = radii
    th = np.linspace(0.0, np.pi, rows)[:, None]
    ph = np.linspace(0.0, TAU, cols + 1)[None, :]
    unit = np.stack(np.broadcast_arrays(np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)), -1)
    p = unit * np.array([rx, ry, rz])
    n = _normalize(unit / np.array([rx, ry, rz]))
    uv = np.stack(np.broadcast_arrays(ph / TAU, th / np.pi), -1)
    return Mesh(p.reshape(-1, 3), n.reshape(-1, 3), uv.reshape(-1, 2), _grid_faces(rows, cols + 1)).oriented()


def leaf(length: float, width: float, curl: float = 0.25, fold: float = 0.3,
         rows: int = 10, cols: int = 5) -> Mesh:
    """A curved leaf blade along +z from the origin, surface roughly facing +y."""
    s = np.linspace(0.0, 1.0, rows)[:, None]
    w = np.linspace(-1.0, 1.0, cols)[None, :]
    half_width = width * 0.5 * np.sin(np.pi * np.clip(s, 0, 1)) ** 0.8
    x = w * half_width
    z = s * length
    y = -curl * length * s ** 2 + fold * np.abs(x)
    p = np.stack(np.broadcast_arrays(x, y, z), -1).reshape(-1, 3)
    f = _grid_faces(rows, cols)
    uv = np.stack(np.broadcast_arrays((w + 1) / 2, s), -1).reshape(-1, 2)
    return Mesh(p, vertex_normals(p, f), uv, f)


# --------------------------------------------------------------------------
# Textures
# --------------------------------------------------------------------------

def fbm(h: int, w: int, octaves: int = 5, base: int = 4, persistence: float = 0.5,
        seed: int = 0, aspect: float | None = None) -> np.ndarray:
    """Fractal value noise in [0, 1] via bicubic upsampling of random grids."""
    rng = np.random.default_rng(seed)
    aspect = (w / h) if aspect is None else aspect
    out = np.zeros((h, w), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = base * 2 ** o + 2
        gw = max(2, int(round(gh * aspect)))
        grid = rng.random((gh, gw)).astype(np.float32)
        out += amp * np.asarray(Image.fromarray(grid, mode="F").resize((w, h), Image.BICUBIC))
        total += amp
        amp *= persistence
    out /= total
    return np.clip((out - out.min()) / max(out.max() - out.min(), 1e-6), 0.0, 1.0)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".png")
    os.close(fd)
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def save_gray(path: Path, gray: np.ndarray) -> None:
    save_rgb(path, np.repeat(np.asarray(gray)[..., None], 3, axis=-1))


def _tint(base, variation, noise):
    base = np.asarray(base, dtype=np.float32)
    return base * (1.0 - variation + 2 * variation * noise[..., None])


def wood_planks(h: int, w: int, planks: int, board_len: float, seed: int = 1):
    """Floorboards running along v. Returns (albedo, roughness) in [0, 1].

    `planks` boards across u; each run of boards along v is `board_len` of the
    full v range, staggered per plank.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(w) + 0.5)[None, :] / w
    v = (np.arange(h) + 0.5)[:, None] / h
    plank = np.floor(u * planks).astype(int)
    lu = u * planks - plank
    offset = rng.random(planks + 1)[plank]
    board_coord = v / board_len + offset
    board = np.floor(board_coord).astype(int)
    board_id = plank * 1000 + board
    ids, inverse = np.unique(board_id, return_inverse=True)
    inverse = inverse.reshape(board_id.shape)
    tone = rng.normal(0.0, 1.0, len(ids))[inverse]
    warp = fbm(h, w, octaves=5, base=6, seed=seed + 1)
    fine = fbm(h, w, octaves=3, base=48, seed=seed + 2)
    grain = 0.5 + 0.5 * np.sin(TAU * (lu * 3.0 + warp * 6.0 + board_coord * 0.8 + tone * 0.3) * 2.2)
    grain = grain ** 2.5
    base = np.array([0.46, 0.29, 0.16], np.float32)
    dark = np.array([0.24, 0.13, 0.06], np.float32)
    mix = np.clip(0.55 * grain + 0.25 * fine + 0.12 * tone, 0, 1)[..., None]
    rgb = base * (1 - mix) + dark * mix
    rgb *= (1.0 + 0.10 * tone)[..., None]
    gap_u = np.minimum(lu, 1 - lu) * (w / planks)
    gap_v = np.minimum(board_coord % 1.0, 1 - board_coord % 1.0) * (h * board_len)
    gap = np.clip(np.minimum(gap_u, gap_v) / 1.5, 0, 1)[..., None]
    rgb = rgb * (0.35 + 0.65 * gap)
    rough = np.clip(0.38 + 0.12 * tone + 0.2 * fine + 0.3 * (1 - gap[..., 0]), 0.15, 0.95)
    return rgb, rough


def straight_wood(h: int, w: int, seed: int = 3, base=(0.42, 0.24, 0.12), dark=(0.20, 0.10, 0.05)):
    u = (np.arange(w) + 0.5)[None, :] / w
    v = (np.arange(h) + 0.5)[:, None] / h
    warp = fbm(h, w, octaves=5, base=3, seed=seed)
    fine = fbm(h, w, octaves=3, base=64, seed=seed + 1)
    rings = 0.5 + 0.5 * np.sin(TAU * (u * 14.0 + warp * 3.0 + 0.2 * np.sin(TAU * v * 2)))
    mix = np.clip(0.6 * rings ** 3 + 0.3 * fine, 0, 1)[..., None]
    rgb = np.asarray(base, np.float32) * (1 - mix) + np.asarray(dark, np.float32) * mix
    return rgb, np.clip(0.3 + 0.2 * fine, 0, 1)


def marble(h: int, w: int, seed: int = 5):
    u = (np.arange(w) + 0.5)[None, :] / w * (w / h)
    v = (np.arange(h) + 0.5)[:, None] / h
    turb = fbm(h, w, octaves=6, base=3, seed=seed)
    fine = fbm(h, w, octaves=4, base=24, seed=seed + 1)
    vein = np.abs(np.sin(TAU * (u * 1.3 + v * 0.7 + turb * 2.2)))
    vein = 1.0 - np.clip(vein * 3.0, 0, 1) ** 0.35
    base = np.array([0.86, 0.85, 0.82], np.float32) * (0.96 + 0.04 * fine[..., None])
    return np.clip(base * (1 - 0.55 * vein[..., None]), 0, 1)


def plaster(h: int, w: int, colour, seed: int = 7, strength: float = 0.07):
    n = 0.6 * fbm(h, w, octaves=6, base=4, seed=seed) + 0.4 * fbm(h, w, octaves=3, base=80, seed=seed + 1)
    return np.clip(_tint(colour, strength, n), 0, 1)


def subway_tiles(h: int, w: int, width_m: float, height_m: float, tile=(0.15, 0.075),
                 grout_m: float = 0.004, colour=(0.18, 0.33, 0.30), seed: int = 9):
    """Brick-bond glazed tiles for a face of width_m x height_m meters."""
    rng = np.random.default_rng(seed)
    x = (np.arange(w) + 0.5)[None, :] / w * width_m
    y = (np.arange(h) + 0.5)[:, None] / h * height_m
    row = np.floor(y / tile[1]).astype(int)
    xs = x / tile[0] + 0.5 * (row % 2)
    col = np.floor(xs).astype(int)
    tone = rng.normal(0, 1, (row.max() + 2, col.max() + 2))[row, col]
    gx = np.minimum(xs % 1.0, 1 - xs % 1.0) * tile[0]
    gy = np.minimum((y / tile[1]) % 1.0, 1 - (y / tile[1]) % 1.0) * tile[1]
    grout = (np.minimum(gx, gy) < grout_m)[..., None]
    rgb = np.asarray(colour, np.float32) * (1 + 0.08 * tone[..., None])
    rgb = np.where(grout, np.array([0.72, 0.70, 0.66], np.float32), rgb)
    rough = np.where(grout[..., 0], 0.9, 0.2 + 0.04 * np.abs(tone))
    return np.clip(rgb, 0, 1), np.clip(rough, 0, 1)


def chalkboard(h: int, w: int, seed: int = 11):
    """A dark board with smudges and rows of chalk 'handwriting'."""
    rng = np.random.default_rng(seed)
    smudge = fbm(h, w, octaves=5, base=3, seed=seed)
    rgb = np.array([0.045, 0.06, 0.055], np.float32) * (0.8 + 0.6 * smudge[..., None])
    chalk = np.zeros((h, w), np.float32)
    lines = rng.integers(7, 10)
    for i in range(lines):
        y0 = int(h * (0.12 + 0.8 * i / lines))
        x = int(w * 0.08)
        heading = i in (0, 4)
        th = max(2, h // (40 if heading else 70))
        while x < w * (0.9 if heading else rng.uniform(0.55, 0.88)):
            word = int(rng.uniform(0.04, 0.12 if heading else 0.09) * w)
            for c in range(x, min(x + word, w - 1), max(2, th)):
                hh = int(th * rng.uniform(1.5, 3.0 if heading else 2.2))
                chalk[max(0, y0 - hh):y0, c:c + max(1, th // 2)] = rng.uniform(0.6, 1.0)
            x += word + int(0.025 * w)
        if not heading:
            price_x = int(w * 0.86)
            chalk[y0 - th * 2:y0, price_x:price_x + int(0.06 * w)] = 0.8
    chalk = np.asarray(Image.fromarray(chalk, mode="F").resize((w, h), Image.BILINEAR))
    grit = fbm(h, w, octaves=2, base=256, seed=seed + 1)
    rgb = rgb + (chalk * (0.55 + 0.45 * grit))[..., None] * np.array([0.85, 0.85, 0.8])
    return np.clip(rgb, 0, 1)


def abstract_art(h: int, w: int, palette, seed: int = 13):
    """Soft colour-field painting for the framed pictures."""
    rng = np.random.default_rng(seed)
    palette = np.asarray(palette, np.float32)
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    rgb = np.tile(palette[0], (h, w, 1))
    for i in range(1, len(palette)):
        cx, cy, r = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.4)
        warp = fbm(h, w, octaves=4, base=3, seed=seed + i) * 0.25
        m = np.clip(1.0 - (np.hypot((x - cx) * 1.2, y - cy) + warp - r) * 8.0, 0, 1)[..., None]
        rgb = rgb * (1 - m) + palette[i] * m
    canvas = fbm(h, w, octaves=2, base=200, seed=seed + 99)[..., None]
    return np.clip(rgb * (0.92 + 0.1 * canvas), 0, 1)


def laptop_screen(h: int, w: int):
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    rgb = np.stack([0.20 + 0.25 * y, 0.35 + 0.2 * y, 0.60 + 0.2 * (1 - x)], -1)
    win = (x > 0.12) & (x < 0.78) & (y > 0.14) & (y < 0.8)
    rgb[win] = [0.93, 0.93, 0.95]
    for i in range(9):
        yy = 0.22 + i * 0.06
        line = win & (y > yy) & (y < yy + 0.018) & (x < 0.2 + 0.5 * ((i * 37) % 10) / 10 + 0.1)
        rgb[line] = [0.35, 0.37, 0.42]
    rgb[(y < 0.06)] = [0.12, 0.12, 0.14]
    return rgb
