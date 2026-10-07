"""Meshes and textures for night_street: cars, lamps, street furniture, trees, ground decals.

Pure numpy + Pillow except `save_exr` (Mitsuba). Deterministic in the arguments. Facades live in
`_night_street_facade.py`. Texture convention: row 0 of a returned array is v = 0, which Mitsuba
puts at the BOTTOM of a quad, so images meant to be read upright are flipped before saving.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import procedural as G
from procedural import TAU, Mesh
from props import bezier

UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

class Acc:
    """Collects many small meshes and concatenates once (Mesh += is quadratic)."""

    def __init__(self):
        self.p, self.n, self.uv, self.f, self.off = [], [], [], [], 0

    def add(self, mesh: Mesh, matrix=None) -> None:
        if matrix is not None:
            mesh = mesh.transformed(matrix)
        if len(mesh.f) == 0:
            return
        self.p.append(mesh.p), self.n.append(mesh.n), self.uv.append(mesh.uv), self.f.append(mesh.f + self.off)
        self.off += len(mesh.p)

    def mesh(self) -> Mesh:
        if not self.p:
            return Mesh()
        return Mesh(np.vstack(self.p), np.vstack(self.n), np.vstack(self.uv), np.vstack(self.f))


class Groups(defaultdict):
    """material name -> Acc"""

    def __init__(self):
        super().__init__(Acc)

    def meshes(self) -> dict[str, Mesh]:
        return {k: v.mesh() for k, v in self.items() if v.off}


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def frame(origin, forward, up=UP) -> np.ndarray:
    """Local +z -> forward, +y -> up (as far as perpendicular)."""
    f = unit(forward)
    r = np.cross(up, f)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r = unit(r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, np.cross(f, r), f, origin
    return m


def quad(p0, p1, p2, p3, uv=None) -> Mesh:
    """Planar quad p0 p1 p2 p3 counter-clockwise seen from the side it faces. uv defaults to meters."""
    p = np.array([p0, p1, p2, p3], float)
    n = unit(np.cross(p[1] - p[0], p[3] - p[0]))
    if uv is None:
        a, b = np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0])
        uv = [(0, 0), (a, 0), (a, b), (0, b)]
    return Mesh(p, np.tile(n, (4, 1)), np.asarray(uv, float), np.array([[0, 1, 2], [0, 2, 3]]))


def join(*meshes: Mesh) -> Mesh:
    acc = Acc()
    for m in meshes:
        acc.add(m)
    return acc.mesh()


def box(size, centre=(0.0, 0.0, 0.0)) -> Mesh:
    return G.box(size, centre)


def tube(points, radius, sides: int = 8, caps: bool = True) -> Mesh:
    return G.sweep(np.asarray(points, float), radius, sides, caps)


def seg_box(p0, p1, w: float, h: float) -> Mesh:
    """A w x h box from p0 to p1 (for railings, stair stringers, sign letters)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    if L < 1e-6:
        return Mesh()
    m = frame((p0 + p1) / 2, p1 - p0, UP if abs(unit(p1 - p0)[1]) < 0.95 else np.array([1.0, 0.0, 0.0]))
    return box((w, h, L)).transformed(m)


def save_exr(path, arr: np.ndarray) -> None:
    import mitsuba as mi

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".exr")
    os.close(fd)
    mi.Bitmap(np.ascontiguousarray(arr, np.float32)).write(tmp)
    os.replace(tmp, path)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------
# Ground textures (tiling, uv in meters scaled by the material)
# ---------------------------------------------------------------------------

ASPHALT_TILE_M = 2.0


def asphalt(size: int = 2048, seed: int = 3, patch: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate asphalt over ASPHALT_TILE_M: binder, stones of mixed size, fines. Returns (albedo, height)."""
    rng = np.random.default_rng(seed)
    n = 0.6 * G.fbm(size, size, octaves=6, base=3, seed=seed) + 0.4 * G.fbm(size, size, octaves=3, base=180, seed=seed + 1)
    base = (0.045 if patch else 0.06) + 0.035 * n
    alb = np.repeat(base[..., None], 3, -1).astype(np.float32)
    height = 0.3 * n.astype(np.float32)
    img = Image.new("L", (size, size), 0)
    hgt = Image.new("L", (size, size), 0)
    d, dh = ImageDraw.Draw(img), ImageDraw.Draw(hgt)
    px_per_m = size / ASPHALT_TILE_M
    for _ in range(int(9000 if not patch else 5000)):
        r = max(1.0, rng.lognormal(math.log(0.0035 * px_per_m), 0.45))
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        tone = int(np.clip(rng.normal(150, 45), 40, 255))
        pts = [(x + r * rng.uniform(0.7, 1.2) * math.cos(a), y + r * rng.uniform(0.7, 1.2) * math.sin(a))
               for a in np.linspace(0, TAU, 7)[:-1] + rng.uniform(0, 1)]
        d.polygon(pts, fill=tone)
        dh.polygon(pts, fill=255)
    stones = np.asarray(img, np.float32) / 255.0
    sh = np.asarray(hgt, np.float32) / 255.0
    grey = 0.10 + 0.22 * stones
    tint = np.stack([grey, grey * 0.98, grey * 0.96], -1)
    alb = np.where(sh[..., None] > 0, tint, alb)
    height = np.clip(height + 0.6 * sh, 0, 1)
    return np.clip(alb, 0, 1), height


PAVING_TILE_M = 2.4  # 4 x 4 slabs of 0.6 m


def paving(size: int = 2048, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Concrete slabs with joints, per-slab tone, chewing-gum spots and grime. Returns (albedo, height)."""
    rng = np.random.default_rng(seed)
    slabs = 4
    idx = (np.arange(size) * slabs // size)
    tone = rng.normal(0.0, 1.0, (slabs, slabs))[idx[:, None], idx[None, :]]
    n = 0.55 * G.fbm(size, size, octaves=6, base=4, seed=seed) + 0.45 * G.fbm(size, size, octaves=3, base=140, seed=seed + 1)
    v = 0.30 + 0.035 * tone + 0.09 * n
    pos = (np.arange(size) * slabs / size) % 1.0
    edge = np.minimum(pos, 1 - pos) * (PAVING_TILE_M / slabs) * 1000.0          # mm to the nearest joint
    joint = np.clip(np.minimum(edge[None, :], edge[:, None]) / 4.0, 0, 1)
    v = v * (0.35 + 0.65 * joint)
    gum = rng.random((size, size)) < 0.00008
    v = np.where(gum, 0.12, v)
    alb = np.stack([v, v * 0.98, v * 0.94], -1)
    height = np.clip(0.5 + 0.2 * n - 0.5 * (1 - joint), 0, 1)
    return np.clip(alb, 0, 1).astype(np.float32), height.astype(np.float32)


def granite(size: int = 512, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = G.fbm(size, size, octaves=4, base=8, seed=seed)
    grain = rng.random((size, size))
    v = 0.33 + 0.08 * n + np.where(grain < 0.08, 0.18, 0.0) - np.where(grain > 0.94, 0.15, 0.0)
    return np.clip(np.stack([v, v, v * 1.02], -1), 0, 1).astype(np.float32)


def road_roughness(wet: bool, seed: int, puddles: list, h: int = 4096, w: int = 768,
                   width: float = 9.0, length: float = 80.0, z_far: float = -70.0) -> np.ndarray:
    """Roughness over the carriageway (col 0 at x = -width/2, row 0 at z = z_far). Wet asphalt is
    rough enough that lights streak; only the puddles are near-mirrors."""
    n = G.fbm(h, w, octaves=5, base=6, seed=seed, aspect=width / length)
    fine = G.fbm(h, w, octaves=3, base=64, seed=seed + 1, aspect=width / length)
    if not wet:
        return np.clip(0.62 + 0.22 * n + 0.08 * fine, 0.4, 0.95).astype(np.float32)
    rough = 0.13 + 0.11 * n + 0.05 * fine
    X = (np.arange(w)[None, :] + 0.5) / w * width - width / 2
    Z = (np.arange(h)[:, None] + 0.5) / h * length + z_far
    wobble = 0.35 * (G.fbm(h, w, octaves=4, base=24, seed=seed + 2, aspect=width / length) - 0.5)
    inside = np.zeros((h, w), np.float32)
    for cx, cz, ax, az in puddles:
        d = ((X - cx) / ax) ** 2 + ((Z - cz) / az) ** 2 + wobble
        inside = np.maximum(inside, np.clip((1.0 - d) * 5.0, 0, 1))
    # wheel ruts hold water: slightly smoother bands along the lanes
    for xc in (-2.6, -0.4, 0.4, 2.6):
        rough = rough * (1 - 0.25 * np.exp(-((X - xc) / 0.35) ** 2))
    return (rough * (1 - inside) + 0.05 * inside).astype(np.float32)      # 0.05, not a mirror: see caustic note in night_street


def puddle_list(seed: int, hero) -> list:
    rng = np.random.default_rng(seed + 7)
    return [hero] + [(rng.uniform(-3.9, 3.9), rng.uniform(-62, 6), rng.uniform(0.4, 1.4), rng.uniform(0.6, 2.8))
                     for _ in range(22)]


def skyglow(env: str, h: int = 256, w: int = 512) -> np.ndarray:
    """Equirect night sky, row 0 = zenith: sodium-orange city glow at the horizon, deep blue overhead,
    a faint cloud texture (lit from below) so the strip of sky between the roofs is not flat."""
    el = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    t = np.clip(np.sin(np.maximum(el, 0.0)), 0, 1)[:, None, None]
    horizon = np.array([0.110, 0.062, 0.030])
    mid = np.array([0.030, 0.026, 0.034])
    zenith = np.array([0.004, 0.007, 0.020])
    if env == "wet":                       # low cloud reflects the city: brighter, flatter, warmer
        horizon, mid, zenith = horizon * 1.4, mid * 2.0, np.array([0.016, 0.014, 0.018])
    a = smooth(t / 0.25)
    b = smooth((t - 0.25) / 0.75)
    img = horizon * (1 - a) + mid * a
    img = img * (1 - b) + zenith * b
    clouds = G.fbm(h, w, octaves=5, base=4, seed=17, aspect=2.0)[..., None]
    img = img * (0.75 + 0.5 * clouds * (1.4 if env == "wet" else 0.8))
    img = np.where((el < 0)[:, None, None], horizon, img)
    return np.broadcast_to(img, (h, w, 3)).astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Signs, plates and other small printed textures (row 0 = top; flipped when saved)
# ---------------------------------------------------------------------------

def _font():
    from scenes._desk_text import GLYPHS

    return GLYPHS


def draw_text(draw: ImageDraw.ImageDraw, text: str, x: float, y_base: float, xh: float, width: float, fill, track=0.15):
    """Stroke-font text in pixel space (y down), using the desk scene's glyph strokes."""
    glyphs = _font()
    for ch in text:
        if ch == " " or ch not in glyphs:
            x += xh * 0.6
            continue
        adv, strokes = glyphs[ch]
        for poly in strokes:
            pts = [(x + gx * xh * adv * 0.9, y_base - gy * xh) for gx, gy in poly]
            draw.line(pts, fill=fill, width=max(1, int(round(width))), joint="curve")
        x += adv * xh * (1 + track)
    return x


def text_width(text: str, xh: float, track=0.15) -> float:
    glyphs = _font()
    return sum((glyphs[c][0] * xh * (1 + track)) if c in glyphs else xh * 0.6 for c in text)


PLATE_CELLS = 8


def plate_atlas(seed: int = 4) -> np.ndarray:
    """PLATE_CELLS EU-style plates stacked vertically (each 520 x 112 px): white, blue band, black characters."""
    rng = np.random.default_rng(seed)
    W, H = 520, 112
    img = Image.new("RGB", (W, H * PLATE_CELLS), (236, 236, 228))
    d = ImageDraw.Draw(img)
    letters, digits = "ABCDEHLMNPRST", "0123456789"
    for k in range(PLATE_CELLS):
        y0 = k * H
        d.rectangle([0, y0, W - 1, y0 + H - 1], outline=(20, 20, 20), width=5)
        d.rectangle([5, y0 + 5, 50, y0 + H - 6], fill=(20, 50, 150))
        txt = "".join(rng.choice(list(letters), 2)) + " " + "".join(rng.choice(list(digits), 3)) + " " + "".join(rng.choice(list(letters), 2))
        draw_text(d, txt, 70, y0 + 88, 46, 9, (15, 15, 15), track=0.25)
    return np.asarray(img, np.float32) / 255.0


def road_sign(kind: str, size: int = 256) -> np.ndarray:
    """Upright RGB images of a few European road signs."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size
    if kind == "no_parking":
        d.ellipse([4, 4, s - 4, s - 4], fill=(200, 25, 30))
        d.ellipse([s * 0.14, s * 0.14, s * 0.86, s * 0.86], fill=(20, 60, 160))
        d.line([(s * 0.22, s * 0.22), (s * 0.78, s * 0.78)], fill=(200, 25, 30), width=int(s * 0.1))
    elif kind == "speed":
        d.ellipse([4, 4, s - 4, s - 4], fill=(200, 25, 30))
        d.ellipse([s * 0.12, s * 0.12, s * 0.88, s * 0.88], fill=(240, 240, 235))
        draw_text(d, "30", s * 0.22, s * 0.72, s * 0.38, s * 0.07, (10, 10, 10), track=0.1)
    elif kind == "one_way":
        d.rectangle([0, 0, s, s], fill=(20, 60, 160))
        d.rectangle([4, 4, s - 5, s - 5], outline=(240, 240, 240), width=6)
        d.rectangle([s * 0.2, s * 0.44, s * 0.66, s * 0.56], fill=(240, 240, 240))
        d.polygon([(s * 0.62, s * 0.30), (s * 0.84, s * 0.50), (s * 0.62, s * 0.70)], fill=(240, 240, 240))
    elif kind == "crossing":
        d.rectangle([0, 0, s, s], fill=(20, 60, 160))
        d.polygon([(s * 0.5, s * 0.1), (s * 0.92, s * 0.88), (s * 0.08, s * 0.88)], fill=(240, 240, 240))
        d.ellipse([s * 0.44, s * 0.36, s * 0.56, s * 0.48], fill=(10, 10, 10))
        d.line([(s * 0.5, s * 0.48), (s * 0.46, s * 0.7), (s * 0.38, s * 0.82)], fill=(10, 10, 10), width=int(s * 0.04))
        d.line([(s * 0.46, s * 0.7), (s * 0.58, s * 0.82)], fill=(10, 10, 10), width=int(s * 0.04))
        for k in range(4):
            d.rectangle([s * (0.2 + 0.16 * k), s * 0.84, s * (0.28 + 0.16 * k), s * 0.88], fill=(10, 10, 10))
    return np.asarray(img, np.float32) / 255.0


def street_name(text: str, w: int = 1024, h: int = 220) -> np.ndarray:
    img = Image.new("RGB", (w, h), (238, 238, 232))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, w - 9, h - 9], outline=(20, 20, 20), width=8)
    xh = 70
    x0 = (w - text_width(text, xh, 0.2)) / 2
    draw_text(d, text, x0, h * 0.7, xh, 11, (15, 15, 15), track=0.2)
    return np.asarray(img, np.float32) / 255.0


def manhole(size: int = 512, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Cast-iron cover: rim, chequer pattern, rust. Returns (albedo, height)."""
    rng = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:size, 0:size] + 0.5) / size * 2 - 1
    r = np.hypot(xx, yy)
    chk = ((np.floor((xx + 1) * 18) + np.floor((yy + 1) * 18)) % 2).astype(np.float32)
    ring = (np.abs(r - 0.93) < 0.04).astype(np.float32)
    rust = G.fbm(size, size, octaves=5, base=6, seed=seed)
    v = 0.06 + 0.05 * chk + 0.05 * ring
    alb = np.stack([v + 0.05 * rust, v + 0.025 * rust, v], -1)
    height = np.clip(0.4 + 0.4 * chk + 0.5 * ring - 0.3 * (np.abs(r - 0.98) < 0.02), 0, 1)
    return np.clip(alb, 0, 1).astype(np.float32), height.astype(np.float32)


def bark(h: int = 512, w: int = 256, seed: int = 81) -> np.ndarray:
    v = np.linspace(0, 1, h)[:, None]
    n = G.fbm(h, w, octaves=5, base=8, seed=seed, aspect=0.3)
    fissure = np.abs(np.sin(TAU * (np.linspace(0, 1, w)[None, :] * 9 + 0.6 * G.fbm(h, w, 4, 3, seed=seed + 1, aspect=0.3))))
    rgb = np.array([0.22, 0.18, 0.14]) * (0.55 + 0.7 * n[..., None]) * (0.55 + 0.45 * fissure[..., None] ** 0.4)
    return np.clip(rgb + 0.0 * v[..., None], 0, 1).astype(np.float32)


LEAF_COLOURS = [(0.07, 0.20, 0.05), (0.09, 0.25, 0.06), (0.06, 0.17, 0.05), (0.12, 0.27, 0.06), (0.05, 0.14, 0.06),
                (0.16, 0.24, 0.05)]
LEAF_WEIGHTS = np.array([0.24, 0.22, 0.16, 0.14, 0.14, 0.10])


def leaf_atlas(h: int = 256, cell_w: int = 64) -> np.ndarray:
    """Variants side by side; v base to tip, u across. Midrib, veins, darker edge."""
    k_n = len(LEAF_COLOURS)
    out = np.zeros((h, cell_w * k_n, 3), np.float32)
    v = np.linspace(0, 1, h)[:, None]
    u = np.linspace(-1, 1, cell_w)[None, :]
    for k, base in enumerate(LEAF_COLOURS):
        noise = G.fbm(h, cell_w, octaves=4, base=6, seed=300 + k)
        col = np.array(base, np.float32)
        rib = np.exp(-(u / 0.06) ** 2)
        lateral = (0.5 + 0.5 * np.cos(TAU * (v * 8.0 - np.abs(u) * 2.0))) ** 6 * (1 - np.abs(u))
        img = col * (0.8 + 0.4 * noise[..., None]) * (1 - 0.3 * (np.abs(u) ** 3)[..., None])
        img = img + (col * 0.9 + 0.03) * (0.5 * rib + 0.2 * lateral)[..., None]
        out[:, k * cell_w:(k + 1) * cell_w] = img
    return np.clip(out, 0, 1)


def awning_stripes(colours, stripes: int = 8, h: int = 256, w: int = 512, seed: int = 0) -> np.ndarray:
    x = np.arange(w)[None, :]
    k = (x * stripes // w) % len(colours)
    img = np.asarray(colours, np.float32)[k[0]][None].repeat(h, 0)
    n = G.fbm(h, w, octaves=4, base=8, seed=seed)
    return np.clip(img * (0.85 + 0.3 * n[..., None]), 0, 1)


# ---------------------------------------------------------------------------
# Cars. Local frame: length along z, FRONT toward +z, y up, ground at y = 0.
# ---------------------------------------------------------------------------

CAR_SPECS = {
    # top: side silhouette (z from rear 0 to front 1, y); belt: shoulder height; side_glass: z range of the side
    # windows; bpillars: B/C pillar centres between side windows
    "sedan": dict(L=4.70, W=1.82, belt=0.97, bottom=0.21, tyre=0.32, wheels=(0.16, 0.80), bpillars=(0.535,),
                  side_glass=(0.27, 0.755),
                  top=[(0.0, 0.93), (0.03, 0.98), (0.12, 1.00), (0.24, 1.03), (0.37, 1.38), (0.45, 1.45),
                       (0.62, 1.45), (0.765, 1.02), (0.94, 0.86), (0.975, 0.81), (1.0, 0.79)]),
    "hatch": dict(L=4.25, W=1.78, belt=0.99, bottom=0.20, tyre=0.32, wheels=(0.15, 0.81), bpillars=(0.46,),
                  side_glass=(0.10, 0.735),
                  top=[(0.0, 1.00), (0.012, 1.12), (0.05, 1.37), (0.13, 1.47), (0.55, 1.48), (0.735, 1.04),
                       (0.93, 0.87), (0.975, 0.82), (1.0, 0.80)]),
    "suv": dict(L=4.75, W=1.92, belt=1.12, bottom=0.30, tyre=0.37, wheels=(0.16, 0.81), bpillars=(0.45, 0.16),
                side_glass=(0.07, 0.765),
                top=[(0.0, 1.08), (0.008, 1.38), (0.045, 1.67), (0.11, 1.72), (0.62, 1.72), (0.775, 1.18),
                     (0.94, 1.04), (0.98, 0.99), (1.0, 0.96)]),
    "van": dict(L=5.10, W=2.00, belt=1.10, bottom=0.27, tyre=0.35, wheels=(0.15, 0.83), bpillars=(),
                side_glass=(0.70, 0.855),
                top=[(0.0, 1.96), (0.02, 2.04), (0.78, 2.05), (0.86, 1.30), (0.95, 1.08), (0.985, 1.02), (1.0, 0.99)]),
}
_RB, _RS, _RR, _DECK = 0.06, 0.07, 0.07, 0.07


class CarShape:
    """Analytic body description so lights, mirrors and seams can be placed on the surface."""

    def __init__(self, kind: str):
        s = CAR_SPECS[kind]
        self.kind, self.s = kind, s
        self.L, self.W = s["L"], s["W"]
        zn = np.linspace(0, 1, 801)
        top = np.interp(zn, *np.array(s["top"]).T)
        k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
        k /= k.sum()
        top = np.convolve(np.pad(top, 12, mode="edge"), k, mode="valid")
        self._zn, self._top = zn, top
        roof = top.max()
        on_roof = zn[top > roof - 0.03]
        self.roof0, self.roof1 = float(on_roof.min()), float(on_roof.max())
        self.tyre = s["tyre"]
        self.wheel_zn = s["wheels"]

    def top(self, zn):
        return np.interp(zn, self._zn, self._top)

    def hw(self, zn):
        t = np.abs(2 * np.asarray(zn) - 1)
        return 0.5 * self.W * (1 - 0.10 * t ** 10 - 0.08 * smooth((t - 0.965) / 0.035))

    def shoulder(self, zn):
        return np.minimum(self.top(zn), self.s["belt"] + 0.03 * (0.5 - np.asarray(zn)))

    def bottom(self, zn):
        zn = np.asarray(zn, float)
        z = (zn - 0.5) * self.L
        y = self.s["bottom"] + 0.10 * smooth((np.abs(2 * zn - 1) - 0.86) / 0.14)
        ra = self.tyre + 0.05
        for wz in self.wheel_zn:
            dz = z - (wz - 0.5) * self.L
            arch = np.where(np.abs(dz) < ra, self.tyre + np.sqrt(np.maximum(ra * ra - dz * dz, 0.0)), 0.0)
            y = np.maximum(y, arch)
        return y

    def gh(self, zn):
        return np.maximum(self.top(zn) - self.shoulder(zn), 0.0)


def _half_section(c: CarShape, zn: float):
    """Right half of a cross-section from bottom centre to top centre, plus a label per segment."""
    hw, yb, ys, gh = float(c.hw(zn)), float(c.bottom(zn)), float(c.shoulder(zn)), float(c.gh(zn))
    yb = min(yb, ys - _RB - _RS - 0.02)
    pts, lab = [(0.0, yb)], []

    def add(p, l):
        pts.append(p)
        lab.append(l)

    for x in np.linspace(0, hw - _RB, 3)[1:]:
        add((x, yb), "lower")
    for a in np.linspace(-90, 0, 5)[1:]:
        add((hw - _RB + _RB * math.cos(math.radians(a)), yb + _RB + _RB * math.sin(math.radians(a))), "lower")
    for y in np.linspace(yb + _RB, ys - _RS, 4)[1:]:
        add((hw, y), "lower")
    for a in np.linspace(0, 90, 5)[1:]:
        add((hw - _RS + _RS * math.cos(math.radians(a)), ys - _RS + _RS * math.sin(math.radians(a))), "lower")
    gb = hw - _RS - _DECK
    add((gb, ys), "lower")
    rr = min(_RR, gh / 2)
    gt = gb - 0.42 * gh
    yr = ys + gh
    for t in np.linspace(0, 1, 5)[1:]:
        add((gb + (gt - gb) * t, ys + (yr - rr - ys) * t), "gside")
    for a in np.linspace(0, 90, 5)[1:]:
        add((gt - rr + rr * math.cos(math.radians(a)), yr - rr + rr * math.sin(math.radians(a))), "gcorner")
    crown = 0.025 if gh > 0.1 else 0.015
    for t in np.linspace(0, 1, 4)[1:]:
        add(((gt - rr) * (1 - t), yr + crown * math.sin(t * math.pi / 2)), "gtop")
    return pts, lab


def car(kind: str) -> dict[str, Mesh]:
    """Body (paint), glass, black trim, tyres, alloy rims, brake discs, mirrors, seams, handles, lamp housings."""
    c = CarShape(kind)
    s = c.s
    zs = np.unique(np.concatenate([np.linspace(0, 1, 72), np.linspace(0, 0.03, 6), np.linspace(0.97, 1, 6),
                                   *[np.linspace(w - 0.1, w + 0.1, 16) for w in c.wheel_zn]]))
    rings, labels = [], None
    for zn in zs:
        half, lab = _half_section(c, zn)
        right = np.array(half)
        left = right[-2:0:-1] * np.array([-1.0, 1.0])
        ring = np.vstack([right, left, right[:1]])
        rings.append(np.column_stack([ring[:, 0], ring[:, 1], np.full(len(ring), (zn - 0.5) * c.L)]))
        if labels is None:
            labels = lab + lab[::-1]
    P = np.array(rings)                                             # sections x ring x 3
    rows, cols = P.shape[:2]
    flat = P.reshape(-1, 3)
    f = G._grid_faces(rows, cols)
    # outward winding
    n0 = G.vertex_normals(flat, f)
    out = flat - P.mean(axis=1, keepdims=True).repeat(cols, 1).reshape(-1, 3)
    if (n0 * out).sum() < 0:
        f = f[:, ::-1]
    nrm = G.vertex_normals(flat, f)
    seam = np.arange(rows) * cols
    nrm[seam] = nrm[seam + cols - 1] = G._normalize(nrm[seam] + nrm[seam + cols - 1])
    uv = np.column_stack([flat[:, 2], flat[:, 1]])
    # per-face material from (segment label, section)
    n_quads = len(f) // 2                       # _grid_faces lists every quad's first triangle, then every second one
    q = np.arange(len(f)) % n_quads
    r_idx, c_idx = q // (cols - 1), q % (cols - 1)
    zmid = 0.5 * (zs[r_idx] + zs[np.minimum(r_idx + 1, rows - 1)])
    seg = np.array(labels)[c_idx]
    gh = c.gh(zmid)
    side_glass = (seg == "gside") & (zmid > s["side_glass"][0]) & (zmid < s["side_glass"][1]) & (gh > 0.12)
    for b in s["bpillars"]:
        side_glass &= np.abs(zmid - b) > 0.018
    top_glass = (seg == "gtop") & (gh > 0.06) & ((zmid > c.roof1 + 0.005) | (zmid < c.roof0 - 0.005))
    is_glass = side_glass | top_glass
    body = Mesh(flat, nrm, uv, f[~is_glass])
    glass = Mesh(flat, nrm, uv, f[is_glass])
    for zn_end, sgn in ((0.0, -1.0), (1.0, 1.0)):                    # flat end caps
        ring = P[0 if zn_end == 0 else -1]
        cp = np.vstack([ring.mean(0), ring])
        cf = np.stack([np.zeros(cols - 1, int), np.arange(1, cols), np.arange(2, cols + 1)], 1)
        body += Mesh(cp, np.tile([0, 0, sgn], (len(cp), 1)), cp[:, [0, 1]], cf).oriented()

    parts: dict[str, Acc] = defaultdict(Acc)
    parts["body"].add(body)
    parts["glass"].add(glass)
    ends = {sgn: (float(c.hw(zn)), float(c.bottom(zn)), float(c.shoulder(zn)), (zn - 0.5) * c.L + sgn * 0.004)
            for zn, sgn in ((0.0, -1.0), (1.0, 1.0))}

    def lens(sgn, cx, cy, hx, hy, key, bezel=True):
        _, _, _, z = ends[sgn]
        parts[key].add(rounded_rect(cx, cy, hx, hy, z + sgn * 0.004, sgn))
        if bezel:
            parts["bezel"].add(rounded_rect(cx, cy, hx + 0.018, hy + 0.018, z + sgn * 0.001, sgn))

    def end_rect(sgn, cx, cy, hx, hy, key):
        hw_e, yb_e, ys_e, z = ends[sgn]
        x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
        pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)] if sgn > 0 else [(x1, y0, z), (x0, y0, z), (x0, y1, z), (x1, y1, z)]
        parts[key].add(quad(*pts, uv=[(0, 0), (1, 0), (1, 1), (0, 1)] if sgn > 0 else [(1, 0), (0, 0), (0, 1), (1, 1)]))

    lay = car_layout(c)
    hw_f, yb_f, ys_f, _ = ends[1.0]
    hw_r, yb_r, ys_r, _ = ends[-1.0]
    hx, hy = lay["head_half"]
    tx, ty = lay["tail_half"]
    for (x, y) in lay["head"]:
        lens(1.0, x, y, hx, hy, "head_lens")
    for (x, y) in lay["tail"]:
        lens(-1.0, x, y, tx, ty, "tail_lens")
        lens(-1.0, x, y - ty - 0.045, tx * 0.8, 0.02, "reverse_lens", bezel=False)
    end_rect(1.0, 0.0, lay["grille_y"], lay["grille_hw"], 0.075, "grille")
    end_rect(1.0, 0.0, yb_f + 0.08, hw_f - 0.10, 0.05, "trim")            # lower intake
    end_rect(-1.0, 0.0, yb_r + 0.10, hw_r - 0.12, 0.05, "trim")           # rear diffuser
    for zn_l, key, (x_, y) in ((0.012, "tail_lens", lay["tail"][1]), (0.985, "head_lens", lay["head"][1])):
        z0 = (zn_l - 0.5) * c.L
        dz = 0.10 if zn_l > 0.5 else -0.10
        for sx in (-1, 1):
            x = sx * (float(c.hw(zn_l)) + 0.003)
            pts = [(x, y - 0.05, z0 - dz), (x, y - 0.05, z0 + dz), (x, y + 0.05, z0 + dz), (x, y + 0.05, z0 - dz)]
            m = quad(*pts)
            if m.n[0, 0] * sx < 0:
                m = quad(*pts[::-1])
            parts[key].add(m)

    # wheels: tyre, rim with 5 spokes, brake disc, arch liner
    r = c.tyre
    tw = 0.21
    tyre = G.lathe([(r - 0.10, -tw / 2), (r - 0.025, -tw / 2), (r - 0.004, -tw * 0.40), (r, -tw * 0.2), (r, tw * 0.2),
                    (r - 0.004, tw * 0.40), (r - 0.025, tw / 2), (r - 0.10, tw / 2)], 48)
    barrel = G.lathe([(r - 0.10, tw * 0.42), (r - 0.095, tw * 0.47), (r - 0.10, -tw * 0.3)], 40)
    hub = G.lathe([(0.0, tw * 0.47), (0.045, tw * 0.47), (0.06, tw * 0.40), (0.07, tw * 0.30)], 24)
    disc = G.lathe([(0.0, -0.02), (r - 0.13, -0.02), (r - 0.13, 0.0), (0.0, 0.0)], 32)
    spokes = Mesh()
    for k in range(5):
        a = TAU * k / 5
        p0 = np.array([0.06 * math.cos(a), tw * 0.42, 0.06 * math.sin(a)])
        p1 = np.array([(r - 0.10) * math.cos(a), tw * 0.40, (r - 0.10) * math.sin(a)])
        sp = seg_box(p0, p1, 0.045, 0.025)
        spokes += sp
    ra = c.tyre + 0.05
    for wz in c.wheel_zn:
        zc = (wz - 0.5) * c.L
        for sx in (-1, 1):
            xw = sx * (float(c.hw(wz)) - 0.13)
            rot = G.rotate((0, 0, 1), -90.0 * sx)                    # lathe axis y -> outward x
            m = G.compose(G.translate((xw, r, zc)), rot)
            parts["tyre"].add(tyre, m)
            parts["rim"].add(barrel, m)
            parts["rim"].add(hub, m)
            parts["rim"].add(spokes, m)
            parts["disc"].add(disc, m)
            xl = sx * (float(c.hw(wz)) - 0.30)
            pts = [(xl, 0.08, zc - ra), (xl, 0.08, zc + ra), (xl, r + ra, zc + ra), (xl, r + ra, zc - ra)]
            parts["trim"].add(quad(*(pts if sx > 0 else pts[::-1])))

    # mirrors at the windscreen base, seams, handles
    z_ws = c.roof1 + 0.10 if kind != "van" else 0.84
    z_ws = float(np.clip(z_ws, 0.6, 0.9))
    zm = (z_ws - 0.5) * c.L
    ym = float(c.shoulder(z_ws)) + 0.12
    for sx in (-1, 1):
        xm = sx * (float(c.hw(z_ws)) + 0.13)
        parts["body"].add(G.ellipsoid((0.10, 0.065, 0.06), 10, 16), G.translate((xm, ym, zm)))
        parts["trim"].add(tube([(sx * float(c.hw(z_ws)) - sx * 0.02, ym - 0.04, zm + 0.02), (xm, ym - 0.02, zm)], 0.015, 6))
        parts["mirror"].add(quad(*([(xm - 0.08, ym - 0.045, zm - 0.062), (xm + 0.08, ym - 0.045, zm - 0.062),
                                    (xm + 0.08, ym + 0.045, zm - 0.062), (xm - 0.08, ym + 0.045, zm - 0.062)][::-1])))
    doors = [s["side_glass"][1] - 0.01] + list(s["bpillars"][:1]) + ([s["side_glass"][0] + 0.03] if kind != "van" else [])
    for zn_d in doors:
        z = (zn_d - 0.5) * c.L
        yb, ys = float(c.bottom(zn_d)), float(c.shoulder(zn_d))
        for sx in (-1, 1):
            x = sx * (float(c.hw(zn_d)) + 0.0015)
            parts["seam"].add(box((0.004, ys - yb - 0.16, 0.005), (x, (yb + ys) / 2 + 0.01, z)))
            if zn_d > s["side_glass"][0] + 0.06:
                parts["rim"].add(box((0.02, 0.025, 0.13), (x + sx * 0.008, ys - 0.12, z - 0.16)))
    # sill skirt
    for sx in (-1, 1):
        z0, z1 = (c.wheel_zn[0] + 0.07 - 0.5) * c.L, (c.wheel_zn[1] - 0.07 - 0.5) * c.L
        zn_mid = 0.5 * (c.wheel_zn[0] + c.wheel_zn[1])
        parts["trim"].add(box((0.02, 0.07, z1 - z0), (sx * (float(c.hw(zn_mid)) - 0.005), float(c.bottom(zn_mid)) + 0.03, (z0 + z1) / 2)))
    return {k: v.mesh() for k, v in parts.items() if v.off}


def rounded_rect(cx, cy, hx, hy, z, sgn, n: int = 28, p: float = 5.0) -> Mesh:
    """A flat superellipse ("squircle") facing sgn * z, uv 0..1 across it (mirrored on the rear so text reads)."""
    a = np.linspace(0, TAU, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    ex, ey = np.sign(c) * np.abs(c) ** (2 / p), np.sign(s) * np.abs(s) ** (2 / p)
    pts = np.column_stack([cx + hx * ex, cy + hy * ey, np.full(n, z)])
    P = np.vstack([[cx, cy, z], pts])
    u = (P[:, 0] - cx) / (2 * hx) + 0.5
    uv = np.column_stack([u if sgn > 0 else 1 - u, (P[:, 1] - cy) / (2 * hy) + 0.5])
    f = np.stack([np.zeros(n, int), np.arange(1, n + 1), np.roll(np.arange(1, n + 1), -1)], 1)
    return Mesh(P, np.tile([0.0, 0.0, sgn], (len(P), 1)), uv, f).oriented()


def lamp_texture(kind: str, w: int = 256, h: int = 96) -> np.ndarray:
    """Lens internals seen through clear plastic: reflector bowls and a DRL strip (head), LED bars (tail).
    Row 0 = v = 0 (bottom). Left/right symmetric, so mirroring does not matter."""
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    if kind == "head":
        img = np.full((h, w, 3), 0.42, np.float32) * (0.85 + 0.3 * yy[..., None])
        for cxp in (0.3, 0.7):
            r = np.hypot((xx - cxp) * w / h, yy - 0.45)
            img = np.where((r < 0.30)[..., None], np.array([0.75, 0.76, 0.78]) * (0.6 + 0.6 * r[..., None] / 0.3), img)
            img = np.where((r < 0.13)[..., None], np.array([0.03, 0.03, 0.035]), img)
        img = np.where((np.abs(yy - 0.88) < 0.05)[..., None] & (np.abs(xx - 0.5) < 0.45)[..., None], 0.92, img)
    else:
        img = np.full((h, w, 3), [0.20, 0.01, 0.01], np.float32)
        bars = (np.abs(((xx * 6) % 1.0) - 0.5) < 0.18) & (np.abs(yy - 0.5) < 0.32)
        img = np.where(bars[..., None], np.array([0.55, 0.03, 0.02]), img)
        img = np.where((np.abs(yy - 0.5) > 0.42)[..., None], np.array([0.12, 0.005, 0.005]), img)
    return np.clip(img, 0, 1).astype(np.float32)


def car_layout(c: "CarShape") -> dict:
    """Lamp, grille and plate rectangles on the flat end faces (local frame, x and y centres)."""
    hw_f, yb_f, ys_f = float(c.hw(1.0)), float(c.bottom(1.0)), float(c.shoulder(1.0))
    hw_r, yb_r, ys_r = float(c.hw(0.0)), float(c.bottom(0.0)), float(c.shoulder(0.0))
    return dict(L=c.L, head=[(sx * (hw_f - 0.30), ys_f - 0.15) for sx in (-1, 1)], head_half=(0.20, 0.06),
                tail=[(sx * (hw_r - 0.27), ys_r - 0.14) for sx in (-1, 1)], tail_half=(0.20, 0.065),
                grille_y=ys_f - 0.17, grille_hw=max(hw_f - 0.56, 0.12),
                plate_front=(0.0, yb_f + 0.21), plate_rear=(0.0, 0.5 * (yb_r + ys_r) - 0.04), plate_half=(0.26, 0.056),
                z_front=0.5 * c.L, z_rear=-0.5 * c.L, roof=float(c._top.max()))


def car_anchors(kind: str) -> dict:
    return car_layout(CarShape(kind))


# ---------------------------------------------------------------------------
# Street lamps
# ---------------------------------------------------------------------------

def cobra_post(height: float = 8.0, reach: float = 2.0) -> Mesh:
    """Galvanised pole with a curved outreach arm toward +x; the arm ends at (reach, height + 0.2, 0)."""
    low = np.stack([np.zeros(20), np.linspace(0.0, height - 0.7, 20), np.zeros(20)], 1)
    arm = bezier([(0, height - 0.7, 0), (0, height + 0.25, 0), (reach * 0.3, height + 0.35, 0), (reach, height + 0.2, 0)], 26)
    path = np.vstack([low, arm[1:]])
    radius = np.concatenate([np.linspace(0.085, 0.06, 20), np.linspace(0.055, 0.032, 25)])
    m = G.sweep(path, radius, 16)
    m += G.lathe([(0.0, 0.0), (0.15, 0.0), (0.15, 0.03), (0.11, 0.08), (0.10, 0.6), (0.085, 0.65), (0.0, 0.65)], 24)
    return m


def cobra_head() -> Mesh:
    """Teardrop LED head, long axis +x (arm end at x = 0), flat lens opening downward at y = 0."""
    prof = [(0.0, 0.0), (0.18, 0.0), (0.22, 0.03), (0.22, 0.06), (0.16, 0.10), (0.0, 0.11)]
    shell = G.lathe(prof, 36).transformed(G.compose(G.translate((0.32, 0.0, 0.0)), G.scale((1.7, 1.0, 0.75))))
    neck = tube([(-0.05, 0.08, 0), (0.08, 0.07, 0)], 0.04, 10)
    return join(shell, neck)


def lantern_post(height: float = 4.2) -> Mesh:
    """Heritage cast-iron column: fluted base, slim shaft, collar; the lantern sits on top at `height`."""
    prof = [(0.0, 0.0), (0.20, 0.0), (0.20, 0.08), (0.15, 0.14), (0.15, 0.55), (0.11, 0.62), (0.085, 0.75),
            (0.065, 1.2), (0.055, height - 0.35), (0.08, height - 0.28), (0.08, height - 0.22),
            (0.05, height - 0.18), (0.05, height - 0.05), (0.10, height), (0.0, height)]
    m = G.lathe(prof, 20)
    for a in (0.0, 90.0):                                                  # ladder bar under the lantern
        m += box((0.55, 0.03, 0.03), (0.0, height - 0.45, 0.0)).transformed(G.rotate((0, 1, 0), a))
    return m


LANTERN_W, LANTERN_H = 0.36, 0.50


def lantern_frame() -> Mesh:
    """Square lantern from y = 0 to LANTERN_H + roof: corner posts, top and bottom rings, pyramid roof, finial."""
    hw, h = LANTERN_W / 2, LANTERN_H
    m = Mesh()
    for sx in (-1, 1):
        for sz in (-1, 1):
            m += box((0.03, h, 0.03), (sx * hw, h / 2, sz * hw))
    for y in (0.0, h):
        for a in (0.0, 90.0):
            for s in (-1, 1):
                m += box((LANTERN_W + 0.05, 0.035, 0.03), (0.0, y, s * hw)).transformed(G.rotate((0, 1, 0), a))
    m += G.lathe([(0.0, h), (hw + 0.07, h), (hw + 0.07, h + 0.03), (0.03, h + 0.24), (0.0, h + 0.25)], 4).transformed(G.rotate((0, 1, 0), 45))
    m += G.lathe([(0.0, h + 0.24), (0.025, h + 0.25), (0.025, h + 0.32), (0.04, h + 0.36), (0.0, h + 0.40)], 10)
    m += G.lathe([(0.0, -0.08), (0.11, -0.08), (0.12, -0.02), (0.06, 0.0), (0.0, 0.0)], 4).transformed(G.rotate((0, 1, 0), 45))
    return m


def lantern_panels() -> Mesh:
    """Four frosted panels (emitters), normals outward."""
    hw, h = LANTERN_W / 2 - 0.012, LANTERN_H
    m = Mesh()
    for a in (0.0, 90.0, 180.0, 270.0):
        q = quad((-hw, 0.02, hw), (hw, 0.02, hw), (hw, h - 0.02, hw), (-hw, h - 0.02, hw))
        m += q.transformed(G.rotate((0, 1, 0), a))
    return m


# ---------------------------------------------------------------------------
# Street furniture
# ---------------------------------------------------------------------------

def bollard() -> dict[str, Mesh]:
    body = G.lathe([(0.0, 0.0), (0.075, 0.0), (0.075, 0.82), (0.07, 0.86), (0.06, 0.9), (0.0, 0.93)], 24)
    band = G.lathe([(0.077, 0.70), (0.077, 0.76)], 24)
    return {"body": body, "band": band}


def litter_bin() -> dict[str, Mesh]:
    body = G.lathe([(0.0, 0.02), (0.20, 0.02), (0.22, 0.82), (0.225, 0.86), (0.0, 0.86)], 28)
    lid = G.lathe([(0.0, 0.86), (0.235, 0.86), (0.235, 0.9), (0.14, 0.95), (0.0, 0.96)], 28)
    hood = box((0.18, 0.08, 0.04), (0.0, 0.78, 0.225))
    return {"body": body, "lid": join(lid, hood)}


def pole(height: float, radius: float = 0.038) -> Mesh:
    return tube([(0, 0, 0), (0, height, 0)], radius, 12)


def sign_plate(w: float, h: float, round_: bool) -> dict[str, Mesh]:
    """Plate facing +z centred on the origin; uv 0..1 (v up). Back is grey metal."""
    if round_:
        n = 40
        a = np.linspace(0, TAU, n, endpoint=False)
        pts = np.stack([w / 2 * np.cos(a), h / 2 * np.sin(a), np.zeros(n)], 1)
        cp = np.vstack([[0, 0, 0.004], pts + [0, 0, 0.004]])
        f = np.stack([np.zeros(n, int), np.arange(1, n + 1), np.roll(np.arange(1, n + 1), -1)], 1)
        face = Mesh(cp, np.tile([0, 0, 1.0], (n + 1, 1)), cp[:, :2] / [w, h] + 0.5, f).oriented()
        back = Mesh(cp * [1, 1, -1], np.tile([0, 0, -1.0], (n + 1, 1)), cp[:, :2] * 0, f).oriented()
    else:
        face = quad((-w / 2, -h / 2, 0.004), (w / 2, -h / 2, 0.004), (w / 2, h / 2, 0.004), (-w / 2, h / 2, 0.004),
                    uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
        back = quad((w / 2, -h / 2, -0.004), (-w / 2, -h / 2, -0.004), (-w / 2, h / 2, -0.004), (w / 2, h / 2, -0.004))
    return {"face": face, "back": back}


def parking_meter() -> Mesh:
    m = tube([(0, 0, 0), (0, 1.05, 0)], 0.04, 10)
    m += box((0.22, 0.34, 0.16), (0, 1.22, 0))
    m += G.lathe([(0.0, 1.39), (0.12, 1.39), (0.10, 1.45), (0.0, 1.47)], 16)
    return m


def bike_rack(n: int = 3, gap: float = 0.9) -> Mesh:
    m = Mesh()
    for k in range(n):
        z = k * gap
        path = bezier([(0, 0, z - 0.35), (0, 0.95, z - 0.40), (0, 0.95, z + 0.40), (0, 0, z + 0.35)], 24)
        m += tube(path, 0.024, 10)
    return m


def bicycle(seed: int = 0) -> dict[str, Mesh]:
    """Diamond-frame bike in the y-z plane (front toward +z), wheels touching y = 0."""
    rng = np.random.default_rng(seed)
    R = 0.34
    rear, front = np.array([0, R, -0.52]), np.array([0, R, 0.52])
    bb = np.array([0, 0.30, -0.02])
    seat_top = np.array([0, 0.86, -0.22])
    head_top = np.array([0, 0.84, 0.40])
    head_low = np.array([0, 0.66, 0.44])
    frame_m = Mesh()
    for a, b in ((rear, bb), (rear, seat_top + [0, -0.08, 0.02]), (bb, seat_top), (bb, head_low), (seat_top + [0, -0.05, 0.01], head_top + [0, -0.03, 0]),
                 (head_low, head_top), (head_low, front), (head_top, head_top + [0, 0.12, -0.04]), (seat_top, seat_top + [0, 0.10, -0.03])):
        frame_m += tube([a, b], 0.017, 8)
    bar = head_top + [0, 0.12, -0.04]
    frame_m += tube([bar + [-0.28, 0, 0], bar + [0.28, 0, 0]], 0.012, 8)
    tyres, spokes = Mesh(), Mesh()
    a = np.linspace(0, TAU, 49)
    for c in (rear, front):
        ring = np.stack([np.zeros_like(a), c[1] + R * np.cos(a), c[2] + R * np.sin(a)], 1)
        tyres += tube(ring, 0.017, 8, caps=False)
        rim = np.stack([np.zeros_like(a), c[1] + (R - 0.03) * np.cos(a), c[2] + (R - 0.03) * np.sin(a)], 1)
        spokes += tube(rim, 0.008, 6, caps=False)
        for k in range(28):
            t = TAU * k / 28
            side = 0.02 if k % 2 else -0.02
            spokes += tube([c + [side, 0, 0], c + [0, (R - 0.03) * math.cos(t), (R - 0.03) * math.sin(t)]], 0.0012, 3, caps=False)
        spokes += tube([c + [-0.04, 0, 0], c + [0.04, 0, 0]], 0.02, 10)
    saddle = G.ellipsoid((0.07, 0.03, 0.13), 8, 14).transformed(G.translate(seat_top + [0, 0.12, -0.05]))
    crank = join(tube([bb + [0.06, 0, 0], bb + [0.06, -0.12, 0.08]], 0.012, 6), tube([bb + [-0.06, 0, 0], bb + [-0.06, 0.12, -0.08]], 0.012, 6))
    ring = np.stack([np.full_like(a, 0.05), bb[1] + 0.10 * np.cos(a), bb[2] + 0.10 * np.sin(a)], 1)
    crank += tube(ring, 0.006, 5, caps=False)
    spokes += crank
    return {"frame": frame_m, "tyre": tyres, "metal": spokes, "saddle": saddle}


def traffic_signal_head() -> dict[str, Mesh]:
    """Three-aspect head facing +z, centred at the origin; hoods over each aspect; backplate."""
    body = box((0.30, 0.92, 0.26))
    plate = box((0.52, 1.10, 0.02), (0.0, 0.0, -0.14))
    hoods = Mesh()
    for y in (0.29, 0.0, -0.29):
        a = np.linspace(0, math.pi, 12)
        for t0, t1 in zip(a[:-1], a[1:]):
            p = lambda t, z: (0.115 * math.cos(t), y + 0.115 * math.sin(t), z)
            hoods += quad(p(t1, 0.13), p(t0, 0.13), p(t0, 0.30), p(t1, 0.30))
            hoods += quad(p(t0, 0.13), p(t1, 0.13), p(t1, 0.30), p(t0, 0.30))
    return {"body": join(body, plate), "hood": hoods}


def tree(seed: int, height: float = 7.5) -> dict[str, Mesh]:
    """Street tree (lime/plane-like): trunk, limbs, twigs, leaves clustered at twig ends; uv of leaves picks an
    atlas variant. Base at the origin."""
    rng = np.random.default_rng(seed)
    grp = Groups()
    trunk_top = np.array([rng.normal(0, 0.08), height * 0.42, rng.normal(0, 0.08)])
    trunk = bezier([(0, 0, 0), (0.02, height * 0.2, -0.02), trunk_top], 16)
    grp["bark"].add(tube(trunk, np.linspace(0.20, 0.13, 16), 14))
    k_n = len(LEAF_COLOURS)
    proto = G.leaf(0.16, 0.11, curl=0.25, fold=0.25, rows=4, cols=3)
    tips = []
    for k in range(7):
        a = TAU * k / 7 + rng.uniform(-0.35, 0.35)
        up = rng.uniform(0.55, 0.95)
        reach = rng.uniform(1.4, 2.4)
        end = trunk_top + np.array([math.cos(a) * reach, height * 0.42 * up, math.sin(a) * reach])
        mid = trunk_top + (end - trunk_top) * 0.45 + [0, 0.5, 0]
        limb = bezier([trunk_top, mid, end], 12)
        grp["bark"].add(tube(limb, np.linspace(0.10, 0.035, 12), 9))
        for j in range(5):
            s = limb[int(rng.integers(5, 12))]
            d = unit(np.array([math.cos(a), 0.6, math.sin(a)]) + rng.normal(0, 0.5, 3))
            twig_end = s + d * rng.uniform(0.6, 1.1)
            grp["bark"].add(tube([s, (s + twig_end) / 2 + rng.normal(0, 0.08, 3), twig_end], 0.016, 5))
            tips.append(twig_end)
        tips.append(end)
    for t in tips:
        for _ in range(int(rng.integers(120, 170))):
            p = t + rng.normal(0, 0.42, 3) * [1.0, 0.75, 1.0]
            m = proto.transformed(frame(p, unit(rng.normal(size=3) + [0, 0.4, 0])))
            v = int(rng.choice(k_n, p=LEAF_WEIGHTS / LEAF_WEIGHTS.sum()))
            m.uv = np.stack([(v + 0.05 + 0.9 * m.uv[:, 0]) / k_n, 0.02 + 0.96 * m.uv[:, 1]], 1)
            grp["leaf"].add(m)
    return grp.meshes()


def tree_grate(size: float = 1.2) -> Mesh:
    """Cast-iron tree grate on the pavement surface (y = 0), square frame with radial slots."""
    m = Mesh()
    h = size / 2
    for a in (0.0, 90.0, 180.0, 270.0):
        m += box((size, 0.02, 0.06), (0.0, 0.01, h - 0.03)).transformed(G.rotate((0, 1, 0), a))
    for k in range(36):
        t = TAU * k / 36
        p0 = np.array([0.25 * math.cos(t), 0.012, 0.25 * math.sin(t)])
        r1 = min(h / max(abs(math.cos(t)), abs(math.sin(t))), h) - 0.05
        p1 = np.array([r1 * math.cos(t), 0.012, r1 * math.sin(t)])
        m += seg_box(p0, p1, 0.025, 0.02)
    a = np.linspace(0, TAU, 37)
    m += tube(np.stack([0.24 * np.cos(a), np.full_like(a, 0.012), 0.24 * np.sin(a)], 1), 0.015, 6, caps=False)
    return m


# ---------------------------------------------------------------------------
# Ground decals (flat ribbons a few mm above the surface)
# ---------------------------------------------------------------------------

def worn_marking(x0, x1, z0, z1, y, rng, cell=0.015, wear=0.06) -> Mesh:
    """Painted rectangle with clustered wear (tyre tracks, scuffs) and slightly ragged edges. Kept cells
    are merged into one quad per run along each row, so fine cells stay cheap."""
    nx, nz = max(1, int(round((x1 - x0) / cell))), max(1, int(round((z1 - z0) / cell)))
    xs, zs = np.linspace(x0, x1, nx + 1), np.linspace(z0, z1, nz + 1)
    seed = int(rng.integers(1 << 30))
    if nx > 2 and nz > 2:
        big = G.fbm(nz, nx, octaves=4, base=max(2, int(4 * (z1 - z0))), seed=seed, aspect=(x1 - x0) / (z1 - z0))
        keep = big > np.quantile(big, wear)                           # `wear` = fraction of paint gone, in clusters
    else:
        keep = np.ones((nz, nx), bool)
    keep &= rng.random((nz, nx)) > 0.004
    for j in (0, nx - 1):                                            # slightly ragged long edges
        keep[:, j] &= rng.random(nz) > 0.08
    P, F = [], []
    for i in range(nz):
        j = 0
        while j < nx:
            if not keep[i, j]:
                j += 1
                continue
            j0 = j
            while j < nx and keep[i, j]:
                j += 1
            b = len(P)
            P += [(xs[j0], y, zs[i]), (xs[j0], y, zs[i + 1]), (xs[j], y, zs[i + 1]), (xs[j], y, zs[i])]
            F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    if not P:
        return Mesh()
    P = np.array(P, float)
    return Mesh(P, np.tile(UP, (len(P), 1)), P[:, [0, 2]], np.array(F)).oriented()


def crack(rng, start, length, y=0.0012) -> Mesh:
    """Random-walk crack as a thin flat ribbon with forks."""
    pts = [np.array(start, float)]
    a = rng.uniform(0, TAU)
    for _ in range(int(length / 0.08)):
        a += rng.normal(0, 0.45)
        pts.append(pts[-1] + 0.08 * np.array([math.cos(a), 0, math.sin(a)]))
    pts = np.array(pts)
    m = Mesh()
    for p0, p1 in zip(pts[:-1], pts[1:]):
        d = unit(p1 - p0)
        side = np.array([-d[2], 0, d[0]]) * rng.uniform(0.003, 0.008)
        q = quad(p0 - side + [0, y, 0], p1 - side + [0, y, 0], p1 + side + [0, y, 0], p0 + side + [0, y, 0])
        if q.n[0, 1] < 0:
            q = quad(p0 + side + [0, y, 0], p1 + side + [0, y, 0], p1 - side + [0, y, 0], p0 - side + [0, y, 0])
        m += q
    return m


def catenary(p0, p1, sag: float, n: int = 40) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)[:, None]
    return p0 + (p1 - p0) * t - np.array([0.0, sag, 0.0]) * (4 * t * (1 - t))
