"""Procedural textures for the workshop scene. Pure numpy + Pillow (no OpenCV), deterministic per seed.

Convention: arrays are float32 RGB in [0, 1] (or HxW for roughness), row 0 = the TOP of the image, which
is how Mitsuba's bitmap lookup reads v = 0 for the `rectangle` shape (v = 1 at its +y edge is flipped by
the caller with np.flipud where noted).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

import procedural as G

fbm = G.fbm


class Mask:
    """A float coverage mask drawn with Pillow; shapes accumulate, `a` is the array."""

    def __init__(self, h: int, w: int):
        self.img = Image.new("F", (w, h), 0.0)
        self.d = ImageDraw.Draw(self.img)

    @property
    def a(self) -> np.ndarray:
        return np.asarray(self.img, dtype=np.float32)

    def line(self, p0, p1, width, v=1.0):
        self.d.line([tuple(map(float, p0)), tuple(map(float, p1))], fill=float(v), width=max(1, int(width)))

    def polyline(self, pts, width, v=1.0):
        self.d.line([tuple(map(float, p)) for p in pts], fill=float(v), width=max(1, int(width)), joint="curve")

    def circle(self, c, r, v=1.0, outline: int = 0):
        box = [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
        if outline:
            self.d.ellipse(box, outline=float(v), width=int(outline))
        else:
            self.d.ellipse(box, fill=float(v))

    def ellipse(self, c, rx, ry, angle_deg, v=1.0):
        a = np.radians(angle_deg)
        s = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        x, y = rx * np.cos(s), ry * np.sin(s)
        pts = np.stack([c[0] + x * np.cos(a) - y * np.sin(a), c[1] + x * np.sin(a) + y * np.cos(a)], 1)
        self.d.polygon([tuple(p) for p in pts.tolist()], fill=float(v))

    def rect(self, p0, p1, v=1.0, outline: int = 0):
        box = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
        if outline:
            self.d.rectangle(box, outline=float(v), width=int(outline))
        else:
            self.d.rectangle(box, fill=float(v))


def paint(rgb: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    m = np.clip(mask, 0, 1)[..., None]
    return rgb * (1 - m) + np.asarray(colour, np.float32) * m


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-like blur: three box passes per axis (cumulative sums), edge-clamped."""
    if sigma <= 0.3:
        return a
    r = max(1, int(round(np.sqrt(12 * sigma * sigma / 3 + 1) / 2)))
    out = a.astype(np.float32)
    for axis in (0, 1):
        for _ in range(3):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
            n = out.shape[axis]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
            lo = np.take(c, np.arange(0, n), axis=axis)
            out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def _stack(r, g, b):
    return np.stack([r, g, b], -1).astype(np.float32)


def _mix(a, b, t):
    return a + (b - a) * t[..., None]


def concrete(h: int, w: int, width_m: float, height_m: float, seed: int, stains: bool = True):
    """Floor slab. Returns (albedo, roughness). Saw-cut joints every 2.67 m, oil stains and tyre marks by seed."""
    rng = np.random.default_rng(seed)
    n1 = fbm(h, w, 6, 6, 0.55, seed=seed)
    n2 = fbm(h, w, 4, 40, 0.5, seed=seed + 1)
    base = 0.30 + 0.10 * n1 + 0.04 * (n2 - 0.5)
    alb = np.repeat(base[..., None], 3, -1) * np.array([1.0, 0.99, 0.96], np.float32)
    rough = 0.62 + 0.25 * n1 - 0.10 * n2
    ppm_x, ppm_y = w / width_m, h / height_m
    # Saw-cut control joints in a grid.
    joints = Mask(h, w)
    for gx in np.arange(0.0, width_m + 1e-6, 2.67):
        joints.line((gx * ppm_x, 0), (gx * ppm_x, h - 1), 2)
    for gz in np.arange(0.0, height_m + 1e-6, 3.0):
        joints.line((0, gz * ppm_y), (w - 1, gz * ppm_y), 2)
    alb = paint(alb, joints.a, (0.12, 0.12, 0.12))
    if stains:
        mask, sheen = Mask(h, w), Mask(h, w)
        for _ in range(int(rng.integers(7, 11))):                      # oil blotches
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            rx, ry = rng.uniform(0.08, 0.35) * ppm_x, rng.uniform(0.06, 0.25) * ppm_y
            ang = float(rng.uniform(0, 180))
            mask.ellipse((cx, cy), rx, ry, ang)
            sheen.ellipse((cx, cy), rx * 0.6, ry * 0.6, ang)
        for _ in range(int(rng.integers(5, 9))):                       # drips: small dark dots
            mask.circle((rng.uniform(0, w), rng.uniform(0, h)), rng.uniform(0.015, 0.05) * ppm_x)
        for _ in range(int(rng.integers(3, 5))):                       # tyre marks from the door toward the middle
            x0 = rng.uniform(0.2, 0.8) * w
            pts = [(x0 + 0.04 * w * np.sin(s * 3.0 + x0), s * h * 0.8 + 0.05 * h) for s in np.linspace(0, 1, 30)]
            mask.polyline(pts, max(2, int(0.17 * ppm_x)), 0.7)
        m = blur(mask.a, 0.012 * ppm_x) * (0.5 + 0.5 * fbm(h, w, 5, 20, seed=seed + 5))
        sh = blur(sheen.a, 0.02 * ppm_x) * m
        alb *= (1.0 - 0.62 * np.clip(m, 0, 1))[..., None]
        rough = rough - 0.42 * np.clip(m, 0, 1) - 0.12 * sh
    return np.clip(alb, 0, 1).astype(np.float32), np.clip(rough, 0.12, 1.0).astype(np.float32)


def breeze_block(h: int, w: int, width_m: float, height_m: float, seed: int, tone=(0.78, 0.77, 0.73)):
    """Painted breeze-block wall: 0.4 x 0.2 m blocks, recessed mortar, grime toward the floor."""
    rng = np.random.default_rng(seed)
    n = fbm(h, w, 5, 10, 0.55, seed=seed)
    alb = np.ones((h, w, 3), np.float32) * np.array(tone, np.float32) * (0.9 + 0.1 * n)[..., None]
    ppm_x, ppm_y = w / width_m, h / height_m
    bw, bh = 0.4, 0.2
    rows = int(np.ceil(height_m / bh))
    mortar = Mask(h, w)
    for r in range(rows):
        y0, y1 = int(r * bh * ppm_y), int((r + 1) * bh * ppm_y)
        off = (r % 2) * bw / 2
        alb[y0:y1] *= 0.94 + 0.08 * rng.random()
        mortar.line((0, y0), (w - 1, y0), max(2, int(0.01 * ppm_y)))
        x = -off
        while x < width_m:
            xi = int(x * ppm_x)
            mortar.line((xi, y0), (xi, y1), max(2, int(0.01 * ppm_x)))
            alb[y0:y1, max(xi, 0):int((x + bw) * ppm_x)] *= 0.95 + 0.09 * rng.random()
            x += bw
    alb = paint(alb, mortar.a, (0.55, 0.55, 0.52))
    v = np.linspace(1.0, 0.0, h)[:, None]                 # row 0 = top of the wall
    grime = np.clip(0.6 * (1 - v) ** 3, 0, 0.45) * (0.5 + n)
    alb *= (1.0 - grime)[..., None] * np.array([1.0, 0.98, 0.93], np.float32)
    return np.clip(alb, 0, 1).astype(np.float32)


def pegboard_cell(px: int = 128):
    """One 25 mm cell of perforated hardboard: a 6 mm hole, brown board with grain."""
    n = fbm(px, px, 3, 4, 0.5, seed=3)
    alb = _stack(0.34 + 0.07 * n, 0.22 + 0.05 * n, 0.12 + 0.03 * n)
    yy, xx = np.mgrid[0:px, 0:px]
    d = np.hypot(xx - px / 2 + 0.5, yy - px / 2 + 0.5) / px * 0.025
    hole = d < 0.0032
    rim = (d >= 0.0032) & (d < 0.0042)
    alb[rim] *= 0.7
    alb[hole] = (0.03, 0.025, 0.02)
    return alb


def pegboard_painted(px: int = 128, tone=(0.66, 0.70, 0.66)):
    """Painted pegboard cell (the shop painted it a pale green long ago)."""
    cell = pegboard_cell(px)
    n = fbm(px, px, 3, 4, 0.5, seed=9)
    paint = np.array(tone, np.float32) * (0.9 + 0.1 * n)[..., None]
    hole = cell.sum(-1) < 0.2
    out = np.where(hole[..., None], cell, paint)
    return out.astype(np.float32)


def brushed(h: int, w: int, seed: int, lo: float = 0.25, hi: float = 0.5, vertical: bool = False):
    """Roughness map with fine directional streaks: brushed or machined steel."""
    rng = np.random.default_rng(seed)
    n = rng.random((h, 1) if not vertical else (1, w)).astype(np.float32)
    n = blur(np.repeat(n, w if not vertical else h, 1 if not vertical else 0), 0.6)
    n = (n - n.min()) / max(n.max() - n.min(), 1e-6)
    return (lo + (hi - lo) * n).astype(np.float32)


def painted_steel(h: int, w: int, colour, seed: int, rust: float = 0.4, chips: float = 1.0):
    """Painted sheet steel with chips, scuffs and rust bleeding through. Returns (albedo, roughness)."""
    rng = np.random.default_rng(seed)
    n = fbm(h, w, 6, 6, 0.55, seed=seed)
    n2 = fbm(h, w, 5, 24, 0.55, seed=seed + 1)
    base = np.array(colour, np.float32) * (0.85 + 0.2 * n)[..., None]
    wear = np.clip((n2 - (0.78 - 0.12 * chips)) * 8.0, 0, 1) * np.clip(n * 1.4, 0, 1)
    steel = np.array([0.30, 0.29, 0.28], np.float32)
    rusty = _mix(np.broadcast_to(np.array([0.34, 0.15, 0.06], np.float32), base.shape),
                 np.broadcast_to(np.array([0.50, 0.24, 0.08], np.float32), base.shape), n2)
    under = _mix(np.broadcast_to(steel, base.shape), rusty, np.clip(rust * 1.4 * (0.4 + n), 0, 1))
    alb = _mix(base, under, wear)
    # grime film on everything
    alb *= (0.9 + 0.1 * fbm(h, w, 3, 3, seed=seed + 2))[..., None]
    rough = np.clip(0.38 + 0.4 * wear + 0.1 * n2, 0.2, 1.0)
    return np.clip(alb, 0, 1).astype(np.float32), rough.astype(np.float32)


def galvanised(h: int, w: int, seed: int):
    n = fbm(h, w, 6, 12, 0.6, seed=seed)
    spangle = fbm(h, w, 3, 60, 0.5, seed=seed + 2)
    v = 0.55 + 0.12 * n + 0.08 * spangle
    alb = _stack(v, v * 1.01, v * 1.03)
    grime = np.linspace(0.0, 1.0, h)[:, None]
    alb *= (1.0 - 0.25 * grime * (0.4 + n))[..., None]
    return np.clip(alb, 0, 1), np.clip(0.3 + 0.3 * n, 0.2, 1.0).astype(np.float32)


def cardboard(h: int, w: int, seed: int):
    rng = np.random.default_rng(seed)
    n = fbm(h, w, 5, 8, 0.55, seed=seed)
    base = _stack(0.55 + 0.1 * n, 0.41 + 0.08 * n, 0.26 + 0.06 * n)
    # faint corrugation lines and packing tape
    ys = np.arange(h)[:, None]
    base *= (0.97 + 0.03 * np.sin(ys * 0.9))[..., None]
    t0 = int(h * (0.45 + 0.1 * rng.random()))
    base[t0:t0 + max(2, h // 12)] = _mix(base[t0:t0 + max(2, h // 12)], np.broadcast_to(np.array([0.62, 0.52, 0.34], np.float32),
                                         base[t0:t0 + max(2, h // 12)].shape), np.full(base[t0:t0 + max(2, h // 12)].shape[:2], 0.7))
    # printed label block with lines of print
    lx0, ly0 = int(w * 0.15), int(h * 0.15)
    lab, ink = Mask(h, w), Mask(h, w)
    lab.rect((lx0, ly0), (lx0 + int(w * 0.35), ly0 + int(h * 0.22)))
    for k in range(4):
        y = ly0 + 8 + k * int(h * 0.04)
        ink.line((lx0 + 6, y), (lx0 + int(w * 0.35) - 8 - int(rng.integers(0, 40)), y), 2)
    base = paint(paint(base, lab.a, (0.82, 0.8, 0.74)), ink.a, (0.15, 0.14, 0.13))
    return np.clip(base, 0, 1).astype(np.float32)


def diamond_plate(px: int = 256, cells: int = 8):
    """Checker/tread plate tile: raised lozenges alternating 45 and 135 degrees. Returns (albedo, roughness)."""
    m = Mask(px, px)
    step = px // cells
    for i in range(cells):
        for j in range(cells):
            ang = 45 if (i + j) % 2 == 0 else -45
            m.ellipse(((i + 0.5) * step, (j + 0.5) * step), step * 0.4, step * 0.12, ang)
    img = blur(m.a, 1.0)
    alb = _stack(0.34 + 0.12 * img, 0.35 + 0.12 * img, 0.36 + 0.12 * img)
    return alb, (0.38 - 0.12 * img).astype(np.float32)


def drum_paint(h: int, w: int, colour, seed: int):
    """Oil-drum body: painted, dented-looking mottling, rusty seams (the ring ridges are geometry)."""
    alb, rough = painted_steel(h, w, colour, seed, rust=0.9, chips=1.3)
    return alb, rough


def rubber(h: int, w: int, seed: int):
    n = fbm(h, w, 5, 20, 0.6, seed=seed)
    v = 0.035 + 0.025 * n
    return _stack(v, v, v * 1.02), np.clip(0.7 + 0.2 * n, 0, 1).astype(np.float32)


def wood_worn(h: int, w: int, seed: int):
    """Butcher-block bench top: planks running along u, ring stains, tool scars."""
    rng = np.random.default_rng(seed)
    alb, rough = G.wood_planks(h, w, planks=14, board_len=0.5, seed=seed)
    alb = alb * np.array([0.78, 0.74, 0.70], np.float32)
    n = fbm(h, w, 5, 14, seed=seed + 4)
    alb *= (0.8 + 0.3 * n)[..., None]
    rings, scars = Mask(h, w), Mask(h, w)
    for _ in range(int(rng.integers(4, 9))):                  # dark oil rings and burns
        rings.circle((rng.uniform(0, w), rng.uniform(0, h)), rng.uniform(0.02, 0.06) * w, outline=int(rng.integers(2, 6)))
    for _ in range(int(rng.integers(30, 60))):               # knife and saw scars
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        a = rng.uniform(0, np.pi)
        L = rng.uniform(0.02, 0.1) * w
        scars.line((x0, y0), (x0 + L * np.cos(a), y0 + L * np.sin(a)), 1)
    alb = paint(paint(alb, rings.a, (0.07, 0.05, 0.035)), scars.a, (0.12, 0.08, 0.05))
    return np.clip(alb, 0, 1).astype(np.float32), np.clip(rough * 0.9 + 0.1, 0.2, 1).astype(np.float32)


def brick_wall(h: int, w: int, width_m: float, height_m: float, seed: int, windows: int = 6):
    """Red brick facade across the yard: 0.225 x 0.075 m bricks in running bond, light mortar, dark window openings."""
    rng = np.random.default_rng(seed)
    n = fbm(h, w, 5, 12, 0.55, seed=seed)
    alb = np.ones((h, w, 3), np.float32) * np.array([0.78, 0.76, 0.70], np.float32)          # mortar
    ppm_x, ppm_y = w / width_m, h / height_m
    bw, bh, mt = 0.225, 0.075, 0.01
    for r in range(int(np.ceil(height_m / bh))):
        y0 = int(r * bh * ppm_y)
        y1 = int((r + 1) * bh * ppm_y - mt * ppm_y)
        off = (r % 2) * bw / 2
        x = -off
        while x < width_m:
            x0, x1 = int(max(x, 0) * ppm_x), int(min(x + bw - mt, width_m) * ppm_x)
            if x1 > x0 and y1 > y0:
                c = np.array([0.50, 0.22, 0.15]) * (0.75 + 0.5 * rng.random()) * np.array([1.0, 1.0 + 0.2 * rng.random(), 1.0])
                alb[y0:y1, x0:x1] = c
            x += bw
    alb *= (0.85 + 0.25 * n)[..., None]
    frames = Mask(h, w)
    for k in range(windows):
        wx = (k + 0.5) / windows * width_m
        for row in range(2):
            cx0, cx1 = int((wx - 0.9) * ppm_x), int((wx + 0.9) * ppm_x)
            cy0, cy1 = int((1.2 + row * 2.8) * ppm_y), int((3.0 + row * 2.8) * ppm_y)
            alb[cy0:cy1, cx0:cx1] = (0.07, 0.09, 0.11)
            frames.rect((cx0, cy0), (cx1, cy1), outline=max(2, int(0.08 * ppm_x)))
            frames.line(((cx0 + cx1) // 2, cy0), ((cx0 + cx1) // 2, cy1), max(2, int(0.04 * ppm_x)))
    alb = paint(alb, frames.a, (0.6, 0.6, 0.58))
    return np.clip(alb, 0, 1).astype(np.float32)


def oil_mask(h: int, w: int, width_m: float, height_m: float, seed: int) -> np.ndarray:
    """Where the floor is oily: blotches, drips and tyre marks from the door (row 0) inward. 0..1, soft-edged.

    Used as the blend weight between the scanned concrete and a dark glossy oil film."""
    rng = np.random.default_rng(seed)
    ppm_x = w / width_m
    mask = Mask(h, w)
    for _ in range(int(rng.integers(8, 13))):
        cx, cy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.15, 0.95) * h
        mask.ellipse((cx, cy), rng.uniform(0.08, 0.4) * ppm_x, rng.uniform(0.06, 0.28) * ppm_x, float(rng.uniform(0, 180)))
    for _ in range(int(rng.integers(10, 20))):
        mask.circle((rng.uniform(0, w), rng.uniform(0, h)), rng.uniform(0.01, 0.04) * ppm_x)
    for _ in range(int(rng.integers(3, 5))):
        x0 = rng.uniform(0.3, 0.7) * w
        pts = [(x0 + 0.05 * w * np.sin(s * 2.5 + x0), s * h * 0.7) for s in np.linspace(0, 1, 30)]
        mask.polyline(pts, max(2, int(0.16 * ppm_x)), 0.45)
    m = blur(mask.a, 0.015 * ppm_x) * (0.35 + 0.65 * fbm(h, w, 5, 16, seed=seed + 5))
    return np.clip(m * 1.4, 0, 1).astype(np.float32)
