"""Textures for scenes/courtyard.py: weathered facades, flagstones, roof tiles, leaves, paint, snow.

Pure numpy (+ procedural.fbm). Every array is (h, w, 3) float32 in [0, 1] (sRGB-encoded albedo) with
ROW 0 AT v = 0, which is how Mitsuba's bitmap lookup reads rows. Wall and paving meshes put uv in
meters with v = height (walls) or v = z (paving), so row 0 is the BOTTOM of a facade and the north
edge (z = -9) of the paving. Facade textures are unique per wall (not tiled): the streaks under each
sill and the rising damp line up with that wall's openings.
"""

from __future__ import annotations

import math

import numpy as np

import procedural as G

TAU = 2 * math.pi


def _grid(h: int, w: int, ppm: float, x0: float = 0.0, y0: float = 0.0):
    x = x0 + (np.arange(w, dtype=np.float32) + 0.5)[None, :] / ppm
    y = y0 + (np.arange(h, dtype=np.float32) + 0.5)[:, None] / ppm
    return x, y


def _noise1d(n: int, rng, sigma_px: float) -> np.ndarray:
    """Smoothed 1D noise in [0, 1]: the column profile of drip streaks."""
    k = max(int(sigma_px * 3), 1)
    t = np.arange(-k, k + 1, dtype=np.float32)
    ker = np.exp(-0.5 * (t / max(sigma_px, 1e-3)) ** 2)
    raw = rng.normal(0, 1, n + 2 * k).astype(np.float32)
    out = np.convolve(raw, ker / ker.sum(), mode="valid")[:n]
    return (out - out.min()) / max(out.max() - out.min(), 1e-6)


def _fbm(h, w, cells_h, seed, octaves=5, persistence=0.5):
    """fbm whose first octave has `cells_h` cells over the image height (physical-scale control)."""
    return G.fbm(h, w, octaves=octaves, base=max(int(cells_h), 1), persistence=persistence, seed=seed)


def _mix(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 2:
        t = t[..., None]
    return a * (1 - t) + b * t


# --------------------------------------------------------------------------
# Masonry patterns
# --------------------------------------------------------------------------
def _ashlar(x, y, L, H, rng, course=(0.30, 0.42), block=(0.45, 1.1), joint=0.009,
            stone_a=(0.66, 0.58, 0.45), stone_b=(0.58, 0.56, 0.50), mortar=(0.70, 0.67, 0.60)):
    """Coursed ashlar. Returns (rgb, distance to the nearest joint in m)."""
    h, w = y.shape[0], x.shape[1]
    edges = [0.0]
    while edges[-1] < H + 0.5:
        edges.append(edges[-1] + rng.uniform(*course))
    edges = np.array(edges, np.float32)
    ci = np.searchsorted(edges, y[:, 0], side="right") - 1
    bid = np.zeros((h, w), np.int32)
    dist = np.zeros((h, w), np.float32)
    nb = 0
    for k in range(len(edges) - 1):
        rows = np.nonzero(ci == k)[0]
        if len(rows) == 0:
            continue
        r0, r1 = rows[0], rows[-1] + 1
        xs = [-rng.uniform(0.0, block[1])]
        while xs[-1] < L + 0.5:
            xs.append(xs[-1] + rng.uniform(*block))
        xs = np.array(xs, np.float32)
        bj = np.clip(np.searchsorted(xs, x[0], side="right") - 1, 0, len(xs) - 2)
        dx = np.minimum(x[0] - xs[bj], xs[bj + 1] - x[0])
        dy = np.minimum(y[r0:r1, 0] - edges[k], edges[k + 1] - y[r0:r1, 0])
        dist[r0:r1] = np.minimum(dx[None, :], dy[:, None])
        bid[r0:r1] = nb + bj[None, :]
        nb += len(xs)
    tone = rng.normal(0, 1, nb).astype(np.float32)[bid]
    pick = rng.random(nb).astype(np.float32)[bid]
    base = _mix(np.array(stone_a, np.float32), np.array(stone_b, np.float32), pick)
    rgb = base * (1.0 + 0.07 * tone[..., None])
    rgb = rgb * (1 - 0.16 * np.clip(1 - dist / 0.025, 0, 1))[..., None]          # weathered arrises
    rgb = np.where((dist < joint / 2)[..., None], np.array(mortar, np.float32), rgb)
    return rgb.astype(np.float32), dist


def _bricks(x, y, rng, pitch=(0.235, 0.077), joint=0.011,
            colours=((0.52, 0.22, 0.13), (0.60, 0.30, 0.17), (0.45, 0.19, 0.12), (0.64, 0.42, 0.28)),
            mortar=(0.58, 0.55, 0.48)):
    px, py = pitch
    course = np.floor(y / py).astype(np.int32)
    xo = x + (course % 2) * px / 2
    bi = np.floor(xo / px).astype(np.int32)
    table_t = rng.normal(0, 1, (97, 89)).astype(np.float32)
    table_c = rng.integers(0, len(colours), (97, 89))
    tone = table_t[course % 97, bi % 89]
    cols = np.array(colours, np.float32)[table_c[course % 97, bi % 89]]
    rgb = cols * (1 + 0.1 * tone[..., None])
    lx, ly = xo % px, y % py
    mort = (lx < joint) | (ly < joint)
    return np.where(mort[..., None], np.array(mortar, np.float32), rgb).astype(np.float32)


# --------------------------------------------------------------------------
# Facades (unique per wall)
# --------------------------------------------------------------------------
def facade(kind: str, L: float, H: float, openings, ppm: float, seed: int, wet: bool = False) -> np.ndarray:
    """Weathered facade texture covering the whole wall, `openings` in wall uv (scene_kit-free tuples
    (u0, u1, v0, v1, arch)). kind: ashlar | limestone | stucco | brick."""
    rng = np.random.default_rng(seed)
    h, w = int(round(H * ppm)), int(round(L * ppm))
    x, y = _grid(h, w, ppm)
    big = _fbm(h, w, H / 3.0, seed + 1, octaves=4)
    mid = _fbm(h, w, H / 0.6, seed + 2, octaves=4)
    fine = _fbm(h, w, H / 0.08, seed + 3, octaves=3)

    if kind in ("ashlar", "limestone"):
        if kind == "ashlar":
            rgb, dist = _ashlar(x, y, L, H, rng)
        else:
            rgb, dist = _ashlar(x, y, L, H, rng, course=(0.26, 0.33), block=(0.35, 0.8),
                                stone_a=(0.74, 0.70, 0.60), stone_b=(0.66, 0.64, 0.58), mortar=(0.76, 0.73, 0.66))
        rgb = rgb * (0.86 + 0.22 * mid[..., None]) * (0.93 + 0.12 * fine[..., None])
        pits = (_fbm(h, w, H / 0.03, seed + 4, octaves=2) > 0.82)
        rgb = np.where(pits[..., None], rgb * 0.72, rgb)
    elif kind == "stucco":
        plaster = np.array([0.80, 0.60, 0.38], np.float32)
        rgb = plaster * (0.84 + 0.26 * big[..., None]) * (0.92 + 0.14 * mid[..., None]) * (0.96 + 0.07 * fine[..., None])
        # spalled render: brick shows through, more near the ground and the window corners
        patch_field = _fbm(h, w, H / 1.2, seed + 5, octaves=5)
        near_base = np.exp(-y / 1.1)
        corner = np.zeros((h, w), np.float32)
        for (u0, u1, v0, v1, arch) in openings:
            for cu, cv in ((u0, v0), (u1, v0), (u0, v1), (u1, v1)):
                corner += np.exp(-((x - cu) ** 2 + (y - cv) ** 2) / 0.35)
        score = patch_field + 0.25 * near_base + 0.18 * np.clip(corner, 0, 1)
        holes = score > 0.86
        rim = (score > 0.83) & ~holes
        brick = _bricks(x, y, rng)
        rgb = np.where(holes[..., None], brick * 0.92, rgb)
        rgb = np.where(rim[..., None], rgb * 0.78, rgb)
        cracks = np.exp(-((_fbm(h, w, H / 2.0, seed + 6, octaves=3) - 0.5) / 0.004) ** 2) * (mid > 0.55)
        rgb = rgb * (1 - 0.35 * cracks[..., None])
        dist = np.full((h, w), 1.0, np.float32)
    elif kind == "brick":
        rgb = _bricks(x, y, rng) * (0.88 + 0.22 * mid[..., None]) * (0.95 + 0.08 * fine[..., None])
        wash = _fbm(h, w, H / 1.5, seed + 7, octaves=4)
        lime = np.clip((wash - 0.62) / 0.12, 0, 1) * (y / H)                       # old limewash, upper wall
        rgb = _mix(rgb, np.array([0.80, 0.78, 0.71], np.float32) * (0.9 + 0.1 * fine[..., None]), lime * 0.85)
        dist = np.full((h, w), 1.0, np.float32)
    else:
        raise ValueError(kind)

    # --- weathering -------------------------------------------------------------------------------
    rgb = rgb * (0.90 + 0.14 * big[..., None])
    damp_h = 0.55 + 0.45 * _noise1d(w, rng, 0.6 * ppm)[None, :]
    damp = np.clip(1 - y / damp_h, 0, 1) ** 1.3
    rgb = _mix(rgb, rgb * np.array([0.55, 0.60, 0.50], np.float32), damp * 0.85)  # rising damp, greenish
    splash = np.clip(1 - y / 0.22, 0, 1) * (0.6 + 0.4 * fine)
    rgb = rgb * (1 - 0.30 * splash[..., None])
    moss = np.clip((mid - 0.45) * 3, 0, 1) * np.clip(1 - y / 0.9, 0, 1) * (fine > 0.35)
    rgb = _mix(rgb, np.array([0.20, 0.27, 0.10], np.float32) * (0.7 + 0.6 * fine[..., None]), moss * 0.75)

    streak = np.zeros((h, w), np.float32)
    cols = _noise1d(w, rng, 0.012 * ppm) ** 2.2
    length = 0.5 + 1.6 * _noise1d(w, rng, 0.05 * ppm)
    for (u0, u1, v0, v1, arch) in openings:
        if v0 < 0.3:
            continue
        inside = (x >= u0 - 0.06) & (x <= u1 + 0.06) & (y < v0 - 0.07)
        streak = np.maximum(streak, inside * cols * np.exp(-np.maximum(v0 - 0.07 - y, 0) / length))
    top_streak = cols * np.exp(-np.maximum(H - 0.3 - y, 0) / (0.6 * length)) * (y < H - 0.3)
    streak = np.maximum(streak, 0.8 * top_streak)
    rgb = rgb * (1 - 0.42 * streak[..., None])

    grime = np.zeros((h, w), np.float32)
    for (u0, u1, v0, v1, arch) in openings:                     # dirt around opening edges
        r, c = (u1 - u0) / 2, (u0 + u1) / 2
        top = v1 + (np.sqrt(np.maximum(r * r - (x - c) ** 2, 0)) if arch else 0)
        dx = np.maximum(np.maximum(u0 - x, x - u1), 0)
        dy = np.maximum(np.maximum(v0 - y, y - top), 0)
        d = np.sqrt(dx * dx + dy * dy)
        grime = np.maximum(grime, np.exp(-d / 0.09) * ((x > u0 - 0.4) & (x < u1 + 0.4)))
    rgb = rgb * (1 - 0.18 * grime[..., None] * (0.5 + 0.5 * fine[..., None]))

    if kind != "brick":
        lichen = (fine > 0.74) & (mid > 0.55) & (dist > 0.02)
        rgb = np.where(lichen[..., None], _mix(rgb, np.array([0.78, 0.72, 0.48], np.float32), 0.45), rgb)

    if wet:
        rgb = wet_variant(rgb, streak)
    return np.clip(rgb, 0, 1).astype(np.float32)


def wet_variant(rgb: np.ndarray, streak=None) -> np.ndarray:
    """Rain-soaked: darker (sRGB x0.76, about x0.55 linear), a little more saturated, streaks darker still."""
    lum = rgb.mean(-1, keepdims=True)
    out = (lum + (rgb - lum) * 1.15) * 0.76
    if streak is not None:
        out = out * (1 - 0.2 * streak[..., None])
    return np.clip(out, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Tiling textures (uv in meters, `size` meters per tile)
# --------------------------------------------------------------------------
def stone_tile(size: float = 2.0, ppm: int = 256, seed: int = 11, pale: bool = False) -> np.ndarray:
    n = int(size * ppm)
    x, y = _grid(n, n, ppm)
    rng = np.random.default_rng(seed)
    a, b = ((0.74, 0.71, 0.62), (0.68, 0.66, 0.60)) if pale else ((0.62, 0.57, 0.47), (0.56, 0.55, 0.50))
    rgb, _ = _ashlar(x, y, size, size, rng, course=(0.3, 0.4), block=(0.4, 0.9), stone_a=a, stone_b=b)
    rgb = rgb * (0.88 + 0.2 * _fbm(n, n, size / 0.5, seed + 1, 4)[..., None])
    return np.clip(rgb, 0, 1).astype(np.float32)


def plaster(size: float = 4.0, ppm: int = 128, seed: int = 21, colour=(0.84, 0.81, 0.73)) -> np.ndarray:
    n = int(size * ppm)
    x, y = _grid(n, n, ppm)
    m = _fbm(n, n, size / 1.0, seed, 5)
    f = _fbm(n, n, size / 0.1, seed + 1, 3)
    rgb = np.array(colour, np.float32) * (0.86 + 0.2 * m[..., None]) * (0.95 + 0.08 * f[..., None])
    return np.clip(rgb, 0, 1).astype(np.float32)


def roof_tiles(size=(2.1, 2.16), ppm: int = 240, seed: int = 31, wet: bool = False) -> np.ndarray:
    """Roman barrel tiles: u across the slope (0.21 m per tile), v up the slope (0.36 m per course)."""
    w, h = int(size[0] * ppm), int(size[1] * ppm)
    x, y = _grid(h, w, ppm)
    rng = np.random.default_rng(seed)
    col = np.floor(x / 0.21).astype(np.int32)
    row = np.floor(y / 0.36).astype(np.int32)
    t = rng.normal(0, 1, (8, 12)).astype(np.float32)[row % 8, col % 12]
    pick = rng.random((8, 12)).astype(np.float32)[row % 8, col % 12]
    base = _mix(np.array([0.66, 0.34, 0.20], np.float32), np.array([0.55, 0.42, 0.33], np.float32), pick * 0.7)
    rgb = base * (1 + 0.1 * t[..., None])
    lx, ly = (x % 0.21) / 0.21, (y % 0.36) / 0.36
    rgb = rgb * (0.72 + 0.28 * np.sin(np.pi * lx) ** 0.5)[..., None]            # troughs between barrels
    rgb = rgb * (0.78 + 0.22 * np.clip(ly * 4, 0, 1))[..., None]                 # shadow under the lip
    m = _fbm(h, w, size[1] / 0.6, seed + 1, 4)
    lichen = (m > 0.62) & (_fbm(h, w, size[1] / 0.05, seed + 2, 3) > 0.55)
    rgb = np.where(lichen[..., None], _mix(rgb, np.array([0.70, 0.66, 0.45], np.float32), 0.6), rgb)
    rgb = rgb * (0.88 + 0.2 * m[..., None])
    if wet:
        rgb = wet_variant(rgb)
    return np.clip(rgb, 0, 1).astype(np.float32)


def cobbles(size: float = 2.0, ppm: int = 256, seed: int = 41, wet: bool = False) -> np.ndarray:
    """Granite setts, 0.1 x 0.2 m in rows, dark sandy joints."""
    n = int(size * ppm)
    x, y = _grid(n, n, ppm)
    rng = np.random.default_rng(seed)
    row = np.floor(y / 0.105).astype(np.int32)
    xo = x + (row % 2) * 0.1
    ci = np.floor(xo / 0.2).astype(np.int32)
    t = rng.normal(0, 1, (64, 64)).astype(np.float32)[row % 64, ci % 64]
    pick = rng.random((64, 64)).astype(np.float32)[row % 64, ci % 64]
    base = _mix(np.array([0.45, 0.44, 0.43], np.float32), np.array([0.52, 0.47, 0.42], np.float32), pick)
    lx, ly = (xo % 0.2) / 0.2, (y % 0.105) / 0.105
    dome = np.sin(np.pi * np.clip(lx, 0, 1)) * np.sin(np.pi * np.clip(ly, 0, 1))
    rgb = base * (1 + 0.12 * t[..., None]) * (0.7 + 0.35 * dome[..., None] ** 0.4)
    joint = (lx < 0.06) | (lx > 0.94) | (ly < 0.1) | (ly > 0.9)
    rgb = np.where(joint[..., None], np.array([0.22, 0.20, 0.17], np.float32), rgb)
    rgb = rgb * (0.85 + 0.25 * _fbm(n, n, size / 0.4, seed + 1, 4)[..., None])
    if wet:
        rgb = wet_variant(rgb)
    return np.clip(rgb, 0, 1).astype(np.float32)


def floor_tiles(size: float = 2.4, ppm: int = 200, seed: int = 51) -> np.ndarray:
    """Worn terracotta 0.3 m tiles for the cloister walk."""
    n = int(size * ppm)
    x, y = _grid(n, n, ppm)
    rng = np.random.default_rng(seed)
    i, j = np.floor(x / 0.3).astype(np.int32), np.floor(y / 0.3).astype(np.int32)
    t = rng.normal(0, 1, (8, 8)).astype(np.float32)[j % 8, i % 8]
    rgb = np.array([0.58, 0.30, 0.18], np.float32) * (1 + 0.1 * t[..., None])
    rgb = rgb * (0.82 + 0.25 * _fbm(n, n, size / 0.5, seed + 1, 4)[..., None])
    lx, ly = (x % 0.3) / 0.3, (y % 0.3) / 0.3
    grout = (lx < 0.025) | (lx > 0.975) | (ly < 0.025) | (ly > 0.975)
    rgb = np.where(grout[..., None], np.array([0.42, 0.39, 0.33], np.float32), rgb)
    return np.clip(rgb, 0, 1).astype(np.float32)


def planks(size=(1.4, 2.8), ppm: int = 200, seed: int = 61, tint=(0.36, 0.25, 0.16)) -> np.ndarray:
    """Weathered vertical door planks, 0.14 m wide; u across, v up."""
    w, h = int(size[0] * ppm), int(size[1] * ppm)
    x, y = _grid(h, w, ppm)
    rng = np.random.default_rng(seed)
    k = np.floor(x / 0.14).astype(np.int32)
    t = rng.normal(0, 1, 32).astype(np.float32)[k % 32]
    grain = G.fbm(h, w, octaves=4, base=2, seed=seed + 1, aspect=25.0)       # long streaks along v
    rgb = np.array(tint, np.float32) * (0.78 + 0.45 * grain[..., None]) * (1 + 0.1 * t[..., None])
    gap = (x % 0.14) / 0.14
    rgb = rgb * np.where(((gap < 0.03) | (gap > 0.97))[..., None], 0.35, 1.0)
    grey = _fbm(h, w, size[1] / 0.8, seed + 2, 4)
    rgb = _mix(rgb, rgb.mean(-1, keepdims=True) * np.array([1.05, 1.0, 0.95], np.float32) * 1.3, np.clip(grey - 0.3, 0, 0.6))
    return np.clip(rgb, 0, 1).astype(np.float32)


def paint(colour, size: float = 0.5, ppm: int = 1024, seed: int = 71) -> np.ndarray:
    """Old louvre-shutter paint: brushed, chalky, chipped to grey wood in places."""
    n = int(size * ppm)
    brush = G.fbm(n, n, octaves=4, base=4, seed=seed, aspect=0.08)
    blot = _fbm(n, n, 6, seed + 1, 5)
    rgb = np.array(colour, np.float32) * (0.85 + 0.22 * brush[..., None]) * (0.9 + 0.15 * blot[..., None])
    chalk = np.clip(blot - 0.5, 0, 0.5)
    rgb = _mix(rgb, np.array([0.85, 0.84, 0.80], np.float32), chalk * 0.5)
    chips = _fbm(n, n, 30, seed + 2, 3) > 0.78
    wood = np.array([0.42, 0.38, 0.32], np.float32) * (0.8 + 0.3 * brush[..., None])
    rgb = np.where(chips[..., None], wood, rgb)
    return np.clip(rgb, 0, 1).astype(np.float32)


def terracotta(h: int = 256, w: int = 512, seed: int = 81) -> np.ndarray:
    """Pot texture in lathe uv (u around, v along the profile from the foot): efflorescence near the
    rim, green algae near the foot."""
    v = (np.arange(h, dtype=np.float32) + 0.5)[:, None] / h
    n = G.fbm(h, w, octaves=5, base=4, seed=seed, aspect=4.0)
    f = G.fbm(h, w, octaves=3, base=30, seed=seed + 1, aspect=4.0)
    rgb = np.array([0.66, 0.35, 0.21], np.float32) * (0.82 + 0.3 * n[..., None]) * (0.94 + 0.1 * f[..., None])
    salt = np.clip((v - 0.55) * 2.5, 0, 1) * np.clip(n - 0.35, 0, 1) * 1.5
    rgb = _mix(rgb, np.array([0.82, 0.80, 0.74], np.float32), np.clip(salt, 0, 0.7))
    algae = np.clip(0.3 - v, 0, 0.3) / 0.3 * (f > 0.4)
    rgb = _mix(rgb, np.array([0.25, 0.30, 0.15], np.float32), algae * 0.6)
    return np.clip(rgb, 0, 1).astype(np.float32)


def bark(h: int = 512, w: int = 256, seed: int = 91) -> np.ndarray:
    """London-plane bark in sweep uv (u around, v along): flaking camouflage patches of olive-grey, cream
    and tan with soft edges, plus fine grain."""
    patches = [G.fbm(h, w, octaves=4, base=3, seed=seed + k, aspect=1.0) for k in range(3)]
    cols = [np.array(c, np.float32) for c in ((0.52, 0.50, 0.40), (0.76, 0.73, 0.60), (0.62, 0.55, 0.42),
                                               (0.42, 0.42, 0.34))]
    rgb = np.broadcast_to(cols[0], (h, w, 3)).copy()
    for k, p in enumerate(patches):
        m = np.clip((p - 0.55) / 0.06, 0, 1)
        rgb = _mix(rgb, np.broadcast_to(cols[k + 1], (h, w, 3)), m)
    fine = G.fbm(h, w, octaves=3, base=24, seed=seed + 9, aspect=1.0)
    return np.clip(rgb * (0.88 + 0.2 * fine[..., None]), 0, 1).astype(np.float32)


def snow(size: float = 2.0, ppm: int = 128, seed: int = 101) -> np.ndarray:
    n = int(size * ppm)
    m = _fbm(n, n, size / 0.8, seed, 4)
    rgb = np.array([0.90, 0.92, 0.95], np.float32) * (0.96 + 0.05 * m[..., None])
    return np.clip(rgb, 0, 1).astype(np.float32)


def joints(size: float = 2.0, ppm: int = 128, seed: int = 111) -> np.ndarray:
    """What shows in the gaps between flagstones: dark soil with moss and grit."""
    n = int(size * ppm)
    m = _fbm(n, n, size / 0.3, seed, 5)
    f = _fbm(n, n, size / 0.04, seed + 1, 3)
    soil = np.array([0.16, 0.13, 0.10], np.float32) * (0.7 + 0.6 * f[..., None])
    moss = np.array([0.20, 0.30, 0.09], np.float32) * (0.7 + 0.6 * f[..., None])
    return np.clip(_mix(soil, moss, np.clip((m - 0.35) * 2.5, 0, 1)), 0, 1).astype(np.float32)


def street_facade(L: float = 40.0, H: float = 14.0, ppm: int = 40, seed: int = 121) -> np.ndarray:
    """The building across the street, seen only from the archway looking out: render, windows, a shop."""
    rng = np.random.default_rng(seed)
    h, w = int(H * ppm), int(L * ppm)
    x, y = _grid(h, w, ppm)
    rgb = np.array([0.74, 0.66, 0.52], np.float32) * (0.85 + 0.25 * _fbm(h, w, H / 3, seed, 4)[..., None])
    for k in range(int(L / 3.2)):
        cx = 1.6 + 3.2 * k
        for fy in (4.2, 7.4, 10.6):
            win = (np.abs(x - cx) < 0.55) & (y > fy) & (y < fy + 1.6)
            rgb = np.where(win[..., None], np.array([0.06, 0.07, 0.08], np.float32) * (1 + rng.uniform(-0.3, 0.6)), rgb)
        shop = (np.abs(x - cx) < 1.2) & (y > 0.2) & (y < 2.8)
        if rng.random() < 0.5:
            rgb = np.where(shop[..., None], np.array([0.10, 0.11, 0.12], np.float32), rgb)
    return np.clip(rgb, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Paving (unique, world-mapped: u = (x + 8) / 16, v = (z + 9) / 16)
# --------------------------------------------------------------------------
PAVE_X0, PAVE_Z0, PAVE_SIZE = -8.0, -9.0, 16.0


def _window(x0, x1, z0, z1, ppm, n):
    """Row/column slice of the paving map covering world rect [x0, x1] x [z0, z1]."""
    c0, c1 = int((x0 - PAVE_X0) * ppm), int((x1 - PAVE_X0) * ppm) + 1
    r0, r1 = int((z0 - PAVE_Z0) * ppm), int((z1 - PAVE_Z0) * ppm) + 1
    return max(r0, 0), min(r1, n), max(c0, 0), min(c1, n)


def paving(stones, ppm: int, seed: int, mode: str = "dry", paths=(), puddles=()) -> tuple[np.ndarray, np.ndarray]:
    """Flagstone tops (stone layout from `stones`: list of (x0, x1, z0, z1) in world m). Returns (rgb, rough)
    where rough is the roughness map (grayscale, raw). mode: dry | wet | snow."""
    rng = np.random.default_rng(seed)
    n = int(PAVE_SIZE * ppm)
    x, z = _grid(n, n, ppm, PAVE_X0, PAVE_Z0)
    xx, zz = np.broadcast_to(x, (n, n)), np.broadcast_to(z, (n, n))
    sid = np.full((n, n), -1, np.int32)
    rect = np.zeros((len(stones), 4), np.float32)
    for k, (x0, x1, z0, z1) in enumerate(stones):
        c0, c1 = int((x0 - PAVE_X0) * ppm), int(math.ceil((x1 - PAVE_X0) * ppm))
        r0, r1 = int((z0 - PAVE_Z0) * ppm), int(math.ceil((z1 - PAVE_Z0) * ppm))
        sid[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = k
        rect[k] = (x0, x1, z0, z1)
    ok = sid >= 0
    r = rect[np.maximum(sid, 0)]
    edge = np.minimum(np.minimum(xx - r[..., 0], r[..., 1] - xx), np.minimum(zz - r[..., 2], r[..., 3] - zz))
    edge = np.where(ok, edge, 0.0)
    m = len(stones)
    tone = rng.normal(0, 1, m + 1).astype(np.float32)[sid]
    pick = rng.random(m + 1).astype(np.float32)[sid]
    pal = np.array([[0.60, 0.58, 0.52], [0.66, 0.60, 0.50], [0.54, 0.54, 0.53], [0.62, 0.55, 0.45]], np.float32)
    cidx = rng.integers(0, len(pal), m + 1)[sid]
    base = pal[cidx]
    big = _fbm(n, n, PAVE_SIZE / 2.5, seed + 1, 4)
    mid = _fbm(n, n, PAVE_SIZE / 0.35, seed + 2, 4)
    fine = _fbm(n, n, PAVE_SIZE / 0.05, seed + 3, 3)
    rgb = base * (1 + 0.08 * tone[..., None]) * (0.86 + 0.2 * mid[..., None]) * (0.94 + 0.1 * fine[..., None])
    rgb = rgb * (0.9 + 0.14 * big[..., None])
    wear = np.zeros((n, n), np.float32)
    for poly in paths:                                         # footfall: smoother, paler, cleaner
        p = np.asarray(poly, np.float32)
        for a, b in zip(p[:-1], p[1:]):
            r0, r1, c0, c1 = _window(min(a[0], b[0]) - 2.5, max(a[0], b[0]) + 2.5, min(a[1], b[1]) - 2.5,
                                     max(a[1], b[1]) + 2.5, ppm, n)
            if r1 <= r0 or c1 <= c0:
                continue
            sx, sz = xx[r0:r1, c0:c1], zz[r0:r1, c0:c1]
            ab = b - a
            t = np.clip(((sx - a[0]) * ab[0] + (sz - a[1]) * ab[1]) / max(float(ab @ ab), 1e-6), 0, 1)
            d2 = (sx - a[0] - t * ab[0]) ** 2 + (sz - a[1] - t * ab[1]) ** 2
            wear[r0:r1, c0:c1] = np.maximum(wear[r0:r1, c0:c1], np.exp(-d2 / (2 * 0.7 ** 2)))
    rgb = rgb * (1 + 0.10 * wear[..., None])
    arris = np.clip(1 - edge / 0.03, 0, 1)
    mossy = np.clip((big - 0.35) * 2.0, 0, 1) * (1 - wear)
    rgb = _mix(rgb, rgb * 0.7, arris * 0.6)
    rgb = _mix(rgb, np.array([0.22, 0.30, 0.10], np.float32) * (0.7 + 0.6 * fine[..., None]),
               arris ** 2 * mossy * (fine > 0.3) * 0.9)
    lichen = (fine > 0.8) & (mid > 0.5) & (edge > 0.03) & (wear < 0.5)
    rgb = np.where(lichen[..., None], _mix(rgb, np.array([0.80, 0.78, 0.66], np.float32), 0.5), rgb)
    stains = _fbm(n, n, PAVE_SIZE / 1.2, seed + 4, 4)
    rgb = rgb * (1 - 0.25 * np.clip((stains - 0.7) * 4, 0, 1)[..., None])
    rough = np.full((n, n), 0.78, np.float32) - 0.12 * wear

    if mode == "wet":
        pud = np.zeros((n, n), np.float32)
        for (px, pz, rx, rz) in puddles:
            r0, r1, c0, c1 = _window(px - 1.6 * rx, px + 1.6 * rx, pz - 1.6 * rz, pz + 1.6 * rz, ppm, n)
            if r1 <= r0 or c1 <= c0:
                continue
            d = ((xx[r0:r1, c0:c1] - px) / rx) ** 2 + ((zz[r0:r1, c0:c1] - pz) / rz) ** 2
            pud[r0:r1, c0:c1] = np.maximum(pud[r0:r1, c0:c1],
                                           np.clip((1.0 - d) * 3 + (mid[r0:r1, c0:c1] - 0.5) * 2.2, 0, 1))
        rgb = wet_variant(rgb)
        rgb = _mix(rgb, rgb * 0.72, pud)
        rough = np.where(pud > 0.5, 0.03, 0.30 + 0.12 * fine).astype(np.float32)
        rough = np.where(edge < 0.012, 0.2, rough)
    elif mode == "snow":
        snowc = np.array([0.90, 0.92, 0.95], np.float32) * (0.95 + 0.06 * mid[..., None])
        trod = np.clip(wear * 1.4 - 0.35 + (fine - 0.5) * 0.5, 0, 1)
        slush = _mix(np.array([0.62, 0.62, 0.63], np.float32), wet_variant(rgb) * 0.9, 0.55)
        prints = np.zeros((n, n), np.float32)
        for _ in range(int(220 * len(paths))):
            seg = np.asarray(paths[rng.integers(len(paths))], np.float32)
            k = int(rng.integers(len(seg) - 1))
            p = seg[k] + (seg[k + 1] - seg[k]) * rng.random() + rng.normal(0, 0.45, 2)
            r0, r1, c0, c1 = _window(p[0] - 0.15, p[0] + 0.15, p[1] - 0.15, p[1] + 0.15, ppm, n)
            if r1 <= r0 or c1 <= c0:
                continue
            ang = rng.uniform(0, TAU)
            dx, dz = xx[r0:r1, c0:c1] - p[0], zz[r0:r1, c0:c1] - p[1]
            u = dx * math.cos(ang) + dz * math.sin(ang)
            v = -dx * math.sin(ang) + dz * math.cos(ang)
            prints[r0:r1, c0:c1] = np.maximum(prints[r0:r1, c0:c1], ((u / 0.13) ** 2 + (v / 0.05) ** 2 < 1))
        rgb = _mix(snowc, slush, trod)
        rgb = _mix(rgb, rgb * 0.82, prints * (1 - trod) * 0.8)
        rough = np.full((n, n), 0.85, np.float32) - 0.4 * trod
    rgb = np.where(ok[..., None], rgb, np.array([0.3, 0.3, 0.3], np.float32))
    return np.clip(rgb, 0, 1).astype(np.float32), np.clip(rough, 0.02, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Leaves: atlases of K colour variants side by side; v along the blade (base row 0), u across
# --------------------------------------------------------------------------
# sRGB-encoded albedo: 0.30 sRGB is only 0.07 linear, so leaf greens need these values to read green, not black.
IVY = [(0.16, 0.30, 0.10), (0.13, 0.26, 0.09), (0.20, 0.34, 0.12), (0.14, 0.27, 0.13), (0.22, 0.33, 0.10),
       (0.11, 0.23, 0.09)]
CREEPER = [(0.27, 0.45, 0.13), (0.23, 0.40, 0.12), (0.31, 0.49, 0.15), (0.25, 0.42, 0.16), (0.34, 0.47, 0.13),
           (0.21, 0.37, 0.11)]
CREEPER_WINTER = [(0.42, 0.10, 0.06), (0.50, 0.16, 0.05), (0.33, 0.12, 0.08), (0.45, 0.26, 0.10),
                  (0.30, 0.18, 0.10), (0.38, 0.08, 0.07)]
TREE = [(0.30, 0.46, 0.14), (0.26, 0.42, 0.12), (0.34, 0.50, 0.16), (0.28, 0.44, 0.17), (0.38, 0.50, 0.14),
        (0.24, 0.38, 0.11)]
TREE_WINTER = [(0.45, 0.30, 0.15), (0.52, 0.36, 0.18), (0.38, 0.26, 0.14), (0.58, 0.42, 0.22),
               (0.42, 0.33, 0.20), (0.35, 0.22, 0.12)]
CITRUS = [(0.12, 0.27, 0.09), (0.15, 0.30, 0.10), (0.10, 0.24, 0.08), (0.13, 0.28, 0.11)]
GERANIUM = [(0.26, 0.42, 0.13), (0.23, 0.39, 0.12), (0.29, 0.45, 0.14), (0.21, 0.36, 0.11)]
N_VARIANTS = 6


def leaf_atlas(colours, seed: int, veins: str = "pinnate", h: int = 256, cell_w: int = 128,
               vein_colour=None, zonal: bool = False) -> np.ndarray:
    """veins: 'pinnate' (midrib + laterals), 'palmate' (rays from the base, ivy/creeper/geranium)."""
    k_n = len(colours)
    out = np.zeros((h, cell_w * k_n, 3), np.float32)
    v = (np.arange(h, dtype=np.float32) + 0.5)[:, None] / h
    u = (np.arange(cell_w, dtype=np.float32) + 0.5)[None, :] / cell_w * 2 - 1
    for k, base in enumerate(colours):
        n = G.fbm(h, cell_w, octaves=4, base=5, seed=seed + k)
        f = G.fbm(h, cell_w, octaves=3, base=24, seed=seed + 50 + k)
        col = np.array(base, np.float32)
        img = col * (0.82 + 0.3 * n[..., None]) * (0.94 + 0.1 * f[..., None])
        if veins == "pinnate":
            rib = np.exp(-(u / 0.035) ** 2)
            lat = (0.5 + 0.5 * np.cos(TAU * (v * 8.0 - np.abs(u) * 1.8))) ** 8 * (1 - np.abs(u))
            vm = np.clip(0.7 * rib + 0.3 * lat, 0, 1)
        else:
            bx, by = 0.0, 0.10
            ang = np.arctan2(u - bx, v - by)
            rays = np.zeros_like(ang)
            for a in (0.0, 0.95, -0.95, 1.9, -1.9):
                rays = np.maximum(rays, np.exp(-((ang - a) * np.hypot(u - bx, v - by) / 0.03) ** 2))
            vm = rays * (v > by - 0.02)
        vcol = col * 1.8 + 0.05 if vein_colour is None else np.array(vein_colour, np.float32)
        img = _mix(img, np.broadcast_to(vcol, img.shape), vm * 0.55)
        edge = np.clip(np.abs(u) - 0.6, 0, 0.4) / 0.4
        img = img * (1 - 0.25 * edge[..., None])
        if zonal:                                              # geranium: dark horseshoe band
            r = np.hypot(u, (v - 0.45) * 1.1)
            img = img * (1 - 0.45 * np.exp(-((r - 0.42) / 0.07) ** 2))[..., None]
        out[:, k * cell_w:(k + 1) * cell_w] = img
    return np.clip(out, 0, 1).astype(np.float32)
