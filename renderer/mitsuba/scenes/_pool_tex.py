"""Procedural textures for the pool scene. Pure numpy + Pillow, deterministic in their arguments.

Row 0 of every array is the TOP of the image. Emission maps are linear radiance (HDR); albedo and
roughness maps are 0..1.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import procedural as G
from scenes._pool_text import Ink, draw_text, word_width

TAU = 2 * np.pi


def save_png8(path: Path, arr: np.ndarray) -> None:
    """Atomic 8-bit PNG from a float 0..1 array (RGB or gray)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".png")
    os.close(fd)
    Image.fromarray(a).save(tmp, format="PNG")
    os.replace(tmp, path)


def _up(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC))


# ----------------------------------------------------------------------------
# Ceramic tiles: the pool mosaic, lane lines, the deck
# ----------------------------------------------------------------------------

def mosaic(seed: int, palette, weights, tile_m: float = 0.0625, grout_m: float = 0.0028, ppm: int = 1024,
           grout=(0.70, 0.74, 0.74)):
    """One square metre of glazed mosaic that repeats seamlessly: (albedo, roughness)."""
    rng = np.random.default_rng(seed)
    n = int(round(1.0 / tile_m))
    pal = np.asarray(palette, np.float32)
    idx = rng.choice(len(pal), size=(n, n), p=np.asarray(weights) / np.sum(weights))
    col = pal[idx] * rng.uniform(0.94, 1.06, (n, n, 1)).astype(np.float32)
    px = ppm // n
    alb = np.repeat(np.repeat(col, px, 0), px, 1)
    ys, xs = np.mgrid[0:n * px, 0:n * px] % px
    g = max(1, int(round(grout_m * ppm)))
    is_grout = (ys < g) | (xs < g)
    shade = G.fbm(n * px // 4, n * px // 4, octaves=3, base=8, seed=seed + 1)
    shade = _up(shade, n * px, n * px)
    alb = alb * (0.97 + 0.06 * shade[..., None])
    alb = np.where(is_grout[..., None], np.asarray(grout, np.float32) * (0.9 + 0.15 * shade[..., None]), alb)
    rough = np.where(is_grout, 0.75, 0.05 + 0.04 * shade)
    return np.clip(alb, 0, 1).astype(np.float32), np.clip(rough, 0, 1).astype(np.float32)


POOL_PALETTE = [(0.52, 0.74, 0.82), (0.58, 0.79, 0.86), (0.46, 0.68, 0.79), (0.80, 0.87, 0.88), (0.40, 0.62, 0.75)]
POOL_WEIGHTS = [0.35, 0.25, 0.2, 0.12, 0.08]
LANE_PALETTE = [(0.025, 0.05, 0.15), (0.035, 0.06, 0.17), (0.02, 0.04, 0.12)]
WHITE_PALETTE = [(0.85, 0.87, 0.86), (0.80, 0.83, 0.83), (0.88, 0.89, 0.88)]


def deck(seed: int, width_m: float, depth_m: float, ppm: int = 96, tile_m: float = 0.25, pool_rect=None):
    """The whole hall floor in one map (no repeats, so wet patches never tile): (albedo, roughness).

    Columns run along +x from the hall's min x, rows along +z from its min z (row 0 = min z). Wet patches
    gather near the pool edge (`pool_rect` = x0, x1, z0, z1 in map metres).
    """
    rng = np.random.default_rng(seed)
    W, H = int(width_m * ppm), int(depth_m * ppm)
    nx, nz = int(np.ceil(width_m / tile_m)) + 1, int(np.ceil(depth_m / tile_m)) + 1
    base = np.array([0.60, 0.58, 0.53], np.float32)
    tone = rng.normal(0, 1, (nz, nx)).astype(np.float32)
    x = (np.arange(W) + 0.5) / ppm
    z = (np.arange(H) + 0.5) / ppm
    ix, iz = (x / tile_m).astype(int), (z / tile_m).astype(int)
    alb = base * (1.0 + 0.05 * tone[iz][:, ix])[..., None]
    fx, fz = (x / tile_m) % 1.0, (z / tile_m) % 1.0
    g = 0.0075 / tile_m
    grout = (np.minimum(fx, 1 - fx)[None, :] < g) | (np.minimum(fz, 1 - fz)[:, None] < g)
    speck = G.fbm(H // 2, W // 2, octaves=3, base=60, seed=seed + 1)
    speck = _up(speck, W, H)
    alb = alb * (0.92 + 0.14 * speck[..., None])
    alb = np.where(grout[..., None], np.array([0.50, 0.49, 0.46], np.float32), alb)
    rough = np.where(grout, 0.85, 0.55 + 0.1 * speck)
    # wet patches: low-frequency noise, thresholded, strongest within ~1.5 m of the pool edge
    wet = _up(G.fbm(H // 8, W // 8, octaves=4, base=5, seed=seed + 2), W, H)
    if pool_rect is not None:
        x0, x1, z0, z1 = pool_rect
        dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)[None, :]
        dz = np.maximum(np.maximum(z0 - z, z - z1), 0.0)[:, None]
        near = np.exp(-np.hypot(dx, dz) / 1.4)
    else:
        near = 1.0
    w = np.clip((wet * (0.55 + 0.6 * near) - 0.55) / 0.12, 0, 1)
    alb = alb * (1.0 - 0.28 * w[..., None])
    rough = rough * (1 - w) + 0.06 * w
    return np.clip(alb, 0, 1).astype(np.float32), np.clip(rough, 0.02, 1).astype(np.float32)


def grass(seed: int, size: int = 1024):
    rng = np.random.default_rng(seed)
    a = G.fbm(size, size, octaves=6, base=6, seed=seed)
    b = G.fbm(size, size, octaves=3, base=120, seed=seed + 1)
    base = np.array([0.10, 0.20, 0.05], np.float32)
    dry = np.array([0.22, 0.24, 0.10], np.float32)
    m = np.clip(0.6 * a + 0.4 * b, 0, 1)[..., None]
    return np.clip(base * (1 - m) + dry * m + rng.normal(0, 0.01, (size, size, 1)).astype(np.float32), 0, 1)


def window_grid(seed: int, size: int = 1024, cells: int = 4):
    """A 12 m patch of a far building's wall: cladding and a grid of dark glazing (albedo)."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.45, 0.65)
    wall = np.array([v * rng.uniform(0.95, 1.05), v, v * rng.uniform(0.9, 1.0)], np.float32)
    img = np.tile(wall, (size, size, 1))
    c = size // cells
    for i in range(cells):
        for j in range(cells):
            x0, y0 = i * c + int(c * 0.15), j * c + int(c * 0.2)
            x1, y1 = (i + 1) * c - int(c * 0.15), (j + 1) * c - int(c * 0.2)
            img[y0:y1, x0:x1] = np.array([0.05, 0.07, 0.09], np.float32) * rng.uniform(0.7, 1.6)
    return np.clip(img * (0.95 + 0.1 * G.fbm(size, size, octaves=3, base=30, seed=seed)[..., None]), 0, 1)


# ----------------------------------------------------------------------------
# Lettering: wall signs, deck depth marks, lane numbers, the scoreboard, the pace clock
# ----------------------------------------------------------------------------

def sign(lines, bg, fg, w_px: int = 900, h_px: int = 450, icon: str = "", border=None):
    """A painted wall sign: centred capital lines, optional 'nodive' icon on the left (albedo)."""
    bg, fg = np.asarray(bg, np.float32), np.asarray(fg, np.float32)
    img = np.tile(bg, (h_px, w_px, 1))
    ink = Ink(w_px, h_px, ["t", "red"])
    x_left = 0.0
    if icon == "nodive":
        r = h_px * 0.36
        cx, cy = h_px * 0.48, h_px * 0.5
        ink.ellipse("red", (cx - r, cy - r, cx + r, cy + r), r * 0.16)
        ink.line("red", [(cx - r * 0.7, cy - r * 0.7), (cx + r * 0.7, cy + r * 0.7)], r * 0.16)
        # a diver: an arc for the body and a dot for the head, under the slash
        ink.line("t", [(cx - r * 0.55, cy + r * 0.2), (cx - r * 0.1, cy - r * 0.35), (cx + r * 0.35, cy - r * 0.15),
                       (cx + r * 0.55, cy + r * 0.3)], r * 0.12)
        ink.ellipse("t", (cx - r * 0.42, cy - r * 0.62, cx - r * 0.18, cy - r * 0.38), r * 0.12)
        x_left = cx + r * 1.15
    n = len(lines)
    xh = min(h_px * 0.42 / (1.5 * n) * 1.4, h_px * 0.24)
    widest = max(word_width(line, 1.0) for line in lines)
    xh = min(xh, 0.86 * (w_px - x_left) / widest)
    for i, line in enumerate(lines):
        tw = word_width(line, xh)
        avail = w_px - x_left
        x = x_left + (avail - tw) / 2
        y = h_px * (i + 1) / (n + 1) + 0.75 * xh
        draw_text(ink, "t", x, y, line, xh, max(2.5, xh * 0.16))
    cov, red = ink.coverage("t")[..., None], ink.coverage("red")[..., None]
    img = img * (1 - cov) + fg * cov
    img = img * (1 - red) + np.array([0.75, 0.06, 0.05], np.float32) * red
    b = max(4, h_px // 30)
    if border is not None:
        img[:b], img[-b:], img[:, :b], img[:, -b:] = border, border, border, border
    return np.clip(img, 0, 1).astype(np.float32)


def depth_mark(text: str, w_px: int = 600, h_px: int = 200):
    """A deck-edge depth marking: dark blue lettering on a white tile strip (albedo)."""
    return sign([text], (0.84, 0.86, 0.85), (0.03, 0.08, 0.30), w_px, h_px)


def lane_number(n: int, size: int = 256):
    img = np.full((size, size, 3), 0.03, np.float32)
    ink = Ink(size, size, ["t"])
    xh = size * 0.42
    s = str(n)
    draw_text(ink, "t", (size - word_width(s, xh)) / 2 - xh * 0.05, size * 0.5 + 0.75 * xh, s, xh, xh * 0.2)
    cov = ink.coverage("t")[..., None]
    return np.clip(img * (1 - cov) + 0.9 * cov, 0, 1).astype(np.float32)


NAMES = ["KOVAC", "SATO", "MILLER", "OKAFOR", "DUBOIS", "LINDQVIST", "ROSSI", "NAKAMURA", "PETROV", "SILVA",
         "HANSEN", "MORALES", "CHEN", "WALSH", "BAKER", "IBRAHIM"]


def scoreboard(seed: int, w_px: int = 1800, h_px: int = 600, level: float = 3.0):
    """Results board: (albedo, emission). Amber names and times on black, a white header."""
    rng = np.random.default_rng(seed)
    alb = np.full((h_px, w_px, 3), 0.015, np.float32)
    emi = np.zeros((h_px, w_px, 3), np.float32)
    ink = {k: Ink(w_px, h_px, [k]) for k in ("amber", "white", "green")}
    head = int(0.17 * h_px)
    event = ["100 M FREESTYLE", "50 M BACKSTROKE", "200 M MEDLEY", "100 M BREASTSTROKE", "50 M BUTTERFLY"][int(rng.integers(5))]
    draw_text(ink["white"], "white", 0.03 * w_px, head * 0.72, event, head * 0.36, head * 0.05)
    draw_text(ink["green"], "green", 0.80 * w_px, head * 0.72, "FINAL", head * 0.36, head * 0.05)
    top = head + int(0.03 * h_px)
    rows = 6
    pitch = (h_px - top - int(0.03 * h_px)) / rows
    xh = pitch * 0.42
    base_t = rng.uniform(48.0, 62.0)
    times = np.sort(base_t + rng.exponential(0.9, rows))
    names = [NAMES[i] for i in rng.permutation(len(NAMES))[:rows]]
    for r in range(rows):
        y = top + (r + 0.8) * pitch
        lane = int(rng.integers(1, 7))
        t = times[r]
        tt = f"{int(t // 60)}:{t % 60:05.2f}" if t >= 60 else f"{t:05.2f}"
        for text, fx in ((str(r + 1), 0.03), (str(lane), 0.11), (names[r], 0.19), (tt, 0.72)):
            draw_text(ink["amber"], "amber", fx * w_px, y, text, xh, max(2.0, xh * 0.11))
    emi += level * ink["amber"].coverage("amber")[..., None] * np.array([1.0, 0.70, 0.16], np.float32)
    emi += level * 1.2 * ink["white"].coverage("white")[..., None]
    emi += level * ink["green"].coverage("green")[..., None] * np.array([0.25, 1.0, 0.35], np.float32)
    emi[:head - 4] += np.array([0.015, 0.02, 0.05], np.float32)
    return alb, emi


def pace_clock(size: int = 512):
    """A swimming pace clock: white face, 60 second ticks, big numbers every 5 s, red sweep hand (albedo)."""
    ink = Ink(size, size, ["m", "red"])
    c, r = size / 2, size * 0.46
    for i in range(60):
        a = i / 60 * TAU
        ln = 0.12 if i % 5 == 0 else 0.05
        ink.line("m", [(c + np.sin(a) * r, c - np.cos(a) * r), (c + np.sin(a) * r * (1 - ln), c - np.cos(a) * r * (1 - ln))],
                 size * (0.014 if i % 5 == 0 else 0.006))
    for k in range(12):
        a = k / 12 * TAU
        s = str(60 if k == 0 else k * 5)
        xh = size * 0.055
        px, py = c + np.sin(a) * r * 0.74, c - np.cos(a) * r * 0.74
        draw_text(ink, "m", px - word_width(s, xh) / 2, py + 0.75 * xh, s, xh, size * 0.010)
    ink.line("red", [(c, c), (c + np.sin(1.3) * r * 0.9, c - np.cos(1.3) * r * 0.9)], size * 0.012)
    ink.line("m", [(c, c), (c + np.sin(4.0) * r * 0.55, c - np.cos(4.0) * r * 0.55)], size * 0.02)
    cov, red = ink.coverage("m")[..., None], ink.coverage("red")[..., None]
    yy, xx = np.mgrid[0:size, 0:size]
    face = (np.hypot(xx - c, yy - c) < r * 1.05)[..., None]
    img = np.where(face, 0.86, 0.03) * np.ones(3, np.float32)
    img = img * (1 - cov) + 0.02 * cov
    img = img * (1 - red) + np.array([0.8, 0.05, 0.04], np.float32) * red
    return np.clip(img, 0, 1).astype(np.float32)


def banner(seed: int, text: str, w_px: int = 360, h_px: int = 900):
    """A vertical club banner: a colour field, a stripe and a word down the length (albedo)."""
    rng = np.random.default_rng(seed)
    pal = np.array([[0.05, 0.15, 0.45], [0.55, 0.05, 0.08], [0.02, 0.35, 0.25], [0.85, 0.65, 0.05], [0.25, 0.08, 0.40]], np.float32)
    bg = pal[int(rng.integers(len(pal)))]
    img = np.tile(bg, (h_px, w_px, 1))
    img[int(h_px * 0.62):int(h_px * 0.68)] = 0.85
    ink = Ink(h_px, w_px, ["t"])
    xh = w_px * 0.3
    draw_text(ink, "t", (h_px - word_width(text, xh)) / 2, w_px * 0.5 + 0.75 * xh, text, xh, xh * 0.16)
    cov = np.rot90(ink.coverage("t"), 1)[..., None]
    img = img * (1 - cov) + 0.92 * cov
    return np.clip(img, 0, 1).astype(np.float32)
