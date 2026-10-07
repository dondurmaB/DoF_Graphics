"""Procedural textures for the terminal scene. Pure numpy + Pillow, deterministic in their arguments.

Row 0 of every array is the TOP of the image (Mitsuba reads bitmap rows that way, with uv v = 0 at the top).
Emission maps are linear radiance in the scene's units (HDR); albedo and roughness maps are 0..1.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import procedural as G
from scenes._terminal_text import Ink, draw_text, word_width

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


def _up(a: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((size, size), Image.BICUBIC))


def _fill(a: np.ndarray, x0, y0, x1, y1, value) -> None:
    a[int(y0):int(y1), int(x0):int(x1)] = value


# ----------------------------------------------------------------------------
# Floor: polished stone tiles with grout, slate inlay stripes, veining and scuffs
# ----------------------------------------------------------------------------

def floor_textures(seed: int, tiles: int = 10, tile_px: int = 340):
    """Albedo and roughness for a floor that repeats every `tiles` tiles. Tile colours depend on `seed`."""
    rng = np.random.default_rng(seed)
    n = tiles * tile_px
    palette = np.array([[0.66, 0.60, 0.50], [0.70, 0.65, 0.55], [0.61, 0.56, 0.47], [0.73, 0.69, 0.60], [0.58, 0.55, 0.49]], np.float32)
    col = palette[rng.choice(len(palette), size=(tiles, tiles), p=[0.3, 0.3, 0.15, 0.15, 0.1])]
    col *= rng.uniform(0.93, 1.07, (tiles, tiles, 1)).astype(np.float32)
    inlay = (0, 5)
    for r in inlay:
        col[r, :] = np.array([0.28, 0.29, 0.31], np.float32) * rng.uniform(0.9, 1.1, (tiles, 1)).astype(np.float32)
    low = n // 4
    turb = G.fbm(low, low, octaves=5, base=3, seed=seed + 1)
    fine = G.fbm(low, low, octaves=4, base=40, seed=seed + 2)
    u = np.linspace(0, 1, low, dtype=np.float32)[None, :]
    v = np.linspace(0, 1, low, dtype=np.float32)[:, None]
    vein = np.abs(np.sin(TAU * (u * 3.1 + v * 1.7 + turb * 2.5)))
    vein = (1.0 - np.clip(vein * 3.5, 0, 1) ** 0.4) * 0.18
    detail = _up(0.55 * turb + 0.45 * fine, n)
    veins = _up(vein, n)
    scuff = _up(G.fbm(low, low, octaves=3, base=12, seed=seed + 3), n)
    alb = np.empty((n, n, 3), np.float32)
    rough = np.empty((n, n), np.float32)
    for r in range(tiles):
        rows = slice(r * tile_px, (r + 1) * tile_px)
        band = np.repeat(col[r], tile_px, axis=0).reshape(tiles * tile_px, 3)[None].repeat(tile_px, 0)
        d, vn, sc = detail[rows], veins[rows], scuff[rows]
        a = band * (0.94 + 0.12 * d[..., None]) * (1.0 - vn[..., None] * (0.4 if r not in inlay else 0.1))
        rr = 0.085 + 0.06 * d + np.where(sc > 0.74, 0.22 * (sc - 0.74) / 0.26 + 0.05, 0.0)
        ys = np.arange(tile_px)[:, None]
        xs = np.arange(n)[None, :] % tile_px
        grout = (ys < 2) | (ys >= tile_px - 1) | (xs < 2) | (xs >= tile_px - 1)
        a = np.where(grout[..., None], np.array([0.30, 0.29, 0.27], np.float32), a)
        rr = np.where(grout, 0.65, rr)
        alb[rows], rough[rows] = a, rr
    alb += rng.normal(0, 0.006, alb.shape).astype(np.float32)
    return np.clip(alb, 0, 1), np.clip(rough, 0.02, 1)


# ----------------------------------------------------------------------------
# Shopfronts: one atlas of shops, each 6 m x 4.4 m (glazed front below, lit sign above)
# ----------------------------------------------------------------------------

SHOP_W, SHOP_H = 6.0, 4.4
SHOP_NAMES = ["CAFE LUNA", "BOOKS", "GIFTS", "PHARMACY", "NEWS", "DUTY FREE", "BAKERY", "TECH", "FLOWERS", "SUSHI",
              "CHOCOLATE", "TRAVEL", "WATCHES", "PIZZA", "MARKET", "FASHION"]
SIGN_BG = [(0.9, 0.2, 0.15), (0.1, 0.45, 0.85), (0.95, 0.75, 0.1), (0.1, 0.65, 0.4), (0.55, 0.2, 0.7), (0.9, 0.9, 0.9),
           (0.95, 0.45, 0.1), (0.1, 0.1, 0.12)]


def shop_atlas(seed: int, count: int = 6, ppm: int = 80):
    """(albedo, emission, names): `count` shops side by side, picked and styled by `seed`."""
    rng = np.random.default_rng(seed)
    W, H = int(SHOP_W * ppm), int(SHOP_H * ppm)
    names = [SHOP_NAMES[i] for i in rng.permutation(len(SHOP_NAMES))[:count]]
    alb = np.zeros((H, W * count, 3), np.float32)
    emi = np.zeros_like(alb)
    ink_all = Ink(W * count, H, ["t"])
    for k, name in enumerate(names):
        x0 = k * W
        stone = np.array([0.50, 0.47, 0.42], np.float32) * rng.uniform(0.85, 1.1)
        alb[:, x0:x0 + W] = stone
        # glazed front: 0.15 m to 3.0 m above the floor, inset 0.45 m from each side
        gy0, gy1 = H - 3.0 * ppm, H - 0.15 * ppm
        gx0, gx1 = x0 + 0.45 * ppm, x0 + W - 0.45 * ppm
        warm = rng.random() < 0.65
        light = np.array([1.0, 0.80, 0.52] if warm else [0.80, 0.90, 1.0], np.float32) * rng.uniform(0.9, 1.4)
        gh, gw = int(gy1 - gy0), int(gx1 - gx0)
        grad = np.linspace(1.0, 0.25, gh, dtype=np.float32)[:, None, None]
        glow = light * (0.08 + 0.22 * grad) * np.ones((gh, gw, 1), np.float32)         # dim back wall, brighter high up
        glow[: max(3, gh // 18)] = light * 1.8                                           # ceiling strip light
        glow[-max(6, gh // 12):] = light * 0.28                                          # lit floor
        n_shelf = int(rng.integers(3, 6))
        for s in range(1, n_shelf + 1):
            yy = int(gh * s / (n_shelf + 1))
            glow[yy:yy + 3] = light * 0.9                                                # shelf edge
            x = 2
            while x < gw - 6:                                                            # products standing on the shelf
                pw, ph = int(rng.integers(5, 18)), int(rng.integers(8, 26))
                c = np.array([rng.uniform(0.05, 1), rng.uniform(0.05, 1), rng.uniform(0.05, 1)], np.float32)
                c = c / max(c.max(), 1e-3) * rng.uniform(0.6, 2.4)
                glow[max(0, yy - ph):yy, x:x + pw] = c
                x += pw + int(rng.integers(1, 8))
        door = slice(int(gw * 0.42), int(gw * 0.58))
        glow[:, door] = light * 0.55
        emi[int(gy0):int(gy0) + gh, int(gx0):int(gx0) + gw] = glow
        alb[int(gy0):int(gy1), int(gx0):int(gx1)] = 0.03
        for mx in np.arange(1.7, 5.1, 1.7):                             # mullions: dark, no emission
            xx = int(x0 + (mx + 0.0) * ppm)
            if not (door.start + gx0 - 6 < xx < door.stop + gx0 + 6):
                _fill(alb, xx - 2, gy0, xx + 2, gy1, 0.05)
                _fill(emi, xx - 2, gy0, xx + 2, gy1, 0.0)
        _fill(alb, gx0, gy0 + 0.6 * ppm, gx1, gy0 + 0.6 * ppm + 4, 0.05)
        _fill(emi, gx0, gy0 + 0.6 * ppm, gx1, gy0 + 0.6 * ppm + 4, 0.0)
        # lit sign, 3.25 m to 4.25 m
        sy0, sy1 = H - 4.25 * ppm, H - 3.25 * ppm
        sx0, sx1 = x0 + 0.3 * ppm, x0 + W - 0.3 * ppm
        bg = np.array(SIGN_BG[int(rng.integers(len(SIGN_BG)))], np.float32)
        _fill(alb, sx0, sy0, sx1, sy1, bg * 0.5)
        _fill(emi, sx0, sy0, sx1, sy1, bg * 2.2)
        xh = 0.34 * ppm / 1.5 * 1.15
        tw = word_width(name, xh)
        draw_text(ink_all, "t", (sx0 + sx1) / 2 - tw / 2, (sy0 + sy1) / 2 + 0.75 * xh, name, xh, max(2.0, xh * 0.09))
    cov = ink_all.coverage("t")[..., None]
    sign = np.zeros(cov.shape[:2], bool)
    for k in range(count):
        sign[int(H - 4.25 * ppm):int(H - 3.25 * ppm), k * W:(k + 1) * W] = True
    emi = np.where(sign[..., None], emi * (1 - cov) + 6.5 * cov, emi)
    alb = np.where(sign[..., None], alb * (1 - cov) + 0.9 * cov, alb)
    return np.clip(alb, 0, 1), emi, names


# ----------------------------------------------------------------------------
# Departure board, hanging signs, clock, posters
# ----------------------------------------------------------------------------

CITIES = ["AMSTERDAM", "BERLIN", "TOKYO", "DUBAI", "MADRID", "ZURICH", "VIENNA", "LISBON", "OSLO", "SEOUL", "ROME", "PRAGUE",
          "ATHENS", "CAIRO", "DELHI", "MILAN", "LONDON", "PARIS", "BOSTON", "SYDNEY", "SINGAPORE", "ISTANBUL", "DOHA", "MUNICH"]
STATUS = [("ON TIME", (1.0, 0.95, 0.8)), ("BOARDING", (0.3, 1.0, 0.45)), ("DELAYED", (1.0, 0.55, 0.15)),
          ("GATE OPEN", (0.3, 1.0, 0.45)), ("FINAL CALL", (1.0, 0.25, 0.2)), ("CANCELLED", (1.0, 0.25, 0.2))]


def board(seed: int, w_px: int = 2400, h_px: int = 1020, rows: int = 12, level: float = 3.2):
    """Departure board: (albedo, emission). Amber text on near-black, status colours, header and column titles."""
    rng = np.random.default_rng(seed)
    emi = np.zeros((h_px, w_px, 3), np.float32)
    alb = np.full((h_px, w_px, 3), 0.015, np.float32)
    amber = np.array([1.0, 0.72, 0.18], np.float32)
    ink = {n: Ink(w_px, h_px, [n]) for n in ("amber", "white")}
    pad = int(0.025 * w_px)
    head_h = int(0.16 * h_px)
    draw_text(ink["white"], "white", pad, head_h * 0.72, "DEPARTURES", head_h * 0.45, head_h * 0.06)
    cols = {"TIME": 0.03, "DESTINATION": 0.19, "FLIGHT": 0.55, "GATE": 0.73, "STATUS": 0.84}
    col_y = head_h + int(0.05 * h_px)
    for name, fx in cols.items():
        draw_text(ink["white"], "white", fx * w_px, col_y, name, 0.022 * h_px, 2.0)
    top = col_y + int(0.03 * h_px)
    pitch = (h_px - top - int(0.02 * h_px)) / rows
    xh = pitch * 0.40
    minute = int(rng.integers(5 * 60, 20 * 60))
    status_ink = {}
    for r in range(rows):
        y = top + (r + 0.78) * pitch
        minute += int(rng.integers(3, 14))
        t = f"{(minute // 60) % 24:02d}:{minute % 60:02d}"
        city = CITIES[int(rng.integers(len(CITIES)))]
        flight = f"{rng.choice(list('ABCDEFGHJKLMNPRSTUVWXYZ'))}{rng.choice(list('ABCDEFGHJKLMNPRSTUVWXYZ'))} {int(rng.integers(100, 9999))}"
        gate = f"{rng.choice(list('ABCDEF'))}{int(rng.integers(1, 40))}"
        st, stc = STATUS[int(rng.choice(len(STATUS), p=[0.45, 0.15, 0.2, 0.1, 0.07, 0.03]))]
        for text, fx in ((t, cols["TIME"]), (city, cols["DESTINATION"]), (flight, cols["FLIGHT"]), (gate, cols["GATE"])):
            draw_text(ink["amber"], "amber", fx * w_px, y, text, xh, max(2.0, xh * 0.10))
        if st not in status_ink:
            status_ink[st] = (stc, Ink(w_px, h_px, ["s"]))
        draw_text(status_ink[st][1], "s", cols["STATUS"] * w_px, y, st, xh, max(2.0, xh * 0.10))
        if r % 2 == 1:
            emi[int(top + r * pitch):int(top + (r + 1) * pitch)] += np.array([0.02, 0.014, 0.004], np.float32)
    emi += level * ink["amber"].coverage("amber")[..., None] * amber
    emi += level * 1.25 * ink["white"].coverage("white")[..., None] * np.array([1.0, 1.0, 1.0], np.float32)
    for st, (stc, ik) in status_ink.items():
        emi += level * ik.coverage("s")[..., None] * np.array(stc, np.float32)
    emi[:head_h - 6] += np.array([0.02, 0.03, 0.06], np.float32)
    return alb, emi


def sign(text: str, bg, w_px: int = 900, h_px: int = 270, level: float = 5.0, arrow: str = ""):
    """A back-lit directional sign: white text (and optional arrow glyph) on a coloured panel. (albedo, emission)."""
    bg = np.array(bg, np.float32)
    ink = Ink(w_px, h_px, ["t"])
    xh = h_px * 0.42 / 1.5
    full = text + ("  " + arrow if arrow else "")
    tw = word_width(full, xh)
    draw_text(ink, "t", (w_px - tw) / 2, h_px * 0.5 + 0.75 * xh, full, xh, max(2.5, xh * 0.14))
    cov = ink.coverage("t")[..., None]
    edge = np.zeros((h_px, w_px), bool)
    b = max(3, h_px // 36)
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = True, True, True, True
    emi = bg * 0.9 * (1 - cov) + level * cov
    emi = np.where(edge[..., None], 0.04, emi)
    alb = np.where(edge[..., None], 0.03, bg * 0.4 * (1 - cov) + 0.9 * cov)
    return alb.astype(np.float32), emi.astype(np.float32)


def clock_face(size: int = 512, level: float = 2.6):
    """Round station clock on a dark square (the scene hides the corners behind a bezel). (albedo, emission)."""
    ink = Ink(size, size, ["m"])
    c = size / 2
    r = size * 0.46
    for i in range(60):
        a = i / 60 * TAU
        ln = 0.10 if i % 5 == 0 else 0.04
        ink.line("m", [(c + np.sin(a) * r, c - np.cos(a) * r), (c + np.sin(a) * r * (1 - ln), c - np.cos(a) * r * (1 - ln))],
                 size * (0.012 if i % 5 == 0 else 0.005))
    for h, (px, py) in {"12": (0, -0.72), "3": (0.72, 0), "6": (0, 0.72), "9": (-0.72, 0)}.items():
        xh = size * 0.075
        draw_text(ink, "m", c + px * r - word_width(h, xh) / 2, c + py * r + 0.75 * xh, h, xh, size * 0.012)
    for ang, ln, wd in ((-60.0, 0.5, 0.025), (60.0, 0.78, 0.016)):                      # 10:10
        a = np.radians(ang)
        ink.line("m", [(c, c), (c + np.sin(a) * r * ln, c - np.cos(a) * r * ln)], size * wd)
    cov = ink.coverage("m")
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(xx - c, yy - c)
    face = (rr < r * 1.04).astype(np.float32)
    emi = (face * (1 - cov) * level)[..., None] * np.array([1.0, 0.98, 0.92], np.float32)
    alb = (face * (1 - cov) * 0.8 + 0.03)[..., None] * np.ones(3, np.float32)
    return alb.astype(np.float32), emi.astype(np.float32)


def poster(seed: int, slogan: str, h_px: int = 600, w_px: int = 400, level: float = 3.0):
    """A lit advertising panel: abstract colour field plus a slogan. (albedo, emission)."""
    rng = np.random.default_rng(seed)
    pal = [rng.uniform(0.05, 0.9, 3) for _ in range(4)]
    art = G.abstract_art(h_px, w_px, pal, seed=seed + 5).astype(np.float32)
    ink = Ink(w_px, h_px, ["t"])
    xh = w_px * 0.10
    y = h_px * 0.86
    for line in slogan.split("|"):
        draw_text(ink, "t", (w_px - word_width(line, xh)) / 2, y, line, xh, max(2.0, xh * 0.12))
        y += xh * 1.9
    cov = ink.coverage("t")[..., None]
    img = art * (1 - 0.5 * cov) + cov
    return (img * 0.9).astype(np.float32), (img * level).astype(np.float32)


def window_grid(seed: int, size: int = 1024, cells: int = 4):
    """A 12 m x 12 m patch of office-tower wall: tinted cladding and a grid of dark glazing. (albedo)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.38, 0.62)
    wall = np.array([base * rng.uniform(0.94, 1.04), base * rng.uniform(0.96, 1.03), base * rng.uniform(0.95, 1.08)], np.float32)
    img = np.tile(wall, (size, size, 1))
    c = size // cells
    for i in range(cells):
        for j in range(cells):
            x0, y0 = i * c + int(c * 0.14), j * c + int(c * 0.18)
            x1, y1 = (i + 1) * c - int(c * 0.14), (j + 1) * c - int(c * 0.18)
            g = np.array([0.05, 0.08, 0.11], np.float32) * rng.uniform(0.7, 1.8)
            img[y0:y1, x0:x1] = g * np.linspace(1.0, 1.5, y1 - y0, dtype=np.float32)[:, None, None]
    return np.clip(img * (0.95 + 0.1 * G.fbm(size, size, octaves=3, base=30, seed=seed)[..., None]), 0, 1)
