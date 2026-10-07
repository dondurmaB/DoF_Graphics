"""Procedural text, paper and screen textures for the desk scene. Pure numpy + Pillow, deterministic.

The scene exists to stress fine, high-frequency detail, so all lettering is drawn here from
stroke templates (no system fonts, which differ between machines). Ink is accumulated into
coverage masks at 2x supersampling, then box-filtered down and composited over the paper.

Texture convention: row 0 of the returned arrays is the TOP of the page. Meshes in
`_desk_props` put uv v = 0 at the top edge, which is how Mitsuba's bitmap lookup reads rows.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

import procedural as G

SS = 2  # supersampling of ink masks


# ----------------------------------------------------------------------------
# Glyph strokes. Unit cell: x in [0, 1], y = 0 baseline, 1 = x-height, ~1.6 = ascender, -0.55 = descender.
# ----------------------------------------------------------------------------

def _arc(cx, cy, rx, ry, a0, a1, n=14):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]


def _glyphs() -> dict[str, tuple[float, list]]:
    """letter -> (advance width relative to x-height, list of polylines)."""
    g: dict[str, tuple[float, list]] = {}
    bowl = lambda cx=0.45: _arc(cx, 0.5, 0.38, 0.5, 0, 360, 20)
    g["o"] = (0.95, [_arc(0.5, 0.5, 0.4, 0.5, 0, 360, 20)])
    g["a"] = (0.95, [bowl(0.42), [(0.8, 1.0), (0.8, 0.0)]])
    g["d"] = (0.95, [bowl(0.42), [(0.8, 1.6), (0.8, 0.0)]])
    g["b"] = (0.95, [[(0.15, 1.6), (0.15, 0.0)], bowl(0.55)])
    g["p"] = (0.95, [[(0.15, 1.0), (0.15, -0.55)], bowl(0.55)])
    g["q"] = (0.95, [bowl(0.42), [(0.8, 1.0), (0.8, -0.55)]])
    g["g"] = (0.95, [bowl(0.42), [(0.8, 1.0), (0.8, -0.2)] + _arc(0.45, -0.2, 0.35, 0.35, 0, -150, 8)])
    g["c"] = (0.85, [_arc(0.55, 0.5, 0.4, 0.5, 40, 320, 16)])
    g["e"] = (0.9, [_arc(0.5, 0.5, 0.4, 0.5, 15, 330, 18), [(0.1, 0.5), (0.9, 0.5)]])
    g["n"] = (0.9, [[(0.15, 1.0), (0.15, 0.0)], _arc(0.5, 0.58, 0.35, 0.42, 180, 0, 10) + [(0.85, 0.0)]])
    g["h"] = (0.9, [[(0.15, 1.6), (0.15, 0.0)], _arc(0.5, 0.58, 0.35, 0.42, 180, 0, 10) + [(0.85, 0.0)]])
    g["m"] = (1.45, [[(0.1, 1.0), (0.1, 0.0)], _arc(0.32, 0.6, 0.22, 0.4, 180, 0, 8) + [(0.54, 0.0)],
                     _arc(0.76, 0.6, 0.22, 0.4, 180, 0, 8) + [(0.98, 0.0)]])
    g["r"] = (0.7, [[(0.2, 1.0), (0.2, 0.0)], _arc(0.55, 0.62, 0.35, 0.38, 170, 40, 8)])
    g["u"] = (0.9, [[(0.15, 1.0), (0.15, 0.45)] + _arc(0.5, 0.45, 0.35, 0.45, 180, 360, 10) + [(0.85, 1.0)],
                    [(0.85, 1.0), (0.85, 0.0)]])
    g["i"] = (0.45, [[(0.5, 0.78), (0.5, 0.0)], [(0.5, 1.22), (0.5, 1.3)]])
    g["l"] = (0.45, [[(0.5, 1.6), (0.5, 0.0)]])
    g["j"] = (0.5, [[(0.5, 0.78), (0.5, -0.3)] + _arc(0.25, -0.3, 0.25, 0.25, 0, -90, 6), [(0.5, 1.22), (0.5, 1.3)]])
    g["t"] = (0.6, [[(0.5, 1.35), (0.5, 0.15)] + _arc(0.75, 0.15, 0.25, 0.15, 180, 270, 5), [(0.2, 1.0), (0.85, 1.0)]])
    g["f"] = (0.6, [_arc(0.8, 1.35, 0.3, 0.25, 0, 180, 8) + [(0.5, 0.0)], [(0.2, 1.0), (0.85, 1.0)]])
    g["s"] = (0.8, [_arc(0.5, 0.76, 0.32, 0.24, 30, 270, 10) + _arc(0.5, 0.27, 0.34, 0.27, 90, -150, 10)])
    g["v"] = (0.9, [[(0.08, 1.0), (0.5, 0.0), (0.92, 1.0)]])
    g["w"] = (1.3, [[(0.05, 1.0), (0.28, 0.0), (0.5, 0.8), (0.72, 0.0), (0.95, 1.0)]])
    g["x"] = (0.85, [[(0.1, 1.0), (0.9, 0.0)], [(0.1, 0.0), (0.9, 1.0)]])
    g["y"] = (0.9, [[(0.08, 1.0), (0.5, 0.05)], [(0.92, 1.0), (0.3, -0.55)]])
    g["z"] = (0.8, [[(0.12, 1.0), (0.88, 1.0), (0.12, 0.0), (0.88, 0.0)]])
    g["k"] = (0.85, [[(0.15, 1.6), (0.15, 0.0)], [(0.85, 1.0), (0.15, 0.4)], [(0.35, 0.58), (0.85, 0.0)]])
    # capitals and digits, 1.5 x-heights tall
    g["T"] = (1.0, [[(0.05, 1.5), (0.95, 1.5)], [(0.5, 1.5), (0.5, 0.0)]])
    g["H"] = (1.1, [[(0.1, 1.5), (0.1, 0.0)], [(0.9, 1.5), (0.9, 0.0)], [(0.1, 0.75), (0.9, 0.75)]])
    g["E"] = (0.9, [[(0.85, 1.5), (0.1, 1.5), (0.1, 0.0), (0.85, 0.0)], [(0.1, 0.75), (0.7, 0.75)]])
    g["A"] = (1.1, [[(0.05, 0.0), (0.5, 1.5), (0.95, 0.0)], [(0.25, 0.55), (0.75, 0.55)]])
    g["S"] = (0.9, [_arc(0.5, 1.12, 0.38, 0.38, 20, 270, 12) + _arc(0.5, 0.4, 0.42, 0.4, 90, -160, 12)])
    g["0"] = (0.85, [_arc(0.5, 0.75, 0.4, 0.75, 0, 360, 20)])
    g["1"] = (0.6, [[(0.25, 1.2), (0.55, 1.5), (0.55, 0.0)]])
    g["2"] = (0.85, [_arc(0.5, 1.1, 0.38, 0.4, 160, -30, 10) + [(0.1, 0.0), (0.9, 0.0)]])
    g["3"] = (0.85, [_arc(0.5, 1.15, 0.36, 0.35, 150, -90, 10) + _arc(0.5, 0.4, 0.4, 0.4, 90, -150, 10)])
    g["4"] = (0.9, [[(0.7, 0.0), (0.7, 1.5), (0.05, 0.5), (0.95, 0.5)]])
    g["5"] = (0.85, [[(0.85, 1.5), (0.2, 1.5), (0.15, 0.85)] + _arc(0.5, 0.45, 0.4, 0.45, 110, -150, 12)])
    g["7"] = (0.85, [[(0.1, 1.5), (0.9, 1.5), (0.35, 0.0)]])
    g["8"] = (0.85, [_arc(0.5, 1.1, 0.33, 0.4, 0, 360, 14), _arc(0.5, 0.38, 0.4, 0.38, 0, 360, 14)])
    g["9"] = (0.85, [_arc(0.5, 1.05, 0.38, 0.45, 0, 360, 14), [(0.88, 1.05), (0.8, 0.0)]])
    g["R"] = (1.0, [[(0.1, 0.0), (0.1, 1.5)] + _arc(0.5, 1.12, 0.4, 0.38, 90, -90, 10) + [(0.1, 0.75)], [(0.45, 0.75), (0.9, 0.0)]])
    g["P"] = (0.95, [[(0.1, 0.0), (0.1, 1.5)] + _arc(0.5, 1.12, 0.4, 0.38, 90, -90, 10) + [(0.1, 0.75)]])
    g["N"] = (1.1, [[(0.1, 0.0), (0.1, 1.5), (0.9, 0.0), (0.9, 1.5)]])
    g["C"] = (1.0, [_arc(0.55, 0.75, 0.45, 0.75, 40, 320, 18)])
    g["D"] = (1.05, [[(0.1, 0.0), (0.1, 1.5)] + _arc(0.45, 0.75, 0.45, 0.75, 90, -90, 14) + [(0.1, 0.0)]])
    g["L"] = (0.85, [[(0.12, 1.5), (0.12, 0.0), (0.85, 0.0)]])
    g["M"] = (1.3, [[(0.08, 0.0), (0.08, 1.5), (0.5, 0.5), (0.92, 1.5), (0.92, 0.0)]])
    g["B"] = (0.95, [[(0.1, 0.0), (0.1, 1.5)] + _arc(0.5, 1.12, 0.38, 0.38, 90, -90, 10) + [(0.1, 0.75)],
                     [(0.1, 0.75)] + _arc(0.5, 0.38, 0.42, 0.38, 90, -90, 10) + [(0.1, 0.0)]])
    g["6"] = (0.85, [_arc(0.5, 0.42, 0.4, 0.42, 0, 360, 14), _arc(0.85, 0.95, 0.4, 0.55, 190, 100, 8)])
    g["-"] = (0.5, [[(0.15, 0.55), (0.85, 0.55)]])
    g["."] = (0.35, [[(0.5, 0.0), (0.5, 0.06)]])
    g[","] = (0.35, [[(0.55, 0.05), (0.4, -0.25)]])
    return g


GLYPHS = _glyphs()
LOWER = "etaoinshrdlcumwfgypbvkxjqz"
LOWER_P = np.array([12.7, 9.1, 8.2, 7.5, 7.0, 6.7, 6.3, 6.1, 6.0, 4.3, 4.0, 2.8, 2.8, 2.4, 2.4, 2.2, 2.0, 2.0,
                    1.9, 1.5, 1.0, 0.8, 0.15, 0.15, 0.1, 0.07])
LOWER_P = LOWER_P / LOWER_P.sum()
CAPS = "TEHASRPNCDLMB"


class Ink:
    """A set of named coverage masks over one page, drawn at SS x supersampling."""

    def __init__(self, w: int, h: int, names):
        self.w, self.h = w, h
        self.masks = {n: Image.new("L", (w * SS, h * SS), 0) for n in names}
        self.draw = {n: ImageDraw.Draw(m) for n, m in self.masks.items()}

    def line(self, name, pts, width_px, value=255):
        pts = [(x * SS, y * SS) for x, y in pts]
        w = max(1, int(round(width_px * SS)))
        self.draw[name].line(pts, fill=value, width=w, joint="curve")

    def rect(self, name, x0, y0, x1, y1, value=255):
        self.draw[name].rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=value)

    def ellipse(self, name, box, width_px, value=255):
        x0, y0, x1, y1 = (v * SS for v in box)
        self.draw[name].ellipse([x0, y0, x1, y1], outline=value, width=max(1, int(round(width_px * SS))))

    def coverage(self, name) -> np.ndarray:
        return np.asarray(self.masks[name].reduce(SS), np.float32) / 255.0


def glyph_strokes(ch, x, y_base, xh, slant=0.0, wobble=None, rng=None):
    """Polylines in pixel space for one glyph, with the cell's left edge at x and baseline at y_base."""
    adv, strokes = GLYPHS[ch]
    out = []
    for poly in strokes:
        pts = []
        for gx, gy in poly:
            px = x + gx * xh * adv * 0.9 + slant * gy * xh
            py = y_base - gy * xh
            if wobble:
                px += rng.normal(0, wobble)
                py += rng.normal(0, wobble)
            pts.append((px, py))
        out.append(pts)
    return out, adv * xh


def print_word(ink: Ink, name, x, y_base, word, xh, width, rng, tone=255, wobble=0.0, slant=0.0, track=0.12):
    for ch in word:
        if ch not in GLYPHS:
            x += xh * 0.5
            continue
        strokes, adv = glyph_strokes(ch, x, y_base, xh, slant, wobble or None, rng)
        for s in strokes:
            ink.line(name, s, width, tone)
        x += adv * (1.0 + track)
    return x


def word_width(word, xh, track=0.12):
    return sum(GLYPHS.get(c, (0.5, []))[0] * xh * (1.0 + track) for c in word)


def random_word(rng, minlen=2, maxlen=9, caps_p=0.0) -> str:
    n = int(rng.integers(minlen, maxlen + 1))
    w = "".join(rng.choice(list(LOWER), size=n, p=LOWER_P))
    if rng.random() < caps_p:
        w = str(rng.choice(list(CAPS))) + w[1:]
    return w


def text_lines(ink: Ink, name, box, pitch, xh, width, rng, tone=255, paragraph=True, caps_p=0.04, track=0.12,
               slant=0.0, wobble=0.0, ragged=True, numbers_p=0.0):
    """Fill `box` = (x0, y0, x1, y1) with lines of pseudo-words; returns the y after the last line."""
    x0, y0, x1, y1 = box
    y = y0 + pitch
    left_in_par = int(rng.integers(3, 8))
    while y < y1:
        x = x0
        last = paragraph and left_in_par == 1
        limit = x1 - (rng.uniform(0.15, 0.6) * (x1 - x0) if last else (rng.uniform(0.0, 0.05) * (x1 - x0) if ragged else 0))
        indent = xh * 2.2 if (paragraph and left_in_par == -1) else 0
        x += indent
        while True:
            w = str(rng.integers(10, 9999)) if rng.random() < numbers_p else random_word(rng, 2, 9, caps_p)
            ww = word_width(w, xh, track)
            if x + ww > limit:
                break
            print_word(ink, name, x, y, w, xh, width, rng, tone, wobble, slant, track)
            x += ww + xh * 0.55
        y += pitch
        left_in_par -= 1
        if paragraph and left_in_par <= 0:
            y += pitch * 0.6
            left_in_par = int(rng.integers(3, 8))
    return y


# ----------------------------------------------------------------------------
# Handwriting: connected wavy strokes through random control points.
# ----------------------------------------------------------------------------

def _catmull(pts: np.ndarray, per: int = 6) -> np.ndarray:
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2])
    return np.array(out)


def cursive_word(ink: Ink, name, x, y_base, nletters, xh, width, rng, slant=0.25, tone=255):
    step = xh * rng.uniform(0.38, 0.55)
    pts = []
    for i in range(nletters * 2 + 1):
        r = rng.random()
        if i % 2 == 0:
            h = 0.0 + rng.normal(0, 0.05)
        elif r < 0.18:
            h = rng.uniform(1.5, 1.9)          # ascender loop
        elif r < 0.28:
            h = rng.uniform(-0.6, -0.3)        # descender
        else:
            h = rng.uniform(0.7, 1.15)
        pts.append((x + i * step + slant * h * xh + rng.normal(0, step * 0.12), y_base - h * xh))
    path = _catmull(np.array(pts), 7)
    ink.line(name, [tuple(p) for p in path], width, tone)
    return x + (nletters * 2 + 1) * step


def hand_word(ink: Ink, name, x, y_base, word, xh, width, rng, slant=0.22, tone=255):
    """A word of the print glyphs, drawn the way people print by hand: slanted, uneven in size and baseline."""
    for ch in word:
        s = rng.uniform(0.9, 1.12)
        dy = rng.normal(0, 0.07 * xh)
        strokes, adv = glyph_strokes(ch, x, y_base + dy, xh * s, slant + rng.normal(0, 0.04), 0.035 * xh, rng)
        for st in strokes:
            ink.line(name, st, width * rng.uniform(0.85, 1.2), tone)
        x += adv * rng.uniform(0.98, 1.1)
    return x


def handwriting(ink: Ink, name, box, pitch, xh, width, rng, tone=255, slant=0.22, density=1.0, para=True):
    x0, y0, x1, y1 = box
    y = y0 + pitch
    drift = 0.0
    while y < y1:
        if rng.random() > density:
            y += pitch
            continue
        x = x0 + (xh * 2.5 if rng.random() < 0.12 else 0)
        limit = x1 - (rng.uniform(0.1, 0.6) * (x1 - x0) if rng.random() < 0.25 else rng.uniform(0, 0.05) * (x1 - x0))
        while True:
            w = random_word(rng, 2, 8, 0.12)
            ww = word_width(w, xh, 0.0) * 1.05
            if x + ww > limit:
                break
            drift = float(np.clip(drift + rng.normal(0, 0.05 * xh), -0.25 * xh, 0.25 * xh))
            x = hand_word(ink, name, x, y + drift, w, xh, width, rng, slant, tone) + xh * rng.uniform(0.45, 0.8)
        y += pitch
    return y


# ----------------------------------------------------------------------------
# Paper base
# ----------------------------------------------------------------------------

def paper_base(h: int, w: int, tint=(0.93, 0.92, 0.88), seed=0, fibre=0.025, creases=0, rng=None) -> np.ndarray:
    n = 0.5 * G.fbm(h, w, octaves=3, base=6, seed=seed) + 0.5 * G.fbm(h, w, octaves=2, base=300, seed=seed + 1)
    img = np.asarray(tint, np.float32)[None, None, :] * (1 - fibre + 2 * fibre * n[..., None])
    rng = rng or np.random.default_rng(seed)
    for _ in range(creases):
        if rng.random() < 0.5:
            x = int(rng.uniform(0.15, 0.85) * w)
            img[:, max(0, x - 1):x + 1] *= 0.955
            img[:, x + 1:x + 3] *= 1.015
        else:
            y = int(rng.uniform(0.15, 0.85) * h)
            img[max(0, y - 1):y + 1, :] *= 0.955
            img[y + 1:y + 3, :] *= 1.015
    return img


def composite(base: np.ndarray, layers: list[tuple[np.ndarray, tuple]]) -> np.ndarray:
    out = base.copy()
    for cov, colour in layers:
        c = np.asarray(colour, np.float32)[None, None, :]
        out = out * (1 - cov[..., None]) + c * cov[..., None]
    return np.clip(out, 0, 1)


def coffee_ring(h, w, cx, cy, r, seed=0) -> np.ndarray:
    """Multiplicative darkening mask for a dried coffee ring, 1 = untouched."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(x - cx, y - cy)
    wob = (G.fbm(h, w, octaves=4, base=8, seed=seed) - 0.5) * 0.12 * r
    ring = np.exp(-(((d + wob) - r) / (0.045 * r)) ** 2)
    fill = np.clip(1.0 - (d / r) ** 6, 0, 1) * 0.15
    m = 1.0 - 0.28 * ring - 0.05 * fill
    return np.stack([m, m * 0.97, m * 0.9], -1).astype(np.float32)


# ----------------------------------------------------------------------------
# Page generators. All return (h, w, 3) float32 with row 0 at the top.
# ----------------------------------------------------------------------------

def notebook_spread(h: int, w: int, seed: int) -> np.ndarray:
    """An open ruled notebook: left page handwritten notes, right page diagram, list and numbers."""
    rng = np.random.default_rng(seed)
    px_per_m = w / 0.42
    mm = px_per_m / 1000.0
    base = paper_base(h, w, (0.95, 0.93, 0.86), seed, fibre=0.02)
    ink = Ink(w, h, ["blue", "pencil", "rule", "red", "black"])
    # ruled lines every 7 mm, margin line on each page
    pitch = 7.0 * mm
    top = 22 * mm
    half = w // 2
    for page_x0, page_x1 in ((10 * mm, half - 8 * mm), (half + 8 * mm, w - 10 * mm)):
        y = top
        while y < h - 14 * mm:
            ink.line("rule", [(page_x0 - 4 * mm, y), (page_x1 + 4 * mm, y)], 0.25 * mm * 1.2, 255)
            y += pitch
    ink.line("red", [(22 * mm, 8 * mm), (22 * mm, h - 8 * mm)], 0.3 * mm, 150)
    ink.line("red", [(half + 24 * mm, 8 * mm), (half + 24 * mm, h - 8 * mm)], 0.3 * mm, 150)
    # left page: dated heading, underlined, then notes in blue ink
    hand_word(ink, "blue", half - 48 * mm, top - 3 * mm, "12-06", 3.0 * mm, 0.5 * mm * 1.5, rng)
    hand_word(ink, "blue", 26 * mm, top + pitch - 1.5 * mm, "Notes", 4.6 * mm, 0.62 * mm * 1.5, rng)
    ink.line("blue", [(26 * mm, top + pitch + 0.8 * mm), (78 * mm, top + pitch + 0.8 * mm + rng.normal(0, 0.3) * mm)], 0.6 * mm * 1.2, 255)
    handwriting(ink, "blue", (26 * mm, top + pitch - 0.7 * mm, half - 12 * mm, h - 16 * mm), pitch, 2.6 * mm, 0.45 * mm * 1.5, rng,
                slant=0.25, density=0.93)
    # a few pencil annotations in the margin
    handwriting(ink, "pencil", (half + 8 * mm, top, half + 22 * mm, top + 6 * pitch), pitch, 2.0 * mm, 0.4 * mm * 1.4, rng, density=0.5)
    # right page: bulleted list, then a small diagram of boxes and arrows, then a table of numbers
    rx0, rx1 = half + 28 * mm, w - 14 * mm
    for i in range(6):
        y = top + pitch * (1 + i) - 0.5 * mm
        ink.ellipse("blue", (rx0 - 5 * mm, y - 2.0 * mm, rx0 - 2.2 * mm, y + 0.8 * mm), 0.5 * mm * 1.4, 255)
        handwriting(ink, "blue", (rx0, y - pitch - 0.2 * mm, rx1 - rng.uniform(0, 40) * mm, y + 1), pitch, 2.6 * mm, 0.45 * mm * 1.5, rng, slant=0.25)
    dy = top + pitch * 8.5
    boxes = []
    for i in range(3):
        bx = rx0 + i * 38 * mm
        ink.line("black", [(bx, dy), (bx + 28 * mm, dy), (bx + 28 * mm, dy + 14 * mm), (bx, dy + 14 * mm), (bx, dy)], 0.55 * mm * 1.2, 255)
        handwriting(ink, "black", (bx + 3 * mm, dy + 1 * mm, bx + 26 * mm, dy + 14 * mm), 7 * mm, 2.4 * mm, 0.45 * mm * 1.3, rng, density=1.0)
        boxes.append(bx)
        if i:
            ink.line("black", [(bx - 10 * mm, dy + 7 * mm), (bx - 1.5 * mm, dy + 7 * mm)], 0.5 * mm * 1.2, 255)
            ink.line("black", [(bx - 3.5 * mm, dy + 5 * mm), (bx - 1.5 * mm, dy + 7 * mm), (bx - 3.5 * mm, dy + 9 * mm)], 0.5 * mm * 1.2, 255)
    ink.ellipse("red", (rx0 + 70 * mm, dy - 6 * mm, rx0 + 118 * mm, dy + 22 * mm), 0.6 * mm * 1.3, 220)
    ty = dy + 30 * mm
    for r in range(5):
        for c in range(4):
            print_word(ink, "pencil", rx0 + c * 24 * mm, ty + r * 6.5 * mm, str(int(rng.integers(10, 999))), 2.3 * mm, 0.4 * mm * 1.3, rng,
                       200, wobble=0.08 * mm, slant=0.2)
    handwriting(ink, "blue", (rx0, ty + 38 * mm, rx1, h - 16 * mm), pitch, 2.6 * mm, 0.45 * mm * 1.5, rng, slant=0.25, density=0.7)
    layers = [(ink.coverage("rule") * 0.5, (0.45, 0.62, 0.78)), (ink.coverage("pencil") * 0.75, (0.30, 0.30, 0.32)),
              (ink.coverage("red") * 0.9, (0.78, 0.20, 0.16)), (ink.coverage("blue"), (0.10, 0.14, 0.52)),
              (ink.coverage("black"), (0.06, 0.06, 0.07))]
    img = composite(base, layers)
    img *= 1.0 - 0.7 * (1.0 - coffee_ring(h, w, w * 0.77, h * 0.80, 22 * mm, seed))
    # gutter shadow
    x = np.arange(w, dtype=np.float32)[None, :, None]
    img *= 1.0 - 0.28 * np.exp(-(np.abs(x - half) / (7 * mm)) ** 2)
    return np.clip(img, 0, 1)


def printed_page(h: int, w: int, seed: int, kind: str = "report") -> np.ndarray:
    """A4 portrait: 'report' (heading, paragraphs, bar chart), 'table' (grid of numbers) or 'marked'
    (report text with red pen annotations)."""
    rng = np.random.default_rng(seed)
    mm = w / 210.0
    base = paper_base(h, w, (0.95, 0.95, 0.93), seed, fibre=0.015, creases=int(rng.integers(0, 3)), rng=rng)
    ink = Ink(w, h, ["black", "grey", "red", "blue"])
    xh = 1.9 * mm
    pitch = 5.0 * mm
    x0, x1 = 20 * mm, w - 20 * mm
    sw = 0.2 * mm * 1.5
    if kind in ("report", "marked"):
        # title and subtitle (thicker strokes, larger)
        print_word(ink, "black", x0, 30 * mm, "Report", 5.4 * mm, 0.55 * mm * 1.4, rng, 255, track=0.1)
        text_lines(ink, "grey", (x0, 34 * mm, x1, 44 * mm), pitch, xh, sw, rng, 190, paragraph=False)
        ink.line("black", [(x0, 47 * mm), (x1, 47 * mm)], 0.35 * mm * 1.3, 255)
        y = text_lines(ink, "black", (x0, 50 * mm, x1, 115 * mm), pitch, xh, sw, rng, 255)
        # bar chart with axes, ticks and tiny labels
        cx0, cy0, cx1, cy1 = x0 + 8 * mm, y + 6 * mm, x1 - 10 * mm, y + 58 * mm
        ink.line("black", [(cx0, cy0), (cx0, cy1), (cx1, cy1)], 0.3 * mm * 1.3, 255)
        n = 7
        for i in range(n):
            bh = rng.uniform(0.2, 0.95) * (cy1 - cy0 - 4 * mm)
            bx = cx0 + 4 * mm + i * (cx1 - cx0 - 8 * mm) / n
            bw = (cx1 - cx0 - 8 * mm) / n * 0.62
            ink.rect("grey", bx, cy1 - bh, bx + bw, cy1 - 0.4 * mm, 150 + int(rng.integers(0, 90)))
            print_word(ink, "black", bx, cy1 + 4.2 * mm, random_word(rng, 2, 4), 1.2 * mm, 0.15 * mm * 1.4, rng, 255)
        for t in range(5):
            ty = cy1 - t * (cy1 - cy0) / 5
            ink.line("black", [(cx0 - 1.5 * mm, ty), (cx0, ty)], 0.2 * mm * 1.3, 255)
            print_word(ink, "black", cx0 - 7 * mm, ty + 0.9 * mm, str(t * 20), 1.2 * mm, 0.15 * mm * 1.4, rng, 255)
        text_lines(ink, "black", (x0, cy1 + 14 * mm, x1, h - 24 * mm), pitch, xh, sw, rng, 255)
        print_word(ink, "grey", w / 2 - 3 * mm, h - 14 * mm, str(int(rng.integers(1, 40))), 1.8 * mm, 0.2 * mm * 1.4, rng, 220)
        if kind == "marked":
            for _ in range(5):
                yy = rng.uniform(55, 200) * mm
                xx = rng.uniform(25, 120) * mm
                ink.line("red", [(xx, yy), (xx + rng.uniform(25, 70) * mm, yy + rng.normal(0, 0.6) * mm)], 0.45 * mm * 1.3, 240)
            ink.ellipse("red", (x0 + 30 * mm, 60 * mm, x0 + 95 * mm, 78 * mm), 0.5 * mm * 1.3, 240)
            handwriting(ink, "red", (x1 - 62 * mm, 52 * mm, x1 + 12 * mm, 74 * mm), 6 * mm, 2.4 * mm, 0.45 * mm * 1.3, rng, density=1.0)
            handwriting(ink, "red", (x0 + 4 * mm, h - 48 * mm, x1 - 20 * mm, h - 28 * mm), 6.5 * mm, 2.6 * mm, 0.45 * mm * 1.3, rng, density=1.0)
    else:
        print_word(ink, "black", x0, 30 * mm, "Table", 4.6 * mm, 0.5 * mm * 1.4, rng, 255)
        cols, rows = 6, 26
        cw, rh = (x1 - x0) / cols, 6.4 * mm
        y0 = 40 * mm
        for r in range(rows + 1):
            ink.line("black", [(x0, y0 + r * rh), (x1, y0 + r * rh)], (0.4 if r in (0, 1, rows) else 0.15) * mm * 1.4, 255)
        for c in range(cols + 1):
            ink.line("black", [(x0 + c * cw, y0), (x0 + c * cw, y0 + rows * rh)], 0.15 * mm * 1.4, 255)
        for c in range(cols):
            print_word(ink, "black", x0 + c * cw + 2 * mm, y0 + 4.4 * rh / 6.4, random_word(rng, 3, 7, 0.9), 1.7 * mm, 0.3 * mm * 1.4, rng, 255)
        for r in range(1, rows):
            for c in range(cols):
                v = f"{rng.uniform(0, 9999):.2f}" if c else random_word(rng, 3, 8)
                print_word(ink, "black", x0 + c * cw + 2 * mm, y0 + (r + 1) * rh - 2.0 * mm, v.replace(".", "."), 1.7 * mm, 0.22 * mm * 1.4, rng, 255)
        handwriting(ink, "blue", (x0 + 10 * mm, y0 + (rows + 1) * rh, x1 - 20 * mm, y0 + (rows + 4) * rh), 6 * mm, 2.4 * mm, 0.45 * mm * 1.3, rng)
    layers = [(ink.coverage("grey") * 0.85, (0.35, 0.35, 0.37)), (ink.coverage("black") * 0.92, (0.07, 0.07, 0.08)),
              (ink.coverage("red"), (0.78, 0.15, 0.12)), (ink.coverage("blue"), (0.10, 0.13, 0.50))]
    return composite(base, layers)


def sticky_note(size: int, seed: int, colour) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.tile(np.asarray(colour, np.float32), (size, size, 1))
    n = G.fbm(size, size, octaves=3, base=4, seed=seed)
    base *= (0.96 + 0.07 * n[..., None])
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
    base *= 1.0 - 0.06 * y            # darker toward the glue strip... lighter at the free end
    ink = Ink(size, size, ["ink"])
    mm = size / 76.0
    handwriting(ink, "ink", (7 * mm, 6 * mm, 70 * mm, 66 * mm), 8.5 * mm, 3.3 * mm, 0.6 * mm * 1.4, rng, slant=0.3, density=0.9)
    return composite(base, [(ink.coverage("ink") * 0.9, (0.12, 0.13, 0.30))])


def code_screen(h: int, w: int, seed: int) -> np.ndarray:
    """A dark-theme code editor: sidebar, tab bar, line numbers and syntax-coloured tokens."""
    rng = np.random.default_rng(seed)
    bg = np.array([0.20, 0.22, 0.28], np.float32)       # mid-dark slate: a near-black screen would fail the exposure band
    img = np.tile(bg, (h, w, 1))
    side = int(w * 0.14)
    img[:, :side] = np.array([0.15, 0.16, 0.21], np.float32)
    img[: int(h * 0.045), :] = np.array([0.12, 0.13, 0.17], np.float32)
    img[-int(h * 0.03):, :] = np.array([0.10, 0.35, 0.65], np.float32)
    names = ["white", "blue", "orange", "green", "purple", "grey"]
    ink = Ink(w, h, names)
    xh = h / 120.0
    pitch = xh * 2.9
    wd = max(1.0, xh * 0.22)
    colours = {"white": (0.80, 0.82, 0.85), "blue": (0.35, 0.65, 0.95), "orange": (0.90, 0.60, 0.30),
               "green": (0.55, 0.78, 0.45), "purple": (0.75, 0.50, 0.90), "grey": (0.38, 0.40, 0.45)}
    y = h * 0.075
    ln = 1
    while y < h * 0.95:
        print_word(ink, "grey", side + xh * 1.2, y, f"{ln}", xh, wd, rng, 255, track=0.05)
        indent = int(rng.integers(0, 4)) * xh * 3.6
        x = side + xh * 8 + indent
        for _ in range(int(rng.integers(0, 6))):
            name = str(rng.choice(["white", "blue", "orange", "green", "purple", "white"]))
            w_ = random_word(rng, 2, 9)
            ww = word_width(w_, xh, 0.05)
            if x + ww > w * 0.96:
                break
            print_word(ink, name, x, y, w_, xh, wd, rng, 255, track=0.05)
            x += ww + xh * 0.8
            if rng.random() < 0.2:
                print_word(ink, "white", x - xh * 0.5, y, str(rng.choice(["(", ")", ".", ","])), xh, wd, rng, 255)
        y += pitch
        ln += 1
    # sidebar entries and tabs
    for i in range(14):
        print_word(ink, "grey", xh * 3 + (xh * 3 if i % 4 else 0), h * 0.09 + i * pitch, random_word(rng, 4, 10), xh, wd, rng, 255, track=0.05)
    for i in range(3):
        print_word(ink, "white", side + xh * 3 + i * xh * 22, h * 0.03, random_word(rng, 4, 9) + ".py", xh, wd, rng, 255, track=0.05)
    for name in names:
        c = np.asarray(colours[name], np.float32)
        cov = ink.coverage(name)[..., None]
        img = img * (1 - cov) + c * cov
    # current-line highlight
    img[int(h * 0.42): int(h * 0.42 + pitch), side:] += 0.04
    return np.clip(img, 0, 1)


def pinboard(h: int, w: int, seed: int) -> np.ndarray:
    """Cork with pinned note cards, each carrying handwriting."""
    rng = np.random.default_rng(seed)
    n = G.fbm(h, w, octaves=5, base=10, seed=seed)
    fine = G.fbm(h, w, octaves=2, base=400, seed=seed + 1)
    cork = np.array([0.55, 0.40, 0.25], np.float32) * (0.75 + 0.35 * n[..., None]) * (0.9 + 0.2 * fine[..., None])
    img = cork.copy()
    mm = w / 1000.0
    for i in range(9):
        cw, ch = int(rng.uniform(90, 150) * mm * 1.6), int(rng.uniform(70, 110) * mm * 1.6)
        x, y = int(rng.uniform(0.02, 0.85) * w), int(rng.uniform(0.04, 0.8) * h)
        x, y = min(x, w - cw - 2), min(y, h - ch - 2)
        col = [(0.93, 0.93, 0.90), (0.95, 0.85, 0.35), (0.90, 0.55, 0.60), (0.55, 0.78, 0.60), (0.55, 0.70, 0.90)][i % 5]
        card = sticky_note(max(cw, ch), seed + i, col)[:ch, :cw]
        img[y:y + ch, x:x + cw] = card
        py, px = y + 6, x + cw // 2
        img[py - 3:py + 3, px - 3:px + 3] = np.array([0.8, 0.12, 0.1], np.float32)
    return np.clip(img, 0, 1)


def rug(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = G.fbm(h, w, octaves=5, base=6, seed=seed)
    fine = G.fbm(h, w, octaves=2, base=500, seed=seed + 1)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = x / w, y / h
    border = np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v))
    band = ((border > 0.04) & (border < 0.07)) | ((border > 0.11) & (border < 0.125))
    diamond = (np.abs(((u * 6) % 1) - 0.5) + np.abs(((v * 4) % 1) - 0.5)) < 0.32
    base = np.where(band[..., None], np.array([0.78, 0.66, 0.42], np.float32), np.array([0.32, 0.12, 0.10], np.float32))
    base = np.where((diamond & (border > 0.15))[..., None], np.array([0.12, 0.20, 0.34], np.float32), base)
    return np.clip(base * (0.78 + 0.4 * n[..., None]) * (0.88 + 0.24 * fine[..., None]), 0, 1)


def spine_atlas(seed: int = 5) -> np.ndarray:
    """16 x 4 cells of 128 x 512 px: book spines with colour, bands and vertical titles. The last cell is
    plain paper (page edges). Each cell keeps an 8 px plain border so covers can sample a flat colour."""
    rng = np.random.default_rng(seed)
    cw, ch, cols, rows = 128, 512, 16, 4
    atlas = np.zeros((rows * ch, cols * cw, 3), np.float32)
    palette = [(0.45, 0.08, 0.06), (0.08, 0.20, 0.30), (0.55, 0.42, 0.12), (0.12, 0.25, 0.14), (0.30, 0.12, 0.25),
               (0.75, 0.70, 0.60), (0.12, 0.12, 0.14), (0.62, 0.28, 0.12), (0.20, 0.32, 0.45), (0.40, 0.45, 0.30),
               (0.68, 0.58, 0.42), (0.35, 0.10, 0.12), (0.18, 0.18, 0.22), (0.80, 0.78, 0.72), (0.25, 0.40, 0.35),
               (0.55, 0.20, 0.30)]
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx == rows * cols - 1:
                paper = np.tile(np.array([0.86, 0.82, 0.70], np.float32), (ch, cw, 1))
                paper *= (0.94 + 0.06 * G.fbm(ch, cw, octaves=2, base=120, seed=seed + 7)[..., None])
                atlas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = paper
                continue
            col = np.asarray(palette[int(rng.integers(0, len(palette)))], np.float32) * rng.uniform(0.85, 1.15)
            wear = G.fbm(ch, cw, octaves=4, base=8, seed=seed + 10 + idx)
            cell = np.tile(col, (ch, cw, 1)) * (0.88 + 0.22 * wear[..., None])
            lum = float(col @ np.array([0.2, 0.7, 0.1]))
            fg = np.array([0.88, 0.80, 0.55], np.float32) if lum < 0.4 else np.array([0.10, 0.09, 0.08], np.float32)
            ink = Ink(cw, ch, ["t"])
            yb = int(ch * rng.uniform(0.07, 0.12))
            for y in (yb, yb + 12, ch - yb - 12, ch - yb):
                if rng.random() < 0.8:
                    ink.line("t", [(14, y), (cw - 14, y)], 2.0 + rng.uniform(0, 2), 255)
            # title: render horizontally on a side canvas then rotate to read bottom to top
            tw, th = int(ch * 0.62), cw - 40
            tin = Ink(tw, th, ["t"])
            xh = th * rng.uniform(0.17, 0.26)
            x = 6
            for _ in range(int(rng.integers(1, 4))):
                wd = random_word(rng, 3, 8, 0.6)
                if x + word_width(wd, xh) > tw - 6:
                    break
                x = print_word(tin, "t", x, th * 0.62, wd, xh, max(1.6, xh * 0.2), rng, 255) + xh * 0.6
            title = np.asarray(Image.fromarray((tin.coverage("t") * 255).astype(np.uint8)).rotate(90, expand=True), np.float32) / 255
            y0 = int(ch * 0.17)
            cov = ink.coverage("t")
            cov[y0:y0 + title.shape[0], 20:20 + title.shape[1]] = np.maximum(
                cov[y0:y0 + title.shape[0], 20:20 + title.shape[1]], title[: ch - y0, :cw - 20])
            cov[:, :8] = 0
            cov[:, -8:] = 0
            cell = cell * (1 - cov[..., None] * 0.95) + fg * cov[..., None] * 0.95
            cell[:, :8] = col * 0.9
            cell[:, -8:] = col * 0.9
            atlas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
    return np.clip(atlas, 0, 1)


def wood_top(h: int, w: int, seed: int = 3, base=(0.60, 0.42, 0.25), dark=(0.33, 0.19, 0.09)) -> np.ndarray:
    """Oak veneer for the desk top: irregular straight grain running along u, with fibre streaks and pores."""
    v = (np.arange(h, dtype=np.float32) + 0.5)[:, None] / h
    warp = G.fbm(h, w, octaves=4, base=3, seed=seed, aspect=0.06)
    tone = G.fbm(h, w, octaves=3, base=2, seed=seed + 5, aspect=0.2)
    streak = G.fbm(h, w, octaves=5, base=24, seed=seed + 1, aspect=0.015)
    fine = G.fbm(h, w, octaves=3, base=180, seed=seed + 2, aspect=0.008)
    pores = G.fbm(h, w, octaves=2, base=500, seed=seed + 3, aspect=0.02)
    rings = np.abs(np.sin(np.pi * (v * 24.0 + 5.0 * warp)))
    mix = np.clip(0.10 * rings ** 3 + 0.42 * streak + 0.30 * fine + 0.14 * pores + 0.18 * (tone - 0.5), 0, 1)[..., None]
    rgb = np.asarray(base, np.float32) * (1 - mix) + np.asarray(dark, np.float32) * mix
    return np.clip(rgb * (0.94 + 0.12 * tone[..., None]), 0, 1)


def leather_mat(h: int, w: int, seed: int = 4) -> np.ndarray:
    n = G.fbm(h, w, octaves=3, base=60, seed=seed)
    n2 = G.fbm(h, w, octaves=2, base=300, seed=seed + 1)
    return np.clip(np.array([0.08, 0.075, 0.07], np.float32)[None, None, :] * (0.7 + 0.5 * n[..., None]) * (0.85 + 0.3 * n2[..., None]), 0, 1)
