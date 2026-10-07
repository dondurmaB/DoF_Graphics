"""Mesh recipes for the desk scene: keyboard, mouse, monitor, pens, mug, headphones, chair, books, paper.

Same conventions as props.py (meters, y up, local frame with the base at y = 0). Each recipe returns a
Mesh or a dict of Meshes keyed by material. Pure numpy; nothing here imports Mitsuba. The text and
paper textures these meshes expect are in _desk_text.py (uv v = 0 at the top edge of a page).
"""

from __future__ import annotations

import numpy as np

import procedural as G
from procedural import Mesh, box, compose, ellipsoid, lathe, rotate, scale, sweep, translate
from props import arc, bezier, slab_disc


def rot_between(a, b) -> np.ndarray:
    """4x4 rotation taking unit vector a onto unit vector b."""
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


def _quad(m: Mesh, pts, normal, uv) -> None:
    m += Mesh(pts, np.tile(normal, (4, 1)), uv, np.array([[0, 1, 2], [0, 2, 3]]))


# ----------------------------------------------------------------------------
# Keyboard and mouse
# ----------------------------------------------------------------------------

KEY_COLS, KEY_ROWS = 16, 8


def keycap(cx: float, cz: float, cell: tuple[int, int], bw=0.0178, tw=0.0138, h=0.0085, depth=None) -> Mesh:
    """A tapered key; the top uv shows one legend cell of the key atlas, the sides a plain corner of it."""
    depth = depth or bw
    hb, ht, db, dt = bw / 2, tw / 2, depth / 2, tw / 2 * depth / bw
    col, row = cell
    u0, v0 = col / KEY_COLS, row / KEY_ROWS
    du, dv = 1.0 / KEY_COLS, 1.0 / KEY_ROWS
    plain = [(u0 + 0.03 * du, v0 + 0.03 * dv)] * 4
    top_uv = [(u0 + 0.15 * du, v0 + 0.85 * dv), (u0 + 0.85 * du, v0 + 0.85 * dv),
              (u0 + 0.85 * du, v0 + 0.15 * dv), (u0 + 0.15 * du, v0 + 0.15 * dv)]
    b = [(cx - hb, 0, cz + db), (cx + hb, 0, cz + db), (cx + hb, 0, cz - db), (cx - hb, 0, cz - db)]
    t = [(cx - ht, h, cz + dt), (cx + ht, h, cz + dt), (cx + ht, h, cz - dt), (cx - ht, h, cz - dt)]
    m = Mesh()
    _quad(m, t, (0, 1, 0), top_uv)
    for i in range(4):
        j = (i + 1) % 4
        quad = [b[i], b[j], t[j], t[i]]
        e = np.subtract(b[j], b[i])
        n = np.cross(e, (0, 1, 0))
        n = n / np.linalg.norm(n)
        mid = np.mean(quad, axis=0) - np.array([cx, h / 2, cz])
        if np.dot(n, mid) < 0:
            n = -n
        _quad(m, quad, n, plain)
    return m.oriented()


def keyboard(seed: int = 0) -> dict[str, Mesh]:
    """Compact keyboard, front edge at +z, keys on an aluminium plate. The tilt is applied when placing."""
    rng = np.random.default_rng(seed)
    body = box((0.436, 0.012, 0.134), (0, 0.006, 0))
    keys = Mesh()
    layout = [(14, 0.0), (14, 0.0), (13, 0.004), (12, 0.012), (1, 0.0)]
    z0 = -0.049
    pitch = 0.0192
    for r, (n, off) in enumerate(layout):
        z = z0 + r * pitch
        if r == 4:
            keys += keycap(0.0, z, (int(rng.integers(0, KEY_COLS)), int(rng.integers(0, KEY_ROWS))), 0.09, 0.0715, 0.0085, 0.0178).transformed(translate((0, 0.012, 0)))
            for x in (-0.152, -0.126, 0.114, 0.14):
                keys += keycap(x, z, (int(rng.integers(0, KEY_COLS)), int(rng.integers(0, KEY_ROWS)))).transformed(translate((0, 0.012, 0)))
            continue
        width = n * pitch
        for k in range(n):
            x = -width / 2 + (k + 0.5) * pitch + off
            keys += keycap(x, z, (int(rng.integers(0, KEY_COLS)), int(rng.integers(0, KEY_ROWS)))).transformed(translate((0, 0.012, 0)))
    return {"body": body, "keys": keys}


def keycap_atlas(seed: int = 3) -> np.ndarray:
    """16 x 8 legend cells of 64 px: dark keycap with a light legend glyph; row 0 is the top."""
    import scenes._desk_text as T

    rng = np.random.default_rng(seed)
    cell = 64
    img = np.tile(np.array([0.045, 0.045, 0.05], np.float32), (KEY_ROWS * cell, KEY_COLS * cell, 1))
    ink = T.Ink(KEY_COLS * cell, KEY_ROWS * cell, ["g"])
    chars = list("abcdefghijklmnoprstuvwxyz0123456789")
    for r in range(KEY_ROWS):
        for c in range(KEY_COLS):
            ch = chars[int(rng.integers(0, len(chars)))]
            x = c * cell + cell * 0.32
            strokes, _ = T.glyph_strokes(ch, x, r * cell + cell * 0.66, cell * 0.30)
            for s in strokes:
                ink.line("g", s, 2.4)
    cov = ink.coverage("g")[..., None]
    img = img * (1 - cov) + np.array([0.82, 0.82, 0.84], np.float32) * cov
    return np.clip(img, 0, 1)


def mouse() -> dict[str, Mesh]:
    shell = ellipsoid((0.033, 0.021, 0.057), 20, 36).transformed(translate((0, 0.006, 0)))
    wheel = tube((0, 0.0255, -0.025), (0, 0.0255, -0.012), 0.0035, 10)
    seam = box((0.0016, 0.001, 0.03), (0, 0.0265, -0.032))
    return {"shell": shell, "dark": join(wheel, seam)}


# ----------------------------------------------------------------------------
# Monitor
# ----------------------------------------------------------------------------

MONITOR_W, MONITOR_H = 0.60, 0.3375
MONITOR_CENTRE_Y = 0.285


def monitor() -> dict[str, Mesh]:
    """Display on a stand, screen facing +z. The screen itself is an emissive rectangle added by the scene
    at (0, MONITOR_CENTRE_Y, 0.0095), tilted back by 5 degrees like the panel here."""
    base = lathe([(0.0, 0.0), (0.105, 0.0), (0.11, 0.003), (0.11, 0.009), (0.095, 0.012), (0.0, 0.012)], 48).transformed(scale((1.0, 1.0, 0.75)))
    neck = box((0.045, 0.2, 0.022), (0, 0.11, -0.03))
    tilt = compose(translate((0, MONITOR_CENTRE_Y, 0)), rotate((1, 0, 0), -5), translate((0, -MONITOR_CENTRE_Y, 0)))
    bezel = box((MONITOR_W + 0.02, MONITOR_H + 0.02, 0.012), (0, MONITOR_CENTRE_Y, 0)).transformed(tilt)
    back = box((0.34, 0.2, 0.026), (0, MONITOR_CENTRE_Y, -0.018)).transformed(tilt)
    logo = ellipsoid((0.006, 0.006, 0.0015), 8, 12).transformed(translate((0, MONITOR_CENTRE_Y - MONITOR_H / 2 - 0.0035, 0.007)))
    return {"body": join(bezel, back), "stand": join(base, neck), "logo": logo}


# ----------------------------------------------------------------------------
# Pens, pencils, mug, cup
# ----------------------------------------------------------------------------

def ballpoint(length: float = 0.142) -> dict[str, Mesh]:
    """Along +y from the tip at y = 0: barrel, chrome tip and clip."""
    r = 0.0052
    path = [(0, 0.0, 0), (0, 0.012, 0), (0, 0.04, 0), (0, length, 0)]
    radii = [0.0012, 0.0036, r, r * 0.95]
    barrel = sweep(path, radii, 14)
    tip = sweep([(0, 0.0, 0), (0, 0.016, 0)], [0.0007, 0.0034], 10)
    clip = box((0.0014, 0.05, 0.003), (0.0057, length - 0.03, 0))
    return {"body": barrel, "metal": join(tip, clip)}


def pencil(length: float = 0.178) -> dict[str, Mesh]:
    """Hexagonal painted pencil with a sharpened tip (wood and lead) and a ferrule with eraser."""
    r = 0.0036
    body = sweep([(0, 0.014, 0), (0, length - 0.02, 0)], r, 6, caps=False)
    wood = sweep([(0, 0.0045, 0), (0, 0.014, 0)], [0.0016, r], 10, caps=False)
    lead = sweep([(0, 0.0, 0), (0, 0.0048, 0)], [0.0003, 0.0016], 8)
    ferrule = sweep([(0, length - 0.02, 0), (0, length - 0.008, 0)], r * 1.02, 12)
    eraser = sweep([(0, length - 0.008, 0), (0, length, 0)], r * 0.95, 12)
    return {"paint": body, "wood": wood, "lead": lead, "metal": ferrule, "eraser": eraser}


def mug() -> dict[str, Mesh]:
    outer = [(0.0, 0.0), (0.036, 0.0), (0.0405, 0.004), (0.0425, 0.02), (0.0435, 0.06), (0.0445, 0.092), (0.0445, 0.0955)]
    rim = arc(0.0012, 0, 180, (0.0433, 0.0955), 6)
    inner = [(0.0421, 0.0955), (0.0415, 0.09), (0.0405, 0.012), (0.036, 0.006), (0.0, 0.0055)]
    body = lathe(outer + rim + inner, 64)
    handle = bezier([(0.043, 0.082, 0), (0.092, 0.088, 0), (0.092, 0.025, 0), (0.0425, 0.025, 0)], 28)
    body += sweep(handle, 0.0052, 12)
    coffee = lathe([(0.0412, 0.072), (0.0, 0.072)], 48)
    return {"mug": body, "coffee": coffee}


def pen_cup() -> Mesh:
    outer = [(0.0, 0.0), (0.043, 0.0), (0.045, 0.004), (0.045, 0.098), (0.0452, 0.1)]
    rim = arc(0.0012, 0, 180, (0.0443, 0.1), 6)
    inner = [(0.0435, 0.098), (0.0432, 0.01), (0.0, 0.008)]
    return lathe(outer + rim + inner, 56)


# ----------------------------------------------------------------------------
# Headphones
# ----------------------------------------------------------------------------

def headphones() -> dict[str, Mesh]:
    """Over-ear headphones sitting on their ear cups, headband arching up. Centred at the origin."""
    cup_r = 0.047
    plastic, cushion, metal = Mesh(), Mesh(), Mesh()
    for s in (-1, 1):
        x = s * 0.087
        cup = lathe([(0.0, 0.0), (0.04, 0.0), (0.047, 0.006), (0.047, 0.03), (0.04, 0.04), (0.0, 0.042)], 40)
        plastic += cup.transformed(compose(translate((x, cup_r, 0)), rotate((0, 0, 1), -90 * s)))
        pad = lathe([(0.0, 0.0), (0.05, 0.0), (0.054, 0.005), (0.054, 0.012), (0.05, 0.017), (0.036, 0.017), (0.032, 0.012),
                     (0.032, 0.004), (0.0, 0.004)], 40)
        cushion += pad.transformed(compose(translate((x, cup_r, 0)), rotate((0, 0, 1), 90 * s)))
        metal += ellipsoid((0.002, 0.014, 0.014), 10, 16).transformed(translate((s * 0.1296, cup_r, 0)))
    a = np.radians(np.linspace(0, 180, 40))
    plastic += sweep(np.stack([0.108 * np.cos(a), 0.059 + 0.10 * np.sin(a), 0 * a], 1), 0.0075, 12)
    inner = a[8:-8]
    cushion += sweep(np.stack([0.075 * np.cos(inner), 0.059 + 0.088 * np.sin(inner), 0 * inner], 1), 0.011, 12)
    return {"plastic": plastic, "cushion": cushion, "metal": metal}


# ----------------------------------------------------------------------------
# Chair
# ----------------------------------------------------------------------------

def office_chair() -> dict[str, Mesh]:
    """Swivel chair facing +z. Seat at 0.46 m."""
    fabric, plastic, chrome = Mesh(), Mesh(), Mesh()
    seat = slab_disc(0.235, 0.075, 0.032).transformed(compose(translate((0, 0.435, 0)), scale((1.0, 1.0, 0.95))))
    fabric += seat
    back = ellipsoid((0.225, 0.26, 0.04), 20, 40).transformed(compose(translate((0, 0.82, -0.215)), rotate((1, 0, 0), -8)))
    fabric += back
    plastic += tube((0, 0.43, 0), (0, 0.21, 0), 0.03, 16)
    plastic += lathe([(0.0, 0.2), (0.05, 0.2), (0.06, 0.215), (0.045, 0.235), (0.0, 0.235)], 24)
    plastic += tube((0, 0.52, -0.14), (0, 0.70, -0.2), 0.012, 10)
    for k in range(5):
        a = np.radians(72 * k + 18)
        d = np.array([np.cos(a), 0, np.sin(a)])
        foot = d * 0.30
        pts = [np.array([0, 0.22, 0]) + d * 0.02, np.array([0, 0.19, 0]) + d * 0.15, np.array([0, 0.075, 0]) + foot]
        plastic += sweep(np.array(pts), [0.017, 0.014, 0.011], 10)
        chrome += ellipsoid((0.022, 0.022, 0.022), 10, 16).transformed(translate(np.array([0, 0.035, 0]) + foot))
    for s in (-1, 1):
        arm = bezier([(s * 0.235, 0.52, 0.02), (s * 0.25, 0.64, 0.02), (s * 0.25, 0.68, -0.08), (s * 0.245, 0.66, -0.17)], 16)
        plastic += sweep(arm, 0.012, 8)
        plastic += tube((s * 0.235, 0.45, 0.02), (s * 0.235, 0.54, 0.02), 0.011, 8)
    return {"fabric": fabric, "plastic": plastic, "chrome": chrome}


# ----------------------------------------------------------------------------
# Paper and notebook (uv v = 0 at the top edge, which is the -z edge)
# ----------------------------------------------------------------------------

def paper_sheet(width: float, depth: float, curl: float = 0.004, seed: int = 0, rows: int = 28, cols: int = 20) -> Mesh:
    """A sheet lying flat (xz plane, faces +y), slightly warped at the corners and edges."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-width / 2, width / 2, cols)[None, :]
    z = np.linspace(-depth / 2, depth / 2, rows)[:, None]
    nx, nz = x / (width / 2), z / (depth / 2)
    ph = rng.uniform(0, 6.28)
    y = curl * (np.abs(nx) ** 3 + 0.6 * np.abs(nz) ** 3 * (nz > 0) + 0.3 * np.sin(2.5 * nz + ph) * np.abs(nx)) \
        + 0.0006 * np.sin(8 * nx + ph)
    p = np.stack(np.broadcast_arrays(x, y, z), -1).reshape(-1, 3)
    f = G._grid_faces(rows, cols)
    u = np.broadcast_to((x / width + 0.5), (rows, cols))
    v = np.broadcast_to((z / depth + 0.5), (rows, cols))
    return Mesh(p, G.vertex_normals(p, f), np.stack([u, v], -1).reshape(-1, 2), f)


def notebook_pages(width: float = 0.42, depth: float = 0.297, rows: int = 40, cols: int = 90) -> dict[str, Mesh]:
    """Open notebook lying flat. 'pages' is the textured top surface (uv covers the whole spread);
    'edges' are the page-block sides; 'cover' is the hard cover beneath and around."""
    x = np.linspace(-width / 2, width / 2, cols)[None, :]
    z = np.linspace(-depth / 2, depth / 2, rows)[:, None]
    nx = np.abs(x) / (width / 2)
    y = 0.0145 - 0.0075 * np.exp(-(np.abs(x) / 0.022) ** 2) + 0.0042 * np.clip(nx - 0.55, 0, 1) ** 2 / 0.2 + 0.0012 * (z / (depth / 2)) ** 2
    p = np.stack(np.broadcast_arrays(x, y, z), -1).reshape(-1, 3)
    f = G._grid_faces(rows, cols)
    u = np.broadcast_to(x / width + 0.5, (rows, cols))
    v = np.broadcast_to(z / depth + 0.5, (rows, cols))
    pages = Mesh(p, G.vertex_normals(p, f), np.stack([u, v], -1).reshape(-1, 2), f)
    grid = p.reshape(rows, cols, 3)
    edges = Mesh()
    y0 = 0.004
    for ring, outward in ((grid[0, :, :], (0, 0, -1)), (grid[-1, :, :], (0, 0, 1)), (grid[:, 0, :], (-1, 0, 0)), (grid[:, -1, :], (1, 0, 0))):
        n_pts = len(ring)
        top = ring
        bot = ring.copy()
        bot[:, 1] = y0
        pts = np.vstack([top, bot])
        uv = np.stack([np.linspace(0, 1, n_pts).repeat(1), np.zeros(n_pts)], -1)
        uv = np.vstack([np.stack([np.linspace(0, 1, n_pts), np.zeros(n_pts)], -1), np.stack([np.linspace(0, 1, n_pts), np.ones(n_pts)], -1)])
        fa = np.array([[i, i + n_pts, i + 1] for i in range(n_pts - 1)] + [[i + 1, i + n_pts, i + n_pts + 1] for i in range(n_pts - 1)])
        m = Mesh(pts, np.tile(outward, (len(pts), 1)), uv, fa).oriented()
        edges += m
    cover = box((width + 0.012, 0.004, depth + 0.012), (0, 0.002, 0))
    return {"pages": pages, "edges": edges, "cover": cover}


# ----------------------------------------------------------------------------
# Books for the bookcase, one mesh, uv into the spine atlas (16 x 4 cells; last cell is paper)
# ----------------------------------------------------------------------------

ATLAS_COLS, ATLAS_ROWS = 16, 4


def book(thick: float, height: float, depth: float, cell: tuple[int, int]) -> Mesh:
    """Standing book, spine facing +z, front of the spine at z = 0, resting on y = 0, centred in x."""
    c, r = cell
    u0, v0 = c / ATLAS_COLS, r / ATLAS_ROWS
    du, dv = 1 / ATLAS_COLS, 1 / ATLAS_ROWS
    cover_uv = [(u0 + 0.03 * du, v0 + 0.5 * dv)] * 4
    paper_uv = [((ATLAS_COLS - 1 + 0.5) / ATLAS_COLS, (ATLAS_ROWS - 1 + 0.5) / ATLAS_ROWS)] * 4
    t, d = thick / 2, depth
    m = Mesh()
    _quad(m, [(-t, 0, 0), (t, 0, 0), (t, height, 0), (-t, height, 0)], (0, 0, 1),
          [(u0, v0 + dv), (u0 + du, v0 + dv), (u0 + du, v0), (u0, v0)])          # spine, v = 0 at the top
    _quad(m, [(-t, 0, 0), (-t, 0, -d), (-t, height, -d), (-t, height, 0)], (-1, 0, 0), cover_uv)
    _quad(m, [(t, 0, -d), (t, 0, 0), (t, height, 0), (t, height, -d)], (1, 0, 0), cover_uv)
    _quad(m, [(-t, height, 0), (t, height, 0), (t, height, -d), (-t, height, -d)], (0, 1, 0), paper_uv)
    _quad(m, [(-t, 0, -d), (t, 0, -d), (t, height, -d), (-t, height, -d)], (0, 0, -1), paper_uv)
    return m.oriented()


def shelf_row(rng, x0: float, x1: float, y: float, z_front: float, max_h: float, gaps: float = 0.12) -> Mesh:
    """A row of books across [x0, x1] on a shelf at height y, spines flush at z_front, some leaning."""
    out = Mesh()
    x = x0 + 0.01
    while x < x1 - 0.06:
        if rng.random() < gaps * 0.5:
            x += rng.uniform(0.04, 0.16)
            continue
        th = rng.uniform(0.018, 0.052)
        h = rng.uniform(0.62, 1.0) * max_h
        dp = rng.uniform(0.15, 0.24)
        cell = (int(rng.integers(0, ATLAS_COLS)), int(rng.integers(0, ATLAS_ROWS)))
        if (cell == (ATLAS_COLS - 1, ATLAS_ROWS - 1)):
            cell = (0, 0)
        b = book(th, h, dp, cell)
        if x + th > x1:
            break
        out += b.transformed(translate((x + th / 2, y, z_front + rng.uniform(-0.012, 0.0))))
        x += th + rng.uniform(0.0, 0.004)
    if rng.random() < 0.7 and x < x1 - 0.2:   # a short leaning stack at the end
        lean = rng.uniform(8, 20)
        b = book(rng.uniform(0.02, 0.04), rng.uniform(0.7, 1.0) * max_h, 0.2, (int(rng.integers(0, ATLAS_COLS)), int(rng.integers(0, ATLAS_ROWS - 1))))
        out += b.transformed(compose(translate((x + 0.06, y, z_front)), rotate((0, 0, 1), lean)))
    return out
