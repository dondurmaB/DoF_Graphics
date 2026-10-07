"""Facade geometry for night_street: real openings, recessed joinery, lit rooms, shopfronts and signs.

`build_facade(spec, rng)` returns ({material or emitter key: Mesh}, {emitter key: rgb radiance}, sign points)
in the facade's local frame: x along the facade (-w/2 .. w/2), y up, +z out of the wall toward the street,
wall plane at z = 0. Keys starting with "emit_" are emitters. The building's layout depends on spec["seed"]
only; which rooms are lit, their furnishing, shop types and sign colours depend on `rng` (the scene seed).

No glass sits in front of lit rooms: a delta-transmission pane would block light-sampling shadow rays and
turn every room light into caustic fireflies on the street. Unlit windows get an opaque dark reflective pane.
Shopfronts do have glass; their interiors are lit by large ceiling panels, which BSDF sampling finds easily.
"""

from __future__ import annotations

import math

import numpy as np

import procedural as G
from procedural import Mesh

from scenes._night_street_props import Groups, box, quad, seg_box, tube, unit, _font, text_width

FLOOR_H = 3.15
SHOP_H = 4.30                 # ground storey including the fascia zone
FASCIA = (3.18, 3.86)
SHOP_OPEN_Y = 3.02
BODY_DEPTH = 8.8             # interiors stay in front of this; the solid building body starts here
ROOM_L = 60.0                 # ceiling-lamp radiance; one 0.3 m box lights a room

ROOM_LIGHT = {"warm": (1.0, 0.70, 0.40), "neutral": (1.0, 0.85, 0.66), "cool": (0.80, 0.90, 1.0)}
SHOP_LIGHT = {"warm": (1.0, 0.78, 0.52), "cool": (0.86, 0.93, 1.0)}
NEON = {"red": (1.0, 0.06, 0.08), "pink": (1.0, 0.12, 0.55), "blue": (0.12, 0.42, 1.0), "green": (0.15, 1.0, 0.45),
        "amber": (1.0, 0.55, 0.06), "white": (0.95, 0.95, 1.0), "violet": (0.55, 0.15, 1.0)}
NEON_L, BOX_L, GOOSE_L, HALL_L, TV_L, SHOP_L, OFFICE_L = 55.0, 5.5, 160.0, 2.5, 3.0, 7.0, 6.5
WORDS = ["cafe", "bar", "pizza", "noodles", "books", "deli", "wine", "sushi", "hair", "music", "kebab", "bakery",
         "market", "tattoo", "repairs", "shoes", "laundry", "ramen", "tapas", "records", "hotel", "lounge", "dental"]
PRODUCTS = 16


def oq(p0, p1, p2, p3, normal, uv=None) -> Mesh:
    """Quad facing `normal`, whatever the vertex order."""
    m = quad(p0, p1, p2, p3, uv)
    if np.dot(m.n[0], normal) < 0:
        m = quad(p3, p2, p1, p0, None if uv is None else list(uv)[::-1])
    return m


def wall_quad(x0, x1, y0, y1, z, u0) -> Mesh:
    return oq((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (0, 0, 1),
              uv=[(x0 + u0, y0), (x1 + u0, y0), (x1 + u0, y1), (x0 + u0, y1)])


def wall_with_openings(x0, x1, y0, y1, holes, z=0.0, u0=0.0, normal=(0, 0, 1)) -> Mesh:
    """The rectangle minus axis-aligned holes, as a grid of quads (uv in meters, continuous across cells)."""
    xs = sorted({x0, x1, *[h[0] for h in holes], *[h[1] for h in holes]})
    ys = sorted({y0, y1, *[h[2] for h in holes], *[h[3] for h in holes]})
    xs = [x for x in xs if x0 <= x <= x1]
    ys = [y for y in ys if y0 <= y <= y1]
    out = Mesh()
    P, F, UV = [], [], []
    for xa, xb in zip(xs[:-1], xs[1:]):
        for ya, yb in zip(ys[:-1], ys[1:]):
            cx, cy = 0.5 * (xa + xb), 0.5 * (ya + yb)
            if any(h[0] <= cx <= h[1] and h[2] <= cy <= h[3] for h in holes):
                continue
            b = len(P)
            P += [(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)]
            UV += [(xa + u0, ya), (xb + u0, ya), (xb + u0, yb), (xa + u0, yb)]
            F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    if not P:
        return out
    P = np.array(P, float)
    m = Mesh(P, np.tile(normal, (len(P), 1)).astype(float), np.array(UV, float), np.array(F))
    return m.oriented()


def reveals(x0, x1, y0, y1, d, u0=0.0) -> list[Mesh]:
    """Inner faces of an opening from z = 0 back to z = -d."""
    return [oq((x0, y1, 0), (x1, y1, 0), (x1, y1, -d), (x0, y1, -d), (0, -1, 0)),
            oq((x0, y0, 0), (x1, y0, 0), (x1, y0, -d), (x0, y0, -d), (0, 1, 0)),
            oq((x0, y0, 0), (x0, y1, 0), (x0, y1, -d), (x0, y0, -d), (1, 0, 0)),
            oq((x1, y0, 0), (x1, y1, 0), (x1, y1, -d), (x1, y0, -d), (-1, 0, 0))]


def room_box(x0, x1, y0, y1, z0, z1) -> dict[str, Mesh]:
    """Inward-facing room faces between z0 (front, the window plane) and z1 (< z0, back)."""
    return {"back": oq((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1)),
            "left": oq((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (1, 0, 0)),
            "right": oq((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (-1, 0, 0)),
            "floor": oq((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, 1, 0)),
            "ceiling": oq((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1), (0, -1, 0))}


def frame_ring(x0, x1, y0, y1, z, face=0.06, depth=0.07) -> list[Mesh]:
    return [box((x1 - x0, face, depth), ((x0 + x1) / 2, y0 + face / 2, z)),
            box((x1 - x0, face, depth), ((x0 + x1) / 2, y1 - face / 2, z)),
            box((face, y1 - y0 - 2 * face, depth), (x0 + face / 2, (y0 + y1) / 2, z)),
            box((face, y1 - y0 - 2 * face, depth), (x1 - face / 2, (y0 + y1) / 2, z))]


def glyph_lines(word: str, x0: float, y_base: float, xh: float, z: float, track=0.18):
    glyphs = _font()
    out, x = [], x0
    for ch in word:
        if ch not in glyphs:
            x += xh * 0.6
            continue
        adv, strokes = glyphs[ch]
        for poly in strokes:
            out.append(np.array([(x + gx * xh * adv * 0.9, y_base + gy * xh, z) for gx, gy in poly], float))
        x += adv * xh * (1 + track)
    return out


def pleated(x0, x1, y0, y1, z, folds=7, amp=0.04) -> Mesh:
    nx, ny = folds * 6 + 1, 2
    xs = np.linspace(x0, x1, nx)
    zz = z + amp * np.sin(np.linspace(0, folds * 2 * math.pi, nx))
    P = np.array([(x, y, zv) for y in (y0, y1) for x, zv in zip(xs, zz)])
    f = G._grid_faces(ny, nx)
    n = G.vertex_normals(P, f)
    if n[:, 2].mean() < 0:
        f = f[:, ::-1]
        n = -n
    uv = P[:, [0, 1]]
    return Mesh(P, n, uv, f)


# ---------------------------------------------------------------------------
# Layout (building seed only)
# ---------------------------------------------------------------------------

def layout(spec: dict) -> dict:
    rng = np.random.default_rng(spec["seed"])
    w, storeys, style = spec["width"], spec["storeys"], spec["style"]
    parapet = float(rng.uniform(0.7, 1.3)) if style != "glass" else 0.5
    floor_h = FLOOR_H if style != "glass" else 3.6
    h = SHOP_H + storeys * floor_h + parapet
    # ground floor units between pilasters; one narrow residential entrance
    units, x = [], -w / 2 + 0.25
    entrance_at = int(rng.integers(0, max(1, int(w // 4.5))))
    k = 0
    while x < w / 2 - 0.25 - 1.5:
        if k == entrance_at and style != "glass":
            uw = 1.7
            kind = "entrance"
        else:
            uw = float(rng.uniform(3.4, 6.0))
            kind = "shop"
        if w / 2 - 0.25 - (x + uw) < 2.6:
            uw = w / 2 - 0.25 - x
        units.append(dict(x0=x, x1=x + uw, kind=kind))
        x += uw + 0.35
        k += 1
    nb = max(2, int(round(w / spec.get("bay", 2.9))))
    bay = w / nb
    ww = float(rng.uniform(1.0, 1.35)) if style != "glass" else bay - 0.1
    wh = float(rng.uniform(1.55, 1.95)) if style != "glass" else floor_h - 1.0
    windows = []
    for f in range(storeys):
        base = SHOP_H + f * floor_h
        for b in range(nb):
            cx = -w / 2 + (b + 0.5) * bay
            y0 = base + (0.85 if style != "glass" else 0.95)
            windows.append(dict(x0=cx - ww / 2, x1=cx + ww / 2, y0=y0, y1=y0 + wh, floor=f, bay=b,
                                bx0=-w / 2 + b * bay, bx1=-w / 2 + (b + 1) * bay, base=base, top=base + floor_h))
    return dict(w=w, h=h, parapet=parapet, floor_h=floor_h, units=units, windows=windows, nb=nb, bay=bay,
                style=style, joinery=str(rng.choice(["sash", "casement", "georgian"])) if style != "glass" else "curtain",
                frame=str(rng.choice(["frame_white", "frame_dark", "frame_wood"])),
                reveal=0.22 if style in ("brick", "stone") else 0.16)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_facade(spec: dict, rng: np.random.Generator):
    lay = layout(spec)
    g = Groups()
    emit: dict[str, tuple] = {}
    signs: list = []
    w, h, style = lay["w"], lay["h"], lay["style"]
    wall = spec["wall"]
    trim = spec["trim"]
    u0 = spec.get("u0", 0.0)
    d = lay["reveal"]

    # --- wall with every opening cut out -----------------------------------
    holes = [(wd["x0"], wd["x1"], wd["y0"], wd["y1"]) for wd in lay["windows"]]
    for un in lay["units"]:
        if un["kind"] == "entrance":
            holes.append((un["x0"] + 0.25, un["x1"] - 0.25, 0.0, 2.9))
        else:
            holes.append((un["x0"], un["x1"], 0.0, SHOP_OPEN_Y))
    if style == "glass":
        g["wall_glass_frame"].add(wall_with_openings(-w / 2, w / 2, 0.0, h, holes, u0=u0))
    else:
        g[f"wall_{wall}"].add(wall_with_openings(-w / 2, w / 2, 0.0, h, holes, u0=u0))
    for hx0, hx1, hy0, hy1 in holes:
        for m in reveals(hx0, hx1, hy0, hy1, d if hy0 > 0.5 else 0.32):
            g["wall_glass_frame" if style == "glass" else f"wall_{wall}"].add(m)

    # --- horizontal articulation --------------------------------------------
    if style in ("brick", "stone", "stucco"):
        g[trim].add(box((w, 0.16, 0.30), (0.0, SHOP_H - 0.08 + 0.02, 0.15)))                       # shopfront cornice
        for f in range(1, spec["storeys"]):
            g[trim].add(box((w, 0.11, 0.06), (0.0, SHOP_H + f * lay["floor_h"] - 0.06, 0.03)))
        cy = h - lay["parapet"]
        g[trim].add(box((w + 0.1, 0.34, 0.42), (0.0, cy + 0.17, 0.21)))                              # main cornice
        g[trim].add(box((w + 0.02, 0.08, 0.32), (0.0, h + 0.04, 0.02)))                              # coping
        if style == "stone":
            for k in range(int(w / 0.28)):                                                            # dentils
                g[trim].add(box((0.12, 0.10, 0.10), (-w / 2 + 0.14 + k * 0.28, cy - 0.05, 0.05)))
    # roof slab, so a camera looking up never sees into the hollow behind the facade
    g["roof"].add(oq((-w / 2, h, 0.0), (w / 2, h, 0.0), (w / 2, h, -BODY_DEPTH), (-w / 2, h, -BODY_DEPTH), (0, 1, 0)))

    # --- ground floor ---------------------------------------------------------
    for un in lay["units"]:
        if un["kind"] == "entrance":
            entrance(g, emit, un, rng, spec)
        else:
            shopfront(g, emit, signs, un, rng, spec, lay)
    for un_a, un_b in zip(lay["units"][:-1], lay["units"][1:]):                                     # pilasters
        xp = 0.5 * (un_a["x1"] + un_b["x0"])
        g[trim if style != "glass" else "frame_alu"].add(box((0.42, SHOP_H - 0.2, 0.12), (xp, (SHOP_H - 0.2) / 2, 0.06)))

    # --- upper floors ---------------------------------------------------------
    if style == "glass":
        curtain_wall(g, emit, lay, rng)
    else:
        for wd in lay["windows"]:
            window(g, emit, wd, lay, rng, spec)
    if spec.get("balconies"):
        balconies(g, lay, rng)
    if spec.get("fire_escape"):
        fire_escape(g, lay)
    if style != "glass":
        # drainpipe with hopper head and brackets
        xd = w / 2 - 0.18
        g["drain"].add(tube([(xd, 0.0, 0.09), (xd, h - lay["parapet"] - 0.2, 0.09)], 0.045, 10))
        g["drain"].add(box((0.22, 0.22, 0.18), (xd, h - lay["parapet"] - 0.12, 0.10)))
        for y in np.arange(1.0, h - 1.0, 1.8):
            g["drain"].add(box((0.12, 0.035, 0.10), (xd, y, 0.05)))
    if spec.get("mansard"):
        mansard(g, emit, lay, rng)
    if spec.get("tank"):
        tank(g, w, h)
    for _ in range(int(rng.integers(0, 3))):                                                         # antennas
        x = float(rng.uniform(-w / 2 + 0.5, w / 2 - 0.5))
        top = h + float(rng.uniform(1.0, 2.6))
        g["drain"].add(tube([(x, h, -1.0), (x, top, -1.0)], 0.02, 6))
        for y in np.linspace(h + 0.6, top - 0.1, 3):
            g["drain"].add(tube([(x - 0.35, y, -1.0), (x + 0.35, y, -1.0)], 0.008, 4))
    return {k: v for k, v in g.meshes().items()}, emit, signs


# ---------------------------------------------------------------------------
# Upper-floor windows and rooms
# ---------------------------------------------------------------------------

def window(g: Groups, emit: dict, wd: dict, lay: dict, rng, spec) -> None:
    x0, x1, y0, y1 = wd["x0"], wd["x1"], wd["y0"], wd["y1"]
    d = lay["reveal"]
    xc = 0.5 * (x0 + x1)
    trim = spec["trim"]
    zf = -d + 0.05
    fr = lay["frame"]
    for m in frame_ring(x0, x1, y0, y1, zf):
        g[fr].add(m)
    if lay["joinery"] == "sash":
        g[fr].add(box((x1 - x0, 0.05, 0.08), (xc, 0.5 * (y0 + y1) + 0.1, zf + 0.01)))
        g[fr].add(box((0.03, y1 - y0, 0.03), (xc, 0.5 * (y0 + y1), zf)))
    elif lay["joinery"] == "casement":
        g[fr].add(box((0.05, y1 - y0, 0.07), (xc, 0.5 * (y0 + y1), zf)))
        g[fr].add(box((x1 - x0, 0.05, 0.07), (xc, y1 - 0.45, zf)))
    else:
        for t in (1 / 3, 2 / 3):
            g[fr].add(box((0.025, y1 - y0, 0.03), (x0 + t * (x1 - x0), 0.5 * (y0 + y1), zf)))
        for t in (0.25, 0.5, 0.75):
            g[fr].add(box((x1 - x0, 0.025, 0.03), (xc, y0 + t * (y1 - y0), zf)))
    # sill and lintel / surround
    g[trim].add(box((x1 - x0 + 0.18, 0.07, 0.12), (xc, y0 - 0.035, 0.03)))
    if lay["style"] == "brick":
        g[trim].add(box((x1 - x0 + 0.24, 0.20, 0.03), (xc, y1 + 0.10, 0.015)))
    elif lay["style"] == "stucco":
        for m in frame_ring(x0 - 0.10, x1 + 0.10, y0 - 0.02, y1 + 0.10, 0.02, face=0.10, depth=0.04):
            g[trim].add(m)
        if wd["floor"] == 0:
            g[trim].add(box((x1 - x0 + 0.5, 0.10, 0.14), (xc, y1 + 0.22, 0.07)))
    elif lay["style"] == "stone":
        g[trim].add(box((x1 - x0 + 0.30, 0.26, 0.05), (xc, y1 + 0.13, 0.025)))

    lit = rng.random() < 0.40 - 0.03 * wd["floor"]
    if not lit:
        g["glass_dark"].add(oq((x0, y0, zf - 0.04), (x1, y0, zf - 0.04), (x1, y1, zf - 0.04), (x0, y1, zf - 0.04), (0, 0, 1)))
        if rng.random() < 0.12 and lay["style"] != "stone":
            g["ac_unit"].add(box((0.78, 0.48, 0.34), (xc + rng.uniform(-0.2, 0.2), y0 - 0.42, 0.17)))
        return

    # a lit room: interior box, ceiling lamp, furniture, and one of curtains / sheer / blinds / nothing
    rx0, rx1 = wd["bx0"] + 0.08, wd["bx1"] - 0.08
    ry0, ry1 = wd["base"] + 0.02, wd["top"] - 0.32
    depth = float(rng.uniform(3.0, 5.0))
    zb = -d - depth
    walls = rng.choice(["room_a", "room_b", "room_c"])
    faces = room_box(rx0, rx1, ry0, ry1, -d, zb)
    for k, m in faces.items():
        g["room_floor" if k == "floor" else "room_ceiling" if k == "ceiling" else str(walls)].add(m)
    g[str(walls)].add(wall_with_openings(rx0, rx1, ry0, ry1, [(x0, x1, y0, y1)], z=-d, normal=(0, 0, -1)))
    cls = str(rng.choice(list(ROOM_LIGHT), p=[0.55, 0.3, 0.15]))
    if rng.random() < 0.75:
        lx, lz = rng.uniform(rx0 + 0.6, rx1 - 0.6), zb * rng.uniform(0.35, 0.6)
        g[f"emit_room_{cls}"].add(box((0.32, 0.05, 0.32), (lx, ry1 - 0.03, lz)))
        g["drain"].add(box((0.34, 0.02, 0.34), (lx, ry1 - 0.005, lz)))
    else:
        lx, lz = (rx0 + 0.45) if rng.random() < 0.5 else (rx1 - 0.45), zb + 0.5
        g["furn_dark"].add(tube([(lx, ry0, lz), (lx, ry0 + 1.45, lz)], 0.015, 6))
        g[f"emit_room_{cls}"].add(G.lathe([(0.12, ry0 + 1.45), (0.18, ry0 + 1.45), (0.12, ry0 + 1.72), (0.07, ry0 + 1.72)], 16)
                                  .transformed(G.translate((lx, 0, lz))))
    emit[f"emit_room_{cls}"] = tuple(ROOM_L * c for c in ROOM_LIGHT[cls])
    for _ in range(int(rng.integers(2, 5))):
        kind = rng.random()
        if kind < 0.35:                                   # sofa or sideboard against the back wall
            bw = rng.uniform(1.0, 1.9)
            g[str(rng.choice(["furn_dark", "furn_light", "furn_red"]))].add(
                box((bw, rng.uniform(0.6, 0.9), 0.8), (rng.uniform(rx0 + bw / 2, max(rx0 + bw / 2, rx1 - bw / 2)), ry0 + 0.4, zb + 0.42)))
        elif kind < 0.6:                                  # bookshelf
            g["shelf_books"].add(box((0.9, 1.9, 0.32), (rng.uniform(rx0 + 0.5, rx1 - 0.5), ry0 + 0.95, zb + 0.17)))
        elif kind < 0.8:                                  # picture on the back wall
            g["art"].add(box((rng.uniform(0.5, 1.0), rng.uniform(0.4, 0.7), 0.03), (rng.uniform(rx0 + 0.6, rx1 - 0.6), ry0 + 1.6, zb + 0.02)))
        else:                                             # television
            tx = rng.uniform(rx0 + 0.6, rx1 - 0.6)
            g["furn_dark"].add(box((0.95, 0.56, 0.05), (tx, ry0 + 1.1, zb + 0.03)))
            g["emit_tv"].add(oq((tx - 0.44, ry0 + 0.85, zb + 0.06), (tx + 0.44, ry0 + 0.85, zb + 0.06),
                                (tx + 0.44, ry0 + 1.35, zb + 0.06), (tx - 0.44, ry0 + 1.35, zb + 0.06), (0, 0, 1)))
            emit["emit_tv"] = tuple(TV_L * c for c in (0.45, 0.6, 1.0))
    dressing = rng.random()
    zc = -d - 0.06
    if dressing < 0.35:                                   # curtains drawn to the sides
        mat = f"curtain_{int(rng.integers(0, 4))}"
        cw = (x1 - x0) * rng.uniform(0.18, 0.4)
        g[mat].add(pleated(x0 - 0.15, x0 + cw, y0 - 0.1, y1 + 0.12, zc))
        g[mat].add(pleated(x1 - cw, x1 + 0.15, y0 - 0.1, y1 + 0.12, zc))
    elif dressing < 0.55:                                 # backlit sheer: glows in the light colour
        g[f"emit_sheer_{cls}"].add(pleated(x0, x1, y0, y1, zc - 0.02, folds=9, amp=0.02))
        emit[f"emit_sheer_{cls}"] = tuple(1.1 * c for c in ROOM_LIGHT[cls])
    elif dressing < 0.75:                                 # venetian blinds, partly lowered
        y_bot = y1 - (y1 - y0) * rng.uniform(0.35, 1.0)
        for y in np.arange(y1 - 0.04, y_bot, -0.07):
            g["blind"].add(box((x1 - x0 - 0.04, 0.008, 0.06), (xc, y, zc)).transformed(
                G.compose(G.translate((xc, y, zc)), G.rotate((1, 0, 0), 35.0), G.translate((-xc, -y, -zc)))))
        g["blind"].add(box((x1 - x0, 0.05, 0.07), (xc, y1 - 0.02, zc)))


def curtain_wall(g: Groups, emit: dict, lay: dict, rng) -> None:
    """Office floors behind a grid of mullions; some floors lit (ceiling light rows), the rest dark glass."""
    w = lay["w"]
    for f in range(lay["windows"][-1]["floor"] + 1):
        base = SHOP_H + f * lay["floor_h"]
        top = base + lay["floor_h"]
        lit_floor = rng.random() < 0.45
        lit_from, lit_to = (rng.integers(0, lay["nb"]), rng.integers(0, lay["nb"])) if lit_floor else (0, -1)
        lo, hi = min(lit_from, lit_to), max(lit_from, lit_to)
        g["spandrel"].add(oq((-w / 2, base, 0.02), (w / 2, base, 0.02), (w / 2, base + 0.95, 0.02), (-w / 2, base + 0.95, 0.02), (0, 0, 1)))
        for wd in [x for x in lay["windows"] if x["floor"] == f]:
            x0, x1, y0, y1 = wd["x0"], wd["x1"], wd["y0"], wd["y1"]
            if lit_floor and lo <= wd["bay"] <= hi:
                rx0, rx1 = wd["bx0"], wd["bx1"]
                faces = room_box(rx0, rx1, base + 0.05, top - 0.35, -lay["reveal"], -lay["reveal"] - 8.0)
                for k, m in faces.items():
                    g["office_floor" if k == "floor" else "room_ceiling" if k == "ceiling" else "room_b"].add(m)
                for zz in (-1.5, -3.6, -5.7):
                    g["emit_office"].add(box((rx1 - rx0 - 0.3, 0.03, 0.22), ((rx0 + rx1) / 2, top - 0.37, zz)))
                for zz in (-1.4, -3.4, -5.4):                                   # desk rows with monitors and chairs
                    g["furn_light"].add(box((rx1 - rx0 - 0.4, 0.04, 0.8), ((rx0 + rx1) / 2, base + 0.75, zz)))
                    for xm in np.linspace(rx0 + 0.45, rx1 - 0.45, 2):
                        g["furn_dark"].add(box((0.55, 0.34, 0.03), (xm, base + 1.02, zz - 0.25)))
                        if rng.random() < 0.6:
                            g["emit_screen"].add(oq((xm - 0.25, base + 0.87, zz - 0.233), (xm + 0.25, base + 0.87, zz - 0.233),
                                                    (xm + 0.25, base + 1.17, zz - 0.233), (xm - 0.25, base + 1.17, zz - 0.233), (0, 0, 1)))
                            emit["emit_screen"] = (1.0, 1.3, 1.8)
                        g["furn_dark"].add(box((0.48, 0.5, 0.48), (xm, base + 0.25, zz + 0.6)))
                    g["shop_wall_c"].add(box((rx1 - rx0, 1.3, 0.05), ((rx0 + rx1) / 2, base + 0.65, zz + 1.0)))  # partition
                if rng.random() < 0.5:
                    g["leaf_plant"].add(G.lathe([(0.0, 0.0), (0.22, 0.0), (0.3, 0.5), (0.0, 0.5)], 12).transformed(
                        G.translate((rx0 + 0.4, base + 0.05, -1.0))))
                    g["leaf_plant"].add(G.ellipsoid((0.4, 0.55, 0.4), 8, 12).transformed(G.translate((rx0 + 0.4, base + 1.0, -1.0))))
                emit["emit_office"] = tuple(OFFICE_L * c for c in (0.88, 0.94, 1.0))
            else:
                g["glass_curtain"].add(oq((x0, y0, -0.02), (x1, y0, -0.02), (x1, y1, -0.02), (x0, y1, -0.02), (0, 0, 1)))
    for k in range(lay["nb"] + 1):
        x = -w / 2 + k * lay["bay"]
        g["frame_alu"].add(box((0.07, lay["h"] - SHOP_H, 0.14), (x, SHOP_H + (lay["h"] - SHOP_H) / 2, 0.05)))


def balconies(g: Groups, lay: dict, rng) -> None:
    for wd in lay["windows"]:
        if wd["floor"] < 1 or rng.random() > 0.55:
            continue
        x0, x1 = wd["x0"] - 0.35, wd["x1"] + 0.35
        y = wd["base"] + 0.05
        g["stone_trim"].add(box((x1 - x0, 0.14, 0.95), ((x0 + x1) / 2, y, 0.47)))
        g["iron"].add(box((x1 - x0, 0.035, 0.035), ((x0 + x1) / 2, y + 1.0, 0.92)))
        for x in np.arange(x0 + 0.03, x1, 0.12):
            g["iron"].add(box((0.016, 0.93, 0.016), (x, y + 0.53, 0.92)))
        for zz in np.arange(0.1, 0.92, 0.12):
            for xs in (x0 + 0.02, x1 - 0.02):
                g["iron"].add(box((0.016, 0.93, 0.016), (xs, y + 0.53, zz)))
        g["iron"].add(box((0.035, 0.035, 0.9), (x0 + 0.02, y + 1.0, 0.47)))
        g["iron"].add(box((0.035, 0.035, 0.9), (x1 - 0.02, y + 1.0, 0.47)))


def fire_escape(g: Groups, lay: dict) -> None:
    """Platforms across two bays at every upper floor, railings, and switch-back stairs between them."""
    xa, xb = -lay["w"] / 2 + 0.3, -lay["w"] / 2 + 2 * lay["bay"] - 0.3
    depth = 1.15
    floors = sorted({wd["floor"] for wd in lay["windows"]})
    for f in floors:
        y = SHOP_H + f * lay["floor_h"] + 0.35
        g["iron"].add(box((xb - xa, 0.05, depth), ((xa + xb) / 2, y, depth / 2)))
        for x in np.arange(xa, xb + 0.01, 0.6):
            g["iron"].add(box((0.03, 0.06, depth), (x, y - 0.05, depth / 2)))
        g["iron"].add(box((xb - xa, 0.04, 0.04), ((xa + xb) / 2, y + 1.0, depth)))
        g["iron"].add(box((xb - xa, 0.03, 0.03), ((xa + xb) / 2, y + 0.5, depth)))
        for x in np.arange(xa, xb + 0.01, 0.15):
            g["iron"].add(box((0.014, 1.0, 0.014), (x, y + 0.5, depth)))
        for xs in (xa, xb):
            g["iron"].add(box((0.04, 1.0, 0.04), (xs, y + 0.5, depth)))
        if f > floors[0]:
            y_low = y - lay["floor_h"]
            s0 = np.array([xa + 0.4 if f % 2 else xb - 0.4, y_low + 0.05, depth * 0.55])
            s1 = np.array([xb - 0.9 if f % 2 else xa + 0.9, y - 0.02, depth * 0.55])
            for off in (-0.3, 0.3):
                g["iron"].add(seg_box(s0 + [0, 0, off], s1 + [0, 0, off], 0.04, 0.12))
            for t in np.linspace(0.05, 0.95, 14):
                p = s0 + (s1 - s0) * t
                g["iron"].add(box((0.22, 0.02, 0.62), tuple(p)))
            g["iron"].add(seg_box(s0 + [0, 0.9, 0.32], s1 + [0, 0.9, 0.32], 0.035, 0.035))
    y0 = SHOP_H + floors[0] * lay["floor_h"] + 0.35
    for xs in (xa + 0.9, xa + 1.3):                                   # retracted drop ladder
        g["iron"].add(box((0.04, 2.2, 0.04), (xs, y0 - 1.1, depth - 0.1)))
    for y in np.arange(y0 - 2.1, y0, 0.3):
        g["iron"].add(box((0.40, 0.025, 0.025), (xa + 1.1, y, depth - 0.1)))


def mansard(g: Groups, emit: dict, lay: dict, rng) -> None:
    w, h = lay["w"], lay["h"]
    g["slate"].add(oq((-w / 2, h, -0.15), (w / 2, h, -0.15), (w / 2, h + 2.6, -1.25), (-w / 2, h + 2.6, -1.25), (0.0, 0.39, 0.92)))
    g["slate"].add(oq((-w / 2, h + 2.6, -1.25), (w / 2, h + 2.6, -1.25), (w / 2, h + 2.9, -4.0), (-w / 2, h + 2.9, -4.0), (0, 1, 0)))
    for k in range(max(2, int(w // 3.2))):
        xc = -w / 2 + (k + 0.5) * w / max(2, int(w // 3.2))
        g["stucco_trim"].add(box((1.25, 1.55, 0.9), (xc, h + 1.0, -0.42)))
        g["stucco_trim"].add(box((1.45, 0.12, 1.05), (xc, h + 1.82, -0.40)))
        lit = rng.random() < 0.45
        g["emit_dormer" if lit else "glass_dark"].add(oq((xc - 0.42, h + 0.45, 0.04), (xc + 0.42, h + 0.45, 0.04),
                                                        (xc + 0.42, h + 1.5, 0.04), (xc - 0.42, h + 1.5, 0.04), (0, 0, 1)))
        if lit:
            emit["emit_dormer"] = tuple(1.6 * c for c in ROOM_LIGHT["warm"])
        for m in frame_ring(xc - 0.46, xc + 0.46, h + 0.41, h + 1.54, 0.06, face=0.05, depth=0.05):
            g["frame_white"].add(m)


def tank(g: Groups, w: float, h: float) -> None:
    x, z = -w / 4, -3.0
    for sx in (-0.7, 0.7):
        for sz in (-0.7, 0.7):
            g["iron"].add(box((0.08, 2.0, 0.08), (x + sx, h + 1.0, z + sz)))
    g["wood_tank"].add(G.lathe([(0.0, 0.0), (1.05, 0.0), (1.05, 2.4), (0.0, 2.4)], 28).transformed(G.translate((x, h + 2.0, z))))
    g["iron"].add(G.lathe([(0.0, 0.0), (1.12, 0.0), (0.0, 0.9)], 28).transformed(G.translate((x, h + 4.4, z))))
    for y in (0.4, 1.2, 2.0):
        g["iron"].add(G.lathe([(1.06, y), (1.08, y), (1.08, y + 0.05), (1.06, y + 0.05)], 28).transformed(G.translate((x, h + 2.0, z))))


# ---------------------------------------------------------------------------
# Ground floor
# ---------------------------------------------------------------------------

def entrance(g: Groups, emit: dict, un: dict, rng, spec) -> None:
    x0, x1 = un["x0"] + 0.25, un["x1"] - 0.25
    xc = 0.5 * (x0 + x1)
    d = 0.32
    door = f"door_{int(rng.integers(0, 4))}"
    g[door].add(box((x1 - x0, 2.42, 0.05), (xc, 1.21, -d + 0.03)))
    for yy in (0.55, 1.45):                                          # raised panels
        for xx in (xc - (x1 - x0) / 4, xc + (x1 - x0) / 4):
            g[door].add(box(((x1 - x0) / 2 - 0.14, 0.7, 0.02), (xx, yy, -d + 0.065)))
    g["rim_brass"].add(box((0.02, 0.18, 0.04), (x1 - 0.12, 1.05, -d + 0.08)))
    g[spec["trim"]].add(box((x1 - x0, 0.08, 0.10), (xc, 2.46, -d + 0.04)))
    g["emit_hall"].add(oq((x0, 2.50, -d), (x1, 2.50, -d), (x1, 2.86, -d), (x0, 2.86, -d), (0, 0, 1)))
    emit["emit_hall"] = tuple(HALL_L * c for c in ROOM_LIGHT["warm"])
    g["frame_dark"].add(box((0.04, 0.36, 0.04), (xc, 2.68, -d + 0.03)))
    g["ac_unit"].add(box((0.16, 0.24, 0.05), (x0 - 0.18, 1.35, 0.03)))           # intercom panel


def shopfront(g: Groups, emit: dict, signs: list, un: dict, rng, spec, lay) -> None:
    x0, x1 = un["x0"], un["x1"]
    xc, wu = 0.5 * (x0 + x1), x1 - x0
    d = 0.32
    kind = str(rng.choice(["shop", "cafe", "shop", "closed", "pharmacy"], p=[0.38, 0.24, 0.12, 0.14, 0.12]))
    fascia = f"fascia_{int(rng.integers(0, 6))}"
    zf = 0.14
    g[fascia].add(box((wu, FASCIA[1] - FASCIA[0], zf), (xc, 0.5 * sum(FASCIA), zf / 2)))
    if kind == "closed":
        _shutter(g, x0, x1)
        _sign(g, emit, signs, x0, x1, zf, rng, lit=False)
        return
    riser = 0.45
    zg = -d + 0.05
    g["riser"].add(box((wu, riser, 0.26), (xc, riser / 2, -d + 0.13)))
    door_left = rng.random() < 0.5
    dx0, dx1 = (x0 + 0.1, x0 + 1.15) if door_left else (x1 - 1.15, x1 - 0.1)
    frame = "frame_alu" if rng.random() < 0.6 else "frame_dark"
    # glazing: display window above the riser, glazed door, transom bar
    gx0, gx1 = (dx1 + 0.05, x1 - 0.05) if door_left else (x0 + 0.05, dx0 - 0.05)
    g["shop_glass"].add(oq((gx0, riser, zg), (gx1, riser, zg), (gx1, SHOP_OPEN_Y - 0.06, zg), (gx0, SHOP_OPEN_Y - 0.06, zg), (0, 0, 1)))
    g["shop_glass"].add(oq((dx0 + 0.06, 0.06, zg - 0.02), (dx1 - 0.06, 0.06, zg - 0.02), (dx1 - 0.06, 2.35, zg - 0.02),
                           (dx0 + 0.06, 2.35, zg - 0.02), (0, 0, 1)))
    for m in frame_ring(x0, x1, riser - 0.04, SHOP_OPEN_Y, zg, face=0.06, depth=0.08):
        g[frame].add(m)
    for m in frame_ring(dx0, dx1, 0.0, 2.41, zg - 0.01, face=0.06, depth=0.06):
        g[frame].add(m)
    g[frame].add(box((wu, 0.06, 0.08), (xc, 2.45, zg)))
    g["rim_brass"].add(box((0.03, 0.5, 0.03), ((dx1 - 0.15) if door_left else (dx0 + 0.15), 1.1, zg + 0.06)))
    # interior
    depth = float(rng.uniform(5.5, 7.5))
    zb = -d - depth
    ix0, ix1 = x0 - 0.1, x1 + 0.1
    faces = room_box(ix0, ix1, 0.0, 3.55, -d, zb)
    wallmat = "shop_wall_cool" if kind == "pharmacy" else str(rng.choice(["shop_wall_a", "shop_wall_b", "shop_wall_c"]))
    for k, m in faces.items():
        g["shop_floor" if k == "floor" else "room_ceiling" if k == "ceiling" else wallmat].add(m)
    g[wallmat].add(wall_with_openings(ix0, ix1, 0.0, 3.55, [(x0, x1, 0.0, SHOP_OPEN_Y)], z=-d, normal=(0, 0, -1)))
    cls = "cool" if kind == "pharmacy" or rng.random() < 0.3 else "warm"
    key = f"emit_shop_{cls}"
    for zz in np.arange(-d - 0.9, zb + 0.5, -1.5):
        for xx in np.arange(ix0 + 0.8, ix1 - 0.5, 1.5):
            g[key].add(box((1.1, 0.03, 0.55), (xx, 3.53, zz)))
    emit[key] = tuple(SHOP_L * c for c in SHOP_LIGHT[cls])
    if kind == "cafe":
        for zz in np.arange(-d - 1.2, zb + 1.5, -1.6):
            for xx in np.arange(ix0 + 0.8, ix1 - 0.6, 1.4):
                g["furn_wood"].add(G.lathe([(0.0, 0.74), (0.35, 0.74), (0.35, 0.77), (0.0, 0.77)], 20).transformed(G.translate((xx, 0, zz))))
                g["furn_dark"].add(tube([(xx, 0.0, zz), (xx, 0.74, zz)], 0.03, 6))
                for sx in (-0.55, 0.55):
                    g["furn_dark"].add(box((0.40, 0.45, 0.40), (xx + sx, 0.225, zz)))
                    g["furn_dark"].add(box((0.40, 0.45, 0.04), (xx + sx * 1.1, 0.67, zz)))
        g["furn_wood"].add(box((min(3.0, wu - 0.8), 1.05, 0.65), (xc, 0.525, zb + 0.9)))
        g["emit_display"].add(box((min(2.6, wu - 1.0), 0.03, 0.5), (xc, 1.06, zb + 0.9)))
        emit["emit_display"] = tuple(3.0 * c for c in SHOP_LIGHT["warm"])
        _shelves(g, ix0 + 0.4, ix1 - 0.4, zb + 0.18, rng, levels=3, y0=1.3)
    else:
        _shelves(g, ix0 + 0.4, ix1 - 0.4, zb + 0.18, rng, levels=5)
        for side, xs in ((1, ix0 + 0.2), (-1, ix1 - 0.2)):
            _side_shelves(g, xs, -d - 1.6, zb + 0.6, side, rng)
        if wu > 4.2:
            _gondola(g, xc, -d - 1.8, zb + 1.8, rng)
        g["furn_light"].add(box((1.4, 1.0, 0.6), (xc + (1.0 if door_left else -1.0), 0.5, -d - 1.2)))
    _sign(g, emit, signs, x0, x1, zf, rng, lit=True)
    if rng.random() < 0.3 and kind != "pharmacy":
        _awning(g, x0, x1, rng)
    if kind == "pharmacy" or rng.random() < 0.15:                     # projecting blade sign
        colour = "green" if kind == "pharmacy" else str(rng.choice(list(NEON)))
        xb = x1 - 0.2 if rng.random() < 0.5 else x0 + 0.2
        g["iron"].add(box((0.04, 0.04, 0.95), (xb, 3.95, 0.47)))
        for zs in (0.006, -0.006):
            n = (1, 0, 0) if zs > 0 else (-1, 0, 0)
            xx = xb + zs + (0.012 if zs > 0 else -0.012)
            g[f"emit_box_{colour}"].add(oq((xx, 3.3, 0.25), (xx, 3.3, 0.85), (xx, 3.85, 0.85), (xx, 3.3 + 0.55, 0.25), n))
        g["fascia_0"].add(box((0.024, 0.6, 0.65), (xb, 3.575, 0.55)))
        emit[f"emit_box_{colour}"] = tuple(BOX_L * c for c in NEON[colour])


def _shutter(g: Groups, x0, x1) -> None:
    nx, ny = 4, int(SHOP_OPEN_Y / 0.025)
    ys = np.linspace(0.0, SHOP_OPEN_Y - 0.05, ny)
    zz = -0.08 + 0.012 * np.sin(ys / 0.08 * 2 * math.pi)
    xs = np.linspace(x0, x1, nx)
    P = np.array([(x, y, z) for y, z in zip(ys, zz) for x in xs])
    f = G._grid_faces(ny, nx)
    n = G.vertex_normals(P, f)
    if n[:, 2].mean() < 0:
        f, n = f[:, ::-1], -n
    g["shutter"].add(Mesh(P, n, P[:, [0, 1]], f))
    g["shutter"].add(box((x1 - x0, 0.3, 0.3), (0.5 * (x0 + x1), SHOP_OPEN_Y - 0.1, -0.02)))


def _shelves(g: Groups, x0, x1, z_back, rng, levels=5, y0=0.15) -> None:
    for y in np.linspace(y0, 2.3, levels):
        g["shelf"].add(box((x1 - x0, 0.025, 0.42), ((x0 + x1) / 2, y, z_back + 0.21)))
        x = x0 + 0.02
        while x < x1 - 0.1:
            pw, ph, pd = rng.uniform(0.07, 0.22), rng.uniform(0.10, 0.32), rng.uniform(0.08, 0.3)
            if rng.random() < 0.88:
                m = box((pw, ph, pd), (x + pw / 2, y + 0.0125 + ph / 2, z_back + 0.05 + pd / 2))
                k = int(rng.integers(PRODUCTS))
                m.uv[:] = ((k + 0.5) / PRODUCTS, 0.5)
                g["products"].add(m)
            x += pw + rng.uniform(0.005, 0.03)
    for xs in (x0, x1):
        g["shelf"].add(box((0.03, 2.4, 0.44), (xs, 1.2, z_back + 0.21)))


def _side_shelves(g: Groups, x, z0, z1, side, rng) -> None:
    for y in np.linspace(0.15, 2.0, 4):
        g["shelf"].add(box((0.42, 0.025, abs(z1 - z0)), (x + side * 0.21, y, 0.5 * (z0 + z1))))
        z = max(z0, z1) - 0.02
        while z > min(z0, z1) + 0.1:
            pd_, ph = rng.uniform(0.07, 0.2), rng.uniform(0.1, 0.3)
            if rng.random() < 0.85:
                m = box((rng.uniform(0.1, 0.3), ph, pd_), (x + side * 0.2, y + 0.0125 + ph / 2, z - pd_ / 2))
                k = int(rng.integers(PRODUCTS))
                m.uv[:] = ((k + 0.5) / PRODUCTS, 0.5)
                g["products"].add(m)
            z -= pd_ + rng.uniform(0.005, 0.03)


def _gondola(g: Groups, x, z0, z1, rng) -> None:
    g["shelf"].add(box((0.06, 1.5, abs(z1 - z0)), (x, 0.75, 0.5 * (z0 + z1))))
    for side in (-1, 1):
        for y in (0.2, 0.65, 1.1):
            g["shelf"].add(box((0.38, 0.02, abs(z1 - z0)), (x + side * 0.22, y, 0.5 * (z0 + z1))))
            z = max(z0, z1)
            while z > min(z0, z1) + 0.1:
                pw = rng.uniform(0.07, 0.18)
                ph = rng.uniform(0.1, 0.28)
                m = box((0.3, ph, pw), (x + side * 0.22, y + 0.01 + ph / 2, z - pw / 2))
                k = int(rng.integers(PRODUCTS))
                m.uv[:] = ((k + 0.5) / PRODUCTS, 0.5)
                g["products"].add(m)
                z -= pw + 0.01


def _awning(g: Groups, x0, x1, rng) -> None:
    mat = f"awning_{int(rng.integers(0, 4))}"
    y_top, y_low, out = FASCIA[0] - 0.04, 2.45, 1.25
    g[mat].add(oq((x0 + 0.05, y_top, 0.16), (x1 - 0.05, y_top, 0.16), (x1 - 0.05, y_low, out), (x0 + 0.05, y_low, out), (0, 0.85, 0.5)))
    g[mat].add(oq((x0 + 0.05, y_low, out), (x1 - 0.05, y_low, out), (x1 - 0.05, y_low - 0.22, out), (x0 + 0.05, y_low - 0.22, out), (0, 0, 1)))
    g[mat].add(oq((x0 + 0.05, y_top, 0.16), (x1 - 0.05, y_top, 0.16), (x1 - 0.05, y_low, out), (x0 + 0.05, y_low, out), (0, -0.85, -0.5)))
    for xs in (x0 + 0.05, x1 - 0.05):
        g["iron"].add(seg_box((xs, y_top, 0.16), (xs, y_low, out), 0.02, 0.02))


def _sign(g: Groups, emit: dict, signs: list, x0, x1, zf, rng, lit: bool) -> None:
    word = str(rng.choice(WORDS))
    wu = x1 - x0
    xh = min(0.34, (wu - 0.6) / max(text_width(word, 1.0, 0.18), 1e-3))
    tw = text_width(word, xh, 0.18)
    xs = 0.5 * (x0 + x1) - tw / 2
    yb = 0.5 * sum(FASCIA) - xh * 0.55
    style = rng.random() if lit else 0.5
    signs.append([0.5 * (x0 + x1), 0.5 * sum(FASCIA), zf + 0.03])
    if lit and style < 0.42:                                           # neon tubes standing off the fascia
        colour = str(rng.choice(list(NEON)))
        for line in glyph_lines(word, xs, yb, xh, zf + 0.05):
            if len(line) > 1:
                g[f"emit_neon_{colour}"].add(tube(line, 0.011, 8))
                g["iron"].add(tube([line[0], line[0] - (0, 0, 0.05)], 0.004, 4))
        emit[f"emit_neon_{colour}"] = tuple(NEON_L * c for c in NEON[colour])
    elif lit and style < 0.70:                                         # backlit box with dark cut-out letters
        colour = str(rng.choice(["white", "amber", "red", "blue", "green"]))
        g[f"emit_box_{colour}"].add(oq((x0 + 0.08, FASCIA[0] + 0.06, zf + 0.002), (x1 - 0.08, FASCIA[0] + 0.06, zf + 0.002),
                                       (x1 - 0.08, FASCIA[1] - 0.06, zf + 0.002), (x0 + 0.08, FASCIA[1] - 0.06, zf + 0.002), (0, 0, 1)))
        for line in glyph_lines(word, xs, yb, xh, zf + 0.012):
            for p0, p1 in zip(line[:-1], line[1:]):
                g["letters_dark"].add(seg_box(p0, p1, 0.045, 0.012))
        emit[f"emit_box_{colour}"] = tuple(BOX_L * c for c in NEON[colour])
    else:                                                              # raised letters lit by goose-neck lamps
        mat = str(rng.choice(["letters_gold", "letters_white"]))
        for line in glyph_lines(word, xs, yb, xh, zf + 0.02):
            for p0, p1 in zip(line[:-1], line[1:]):
                g[mat].add(seg_box(p0, p1, 0.04, 0.03))
        if lit:
            for xg in np.linspace(x0 + 0.5, x1 - 0.5, max(2, int(wu // 1.6))):
                arm = [(xg, FASCIA[1] + 0.02, zf), (xg, FASCIA[1] + 0.20, zf + 0.12), (xg, FASCIA[1] + 0.16, zf + 0.42)]
                g["iron"].add(tube(arm, 0.012, 6))
                # shade opens along local -y; +53 deg about x aims it down and back at the letters
                aim = G.compose(G.translate((xg, FASCIA[1] + 0.16, zf + 0.42)), G.rotate((1, 0, 0), 53.0))
                g["iron"].add(G.lathe([(0.0, 0.0), (0.06, 0.0), (0.03, 0.08), (0.0, 0.09)], 12).transformed(aim))
                g["emit_goose"].add(G.lathe([(0.0, 0.012), (0.05, 0.012)], 12).transformed(aim))
            emit["emit_goose"] = tuple(GOOSE_L * c for c in (1.0, 0.8, 0.55))


# ---------------------------------------------------------------------------
# Wall textures: tiling, uv in meters (the scene scales by 1 / tile size)
# ---------------------------------------------------------------------------

WALLS = {
    "brick_red": ("brick", (0.46, 0.20, 0.13)), "brick_brown": ("brick", (0.34, 0.20, 0.14)),
    "brick_buff": ("brick", (0.62, 0.48, 0.32)), "stucco_cream": ("stucco", (0.78, 0.72, 0.60)),
    "stucco_pink": ("stucco", (0.76, 0.58, 0.54)), "stucco_sage": ("stucco", (0.55, 0.60, 0.52)),
    "stucco_ochre": ("stucco", (0.74, 0.58, 0.34)), "stone_beige": ("stone", (0.66, 0.60, 0.50)),
    "stone_grey": ("stone", (0.52, 0.52, 0.50)),
}
TILE = {"brick": (2.25, 1.95), "stucco": (3.0, 3.0), "stone": (3.6, 2.7)}


def wall_texture(name: str, px_per_m: int = 420) -> np.ndarray:
    kind, colour = WALLS[name]
    tw, th = TILE[kind]
    W, H = int(tw * px_per_m), int(th * px_per_m)
    seed = sum(map(ord, name))                      # stable across runs, unlike hash()
    rng = np.random.default_rng(seed)
    yy = (np.arange(H)[:, None] + 0.5) / px_per_m
    xx = (np.arange(W)[None, :] + 0.5) / px_per_m
    col = np.asarray(colour, np.float32)
    n = G.fbm(H, W, octaves=6, base=4, seed=seed)
    fine = G.fbm(H, W, octaves=3, base=120, seed=seed + 1)
    if kind == "brick":
        course = np.floor(yy / 0.075)
        bx = (xx + 0.1125 * (course % 2)) / 0.225
        brick = np.floor(bx)
        ncol, nrow = int(round(tw / 0.225)), int(round(th / 0.075))
        tone = rng.normal(0, 1, (nrow + 1, ncol + 2))[course.astype(int) % (nrow + 1), brick.astype(int) % (ncol + 2)]
        burnt = rng.random((nrow + 1, ncol + 2))[course.astype(int) % (nrow + 1), brick.astype(int) % (ncol + 2)] < 0.08
        fy, fx = (yy / 0.075) % 1.0, bx % 1.0
        mortar = (fy < 0.13) | (fx < 0.045)
        face = col * (1 + 0.16 * tone[..., None]) * (0.85 + 0.25 * n[..., None]) * (0.9 + 0.2 * fine[..., None])
        face = np.where(burnt[..., None], face * 0.55, face)
        img = np.where(mortar[..., None], np.array([0.30, 0.29, 0.27]) * (0.8 + 0.3 * fine[..., None]), face)
    elif kind == "stone":
        course = np.floor(yy / 0.45)
        bx = (xx + 0.45 * (course % 2)) / 0.9
        tone = rng.normal(0, 1, (8, 8))[course.astype(int) % 8, np.floor(bx).astype(int) % 8]
        joint = ((yy / 0.45) % 1.0 < 0.012) | (bx % 1.0 < 0.006)
        face = col * (1 + 0.06 * tone[..., None]) * (0.85 + 0.25 * n[..., None]) * (0.92 + 0.16 * fine[..., None])
        img = np.where(joint[..., None], face * 0.55, face)
    else:
        stain = G.fbm(H, W, octaves=5, base=3, seed=seed + 2)
        img = col * (0.9 + 0.12 * n[..., None]) * (0.95 + 0.1 * fine[..., None]) * (1 - 0.12 * np.clip(stain - 0.55, 0, 1)[..., None] * 4)
    return np.clip(img, 0, 1).astype(np.float32)


PRODUCT_COLOURS = [(0.85, 0.12, 0.10), (0.95, 0.75, 0.10), (0.10, 0.35, 0.80), (0.95, 0.95, 0.92), (0.15, 0.55, 0.25),
                   (0.90, 0.45, 0.10), (0.55, 0.15, 0.55), (0.05, 0.05, 0.06), (0.85, 0.55, 0.65), (0.30, 0.70, 0.85),
                   (0.70, 0.60, 0.40), (0.95, 0.30, 0.45), (0.45, 0.70, 0.20), (0.60, 0.62, 0.66), (0.25, 0.20, 0.55),
                   (0.98, 0.88, 0.60)]


def product_palette() -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(PRODUCT_COLOURS, np.float32)[None], 4, 0), 4, 1)


def book_spines(h: int = 256, w: int = 512, seed: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    x = 0
    while x < w:
        bw = int(rng.integers(6, 20))
        img[:, x:x + bw] = np.asarray(PRODUCT_COLOURS[int(rng.integers(len(PRODUCT_COLOURS)))]) * rng.uniform(0.35, 0.8)
        top = int(rng.integers(0, h // 4))
        img[:top, x:x + bw] = 0.05
        x += bw + 1
    img[(np.arange(h) % (h // 4)) < 4] = 0.25
    return np.clip(img, 0, 1)
