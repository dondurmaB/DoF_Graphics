"""Mesh recipes for every prop in the cafe, built from procedural primitives.

Each function returns a Mesh (or a dict of Meshes, one per material) in the
prop's local frame: base at y = 0, centred on the y axis, front facing +z.
Glassware is modelled as a closed shell with real wall thickness so the
dielectric refracts in and out correctly.
"""

from __future__ import annotations

import numpy as np

from procedural import (Mesh, box, compose, ellipsoid, lathe, lathe_parts, leaf, rotate,
                        scale, sweep, translate)


def bezier(control, n: int = 32) -> np.ndarray:
    """Evaluate a Bezier curve of any degree (de Casteljau)."""
    pts = np.asarray(control, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)[:, None, None]
    work = np.broadcast_to(pts, (n,) + pts.shape).copy()
    while work.shape[1] > 1:
        work = (1 - t) * work[:, :-1] + t * work[:, 1:]
    return work[:, 0]


def arc(radius: float, start: float, stop: float, centre=(0.0, 0.0), n: int = 8):
    """Profile points on a circular arc, angles in degrees."""
    a = np.radians(np.linspace(start, stop, n))
    return [(centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)) for t in a]


# --------------------------------------------------------------------------
# Furniture
# --------------------------------------------------------------------------

def slab_disc(radius: float, thickness: float, edge: float = 0.008) -> Mesh:
    """A disc with rounded edges (table tops, seats); uv is planar."""
    profile = ([(0.0, 0.0)] + arc(edge, -90, 0, (radius - edge, edge))
               + arc(edge, 0, 90, (radius - edge, thickness - edge)) + [(0.0, thickness)])
    return lathe(profile, 96, planar_uv=2 * radius)


def table_base(height: float = 0.72) -> Mesh:
    foot = [(0.0, 0.0), (0.23, 0.0), (0.23, 0.012)]
    body = [(0.225, 0.02), (0.20, 0.027), (0.07, 0.04), (0.04, 0.065), (0.03, 0.1),
            (0.027, height - 0.08), (0.032, height - 0.05), (0.06, height - 0.03),
            (0.13, height - 0.022)]
    top = [(0.13, height - 0.022), (0.13, height), (0.0, height)]
    return lathe_parts([foot + body[:1], body, top], 64)


def bistro_chair() -> dict[str, Mesh]:
    """Bentwood cafe chair: tube frame plus a seat disc. Front faces +z."""
    seat_h = 0.46
    frame = Mesh()
    leg_top, leg_foot = 0.15, 0.21
    for ang in np.radians([45, 135, 225, 315]):
        c, s = np.cos(ang), np.sin(ang)
        pts = bezier([(leg_top * c, seat_h - 0.04, leg_top * s),
                      (0.19 * c, 0.22, 0.19 * s),
                      (leg_foot * c, 0.0, leg_foot * s)], 20)
        frame += sweep(pts, np.linspace(0.0135, 0.011, len(pts)), 14)
    ring = [(0.183 * np.cos(a), 0.2, 0.183 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
    frame += sweep(ring, 0.008, 10, closed=True)
    apron = [(0.19 * np.cos(a), seat_h - 0.045, 0.19 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)]
    frame += sweep(apron, 0.013, 12, closed=True)

    def hoop(width, top, depth, start_y, n=48):
        s = np.linspace(0.0, 1.0, n)
        rise = np.sin(np.pi * s)
        x = -width * np.cos(np.pi * s)
        y = start_y + (top - start_y) * rise ** 0.35
        z = -0.13 - depth * rise ** 0.5
        return np.stack([x, y, z], 1)

    frame += sweep(hoop(0.16, 0.90, 0.08, seat_h - 0.05), 0.013, 14)
    frame += sweep(hoop(0.10, 0.80, 0.075, seat_h + 0.06), 0.009, 12)
    seat = slab_disc(0.205, 0.028, 0.01).transformed(translate((0, seat_h - 0.03, 0)))
    return {"frame": frame, "seat": seat}


# --------------------------------------------------------------------------
# Tableware
# --------------------------------------------------------------------------

CUP_LIQUID_Y = 0.058


def cup() -> dict[str, Mesh]:
    outer = [(0.0, 0.0), (0.024, 0.0), (0.027, 0.003), (0.03, 0.008), (0.036, 0.02),
             (0.041, 0.035), (0.044, 0.05), (0.045, 0.065), (0.0452, 0.07)]
    rim = arc(0.0012, 0, 180, (0.0441, 0.0705), 6)
    inner = [(0.0418, 0.065), (0.041, 0.05), (0.038, 0.035), (0.033, 0.02), (0.026, 0.011),
             (0.015, 0.008), (0.0, 0.0075)]
    body = lathe(outer + rim + inner, 64)
    handle = bezier([(0.043, 0.058, 0), (0.078, 0.066, 0), (0.08, 0.02, 0), (0.037, 0.022, 0)], 28)
    body += sweep(handle, 0.0045, 12)
    coffee = lathe([(0.043, CUP_LIQUID_Y), (0.0, CUP_LIQUID_Y)], 48)
    return {"cup": body, "coffee": coffee}


def saucer() -> Mesh:
    return lathe([(0.0, 0.0), (0.045, 0.0), (0.05, 0.004), (0.075, 0.012), (0.08, 0.017)]
                 + arc(0.0015, 0, 180, (0.0785, 0.0172), 5)
                 + [(0.072, 0.015), (0.05, 0.0085), (0.03, 0.0075), (0.0, 0.0075)], 64)


def plate(radius: float = 0.11) -> Mesh:
    k = radius / 0.11
    pts = [(0.0, 0.0), (0.06, 0.0), (0.064, 0.004), (0.1, 0.012), (0.11, 0.018)]
    pts = [(r * k, y) for r, y in pts]
    return lathe(pts + arc(0.0016, 0, 180, (0.1085 * k, 0.0185), 5)
                 + [(0.1 * k, 0.0155), (0.064 * k, 0.0085), (0.0, 0.0085)], 72)


def bowl(radius: float = 0.1, height: float = 0.07, wall: float = 0.005) -> Mesh:
    a = np.linspace(-80, 0, 12)
    outer = [(0.0, 0.0), (radius * 0.45, 0.0)] + [
        (radius * np.cos(np.radians(t)), height * (1 + np.sin(np.radians(t)))) for t in a[3:]]
    inner = [((radius - wall) * np.cos(np.radians(t)), height * (1 + np.sin(np.radians(t))) + wall)
             for t in a[::-1] if (radius - wall) * np.cos(np.radians(t)) > 0.01]
    return lathe(outer + arc(wall / 2, 0, 180, (radius - wall / 2, height), 5) + inner[1:]
                 + [(0.0, 0.2 * height + wall)], 72)


def spoon() -> Mesh:
    handle = bezier([(0.0, 0.0, 0.0), (0.05, 0.006, 0.0), (0.105, 0.012, 0.0)], 20)
    m = sweep(handle, np.linspace(0.0032, 0.002, len(handle)), 10)
    m += ellipsoid((0.017, 0.0035, 0.011)).transformed(translate((-0.014, 0.0, 0.0)))
    return m


def teapot() -> dict[str, Mesh]:
    body = lathe([(0.0, 0.0), (0.05, 0.0), (0.056, 0.004), (0.062, 0.012), (0.075, 0.035),
                  (0.083, 0.06), (0.082, 0.085), (0.072, 0.108), (0.056, 0.121), (0.046, 0.126),
                  (0.0, 0.127)], 96)
    lid = lathe([(0.0, 0.123), (0.049, 0.123), (0.0495, 0.126), (0.044, 0.133), (0.03, 0.14),
                 (0.012, 0.144), (0.008, 0.147), (0.012, 0.151), (0.015, 0.156), (0.012, 0.162),
                 (0.005, 0.165), (0.0, 0.1655)], 72)
    spout = bezier([(-0.068, 0.035, 0), (-0.11, 0.04, 0), (-0.118, 0.085, 0), (-0.148, 0.113, 0)], 32)
    handle = bezier([(0.07, 0.1, 0), (0.135, 0.118, 0), (0.15, 0.035, 0), (0.078, 0.032, 0)], 32)
    body += sweep(spout, np.linspace(0.019, 0.0075, len(spout)), 20)
    body += sweep(handle, 0.009, 14)
    return {"body": body, "lid": lid}


def croissant() -> Mesh:
    s = np.linspace(0.0, 1.0, 64)
    ang = np.radians(-115 + 230 * s)
    path = np.stack([0.045 * np.cos(ang), 0.022 + 0.004 * np.sin(np.pi * s), 0.045 * np.sin(ang)], 1)
    radius = (0.006 + 0.021 * np.sin(np.pi * s) ** 0.8) * (1 + 0.07 * np.cos(2 * np.pi * 7 * s))
    m = sweep(path, radius, 24)
    return m.transformed(scale((1.0, 0.8, 1.0)))


# --------------------------------------------------------------------------
# Glassware (closed shells, walls a few mm thick)
# --------------------------------------------------------------------------

def wine_glass() -> Mesh:
    outer = [(0.0, 0.0), (0.034, 0.0)] + arc(0.0015, -90, 90, (0.034, 0.0015), 5) + [
        (0.02, 0.0045), (0.006, 0.008), (0.0038, 0.015), (0.0035, 0.07), (0.008, 0.08),
        (0.022, 0.09), (0.034, 0.108), (0.040, 0.135), (0.040, 0.16), (0.036, 0.19)]
    rim = arc(0.0008, 0, 180, (0.0352, 0.19), 5)
    inner = [(0.0385, 0.16), (0.0385, 0.135), (0.0325, 0.11), (0.021, 0.0925), (0.008, 0.087),
             (0.0, 0.086)]
    return lathe(outer + rim + inner, 96)


def tumbler(radius: float = 0.033, height: float = 0.095, wall: float = 0.0022, base: float = 0.011) -> Mesh:
    outer = [(0.0, 0.0), (radius - 0.003, 0.0)] + arc(0.003, -90, 0, (radius - 0.003, 0.003), 4)[1:] + [
        (radius + 0.002, height)]
    rim = arc(wall / 2, 0, 180, (radius + 0.002 - wall / 2, height), 5)
    inner = [(radius - wall, base + 0.004), (radius - wall - 0.004, base), (0.0, base)]
    return lathe(outer + rim + inner, 96)


def bottle() -> Mesh:
    outer = [(0.0, 0.0), (0.033, 0.0), (0.0365, 0.003), (0.0375, 0.012), (0.0375, 0.19),
             (0.031, 0.218), (0.016, 0.248), (0.0135, 0.29), (0.0148, 0.296), (0.0148, 0.305)]
    rim = arc(0.0022, 0, 180, (0.0126, 0.305), 5)
    inner = [(0.0103, 0.29), (0.0105, 0.25), (0.027, 0.218), (0.0345, 0.19), (0.0345, 0.014),
             (0.02, 0.009), (0.0, 0.012)]
    return lathe(outer + rim + inner, 72)


def jar(radius: float = 0.07, height: float = 0.19) -> dict[str, Mesh]:
    glass = tumbler(radius, height, 0.003, 0.008)
    lid = slab_disc(radius + 0.006, 0.022, 0.006).transformed(translate((0, height - 0.012, 0)))
    knob = lathe([(0.0, 0.0), (0.012, 0.0), (0.016, 0.01), (0.014, 0.02), (0.0, 0.024)], 32)
    lid += knob.transformed(translate((0, height + 0.01, 0)))
    return {"glass": glass, "lid": lid}


def cake_dome(radius: float = 0.14) -> dict[str, Mesh]:
    wall = 0.003
    outer = [(radius - wall, 0.0), (radius, 0.0)] + [
        (radius * np.cos(t), radius * 0.95 * np.sin(t)) for t in np.radians(np.linspace(0, 85, 18))[1:]]
    top = radius * 0.95
    knob = [(0.014, top + 0.001), (0.014, top + 0.012), (0.022, top + 0.02), (0.016, top + 0.032), (0.0, top + 0.034)]
    inner = [((radius - wall) * np.cos(t), (radius - wall) * 0.95 * np.sin(t))
             for t in np.radians(np.linspace(90, 0, 18))]
    dome = lathe_parts([outer + knob, inner], 96)
    stand = lathe([(0.0, 0.0), (0.06, 0.0), (0.062, 0.006), (0.02, 0.02), (0.016, 0.07), (0.03, 0.08),
                   (radius + 0.02, 0.085), (radius + 0.022, 0.092), (0.0, 0.092)], 72)
    cake = lathe_parts([[(0.0, 0.0), (radius - 0.03, 0.0), (radius - 0.03, 0.06)],
                        [(radius - 0.03, 0.06), (radius - 0.036, 0.068), (0.0, 0.068)]], 72)
    return {"dome": dome.transformed(translate((0, 0.092, 0))), "stand": stand,
            "cake": cake.transformed(translate((0, 0.092, 0)))}


def candle() -> dict[str, Mesh]:
    holder = tumbler(0.032, 0.07, 0.0025, 0.008)
    wax = lathe_parts([[(0.0, 0.008), (0.028, 0.008), (0.028, 0.04)],
                       [(0.028, 0.04), (0.025, 0.043), (0.0, 0.041)]], 48)
    wick = sweep([(0, 0.04, 0), (0, 0.047, 0), (0.001, 0.05, 0)], 0.0007, 6)
    flame = ellipsoid((0.0045, 0.012, 0.0045)).transformed(translate((0.0005, 0.061, 0)))
    return {"holder": holder, "wax": wax, "wick": wick, "flame": flame}


def vase_with_flowers(seed: int = 21) -> dict[str, Mesh]:
    rng = np.random.default_rng(seed)
    vase = lathe([(0.0, 0.0), (0.035, 0.0), (0.046, 0.02), (0.052, 0.06), (0.047, 0.11),
                  (0.028, 0.15), (0.017, 0.175), (0.016, 0.2), (0.021, 0.212)]
                 + arc(0.0025, 0, 180, (0.0185, 0.213), 5)
                 + [(0.013, 0.2), (0.013, 0.17), (0.0, 0.16)], 72)
    stems, petals, hearts, leaves = Mesh(), Mesh(), Mesh(), Mesh()
    for i in range(3):
        yaw = np.radians(120 * i + rng.uniform(-25, 25))
        lean = rng.uniform(0.04, 0.09)
        top = np.array([lean * np.cos(yaw), 0.36 + rng.uniform(0, 0.08), lean * np.sin(yaw)])
        path = bezier([(0, 0.12, 0), (0, 0.25, 0), top], 24)
        stems += sweep(path, 0.0022, 8)
        tilt = np.degrees(np.arctan2(lean, 0.2))
        head = compose(translate(top), rotate((-np.sin(yaw), 0, np.cos(yaw)), -tilt))
        for k in range(7):
            petal = leaf(0.038, 0.026, curl=-0.35, fold=0.15)
            petals += petal.transformed(compose(head, rotate((0, 1, 0), k * 360 / 7 + 10 * i),
                                                rotate((1, 0, 0), -38)))
        hearts += ellipsoid((0.009, 0.006, 0.009)).transformed(compose(head, translate((0, 0.004, 0))))
        lf = leaf(0.09, 0.028, curl=0.5, fold=0.4)
        mid = path[len(path) // 2]
        leaves += lf.transformed(compose(translate(mid), rotate((0, 1, 0), np.degrees(-yaw) + 60),
                                         rotate((1, 0, 0), -35)))
    return {"vase": vase, "stems": stems, "petals": petals, "hearts": hearts, "leaves": leaves}


def potted_plant(seed: int, stems: int, height: float, leaf_len: float, leaf_w: float,
                 leaves_per_stem: int, pot_radius: float, pot_height: float,
                 spread: float = 0.3) -> dict[str, Mesh]:
    rng = np.random.default_rng(seed)
    r, h = pot_radius, pot_height
    pot = lathe([(0.0, 0.0), (r * 0.72, 0.0), (r * 0.75, 0.01), (r, h - 0.02), (r + 0.012, h - 0.018),
                 (r + 0.012, h)] + arc(0.004, 0, 180, (r + 0.008, h), 5)
                + [(r - 0.006, h - 0.02), (r - 0.012, h - 0.04), (0.0, h - 0.04)], 64)
    soil = lathe([(r - 0.004, h - 0.03), (0.0, h - 0.025)], 32)
    stem_mesh, leaf_mesh = Mesh(), Mesh()
    base_y = h - 0.03
    for s in range(stems):
        yaw = rng.uniform(0, 2 * np.pi)
        reach = spread * rng.uniform(0.3, 1.0)
        top = np.array([reach * np.cos(yaw), base_y + height * rng.uniform(0.55, 1.0), reach * np.sin(yaw)])
        path = bezier([(0, base_y, 0), (0.2 * top[0], base_y + 0.5 * (top[1] - base_y), 0.2 * top[2]), top], 24)
        stem_mesh += sweep(path, np.linspace(0.006, 0.0025, len(path)) * (height / 0.6) ** 0.5, 8)
        for k in range(leaves_per_stem):
            t = 0.35 + 0.65 * (k + rng.uniform(0, 0.6)) / leaves_per_stem
            idx = min(int(t * (len(path) - 1)), len(path) - 1)
            size = rng.uniform(0.75, 1.15) * (0.6 + 0.4 * t)
            lf = leaf(leaf_len * size, leaf_w * size, curl=rng.uniform(0.2, 0.5), fold=rng.uniform(0.2, 0.6))
            leaf_mesh += lf.transformed(compose(translate(path[idx]),
                                                rotate((0, 1, 0), rng.uniform(0, 360)),
                                                rotate((1, 0, 0), -rng.uniform(15, 55)),
                                                rotate((0, 0, 1), rng.uniform(-20, 20))))
    return {"pot": pot, "soil": soil, "stems": stem_mesh, "leaves": leaf_mesh}


# --------------------------------------------------------------------------
# Lighting fixtures
# --------------------------------------------------------------------------

def dome_shade(radius: float = 0.17, height: float = 0.15, wall: float = 0.003) -> dict[str, Mesh]:
    """Enamel pendant shade hanging from its top at y = 0."""
    phi = np.linspace(np.pi / 2, 0.12, 20)
    outer = [(radius * np.sin(p), -height * (1 - np.cos(p))) for p in phi] + [
        (0.022, 0.0), (0.022, 0.05), (0.008, 0.052)]
    inner = [((radius - wall) * np.sin(p), -height * (1 - np.cos(p)) - wall) for p in phi[::-1]]
    rim = [inner[-1], (radius, -height)]
    return {"outer": lathe_parts([outer, rim], 72), "inner": lathe(inner, 72)}


def socket() -> Mesh:
    return lathe([(0.0, 0.0), (0.014, 0.0), (0.0145, 0.004), (0.0145, 0.045), (0.009, 0.05),
                  (0.0035, 0.055), (0.0, 0.056)], 32)


# --------------------------------------------------------------------------
# Counter equipment
# --------------------------------------------------------------------------

def espresso_machine() -> dict[str, Mesh]:
    chrome = box((0.72, 0.42, 0.48), (0, 0.29, 0))
    chrome += box((0.74, 0.03, 0.5), (0, 0.515, 0))
    chrome += box((0.64, 0.02, 0.18), (0, 0.045, 0.2))  # drip tray grille
    panel = box((0.7, 0.14, 0.01), (0, 0.4, 0.242))
    black = Mesh()
    for x in (-0.18, 0.18):
        chrome += lathe([(0.0, 0.0), (0.04, 0.0), (0.045, 0.02), (0.035, 0.05), (0.0, 0.05)], 32
                        ).transformed(translate((x, 0.2, 0.26)))
        handle = bezier([(x, 0.205, 0.28), (x, 0.2, 0.36), (x, 0.19, 0.42)], 12)
        black += sweep(handle, 0.011, 12)
    wand = bezier([(0.32, 0.36, 0.2), (0.35, 0.3, 0.26), (0.34, 0.14, 0.27)], 20)
    chrome += sweep(wand, 0.005, 10)
    feet = Mesh()
    for x in (-0.3, 0.3):
        for z in (-0.18, 0.18):
            feet += lathe([(0.0, 0.0), (0.02, 0.0), (0.02, 0.03), (0.0, 0.03)], 16).transformed(translate((x, 0, z)))
    black += feet
    black += box((0.66, 0.035, 0.2), (0, 0.03, 0.2))
    return {"chrome": chrome, "panel": panel, "black": black}


def frame_ring(radius: float, tube: float) -> Mesh:
    ring = [(radius * np.cos(a), radius * np.sin(a), 0.0) for a in np.linspace(0, 2 * np.pi, 128, endpoint=False)]
    return sweep(ring, tube, 16, closed=True)
