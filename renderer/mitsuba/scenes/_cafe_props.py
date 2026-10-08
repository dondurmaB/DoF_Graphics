"""Cafe helpers: uv-in-meters boxes for scanned textures, scanned-model placement, and the street outside.

The street is real geometry at real distances (depth stays correct through the windows):
our pavement under the windows, a granite kerb, an 8 m asphalt road, the far pavement, and a
row of 2-3 storey buildings 15.5 m from the window wall. Building height is capped so the
late-afternoon sun (elevation about 26 degrees, from over the road) still reaches the windows.
"""

from __future__ import annotations

import numpy as np

import procedural as G
import web_assets as W

NULL = {"type": "null"}          # scanned glass parts convert as opaque; hide them instead
FACADE_X = -19.4                 # face of the buildings across the road
ROAD = (-15.75, -7.6)            # x range of the asphalt
STREET_Z = (-75.0, 75.0)         # long enough that no window view sees an end
MAX_BUILDING_H = 8.3             # sun ray height at the facade line: taller would shade the windows


def box(b, centre, size, material, yaw: float = 0.0) -> None:
    """Like SceneBuilder.cube but with uv in meters, so W.surface textures keep their real scale."""
    size = tuple(float(s) for s in size)
    name = "x".join(f"{s:.3f}" for s in size)
    b.box_mesh(name, centre, size, material, yaw)


def put(b, model_id: str, at, yaw: float = 0.0, scale: float = 1.0, height: float | None = None,
        null_parts=(), prefix: str | None = None, overrides: dict | None = None):
    m = W.model(model_id)
    ov = dict(overrides or {})
    ov.update({p.name: NULL for p in m.parts if p.name in null_parts or p.name.endswith("_glass")})
    W.add(b, m, W.place(m, at=at, yaw=yaw, scale=scale, height=height), prefix=prefix, overrides=ov or None)
    return m


def street_materials(b) -> None:
    b.material("road", W.surface("asphalt_02"))
    b.material("pavement", W.surface("brick_pavement_02"))
    b.material("kerb", W.surface("granite_tile_04"))
    b.material("road_paint", {"type": "principled", "base_color": {"type": "rgb", "value": [0.78, 0.77, 0.72]},
                              "roughness": 0.7})
    b.material("win_glass", {"type": "principled", "base_color": {"type": "rgb", "value": [0.02, 0.025, 0.03]},
                             "roughness": 0.04, "specular": 0.8})
    b.material("win_frame_white", {"type": "principled", "base_color": {"type": "rgb", "value": [0.82, 0.81, 0.77]},
                                   "roughness": 0.45})
    b.material("win_frame_dark", {"type": "principled", "base_color": {"type": "rgb", "value": [0.06, 0.08, 0.07]},
                                  "roughness": 0.4})
    b.material("stone_trim", W.surface("beige_wall_001", tint=(1.05, 1.0, 0.92)))
    b.material("roof_edge", {"type": "principled", "base_color": {"type": "rgb", "value": [0.12, 0.11, 0.1]},
                             "roughness": 0.6})
    facades = [("brick_wall_001", None), ("white_stucco", (1.0, 0.86, 0.66)), ("painted_brick", None),
               ("red_brick_plaster_patch_02", None), ("plastered_wall_02", (0.95, 0.9, 0.8)),
               ("white_stucco", (0.78, 0.86, 0.92))]
    for i, (tex, tint) in enumerate(facades):
        b.material(f"facade_{i}", W.surface(tex, tint=tint))
    return len(facades)


def _window(b, x, y, z, w, h, frame: str) -> None:
    """A window unit standing slightly proud of a facade at x (facing +x): glass, frame, mullion, sill."""
    box(b, (x + 0.03, y + h / 2, z), (0.04, h, w), "win_glass")
    f = 0.07
    for dz in (-w / 2, w / 2):
        box(b, (x + 0.06, y + h / 2, z + dz), (0.08, h + f, f), frame)
    for dy in (0.0, h, h * 0.62):
        box(b, (x + 0.06, y + dy, z), (0.08, f if dy != h * 0.62 else 0.045, w), frame)
    box(b, (x + 0.06, y + h / 2, z), (0.08, h, 0.045), frame)
    box(b, (x + 0.12, y - 0.06, z), (0.2, 0.06, w + 0.16), "stone_trim")


def _shopfront(b, x, z, w, rng, frame: str) -> None:
    """Ground-floor shop window with a fascia board, or a closed roller shutter."""
    if rng.random() < 0.3:
        m = W.model("rollershutter_window_02")
        (x0, _, _), (x1, _, _) = m.bounds
        put(b, "rollershutter_window_02", at=(x + 0.09, 0.05, z), yaw=90, scale=min(w / (x1 - x0), 1.0) * 0.95,
            null_parts=("rollershutter_window_02_graffiti",) if rng.random() < 0.7 else ())
        return
    box(b, (x + 0.03, 1.45, z), (0.04, 2.3, w - 0.3), "win_glass")
    for dz in (-(w - 0.3) / 2, 0.0, (w - 0.3) / 2):
        box(b, (x + 0.06, 1.45, z + dz), (0.08, 2.3, 0.08), frame)
    box(b, (x + 0.06, 0.3, z), (0.1, 0.6, w - 0.2), frame)
    box(b, (x + 0.1, 2.85, z), (0.12, 0.5, w), "win_frame_dark")


def _building(b, z0, z1, h, mat, rng) -> None:
    depth = 10.0
    box(b, (FACADE_X - depth / 2, h / 2 - 0.05, 0.5 * (z0 + z1)), (depth, h + 0.1, z1 - z0), mat)
    frame = "win_frame_white" if rng.random() < 0.6 else "win_frame_dark"
    w = z1 - z0
    n = max(1, int((w - 0.8) // 2.6))
    zs = np.linspace(z0, z1, n + 2)[1:-1]
    for zc in zs:
        _shopfront(b, FACADE_X, zc, min(2.4, w / n - 0.2), rng, frame)
    floors = int((h - 3.4) // 2.9)
    for k in range(floors):
        y = 3.6 + 2.9 * k + 0.55
        for zc in zs:
            _window(b, FACADE_X, y, zc, 1.05, 1.55, frame)
    box(b, (FACADE_X + 0.12, 3.25, 0.5 * (z0 + z1)), (0.24, 0.16, w), "stone_trim")       # string course
    box(b, (FACADE_X + 0.15, h - 0.12, 0.5 * (z0 + z1)), (0.3, 0.24, w + 0.02), "roof_edge")  # cornice


def add_street(b, rng) -> None:
    n_fac = street_materials(b)
    zc, zl = 0.5 * sum(STREET_Z), STREET_Z[1] - STREET_Z[0]
    box(b, (-5.75, -0.15, zc), (3.7, 0.2, zl), "pavement")                       # our pavement, top -0.05
    box(b, (-7.675, -0.16, zc), (0.15, 0.22, zl), "kerb")
    box(b, (0.5 * sum(ROAD), -0.27, zc), (ROAD[1] - ROAD[0], 0.2, zl), "road")  # road top -0.17
    box(b, (-15.825, -0.16, zc), (0.15, 0.22, zl), "kerb")
    box(b, (-17.65, -0.15, zc), (3.5, 0.2, zl), "pavement")
    for z in np.arange(STREET_Z[0], STREET_Z[1], 6.0):                           # dashed centre line
        box(b, (0.5 * sum(ROAD), -0.168, z), (0.12, 0.004, 3.0), "road_paint")
    for x in (ROAD[1] - 0.25, ROAD[0] + 0.25):                                   # edge lines
        box(b, (x, -0.168, zc), (0.1, 0.004, zl), "road_paint")

    z = STREET_Z[0]
    while z < STREET_Z[1]:
        w = float(rng.uniform(7.0, 12.5))
        h = float(rng.uniform(6.6, MAX_BUILDING_H))
        _building(b, z, z + w, h, f"facade_{int(rng.integers(n_fac))}", rng)
        z += w

    # Street furniture.
    for z in (-13.0, 1.0, 15.0):
        put(b, "street_lamp_01", at=(-7.25, -0.05, z), yaw=90)
    for z in (-15.0, 6.0, 24.0):
        put(b, "tree_small_02", at=(-17.0, -0.05, z), yaw=float(rng.uniform(0, 360)), scale=1.25)
    put(b, "covered_car", at=(-14.6, -0.17, -8.0 + float(rng.uniform(-2, 2))), yaw=float(rng.choice([0.0, 180.0])))
    put(b, "water_manhole_cover", at=(-10.6, -0.175, -2.5))
    put(b, "water_manhole_cover", at=(-12.8, -0.175, 9.0))
    put(b, "fire_hydrant", at=(-15.3, -0.05, 2.5), null_parts=("fire_hydrant_aged",))
    put(b, "trashbag", at=(-7.2, -0.05, 6.5), yaw=40)
    put(b, "trashbag", at=(-7.0, -0.05, 7.0), yaw=-20, scale=0.85)
    for z in (-6.0, 11.5):
        put(b, "planter_box_02", at=(-18.9, -0.05, z), yaw=90)
    # A small terrace under our windows and the A-board by the door side.
    for z, yaw in ((-4.45, 0.0), (-1.55, 180.0)):
        put(b, "outdoor_table_chair_set_01", at=(-5.3, -0.05, z), yaw=yaw)
    put(b, "standing_chalkboard_01", at=(-4.55, -0.05, 0.6), yaw=90, scale=0.65)
    b.focus_points["facade"] = [FACADE_X + 0.1, 4.6, -1.5]
    b.focus_points["street_lamp"] = [-7.25, 3.4, 1.0]
    b.focus_points["terrace"] = [-5.3, 0.75, -1.55]
