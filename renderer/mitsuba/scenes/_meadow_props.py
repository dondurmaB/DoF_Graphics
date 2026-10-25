"""Meadow geometry: the terrain with its pond hollow, the pond's water level and water surface, and the terrain cut
into meadow, wet shore and pond bed.

The meadow is a 0.6 m grid, too coarse for a shore line seen from 15 m (a ~27-sided polygon). The square around the
pond is therefore rebuilt on a 0.1 m grid and only that patch is cut along contours (with
`_stream_props.cut_at_height`, exact, so the shore line is smooth); the rest of the meadow is never cut, so a natural
dip elsewhere cannot turn into a patch of mud. The patch's border vertices take the coarse grid's heights along the
shared edges, so the seam has no T-junction cracks (pinholes would show sky as bright dots).
"""

from __future__ import annotations

import math

import numpy as np

import procedural as G
from scenes import _nature_kit as N
from scenes import _stream_props as SP

POND = (-6.5, -13.0)        # centre of the hollow
POND_R = 4.2                # radius of the hollow (Terrain bump); the water is ~5 m across
POND_DEPTH = 0.9            # hollow depth at the centre
RIM_CLEAR = 0.15            # the ground on the ring at POND_R stands at least this far above the water
BED_TOP = -0.08             # mud below this, relative to the water: always under water
SHORE_TOP = 0.05            # wet mud from BED_TOP up to this; meadow above
PATCH = 1.4 * POND_R        # half width of the fine square around the pond (snapped out to the coarse grid)
FINE = 0.1                  # grid spacing inside it

TERRAIN = N.Terrain(-125.0, 125.0, -155.0, 95.0, spacing=0.6,
                    waves=[(0.35, 47.0, 15.0, 0.4), (0.12, 19.0, 100.0, 1.3), (0.05, 7.1, 55.0, 2.2),
                           (0.02, 2.3, 140.0, 0.7)],
                    bumps=[(POND[0], POND[1], POND_R, -POND_DEPTH), (30.0, -95.0, 40.0, 3.5),
                           (-55.0, -70.0, 30.0, 2.0)],
                    tilt=(0.0, -0.02))


def _ring(r, n=720):
    a = np.linspace(0.0, math.tau, n, endpoint=False)
    return POND[0] + r * np.cos(a), POND[1] + r * np.sin(a)


def _water_level() -> float:
    """Water at RIM_CLEAR below the lowest point of the ring at POND_R, so the shore closes all the way round."""
    y = float(TERRAIN.height(*_ring(POND_R)).min()) - RIM_CLEAR
    depth = y - float(TERRAIN.height(*POND))
    assert depth > 0.3, f"pond only {depth:.2f} m deep"
    return y


WATER_Y = _water_level()


def _grid(lo, hi, h) -> np.ndarray:
    """The coordinates Terrain.mesh actually uses (linspace with a rounded count, so not exactly `h` apart)."""
    return np.linspace(lo, hi, int(round((hi - lo) / h)) + 1)


def _patch_box() -> tuple[float, float, float, float]:
    """The fine square: POND +- PATCH, widened to whole coarse cells (on the coarse grid's real lines)."""
    xs, zs = _grid(TERRAIN.x0, TERRAIN.x1, TERRAIN.spacing), _grid(TERRAIN.z0, TERRAIN.z1, TERRAIN.spacing)
    return (float(xs[xs <= POND[0] - PATCH].max()), float(xs[xs >= POND[0] + PATCH].min()),
            float(zs[zs <= POND[1] - PATCH].max()), float(zs[zs >= POND[1] + PATCH].min()))


def _coarse_edge_height(axis: int, v: float, t: np.ndarray) -> np.ndarray:
    """Height of the coarse mesh along its grid line {x = v} (axis 0) or {z = v} (axis 2) at positions t along it:
    linear between the coarse vertices, which is what the coarse triangles' shared edges are."""
    q = _grid(TERRAIN.z0, TERRAIN.z1, TERRAIN.spacing) if axis == 0 else _grid(TERRAIN.x0, TERRAIN.x1, TERRAIN.spacing)
    i = np.clip(np.searchsorted(q, t, side="right") - 1, 0, len(q) - 2)
    a, b = q[i], q[i + 1]
    f = (t - a) / (b - a)
    hv = lambda u: TERRAIN.height(np.full_like(u, v), u) if axis == 0 else TERRAIN.height(u, np.full_like(u, v))
    return (1 - f) * hv(a) + f * hv(b)


def terrain_parts() -> dict[str, G.Mesh]:
    """{'meadow', 'shore', 'bed'}: the coarse meadow without the pond square, plus the fine square cut along the
    BED_TOP and SHORE_TOP contours."""
    xa, xb, za, zb = _patch_box()
    m = TERRAIN.mesh()
    c = m.p[m.f].mean(1)
    outside = ~((c[:, 0] > xa) & (c[:, 0] < xb) & (c[:, 2] > za) & (c[:, 2] < zb))
    used, inv = np.unique(m.f[outside], return_inverse=True)
    coarse = G.Mesh(m.p[used], m.n[used], m.uv[used], inv.reshape(-1, 3))

    fine = N.Terrain(xa, xb, za, zb, spacing=FINE, waves=TERRAIN.waves, bumps=TERRAIN.bumps, tilt=TERRAIN.tilt,
                     base=TERRAIN.base).mesh()
    p = fine.p.copy()
    border = np.zeros(len(p), bool)
    for axis, along, lines in ((0, 2, (xa, xb)), (2, 0, (za, zb))):          # border vertices onto the coarse edges
        for v in lines:
            on = np.isclose(p[:, axis], v, atol=1e-9)
            p[on, axis] = v
            p[on, 1] = _coarse_edge_height(axis, v, p[on, along])
            border |= on
    assert p[border, 1].min() > WATER_Y + SHORE_TOP + 0.02, "the fine square's border would cross the shore band"
    fine = G.Mesh(p, fine.n, fine.uv, fine.f)
    bed, rest = SP.cut_at_height(fine, WATER_Y + BED_TOP)
    shore, above = SP.cut_at_height(rest, WATER_Y + SHORE_TOP)
    return {"meadow": N._concat([coarse, above]), "shore": shore, "bed": bed}


def water(seed: int) -> G.Mesh:
    """Rippled water over the hollow, clipped where the ground is below the water plus a margin the shore covers.
    slope_rms 0.05: a still pond stirred by a breeze, so the sun and the sky break into glints along it."""
    pad = POND_R
    mask = lambda x, z: TERRAIN.height(x, z) < WATER_Y + 0.08
    return N.water_sheet(POND[0] - pad, POND[0] + pad, POND[1] - pad, POND[1] + pad, WATER_Y, seed, spacing=0.03,
                         slope_rms=0.05, waves=32, mask=mask)


def shore_point(angle_deg: float) -> list[float]:
    """A point on the wet shore (2 cm above the water) in the given direction from the pond centre."""
    a = math.radians(angle_deg)
    rs = np.linspace(0.3, POND_R, 400)
    xs, zs = POND[0] + rs * math.cos(a), POND[1] + rs * math.sin(a)
    i = int(np.argmax(TERRAIN.height(xs, zs) > WATER_Y + 0.02))
    return [float(xs[i]), float(TERRAIN.height(xs[i], zs[i])), float(zs[i])]
