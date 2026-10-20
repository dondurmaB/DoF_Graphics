"""Mesh recipes for the stream scene: the meandering channel, its valley terrain, and the rippled water.

Meters, y up, the stream runs along z (the hero cameras look toward -z, upstream). Pure numpy.
"""

from __future__ import annotations

import numpy as np

import procedural as G
from scenes import _nature_kit as N

WATER_Y = -0.15          # water surface (flat: a slow lowland stream with riffles, not a cascade)
BED_DEPTH = 0.45         # bed below the water at the channel centre
BANK = 0.7               # metres over which the bank rises from the water's edge


def centre_x(z):
    """Centre line of the channel: two meanders, so the stream bends out of sight upstream."""
    z = np.asarray(z, float)
    return 1.6 * np.sin(z / 8.5 + 0.5) + 0.6 * np.sin(z / 3.3 + 1.7) + 0.02 * z


def half_width(z):
    z = np.asarray(z, float)
    return 1.35 + 0.35 * np.sin(z / 6.0 + 0.8) + 0.15 * np.sin(z / 2.1)


class StreamTerrain(N.Terrain):
    """The valley floor of N.Terrain with the channel carved into it: a bed BED_DEPTH below the water at the
    centre, banks rising over BANK metres to just above the water, then the valley sides rising gently."""

    def height(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        valley = super().height(x, z)
        d = np.abs(x - centre_x(z)) - half_width(z)                 # <0 in the water, >0 on the banks
        t = np.clip((d + BANK) / (2 * BANK), 0.0, 1.0)                # 0 deep inside, 1 a bank-width out
        s = t * t * (3 - 2 * t)
        floor = (WATER_Y - BED_DEPTH) * (1 - s) + (WATER_Y + 0.12) * s
        side = 0.06 * np.maximum(d - 2.0, 0.0) + 0.012 * np.maximum(d - 2.0, 0.0) ** 1.5   # valley walls
        outside = WATER_Y + 0.12 + side + (valley - self.base) * np.clip((d - BANK) / 3, 0, 1)
        # The valley's undulation must not dip the banks below the water outside the channel: those hollows
        # would be dry ground below the waterline (mud-coloured, and with no water in them).
        return np.where(d < BANK, floor, np.maximum(outside, WATER_Y + SHORE_TOP + 0.02))


BED_TOP = -0.08          # mud below this, relative to the water: always under water, even in a ripple trough
SHORE_TOP = 0.04         # wet mud from BED_TOP up to this, relative to the water; litter or snow above


def _drop_unused(p, n, uv, f) -> G.Mesh:
    used, inv = np.unique(f, return_inverse=True)
    return G.Mesh(p[used], n[used], uv[used], inv.reshape(-1, 3))


def cut_at_height(mesh: G.Mesh, y: float) -> tuple[G.Mesh, G.Mesh]:
    """(below, above): the mesh cut exactly along the contour at height y. Triangles that straddle it are split
    where their edges cross y, so the boundary between the two parts is a smooth contour line, not the
    staircase of whole grid triangles a per-face split gives (visible as saw teeth where snow meets mud).
    Crossings are computed once per edge, so the two parts share their boundary vertices."""
    p, n, uv, f = mesh.p, mesh.n, mesh.uv, mesh.f
    below = p[:, 1] < y
    nb = below[f].sum(1)
    cut = (nb == 1) | (nb == 2)
    fc = f[cut]
    # rotate each cut face so its lone vertex (alone on its side of y) comes first; winding is kept
    lone_is_below = nb[cut] == 1
    lone = np.where(lone_is_below[:, None], below[fc], ~below[fc]).argmax(1)
    idx = (lone[:, None] + np.arange(3)[None, :]) % 3
    a, b_, c = (np.take_along_axis(fc, idx, 1)[:, k] for k in range(3))
    # one new vertex per crossed edge
    edges = np.concatenate([np.stack([a, b_], 1), np.stack([a, c], 1)])
    key = np.sort(edges, 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    e0, e1 = uniq[:, 0], uniq[:, 1]
    t = ((y - p[e0, 1]) / (p[e1, 1] - p[e0, 1]))[:, None]
    newp = p[e0] + t * (p[e1] - p[e0])
    newn = n[e0] + t * (n[e1] - n[e0])
    newn /= np.maximum(np.linalg.norm(newn, axis=1, keepdims=True), 1e-12)
    newuv = uv[e0] + t * (uv[e1] - uv[e0])
    base = len(p)
    ab, ac = base + inv[:len(a)], base + inv[len(a):]
    P, Nn, UV = np.concatenate([p, newp]), np.concatenate([n, newn]), np.concatenate([uv, newuv])
    tri = np.stack([a, ab, ac], 1)                                   # the lone vertex's side
    quad = np.concatenate([np.stack([ab, b_, c], 1), np.stack([ab, c, ac], 1)])
    lb2 = np.concatenate([lone_is_below, lone_is_below])
    lo = np.concatenate([f[nb == 3], tri[lone_is_below], quad[~lb2]])
    hi = np.concatenate([f[nb == 0], tri[~lone_is_below], quad[lb2]])
    return _drop_unused(P, Nn, UV, lo), _drop_unused(P, Nn, UV, hi)


def terrain_parts(terrain) -> dict[str, G.Mesh]:
    """{'bed', 'shore', 'bank'}: the terrain cut along two contours, BED_TOP and SHORE_TOP relative to the water.
    The only edge the camera sees at the water is where the water surface meets the shore (smooth); the mud of
    the bed starts under water."""
    m = terrain.mesh()
    bed, rest = cut_at_height(m, WATER_Y + BED_TOP)
    shore, bank = cut_at_height(rest, WATER_Y + SHORE_TOP)
    return {"bed": bed, "shore": shore, "bank": bank}


def water(z0, z1, seed, spacing=0.04, slope_rms=0.12) -> G.Mesh:
    """Rippled water over the channel from z0 to z1 (z0 < z1): one call to the pool scene's water_surface
    over the stream's bounding rectangle (one call, so the ripple normalisation is the same everywhere and
    there are no seams), clipped to the channel plus a margin the banks cover. slope_rms 0.12 is a stream
    with riffles: facets tilt enough that the sun glints off the water when the camera looks along it, not
    only at the mirror angle (a calm pool is ~0.035)."""
    zz = np.linspace(z0, z1, 400)
    lo = float((centre_x(zz) - half_width(zz)).min()) - BANK
    hi = float((centre_x(zz) + half_width(zz)).max()) + BANK
    mask = lambda x, z: np.abs(x - centre_x(z)) < half_width(z) + BANK * 0.6
    return N.water_sheet(lo, hi, z0, z1, WATER_Y, seed, spacing=spacing, slope_rms=slope_rms, waves=40, mask=mask)
