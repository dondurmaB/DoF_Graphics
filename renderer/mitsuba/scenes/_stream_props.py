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
        return np.where(d < BANK, floor, WATER_Y + 0.12 + side + (valley - self.base) * np.clip((d - BANK) / 3, 0, 1))


def split_by_height(mesh: G.Mesh, y_split: float) -> dict[str, G.Mesh]:
    """{'bed': faces whose centroid is below y_split, 'bank': the rest} with unused vertices dropped."""
    c = mesh.p[mesh.f].mean(1)[:, 1]
    out = {}
    for name, keep in (("bed", c < y_split), ("bank", c >= y_split)):
        used, inv = np.unique(mesh.f[keep], return_inverse=True)
        out[name] = G.Mesh(mesh.p[used], mesh.n[used], mesh.uv[used], inv.reshape(-1, 3))
    return out


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
