"""Dry riverbed geometry: a wide gravel channel with cut banks in a dry valley, the terrain cut into bed and bank
along one contour, the desert-rock albedo for the scanned stones, and the albedo of the far escarpment.

Meters, y up, the channel runs along z (the hero cameras look toward -z, up the riverbed). Pure numpy (and PIL for
the stone albedo).
"""

from __future__ import annotations

import math

import numpy as np

import procedural as G
from scenes import _nature_kit as N
from scenes import _stream_props as SP

BANK = 1.6               # metres over which a cut bank rises from the bed
BANK_H = 0.9             # height of the cut bank above the bed
BED_TOP = 0.2            # bed material below this height (the bed's own bars stay well under it), bank above


def centre_x(z):
    """Centre line: straight round the cameras (z -6..10), then one broad bend that hides the channel upstream."""
    z = np.asarray(z, float)
    return 4.0 * np.sin(np.minimum(z + 6.0, 0.0) / 14.0) + 0.5 * np.sin(z / 4.3 + 0.7) * np.clip(-z / 15.0, 0.0, 1.0)


def half_width(z):
    """4.7-7.5 m: a wide, shallow flood channel (the bed is ~14 m across round the cameras)."""
    z = np.asarray(z, float)
    return 6.0 + 1.2 * np.sin(z / 11.0 + 2.0) + 0.4 * np.sin(z / 3.9)


def beyond_bed(x, z):
    """Signed distance from the bed's edge: < 0 on the bed, > 0 on the banks."""
    return np.abs(np.asarray(x, float) - centre_x(z)) - half_width(z)


def _bars(x, z):
    """Low gravel bars and swales on the bed, +-7 cm, elongated along the flow."""
    return (0.045 * np.sin(x / 1.7 + z / 9.0 + 0.4) * np.sin(z / 5.3 + 1.1) + 0.025 * np.sin(x / 0.9 - z / 3.1 + 2.0))


class RiverbedTerrain(N.Terrain):
    """N.Terrain's undulation for the valley, with the riverbed cut into it: a flat barred bed, a cut bank BANK_H
    high over BANK metres, then valley sides climbing ~10 m over 50 m. The undulation fades in past the bank top,
    so it never dips the banks to the bed's level (no bank-coloured hollows, no bed-coloured patches on the sides)."""

    def height(self, x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        valley = super().height(x, z) - self.base
        d = beyond_bed(x, z)
        t = np.clip(d / BANK, 0.0, 1.0)
        s = t * t * (3 - 2 * t)
        e = np.maximum(d - BANK, 0.0)
        side = 0.06 * e + 0.0025 * e ** 2
        return self.base + _bars(x, z) * (1 - s) + BANK_H * s + side + valley * np.clip(e / 4.0, 0.0, 1.0)


def terrain_parts(terrain) -> dict[str, G.Mesh]:
    """{'bed', 'bank'}: the terrain cut exactly along the BED_TOP contour (`_stream_props.cut_at_height`), so the
    edge where the gravel bed meets the bank is a smooth line, not grid teeth."""
    bed, bank = SP.cut_at_height(terrain.mesh(), terrain.base + BED_TOP)
    return {"bed": bed, "bank": bank}


def bare_rock(src: str) -> np.ndarray:
    """sRGB albedo for the scanned stones in a desert: the scan's own albedo (rock_moss_set_01 has green moss)
    desaturated to its luminance and warmed to a weathered sandstone tone, so the texture detail stays."""
    from PIL import Image

    a = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.clip(lum[..., None] * np.array([1.18, 1.0, 0.78], np.float32), 0.0, 1.0)


def ridge_albedo(seed: int, tile_w: float, y0: float, y_top: float, ppm: int = 24) -> np.ndarray:
    """sRGB albedo tile for the far escarpment (`_nature_kit.backdrop_band`): layered sandstone, strata warped by
    low-frequency noise, darker scree and scrub in the lower half. Tiles seamlessly in u."""
    W, H = int(tile_w * ppm), int((y_top - y0) * ppm)
    O = W // 6
    y = y_top - (np.arange(H)[:, None] + 0.5) / ppm
    aspect = (W + O) / H
    warp = G.fbm(H, W + O, octaves=4, base=6, seed=seed, aspect=aspect)
    grain = G.fbm(H, W + O, octaves=6, base=24, seed=seed + 1, aspect=aspect)
    strata = 0.5 + 0.5 * np.sin(y * math.tau / 2.3 + 6.0 * warp) * np.sin(y * math.tau / 0.9 + 3.0 * warp)
    tone = 0.7 + 0.35 * strata + 0.3 * (grain - 0.5)
    lin = np.array([0.20, 0.135, 0.085])[None, None, :] * tone[..., None]     # linear: ochre sandstone
    low = np.clip((y0 + 0.45 * (y_top - y0) - y) / 4.0, 0.0, 1.0)              # 1 in the lower part
    scrub = G.fbm(H, W + O, octaves=3, base=60, seed=seed + 2, aspect=aspect) > 0.8 - 0.06 * low
    lin[scrub & (low > 0)] = np.array([0.022, 0.024, 0.014])
    out = N._periodic(N._srgb(lin).transpose(2, 0, 1).reshape(3 * H, W + O), O).reshape(3, H, W).transpose(1, 2, 0)
    return out.astype(np.float32)
