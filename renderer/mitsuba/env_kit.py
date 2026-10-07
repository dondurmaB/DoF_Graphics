"""Shared sun/sky and ground presets, so scenes by different authors light and expose alike.

    b.d["sky"] = env_kit.sky(ctx.env, sun_direction=..., sampling_weight=...)
    ground_bsdf = env_kit.ground(ctx.env)

`env` is one of `scene_api.ENVS`. At `scale=1.0` an outdoor scene lands in the contract's
exposure band without per-scene tuning. Interiors lit through windows must balance the
sky against their lamps themselves (the cafe uses scale 60); do not copy that number outdoors.
Importing this file needs no Mitsuba; the dicts it returns are for `mi.load_dict`.
"""

from __future__ import annotations

import math

import numpy as np

from scene_api import ENVS

# Constant sky radiance for overcast and wet: a horizontal white diffuse plane is 1/4 as bright as
# under the clear sky at the default sun (real overcast is 1/3 to 1/10 of a sunny day). Not 1/6:
# at 1/6 a plain pavement scene came out at median 0.014, under the contract's 0.02 floor.
OVERCAST_RADIANCE = 0.0788

# base colour, then principled-BSDF extras. The sunsky ground albedo is taken from the base
# colour, so the light bounced back into the sky matches the ground the scene actually has.
_GROUND = {
    "clear": {"base": (0.25, 0.24, 0.23), "roughness": 0.85},
    "overcast": {"base": (0.25, 0.24, 0.23), "roughness": 0.85},
    "snow": {"base": (0.85, 0.87, 0.90), "roughness": 0.90, "specular": 0.3},
    "sand": {"base": (0.55, 0.42, 0.25), "roughness": 0.95},
    "wet": {"base": (0.05, 0.05, 0.055), "roughness": 0.12, "clearcoat": 0.8, "clearcoat_gloss": 0.9},
}
_TURBIDITY = {"clear": 3.0, "snow": 2.0, "sand": 5.0}   # sunsky envs; others have no direct sun


def default_sun_direction(elevation_deg: float = 40.0, azimuth_deg: float = 35.0) -> list[float]:
    """Unit vector pointing toward the sun (y up), the convention `sunsky` and the cafe use."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    return [math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)]


def _check(env: str) -> None:
    if env not in ENVS:
        raise ValueError(f"unknown env {env!r}; choose from {ENVS}")


def sunsky_to_world():
    """Mitsuba 3.9.1's `sunsky` puts the zenith on local +z. Our scenes are y up, so without this
    rotation the sky is black over the z < 0 half of all directions (and entirely black when the sun
    has z < 0). `sun_direction` is read in world space either way. Measured on the cluster."""
    import mitsuba as mi

    return mi.ScalarTransform4f().rotate(axis=[1, 0, 0], angle=-90)


def sky(env: str, sun_direction=None, scale: float = 1.0, sampling_weight: float = 1.0) -> dict:
    """One emitter dict for the whole sky. `scale` multiplies sun and sky together."""
    _check(env)
    if env in ("overcast", "wet"):
        return {"type": "constant", "radiance": {"type": "rgb", "value": [OVERCAST_RADIANCE * scale] * 3},
                "sampling_weight": float(sampling_weight)}
    d = np.asarray(default_sun_direction() if sun_direction is None else sun_direction, dtype=float)
    d = d / np.linalg.norm(d)
    if d[1] <= 0:
        raise ValueError("sun_direction points below the horizon; sunsky renders black there")
    return {"type": "sunsky", "to_world": sunsky_to_world(), "sun_direction": d.tolist(), "turbidity": _TURBIDITY[env],
            "albedo": {"type": "rgb", "value": list(_GROUND[env]["base"])},
            "sun_scale": float(scale), "sky_scale": float(scale), "sampling_weight": float(sampling_weight)}


def ground(env: str) -> dict:
    """Principled BSDF for flat ground: pavement (clear, overcast), snow, sand, wet asphalt."""
    _check(env)
    g = dict(_GROUND[env])
    base = g.pop("base")
    return {"type": "principled", "base_color": {"type": "rgb", "value": list(base)}, **g}
