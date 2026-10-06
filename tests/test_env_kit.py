"""Static checks for env_kit. No rendering; needs Mitsuba only to confirm the dicts load.

    python tests/test_env_kit.py        (or pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "renderer" / "mitsuba"))

import mitsuba as mi  # noqa: E402

mi.set_variant("scalar_rgb")

import env_kit as E  # noqa: E402
import scene_api as S  # noqa: E402


def test_unknown_env_raises():
    for fn in (E.sky, E.ground):
        for bad in ("night", "haze", ""):
            try:
                fn(bad)
            except ValueError:
                continue
            raise AssertionError(f"{fn.__name__}({bad!r}) was accepted")


def test_every_env_loads_in_mitsuba():
    for env in S.ENVS:
        mi.load_dict(E.sky(env))
        mi.load_dict(E.ground(env))


def test_sky_shape_and_scale():
    for env in ("clear", "snow", "sand"):
        a, b = E.sky(env), E.sky(env, scale=3.0, sampling_weight=7.0)
        assert a["type"] == "sunsky" and a["sun_scale"] == 1.0 and b["sun_scale"] == b["sky_scale"] == 3.0
        assert b["sampling_weight"] == 7.0
        assert abs(sum(v * v for v in a["sun_direction"]) - 1.0) < 1e-9 and a["sun_direction"][1] > 0
        assert a["albedo"]["value"] == list(E._GROUND[env]["base"])
    over, wet = E.sky("overcast"), E.sky("wet")
    assert over == wet and over["type"] == "constant"
    assert E.sky("overcast", scale=2.0)["radiance"]["value"][0] == 2 * over["radiance"]["value"][0]


def test_sun_direction_is_normalised_and_must_be_above_horizon():
    assert abs(E.sky("clear", sun_direction=[0, 5, 5])["sun_direction"][1] - 2 ** -0.5) < 1e-9
    for bad in ([0, 0, 1], [0, -1, 1]):
        try:
            E.sky("clear", sun_direction=bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_ground_presets_differ_as_described():
    g = {e: E.ground(e) for e in S.ENVS}
    assert all(v["type"] == "principled" for v in g.values())
    assert g["wet"]["roughness"] < g["clear"]["roughness"] and g["wet"]["clearcoat"] > 0
    assert g["clear"] == g["overcast"]
    lum = lambda e: sum(g[e]["base_color"]["value"])
    assert lum("snow") > lum("sand") > lum("clear") > lum("wet")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
