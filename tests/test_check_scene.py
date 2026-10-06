"""Metric functions of check_scene.py. Pure numpy, no Mitsuba.

    python tests/test_check_scene.py        (or pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "renderer" / "mitsuba"))

import numpy as np  # noqa: E402

import check_scene as C  # noqa: E402


def test_median5x5_matches_brute_force():
    rng = np.random.default_rng(0)
    img = rng.random((9, 11)).astype(np.float32)
    padded = np.pad(img, 2, mode="reflect")
    brute = np.array([[np.median(padded[y:y + 5, x:x + 5]) for x in range(11)] for y in range(9)])
    assert np.allclose(C.median5x5(img), brute)


def test_firefly_fraction_counts_isolated_outliers_only():
    lum = np.full((40, 40), 0.1, np.float32)
    assert C.firefly_fraction(lum) == 0.0
    for y, x in ((5, 5), (20, 30), (33, 8)):
        lum[y, x] = 50.0                       # isolated hot pixels
    assert abs(C.firefly_fraction(lum) - 3 / 1600) < 1e-9
    lum[10:18, 10:18] = 50.0                   # an 8x8 lit bulb: interior and straight edges are not fireflies,
    assert abs(C.firefly_fraction(lum) - (3 + 12) / 1600) < 1e-9   # only 3 pixels at each corner (< 13 of 25 bright)


def test_luminance_uses_rec709_weights():
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    assert np.allclose(C.luminance(rgb), [[0.2126, 0.7152, 0.0722]])


def test_sky_fraction():
    depth = np.array([[1.0, np.inf], [np.inf, np.inf]])
    assert C.sky_fraction(depth) == 0.75


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
