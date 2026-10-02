"""CPU invariants for the independent oracle; actual GPU/Cycles test is verify_brdf.py."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/raytraced_reference"))
from brdf import cases, specular


class BrdfTests(unittest.TestCase):
    def test_reciprocity_and_finite_nonnegative_results(self):
        for case in cases():
            l, v, r, f = (case[k] for k in ("l", "v", "rough", "f0"))
            result = specular((0, 0, 1), l, v, r, f)
            self.assertTrue(math.isfinite(result))
            self.assertGreaterEqual(result, 0)
            self.assertAlmostEqual(result, specular((0, 0, 1), v, l, r, f), places=8)

    def test_normal_incidence_has_analytic_value(self):
        n = (0, 0, 1)
        for rough in (.05, .16, .35, .85, 1):
            self.assertAlmostEqual(specular(n, n, n, rough, .04),
                                   .04 / (4 * math.pi * rough**4), places=8)

    def test_disabled_and_backfacing_lobes_are_zero(self):
        self.assertEqual(specular((0, 0, 1), (0, 0, 1), (0, 0, 1), .2, 0), 0)
        self.assertEqual(specular((0, 0, 1), (0, 0, -1), (0, 0, 1), .2, .7), 0)
        self.assertEqual(specular((0, 0, 1), (0, 0, 1), (1, 0, 0), .2, .7), 0)


if __name__ == "__main__":
    unittest.main()
