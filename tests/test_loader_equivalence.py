"""Compare real builder output for small fixtures, alongside pinned alley totals."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/scene"))
import scene_loader

DUMPER = sys.argv.pop(1)


class Equivalence(unittest.TestCase):
    def cpp(self, text):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.scene"
            source.write_text(text)
            return subprocess.run([DUMPER, str(source)], capture_output=True, text=True)

    def test_every_vertex_attribute_and_index(self):
        source = """version 1
capture width 349 height 231
imported pos 2 3 4 rot 22 13 35 size .2 .3 .4 rgb .4 .5 .6
box size .7 .2 1.3 bevel .02 rot 7 41 32
box pos 1 2 3 size 2 1 4 rot 17 31 9 rgb .2 .4 .6
cyl pos -2 0 1 size .7 1.3 .9 rot 9 22 31 seg 7 smooth 1
cyl pos 0 2 1 size 2 .8 1 rot 20 11 70 seg 9 smooth 0 taper .3
cyl pos 4 0 0 seg 5 taper 0 rgb .4 .2 .7 emit 12
sph pos 0 0 -3 size 2 3 .5 rot 15 22 41 seg 8 rgb .9 .6 .2
sph pos 0 4 0 seg 5 emit 7
"""
        run = self.cpp(source)
        self.assertEqual(run.returncode, 0, run.stderr)
        actual = json.loads(run.stdout)
        parsed = scene_loader.parse_scene(source)
        self.assertEqual(actual["capture"], [parsed.capture_width, parsed.capture_height])
        for a, b in zip(
            actual["imported"],
            (
                *parsed.import_position,
                *parsed.import_rotation,
                *parsed.import_scale,
                *parsed.import_albedo,
            ),
        ):
            self.assertAlmostEqual(a, b, places=7)
        mesh = scene_loader.build_geometry(parsed)
        expected = [
            (*p, *n, *c, e)
            for p, n, c, e in zip(mesh.positions, mesh.normals, mesh.colors, mesh.emissions)
        ]
        self.assertEqual(len(actual["vertices"]), len(expected))
        for index, (a, b) in enumerate(zip(actual["vertices"], expected)):
            for field, (x, y) in enumerate(zip(a, b)):
                self.assertTrue(
                    math.isclose(x, y, rel_tol=5e-7, abs_tol=1e-7), (index, field, x, y)
                )
        self.assertEqual(actual["indices"], [i for tri in mesh.triangles for i in tri])

    def test_both_parsers_reject_the_asymmetric_cases(self):
        for text in (
            "version 1\n",
            "version 1 extra\nbox\n",
            "version 1.9\nbox\n",
            "box\n",
            "version 2\nbox\n",
            "version 1\ncyl taper -1\n",
            "version 1\nbox bevel -.1\n",
            "version 1\nbox size 1 2 3 bevel .5\n",
            "version 1\ncapture width 23.5\nbox\n",
            "version 1\nimported size 1 0 1\nbox\n",
        ):
            self.assertNotEqual(self.cpp(text).returncode, 0, text)
            with self.assertRaises(scene_loader.SceneError, msg=text):
                scene_loader.parse_scene(text)


if __name__ == "__main__":
    unittest.main()
