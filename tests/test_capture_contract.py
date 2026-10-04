"""Pin the actual C++ filename builder against Python, not a source regex."""

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/raytraced_reference"))
import render_dof

DUMPER = sys.argv.pop(1)


class Contract(unittest.TestCase):
    def test_all_tags(self):
        actual = subprocess.check_output([DUMPER], text=True).splitlines()
        expected = [
            render_dof.capture_tag(focus, stop, lens, sharp)
            for lens in (50, 85, 35.5)
            for focus in (1.6, 2.5, 15)
            for stop in (1.2, 1.4, 22)
            for sharp in (False, True)
        ]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
