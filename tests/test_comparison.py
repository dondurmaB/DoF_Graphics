"""Regression checks for silent comparison failures and stale-report claims."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/compare"))
import compare_renders as compare


class Comparison(unittest.TestCase):
    def test_no_pairs_is_failure(self):
        with patch.object(
            compare, "find_pairs", return_value=([], [], [])
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main([]), 1)

    def test_stale_rejected_and_override_reports_the_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = root / "scene"
            scene.write_text("fixture")
            gl = root / "gl_tag.png"
            rt = root / "rt_tag.png"
            for p in (gl, rt):
                Image.new("RGB", (8, 8), (100, 90, 80)).save(p)
            rt.with_suffix(".json").write_text(json.dumps({"scene_sha256": "stale"}))
            with patch.object(compare, "ROOT", root), patch.object(
                compare, "find_pairs", return_value=([("tag", gl, rt)], [], [])
            ), contextlib.redirect_stdout(io.StringIO()):
                args = ["--scene", "scene", "--output", str(root / "report")]
                self.assertEqual(compare.main(args), 1)
                compare.main(args + ["--allow-stale"])
            records = json.loads((root / "report/metrics.json").read_text())
            self.assertFalse(records[0]["scene_hash_verified"])
            report = (root / "report/README.md").read_text()
            self.assertIn("NOT verified", report)
            self.assertNotIn("Every compared reference sidecar", report)
            # Even valid hashes cannot turn a dimension mismatch into success.
            rt.with_suffix(".json").write_text(
                json.dumps({"scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()})
            )
            Image.new("RGB", (9, 8)).save(rt)
            with patch.object(compare, "ROOT", root), patch.object(
                compare, "find_pairs", return_value=([("tag", gl, rt)], [], [])
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(args), 1)


if __name__ == "__main__":
    unittest.main()
