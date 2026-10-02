"""Checks that the three stages of the comparison pipeline agree.

`tools/compare/run_comparison.py` runs the Cycles script, then the OpenGL
renderer in batch mode, then the metrics. Each stage names its own output, and
the metrics stage pairs them BY NAME. So the naming rules live in three places:

    tools/raytraced_reference/render_dof.py   render_jobs()      rt_<tag>.<ext>
    src/main.cpp                              the batch loop     gl_<tag>.png
    tools/compare/run_comparison.py           expected_pairs()   both

A disagreement between them does not fail loudly at runtime: the comparison
just finds no pairs, or worse, pairs a capture with a reference taken at other
settings. These tests pin the three together.
"""
import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load("render_dof", "tools/raytraced_reference/render_dof.py")
pipeline = load("run_comparison", "tools/compare/run_comparison.py")
compare = load("compare_renders", "tools/compare/compare_renders.py")


class PipelineNamingTests(unittest.TestCase):
    def test_orchestrator_predicts_the_cycles_output_names(self):
        args = pipeline.main.__wrapped__ if hasattr(pipeline.main, "__wrapped__") else None
        del args
        options = pipeline.argparse.Namespace(
            fstops=[1.2, 2.8, 11.0], focus=[1.5], lens=[50.0], gather=[0, 1])
        predicted = {name for name, _ in pipeline.expected_pairs(options, ".exr")}
        actual = {job[0] for job in reference.render_jobs(
            reference.parse_args(["--ground-truth", "--focus", "1.5",
                                  "--fstops", "1.2", "2.8", "11"]))}
        self.assertEqual(predicted, actual)

    def test_orchestrator_predicts_the_opengl_capture_names(self):
        """The batch names are built in C++, so the format is checked literally."""
        source = (ROOT / "src/main.cpp").read_text()
        # One naming site in the batch loop, one for the P key; both must use
        # the same prefix, the same "focus<N>m_f<N>" shape and the same suffix.
        self.assertEqual(source.count('name << "gl_focus"'), 1)
        self.assertIn('matched << "output/gl_focus"', source)
        for snippet in ('<< formatCompact(focusDistanceMeters) << "m_f"',
                        '<< formatCompact(fNumber)',
                        'name << "_" << formatCompact(focalLengthMillimeters) << "mm"',
                        'name << "_naive"'):
            self.assertIn(snippet, source, msg=snippet)

        options = pipeline.argparse.Namespace(
            fstops=[1.2], focus=[1.5], lens=[85.0], gather=[0, 1])
        captures = {capture for _, capture in pipeline.expected_pairs(options, ".exr")}
        self.assertEqual(captures, {"gl_focus1.5m_f1.2_85mm.png",
                                    "gl_focus1.5m_f1.2_85mm_naive.png"})

    def test_variant_suffixes_pair_back_to_one_reference(self):
        """A naive capture must compare against the same reference as a weighted one."""
        self.assertEqual(compare.split_variant("focus1.5m_f1.2_naive"),
                         ("focus1.5m_f1.2", "naive"))
        self.assertEqual(compare.split_variant("focus1.5m_f1.2"),
                         ("focus1.5m_f1.2", "weighted"))
        # Every suffix the C++ side can append must be known to the pairing.
        source = (ROOT / "src/main.cpp").read_text()
        appended = set(re.findall(r'name << "(_[a-z]+)"', source))
        self.assertTrue(appended.issubset(set(compare.VARIANT_SUFFIXES)),
                        msg=f"src/main.cpp appends {appended}, pairing knows "
                            f"{compare.VARIANT_SUFFIXES}")

    def test_comparison_reads_the_scene_path_from_one_place(self):
        """The digest check must hash the file the renderers actually load."""
        self.assertEqual(compare.scene_file_name(), reference.SCENE_FILE)
        source = (ROOT / "src/main.cpp").read_text()
        self.assertIn(f'scenePath = "{reference.SCENE_FILE}"', source)

    def test_quick_mode_is_not_offered_as_ground_truth(self):
        """--quick must not silently produce something that looks like a reference."""
        plan = reference.make_plan(reference.parse_args(["--samples", "64"]))
        self.assertFalse(plan["suitable_as_training_ground_truth"])
        self.assertEqual(plan["denoising"], "OPENIMAGEDENOISE")
        source = (ROOT / "tools/compare/run_comparison.py").read_text()
        self.assertIn("NOT valid as ground truth", source)
        # And quick mode must not pass --require-ground-truth, which would make
        # its own comparison step fail every time.
        self.assertIn("if not args.quick:\n        compare_command.append", source)

    def test_a_multi_lens_sweep_is_refused_rather_than_mismatched(self):
        """One Cycles run has one field of view, so several lenses need several runs."""
        source = (ROOT / "tools/compare/run_comparison.py").read_text()
        self.assertIn("Several focal lengths need one reference run each", source)


if __name__ == "__main__":
    unittest.main()
