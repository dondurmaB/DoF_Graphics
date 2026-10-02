"""Run the whole comparison with one command.

    python3 tools/compare/run_comparison.py

That does three things in order, and stops at the first failure:

  1. Cycles ground-truth references  -> reports/raytraced_dof/rt_*.exr (+ JSON)
  2. OpenGL batch captures           -> output/gl_*.png
  3. Pair, check and measure         -> reports/comparison/

Previously each step was a separate command with its own flags, and the two
renderers had to be given matching settings by hand. Getting that wrong does not
fail loudly: it produces a plausible-looking difference image of two different
camera setups. So the settings are defined once here and passed to both.

Useful switches:

    --quick                 64 samples, denoiser on, PNG. Checks the pipeline
                            runs end to end in a couple of minutes. NOT valid
                            as ground truth, and the output says so.
    --fstops 1.2 2.8 11     which apertures to compare (default: the full set)
    --focus 1.5             focus distance(s) in metres
    --lens 50               focal length(s) in mm
    --gather 0 1            0 naive, 1 CoC-weighted; both by default, so the
                            two baselines are measured against one reference
    --skip-reference        reuse existing references (they are the slow part)
    --skip-opengl           reuse existing captures
    --blender PATH          Blender executable
    --renderer PATH         DepthResearch executable

Exit code is 0 only if every step succeeded and every pair was measured.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIRECTORY = ROOT / "reports/raytraced_dof"
OPENGL_DIRECTORY = ROOT / "output"
COMPARISON_DIRECTORY = ROOT / "reports/comparison"

# Where Blender usually lives. Checked in order; --blender overrides.
BLENDER_CANDIDATES = (
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
    "blender",
)
# Where the build usually puts the renderer.
RENDERER_CANDIDATES = (
    ROOT / "build/DepthResearch",
    ROOT / "build/Debug/DepthResearch",
    ROOT / "build/Release/DepthResearch",
    ROOT / "cmake-build-debug/DepthResearch",
)


def find_blender(explicit):
    if explicit:
        return explicit if (Path(explicit).is_file() or shutil.which(explicit)) else None
    for candidate in BLENDER_CANDIDATES:
        if Path(candidate).is_file() or shutil.which(candidate):
            return candidate
    return None


def find_renderer(explicit):
    if explicit:
        return explicit if Path(explicit).is_file() else None
    for candidate in RENDERER_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def run(label, command, cwd=ROOT):
    """Run a step, streaming its output, and report how long it took."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print("$ " + " ".join(str(part) for part in command) + "\n", flush=True)
    started = time.monotonic()
    result = subprocess.run([str(part) for part in command], cwd=str(cwd))
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        print(f"\n{label} FAILED after {elapsed:.1f} s (exit {result.returncode}).")
        return False
    print(f"\n{label} finished in {elapsed:.1f} s.")
    return True


def number_list(values):
    return [f"{value:g}" for value in values]


def expected_pairs(args, suffix):
    """Every (reference, capture) filename the two steps should produce.

    Built from the same lists given to both renderers, so step 3 can say which
    pair is missing rather than silently comparing whatever happens to be on
    disk. The naming rules here mirror render_jobs() in render_dof.py and the
    batch naming in src/main.cpp; tests/test_pipeline.py checks they agree.
    """
    pairs = []
    for lens in args.lens:
        for focus in args.focus:
            for fstop in args.fstops:
                reference = f"rt_focus{focus:g}m_f{fstop:g}{suffix}"
                for gather in args.gather:
                    capture = f"gl_focus{focus:g}m_f{fstop:g}"
                    if abs(lens - 50.0) > 0.01:
                        capture += f"_{lens:g}mm"
                        reference = (f"rt_focus{focus:g}m_f{fstop:g}"
                                     f"_{lens:g}mm{suffix}")
                    if gather == 0:
                        capture += "_naive"
                    pairs.append((reference, capture + ".png"))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fstops", type=float, nargs="+", default=[1.2, 1.4, 2.8, 11.0])
    parser.add_argument("--focus", type=float, nargs="+", default=[1.5])
    parser.add_argument("--lens", type=float, nargs="+", default=[50.0])
    parser.add_argument("--gather", type=int, nargs="+", default=[0, 1], choices=(0, 1))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--samples", type=int, help="Cycles samples; overrides the mode default")
    parser.add_argument("--taps", type=int, default=160, help="OpenGL gather taps")
    parser.add_argument("--quick", action="store_true",
                        help="Fast pipeline check: 64 samples, denoised, PNG. Not ground truth.")
    parser.add_argument("--full-gi", action="store_true",
                        help="Pass through to Cycles: real global illumination rather than the "
                             "shading model matched to basic.frag")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-opengl", action="store_true")
    parser.add_argument("--blender")
    parser.add_argument("--renderer")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the three commands and the expected filenames, run nothing")
    args = parser.parse_args(argv)

    if len(args.lens) > 1 and not args.skip_reference:
        # render_dof.py takes a single lens per invocation, deliberately: the
        # lens changes the field of view and therefore what "the same frame"
        # means. Sweeping it needs one reference run per lens.
        print("Several focal lengths need one reference run each. Run this script once per "
              "--lens value, or pass --skip-reference if the references already exist.")
        return 1

    suffix = ".png" if args.quick else ".exr"
    pairs = expected_pairs(args, suffix)

    reference_command = [
        "--background", "--python-exit-code", "1",
        "--python", "tools/raytraced_reference/render_dof.py", "--",
        "--width", str(args.width), "--height", str(args.height),
        "--lens", f"{args.lens[0]:g}",
        "--focus", *number_list(args.focus),
        "--fstops", *number_list(args.fstops),
    ]
    if args.quick:
        reference_command += ["--samples", str(args.samples or 64)]
    else:
        reference_command += ["--ground-truth", "--convergence"]
        if args.samples:
            reference_command += ["--samples", str(args.samples)]
    if args.full_gi:
        reference_command.append("--full-gi")

    opengl_command = [
        "--batch",
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.taps),
        "--lens", *number_list(args.lens),
        "--focus", *number_list(args.focus),
        "--fstops", *number_list(args.fstops),
        "--gather", *[str(value) for value in args.gather],
        "--output", "output",
    ]

    compare_command = [sys.executable, "tools/compare/compare_renders.py", "--write-images"]
    if not args.quick:
        compare_command.append("--require-ground-truth")

    if args.dry_run:
        print("1. Cycles references:\n   blender " + " ".join(reference_command))
        print("\n2. OpenGL captures:\n   DepthResearch " + " ".join(opengl_command))
        print("\n3. Compare:\n   " + " ".join(str(p) for p in compare_command))
        print(f"\nExpecting {len(pairs)} pair(s):")
        for reference, capture in pairs:
            print(f"   {reference:40s} <-> {capture}")
        if args.quick:
            print("\n--quick output is NOT valid ground truth: the denoiser is on and the "
                  "output is clipped 8-bit PNG.")
        return 0

    if args.quick:
        print("--quick: 64 samples, denoiser ON, 8-bit PNG. This checks that the pipeline "
              "runs end to end.\nIt is NOT valid as ground truth; drop --quick for that.\n")

    # --- Step 1: Cycles.
    if args.skip_reference:
        print("Skipping the reference render (--skip-reference).")
    else:
        blender = find_blender(args.blender)
        if blender is None:
            print("Could not find Blender. Pass --blender /path/to/Blender.\n"
                  "Looked in: " + ", ".join(BLENDER_CANDIDATES))
            return 1
        if not run("1/3  Cycles reference render", [blender, *reference_command]):
            return 1

    # --- Step 2: OpenGL.
    if args.skip_opengl:
        print("Skipping the OpenGL capture (--skip-opengl).")
    else:
        renderer = find_renderer(args.renderer)
        if renderer is None:
            print("Could not find the DepthResearch executable. Build it first:\n"
                  "  cmake -S . -B build -G Ninja && cmake --build build\n"
                  "or pass --renderer /path/to/DepthResearch.")
            return 1
        if not run("2/3  OpenGL batch capture", [renderer, *opengl_command]):
            return 1

    # --- Check both sides produced what the settings asked for, before the
    # comparison step reports a confusing partial result.
    missing = []
    for reference, capture in pairs:
        if not (REFERENCE_DIRECTORY / reference).is_file():
            missing.append(f"reference {reference}")
        if not (OPENGL_DIRECTORY / capture).is_file():
            missing.append(f"capture {capture}")
    if missing:
        print("\nThese expected files are not on disk:")
        for item in missing:
            print(f"  {item}")
        print("The two steps disagree about settings, or one of them wrote elsewhere.")
        return 1

    # --- Step 3: compare.
    if not run("3/3  Pair and measure", compare_command):
        return 1

    report = COMPARISON_DIRECTORY / "README.md"
    metrics = COMPARISON_DIRECTORY / "metrics.json"
    print(f"\n{'=' * 72}\nDone. {len(pairs)} pair(s) compared.\n{'=' * 72}")
    if report.is_file():
        print(f"  Report:  {report.relative_to(ROOT)}")
    if metrics.is_file():
        print(f"  Metrics: {metrics.relative_to(ROOT)}")
        try:
            records = json.loads(metrics.read_text())
        except (OSError, ValueError):
            records = []
        for record in records:
            whole = record.get("whole_frame") or {}
            band = record.get("edge_band") or {}
            quality = record.get("reference_quality") or {}
            noise = quality.get("reference_error_levels")
            print(f"  {record.get('settings', record.get('tag', '?')):34s} "
                  f"frame MAE {whole.get('mae', float('nan')):6.2f}   "
                  f"edge MAE {band.get('mae', float('nan')):6.2f}"
                  + (f"   reference noise {noise:.2f}" if noise else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
