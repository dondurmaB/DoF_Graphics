"""Reproduce Phase 1 inspection captures; NOT a ground-truth/DoF benchmark.

Runs the production OpenGL and Cycles renderers; stops on the first failure.
All outputs stay under ignored output/verification. --dry-run prints commands.
The production batch FBO verifies physical pixel dimensions independently of
macOS window services. Those services are still needed to create a GL context.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import shlex

ROOT = Path(__file__).resolve().parents[2]


def commands(blender):
    gl = str(ROOT / "build/DepthResearch")
    comparison_lens = "85"
    comparison_focus = "2.5"
    comparison_taps = "192"
    rt = [
        blender,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        "tools/raytraced_reference/render_dof.py",
        "--",
    ]
    result = []
    for name, pose in (
        ("scene", []),
        ("wide_front", ["--lens", "22", "--pitch", "-8"]),
        (
            "wide_corner",
            [
                "--lens",
                "24",
                "--x",
                "2.5",
                "--y",
                ".6",
                "--z",
                "4.5",
                "--yaw",
                "-112",
                "--pitch",
                "-17",
            ],
        ),
        (
            "machine",
            [
                "--lens",
                "35",
                "--x",
                "1.65",
                "--y",
                ".15",
                "--z",
                "-5.25",
                "--yaw",
                "-66",
                "--pitch",
                "-20",
            ],
        ),
    ):
        result.append(
            [gl, "--batch", "--sharp", "--output", f"output/verification/gl_{name}.png", *pose]
        )
    for fstop in (1.2, 22):
        result.append(
            [
                gl,
                "--batch",
                "--lens",
                comparison_lens,
                "--focus",
                comparison_focus,
                "--fstop",
                str(fstop),
                "--coc-samples",
                comparison_taps,
                "--output",
                f"output/verification/gl_focus{comparison_focus}m_f{fstop:g}_{comparison_lens}mm.png",
            ]
        )
    result.append(
        rt
        + [
            "--fstops",
            "1.2",
            "22",
            "--sharp",
            "--samples",
            "128",
            "--no-denoise",
            "--output",
            "output/verification/cycles",
        ]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = ROOT / "output/verification"
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    for i, command in enumerate(commands(args.blender)):
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        with (output / f"render-{i}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if "--output" in command:
            path = ROOT / command[command.index("--output") + 1]
            if path.suffix == ".png" and (not path.is_file() or path.stat().st_size == 0):
                raise RuntimeError(f"Missing capture: {path}")
    if not args.dry_run:
        # Fixed names are inspection views, not an aperture sweep pairing API.
        for tag in ("focus2.5m_f1.2_85mm", "focus2.5m_f22_85mm", "focus2.5m_f1.2_85mm_sharp"):
            for extension in ("png", "exr", "pfm", "json"):
                path = output / "cycles" / f"rt_{tag}.{extension}"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Missing {path}")
    print(
        "Inspection commands validated."
        if args.dry_run
        else "All inspection outputs present; assess images and linear measurements before accepting."
    )


if __name__ == "__main__":
    main()
