"""Linear radiance checks for matched fill; uses both production renderers.

--render regenerates isolated calibration patches (including a sealed room).
Without it, summarize already rendered patches and the cafe inspection captures.
Outputs are explicitly inspection evidence, not a ground-truth certificate.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/verification"


def read_pfm(path):
    with Path(path).open("rb") as stream:
        if stream.readline() != b"PF\n":
            raise ValueError("Expected RGB PFM")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        pixels = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
    return pixels.reshape(height, width, 3)[::-1].astype(np.float64) * abs(scale)


def render(blender):
    OUT.mkdir(parents=True, exist_ok=True)
    base = (
        "version 1\ncapture width 64 height 64\n"
        "camera pos 0 0 3 yaw -90 pitch 0 focus 3 fnumber 8 lens 50 sensor 24\n"
        "shadow lo -4 -4 -1 hi 4 4 7\nbox size 6 6 .1 rgb .5 .4 .3\n"
    )
    enclosure = (
        "box pos -3 0 3 size .1 6 6\nbox pos 3 0 3 size .1 6 6\n"
        "box pos 0 -3 3 size 6 .1 6\nbox pos 0 3 3 size 6 .1 6\n"
        "box pos 0 0 6 size 6 6 .1\n"
    )
    for name, sun, ambient in (
        ("ambient", 0, 0.2),
        ("sun", 1, 0),
        ("both", 1, 0.2),
        ("enclosed", 0, 0.2),
    ):
        path = OUT / (name + ".scene")
        path.write_text(
            base
            + f"sun dir 0 0 1 color 1 1 1 energy {sun} angle 0\nambient color 1 1 1 strength {ambient}\n"
            + (enclosure if name == "enclosed" else "")
        )
        gl = [
            str(ROOT / "build/DepthResearch"),
            "--batch",
            "--sharp",
            "--scene",
            str(path),
            "--output",
            str(OUT / (name + "-gl.png")),
        ]
        rt = [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            "tools/raytraced_reference/render_dof.py",
            "--",
            "--scene",
            str(path),
            "--fstops",
            "8",
            "--sharp",
            "--samples",
            "32",
            "--no-denoise",
        ]
        jobs = [gl, rt + ["--output", str(OUT / (name + "-rt"))]]
        if name == "enclosed":
            jobs.append(rt + ["--full-gi", "--output", str(OUT / "enclosed-world-rt")])
        for i, command in enumerate(jobs):
            with (OUT / f"{name}-{i}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def summarize():
    rows = []
    def compare_patch(name, gl_pixels, rt_pixels, **extra):
        gl_mean = gl_pixels.mean((0, 1))
        rt_mean = rt_pixels.mean((0, 1))
        rel = np.abs(gl_pixels - rt_pixels) / np.maximum(np.abs(rt_pixels), 1e-9)
        worst_index = np.unravel_index(np.argmax(rel), rel.shape)
        rows.append(
            dict(
                name=name,
                gl_mean=gl_mean.tolist(),
                cycles_mean=rt_mean.tolist(),
                max_channel_relative_error_of_means=float(
                    np.max(np.abs(gl_mean - rt_mean) / np.maximum(np.abs(rt_mean), 1e-9))
                ),
                max_per_pixel_channel_relative_error=float(rel[worst_index]),
                worst_pixel_yx_channel=[int(worst_index[0]), int(worst_index[1]), int(worst_index[2])],
                **extra,
            )
        )

    for name, sun, ambient in (
        ("ambient", 0, 0.2),
        ("sun", 1, 0),
        ("both", 1, 0.2),
        ("enclosed", 0, 0.2),
    ):
        gl = read_pfm(OUT / (name + "-gl.linear.pfm"))[28:36, 28:36]
        rt = read_pfm(OUT / (name + "-rt") / "rt_focus3m_f8_sharp.pfm")[28:36, 28:36]
        expected = np.array((0.5, 0.4, 0.3)) * (ambient + sun / math.pi)
        compare_patch(name, gl, rt, analytic_mean=expected.tolist(), roi_xywh=[28, 28, 8, 8])
    physical = read_pfm(OUT / "enclosed-world-rt/rt_focus3m_f8_sharp.pfm")[28:36, 28:36].mean(
        (0, 1)
    )
    gl = read_pfm(OUT / "gl_scene.linear.pfm")
    rt = read_pfm(OUT / "cycles/rt_focus2.5m_f1.2_sharp.pfm")
    for name, (x, y, w, h) in {
        "cafe_table": (520, 1010, 90, 40),
        "cafe_sunlit_wall": (380, 310, 60, 50),
    }.items():
        compare_patch(name, gl[y : y + h, x : x + w], rt[y : y + h, x : x + w], roi_xywh=[x, y, w, h])
    result = {
        "quantity": "scene-linear RGB radiance (pre-lens OpenGL; sharp Cycles)",
        "patches": rows,
        "enclosed_physical_world_only": physical.tolist(),
        "qualification": "Flat surface agreement only; does not validate shadow edges, lens integration or noise convergence.",
    }
    (OUT / "lighting.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender")
    args = parser.parse_args()
    if args.render:
        render(args.blender)
    summarize()


if __name__ == "__main__":
    main()
