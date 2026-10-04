"""Pair the OpenGL and Cycles images of the same settings and measure the gap.

    python3 tools/compare/compare_renders.py                 # report what is paired
    python3 tools/compare/compare_renders.py --write-images   # + side-by-side and difference PNGs

Inputs (both produced from the same scene file):
  reports/raytraced_dof/rt_focus5m_f1.4.png   Cycles inspection render (not certified ground truth)
  output/gl_focus5m_f1.4.png                  OpenGL, CoC gather (press P in the renderer)

Pairing is by filename: the renderer writes `gl_<tag>.png` with exactly the tag
render_dof.py writes as `rt_<tag>.png`, so a pair is found by name rather than
by remembering which screenshot was which. Capture the OpenGL side in BasicDoF
mode (key 6) at the same resolution as the reference, 1200x1200 by default.

Metrics are deliberately simple and stated in full, because a single number
with an unexplained definition is worth nothing in a report:

  MAE   mean absolute difference per channel, in 0-255 levels
  RMSE  root mean square difference, same units; punishes large local errors
  PSNR  peak signal-to-noise ratio in dB, 20*log10(255/RMSE); higher is closer
  P95   the 95th percentile absolute difference, i.e. how bad the worst 5% is

PSNR over the whole frame flatters this comparison, because most of the frame
is in focus and identical. The edge-band figure is the one that matters: it
restricts the same metrics to pixels near a depth discontinuity, which is
exactly where the CoC gather is known to be wrong.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - reported, not raised
    print("This tool needs numpy. Install it in the project environment.", file=sys.stderr)
    raise SystemExit(1)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    print("This tool needs Pillow. Install it in the project environment.", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIRECTORY = ROOT / "reports/raytraced_dof"
OPENGL_DIRECTORY = ROOT / "output"
OUTPUT_DIRECTORY = ROOT / "reports/comparison"


def load_rgb(path):
    """Load as float in 0-255 levels, dropping any alpha channel."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def metrics(reference, candidate, mask=None):
    difference = np.abs(candidate - reference)
    if mask is not None:
        if not mask.any():
            return None
        difference = difference[mask]
    mae = float(difference.mean())
    rmse = float(math.sqrt(float((difference ** 2).mean())))
    # A perfect match has no noise floor, so PSNR is reported as infinite.
    psnr = float("inf") if rmse == 0.0 else 20.0 * math.log10(255.0 / rmse)
    return {"mae": mae, "rmse": rmse, "psnr_db": psnr,
            "p95": float(np.percentile(difference, 95)), "pixels": int(difference.size // 3)}


def edge_band_mask(reference, width, threshold):
    """Pixels near a strong luminance edge in the reference image.

    A stand-in for "near a depth discontinuity": the renderer's depth buffer is
    not saved alongside the screenshot, and silhouettes in this scene are also
    luminance edges. Crude but honest, and stated as such in the report.
    """
    luminance = reference @ np.array([0.2126, 0.7152, 0.0722])
    gradient_y, gradient_x = np.gradient(luminance)
    strong = np.hypot(gradient_x, gradient_y) > threshold
    # Dilate by `width` with repeated shifts: no scipy dependency needed.
    band = strong.copy()
    for _ in range(max(width, 0)):
        padded = np.pad(band, 1, mode="edge")
        band = (padded[:-2, 1:-1] | padded[2:, 1:-1] |
                padded[1:-1, :-2] | padded[1:-1, 2:] | band)
    return np.repeat(band[:, :, None], 3, axis=2)


def find_pairs():
    """Match output/gl_<tag>.png to reports/raytraced_dof/rt_<tag>.png."""
    references = {path.stem[3:]: path for path in sorted(REFERENCE_DIRECTORY.glob("rt_*.png"))}
    candidates = {path.stem[3:]: path for path in sorted(OPENGL_DIRECTORY.glob("gl_*.png"))}
    pairs = [(tag, candidates[tag], references[tag]) for tag in sorted(set(references) & set(candidates))]
    return pairs, sorted(set(references) - set(candidates)), sorted(set(candidates) - set(references))


def describe_tag(tag):
    """rt_focus5m_f1.4 -> "focus 5 m, f/1.4"."""
    focus = re.search(r"focus([\d.]+)m", tag)
    fstop = re.search(r"_f([\d.]+)", tag)
    lens = re.search(r"_([\d.]+)mm", tag)
    parts = []
    if focus:
        parts.append(f"focus {focus[1]} m")
    if fstop:
        parts.append(f"f/{fstop[1]}")
    if lens:
        parts.append(f"{lens[1]} mm")
    if tag == "sharp":
        parts.append("DoF disabled")
    return ", ".join(parts) or tag


def write_images(tag, opengl, reference, directory):
    """Side-by-side and amplified absolute difference, for the report."""
    directory.mkdir(parents=True, exist_ok=True)
    height, width, _ = reference.shape
    gap = 8
    strip = np.full((height, width * 2 + gap, 3), 24.0)
    strip[:, :width] = opengl
    strip[:, width + gap:] = reference
    Image.fromarray(strip.clip(0, 255).astype(np.uint8)).save(directory / f"side_{tag}.png")

    # x4 so a 10-level error is visible on screen; the scale is in the filename
    # because an unlabelled difference image invites the wrong conclusion.
    difference = (np.abs(opengl - reference) * 4.0).clip(0, 255)
    Image.fromarray(difference.astype(np.uint8)).save(directory / f"diff_{tag}_x4.png")
    return f"side_{tag}.png", f"diff_{tag}_x4.png"


def scene_digest(scene="scene/cafe.scene"):
    """sha256 of the scene file both renderers are supposed to be reading."""
    import hashlib
    return hashlib.sha256((ROOT / scene).read_bytes()).hexdigest()


def reference_scene_check(reference_path, expected_digest):
    """Was this reference rendered from the scene file that is on disk now?

    render_dof.py writes a sidecar JSON next to every PNG recording the scene
    file's sha256. Comparing it to the current file is the only way to be sure
    the two images in a pair really do show the same environment. A reference
    from an older scene will still compare numerically and produce a
    meaningless result, so this is checked rather than assumed.
    """
    sidecar = reference_path.with_suffix(".json")
    if not sidecar.is_file():
        return "no sidecar JSON; cannot confirm which scene it used"
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return "sidecar JSON could not be read"
    recorded = record.get("scene_sha256")
    if recorded is None:
        return "rendered before the scene file existed (pre-experiment-18); re-render it"
    if recorded != expected_digest:
        return "rendered from a DIFFERENT scene file; re-render it"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-images", action="store_true",
                        help="Also write side-by-side and difference PNGs")
    parser.add_argument("--edge-width", type=int, default=6,
                        help="Half-width in pixels of the edge band (default 6)")
    parser.add_argument("--edge-threshold", type=float, default=12.0,
                        help="Luminance gradient above which a pixel counts as an edge")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Compare even when a reference came from another scene file")
    parser.add_argument("--scene", default="scene/cafe.scene")
    parser.add_argument("--output", default=str(OUTPUT_DIRECTORY))
    args = parser.parse_args(argv)

    expected_digest = scene_digest(args.scene)
    pairs, missing_opengl, missing_reference = find_pairs()
    if missing_reference:
        print("OpenGL captures with no Cycles reference: " + ", ".join(missing_reference))
    if missing_opengl:
        print("Cycles references with no OpenGL capture (press P at those settings): "
              + ", ".join(missing_opengl))
    if not pairs:
        print("\nNo pairs yet. To make one:\n"
              "  1. blender --background --python-exit-code 1 \\\n"
              "       --python tools/raytraced_reference/render_dof.py\n"
              "  2. ./build/DepthResearch, press T then 6, then P\n"
              "  3. rerun this tool")
        return 1

    directory = Path(args.output)
    rows, records = [], []
    stale = []
    for tag, opengl_path, reference_path in pairs:
        problem = reference_scene_check(reference_path, expected_digest)
        if problem is not None:
            stale.append(f"{tag}: {problem}")
            if not args.allow_stale:
                continue
        opengl, reference = load_rgb(opengl_path), load_rgb(reference_path)
        if opengl.shape != reference.shape:
            print(f"{tag}: sizes differ ({opengl.shape[1]}x{opengl.shape[0]} against "
                  f"{reference.shape[1]}x{reference.shape[0]}); render both at the same "
                  f"resolution before comparing.")
            continue
        whole = metrics(reference, opengl)
        band = metrics(reference, opengl, edge_band_mask(reference, args.edge_width, args.edge_threshold))
        images = write_images(tag, opengl, reference, directory) if args.write_images else None
        records.append(
            {
                "tag": tag,
                "settings": describe_tag(tag),
                "whole_frame": whole,
                "edge_band": band,
                "opengl": str(opengl_path.relative_to(ROOT)),
                "reference": str(reference_path.relative_to(ROOT)),
                "expected_scene_sha256": expected_digest,
                "scene_hash_verified": problem is None,
                "verification_problem": problem,
                "images": images,
            }
        )
        rows.append((tag, whole, band, images))
        print(f"{tag:28s} whole frame PSNR {whole['psnr_db']:5.2f} dB  MAE {whole['mae']:5.2f}"
              + (f" | edges PSNR {band['psnr_db']:5.2f} dB  MAE {band['mae']:5.2f}" if band else ""))

    if stale:
        print("\nSame scene file check FAILED for:")
        for message in stale:
            print(f"  {message}")
        print("Re-render those references, or pass --allow-stale to compare anyway "
              "(the numbers will not mean anything).")
    if not records:
        return 1
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")

    lines = [
        "# OpenGL against Cycles: paired comparisons",
        "",
        "Generated by `python3 tools/compare/compare_renders.py --write-images`.",
        "PNG inspection metrics only; these references have no measured convergence certificate.",
        f"Expected scene: `{args.scene}`, sha256 `{expected_digest}`.",
        (
            "Some reference scene hashes were NOT verified; --allow-stale permitted the comparison."
            if any(not r["scene_hash_verified"] for r in records)
            else "Every compared reference sidecar carries the expected scene hash."
        ),
        "Camera pose, capture scene hash and radiometric equivalence are NOT established by filename pairing.",
        "Raster visibility, shadow maps, pixel filtering and lens integration can differ.",
        "",
        "| Settings | PSNR (frame) | MAE (frame) | PSNR (edges) | MAE (edges) | Images |",
        "|---|---|---|---|---|---|",
    ]
    for tag, whole, band, images in rows:
        links = f"[side]({images[0]}), [diff]({images[1]})" if images else "-"
        lines.append(f"| {describe_tag(tag)} | {whole['psnr_db']:.2f} dB | {whole['mae']:.2f} | "
                     + (f"{band['psnr_db']:.2f} dB | {band['mae']:.2f} | " if band else "- | - | ")
                     + links + " |")
    lines += ["",
              "Units are 0-255 levels. The edge columns restrict the same metrics to a band",
              f"{args.edge_width} px either side of a strong luminance edge in the reference, which is where",
              "the CoC gather is known to fail: it averages neighbours without testing their",
              "depth, so a sharp foreground bleeds into a blurred background and vice versa.",
              "Expect the edge numbers to be clearly worse than the whole-frame numbers. If they",
              "are not, the comparison is probably not aligned.", "",
              "Difference images are amplified 4x, as their filenames say.", ""]
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {directory / 'README.md'} and {directory / 'metrics.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
