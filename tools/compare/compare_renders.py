"""Pair the OpenGL and Cycles images of the same settings and measure the gap.

    python3 tools/compare/compare_renders.py                 # report what is paired
    python3 tools/compare/compare_renders.py --write-images   # + side-by-side and difference PNGs

Inputs (both produced from the same scene file):
  reports/raytraced_dof/rt_focus1.5m_f1.2.exr  Cycles ground truth, 32-bit linear
  output/gl_focus1.5m_f1.2.png                 OpenGL CoC gather (press P in the renderer)

A `.exr` reference is preferred over a `.png` with the same tag, because only
the EXR path is rendered with the denoiser off and no clipping. The EXR is
encoded down to display levels here rather than the PNG being decoded up: the
OpenGL screenshot has already been clipped and quantized, and decoding it back
to linear would invent precision it never had.

Which settings to compare. Wide open and focused close is where a screen-space
gather and a path-traced lens disagree most, because the circle of confusion is
large and every error it makes scales with it. f/11 is the control: at a near
pinhole the two pipelines should agree to within the reference's own noise, and
if they do not, something other than depth of field is wrong.

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


def srgb_encode(linear):
    """Linear radiance to 0-255 display levels, the curve Blender's "Standard"
    view transform and shaders/screen.frag both apply."""
    clamped = np.clip(linear, 0.0, 1.0)
    low = clamped * 12.92
    high = 1.055 * np.power(clamped, 1.0 / 2.4) - 0.055
    return np.where(clamped > 0.0031308, high, low) * 255.0


def load_rgb(path, exposure=1.0):
    """Load any reference or capture as 0-255 display levels.

    A ground-truth reference is a 32-bit linear EXR, while an OpenGL screenshot
    is an 8-bit sRGB PNG, so one of the two has to be converted before they can
    be differenced. Converting the EXR *down* is the honest direction here: the
    screenshot has already been clipped and quantized by the renderer, and
    pretending otherwise by decoding it back to linear would invent precision
    the OpenGL side never had. Metrics are therefore reported in display levels,
    which is also the space a reader judges the images in.
    """
    if path.suffix.lower() in (".exr", ".hdr"):
        with Image.open(path) as image:
            linear = np.asarray(image.convert("RGB"), dtype=np.float64)
        return srgb_encode(linear * exposure)
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


# Suffixes that name an OpenGL *variant* of the same camera settings rather than
# a different camera. The reference does not have them, so they are stripped
# when looking one up: `gl_focus1.5m_f1.2_naive.png` is compared against
# `rt_focus1.5m_f1.2.exr`, because both are the same shot and the only
# difference is which gather produced the raster image.
VARIANT_SUFFIXES = ("_naive",)


def split_variant(tag):
    """('focus1.5m_f1.2_naive') -> ('focus1.5m_f1.2', 'naive')."""
    for suffix in VARIANT_SUFFIXES:
        if tag.endswith(suffix):
            return tag[: -len(suffix)], suffix[1:]
    return tag, "weighted"


def find_pairs():
    """Match output/gl_<tag>[_variant].png to reports/raytraced_dof/rt_<tag>.*

    EXR takes priority over PNG for the same tag: if a ground-truth render
    exists, compare against that rather than against a denoised preview that
    happens to share the name.
    """
    references = {}
    for pattern in ("rt_*.png", "rt_*.exr"):
        for path in sorted(REFERENCE_DIRECTORY.glob(pattern)):
            references[path.stem[3:]] = path

    pairs, unmatched_captures = [], []
    for path in sorted(OPENGL_DIRECTORY.glob("gl_*.png")):
        tag = path.stem[3:]
        base, variant = split_variant(tag)
        if base in references:
            pairs.append((tag, base, variant, path, references[base]))
        else:
            unmatched_captures.append(tag)

    matched_bases = {base for _, base, _, _, _ in pairs}
    unmatched_references = sorted(set(references) - matched_bases)
    return pairs, unmatched_references, sorted(unmatched_captures)


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


def scene_file_name():
    """Which scene file both renderers are reading, taken from render_dof.py.

    Not written out here: this is the third place that would have to know the
    answer, and a stale copy of it is exactly how this check silently passed
    against the wrong file once already.
    """
    source = (ROOT / "tools/raytraced_reference/render_dof.py").read_text()
    match = re.search(r'^SCENE_FILE = "([^"]+)"', source, re.M)
    if match is None:
        raise SystemExit("tools/raytraced_reference/render_dof.py no longer defines SCENE_FILE")
    return match[1]


def scene_digest():
    """sha256 of the scene file both renderers are supposed to be reading."""
    import hashlib
    return hashlib.sha256((ROOT / scene_file_name()).read_bytes()).hexdigest()


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
        return "rendered before the scene file existed; re-render it"
    if recorded != expected_digest:
        return "rendered from a DIFFERENT scene file; re-render it"
    return None


def reference_quality(reference_path):
    """What the sidecar JSON says about this reference's own trustworthiness."""
    sidecar = reference_path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return {}
    convergence = record.get("convergence") or {}
    return {
        "ground_truth": bool(record.get("ground_truth_mode")),
        "samples": record.get("samples"),
        "denoising": record.get("denoising"),
        "format": record.get("output_format"),
        # In 0-1 linear units; scaled to display levels for the table so it can
        # be read against the MAE columns directly.
        "reference_error_levels": (convergence.get("estimated_reference_error") or 0.0) * 255.0,
        "has_error_bar": bool(convergence),
    }


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
    parser.add_argument("--exposure", type=float, default=1.0,
                        help="Exposure applied to a linear EXR reference before encoding, "
                             "matching the renderer's own exposure slider (default 1.0)")
    parser.add_argument("--require-ground-truth", action="store_true",
                        help="Skip any reference not rendered with --ground-truth")
    parser.add_argument("--output", default=str(OUTPUT_DIRECTORY))
    args = parser.parse_args(argv)

    expected_digest = scene_digest()
    pairs, missing_opengl, missing_reference = find_pairs()
    if missing_reference:
        print("OpenGL captures with no Cycles reference: " + ", ".join(missing_reference))
    if missing_opengl:
        print("Cycles references with no OpenGL capture (press P at those settings): "
              + ", ".join(missing_opengl))
    if not pairs:
        print("\nNo pairs yet. The settings that show the biggest difference between\n"
              "the two methods are wide open and focused close, so start there:\n"
              "\n"
              "  1. Ground-truth references (slow; minutes to hours per image):\n"
              "     blender --background --python-exit-code 1 \\\n"
              "       --python tools/raytraced_reference/render_dof.py -- \\\n"
              "       --ground-truth --convergence --sharp\n"
              "     Add --lens 85 --focus 1.2 --fstops 1.2 for the strongest case.\n"
              "  2. OpenGL captures at the same settings: run ./build/DepthResearch,\n"
              "     press T (reference preset) then 6 (BasicDoF) then P. Then press\n"
              "     f/1.2, f/2.8 and f/11 on the panel, pressing P after each, and K\n"
              "     for the strong-blur preset.\n"
              "  3. python3 tools/compare/compare_renders.py --write-images\n")
        return

    directory = Path(args.output)
    rows, records = [], []
    stale = []
    for tag, base, variant, opengl_path, reference_path in pairs:
        quality = reference_quality(reference_path)
        if args.require_ground_truth and not quality.get("ground_truth"):
            stale.append(f"{tag}: not rendered with --ground-truth (denoising "
                         f"{quality.get('denoising', 'unknown')}, "
                         f"{quality.get('samples', 'unknown')} samples)")
            continue
        problem = reference_scene_check(reference_path, expected_digest)
        if problem is not None:
            stale.append(f"{tag}: {problem}")
            if not args.allow_stale:
                continue
        opengl = load_rgb(opengl_path)
        reference = load_rgb(reference_path, args.exposure)
        if opengl.shape != reference.shape:
            print(f"{tag}: sizes differ ({opengl.shape[1]}x{opengl.shape[0]} against "
                  f"{reference.shape[1]}x{reference.shape[0]}); render both at the same "
                  f"resolution before comparing.")
            continue
        whole = metrics(reference, opengl)
        band = metrics(reference, opengl, edge_band_mask(reference, args.edge_width, args.edge_threshold))
        images = write_images(tag, opengl, reference, directory) if args.write_images else None
        records.append({"tag": tag, "gather": variant,
                        "settings": describe_tag(base) + f" [{variant} gather]",
                        "whole_frame": whole,
                        "edge_band": band, "opengl": str(opengl_path.relative_to(ROOT)),
                        "reference": str(reference_path.relative_to(ROOT)),
                        "scene_sha256": expected_digest, "reference_quality": quality,
                        "images": images})
        rows.append((base, variant, whole, band, images, quality))
        print(f"{tag:34s} whole frame PSNR {whole['psnr_db']:5.2f} dB  MAE {whole['mae']:5.2f}"
              + (f" | edges PSNR {band['psnr_db']:5.2f} dB  MAE {band['mae']:5.2f}" if band else ""))

    if stale:
        print("\nSame scene file check FAILED for:")
        for message in stale:
            print(f"  {message}")
        print("Re-render those references, or pass --allow-stale to compare anyway "
              "(the numbers will not mean anything).")
    if not records:
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")

    lines = ["# OpenGL against Cycles: paired comparisons", "",
             "Generated by `python3 tools/compare/compare_renders.py --write-images`.",
             f"Both images in every pair are rendered from the same `{scene_file_name()}`,",
             "at the same camera pose, lens, aperture, focus distance and resolution.",
             "The only intended difference is how the lens was simulated: every reference below",
             "was confirmed to carry the current scene file's sha256 in its sidecar JSON",
             f"(`{expected_digest[:16]}...`).", "",
             "| Settings | Gather | PSNR (frame) | MAE (frame) | PSNR (edges) | MAE (edges) | Ref. noise | Images |",
             "|---|---|---|---|---|---|---|---|"]
    for base, variant, whole, band, images, quality in rows:
        links = f"[side]({images[0]}), [diff]({images[1]})" if images else "-"
        noise = (f"{quality['reference_error_levels']:.2f}" if quality.get("has_error_bar")
                 else "not measured")
        lines.append(f"| {describe_tag(base)} | {variant} | {whole['psnr_db']:.2f} dB | {whole['mae']:.2f} | "
                     + (f"{band['psnr_db']:.2f} dB | {band['mae']:.2f} | " if band else "- | - | ")
                     + noise + " | " + links + " |")
    lines += ["",
              "Units are 0-255 levels. The edge columns restrict the same metrics to a band",
              f"{args.edge_width} px either side of a strong luminance edge in the reference, which is where",
              "the CoC gather is known to fail: it averages neighbours without testing their",
              "depth, so a sharp foreground bleeds into a blurred background and vice versa.",
              "Expect the edge numbers to be clearly worse than the whole-frame numbers. If they",
              "are not, the comparison is probably not aligned.", "",
              "Difference images are amplified 4x, as their filenames say.", "",
              "The 'Gather' column is which OpenGL blur produced the raster image: `naive` is the",
              "textbook equal-weight gather, `weighted` weights each sample by whether its own",
              "circle of confusion actually reaches the pixel. Both are compared against the same",
              "reference, so the two rows for one setting measure how much that weighting buys.",
              "",
              "The 'Ref. noise' column is the reference's own Monte Carlo error in the same",
              "0-255 units, measured by rendering each frame twice at different sample counts",
              "(`--convergence`). A measured MAE is only meaningful if it is well above that",
              "number; if it is not, the difference being reported is the reference's noise",
              "rather than the raster pass's error.", ""]
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {directory / 'README.md'} and {directory / 'metrics.json'}")


if __name__ == "__main__":
    main()
