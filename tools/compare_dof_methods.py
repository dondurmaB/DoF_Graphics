#!/usr/bin/env python3
"""Compare the three depth-of-field methods against the ray-traced reference.

The three DoFApproaches methods render the same analytic scene from the same
camera, so their images are directly subtractable. Method 3 at a high sample
count is the reference: it solves visibility per lens sample, so it is the only
one of the three with no structural approximation left in it.

Usage:
    python3 tools/compare_dof_methods.py            # uses output/dof3-*.png
    python3 tools/compare_dof_methods.py --amplify 6

Writes amplified difference maps next to the inputs and prints a metric table,
also saved as output/dof3-comparison.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Boxes are in captured-pixel coordinates for the canonical 1280x860 request,
# which lands as 2560x1688 on a Retina framebuffer. They are rescaled below if
# the actual capture size differs.
REFERENCE_SIZE = (2560, 1688)
ZONES = {
    # Bright far wall seen behind the razor-sharp railing. Any difference here
    # is halo: the railing is on the focus plane and cannot legally spread.
    "zone_a_halo": (1075, 60, 2477, 800),
    # The thin near posts, which a finite aperture must see through.
    "zone_b_near_occluder": (200, 100, 900, 1600),
}


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def scale_box(box, size):
    sx = size[0] / REFERENCE_SIZE[0]
    sy = size[1] / REFERENCE_SIZE[1]
    x0, y0, x1, y1 = box
    return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="output", type=Path)
    parser.add_argument("--reference", default="dof3-m3-raytrace-512.png")
    parser.add_argument("--amplify", type=float, default=4.0)
    args = parser.parse_args()

    reference_path = args.dir / args.reference
    reference = load(reference_path)
    height, width, _ = reference.shape
    size = (width, height)
    luminance = reference @ np.array([0.2126, 0.7152, 0.0722])
    # Defocused highlights: where bokeh shape and energy are decided.
    highlights = luminance > 190.0

    candidates = [
        ("m1_gather", "dof3-m1-gather.png"),
        ("m2_multiview_8", "dof3-m2-multiview-8.png"),
        ("m2_multiview_64", "dof3-m2-multiview-64.png"),
        ("m3_raytrace_32", "dof3-m3-raytrace-32.png"),
    ]

    results = {"reference": str(reference_path), "capture_size": [width, height], "methods": {}}
    print(f"reference: {reference_path.name}  ({width}x{height})\n")
    header = f"{'method':22s} {'overall':>9s} {'zone A halo':>12s} {'zone B near':>12s} {'highlights':>11s}"
    print(header)
    print("-" * len(header))

    for key, name in candidates:
        path = args.dir / name
        if not path.exists():
            print(f"{key:22s} {'missing':>9s}")
            continue
        image = load(path)
        if image.shape != reference.shape:
            print(f"{key:22s} size mismatch {image.shape} vs {reference.shape}")
            continue

        row = {"file": name, "overall_mae": mae(image, reference)}
        for zone, box in ZONES.items():
            x0, y0, x1, y1 = scale_box(box, size)
            row[zone + "_mae"] = mae(image[y0:y1, x0:x1], reference[y0:y1, x0:x1])
        row["highlight_mae"] = mae(image[highlights], reference[highlights])
        results["methods"][key] = row

        print(f"{key:22s} {row['overall_mae']:9.3f} {row['zone_a_halo_mae']:12.3f} "
              f"{row['zone_b_near_occluder_mae']:12.3f} {row['highlight_mae']:11.3f}")

        difference = np.clip(np.abs(image - reference) * args.amplify, 0.0, 255.0)
        out = args.dir / f"dof3-diff-{key}-vs-reference.png"
        Image.fromarray(difference.astype(np.uint8)).save(out)
        row["difference_map"] = out.name

    print("\nAll values are mean absolute error against the reference, 0..255 per channel.")
    print(f"Difference maps written with amplification x{args.amplify:g}.")

    out_json = args.dir / "dof3-comparison.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
