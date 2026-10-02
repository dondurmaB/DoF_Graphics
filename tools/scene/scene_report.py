"""Inventory of a scene file: what is in it, how big, and at what depth.

    python3 tools/scene/scene_report.py                      # human-readable
    python3 tools/scene/scene_report.py --markdown            # table form
    python3 tools/scene/scene_report.py --markdown \
        --output notes/graphics/18_scene_inventory.md

Written to answer the plain question "how large are the things in this scene",
which matters more here than it sounds: the circle of confusion depends on
distance from the camera, so the physical size and depth of every prop is what
decides whether it is sharp, softly blurred, or a bokeh disc. Nothing in the
renderer reads this file; it only reports.

The report groups primitives by the section comments that
tools/scene/build_alley.py writes into the scene file, so the grouping stays
correct when the scene is regenerated.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("scene_loader", Path(__file__).with_name("scene_loader.py"))
scene_loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scene_loader)

SHAPE_NAMES = {"box": "box", "cyl": "cylinder", "sph": "sphere"}


def sections(scene_path):
    """Map each primitive index to the section comment above it.

    The loader deliberately throws comments away, so they are read again here.
    Primitive lines are counted in the same order the loader parses them, which
    is the order they appear in the file.
    """
    labels, current, index = {}, "unsorted", 0
    for raw in Path(scene_path).read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith("# ---"):
            current = stripped.lstrip("# -").rstrip(" -")
            continue
        head = stripped.split("#", 1)[0].split()
        if head and head[0] in SHAPE_NAMES:
            labels[index] = current
            index += 1
    return labels


def primitive_extent(primitive):
    """World-space size of one primitive, in meters.

    Unit shapes span 1 m before scaling, so `size` IS the extent along each
    local axis. Rotation is ignored on purpose: "a 1.2 x 0.8 x 0.9 m crate"
    describes the object better than its axis-aligned bounding box once it has
    been turned a few degrees.
    """
    return tuple(abs(value) for value in primitive.size)


def describe(primitive):
    extent = primitive_extent(primitive)
    if primitive.kind == "sph":
        return f"sphere d={extent[0]:.2f}"
    if primitive.kind == "cyl":
        return f"cylinder d={extent[0]:.2f} h={extent[1]:.2f}"
    return f"box {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f}"


def collect(scene, scene_path):
    labels = sections(scene_path)
    camera_z = scene.camera.position[2]
    groups = {}
    for index, primitive in enumerate(scene.primitives):
        label = labels.get(index, "unsorted")
        group = groups.setdefault(label, {
            "count": 0, "shapes": {}, "emissive": 0,
            "low": [math.inf] * 3, "high": [-math.inf] * 3,
            "near": math.inf, "far": -math.inf,
            "largest": None, "largest_volume": -1.0,
        })
        group["count"] += 1
        group["shapes"][SHAPE_NAMES[primitive.kind]] = \
            group["shapes"].get(SHAPE_NAMES[primitive.kind], 0) + 1
        if primitive.emit > 0.0:
            group["emissive"] += 1
        extent = primitive_extent(primitive)
        # Unrotated primitives (the overwhelming majority here) have exact
        # per-axis half extents. Only for a rotated one is the largest extent
        # used on every axis, which is conservative but keeps this honest
        # without duplicating the builder's transform maths.
        rotated = any(abs(angle) > 1e-9 for angle in primitive.rot)
        reach_per_axis = [0.5 * max(extent)] * 3 if rotated else [0.5 * value for value in extent]
        for axis in range(3):
            group["low"][axis] = min(group["low"][axis], primitive.pos[axis] - reach_per_axis[axis])
            group["high"][axis] = max(group["high"][axis], primitive.pos[axis] + reach_per_axis[axis])
        depth = camera_z - primitive.pos[2]
        group["near"] = min(group["near"], depth)
        group["far"] = max(group["far"], depth)
        volume = extent[0] * extent[1] * extent[2]
        if volume > group["largest_volume"]:
            group["largest_volume"] = volume
            group["largest"] = describe(primitive)
    return groups, camera_z


def coc_radius_pixels(scene, depth_m, frame_height=1200):
    """Background/foreground CoC radius in pixels at the scene's own lens settings."""
    camera = scene.camera
    focal = camera.focal_length_mm / 1000.0
    if depth_m <= 0.0 or camera.focus_distance_m <= focal:
        return 0.0
    aperture = focal / camera.f_number
    coc_m = (aperture * focal * (depth_m - camera.focus_distance_m)) / \
            (depth_m * (camera.focus_distance_m - focal))
    return abs(coc_m / (camera.sensor_height_mm / 1000.0) * frame_height) / 2.0


def report(scene_path, frame_height, markdown):
    scene = scene_loader.load_scene(scene_path)
    geometry = scene_loader.build_geometry(scene)
    summary = scene_loader.summary(scene, geometry)
    groups, camera_z = collect(scene, scene_path)
    camera = scene.camera
    lines = []
    out = lines.append

    half_angle = 0.5 * camera.sensor_height_mm / camera.focal_length_mm
    if markdown:
        out("# Scene inventory: `scene/alley.scene`")
        out("")
        out("Generated by `python3 tools/scene/scene_report.py --markdown`. Regenerate after")
        out("changing the scene; do not edit by hand.")
        out("")
        out("## Camera and lens")
        out("")
        out("| Property | Value |")
        out("|---|---|")
        out(f"| Position | ({camera.position[0]:g}, {camera.position[1]:g}, {camera.position[2]:g}) m |")
        out(f"| Lens | {camera.focal_length_mm:g} mm on a {camera.sensor_height_mm:g} mm sensor height |")
        out(f"| Vertical field of view | {math.degrees(2 * math.atan(half_angle)):.2f} degrees |")
        out(f"| Focus distance | {camera.focus_distance_m:g} m |")
        out(f"| Aperture | f/{camera.f_number:g} (entrance pupil "
            f"{camera.focal_length_mm / camera.f_number:.1f} mm) |")
        out("")
        out("The narrow field of view is the single most important number for reading this scene:")
        out(f"the visible half-height at distance *d* is only **{half_angle:.3f} x d** metres, so at 2 m")
        out(f"the frame is {2 * half_angle * 2:.2f} m tall and at 20 m it is {2 * half_angle * 20:.1f} m tall.")
        out("Objects have to be placed close to eye height to be in frame at all up close.")
        out("")
        out("## Totals")
        out("")
        out("| Quantity | Value |")
        out("|---|---|")
        out(f"| Primitives (boxes, cylinders, spheres) | {summary['primitives']} |")
        out(f"| Triangles | {summary['triangles']} |")
        out(f"| Vertices | {summary['vertices']} |")
        out(f"| Emissive triangles | {summary['emissive_triangles']} |")
        out(f"| Smooth-shaded triangles | {summary['smooth_triangles']} |")
        out(f"| Total surface area | {summary['area_sum']:.0f} m^2 |")
        out(f"| Extent (x, y, z) | {summary['bounds_max'][0] - summary['bounds_min'][0]:.1f} x "
            f"{summary['bounds_max'][1] - summary['bounds_min'][1]:.1f} x "
            f"{summary['bounds_max'][2] - summary['bounds_min'][2]:.1f} m |")
        out("")
        out("## Sections, sizes and depths")
        out("")
        out("Depth is measured from the camera along -Z. The CoC column is the blur radius those")
        out("depths produce at the lens settings above, on a 1200 px frame; anything under about")
        out("1 px reads as sharp.")
        out("")
        out("| Section | Prims | Shapes | Depth from camera | Largest item (m) | CoC radius |")
        out("|---|---|---|---|---|---|")
        for label, group in groups.items():
            shapes = ", ".join(f"{count} {name}" for name, count in sorted(group["shapes"].items()))
            emissive = f" ({group['emissive']} lit)" if group["emissive"] else ""
            # Clamped to the near plane: parts of the floor and walls extend
            # behind the camera, and the CoC formula diverges as depth goes to
            # zero, which would print a meaningless four-digit number.
            near_coc = coc_radius_pixels(scene, max(group["near"], 0.35), frame_height)
            far_coc = coc_radius_pixels(scene, max(group["far"], 0.35), frame_height)
            low, high = min(near_coc, far_coc), max(near_coc, far_coc)
            coc = f"{low:.1f}-{high:.1f} px" if high - low > 0.2 else f"{high:.1f} px"
            out(f"| {label} | {group['count']} | {shapes}{emissive} | "
                f"{group['near']:.1f} to {group['far']:.1f} m | {group['largest']} | {coc} |")
        out("")
        out("## Lighting")
        out("")
        sun = scene.sun
        out("| Property | Value |")
        out("|---|---|")
        out(f"| Sun direction (surface toward sun) | ({sun.direction[0]:g}, {sun.direction[1]:g}, "
            f"{sun.direction[2]:g}) |")
        out(f"| Sun irradiance | {sun.energy:g} W/m^2 |")
        out(f"| Sun colour (linear) | ({sun.color[0]:g}, {sun.color[1]:g}, {sun.color[2]:g}) |")
        out(f"| Sun angular diameter | {sun.angular_diameter_degrees:g} degrees |")
        out(f"| Sky radiance | ({scene.sky_radiance[0]:.4f}, {scene.sky_radiance[1]:.4f}, "
            f"{scene.sky_radiance[2]:.4f}) |")
        out(f"| Shadow map region | {scene.shadow_low} to {scene.shadow_high} |")
        out("")
        out("Both renderers read every one of these from the scene file. The sky radiance is")
        out("simultaneously the OpenGL clear colour, the OpenGL ambient term and the Cycles world")
        out("background, which is why the two images share a background colour exactly.")
    else:
        out(f"{scene_path}")
        out(f"  camera      (0, 0, {camera.position[2]:g}) m, {camera.focal_length_mm:g} mm, "
            f"f/{camera.f_number:g}, focus {camera.focus_distance_m:g} m")
        out(f"  frame       half-height {half_angle:.3f} x depth  "
            f"(at 5 m: {half_angle * 5:.2f} m up and down)")
        out(f"  totals      {summary['primitives']} primitives, {summary['triangles']} triangles, "
            f"{summary['area_sum']:.0f} m^2 of surface")
        out("")
        for label, group in groups.items():
            shapes = ", ".join(f"{count} {name}" for name, count in sorted(group["shapes"].items()))
            out(f"  {label}")
            out(f"    {group['count']} primitives ({shapes})"
                + (f", {group['emissive']} emissive" if group["emissive"] else ""))
            out(f"    depth {group['near']:.1f} to {group['far']:.1f} m from the camera")
            out(f"    x {group['low'][0]:.1f} to {group['high'][0]:.1f} m, "
                f"y {group['low'][1]:.1f} to {group['high'][1]:.1f} m")
            out(f"    largest: {group['largest']}")
            out("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=str(ROOT / "scene/alley.scene"))
    parser.add_argument("--frame-height", type=int, default=1200,
                        help="Frame height in pixels used for the CoC column (default 1200)")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--output", help="Write to this file instead of stdout")
    args = parser.parse_args(argv)
    text = report(args.scene, args.frame_height, args.markdown)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Wrote {path}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
