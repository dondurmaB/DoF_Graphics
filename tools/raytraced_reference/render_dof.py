"""Run with Blender 4.2+; --dry-run validates the reference plan with ordinary Python."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

# Sample counts. "Ground truth" is a claim about convergence, not about using a
# path tracer, so there are three tiers and the default for a reference is the
# converged one:
#   PREVIEW   quick look, denoised, never a reference
#   FINAL     the old default; fine for a visual check, not for training data
#   TRUTH     --ground-truth: denoiser OFF, 4096 samples, 32-bit linear EXR
# A denoiser invents plausible detail. Against a denoised image, a network is
# being trained to match a guess, and a bokeh disc is exactly the kind of
# smooth region a denoiser rewrites. So ground truth means no denoiser and
# enough samples that the residual noise is below the error we care about.
PREVIEW_SAMPLES = 32
FINAL_SAMPLES = 128
TRUTH_SAMPLES = 4096
# Light paths. The defaults are tuned for speed, not correctness; a reference
# should not lose energy to a bounce limit it happens to hit.
TRUTH_MAX_BOUNCES = 32
TRUTH_DIFFUSE_BOUNCES = 16
REFERENCE_WIDTH = 1200  # Framebuffer pixels, not the 600-point macOS window size.
REFERENCE_HEIGHT = 1200
FOCAL_LENGTH_MM = 50.0
SENSOR_HEIGHT_MM = 24.0
# The cafe's focus plane is the cup at 1.5 m. The f-stop list is the set that
# makes the two methods differ most at one end and agree at the other: f/1.2 is
# the hardest case for a gather, f/11 is nearly pinhole and is the control that
# proves the two pipelines agree when depth of field is not in play.
FOCUS_DISTANCES_M = (1.5,)
F_STOPS = (1.2, 1.4, 2.8, 11.0)
IMPORTED_SCENE_SCALE = 0.1
IMPORTED_SCENE_POSITION_GL = (0.0, -0.75, 0.0)
IMPORTED_ROTATION_X_DEG = -90.0
IMPORTED_ROTATION_Y_DEG = 0.0
# Linear albedo for the imported mesh, matching importedSceneAlbedo in src/main.cpp.
# The OBJ carries no usable material, so both renderers assign one explicitly.
IMPORTED_ALBEDO = (0.62, 0.44, 0.20)
CAMERA_POSITION_GL = (0.0, 0.0, 5.0)
# Yaw and pitch, matching the `camera` line in scene/cafe.scene and
# updateCameraFront() in src/main.cpp. The 10-degree downward pitch is what puts
# the table in frame; with no pitch a surface has to sit within 20 cm of eye
# height to be visible at 1.5 m.
CAMERA_YAW_DEG = -90.0
CAMERA_PITCH_DEG = -10.0
CAMERA_UP_GL = (0.0, 1.0, 0.0)
NEAR_M, FAR_M = 0.1, 100.0

# One proper rotation for every world-space object: (x,y,z) -> (x,-z,y).
WORLD_CONVERSION = ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1))

# The environment is no longer written out here. It comes from the same file the
# OpenGL renderer loads, so the two cannot describe different scenes:
#   scene/cafe.scene  ->  tools/scene/scene_loader.py  (this script)
#                     ->  src/SceneFile.cpp            (src/main.cpp)
# The camera and lens literals above stay because argparse needs defaults;
# tests/test_reference_config.py checks they still agree with the scene file
# and with the fallbacks in src/main.cpp.
SCENE_FILE = "scene/cafe.scene"


def load_shared_scene():
    """Parse scene/alley.scene with the shared loader. Needs no Blender."""
    root = project_root()
    loader_directory = str(root / "tools/scene")
    if loader_directory not in sys.path:
        sys.path.insert(0, loader_directory)
    import scene_loader

    scene = scene_loader.load_scene(root / SCENE_FILE)
    return scene_loader, scene, scene_loader.build_geometry(scene)


def camera_forward_gl(yaw_degrees, pitch_degrees):
    """Same yaw/pitch convention as updateCameraFront() in src/main.cpp."""
    yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
    return (math.cos(yaw) * math.cos(pitch), math.sin(pitch), math.sin(yaw) * math.cos(pitch))


# Derived, not written out twice: the scene file carries yaw and pitch, and this
# is the same formula the renderer uses, so the two cannot disagree.
CAMERA_FORWARD_GL = camera_forward_gl(CAMERA_YAW_DEG, CAMERA_PITCH_DEG)


def project_root():
    # Also works for the archived copy; the asset stays at the real repository root.
    for parent in Path(__file__).resolve().parents:
        if (parent / "assets/models/scene.obj").is_file() and (parent / "CMakeLists.txt").is_file():
            return parent
    raise FileNotFoundError("Run inside the project checkout with assets/models/scene.obj supplied.")


@contextmanager
def geometry_only_obj(source):
    """Keep every non-material byte; remove the temporary import even on failure."""
    # Blender 5.2's OBJ operator has no option to skip external material libraries.
    with tempfile.TemporaryDirectory(prefix="dof-reference-obj-") as directory:
        sanitized = Path(directory) / source.name
        removed = 0
        with source.open("rb") as original, sanitized.open("wb") as target:
            for line in original:
                directive = line.split(maxsplit=1)
                if directive and directive[0] in (b"mtllib", b"usemtl"):
                    removed += 1
                else:
                    target.write(line)
        if removed:
            print("OBJ material directives omitted; importing geometry without external materials.", flush=True)
        yield sanitized


def multiply(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4))


def model_matrix(position, rx, ry, scale, imported=False):
    """Exact OpenGL order: boxes T*Ry*Rx*S; imported mesh T*Rx*Ry*S."""
    x, y = math.radians(rx), math.radians(ry)
    cx, sx, cy, sy = math.cos(x), math.sin(x), math.cos(y), math.sin(y)
    rotate_x = ((1, 0, 0, 0), (0, cx, -sx, 0), (0, sx, cx, 0), (0, 0, 0, 1))
    rotate_y = ((cy, 0, sy, 0), (0, 1, 0, 0), (-sy, 0, cy, 0), (0, 0, 0, 1))
    translate = ((1, 0, 0, position[0]), (0, 1, 0, position[1]), (0, 0, 1, position[2]), (0, 0, 0, 1))
    scaling = ((scale[0], 0, 0, 0), (0, scale[1], 0, 0), (0, 0, scale[2], 0), (0, 0, 0, 1))
    rotation = multiply(rotate_x, rotate_y) if imported else multiply(rotate_y, rotate_x)
    return multiply(multiply(translate, rotation), scaling)


def gl_to_blender(vector):
    x, y, z = vector
    return (x, -z, y)


def vertical_fov_degrees(lens, sensor_height):
    if not all(math.isfinite(v) and v > 0 for v in (lens, sensor_height)):
        raise ValueError("Lens and sensor height must be finite and positive.")
    return math.degrees(2.0 * math.atan(sensor_height / (2.0 * lens)))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=REFERENCE_WIDTH)
    parser.add_argument("--height", type=int, default=REFERENCE_HEIGHT)
    parser.add_argument("--lens", type=float, default=FOCAL_LENGTH_MM)
    parser.add_argument("--sensor-height", type=float, default=SENSOR_HEIGHT_MM)
    parser.add_argument("--focus", type=float, nargs="+", default=list(FOCUS_DISTANCES_M))
    parser.add_argument("--fstops", type=float, nargs="+", default=list(F_STOPS))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--samples", type=int, help="Override preview/final sample count")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--sharp", action="store_true", help="Also render rt_sharp.png with lens DoF disabled")
    parser.add_argument("--ground-truth", action="store_true",
                        help="Reference quality: denoiser off, %d samples, 32-bit linear EXR, "
                             "raised bounce limits. Slow, and the only output suitable as "
                             "training ground truth." % TRUTH_SAMPLES)
    parser.add_argument("--convergence", action="store_true",
                        help="Also render each job at half the sample count and report the "
                             "difference, so the reference carries a stated error bar")
    parser.add_argument("--full-gi", action="store_true",
                        help="Let emitters light the scene. Physically complete, but then the "
                             "raster comparison measures global illumination as well as defocus")
    parser.add_argument("--imported-mesh", action="store_true",
                        help="Also load assets/models/scene.obj. Off by default: the cafe has "
                             "its own subject and the teapot was only ever a stand-in")
    parser.add_argument("--seed", type=int, default=16,
                        help="Cycles sampling seed; fixed so a reference is reproducible")
    parser.add_argument("--dry-run", action="store_true", help="Print validated settings; no Blender import or output files")
    args = parser.parse_args(argv)
    if not (16 <= args.width <= 8192 and 16 <= args.height <= 8192):
        parser.error("Width and height must be between 16 and 8192 framebuffer pixels.")
    if not all(math.isfinite(v) for v in (args.lens, args.sensor_height, *args.focus, *args.fstops)):
        parser.error("Camera settings must be finite.")
    if not (1 <= args.lens <= 500 and 1 <= args.sensor_height <= 100):
        parser.error("Lens must be 1–500 mm and sensor height 1–100 mm.")
    if any(v <= args.lens * 0.001 for v in args.focus) or any(v < 0.1 for v in args.fstops):
        parser.error("Focus must exceed focal length in meters; f-stops must be at least 0.1.")
    if len(args.focus) * len(args.fstops) + int(args.sharp) > 12:
        parser.error("Use at most 12 renders in this controlled comparison.")
    if args.preview and args.ground_truth:
        parser.error("--preview and --ground-truth ask for opposite things.")
    if args.samples is None:
        args.samples = (TRUTH_SAMPLES if args.ground_truth
                        else PREVIEW_SAMPLES if args.preview else FINAL_SAMPLES)
    if not 1 <= args.samples <= 65536:
        parser.error("Samples must be between 1 and 65536.")
    if args.convergence and args.samples < 8:
        parser.error("A convergence check needs at least 8 samples to halve.")
    # Denoising is the one setting that disqualifies an image as ground truth,
    # so it is tied to the mode rather than left as a separate flag to forget.
    args.denoise = not args.ground_truth
    args.exr = args.ground_truth
    return args


def render_jobs(args):
    # Extension follows the mode, so a ground-truth EXR can never be mistaken
    # for a display-ready PNG by name alone.
    suffix = ".exr" if getattr(args, "exr", False) else ".png"
    jobs = [(f"rt_focus{focus:g}m_f{fstop:g}{suffix}", focus, fstop, True)
            for focus in args.focus for fstop in args.fstops]
    if args.sharp:
        jobs.append((f"rt_sharp{suffix}", args.focus[0], args.fstops[0], False))
    if len({job[0] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate output names; choose distinct focus distances and f-stops.")
    return jobs


def compare_passes(full_path, half_path, samples):
    """Difference between the full-sample render and a half-sample one.

    Reads both images without Blender so the numbers are independent of the
    renderer that produced them. Returns None, with a printed reason, if numpy
    cannot read the format; a missing error bar is worth saying out loud rather
    than silently reporting zero.
    """
    try:
        import numpy as np
    except ImportError:
        print("  convergence: numpy not available inside Blender; skipped", flush=True)
        return None

    def load(path):
        if path.suffix.lower() == ".exr":
            try:
                import OpenEXR  # noqa: F401
            except ImportError:
                # Blender can always read what it just wrote, and bpy is already
                # imported by the caller, so use it as the EXR reader.
                import bpy
                image = bpy.data.images.load(str(path))
                try:
                    pixels = np.array(image.pixels[:], dtype=np.float64)
                    return pixels.reshape(-1, 4)[:, :3]
                finally:
                    bpy.data.images.remove(image)
        try:
            from PIL import Image
        except ImportError:
            print("  convergence: no reader for this format; skipped", flush=True)
            return None
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0

    full, half = load(full_path), load(half_path)
    if full is None or half is None or full.shape != half.shape:
        return None
    difference = np.abs(full - half)
    mean = float(difference.mean())
    return {
        "half_samples": max(1, samples // 2),
        "mean_abs_difference": mean,
        "max_abs_difference": float(difference.max()),
        "p99_abs_difference": float(np.percentile(difference, 99)),
        # Two independent estimates differ by about sqrt(2) times the error of
        # the better one, so this is the figure to quote for the reference.
        "estimated_reference_error": mean / 2.0 ** 0.5,
        "units": "linear radiance" if full_path.suffix.lower() == ".exr" else "0-1 display",
        "note": "two independent Monte Carlo estimates; the full-sample image's own "
                "error is about 1/sqrt(2) of the mean difference",
    }


def make_plan(args):
    root = project_root()
    loader, scene, geometry = load_shared_scene()
    summary = loader.summary(scene, geometry)
    scene_summary = {
        "primitives": summary["primitives"],
        "vertices": summary["vertices"],
        "triangles": summary["triangles"],
        "emissive_triangles": summary["emissive_triangles"],
        "bounds_min": list(summary["bounds_min"]),
        "bounds_max": list(summary["bounds_max"]),
        "sun_direction_gl": list(scene.sun.direction),
        "sun_energy_w_per_m2": scene.sun.energy,
        "sun_angular_diameter_degrees": scene.sun.angular_diameter_degrees,
        "sky_radiance": list(scene.sky_radiance),
    }
    scene_sky = list(scene.sky_radiance)
    return {
        "asset": "assets/models/scene.obj",
        "asset_sha256": hashlib.sha256((root / "assets/models/scene.obj").read_bytes()).hexdigest(),
        "output_directory": "reports/raytraced_dof",
        "resolution_pixels": [args.width, args.height], "samples": args.samples,
        "requested_device": args.device, "engine": "CYCLES",
        "focal_length_mm": args.lens, "sensor_height_mm": args.sensor_height,
        "sensor_fit": "VERTICAL", "effective_sensor_width_mm": args.sensor_height * args.width / args.height,
        "vertical_fov_degrees": vertical_fov_degrees(args.lens, args.sensor_height),
        "camera_position_gl": CAMERA_POSITION_GL, "camera_forward_gl": CAMERA_FORWARD_GL,
        "camera_up_gl": CAMERA_UP_GL, "camera_position_blender": gl_to_blender(CAMERA_POSITION_GL),
        "camera_forward_blender": gl_to_blender(CAMERA_FORWARD_GL), "camera_up_blender": gl_to_blender(CAMERA_UP_GL),
        "world_conversion": WORLD_CONVERSION, "import_scale": IMPORTED_SCENE_SCALE,
        "import_matrix_gl": model_matrix(IMPORTED_SCENE_POSITION_GL, IMPORTED_ROTATION_X_DEG,
                                         IMPORTED_ROTATION_Y_DEG, (IMPORTED_SCENE_SCALE,) * 3, imported=True),
        "outputs": [job[0] for job in render_jobs(args)],
        # The scene digest goes in every sidecar JSON. Without it a reference PNG
        # cannot be tied to the geometry it was rendered from, which is the one
        # thing that would quietly invalidate a comparison months later.
        "scene_file": SCENE_FILE,
        "scene_sha256": hashlib.sha256((root / SCENE_FILE).read_bytes()).hexdigest(),
        "scene_summary": scene_summary,
        # Everything that decides whether this image can be called ground truth.
        # Recorded per render so a stored reference can be audited later instead
        # of being trusted.
        "ground_truth_mode": args.ground_truth,
        "denoising": "OPENIMAGEDENOISE" if args.denoise else "none",
        "output_format": "OPEN_EXR 32-bit linear" if args.exr else "PNG 8-bit sRGB",
        "sampling_seed": args.seed,
        "emitters_light_scene": args.full_gi,
        "shading_model": ("full global illumination" if args.full_gi else
                          "matched to basic.frag: albedo * (sky + sun * N.L / pi), "
                          "direct only, unoccluded ambient fill as emission"),
        "ambient_fill_radiance": list(scene_sky),
        "imported_mesh": args.imported_mesh,
        "max_bounces": TRUTH_MAX_BOUNCES if args.ground_truth else "cycles default",
        "sample_clamping": "disabled" if args.ground_truth else "cycles default",
        "adaptive_sampling": "disabled" if args.ground_truth else "cycles default",
        "suitable_as_training_ground_truth": bool(args.ground_truth),
    }


def select_device(bpy, scene, requested):
    scene.cycles.device = "CPU"
    if requested == "auto":
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = "METAL"
            preferences.refresh_devices()
            metal = [device for device in preferences.devices if device.type == "METAL"]
            if metal:
                for device in preferences.devices:
                    device.use = device.type == "METAL"
                scene.cycles.device = "GPU"
                return "METAL: " + ", ".join(device.name for device in metal)
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            print(f"Metal unavailable ({error}); using CPU.", flush=True)
    return "CPU"


def create_scene(bpy, args, plan):
    from mathutils import Matrix, Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"  # This is the lens-sampled path-traced reference.
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed
    scene.cycles.use_animated_seed = False
    # Denoising OFF for ground truth. A denoiser is a plausible-detail generator:
    # training against its output teaches a network to match a guess, and smooth
    # out-of-focus discs are exactly what it rewrites most.
    scene.cycles.use_denoising = args.denoise
    if args.denoise:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    if args.ground_truth:
        # Adaptive sampling stops early where it thinks it has converged, which
        # makes the noise floor vary across the frame and therefore unknown.
        # A reference wants a uniform, stated sample count everywhere.
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = False
        # Never clamp: clamping is a variance-reduction cheat that silently
        # removes energy from exactly the bright highlights being measured.
        for attribute in ("sample_clamp_direct", "sample_clamp_indirect"):
            if hasattr(scene.cycles, attribute):
                setattr(scene.cycles, attribute, 0.0)
        scene.cycles.max_bounces = TRUTH_MAX_BOUNCES
        scene.cycles.diffuse_bounces = TRUTH_DIFFUSE_BOUNCES
        for attribute, value in (("glossy_bounces", 8), ("transmission_bounces", 8),
                                 ("volume_bounces", 2), ("transparent_max_bounces", 16)):
            if hasattr(scene.cycles, attribute):
                setattr(scene.cycles, attribute, value)
        # Light tree and blue-noise sampling where available: lower variance at
        # the same sample count, with no bias.
        for attribute in ("use_light_tree", "use_preview_adaptive_sampling"):
            if hasattr(scene.cycles, attribute):
                setattr(scene.cycles, attribute, attribute == "use_light_tree")
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    if args.exr:
        # 32-bit linear EXR, no view transform. An 8-bit PNG through the sRGB
        # curve clips every value above 1.0 and quantizes to 256 levels, which
        # throws away the highlights whose bokeh is the measurement. The raster
        # pass renders to RGBA16F for the same reason.
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "32"
        if hasattr(scene.render.image_settings, "exr_codec"):
            scene.render.image_settings.exr_codec = "ZIP"  # Lossless.
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_compositing = False  # Bypass compositing; DoF comes from lens sampling.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if not args.full_gi:
        # Direct lighting only. The ambient fill is an emission on every
        # surface, so allowing diffuse bounces would scatter it around the room
        # and make Cycles brighter than the raster pass by an amount that
        # depends on the geometry. Zero bounces makes the shading model exactly
        # albedo * (sky + sun * N.L / pi), which is what basic.frag computes.
        scene.cycles.diffuse_bounces = 0
        for attribute in ("glossy_bounces", "transmission_bounces", "volume_bounces"):
            if hasattr(scene.cycles, attribute):
                setattr(scene.cycles, attribute, 0)
        scene.cycles.max_bounces = 1

    conversion = Matrix(WORLD_CONVERSION)

    def new_material(name):
        result = bpy.data.materials.new(name)
        if bpy.app.version < (5, 0, 0):
            result.use_nodes = True  # Blender 5+ creates the node tree automatically.
        result.node_tree.nodes.clear()
        return result

    # The ambient fill, as a radiance. basic.frag adds albedo * sky_radiance to
    # every surface with NO occlusion, and Cycles has no equivalent: its world
    # background cannot reach an enclosed room, so the cafe reference rendered
    # black while the OpenGL image looked correct. The fix is to give every
    # Cycles material the same term explicitly, as an emission of
    # albedo * sky_radiance added to the Diffuse BSDF, and then to switch
    # diffuse bounces off so it is not also bounced around as indirect light.
    #
    # This is a deliberately non-physical fill in BOTH renderers, and that is
    # the point: the experiment measures the lens, so the shading model is
    # pinned to be identical and the only difference left is how depth of field
    # was computed. --full-gi swaps it for real global illumination, at the cost
    # of the two images no longer being comparable.
    sky = tuple(shared_scene.sky_radiance)

    def add_ambient_fill(result, albedo_output):
        """Emission of albedo * sky, added to whatever surface shader is there."""
        nodes, links = result.node_tree.nodes, result.node_tree.links
        surface = nodes["__surface__"]
        output = nodes["__output__"]
        # VectorMath rather than MixRGB: ShaderNodeMixRGB was replaced in
        # Blender 4.x and this node has been stable across every version.
        multiply = nodes.new("ShaderNodeVectorMath")
        multiply.operation = "MULTIPLY"
        multiply.inputs[1].default_value = sky
        links.new(albedo_output, multiply.inputs[0])
        fill = nodes.new("ShaderNodeEmission")
        fill.inputs["Strength"].default_value = 1.0
        links.new(multiply.outputs["Vector"], fill.inputs["Color"])
        combine = nodes.new("ShaderNodeAddShader")
        links.new(surface.outputs[0], combine.inputs[0])
        links.new(fill.outputs[0], combine.inputs[1])
        links.new(combine.outputs[0], output.inputs["Surface"])

    def flat_material(name, color):
        """One albedo for the whole object; used for the imported mesh."""
        result = new_material(name)
        result.diffuse_color = (*color, 1.0)
        nodes = result.node_tree.nodes
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.name = "__surface__"
        diffuse.inputs["Color"].default_value = (*color, 1.0)
        constant = nodes.new("ShaderNodeRGB")
        constant.outputs[0].default_value = (*color, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        output.name = "__output__"
        result.node_tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        if not args.full_gi:
            add_ambient_fill(result, constant.outputs[0])
        return result

    def vertex_color_material(name, emission):
        """Reads the per-vertex colour the scene file supplies.

        A Diffuse BSDF for surfaces, matching basic.frag's albedo/pi Lambert
        term, plus the ambient fill above; an Emission shader at strength 1.0
        for emitters, whose colour the builder has already premultiplied by
        `emit` so both renderers emit the same radiance. Emitters get no fill:
        basic.frag returns their radiance directly without a shading term.
        """
        result = new_material(name)
        nodes = result.node_tree.nodes
        attribute = nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = "Col"
        shader = nodes.new("ShaderNodeEmission" if emission else "ShaderNodeBsdfDiffuse")
        shader.name = "__surface__"
        if emission:
            shader.inputs["Strength"].default_value = 1.0
        output = nodes.new("ShaderNodeOutputMaterial")
        output.name = "__output__"
        links = result.node_tree.links
        links.new(attribute.outputs["Color"], shader.inputs["Color"])
        links.new(shader.outputs[0], output.inputs["Surface"])
        if not emission and not args.full_gi:
            add_ambient_fill(result, attribute.outputs["Color"])
        return result

    # The teapot is opt-in now. It was a stand-in subject before the cafe had a
    # real one at the focus plane; the import path stays because generic OBJ
    # loading is still useful, and the plan records the asset hash either way.
    if args.imported_mesh:
        with geometry_only_obj(project_root() / plan["asset"]) as obj_path:
            bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="Y", up_axis="Z",
                                  global_scale=1.0, clamp_size=0.0,
                                  use_split_objects=False, use_split_groups=False)
        imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        if not imported:
            raise RuntimeError("OBJ import produced no mesh; no fallback is allowed "
                               "for reference renders.")
        subject_material = flat_material("imported_subject", IMPORTED_ALBEDO)
        for obj in imported:
            obj.matrix_world = conversion @ Matrix(plan["import_matrix_gl"]) @ obj.matrix_world
            obj.data.materials.clear()
            obj.data.materials.append(subject_material)
            for face in obj.data.polygons:
                face.material_index = 0
            obj.data.calc_loop_triangles()
            print(f"Imported {obj.name}: {len(obj.data.vertices)} positions, "
                  f"{len(obj.data.loop_triangles)} triangles", flush=True)

    # ---- The cafe, from the shared scene file ----
    # Two objects, because one carries a Diffuse BSDF and the other an Emission
    # shader. The OpenGL pass keeps a single mesh and branches on a per-vertex
    # emission attribute instead; the geometry itself is identical because both
    # come out of the same builder.
    loader, shared_scene, geometry = load_shared_scene()
    shaded, emissive = loader.split_by_emission(geometry)

    def build_part(name, part, material):
        if not part["triangles"]:
            return None
        mesh = bpy.data.meshes.new(name)
        # Already in OpenGL world space, so the one conversion matrix below is
        # the only transform applied, exactly as for the boxes it replaced.
        mesh.from_pydata(part["positions"], [], part["triangles"])
        mesh.validate(verbose=False)

        # Linear albedo per vertex (or premultiplied radiance for emitters),
        # read by a ShaderNodeVertexColor. FLOAT_COLOR rather than BYTE_COLOR:
        # byte colours are sRGB-encoded on write, which would silently gamma
        # the albedo and break the comparison with the raster pass.
        colors = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
        for index, color in enumerate(part["colors"]):
            colors.data[index].color = (color[0], color[1], color[2], 1.0)

        # Custom split normals taken from the builder's own analytic normals, so
        # Cycles shades from exactly the normals basic.frag gets rather than
        # from Blender's averaging. Every polygon is marked smooth first: with
        # custom normals supplied this reproduces hard edges too, because the
        # builder already duplicates vertices per face wherever a flat face is
        # wanted.
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        if hasattr(mesh, "normals_split_custom_set_from_vertices"):
            mesh.normals_split_custom_set_from_vertices(part["normals"])
        else:
            # Older or newer Blender without the custom-normals API: fall back to
            # the per-triangle smooth flags from the scene file. Flat faces then
            # match exactly and smooth ones (cylinders, spheres) are averaged by
            # Blender instead, which differs very slightly at their seams.
            print("normals_split_custom_set_from_vertices unavailable; "
                  "using per-face smooth flags instead.", flush=True)
            for polygon, smooth in zip(mesh.polygons, part["smooth"]):
                polygon.use_smooth = bool(smooth)

        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(material)
        scene.collection.objects.link(obj)
        obj.matrix_world = conversion
        return obj

    alley = build_part("alley_surfaces", shaded, vertex_color_material("alley_diffuse", emission=False))
    lights = build_part("alley_emitters", emissive, vertex_color_material("alley_emission", emission=True))
    if alley is None:
        raise RuntimeError(f"{SCENE_FILE} produced no shaded geometry.")
    if lights is not None:
        # The raster pass draws emitters as self-lit patches of colour and they
        # illuminate nothing. Matching that here is what keeps the two images
        # comparable: otherwise Cycles would bounce light off every window and
        # bulb and come out far brighter than the OpenGL render.
        #
        # Switching the emitters off for diffuse (and glossy/transmission/volume)
        # rays is what actually does it: a diffuse ray that cannot see the
        # emitter collects no light from it, while camera rays still do, so the
        # bulbs stay visibly bright. `visible_shadow` is deliberately left on,
        # because the emitter geometry is in the OpenGL shadow pass too and does
        # block the sun there.
        # --full-gi leaves the emitters lighting the scene. That is physically
        # complete, but then a raster-versus-Cycles difference measures global
        # illumination as well as defocus, and the two are no longer separable.
        # Matched is the default precisely because the experiment is about the
        # lens.
        for attribute in ([] if args.full_gi else
                          ["visible_diffuse", "visible_glossy", "visible_transmission",
                           "visible_volume_scatter"]):
            if hasattr(lights, attribute):
                setattr(lights, attribute, False)
            else:
                # Blender 2.9x and earlier kept these under object.cycles_visibility.
                legacy = getattr(lights, "cycles_visibility", None)
                legacy_name = attribute.replace("visible_", "")
                if legacy is not None and hasattr(legacy, legacy_name):
                    setattr(legacy, legacy_name, False)
        # Some Blender versions also expose a per-object bounce limit. It is a
        # belt-and-braces extra, not the mechanism above, so it is only set when
        # the build actually has it: Blender 5.x does not, and assuming it did
        # is what used to abort this script.
        object_cycles = getattr(lights, "cycles", None)
        if object_cycles is not None and hasattr(object_cycles, "max_bounces"):
            object_cycles.max_bounces = 0
    print(f"Built {SCENE_FILE}: {len(shaded['triangles'])} shaded and "
          f"{len(emissive['triangles'])} emissive triangles", flush=True)

    data = bpy.data.cameras.new("reference_camera")
    camera = bpy.data.objects.new("reference_camera", data)
    scene.collection.objects.link(camera)
    forward, up = Vector(CAMERA_FORWARD_GL), Vector(CAMERA_UP_GL)
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()
    rotation = Matrix((right, up, -forward)).transposed().to_4x4()
    camera.matrix_world = conversion @ Matrix.Translation(Vector(CAMERA_POSITION_GL)) @ rotation
    data.type = "PERSP"
    data.lens = args.lens
    data.sensor_fit = "VERTICAL"
    data.sensor_height = args.sensor_height
    data.sensor_width = plan["effective_sensor_width_mm"]
    data.clip_start, data.clip_end = NEAR_M, FAR_M
    data.shift_x = data.shift_y = 0.0
    data.dof.focus_object = None
    data.dof.aperture_blades = 0  # Circular aperture, no artistic polygonal bokeh.
    data.dof.aperture_ratio = 1.0
    scene.camera = camera
    # Check effective framing, including aspect and sensor-fit behavior, at runtime.
    frame = data.view_frame(scene=scene)
    actual_fov = math.degrees(2 * math.atan(max(abs(v.y / v.z) for v in frame)))
    if not math.isclose(actual_fov, plan["vertical_fov_degrees"], abs_tol=0.001):
        raise RuntimeError(f"Blender vertical FOV mismatch: {actual_fov} versus {plan['vertical_fov_degrees']}")

    # World and sun both come from the scene file, so this is the same light
    # that basic.frag uses. The background colour and strength multiply to the
    # sky radiance the raster pass clears to and adds as its ambient term.
    scene.world = bpy.data.worlds.new("reference_world")
    if bpy.app.version < (5, 0, 0):
        scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_links = scene.world.node_tree.links
    world_output = world_nodes.get("World Output") or world_nodes.new("ShaderNodeOutputWorld")
    visible_sky = world_nodes.get("Background") or world_nodes.new("ShaderNodeBackground")
    visible_sky.inputs["Color"].default_value = (*shared_scene.ambient.color, 1.0)
    visible_sky.inputs["Strength"].default_value = shared_scene.ambient.strength
    if args.full_gi:
        world_links.new(visible_sky.outputs[0], world_output.inputs["Surface"])
    else:
        # The sky is what the camera sees through an opening, and it is the same
        # colour the OpenGL pass clears to. But it must not LIGHT anything,
        # because the ambient fill on every material already accounts for that;
        # letting it do both would double the ambient term in Cycles only.
        #
        # Gating on "Is Camera Ray" does exactly that and works in every Blender
        # version, unlike the per-world ray-visibility toggles which have moved
        # between releases.
        dark = world_nodes.new("ShaderNodeBackground")
        dark.inputs["Strength"].default_value = 0.0
        light_path = world_nodes.new("ShaderNodeLightPath")
        mix = world_nodes.new("ShaderNodeMixShader")
        world_links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
        world_links.new(dark.outputs[0], mix.inputs[1])
        world_links.new(visible_sky.outputs[0], mix.inputs[2])
        world_links.new(mix.outputs[0], world_output.inputs["Surface"])

    light_data = bpy.data.lights.new("directional_light", "SUN")
    # energy is irradiance in W/m^2 on a surface facing the sun, the same
    # quantity uLightEnergy carries into basic.frag.
    light_data.energy = shared_scene.sun.energy
    light_data.color = shared_scene.sun.color
    light_data.angle = math.radians(shared_scene.sun.angular_diameter_degrees)
    light = bpy.data.objects.new("directional_light", light_data)
    scene.collection.objects.link(light)
    light.rotation_euler = (-Vector(gl_to_blender(shared_scene.sun.direction))).to_track_quat("-Z", "Y").to_euler()
    return scene, data


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
        # Blender's own flags must not be interpreted as tool flags when no '--' was given.
        if "bpy" in sys.modules and "--" not in sys.argv:
            argv = []
    args = parse_args(argv)
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only: no Blender execution or renders verified.")
        return
    try:
        import bpy
    except ImportError as error:
        raise RuntimeError("Blender must be installed before the ray-traced reference can be rendered. Run this script through Blender, or use --dry-run.") from error
    if bpy.app.version < (4, 2, 0):
        raise RuntimeError("This script targets Blender 4.2 or newer with the built-in OBJ importer and Cycles.")
    scene, camera = create_scene(bpy, args, plan)
    device = select_device(bpy, scene, args.device)
    output = project_root() / plan["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    jobs = render_jobs(args)
    total_started = time.monotonic()
    generated = []
    print(f"Blender {bpy.app.version_string} | CYCLES | selected {device}", flush=True)
    for index, (filename, focus, fstop, use_dof) in enumerate(jobs, start=1):
        camera.dof.use_dof = use_dof
        camera.dof.focus_distance = focus
        camera.dof.aperture_fstop = fstop
        scene.render.filepath = str(output / filename)
        print(f"Rendering {index}/{len(jobs)}:\n  output: {filename}\n  focus: {focus} m\n"
              f"  aperture: f/{fstop:g} | DoF: {'on' if use_dof else 'off'}\n"
              f"  samples: {args.samples}\n  denoising: {'on' if args.denoise else 'OFF'}\n"
              f"  device: {device}", flush=True)
        started = time.monotonic()
        try:
            result = bpy.ops.render.render(write_still=True)
        except RuntimeError:
            if scene.cycles.device != "GPU":
                raise
            print("GPU render failed; retrying this reference on CPU.", flush=True)
            scene.cycles.device = "CPU"
            device = "CPU (fallback after GPU failure)"
            result = bpy.ops.render.render(write_still=True)
        if ("FINISHED" not in result or not (output / filename).is_file()
                or (output / filename).stat().st_size == 0):
            raise RuntimeError(f"Render did not finish: {filename}")
        render_seconds = time.monotonic() - started
        convergence = None
        if args.convergence:
            # Render the same frame again at half the samples with a different
            # seed, and report how far the two disagree. Two independent Monte
            # Carlo estimates of the same integral differ by roughly their own
            # noise, so this is a measured error bar on the reference rather
            # than an assumption that 4096 samples was "enough". By the usual
            # 1/sqrt(N) argument the full-sample image's own noise is about
            # 1/sqrt(2) of the reported difference.
            half_name = Path(filename).stem + "_half" + Path(filename).suffix
            scene.cycles.samples = max(1, args.samples // 2)
            scene.cycles.seed = args.seed + 1
            scene.render.filepath = str(output / half_name)
            print(f"  convergence pass at {scene.cycles.samples} samples", flush=True)
            half_result = bpy.ops.render.render(write_still=True)
            scene.cycles.samples = args.samples
            scene.cycles.seed = args.seed
            if "FINISHED" not in half_result:
                raise RuntimeError(f"Convergence pass did not finish: {half_name}")
            convergence = compare_passes(output / filename, output / half_name, args.samples)
            if convergence is not None:
                print(f"  convergence: mean |difference| {convergence['mean_abs_difference']:.5f}, "
                      f"max {convergence['max_abs_difference']:.5f} "
                      f"({convergence['note']})", flush=True)

        record = dict(plan, blender_version=bpy.app.version_string, render_device=device,
                      focus_distance_m=focus, f_number=fstop, use_dof=use_dof,
                      render_seconds=render_seconds,
                      convergence=convergence,
                      output=filename, render_completed=True)
        (output / Path(filename).with_suffix(".json")).write_text(json.dumps(record, indent=2) + "\n")
        generated.append(filename)
        print(f"Completed {filename} in {record['render_seconds']:.2f} s using {device}", flush=True)
    print(f"Reference rendering complete in {time.monotonic() - total_started:.2f} s.\nGenerated:", flush=True)
    for filename in generated:
        print(f"- {output / filename}", flush=True)


if __name__ == "__main__":
    main()
