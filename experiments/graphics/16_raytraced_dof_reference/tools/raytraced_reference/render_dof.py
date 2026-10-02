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

PREVIEW_SAMPLES = 32
FINAL_SAMPLES = 128
REFERENCE_WIDTH = 1200  # Framebuffer pixels, not the 600-point macOS window size.
REFERENCE_HEIGHT = 1200
FOCAL_LENGTH_MM = 50.0
SENSOR_HEIGHT_MM = 24.0
FOCUS_DISTANCES_M = (5.0,)
F_STOPS = (1.4, 2.8, 8.0)
IMPORTED_SCENE_SCALE = 0.1
IMPORTED_SCENE_POSITION_GL = (0.0, -0.75, 0.0)
IMPORTED_ROTATION_X_DEG = -90.0
IMPORTED_ROTATION_Y_DEG = 0.0
CAMERA_POSITION_GL = (0.0, 0.0, 5.0)
CAMERA_FORWARD_GL = (0.0, 0.0, -1.0)
CAMERA_UP_GL = (0.0, 1.0, 0.0)
LIGHT_DIRECTION_GL = (-0.4, 0.8, 0.6)  # Surface toward light, as in basic.frag.
NEAR_M, FAR_M = 0.1, 100.0

# One proper rotation for every world-space object: (x,y,z) -> (x,-z,y).
WORLD_CONVERSION = ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1))
# name, position, X rotation, Y rotation, scale; B/C are fallback-only and excluded.
BOXES = (
    ("foreground_A", (-0.55, -0.35, 3.0), 50.0, 70.0, (0.5, 0.5, 0.5)),
    ("background_D", (1.6, 0.4, -10.0), 35.0, -20.0, (2.0, 2.0, 2.0)),
    ("background_E", (-3.8, 0.4, -20.0), -15.0, 60.0, (2.0, 2.0, 2.0)),
    ("floor", (0.0, -0.85, -7.5), 0.0, 0.0, (12.0, 0.1, 25.0)),
    ("wall", (0.0, 1.2, -22.0), 0.0, 0.0, (14.0, 4.0, 0.1)),
)


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
    args.samples = args.samples if args.samples is not None else (PREVIEW_SAMPLES if args.preview else FINAL_SAMPLES)
    if not 1 <= args.samples <= 4096:
        parser.error("Samples must be between 1 and 4096.")
    return args


def render_jobs(args):
    jobs = [(f"rt_focus{focus:g}m_f{fstop:g}.png", focus, fstop, True)
            for focus in args.focus for fstop in args.fstops]
    if args.sharp:
        jobs.append(("rt_sharp.png", args.focus[0], args.fstops[0], False))
    if len({job[0] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate output names; choose distinct focus distances and f-stops.")
    return jobs


def make_plan(args):
    root = project_root()
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
        "supporting_boxes": BOXES,
        "outputs": [job[0] for job in render_jobs(args)],
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
    scene.cycles.seed = 16
    scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_compositing = False  # Bypass compositing; DoF comes from lens sampling.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    conversion = Matrix(WORLD_CONVERSION)

    def material(name, color):
        result = bpy.data.materials.new(name)
        result.diffuse_color = (*color, 1.0)
        if bpy.app.version < (5, 0, 0):
            result.use_nodes = True  # Blender 5+ creates the node tree automatically.
        nodes = result.node_tree.nodes
        nodes.clear()
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (*color, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        result.node_tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        return result

    # Identity importer axis conversion. Apply the same C*M transform used for all objects once.
    with geometry_only_obj(project_root() / plan["asset"]) as obj_path:
        bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="Y", up_axis="Z",
                              global_scale=1.0, clamp_size=0.0, use_split_objects=False, use_split_groups=False)
    imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError("OBJ import produced no mesh; no fallback is allowed for reference renders.")
    teapot_material = material("teapot_warm", (0.85, 0.65, 0.35))
    for obj in imported:
        obj.matrix_world = conversion @ Matrix(plan["import_matrix_gl"]) @ obj.matrix_world
        obj.data.materials.clear()
        obj.data.materials.append(teapot_material)
        for face in obj.data.polygons:
            face.material_index = 0
        obj.data.calc_loop_triangles()
        print(f"Imported {obj.name}: {len(obj.data.vertices)} positions, {len(obj.data.loop_triangles)} triangles", flush=True)

    face_colors = {(2, 1): (1, 0, 0), (2, -1): (0, 1, 0), (0, -1): (0, 0, 1),
                   (0, 1): (1, 1, 0), (1, 1): (0, 1, 1), (1, -1): (1, 0, 1)}
    colors = {key: material(f"cube_face_{key}", color) for key, color in face_colors.items()}
    for name, position, rx, ry, scale in BOXES:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.name = name
        keys = list(colors)
        for key in keys:
            obj.data.materials.append(colors[key])
        for face in obj.data.polygons:
            axis = max(range(3), key=lambda i: abs(face.normal[i]))
            face.material_index = keys.index((axis, 1 if face.normal[axis] > 0 else -1))
        obj.matrix_world = conversion @ Matrix(model_matrix(position, rx, ry, scale))

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

    scene.world = bpy.data.worlds.new("reference_world")
    if bpy.app.version < (5, 0, 0):
        scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.2, 0.3, 0.3, 1.0)
    background.inputs["Strength"].default_value = 0.4
    light_data = bpy.data.lights.new("directional_light", "SUN")
    light_data.energy = 2.0
    light_data.angle = math.radians(0.5)
    light = bpy.data.objects.new("directional_light", light_data)
    scene.collection.objects.link(light)
    light.rotation_euler = (-Vector(gl_to_blender(LIGHT_DIRECTION_GL))).to_track_quat("-Z", "Y").to_euler()
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
              f"  samples: {args.samples}\n  device: {device}", flush=True)
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
        record = dict(plan, blender_version=bpy.app.version_string, render_device=device,
                      focus_distance_m=focus, f_number=fstop, use_dof=use_dof,
                      render_seconds=time.monotonic() - started, denoising="OPENIMAGEDENOISE",
                      output=filename, render_completed=True)
        (output / Path(filename).with_suffix(".json")).write_text(json.dumps(record, indent=2) + "\n")
        generated.append(filename)
        print(f"Completed {filename} in {record['render_seconds']:.2f} s using {device}", flush=True)
    print(f"Reference rendering complete in {time.monotonic() - total_started:.2f} s.\nGenerated:", flush=True)
    for filename in generated:
        print(f"- {output / filename}", flush=True)


if __name__ == "__main__":
    main()
