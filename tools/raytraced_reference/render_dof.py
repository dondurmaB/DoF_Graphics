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
DEFAULT_COMPARISON_LENS_MM = 85.0
DEFAULT_COMPARISON_F_STOPS = (1.2, 22.0)
F_STOPS = DEFAULT_COMPARISON_F_STOPS
NEAR_M, FAR_M = 0.1, 100.0

# One proper rotation for every world-space object: (x,y,z) -> (x,-z,y).
WORLD_CONVERSION = ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1))

# Scene data, including capture size and import transform, is read on each run.
SCENE_FILE = "scene/cafe.scene"


def load_shared_scene(path=SCENE_FILE):
    root = project_root()
    loader_directory = str(root / "tools/scene")
    if loader_directory not in sys.path:
        sys.path.insert(0, loader_directory)
    import scene_loader
    scene = scene_loader.load_scene(root / path)
    return scene_loader, scene, scene_loader.build_geometry(scene)


def camera_forward_gl(yaw_degrees, pitch_degrees):
    """Same yaw/pitch convention as updateCameraFront() in src/main.cpp."""
    yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
    return (math.cos(yaw) * math.cos(pitch), math.sin(pitch), math.sin(yaw) * math.cos(pitch))


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
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--lens", type=float, default=None)
    parser.add_argument("--sensor-height", type=float, default=None)
    parser.add_argument("--focus", type=float, nargs="+", default=None)
    parser.add_argument("--fstops", type=float, nargs="+", default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--samples", type=int, help="Override preview/final sample count")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument(
        "--sharp",
        action="store_true",
        help="Also render a settings-tagged sharp control with lens DoF disabled",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print validated settings; no Blender import or output files")
    parser.add_argument("--scene", default=SCENE_FILE)
    parser.add_argument("--position", type=float, nargs=3)
    parser.add_argument("--yaw", type=float)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--import-mesh", action="store_true")
    parser.add_argument(
        "--full-gi",
        action="store_true",
        help="Physical indirect illumination; NOT comparable to OpenGL",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/raytraced_dof"))
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Inspection renders; does not certify ground truth",
    )
    args = parser.parse_args(argv)
    _, shared, _ = load_shared_scene(args.scene)
    camera = shared.camera
    # The shared scene supplies framing, capture dimensions and the focus plane.
    # The default comparison lens is deliberately longer than the scene-camera
    # lens: 50 mm at f/1.2 only gives ~18 px on the far wall at 1200px, while
    # 85 mm focused on the same hero cup gives ~52 px there. f/22 stays in the
    # list as the near-pinhole diagnostic control.
    default_lens = DEFAULT_COMPARISON_LENS_MM if args.scene == SCENE_FILE else camera.focal_length_mm
    for key, value in (
        ("width", shared.capture_width),
        ("height", shared.capture_height),
        ("lens", default_lens),
        ("sensor_height", camera.sensor_height_mm),
        ("focus", [camera.focus_distance_m]),
        ("fstops", list(DEFAULT_COMPARISON_F_STOPS)),
        ("position", camera.position),
        ("yaw", camera.yaw_degrees),
        ("pitch", camera.pitch_degrees),
    ):
        if getattr(args, key) is None:
            setattr(args, key, value)
    if abs(args.pitch) >= 89.9:
        parser.error("Pitch must be within (-89.9,89.9)")
    if not (16 <= args.width <= 8192 and 16 <= args.height <= 8192):
        parser.error("Width and height must be between 16 and 8192 framebuffer pixels.")
    if not all(
        math.isfinite(v)
        for v in (
            args.lens,
            args.sensor_height,
            *args.focus,
            *args.fstops,
            *args.position,
            args.yaw,
            args.pitch,
        )
    ):
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


def capture_tag(focus, fstop, lens, sharp=False):
    tag = f"focus{focus:g}m_f{fstop:g}"
    if abs(lens - 50) > 0.01:
        tag += f"_{lens:g}mm"
    if sharp:
        tag += "_sharp"
    return tag


def render_jobs(args):
    jobs = [
        (f"rt_{capture_tag(focus,fstop,args.lens)}.png", focus, fstop, True)
        for focus in args.focus
        for fstop in args.fstops
    ]
    if args.sharp:
        jobs.append(
            (
                f"rt_{capture_tag(args.focus[0],args.fstops[0],args.lens,True)}.png",
                args.focus[0],
                args.fstops[0],
                False,
            )
        )
    if len({job[0] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate output names; choose distinct focus distances and f-stops.")
    return jobs


def make_plan(args):
    root = project_root()
    loader, scene, geometry = load_shared_scene(args.scene)
    forward = camera_forward_gl(args.yaw, args.pitch)
    yaw, pitch = math.radians(args.yaw), math.radians(args.pitch)
    up = (-math.cos(yaw) * math.sin(pitch), math.cos(pitch), -math.sin(yaw) * math.sin(pitch))
    rotation = loader.rotation_matrix(*scene.import_rotation)
    import_matrix = tuple(
        tuple(rotation[i][j] * scene.import_scale[j] for j in range(3))
        + (scene.import_position[i],)
        for i in range(3)
    ) + (
        (0, 0, 0, 1),
    )
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
    return {
        "asset": "assets/models/scene.obj" if args.import_mesh else None,
        "asset_sha256": (
            hashlib.sha256((root / "assets/models/scene.obj").read_bytes()).hexdigest()
            if args.import_mesh
            else None
        ),
        "output_directory": str(args.output),
        "lighting_mode": "full_gi_not_comparable" if args.full_gi else "matched_fill_direct",
        "ground_truth": False,
        "fill_visibility": "world_transport" if args.full_gi else "primary_camera_only",
        "max_bounces": 12,
        "diffuse_bounces": 4 if args.full_gi else 0,
        "adaptive_sampling": False,
        "clamp_direct": 0,
        "clamp_indirect": 0,
        "seed": 16,
        "linear_output": "32-bit RGB EXR",
        "resolution_pixels": [args.width, args.height],
        "samples": args.samples,
        "requested_device": args.device,
        "engine": "CYCLES",
        "focal_length_mm": args.lens,
        "sensor_height_mm": args.sensor_height,
        "sensor_fit": "VERTICAL",
        "effective_sensor_width_mm": args.sensor_height * args.width / args.height,
        "vertical_fov_degrees": vertical_fov_degrees(args.lens, args.sensor_height),
        "camera_position_gl": args.position,
        "camera_forward_gl": forward,
        "camera_up_gl": up,
        "camera_position_blender": gl_to_blender(args.position),
        "camera_forward_blender": gl_to_blender(forward),
        "camera_up_blender": gl_to_blender(up),
        "world_conversion": WORLD_CONVERSION,
        "import_matrix_gl": import_matrix,
        "import_albedo": scene.import_albedo,
        "outputs": [job[0] for job in render_jobs(args)],
        # The scene digest goes in every sidecar JSON. Without it a reference PNG
        # cannot be tied to the geometry it was rendered from, which is the one
        # thing that would quietly invalidate a comparison months later.
        "scene_file": args.scene,
        "scene_sha256": hashlib.sha256((root / args.scene).read_bytes()).hexdigest(),
        "scene_summary": scene_summary,
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
    scene.cycles.use_denoising = not args.no_denoise
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
    # Matched mode integrates ONLY primary emission and direct sunlight. Zero
    # diffuse continuation is intentional; fill must never bounce or add twice.
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4 if args.full_gi else 0
    scene.cycles.glossy_bounces = 4 if args.full_gi else 0
    scene.cycles.transmission_bounces = 12 if args.full_gi else 0
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

    def new_material(name):
        result = bpy.data.materials.new(name)
        if bpy.app.version < (5, 0, 0):
            result.use_nodes = True  # Blender 5+ creates the node tree automatically.
        result.node_tree.nodes.clear()
        return result

    _, shared_scene, _ = load_shared_scene(args.scene)

    def matched_surface(result, color_output=None, color=None):
        nodes, links = result.node_tree.nodes, result.node_tree.links
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Roughness"].default_value = 0
        if color_output is None:
            diffuse.inputs["Color"].default_value = (*color, 1)
        else:
            links.new(color_output, diffuse.inputs["Color"])
        output = nodes.new("ShaderNodeOutputMaterial")
        if args.full_gi:
            links.new(diffuse.outputs[0], output.inputs["Surface"])
            return
        # Per-channel multiplication with VectorMath survives Blender 4/5's
        # MixRGB API changes. The unoccluded term is albedo * sky radiance.
        multiply_node = nodes.new("ShaderNodeVectorMath")
        multiply_node.operation = "MULTIPLY"
        multiply_node.inputs[1].default_value = shared_scene.sky_radiance
        if color_output is None:
            multiply_node.inputs[0].default_value = color
        else:
            links.new(color_output, multiply_node.inputs[0])
        fill = nodes.new("ShaderNodeEmission")
        primary = nodes.new("ShaderNodeLightPath")
        links.new(primary.outputs["Is Camera Ray"], fill.inputs["Strength"])
        links.new(multiply_node.outputs["Vector"], fill.inputs["Color"])
        add = nodes.new("ShaderNodeAddShader")
        links.new(diffuse.outputs[0], add.inputs[0])
        links.new(fill.outputs[0], add.inputs[1])
        links.new(add.outputs[0], output.inputs["Surface"])
        # This fill must not be sampled as a mesh light. It is a shading term,
        # not an emitter capable of illuminating another surface.
        if hasattr(result, "cycles") and hasattr(result.cycles, "emission_sampling"):
            result.cycles.emission_sampling = "NONE"
        else:
            raise RuntimeError("Matched fill requires the material emission_sampling=NONE control")

    def flat_material(name, color):
        """One albedo for the whole object; used for the imported mesh."""
        result = new_material(name)
        result.diffuse_color = (*color, 1.0)
        matched_surface(result, color=color)
        return result

    def vertex_color_material(name, emission):
        """Reads the per-vertex colour the scene file supplies.

        A Diffuse BSDF for surfaces, matching basic.frag's albedo/pi Lambert
        term; an Emission shader at strength 1.0 for emitters, whose colour the
        builder has already premultiplied by `emit` so both renderers emit the
        same radiance.
        """
        result = new_material(name)
        nodes = result.node_tree.nodes
        attribute = nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = "Col"
        if emission:
            shader = nodes.new("ShaderNodeEmission")
            shader.inputs["Strength"].default_value = 1.0
            output = nodes.new("ShaderNodeOutputMaterial")
            result.node_tree.links.new(attribute.outputs["Color"], shader.inputs["Color"])
            result.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
            if not args.full_gi and hasattr(result.cycles, "emission_sampling"):
                result.cycles.emission_sampling = "NONE"
        else:
            matched_surface(result, color_output=attribute.outputs["Color"])
        return result

    if args.import_mesh:
        # Identity importer axis conversion. Apply the same C*M transform used for all objects once.
        with geometry_only_obj(project_root() / plan["asset"]) as obj_path:
            bpy.ops.wm.obj_import(
                filepath=str(obj_path),
                forward_axis="Y",
                up_axis="Z",
                global_scale=1.0,
                clamp_size=0.0,
                use_split_objects=False,
                use_split_groups=False,
            )
        imported = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        if not imported:
            raise RuntimeError(
                "OBJ import produced no mesh; no fallback is allowed for reference renders."
            )
        teapot_material = flat_material("teapot_warm", plan["import_albedo"])
        for obj in imported:
            obj.matrix_world = conversion @ Matrix(plan["import_matrix_gl"]) @ obj.matrix_world
            obj.data.materials.clear()
            obj.data.materials.append(teapot_material)
            for face in obj.data.polygons:
                face.material_index = 0
            obj.data.calc_loop_triangles()
            print(
                f"Imported {obj.name}: {len(obj.data.vertices)} positions, {len(obj.data.loop_triangles)} triangles",
                flush=True,
            )

    # ---- The alley, from the shared scene file ----
    # Two objects, because one carries a Diffuse BSDF and the other an Emission
    # shader. The OpenGL pass keeps a single mesh and branches on a per-vertex
    # emission attribute instead; the geometry itself is identical because both
    # come out of the same builder.
    loader, shared_scene, geometry = load_shared_scene(args.scene)
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
    if lights is not None and not args.full_gi:
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
        for attribute in ("visible_diffuse", "visible_glossy", "visible_transmission",
                          "visible_volume_scatter"):
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
    forward, up = Vector(plan["camera_forward_gl"]), Vector(plan["camera_up_gl"])
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()
    rotation = Matrix((right, up, -forward)).transposed().to_4x4()
    camera.matrix_world = conversion @ Matrix.Translation(Vector(args.position)) @ rotation
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
    background = scene.world.node_tree.nodes.get("Background")
    if background is None:
        # Not every Blender version seeds a new world with a Background node.
        nodes = scene.world.node_tree.nodes
        background = nodes.new("ShaderNodeBackground")
        world_output = (nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld"))
        scene.world.node_tree.links.new(background.outputs[0], world_output.inputs["Surface"])
    background.inputs["Color"].default_value = (*shared_scene.ambient.color, 1.0)
    background.inputs["Strength"].default_value = shared_scene.ambient.strength
    if not args.full_gi:
        # Visible camera background, zero world illumination. The explicit
        # material fill above supplies ambient exactly once even inside rooms.
        nodes, links = scene.world.node_tree.nodes, scene.world.node_tree.links
        rays = nodes.new("ShaderNodeLightPath")
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = shared_scene.ambient.strength
        links.new(rays.outputs["Is Camera Ray"], scale.inputs[0])
        links.new(scale.outputs[0], background.inputs["Strength"])
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
        # Inspection EXR: lossless linear radiance, explicitly NOT yet a
        # convergence-certified reference. Keep PNG only as a display preview.
        linear_path = output / Path(filename).with_suffix(".exr")
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"
        bpy.data.images["Render Result"].save_render(str(linear_path), scene=scene)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        # PFM companion makes the exact same floats inspectable without an
        # external EXR package. Blender loads EXR scene-linear, without a view.
        import numpy as np

        linear = bpy.data.images.load(str(linear_path), check_existing=False)
        pixels = np.empty(args.width * args.height * 4, dtype=np.float32)
        linear.pixels.foreach_get(pixels)
        rgb = pixels.reshape(args.height, args.width, 4)[:, :, :3].copy()
        with linear_path.with_suffix(".pfm").open("wb") as stream:
            stream.write(f"PF\n{args.width} {args.height}\n-1.0\n".encode())
            stream.write(rgb.astype("<f4").tobytes())
        bpy.data.images.remove(linear)
        record = dict(
            plan,
            blender_version=bpy.app.version_string,
            render_device=device,
            focus_distance_m=focus,
            f_number=fstop,
            use_dof=use_dof,
            render_seconds=time.monotonic() - started,
            denoising="OFF" if args.no_denoise else "OPENIMAGEDENOISE",
            output=filename,
            render_completed=True,
        )
        (output / Path(filename).with_suffix(".json")).write_text(json.dumps(record, indent=2) + "\n")
        generated.append(filename)
        print(f"Completed {filename} in {record['render_seconds']:.2f} s using {device}", flush=True)
    print(f"Reference rendering complete in {time.monotonic() - total_started:.2f} s.\nGenerated:", flush=True)
    for filename in generated:
        print(f"- {output / filename}", flush=True)


if __name__ == "__main__":
    main()
