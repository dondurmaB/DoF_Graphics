"""Rasterize a scene file to a PNG on the CPU, without OpenGL or Blender.

    python tools/scene/preview_scene.py --output output/scene_preview.png

This exists to check a scene edit in seconds without a rebuild or a Cycles
render. It uses the same loader, the same camera convention and the same
lighting equation as `shaders/basic.frag`:

    Lo = albedo * (sky + sun_colour * sun_energy * max(N.L, 0) / pi) + albedo * emit

with a linear-to-sRGB encode at the end. It deliberately omits shadows,
occlusion and depth of field, so it is a framing and albedo check, not a
reference image. Back faces are culled, which also catches a primitive whose
triangle winding disagrees with its normals.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_loader  # noqa: E402


def look_at(position, yaw_degrees, pitch_degrees):
    yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
    forward = np.array([math.cos(yaw) * math.cos(pitch), math.sin(pitch),
                        math.sin(yaw) * math.cos(pitch)])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[:3, 3] = -view[:3, :3] @ np.array(position)
    return view


def perspective(fov_y_radians, aspect, near, far):
    f = 1.0 / math.tan(fov_y_radians * 0.5)
    projection = np.zeros((4, 4))
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2.0 * far * near) / (near - far)
    projection[3, 2] = -1.0
    return projection


def encode_srgb(linear):
    clamped = np.clip(linear, 0.0, 1.0)
    low = clamped * 12.92
    high = 1.055 * np.power(np.maximum(clamped, 1e-8), 1.0 / 2.4) - 0.055
    return np.where(clamped <= 0.0031308, low, high)


def render(scene, geometry, width, height, focus_override=None):
    camera = scene.camera
    fov_y = 2.0 * math.atan(camera.sensor_height_mm / (2.0 * camera.focal_length_mm))
    view = look_at(camera.position, camera.yaw_degrees, camera.pitch_degrees)
    projection = perspective(fov_y, width / height, 0.1, 200.0)
    view_projection = projection @ view

    positions = np.array(geometry.positions, dtype=np.float64)
    normals = np.array(geometry.normals, dtype=np.float64)
    colors = np.array(geometry.colors, dtype=np.float64)
    emissions = np.array(geometry.emissions, dtype=np.float64)
    triangles = np.array(geometry.triangles, dtype=np.int64)

    homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    clip = homogeneous @ view_projection.T
    w = clip[:, 3]
    safe_w = np.where(np.abs(w) < 1e-6, 1e-6, w)
    ndc = clip[:, :3] / safe_w[:, None]
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * width
    screen_y = (0.5 - ndc[:, 1] * 0.5) * height

    sun_direction = np.array(scene.sun.direction, dtype=np.float64)
    sun_direction /= np.linalg.norm(sun_direction)
    sun_radiance = np.array(scene.sun.color) * scene.sun.energy / math.pi
    sky = np.array(scene.sky_radiance)

    color_buffer = np.tile(sky, (height, width, 1))
    depth_buffer = np.full((height, width), np.inf)

    near_epsilon = 1e-3

    def clip_near(corners):
        """Clip a clip-space triangle against w > 0 and return screen triangles.

        Without this, any triangle with a vertex behind the camera has to be
        dropped whole, which silently removes things like the ground slab that
        starts behind the camera. OpenGL and Cycles both clip properly, so the
        preview has to as well or it lies about the framing.
        """
        polygon = []
        for index in range(len(corners)):
            current, following = corners[index], corners[(index + 1) % len(corners)]
            current_in, following_in = current[3] > near_epsilon, following[3] > near_epsilon
            if current_in:
                polygon.append(current)
            if current_in != following_in:
                t = (near_epsilon - current[3]) / (following[3] - current[3])
                polygon.append(current + (following - current) * t)
        return [(polygon[0], polygon[index], polygon[index + 1])
                for index in range(1, len(polygon) - 1)] if len(polygon) >= 3 else []

    # Flat shading per triangle: enough for a framing check and keeps this loop small.
    tri_normals = normals[triangles].mean(axis=1)
    lengths = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    tri_normals = tri_normals / np.maximum(lengths, 1e-9)
    tri_colors = colors[triangles].mean(axis=1)
    tri_emissions = emissions[triangles].mean(axis=1)
    n_dot_l = np.maximum(tri_normals @ sun_direction, 0.0)
    shaded = tri_colors * (sky + sun_radiance * n_dot_l[:, None])
    emitted = tri_colors * tri_emissions[:, None]
    tri_radiance = np.where(tri_emissions[:, None] > 0.0, emitted, shaded)

    all_behind = (w <= near_epsilon)[triangles].all(axis=1)
    needs_clip = (w <= near_epsilon)[triangles].any(axis=1)
    drawn = 0
    for index in range(len(triangles)):
        if all_behind[index]:
            continue
        i0, i1, i2 = triangles[index]
        if needs_clip[index]:
            pieces = clip_near((clip[i0], clip[i1], clip[i2]))
        else:
            pieces = [None]  # Fast path: use the already projected screen positions.
        for piece in pieces:
            if piece is None:
                x0, x1, x2 = screen_x[i0], screen_x[i1], screen_x[i2]
                y0, y1, y2 = screen_y[i0], screen_y[i1], screen_y[i2]
                inverse = (1.0 / safe_w[i0], 1.0 / safe_w[i1], 1.0 / safe_w[i2])
            else:
                projected = []
                for corner in piece:
                    corner_w = corner[3] if abs(corner[3]) > 1e-9 else 1e-9
                    projected.append(((corner[0] / corner_w * 0.5 + 0.5) * width,
                                      (0.5 - corner[1] / corner_w * 0.5) * height,
                                      1.0 / corner_w))
                (x0, y0, iw0), (x1, y1, iw1), (x2, y2, iw2) = projected
                inverse = (iw0, iw1, iw2)
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if area >= 0.0:  # Screen Y is flipped, so front faces have negative area here.
                continue
            min_x = max(int(math.floor(min(x0, x1, x2))), 0)
            max_x = min(int(math.ceil(max(x0, x1, x2))), width - 1)
            min_y = max(int(math.floor(min(y0, y1, y2))), 0)
            max_y = min(int(math.ceil(max(y0, y1, y2))), height - 1)
            if min_x > max_x or min_y > max_y:
                continue
            xs = np.arange(min_x, max_x + 1) + 0.5
            ys = np.arange(min_y, max_y + 1) + 0.5
            grid_x, grid_y = np.meshgrid(xs, ys)
            w0 = ((x1 - x0) * (grid_y - y0) - (grid_x - x0) * (y1 - y0)) / area
            w1 = ((x2 - x1) * (grid_y - y1) - (grid_x - x1) * (y2 - y1)) / area
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            if not inside.any():
                continue
            # 1/w interpolates linearly in screen space; depth does not.
            inverse_w = w1 * inverse[0] + w2 * inverse[1] + w0 * inverse[2]
            depth = 1.0 / np.where(np.abs(inverse_w) < 1e-12, 1e-12, inverse_w)
            window = depth_buffer[min_y:max_y + 1, min_x:max_x + 1]
            visible = inside & (depth > 0.0) & (depth < window)
            if not visible.any():
                continue
            window[visible] = depth[visible]
            color_window = color_buffer[min_y:max_y + 1, min_x:max_x + 1]
            color_window[visible] = tri_radiance[index]
            drawn += 1
    return color_buffer, depth_buffer, drawn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene/alley.scene")
    parser.add_argument("--output", default="output/scene_preview.png")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--depth-output", help="Also write a linear-depth preview PNG")
    parser.add_argument("--depth-max", type=float, default=40.0)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    scene_path = Path(args.scene) if Path(args.scene).is_absolute() else root / args.scene
    scene = scene_loader.load_scene(scene_path)
    geometry = scene_loader.build_geometry(scene)
    color, depth, drawn = render(scene, geometry, args.width, args.height)

    from PIL import Image
    output = Path(args.output) if Path(args.output).is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((encode_srgb(color) * 255.0 + 0.5).astype(np.uint8)).save(output)
    print(f"{drawn} triangles rasterized -> {output}")
    if args.depth_output:
        depth_output = Path(args.depth_output) if Path(args.depth_output).is_absolute() else root / args.depth_output
        normalized = np.clip(np.where(np.isinf(depth), args.depth_max, depth) / args.depth_max, 0.0, 1.0)
        Image.fromarray((normalized * 255.0 + 0.5).astype(np.uint8)).save(depth_output)
        print(f"depth -> {depth_output}")


if __name__ == "__main__":
    main()
