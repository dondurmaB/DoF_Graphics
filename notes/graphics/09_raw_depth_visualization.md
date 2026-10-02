# Experiment 09 — Raw Depth Visualization

## Goal

Visualize the raw perspective depth values produced for visible fragments. This observes window-space depth directly before any linearization, depth texture, framebuffer, or post-processing work.

## Pipeline

Local/Object Space
→ Model Matrix
→ World Space
→ View Matrix
→ Camera/View Space
→ Projection Matrix
→ Clip Space
→ Perspective Divide
→ Window-Space Depth
→ Depth Test
→ Fragment Shader
→ Framebuffer

## What Changed

- Added `visualizeRawDepth` as the Experiment 09 control.
- Added `uVisualizeRawDepth` to the fragment shader.
- Preserved the normal vertex-color path when raw-depth visualization is disabled.
- Used `gl_FragCoord.z` directly as grayscale when raw-depth visualization is enabled.
- Kept depth testing enabled so only visible surfaces contribute to the visualization.
- Repositioned the active scene cubes across a wider range of world-space depths.

## Key Concepts

### `gl_FragCoord.z`

The fragment's window-space depth after projection, clipping, perspective divide, and viewport depth mapping. With the default OpenGL depth range, it is approximately in `[0, 1]`.

### Window-Space Depth

The value compared against the depth buffer for a fragment. This is not raw world-space `z`, camera-space distance, or Euclidean distance from the camera.

### Normalized Depth Range

OpenGL's default depth range maps near-plane depth toward `0` and far-plane depth toward `1`.

### Visible Fragment Depth

With `GL_DEPTH_TEST` and `GL_LESS`, overlapping fragments compete per pixel. The grayscale output shows the raw depth of the nearest visible surface, not every generated fragment.

### Nonlinear Perspective Depth

Perspective projection distributes depth nonlinearly. Much of the precision is concentrated near the near plane, so distant objects may look very similar in raw grayscale.

### Near and Far Planes

`nearPlane` and `farPlane` affect how camera/view-space depth maps into window-space depth. Moving the near plane has a large effect on the distribution.

## Important API / Shader Values

| Value / Function | Purpose |
|---|---|
| `gl_FragCoord.z` | Reads the current fragment's raw window-space depth. |
| `uVisualizeRawDepth` | Float-backed switch between normal color and raw-depth grayscale. |
| `glUniform1f()` | Uploads the visualization toggle through the project's existing GLAD surface. |
| `glGetUniformLocation()` | Finds the raw-depth uniform once after shader linking. |
| `glEnable(GL_DEPTH_TEST)` | Keeps nearest-surface visibility behavior active. |
| `glDepthFunc(GL_LESS)` | Accepts a fragment when it is closer than the stored depth. |
| `glClear(GL_COLOR_BUFFER_BIT \| GL_DEPTH_BUFFER_BIT)` | Resets color and depth each frame. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `visualizeRawDepth` | Shows raw window-space depth as grayscale | `false`, `true` |
| `fieldOfViewDegrees` | Changes the perspective projection | `30.0f`, `45.0f`, `75.0f` |
| `nearPlane` | Changes the start of the projected depth range | `0.1f`, `0.5f`, `1.0f` |
| `farPlane` | Changes the end of the projected depth range | `20.0f`, `50.0f`, `100.0f` |
| `cubeAPosition` ... `cubeEPosition` | Places cubes at different world-space depths | Move the `z` values |

## Observations

- Near objects appear darker because their window-space depth is closer to `0`.
- Far objects quickly approach white because raw perspective depth moves toward `1`.
- Large world-space depth changes can produce small grayscale changes far from the camera.
- Raw depth is useful for observing the depth buffer input, but it is not camera distance.
- Depth testing means hidden surfaces do not appear in the visualization.

## Remember

- This experiment intentionally does not linearize depth.
- This experiment intentionally does not use an FBO or depth texture.
- `gl_FragCoord.z` is post-projection/window-space depth.
- Raw perspective depth is nonlinear.
- Near/far plane choices strongly affect the depth distribution.
