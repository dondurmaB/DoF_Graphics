# Experiment 10 — Linear Depth Reconstruction

## Goal

Convert nonlinear raw perspective depth from `gl_FragCoord.z` into a linear positive view-depth quantity in the fragment shader.

## Pipeline

View-Space Geometry
→ Perspective Projection
→ Clip-Space Z/W
→ Perspective Divide
→ NDC Depth
→ Window Depth (`gl_FragCoord.z`)
→ Reverse Depth Mapping
→ Linear View Depth
→ Display Normalization
→ Grayscale

## What Changed

- Replaced the raw-depth boolean with a source-code `VisualizationMode`.
- Preserved normal color rendering and raw depth visualization.
- Added linear depth reconstruction directly in the existing fragment shader.
- Passed `nearPlane`, `farPlane`, and `depthVisualizationMax` as uniforms.
- Used a display-only normalization range for linear-depth grayscale output.
- Kept the existing camera, MVP pipeline, cube buffers, depth testing, and screenshot workflow.

## Key Concepts

### Raw Depth

`gl_FragCoord.z` is window-space depth, normally approximately in `[0, 1]`. Under perspective projection, it is nonlinear.

### NDC Depth

OpenGL NDC depth is normally in `[-1, 1]`. Convert raw/window depth back to NDC with:

```text
z_ndc = 2 * z_raw - 1
```

### Linear View Depth

The reconstructed positive depth along the camera/view depth axis. It is useful for depth comparisons and later Depth of Field calculations.

### Depth Reconstruction

The shader reverses the conventional OpenGL perspective depth mapping using the same near and far planes used by `glm::perspective()`.

### Near Plane

`nearPlane` defines the near clipping distance and strongly affects depth precision and reconstruction. The shader's `uNearPlane` must match the projection's near plane.

### Far Plane

`farPlane` defines the far clipping distance. The shader's `uFarPlane` must match the projection's far plane.

### Visualization Normalization

Linear depth can be values like `5`, `10`, or `25`, which would clamp if written directly to RGB. `uDepthVisualizationMax` scales only the grayscale display.

It does not modify camera projection, near plane, far plane, depth testing, or the reconstructed linear depth.

### Linear Depth vs Euclidean Distance

Linear view depth is not the full 3D distance from the camera to the fragment. For off-axis points:

```text
view-space depth != length(cameraPosition - worldPosition)
```

## Important Formula

```text
z_ndc = 2 * z_raw - 1

z_linear = (2*n*f) / (f+n - z_ndc*(f-n))
```

Variables:

- `z_raw`: raw window-space depth from `gl_FragCoord.z`.
- `z_ndc`: OpenGL normalized device coordinate depth.
- `n`: near plane used by the projection matrix.
- `f`: far plane used by the projection matrix.
- `z_linear`: reconstructed positive view-depth.

## Visualization Modes

| Mode | Value | Output |
|---|---:|---|
| `VisualizationMode::Color` | `0` | Interpolated vertex color |
| `VisualizationMode::RawDepth` | `1` | `gl_FragCoord.z` as grayscale |
| `VisualizationMode::LinearDepth` | `2` | Reconstructed linear depth divided by `depthVisualizationMax` |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `visualizationMode` | Chooses color, raw depth, or linear depth | `Color`, `RawDepth`, `LinearDepth` |
| `nearPlane` | Projection near plane and reconstruction input | `0.1f`, `0.5f`, `1.0f` |
| `farPlane` | Projection far plane and reconstruction input | `30.0f`, `50.0f`, `100.0f` |
| `depthVisualizationMax` | Display-only maximum for linear-depth grayscale | `10.0f`, `25.0f`, `50.0f` |
| `fieldOfViewDegrees` | Camera projection width | `30.0f`, `45.0f`, `75.0f` |

## Expected Known-Distance Relationship

With the default camera at `z = 5` looking down negative `z`, an object center at world `z = 1` is roughly `4` units of view depth, while an object center at world `z = -10` is roughly `15` units. Cube faces differ slightly because their vertices occupy space around each center.

## Observations

- Raw depth stays close to white across much of the scene when the far plane is large.
- Linear depth changes more evenly as objects move away from the camera.
- Reducing `depthVisualizationMax` makes closer depth differences brighter sooner.
- Changing near/far in C++ changes both the projection and the reconstruction uniforms.

## Remember

- Raw/window depth is `gl_FragCoord.z`.
- NDC depth is recovered with `2 * rawDepth - 1`.
- Linear view depth is not Euclidean camera-to-fragment distance.
- Reconstruction is wrong if shader near/far values differ from the projection near/far values.
- This experiment still does not use an FBO, depth texture, readback, post-processing, or blur.
