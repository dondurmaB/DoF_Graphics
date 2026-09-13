# Experiment 06 — Depth Testing and the Depth Buffer

## Goal

Introduce the depth buffer so overlapping 3D geometry is resolved by per-fragment depth instead of draw order. This prepares for later Depth-of-Field work where depth values become an important image-space signal.

## Pipeline

Triangle → Rasterization → Fragment → Depth Test → Fragment Shader Output → Color Buffer

## What Changed

- Enabled configurable depth testing with `enableDepthTest`.
- Added `depthFunction` so comparison modes can be changed manually.
- Cleared both the color buffer and depth buffer every frame.
- Rendered the same cube mesh twice with different model transforms.
- Added draw-order reversal to compare depth-tested and non-depth-tested visibility.

## Key Concepts

### Depth Buffer

Stores one depth value per framebuffer location/sample. It tracks which fragment is currently closest at each screen location.

### Fragment Depth

Each rasterized fragment has a depth value after vertex transformation and clipping. Smaller depth values are closer when using the default `GL_LESS` comparison.

### Depth Test

The depth test compares an incoming fragment against the stored depth value. Passing fragments can update both the color buffer and the depth buffer.

### GL_DEPTH_TEST

`GL_DEPTH_TEST` enables or disables depth comparison. When disabled, later draw calls can overwrite earlier fragments regardless of geometric depth.

### GL_LESS

`GL_LESS` accepts a fragment if its depth is smaller than the currently stored value. This is the normal default for drawing nearer geometry in front.

### Clearing the Depth Buffer

The depth buffer must be cleared every frame. Clearing only the color buffer leaves old depth values behind.

### Draw Order vs Geometric Visibility

With depth testing disabled, submission order controls visibility. With depth testing enabled, visibility mostly follows geometry depth instead.

### Multiple Draw Calls

The same VAO/VBO/EBO mesh can be drawn multiple times by uploading a different `uTransform` before each `glDrawElements()` call.

## Important API Calls

| Function | Purpose |
|---|---|
| `glEnable(GL_DEPTH_TEST)` | Enable per-fragment depth testing. |
| `glDisable(GL_DEPTH_TEST)` | Disable depth testing so draw order controls overlap. |
| `glDepthFunc()` | Choose the comparison used by the depth test. |
| `glClear()` | Clear color and depth buffers at the start of each frame. |
| `glDrawElements()` | Draw each cube instance using the shared indexed mesh. |
| `glUniformMatrix4fv()` | Upload a different model transform before each cube draw. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `enableDepthTest` | Enables or disables depth testing | `true`, `false` |
| `depthFunction` | Changes the depth comparison rule | `GL_LESS`, `GL_LEQUAL`, `GL_ALWAYS`, `GL_NEVER`, `GL_GREATER` |
| `drawSecondCube` | Draws one cube or two overlapping cubes | `true`, `false` |
| `reverseCubeDrawOrder` | Draws cube B before cube A | `false`, `true` |
| `cubeAOffsetX` / `cubeBOffsetX` | Changes horizontal overlap | `-0.25f`, `0.0f`, `0.25f` |
| `cubeAOffsetY` / `cubeBOffsetY` | Moves cubes vertically | `-0.2f`, `0.0f`, `0.2f` |
| `cubeAOffsetZ` / `cubeBOffsetZ` | Changes which cube is nearer | `-0.3f`, `0.0f`, `0.3f` |
| `cubeARotationXDegrees` / `cubeBRotationXDegrees` | Rotates cubes around X | `-30.0f`, `0.0f`, `45.0f` |
| `cubeARotationYDegrees` / `cubeBRotationYDegrees` | Rotates cubes around Y | `-45.0f`, `0.0f`, `45.0f` |
| `cubeAUniformScale` / `cubeBUniformScale` | Changes cube size | `0.35f`, `0.55f`, `0.8f` |
| `wireframeMode` | Shows filled faces or mesh edges | `false`, `true` |

## Observations

- With `enableDepthTest = false`, reversing draw order changes which cube appears on top.
- With `enableDepthTest = true` and `GL_LESS`, reversing draw order does not materially change the correct overlap.
- `GL_ALWAYS` behaves like there is no useful depth rejection.
- `GL_NEVER` rejects cube fragments, leaving only the clear color.
- The color buffer stores visible color; the depth buffer stores visibility information.
- Depth values are useful for future DoF, but raw depth buffer values are not necessarily linear camera distance.

## Remember

- Depth testing happens per fragment.
- The depth buffer stores depth per framebuffer location/sample.
- `GL_LESS` means smaller depth wins.
- Clear the depth buffer every frame.
- Color and depth buffers store different information.
- Depth testing separates visibility from draw order.
- Raw depth values are not the same as linear world distance.
