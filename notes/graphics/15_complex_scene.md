# Experiment 15 — Complex Scene Loading

## Goal

Load an external OBJ into the existing raster renderer and use its geometry for DoF experiments. The supplied Brown Betty teapot at `assets/models/scene.obj` loads successfully. Build, CPU loader tests and final manual runtime verification passed.

## Pipeline

```text
OBJ / External Mesh → tinyobjloader → CPU Vertex/Index Data
→ VAO/VBO/EBO → Existing Scene Pass → Color + Depth
→ Linear Depth → Physical CoC → Existing BasicDoF
```

## What Changed

- Added tinyobjloader through CMake FetchContent, pinned to `v2.0.0rc13`'s commit.
- Added `Mesh.h` / `Mesh.cpp` for loading, indexed upload, drawing and cleanup.
- Added explicit asset path, scale, position and Y-rotation controls.
- Added a simple lit mesh color, a floor and a back wall. Existing cubes provide depth references and the missing-asset fallback.
- Added automated CPU loader tests. The Experiment 14 screen shaders, including physical CoC and all six display modes, remain unchanged.

## Key Concepts

### External Scene Loading and OBJ

OBJ can describe positions, normals, texture coordinates and polygon faces. A loader saves us from writing vertex arrays manually. tinyobjloader triangulates the faces; all mesh shapes are combined into one indexed draw. Points and lines are not rendered.

OBJ face corners can use separate position, normal and UV indices. OpenGL uses one index for the complete vertex, so the loader combines these indices and splits vertices at normal/UV seams. Supplied normals are retained and normalized. Missing or unusable normals receive flat triangle normals; missing UVs become zero.

### CPU Mesh Data vs GPU Buffers

`MeshData` stores positions, normals, UVs and triangle indices in CPU memory. `Mesh` holds a VAO, VBO, EBO and index count. Data is uploaded once before the render loop using `GL_STATIC_DRAW`, then CPU arrays are released. Each frame only updates transforms and draws. GPU buffers are deleted before GLFW terminates.

### Physical Scene Scale and Model Transform

**1 world unit = 1 meter**, but OBJ source units are not guaranteed to be meters. `importedSceneScale = 0.1` and the explicit import rotation/transform intentionally map this asset into the project's meter-based world. For another asset, choose meters per authored unit explicitly: use `0.01` for a centimeter-authored model. The loader does not center or normalize geometry to a unit cube.

For this asset the model matrix applies scale `0.1`, Y rotation `0°`, X rotation `−90°` (authored Z-up to world Y-up), then translation `(0, -0.75, 0)`. The resulting teapot is approximately 0.994 m wide, 0.578 m tall and 0.559 m deep. This is an explicit demo scale, not a claim about a real teapot's physical dimensions. From the initial camera `(0, 0, 5)`, its view depths span about 4.72–5.28 m. Foreground and background cube centers are about 2 m and 15–25 m away. These distances change when the camera moves.

### Actual Asset

The OBJ header identifies `20900_Brown_Betty_Teapot_v1`. The supplied file has 61,638 position records and 61,632 polygon faces. The real application reports **246,528 GPU vertices, 369,792 indices and 123,264 triangles** after triangulation and attribute-seam splitting. Normals and UVs are present; no normals needed generation. UVs remain unused by the simple shading path.

The referenced `20900_Brown_Betty_Teapot_v1.mtl` was not supplied. tinyobjloader warns about it, but geometry loading succeeds and the renderer uses its intended constant mesh color and directional light. The OBJ is approximately 14.45 MB and is not duplicated in the experiment archive.

### Normals and Basic Lighting

Imported geometry gets a warm constant color with one directional light: 25% ambient plus 75% Lambert diffuse. The shader transforms normals with the inverse transpose of the model matrix, preserving perpendicularity under non-uniform scale, then normalizes them for lighting. Cube vertex colors retain their previous unlit behavior. UVs are loaded but textures and materials are not rendered.

### Depth Testing with Imported Geometry

Imported meshes draw with `GL_DEPTH_TEST` and `GL_LESS` into the same FBO as the cubes. Ordinary raster depth writes populate `sceneDepthTexture`; there is no separate imported-mesh depth path. The screen pass still provides Color, RawDepth, LinearDepth, CoCMagnitude, CoCSigned and BasicDoF.

### DoF on Complex Geometry

Irregular silhouettes and overlapping depth layers expose haloing and foreground/background bleeding more clearly than isolated cubes. BasicDoF still uses a 17-tap gather with no neighboring-depth rejection. Adding geometry does not fix those occlusion limitations.

## Important Controls

| Control | Default / use |
|---|---|
| `screenMode` | `BasicDoF`; select any of the six modes |
| `focusDistanceMeters` | `5.0`; compare `2.0`, `5.0`, `15.0` |
| `focalLengthMillimeters` / `fNumber` | `50.0` / `1.4`; compare f/8 |
| `sensorHeightMillimeters` | `24.0` |
| `maxBlurRadiusPixels` | `12.0` |
| `importedScenePath` | `assets/models/scene.obj`, relative to the project |
| `importedSceneScale` | `0.1` meters per OBJ unit; must be positive |
| `importedScenePosition` / Y rotation | `(0, -0.75, 0)` / `0` degrees |
| Imported X rotation | `−90°` in the model transform for this Z-up asset |
| `showEnvironment` | `true`; disable for a self-contained scene |

Defaults remain source controls, but the following runtime keys avoid rebuilding during verification. Perspective FOV is still 45 degrees; near/far planes remain 0.1/100 m.

| Key | Runtime action |
|---|---|
| `1` / `2` / `3` | Color / RawDepth / LinearDepth |
| `4` / `5` / `6` | CoCMagnitude / CoCSigned / BasicDoF |
| `7` / `8` / `9` | Focus at 2 / 5 / 15 m |
| `F` | Toggle f/1.4 and f/8 |
| `R` | Reset camera position and orientation; retain current focus/aperture |
| `Tab` | Release/capture cursor and pause/resume camera navigation |
| WASD / mouse | Move / look while cursor is captured |
| `H` | Print key help and current settings |
| `P` / Escape | Save `output/latest.png` / exit |

The window title shows mode, focus distance, f-number, actual framebuffer pixel dimensions and cursor state. Setting changes also print to the terminal. Changes last for the current run only. See the [quick manual checklist](../../experiments/graphics/15_complex_scene/README.md#quick-manual-verification) for an ordered check.

## Observations and Known Limitations

`cmake --build build` passed, CTest passed 1/1, and the real application loaded the teapot and printed the counts above. Automated visual verification was blocked by macOS window-service/XPC restrictions. Final visual verification was performed manually by the user outside the sandbox and is authoritative; no successful automated visual capture is claimed.

| Final manual check | Result |
|---|---|
| Teapot orientation and scale | Acceptable in the scene; explicit rotation and scale are intentional import adjustments |
| Color / RawDepth / LinearDepth | PASS for all three modes |
| CoCMagnitude / CoCSigned / BasicDoF | PASS for all three modes |
| Focus at 2 / 5 / 15 m | PASS; focus changes were clearly visible |
| f/1.4 versus f/8 | PASS; aperture changes were clearly visible |
| Camera movement / resizing | PASS for both |

No major visual bugs were obvious in the final manual verification. Haloing, foreground/background bleeding, sparse 17-tap sampling and missing hidden-surface information remain theoretical/expected limitations of the unchanged BasicDoF baseline. They were not reported as strongly observed in this final verification; the earlier pre-adjustment render does not characterize the final result.

Materials, textures, smoothing-group reconstruction, automatic unit detection, automatic placement and scene hierarchies are not implemented. Triangulate difficult polygons in an authoring tool.

## Remember

Imported geometry follows the same pipeline as cubes. Loading removes manual geometry entry, but physical scale and placement remain explicit choices. See the [asset setup guide](../../assets/models/README.md) and [archived verification report](../../experiments/graphics/15_complex_scene/README.md#verification-report) for reproduction details and evidence.
