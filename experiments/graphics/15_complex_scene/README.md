# Experiment 15 — Complex Scene Loading

Status: complete. Build, CPU loader tests, real OBJ loading and final manual runtime verification passed. Final visual verification was performed by the user outside the sandbox.

## Git Before Experiment

The working tree was clean on `main`. `origin` was `https://github.com/dondurmaB/DoF_Graphics.git`. A live `git ls-remote origin refs/heads/main` confirmed `5a1111fe26b4fa5acd95b14fc6c32e8579d6ad00` (Experiment 14), matching local HEAD and `origin/main`.

## Loader and Snapshot

[tinyobjloader v2.0.0rc13](https://github.com/tinyobjloader/tinyobjloader/tree/v2.0.0rc13) is fetched by CMake at commit `2945a967c5303b2c8c14174117c45f3302591150`. It provides lightweight OBJ loading and triangulation without an engine or Assimp dependency.

This archive contains exact copies of:

- `CMakeLists.txt`
- `src/main.cpp`, `src/Mesh.cpp`, `include/Mesh.h`
- `shaders/basic.vert`, `shaders/basic.frag`, `shaders/screen.vert`, `shaders/screen.frag`
- `tests/mesh_loading.cpp`

The root project builds the active sources; this is a historical snapshot, not a separately configured target. Dependency sources, build products, temporary test fixtures and model assets are not duplicated here. Earlier experiment archives are unchanged.

## Asset and Physical Scene

Configured asset path: **`assets/models/scene.obj`**, relative to the project root. The user supplied the OBJ identified in its header as `20900_Brown_Betty_Teapot_v1`. It is included once in the project's asset folder and is not duplicated in this archive.

| Actual asset measurement | Value |
|---|---|
| File size | 14,446,756 bytes (approximately 14.45 MB) |
| SHA-256 | `1c96d356c5198b7dd45c4cbf4fec8cc01f1b7e42b3372b1ffc40cc270682b215` |
| Authored positions / polygon faces | 61,638 / 61,632 |
| Loaded GPU vertices | 246,528 |
| Loaded indices / triangles | 369,792 / 123,264 |
| Source normals / UVs | Both present |
| Generated flat normals | No |
| `importedSceneScale` | `0.1` meters per OBJ unit |
| Translation | `(0, -0.75, 0)` meters |
| Rotation | `0°` about Y, then `−90°` about X |
| Authored bounds | `(-4.9784, -2.7938, -0.0347)` to `(4.9596, 2.7938, 5.7429)` |
| Transformed world bounds | `(-0.49784, -0.75347, -0.27938)` to `(0.49596, -0.17571, 0.27938)` meters |
| Initial camera view-depth range | Approximately 4.72–5.28 m |

Counts, normal/UV availability and scale were printed by the real application. Bounds were independently measured from the OBJ and transformed using the current model matrix. The teapot measures roughly 0.994 × 0.578 × 0.559 m in world space; this is a chosen demo scale, not an assertion about its real physical size. No automatic normalization or centering occurs.

OBJ source units are not guaranteed to be meters. `importedSceneScale = 0.1` and the explicit import rotation/transform are intentional adjustments that map this model into the project convention **1 world unit = 1 meter**. The user accepted the resulting orientation and scale during final manual verification.

The referenced `20900_Brown_Betty_Teapot_v1.mtl` was not supplied. The application reports a material warning and successfully loads the geometry. The intended simple warm mesh color and directional lighting do not depend on the MTL or UV textures.

The initial camera is `(0, 0, 5)`, looking down negative Z. The layout provides a foreground cube around 2 m, a subject position around 5 m, background references at 15 and 25 m, a floor, and a back wall. When the asset is missing or invalid, two additional cubes fill the subject/depth-reference positions. Arbitrary assets require placement and scale adjustment; the default cannot guarantee their bounds or silhouettes.

Defaults: BasicDoF, 5 m focus, 50 mm focal length, f/1.4, 24 mm sensor height, maximum blur radius 12 framebuffer pixels. Model transforms remain source settings. Runtime verification keys select display modes, focus presets and aperture without rebuilding; see the checklist below. The title displays the current mode, focus, aperture, framebuffer pixel dimensions and cursor state.

Imported meshes use an indexed draw and the existing scene color/depth attachments. The screen shaders are byte-for-byte identical to Experiment 14, preserving linear depth, physical CoC and all six modes. Only imported geometry receives simple ambient plus Lambert lighting, with inverse-transpose normal transformation. UVs are retained but materials/textures are not rendered.

## Build and Tests

From the repository root:

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
./build/DepthResearch
```

CMake configuration and compilation passed. CTest passed **1/1**, covering quad triangulation and area, negative indices, vertex reuse, UV seams, multiple shapes, authored-unit preservation, supplied/missing/partial/zero normals, missing materials, degenerate faces, invalid indices, empty/missing files and rejection of empty GPU uploads before any GL call.

The first configure attempt hit sandbox DNS restrictions; configuring with approved network access fetched the dependencies successfully. Existing GLM CMake deprecation warnings remain non-blocking.

## Verification Report

Build and CTest passed as described above. The real application loaded `assets/models/scene.obj` at scale `0.1`, reported the measured counts and attributes, and did not select the cube fallback. The sandbox run encountered macOS window-service/XPC errors before FBO completion or a rendered frame could be confirmed and was stopped. **Automated visual verification was blocked; final runtime verification was performed manually by the user outside the sandbox.** The final manual report is authoritative; no automated visual success is claimed.

| Final manual check | User-reported result |
|---|---|
| Teapot orientation / scale | Acceptable in the scene; explicit rotation and scale are intentional import adjustments |
| Color | PASS |
| RawDepth | PASS |
| LinearDepth | PASS |
| CoCMagnitude | PASS |
| CoCSigned | PASS |
| BasicDoF | PASS |
| Focus at 2 / 5 / 15 m | PASS |
| f/1.4 versus f/8 | PASS |
| Camera movement | PASS |
| Resize | PASS |

The user reported no obvious major visual bugs and clearly visible changes in focus distance and aperture. Strong haloing, bleeding or sparse sampling were not reported in this final verification.

Additional evidence from the implementation review and CPU tests:

- The FBO setup and screen shaders match Experiment 14. The manually verified display modes exercise that scene color/depth path; the sandbox did not independently confirm FBO completion.
- Imported geometry uses the same scene-pass `GL_DEPTH_TEST` / `GL_LESS` and ordinary depth writes as the cubes. No separate imported depth path was added.
- OBJ loading and GPU upload occur before the frame loop; transforms and indexed drawing occur inside it. Buffers are destroyed before GLFW exits.
- CPU tests cover missing/invalid assets; the earlier no-asset application startup selected the fallback. The final real-asset startup loads the teapot.
- The physical scale convention and explicit transform are documented above. The archive contains exact copies of the final active code and shaders.

## Quick Manual Verification

Run `./build/DepthResearch` from the repository root. No source edits or rebuilds are needed between the following checks. Defaults remain BasicDoF, 5 m focus and f/1.4; press `H` to reprint help and settings at any time.

| Key | Action |
|---|---|
| `1` | Color |
| `2` | RawDepth |
| `3` | LinearDepth |
| `4` | CoCMagnitude |
| `5` | CoCSigned |
| `6` | BasicDoF |
| `7`, `8`, `9` | Focus at 2, 5, 15 m respectively |
| `F` | Toggle f/1.4 and f/8; confirm current value in title |
| `R` | Restore initial camera `(0, 0, 5)`, looking down −Z; focus/aperture stay unchanged |
| `Tab` | Release/capture cursor; camera movement pauses while released |
| WASD + mouse | Navigate while cursor is captured |
| `P` | Save `output/latest.png` (overwrites the previous capture) |
| `H` | Print help and current settings, including camera position/yaw/pitch |
| Escape | Exit |

1. **Startup/FBO:** Confirm `Loaded assets/models/scene.obj: 246528 vertices, 369792 indices, 123264 triangles` and `Scene framebuffer complete: ...`. The missing MTL warning is expected; `Using fallback scene.` is not expected with this asset.
2. **Orientation, lighting and placement:** Press `1`, `R`, then `Tab` to freeze the starting view. Inspect the upright teapot, scale, readable shading, floor and background. The current bounds imply the teapot base is about 4.7 cm above the floor top (`−0.75347` versus `−0.8` m); this placement is preserved for review, not silently adjusted. Confirm whether that spacing is acceptable. Model setup is in the [asset guide](../../../assets/models/README.md).
3. **Depth/occlusion:** Press `Tab` to capture the cursor, then use WASD/mouse to view overlapping mesh/environment surfaces. They should occlude by depth, not draw order. Press `R` and `Tab` again for a stationary comparison.
4. **Diagnostic modes:** Press `2`, `3`, `4`, `5` in turn. RawDepth may look almost white because perspective depth is nonlinear. LinearDepth should separate depth layers. CoCMagnitude should be dark around the focus plane; CoCSigned uses red for nearer defocus and blue for farther defocus. Press `8` for 5 m focus if needed.
5. **Focus:** Press `6` and use `7`, `8`, `9` to compare foreground, teapot and background focus. Use `4`/`5` alongside `6` to distinguish CoC response from color-gather artifacts. Keep the camera fixed.
6. **Aperture:** In `6`, keep the same focus/camera and press `F` to compare f/1.4 with f/8. The title confirms the setting. Defocused regions should generally blur more at f/1.4, subject to the unchanged 12-pixel cap; inspect a silhouette or detailed boundary rather than a uniform face.
7. **Resize:** With the cursor released, drag a window edge. Confirm the image and title's framebuffer pixel dimensions update, and the terminal reports a complete FBO at the new size. On HiDPI displays these dimensions can exceed logical window size. Press `Tab` again to resume mouse look and check for a smooth handoff.
8. **Record results:** For each step report passed / failed / untested. For failures, include the mode, focus, f-number, viewpoint and visible symptom. `P` can capture the current frame; copy it to another filename before taking another if retaining comparisons.

This checklist is retained for reproduction. The final user-reported results are recorded above; passing a build alone is not visual verification. Experiment 16 is outside this archive's scope.

## Limitations and Observations

In the final manual verification, no major visual bugs were obvious, and changes in focus distance and aperture were clearly visible. The teapot's corrected orientation and scale were accepted as intentional import adjustments.

Haloing, foreground/background bleeding, sparse 17-tap sampling and missing hidden-surface information remain theoretical/expected limitations of BasicDoF. Do not interpret these as strongly observed artifacts in the final verification. Earlier pre-adjustment observations do not describe the accepted final result, and no blur redesign or artifact fix is claimed.

Missing textures/materials, generated flat normals for assets without them, arbitrary OBJ units/origins, and approximate polygon triangulation limit asset fidelity. No successful automated rendered-frame verification is claimed because macOS window services blocked it.

No ray tracing, BVH, AI or depth prediction was added.

[Experiment notes](../../../notes/graphics/15_complex_scene.md)
