# OBJ Models

The supplied `assets/models/scene.obj` identifies itself as `20900_Brown_Betty_Teapot_v1` in its OBJ header. It is 14,446,756 bytes. The application loads it as 246,528 GPU vertices, 369,792 indices and 123,264 triangles, with authored normals and UVs and no generated normals. The referenced `20900_Brown_Betty_Teapot_v1.mtl` is absent; this produces a warning but does not prevent geometry loading.

To use another OBJ, replace the asset or change `importedScenePath` in `src/main.cpp` to a project-relative path. A missing or invalid asset prints an error and selects the fallback scene.

The loader reads triangle/polygon faces, positions, normals and UVs. tinyobjloader triangulates faces. Missing or unusable normals receive flat triangle normals; missing UVs become zero. Materials and textures are not rendered: imported geometry uses a warm mesh color with ambient plus directional diffuse lighting. Prefer triangulating complicated polygons in the authoring tool.

`importedSceneScale` is meters per OBJ unit, currently `0.1`. For a model authored in centimeters, use `0.01`. Units are never automatically normalized. The current model transform applies scale, zero Y rotation, a fixed `−90°` X rotation to convert this asset from Z-up to Y-up, then translation `(0, -0.75, 0)`. Adjust the scale, position and rotations for a different asset.

The transformed teapot bounds in meters are approximately `(-0.49784, -0.75347, -0.27938)` to `(0.49596, -0.17571, 0.27938)`. This chosen demo scale gives a roughly 0.578 m tall subject; it is not an inferred real-world size.

The initial camera is at `(0, 0, 5)` facing negative Z. The subject should sit around 5 m view depth, with a foreground reference around 2 m and background references around 15–25 m. Set `showEnvironment = false` for a self-contained imported scene.

Before committing an asset, check its license, attribution requirements and size. Keep large unrelated models and generated captures out of the repository. Preserve any required asset attribution alongside the model. The experiment archive refers to this folder instead of duplicating model files.
