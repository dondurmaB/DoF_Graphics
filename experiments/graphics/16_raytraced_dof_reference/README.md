# Experiment 16 — Physically Matched Camera and Ray-Traced DoF Reference

This snapshot adds physical raster projection and a separate Blender Cycles reference workflow. Experiment 15 was preserved at `42c46d2`, confirmed on local `main` and live `origin/main` before changes. Verification is complete only as far as this environment allows: Blender was unavailable and macOS window services blocked OpenGL visual checks.

## Implementation

The default projection uses `2 atan(sensorHeight / (2 focalLength))`: 50 mm lens, 24 mm sensor height, **26.9915° vertical FOV**. `usePhysicalCameraProjection` defaults to true; the legacy 45° setting remains available. The six display modes, original CoC calculation and 17-tap BasicDoF shaders are unchanged. Focus remains 5 m, aperture f/1.4, clipping 0.1–100 m, and maximum gather radius 12 pixels.

`T` restores the reference camera/settings and releases the cursor; `V` toggles projection. `[/]` adjust focal length, `,/.` adjust sensor height, `B` selects f/2.8, `F` toggles f/1.4/f/8, and `7/8/9` select 2/5/15 m focus. `H` reports settings and framebuffer pixels. Existing navigation, modes, resizing and screenshot controls remain implemented.

Cycles uses the same OBJ and explicit `0.1` import scale, with `T(0,-0.75,0) * Rx(-90°) * Ry(0°) * S(0.1)`. OBJ units are not assumed to be meters; this transform maps them into the project's meter convention. A common `(x,y,z) → (x,-z,y)` conversion applies to the model, supporting boxes, floor, wall, camera and lighting direction. The OpenGL camera `(0,0,5)`, forward `(0,0,-1)`, up `(0,1,0)` becomes Blender `(0,-5,0)`, forward `(0,1,0)`, up `(0,0,1)`.

The script configures Cycles lens sampling, vertical sensor fit, 5 m focus and f/1.4, f/2.8, f/8 outputs. Defaults are 1200×1200 framebuffer pixels, 32 preview or 128 final samples, and OpenImageDenoise. Device selection attempts Metal, with CPU fallback; no actual rendering device has been verified. Optional controls support 2/15 m focus, other image dimensions and a sharp reference. Generated PNG/JSON files stay outside this archive and are ignored by Git.

## Verification Record

| Check | Result |
|---|---|
| `cmake --build build` | PASS |
| `ctest --test-dir build --output-on-failure` | PASS, 3/3: mesh loading, physical camera, reference configuration |
| Actual application startup and OBJ loading | PASS: 246,528 GPU vertices, 369,792 indices, 123,264 triangles; normals and UVs present |
| Physical FOV | PASS in unit tests and startup output: 26.9915° |
| Lens/sensor response and non-square projection | PASS in mathematical/configuration tests; rendered response unverified |
| Scene conversion, import transform and reference jobs | PASS in Python configuration tests and dry run |
| Color, depth/CoC modes, BasicDoF, FBO completion | Experiment 16 visual checks blocked by macOS window-service restrictions |
| Focus/aperture response, navigation, resize, reference framing | Implementations preserved; new runtime visual verification pending |
| Blender on PATH and macOS Applications path | Not installed at either checked location |
| Blender version, Cycles execution, OBJ import, orientation, scene/camera matching | Not runtime-verified |
| Cycles f/1.4 versus f/8, output PNGs and actual rendering device | Not verified; no reference images generated |
| Paired BasicDoF/Cycles comparison | Not performed; no observed differences claimed |

Experiment 15's manual results remain valid for that experiment; they do not establish visual correctness of Experiment 16's narrower projection. An existing 1200×1200 OpenGL capture supplied the resolution default, but is not an Experiment 16 verification image.

## Run and Review Locally

From the repository root:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
./build/DepthResearch
```

Press `T`, then `H`. Check all six modes, focus/aperture changes, lens/sensor FOV changes, camera movement and resizing. Capture sharp Color and BasicDoF with `P`, preserving each `output/latest.png` before the next capture. Record actual framebuffer dimensions.

After installing Blender, run the matching reference (replace width/height if needed):

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
  --python tools/raytraced_reference/render_dof.py -- --preview --sharp --width 1200 --height 1200
```

Use `blender` instead of the absolute executable when it is on PATH. Omit `--preview` for 128 samples; `--device cpu` forces CPU. Confirm version/device, imported geometry, camera framing, focus, aperture response and generated images. The [archived tool guide](tools/raytraced_reference/README.md) contains all commands and settings.

## Limits and Snapshot Contents

Haloing, foreground/background bleeding, sparse sampling and missing hidden surfaces remain expected BasicDoF limitations; none are claimed as newly observed here. The inherited CoC expression describes a diameter but BasicDoF uses its magnitude as a capped radius. Cycles should improve lens-dependent visibility, while lighting, triangulation, color management, noise and denoising also affect comparisons.

**Cycles is a quality reference. The ≤20 ms/frame target applies to the future AI solution, not Cycles.** No AI stage is included.

The snapshot contains the application/mesh sources, physical camera helper, four shaders, CMake configuration, tests and separate reference tool. It is an archive, not a standalone checkout: building requires the repository's existing `external/` dependencies and original `assets/models/scene.obj`. No large model copy, generated renders, caches or temporary harnesses are included. Earlier archives are unchanged.

See [research notes](../../../notes/graphics/16_raytraced_dof_reference.md).
