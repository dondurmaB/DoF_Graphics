# Matched specular and café geometry — dof-research 6

Follow-up: [expanded validation, numerical limits, and roughness authoring floor](ggx_validation_limits.md)
records the October 4 revalidation, the five low-roughness oracle failures,
and the conservative authoring rule `roughness >= 0.12`.

This follow-up fixes the dome profile, adds a numerically verified direct-sun
specular term to both renderers, and improves visible edges and machine hardware.
It does not change the research camera, focus, lens, f-stop ladder, or DoF gather.
It is a controlled shading comparison, not a claim of full photorealism.

This folder arrived without Git metadata. Input files were preserved in local
baseline commit 82ffa30; work is on fix/matched-specular-and-cafe-geometry.
No remote is configured, and this branch has not been pushed or merged.
The existing untracked imgui.ini is preserved outside the commit.

## Geometry and material changes

The dome now uses contiguous frusta: r(v) = R sqrt(1-v²), bottom radius r(i/n),
top/bottom ratio r((i+1)/n)/r(i/n). The last ring has taper zero. The old 2 mm
overlap is removed; nine-decimal serialization avoids small gaps at joins.
There are still internal caps and piecewise slope changes, but no radial ledges.
The cylinder-loop audit found no other remaining untapered curved-profile
approximation: cup/tumbler/pot/lamp profiles already taper; plate/cup stacks,
chair hoops, grate bars and supports are separate physical pieces.

Boxes accept an optional bevel width in metres. Six inset faces, twelve edge
strips and eight corners cost 44 triangles instead of 12; bevel zero retains
the old path. The counter top/front, cabinet doors, shelves and machine are
chamfered. The hero table uses three contiguous rings with a tapered rim.

The machine was outside the fixed 50 mm camera, contrary to the brief's
mid-ground assumption. It moves along the same counter from (2.08, -0.23, 1.05)
to (1.88, -0.23, -4.62). The lowest shelf/bottles leave an alcove, the front
panel is shorter, and group heads, handles and wands project in front of it.
Steel, chrome and dark painted steel provide contrast. It is visible near the
right edge of the fixed camera and deliberately defocused at f/1.2.
Camera rays to both heads, handles and wands first intersect within their
surface extents (about 74, 67 and 14 mm from the respective centre targets).
The wide inspection also exposed a tray overhang: the pastry case moves from
z=2.35 to z=1.85 so its entire 0.52 m base rests on the counter.

| Metric | Input snapshot | Final |
|---|---:|---:|
| Primitives | 2,692 | 2,678 |
| Vertices | 135,050 | 134,957 |
| Triangles | 115,864 | 115,240 |
| Emissive triangles | 5,820 | 5,820 |

Triangle counts exclude the optional imported teapot, which is not drawn in the
default café. Deleting obstructing bottles offsets the added chamfers.

Scene format stays version 1. All primitives accept rough (perceptual roughness,
alpha=rough², range 0.05–1) and spec (scalar normal-incidence F0, range 0–1).
Defaults rough=0.5/spec=0 reproduce the legacy material equation exactly:
the ENTIRE specular lobe is disabled at zero, including grazing Fresnel.
Changing the matched sun angle, described below, can still change visibility
relative to historical Cycles renders; legacy images are not byte-identical.
Only boxes accept bevel; it must be nonnegative and below half the smallest
absolute dimension. Attributes 5 and 6 carry roughness/F0; 3 remains reserved
for imported UVs and 4 remains emission.

Representative authored rough/F0: ceramic .16/.04, warm ceramic .20/.04,
chrome .20/.70, steel .36/.60, painted steel .40/.045, wood .36–.42/.04,
frosted glass .34/.04, terracotta .85/.008, tile .24–.32/.04.
Glass is an opaque frosted approximation, without refraction. Scalar F0 is
not a spectral conductor model. Unassigned materials retain spec=0.

## Exact shared formulation

For unit N, L and V, let H=normalize(L+V), nl=N·L, nv=N·V,
nh=N·H, vh=V·H, alpha=rough² and a²=alpha²:

    D = a² / (pi * ((1-nh²) + nh²*a²)²)
    Lambda(x) = (sqrt(1 + a²*(1-x²)/x²) - 1) / 2
    G2 = 1 / (1 + Lambda(nl) + Lambda(nv))
    F = F0 + (1-F0)*(1-vh)^5
    f_spec = D*G2*F / (4*nl*nv)
    f_spec = 0 if F0=0, nl<=0 or nv<=0
    E = sunColor * energy * max(nl,0) * visibility
    radiance = (1-F0)*albedo*(sky + E/pi) + E*f_spec

This is isotropic single-scatter GGX with height-correlated Smith masking.
The constant (1-F0) diffuse allocation also applies to the explicit ambient
emission in both renderers. It is not a layered dielectric energy-compensation
model. Emitters keep their previous independent radiance path.

GLSL evaluates this directly. Cycles adds a zero-roughness Diffuse BSDF weighted
by (1-F0) to Metallic BSDF with GGX distribution, F82 Fresnel, white Edge Tint,
zero anisotropy and zero thin-film thickness. White Edge Tint makes the F82
correction B zero, leaving Schlick's half-vector Fresnel exactly. A gate disables
that closure at F0=0. No Principled, clearcoat, sheen or multiscatter GGX is used.
The production graph lives in materials.py and is reused by the numerical probe.

Sources checked against the installed Blender 5.2.2 implementation:
[GGX/Smith and F82 setup](https://raw.githubusercontent.com/blender/blender/v5.2.2/intern/cycles/kernel/closure/bsdf_microfacet.h),
[Fresnel evaluation](https://raw.githubusercontent.com/blender/blender/v5.2.2/intern/cycles/kernel/closure/bsdf_util.h).
The new graph requires the Metallic node; Blender 4.2 compatibility is not claimed.

The input Cycles setup used a 1.4° sun disc while GL evaluated one light
direction. That integrates different highlights even with identical BRDFs.
Matched mode now uses a zero-angle sun; full-GI mode retains the authored angle.
Both effective and authored angles are recorded in render metadata.
Matched diffuse/glossy/transmission/volume bounce limits remain zero.

## Numerical evidence

The probe compiles the marked function from the actual basic.frag, renders into
RGBA32F and reads its linear float result. It independently renders the actual
production Cycles graph on a planar patch with a unit directional sun and
constant ambient fill. An orthographic camera, black world, zero sun angle,
64 CPU samples, no denoising and 32-bit EXR isolate the material term.

24 cases cover six light/view/relative-azimuth configurations, including normal
and grazing views, with rough/F0 pairs (.12,.04), (.35,.04), (.65,.65), (.5,0).
Relative azimuth is necessary: N·L and N·V alone do not specify the BRDF.
A separate Python oracle uses the equivalent square-root form of Smith G2.

Measured on Blender 5.2.2 LTS and Apple M4 Max OpenGL:

- Maximum Cycles versus GLSL relative RGB error: 1.618224e-6 (0.000162%).
- Maximum absolute linear RGB error: 8.741472e-5.
- Maximum GLSL versus double-precision oracle relative BRDF error: 0.00114852 (0.114852%).
- Acceptance: Cycles error <= 2e-5 + 0.005*max(expected RGB);
  GLSL/oracle error <= 1e-6 + 0.003*abs(expected BRDF).

The complete per-case measurements are in
[matched_specular_agreement.json](matched_specular_agreement.json).
These are measured patch tests, not whole-scene image error or proof for every
normal/material configuration.

## Verification and images

Build and all seven CTest suites passed. New checks cover material propagation,
legacy defaults, invalid material/bevel inputs, bevel topology, dome joins,
alley loading and BRDF reciprocity/limits. Both loaders use the regenerated,
identical pinned geometry numbers. C++ additionally checks byte identity for
taper=1. Full CTest output follows at the end of this note.

Completed:

    cmake -S . -B build -G Ninja
    cmake --build build
    ctest --test-dir build --output-on-failure
    python3 tools/scene/build_cafe.py
    .venv/bin/python tools/scene/preview_scene.py --scene scene/cafe.scene --output output/after.png
    python3 tools/raytraced_reference/render_dof.py --ground-truth --dry-run
    python3 tools/compare/run_comparison.py --dry-run
    /Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 --python tools/raytraced_reference/verify_brdf.py
    /Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 --python tools/raytraced_reference/render_dof.py -- --preview --fstops 1.2 11
    ./build/DepthResearch --batch --fstops 1.2 11 --width 600 --height 600 --output output/matched

The local venv supplies NumPy/Pillow for the CPU preview. Build emits existing
GLM compatibility warnings; the native-GL probe also emits macOS OpenGL
deprecation warnings. The application logs the pre-existing missing teapot MTL;
the optional imported mesh is not drawn in the café view.

macOS graphics access was approved for these runs. Actual OpenGL scene and
shadow FBOs completed; final captures are 1200×1200. A 1200-point batch window
initially produced 2400 pixels on Retina. This is an existing batch sizing issue,
so the final capture explicitly uses 600 window points for 1200 framebuffer
pixels. No pipeline filename convention was changed.

Local review artifacts (ignored output, intentionally not committed):

- [Before CPU scene camera](../../output/before.png), [after](../../output/after.png).
- [Before CPU 12 mm wide view](../../output/before-wide.png), [after](../../output/after-wide.png).
- [Cycles 12 mm wide inspection](../../output/after-wide-cycles.png): complete dome, tray support and machine alcove.
- [Cycles f/1.2](../../reports/raytraced_dof/rt_focus1.5m_f1.2.png), [f/11](../../reports/raytraced_dof/rt_focus1.5m_f11.png).
- [OpenGL f/1.2](../../output/matched/gl_focus1.5m_f1.2.png), [f/11](../../output/matched/gl_focus1.5m_f11.png).

CPU previews omit shadows/DoF and shade per triangle; use them for geometry,
not BRDF validation. Cycles scene previews use 32 samples and denoising; they
are not converged training ground truth. The 4096-sample ground-truth and
comparison commands were dry runs only. The 12 mm lens is an inspection-only
override; the scene file, renderer defaults and standard captures stay 50 mm,
24 mm sensor, 1.5 m focus and the original 1.2/1.4/2.8/11 f-stop ladder.

## Limits and approaches not used

- The brief's claim that only DoF differs was already too strong: PCF shadow
  visibility/bias differs from Cycles ray visibility. Making the sun directional
  fixes light integration, not that shadow discrepancy.
- Thin-lens Cycles samples V at each aperture point; screen-space DoF blurs
  centrally shaded radiance. View-dependent highlight variation is therefore
  another limitation of the approximation, even with the same BRDF.
- Cycles may correct shading normals near silhouettes; the planar probe has
  N=geometric N and does not establish equivalence for that correction.
- Raster coverage, half-float radiance, pixel filtering, lens occlusion,
  sampling noise, denoising and gather truncation still differ. No new
  whole-image convergence or quantitative DoF-quality claim is made.
- Both matched renderers omit GI, reflected environment detail, refraction and
  physical ambient occlusion. Metallic colours are approximate; frosted glass
  stays opaque. The scene remains visibly stylized, with faceted handles and
  simplified objects. Highlights help, but do not make it a photograph.
- A Fresnel/Layer Weight mix was rejected before implementation because its
  view-normal Fresnel is not the required view-half-vector Fresnel. Principled
  and multiscatter GGX were not used because their extra terms would need
  matching implementations. No unmatched shading lobe was retained.
- Full-scene setup initially failed due to loading the shared scene after
  material initialization; fixed before successful final renders. An early dome
  test caught serialization gaps; precision was fixed, not the tolerance.
  Initial wider shots still cropped the dome, so the final inspection uses 12 mm.

## File-by-file review

| File | Change and reason |
|---|---|
| CMakeLists.txt | Build the real GPU BRDF probe and register the CPU BRDF suite. |
| include/SceneFile.h | Add roughness/F0 vertex fields; reserve UV attribute 3. |
| src/SceneFile.cpp | Parse/validate materials and bevel; tessellate chamfered boxes. |
| src/main.cpp | Upload attributes 5/6 and camera position for view-dependent shading. |
| shaders/basic.vert | Carry the new material attributes. |
| shaders/basic.frag | Evaluate the verified GGX term with explicit legacy-zero behavior. |
| tools/scene/scene_loader.py | Python twin of the grammar, chamfer and material propagation. |
| tools/scene/authoring.py | Tapered dome, precision control and material-aware colors. |
| tools/scene/build_cafe.py | Restrained materials, visible chamfers, machine alcove and supported pastry tray. |
| scene/cafe.scene | Regenerated deterministic shared scene; camera unchanged. |
| tools/scene/preview_scene.py | Approximate GGX preview and temporary wide-view camera overrides. |
| tools/raytraced_reference/materials.py | Shared, explicit Cycles Lambert/GGX/fill node graph. |
| tools/raytraced_reference/render_dof.py | Upload mesh materials; use shared graph and matched directional sun. |
| tools/raytraced_reference/brdf.py | Independent scalar reference and numerical cases. |
| tools/raytraced_reference/verify_brdf.py | Render/compare actual GLSL and Cycles with failing tolerances. |
| tests/brdf_gpu.cpp | Compile production GLSL and read unclipped float pixels. |
| tests/test_brdf.py | Reciprocity, analytic normal-incidence and disabled/backface limits. |
| tests/scene_file.cpp | Updated pinned values, material/bevel checks and byte-identical taper check. |
| tests/test_scene_file.py | Same pinned values plus dome/material/bevel/alley regression coverage. |
| tools/raytraced_reference/README.md | Current graph requirements and verification command. |
| notes/EXPLAINER.md | Link the current material model and caveats. |
| notes/graphics/README.md | Index this follow-up without renumbering historical experiments. |
| notes/graphics/matched_specular_and_geometry.md | This complete change and verification record. |
| notes/graphics/matched_specular_agreement.json | Recorded numerical evidence for the reviewed implementation. |

## Full CTest output

    Test project /Users/beibarys.myrzash/Documents/Classes Fall 2026-27/Directed Research/dof-research 6/build
        Start 1: mesh_loading
    1/7 Test #1: mesh_loading .....................   Passed    0.03 sec
        Start 2: physical_camera
    2/7 Test #2: physical_camera ..................   Passed    0.00 sec
        Start 3: scene_file
    3/7 Test #3: scene_file .......................   Passed    0.08 sec
        Start 4: reference_config
    4/7 Test #4: reference_config .................   Passed    3.09 sec
        Start 5: scene_file_python
    5/7 Test #5: scene_file_python ................   Passed    1.21 sec
        Start 6: brdf_oracle
    6/7 Test #6: brdf_oracle ......................   Passed    0.04 sec
        Start 7: pipeline
    7/7 Test #7: pipeline .........................   Passed    0.77 sec

    100% tests passed, 0 tests failed out of 7

    Total Test time (real) =   5.23 sec
