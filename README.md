# Depth of Field Research

Educational C++17 / OpenGL 3.3 Core project for learning the graphics pipeline before building monocular-depth-based Depth of Field experiments.

The longer-term pipeline is:

RGB image -> monocular AI depth estimation -> predicted depth map -> OpenGL Depth-of-Field post-processing

The OpenGL renderer starts as a controlled source of ground-truth depth for later dataset generation, training, and evaluation.

## Architecture Roadmap

Stage 1 - OpenGL fundamentals: window, context, VAO, VBO, shaders, triangle.

Stage 2 - Camera and projection: model, view, projection matrices and coordinate systems.

Stage 3 - Depth buffer: depth testing, depth precision, and depth texture output.

Stage 4 - Framebuffer/post-processing: render targets and screen-space passes.

Stage 5 - Depth of Field: circle of confusion, blur passes, foreground/background separation.

Stage 6 - Dataset generation: RGB frames, ground-truth depth maps, camera metadata.

Stage 7 - AI depth estimation: PyTorch monocular depth model experiments.

Stage 8 - Integration: compare predicted depth with renderer depth and drive DoF from both.

## Environment Setup

Clone the repository:

```sh
git clone https://github.com/dondurmaB/DoF_Graphics.git
cd DoF_Graphics
```

Install Apple's Command Line Tools if they are missing:

```sh
xcode-select --install
```

Create and activate the Conda environment:

```sh
conda env create -f environment.yml
conda activate dof-graphics
```

Update the environment later after editing `environment.yml`:

```sh
conda env update -f environment.yml --prune
conda activate dof-graphics
```

Configure and build:

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

Run:

```sh
./build/DepthResearch
```

## Current checkpoint: café and matched lighting

Start with [the Phase 1 report](notes/graphics/cafe_phase1.md), including framing arithmetic,
linear-light measurements, images and unfinished phases. The accepted starting point was
`dof-research 3`; previous café attempts were not reused. Previous experiment archives are unchanged.

`scene/cafe.scene` supplies geometry, camera, lighting, capture dimensions and optional imported
transform to both `src/SceneFile.cpp` and `tools/scene/scene_loader.py`. Geometry is generated once
by Python; renderers only load it. The alley remains a regression scene.

```sh
python tools/scene/build_cafe.py
./build/DepthResearch                         # café, 50 mm, focus 2.5 m, f/1.2
./build/DepthResearch --scene scene/alley.scene
python tools/scene/preview_scene.py           # CPU framing preview, no shadows or DoF
ctest --test-dir build --output-on-failure
```

The café has a finite table, identical 110 mm cups at view depths 2.5/4/6/9/12 m, curved foliage,
round chair stock, chamfered fixtures, wall boards/prints, hanging mugs, counter dressing,
condiment caddies, trailing plants, and HDR emitters. Cup/pot bodies are single frusta.
The shader remains diffuse-only: the measured GGX/Schlick-to-Cycles Glossy mapping failed.

Matched lighting is `albedo * (ambient + sun * max(N.L,0) * visibility / pi)`.
Cycles adds the ambient as primary-camera-only fill; it is intentionally non-physical.
Geometry, camera and the direct shading equation match by construction and tests.
Shadow visibility, pixel filtering, finite precision, and DoF integration still differ.
`--full-gi` on the Blender tool selects physical indirect transport, explicitly not comparable.

## Controls

| Input | Action |
|---|---|
| `1`–`6` | Sharp color, raw depth, linear depth, CoC magnitude, CoC signed, basic DoF |
| `0` | Sharp/DoF split view |
| Aperture buttons | f/1.2, 1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22 |
| `K` | Original scene framing, focus and lens at f/1.2 |
| `T` | Restore scene camera and lens defaults |
| `7` / `8` / `9` | Focus 2 / 5 / 15 m (panel also has the café's 2.5 m) |
| `F` / `B` | Legacy f/1.4↔f/8 toggle / f/2.8 |
| `L` / `R` | Reload selected scene / reset camera pose |
| WASD, right mouse | Move / look |
| `Tab`, `G` | Capture/free pointer; cycle compact/full/hidden panel |
| `[` / `]`, `,` / `.` | Change focal length; sensor height |
| `P` | Save UI-free interactive screenshot and settings-tagged copy |
| Escape | Exit |

The CoC display is labelled as a **near/far plane bound**, not the maximum present in the image.
The radius ceiling is computed for the widest aperture at the current resolution/lens/focus.
The original 100-tap gather is still the baseline; an improved gather is pending.

## Reproducible inspection captures

```sh
./build/DepthResearch --batch --sharp --output output/sharp.png
./build/DepthResearch --batch --lens 85 --focus 2.5 --fstop 1.2 \
  --coc-samples 192 --output output/gl_focus2.5m_f1.2_85mm.png
./build/DepthResearch --batch --lens 85 --focus 2.5 --fstop 22 \
  --coc-samples 192 --output output/gl_focus2.5m_f22_85mm.png
./build/DepthResearch --dry-run
python tools/raytraced_reference/render_dof.py --dry-run
python tools/verify/check_cafe.py --dry-run
python tools/verify/check_cafe.py
python tools/verify/measure_lighting.py --render
```

Batch renders through the normal shaders and explicitly allocated 1200×1200 pixel FBOs,
independent of window/Retina scaling; it queries dimensions and fails on mismatch. `--width`
and `--height` override the shared scene size. `--scene`, `--lens`, `--sensor-height`, `--focus`,
`--fstop`, `--coc-samples`, `--x`, `--y`, `--z`, `--yaw`, and `--pitch` support inspection views.
`--import-mesh` is required to add the OBJ. A companion `.linear.pfm` is the **sharp pre-lens**
radiance buffer, not the defocused output. Interactive screenshots still use actual window
framebuffer size.

`check_cafe.py` produces scene-camera inspection views plus the default 85 mm comparison pair:
focus 2.5 m at f/1.2 and f/22. The wide-open setting makes far-wall blur large enough to inspect;
f/22 is the near-pinhole control. It is a Phase 1 inspection runner, **not** the planned
convergence-certified reference/comparison pipeline. See [Cycles instructions](tools/raytraced_reference/README.md).
The legacy PNG comparison now fails with no usable pairs and reports stale hashes truthfully;
it remains a display-space diagnostic, not a linear HDR benchmark or proof of matching cameras.

## macOS Notes

This project uses the OpenGL framework supplied by macOS. It does not install an OpenGL implementation through Conda.

The GLFW context requests OpenGL 3.3 Core Profile and sets `GLFW_OPENGL_FORWARD_COMPAT = GL_TRUE`, which is required for modern OpenGL contexts on macOS.

On Retina / HiDPI displays, the GLFW window size and the actual framebuffer size can differ. Rendering uses `glfwGetFramebufferSize()` and a framebuffer resize callback so `glViewport()` always matches the real drawable pixel size.

The CMake configuration avoids hard-coded Homebrew paths, so it works on both Apple Silicon and Intel Macs as long as developer tools and the Conda environment are available.

## Understanding the First Triangle

The data path is:

CPU vertex array -> VBO -> VAO configuration -> vertex shader -> primitive assembly -> rasterization -> fragment shader -> framebuffer

A VBO stores raw vertex bytes in GPU memory. In this program, each vertex contains three position floats followed by three color floats.

A VAO remembers how vertex attributes are read from the currently configured vertex buffers. Here it records attribute 0 as position and attribute 1 as color.

`layout(location = 0)` and `layout(location = 1)` give the shader inputs explicit attribute indices. Those indices match the calls to `glVertexAttribPointer()` in C++.

Three vertices form one triangle because `glDrawArrays(GL_TRIANGLES, 0, 3)` groups every three submitted vertices into an independent triangle.

The vertex shader runs once per vertex, so it runs three times here. The fragment shader runs for every covered screen fragment, so it runs many more times.

Vertex colors become gradients because rasterization interpolates vertex outputs across the triangle before each fragment reaches the fragment shader.

## Troubleshooting

If CMake cannot find a compiler, install or repair Xcode Command Line Tools:

```sh
xcode-select --install
```

If the window cannot create an OpenGL 3.3 Core context, check that you are running on macOS hardware and drivers that still expose the macOS OpenGL framework.

If GLAD initialization fails, make sure `gladLoadGLLoader()` runs after `glfwMakeContextCurrent()`. OpenGL function pointers are context-dependent.

If shaders are not found, rebuild with CMake. The build copies `shaders/` next to the executable, and the program also falls back to the source-tree shader directory.

If the triangle looks stretched or clipped after moving the window between displays, confirm the framebuffer resize callback is firing and that `glViewport()` receives framebuffer dimensions rather than logical window dimensions.

If `cmake -G Ninja` fails because Ninja is missing, activate the Conda environment first. `environment.yml` includes `ninja`.
