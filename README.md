# Depth of Field Research

The current, qualified stage-2 experiment is documented in
[the run guide](renderer/mitsuba/STAGE2.md) and
[the meeting account](notes/graphics/stage2_qualified_gather.md).
It feeds Mitsuba sharp/depth into the production OpenGL gather and compares
against independently noise-qualified Mitsuba thin-lens renders. It does not
compare the independently authored OpenGL and Mitsuba cafés.

Educational C++17 / OpenGL 3.3 Core project for learning the graphics pipeline before building monocular-depth-based Depth of Field experiments.

The longer-term pipeline is:

RGB image -> monocular AI depth estimation -> predicted depth map -> OpenGL Depth-of-Field post-processing

The OpenGL renderer starts as a controlled source of ground-truth depth for later dataset generation, training, and evaluation.

## Project Memory

- [project_description.md](project_description.md) records the stable project purpose, constraints, and current direction.
- [feature-list.md](feature-list.md) tracks shipped, in-progress, and deferred features.
- [progress.md](progress.md) records current execution status, validation, and known risks.

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
cmake -S . -B build
cmake --build build --target DepthResearch DoFScene DoFApproaches DoFSceneTests
```

Run the original teaching renderer:

```sh
./build/DepthResearch
```

Run the procedural cafe depth-of-field scene:

```sh
cmake --build build --target DoFScene
./build/DoFScene.app/Contents/MacOS/DoFScene
```

Run the three-way depth-of-field comparison:

```sh
cmake --build build --target DoFApproaches
./build/DoFApproaches.app/Contents/MacOS/DoFApproaches --method 2 --samples 64
```

Run the available tests:

```sh
cmake --build build --target DoFSceneTests
ctest --test-dir build --output-on-failure
```

## Render Targets

`DepthResearch` is the original numbered-experiment renderer. It remains the learning path for OpenGL fundamentals and the archived experiments under `experiments/graphics/`.

`DoFScene` is a separate native macOS GLFW application for the cafe tabletop depth-of-field demo. It keeps the teaching snapshots intact while providing a richer procedural scene with an interactive camera, focus controls, diagnostic render modes, validation capture paths, and a screenshot path for visual review.

`DoFApproaches` renders one shared analytic scene through three different depth-of-field methods so they can be subtracted from each other: a single-layer screen-space gather, multi-view aperture accumulation, and lens-sampled ray tracing. It is the code behind Experiments 15-17.

`renderer/mitsuba/` is the physically based path: a Mitsuba 3 cafe interior with global illumination, sunlight through windows, refraction and metals. It renders pixel-aligned all-in-focus, thin-lens depth-of-field and ground-truth depth, locally on Apple Metal or on the HPC with CUDA. See [renderer/mitsuba/README.md](renderer/mitsuba/README.md).

## DoFApproaches

A three-way comparison of depth-of-field methods on one scene. All three share the same geometry and the same surface shading, so they differ in exactly one respect: how visibility through the aperture is resolved. That is what makes their difference images meaningful.

```sh
cmake --build build --target DoFApproaches
BIN=./build/DoFApproaches.app/Contents/MacOS/DoFApproaches

$BIN --method 1 --samples 64  --size 1280x860 --capture output/dof3-m1-gather.png
$BIN --method 2 --samples 64  --size 1280x860 --capture output/dof3-m2-multiview-64.png
$BIN --method 3 --samples 512 --size 1280x860 --capture output/dof3-m3-raytrace-512.png

python3 tools/compare_dof_methods.py
```

| Method | What it does | Error type |
|---|---|---|
| `--method 1` | Screen-space gather over one colour and depth image | Bias: more taps do not fix it |
| `--method 2` | N full scene renders from N points on the aperture, sheared to hold the focus plane | Banding from N discrete views, falls as 1/N |
| `--method 3` | One stochastic lens sample per pixel per frame, accumulated | Noise, falls as 1/sqrt(N) |

Options: `--method 1..3`, `--samples N` (64 max for methods 1-2, 512 for method 3), `--frames N`, `--focus meters`, `--f-number`, `--lens mm`, `--sensor-height mm`, `--max-radius pixels`, `--size WIDTHxHEIGHT`, `--capture path`.

Controls: `1`/`2`/`3` switch method, `Up`/`Down` double and halve the sample count, `RMB` looks, `WASD` and `Q`/`E` move, scroll or `[`/`]` change focus distance, `-`/`=` change the f-number, `R` restarts accumulation, `Esc` quits.

`tools/compare_dof_methods.py` measures each method against the 512-sample ray-traced reference and writes amplified difference maps plus `output/dof3-comparison.json`. Measured mean absolute error on a 0..255 range, 64 samples each:

| Method | Overall | Sharp-silhouette halo | Near occluder | Defocused highlights |
|---|---|---|---|---|
| 1: gather | 4.946 | 2.925 | 12.641 | 4.014 |
| 2: multi-view | 0.682 | 0.558 | 1.364 | 0.626 |

The floor of the comparison is 0.518, measured by rendering the rasterized and ray-traced paths at f/22 where depth of field vanishes; it is nonzero because the rasterizer tessellates spheres the ray tracer intersects exactly.

## DoFScene Controls

- `RMB` drag: look around
- `WASD`: move forward/backward along the view direction and strafe sideways
- `Q` / `E`: move down / up
- `Shift`: faster movement
- `LMB`: click a visible surface to focus on its depth
- `F`: focus the center crosshair
- Mouse wheel: adjust focus distance
- `Shift` + mouse wheel: adjust lens focal length from 24 mm to 100 mm
- `[` / `]`: decrease / increase focus distance
- `-` / `+`: open / close aperture by changing f-number
- `1`: DoF view
- `2`: sharp color view
- `3`: depth view
- `4`: circle-of-confusion view
- `5`: split sharp/DoF view
- `0`: reset home view
- `H`: toggle help overlay
- `Tab`: toggle HUD
- `P`: capture `output/cafe-dof.png`
- `Esc`: quit

The HUD also exposes mode buttons plus focus and aperture minus/plus widgets.

## DoFScene CLI

```sh
./build/DoFScene.app/Contents/MacOS/DoFScene \
  --capture output/cafe-dof.png \
  --frames 30 \
  --mode 0 \
  --focus 4.0 \
  --f-number 1.2 \
  --lens 50 \
  --view home \
  --size 1120x760 \
  --verify
```

Supported options include `--capture path`, `--frames N`, `--mode 0..4`, `--focus meters`, `--f-number`, `--lens`, `--view home|close|wide`, `--size WIDTHxHEIGHT`, `--no-hud`, and `--verify`.

`--verify` runs a deterministic runtime smoke exercise. It checks the native render path across resize, camera movement, center and click autofocus, HUD mode/focus buttons, mouse capture, all five display modes, aperture changes, and reset behavior. It is a practical regression check for the demo, not an exhaustive physical input-device test.

## Expected Results

A window titled `DOF_Research` should open and display the current 3D cube scene through the active rendering path. The current project includes an interactive camera, depth testing, model/view/projection matrices, depth visualization modes, and an off-screen framebuffer presentation pass. The terminal prints framebuffer setup diagnostics such as the completed scene framebuffer size.

The cafe target opens a window titled `Cafe - Depth of Field`. It shows a procedural cafe tabletop with a foreground mug, central teapot focus subject, background furniture, shelves, plants, pendant lights, procedural material variation, shadow mapping, and a screen-space depth-of-field post-process. The current scene has 126 procedural objects and 8 mesh shapes, including smooth lathed teapot, cup, vase, and curved spout geometry. The effect is an illustrative real-time OpenGL DoF demonstration rather than a ray-traced or photoreal renderer.

Default camera/lens validation uses camera position `(0.25, 1.55, 3.8)`, aimed 0.27 m above the focus point, with focus distance 4.37 m, a 50 mm lens, and f/1.2 aperture. The app enforces a minimum CLI window size of 900x560 to keep HUD controls from overlapping.

Reference validation artifacts include `output/cafe-final.png` with the HUD, no-HUD captures `output/cafe-sharp.png`, `output/cafe-wide-aperture.png`, `output/cafe-narrow-aperture.png`, `output/cafe-near-focus.png`, `output/cafe-far-focus.png`, `output/cafe-depth.png`, and `output/cafe-coc.png`, plus `output/cafe-validation.json`.

Final validation also includes clean Clang static analysis for `src/dof_scene/main.cpp` and `src/dof_scene/scene.cpp`, invalid CLI rejection checks, and a live interactive GUI run with title `Cafe - Depth of Field | DOF | 4.37 m | f/1.2`.

Press Escape or close the window to exit.

## Development Workflow

The renderer opens as a separate native macOS GLFW window. VS Code cannot directly embed this running GLFW/OpenGL context, so the window is intentionally sized and positioned to sit beside the editor.

Press `P` while the render window is focused to save the current frame: `DepthResearch` writes `output/latest.png`, and `DoFScene` writes `output/cafe-dof.png`. Screenshots use the actual OpenGL framebuffer size, so Retina / HiDPI captures preserve the rendered resolution. Both images can be opened directly in VS Code.

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

`DepthResearch` loads shaders from the source-tree directory. `DoFScene` and `DoFApproaches` first load their copied shaders beside the executable, then fall back to the source tree. Rebuild the target after editing shaders to refresh its bundled copies.

If the triangle looks stretched or clipped after moving the window between displays, confirm the framebuffer resize callback is firing and that `glViewport()` receives framebuffer dimensions rather than logical window dimensions.

If `cmake -G Ninja` fails because Ninja is missing, activate the Conda environment first. `environment.yml` includes `ninja`.
