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

## Start Here

[`notes/EXPLAINER.md`](notes/EXPLAINER.md) explains the whole project in one read: what depth of
field is physically, how the OpenGL gather and the Cycles path tracer each produce it, why the two
can be compared, and everything that changed in experiment 18. Object sizes and depths are in
[`notes/graphics/18_scene_inventory.md`](notes/graphics/18_scene_inventory.md).

## The Scene

The environment is a dusk back-alley described once in `scene/alley.scene` and loaded by **both**
renderers: by `src/SceneFile.cpp` for the OpenGL pass and by `tools/scene/scene_loader.py` for the
Cycles reference in `tools/raytraced_reference/render_dof.py`. Neither one authors geometry, so the
two cannot drift apart and end up comparing different scenes.

There are no image textures. Every surface detail is real geometry (mortar courses, protruding
bricks, crate panels, railings), which keeps the two renderers matching exactly and makes every
shadow a real shadow.

```sh
python3 tools/scene/build_alley.py      # regenerate scene/alley.scene
python3 tools/scene/preview_scene.py    # CPU preview PNG; needs no GPU
```

Single-line tweaks can be made directly in `scene/alley.scene` and reloaded in the running
renderer with `L`. Larger changes belong in `build_alley.py`, which regenerates the file
deterministically. After regenerating, refresh the cross-language test numbers:

```sh
python3 tests/test_scene_file.py --print-expected
```

## Expected Result

A window titled `DOF_Research` should open and display the alley through the active rendering path.
The project includes an interactive camera, depth testing, model/view/projection matrices, depth and
circle-of-confusion visualization modes, a shadow map, an HDR off-screen framebuffer, and a
defocus-gather presentation pass. The terminal prints setup diagnostics: the loaded scene's
primitive/triangle counts and bounds, the sun and sky values, the shadow-map resolution in
centimetres per texel, and the scene framebuffer size.

Press Escape or close the window to exit.

## Controls

The cursor is free on launch so the panel is immediately clickable. The button at the top of the
panel switches between "pointer for the panel" and "camera follows the mouse"; `Tab` does the same
from the keyboard. While the camera has the mouse the panel cannot be clicked, so `Tab` is the way
back, and the panel says so in that mode. Holding the right mouse button is still available as a
quick look-around without toggling.

| Key | Action |
|---|---|
| `1`-`6` | Color, raw depth, linear depth, CoC magnitude, CoC signed, basic DoF |
| `0` | Split view: sharp on the left of the divider, defocused on the right |
| `7` `8` `9` | Focus 2 m / 5 m / 15 m |
| `F` `B` | Toggle f/1.4 and f/8 / set f/2.8 |
| `T` | Reference preset: 50 mm f/1.4 focused 5 m, matching the Cycles jobs |
| `K` | Strong-DoF preset: 85 mm f/1.4 focused 1.6 m, an unmistakable blur |
| `L` | Reload `scene/alley.scene` |
| `R` | Reset the camera to the scene file's pose |
| `[` `]` | Focal length -/+5 mm |
| `,` `.` | Sensor height -/+2 mm |
| `V` | Physical / legacy projection |
| `WASD` | Move |
| `G` | Cycle the panel: compact -> full -> hidden |
| `Tab` | Free / capture the mouse |
| `P` | Screenshot: `output/latest.png` plus a settings-named copy such as `output/gl_focus5m_f1.4.png` |
| `H` | Print the key list |

The panel reports the circle-of-confusion radius the current lens settings actually reach next to
the configured ceiling. That readout matters: at 50 mm f/1.4 focused 5 m the background CoC radius
is only about 9 px, so the 120 px ceiling does nothing and the panel says "lens-limited". Press `K`
for settings that produce a large blur. See
[notes/graphics/18_alley_scene_and_ui.md](notes/graphics/18_alley_scene_and_ui.md) for the numbers.

## Comparing the Two Renderers

```sh
# 1. Cycles references -> reports/raytraced_dof/rt_*.png (+ sidecar JSON)
/Applications/Blender.app/Contents/MacOS/Blender --background --python-exit-code 1 \
  --python tools/raytraced_reference/render_dof.py

# 2. OpenGL captures: run the renderer, press T then 6, then P. Repeat for F and B.
#    Writes output/gl_focus5m_f1.4.png, matching the reference's name.

# 3. Pair, check and measure
python3 tools/compare/compare_renders.py --write-images
```

Step 3 writes side-by-side and 4x difference images plus a metrics table to
`reports/comparison/`. It refuses to compare a reference whose recorded scene sha256 does not
match the scene file on disk, so a stale reference cannot silently produce plausible numbers.

## Tests

```sh
cmake --build build && ctest --test-dir build
```

`scene_file` and `scene_file_python` are two halves of one check: the C++ and Python scene loaders
are compared against the same expected geometry numbers, so changing one without the other fails
with the exact number that moved.

## Development Workflow

The renderer opens as a separate native macOS GLFW window. VS Code cannot directly embed this running GLFW/OpenGL context, so the window is intentionally sized and positioned to sit beside the editor.

Press `P` while the render window is focused to save the current frame to `output/latest.png`. The screenshot uses the actual OpenGL framebuffer size, so Retina / HiDPI screenshots preserve the real rendered resolution. You can open `output/latest.png` directly inside VS Code for inspection.

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
