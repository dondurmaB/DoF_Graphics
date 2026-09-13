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

## Expected Result

A window titled `DOF_Research` should open and display the current 3D cube scene through the active rendering path. The current project includes an interactive camera, depth testing, model/view/projection matrices, depth visualization modes, and an off-screen framebuffer presentation pass. The terminal prints framebuffer setup diagnostics such as the completed scene framebuffer size.

Press Escape or close the window to exit.

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
