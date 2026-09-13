# Experiment 03 — Uniforms and Animation

## Goal

Learn how CPU-side runtime values are sent to shaders with uniforms. This experiment contrasts per-vertex attributes with draw-call-wide uniforms and uses elapsed time to animate brightness and horizontal position.

## Pipeline

CPU Runtime Values  
→ Uniforms  
→ Vertex / Fragment Shader

CPU Vertex Data  
→ VBO + EBO  
→ Vertex Shader  
→ Rasterization  
→ Fragment Shader  
→ Framebuffer

## What Changed

- Added `uOffsetX` in the vertex shader to move all vertices horizontally.
- Added `uIntensity` in the fragment shader to scale interpolated RGB color.
- Retrieved uniform locations once after shader-program linking.
- Updated uniform values every frame before drawing.
- Used `glfwGetTime()` and `sin()` for smooth periodic animation.
- Preserved the indexed rectangle, VBO, VAO, EBO, and `glDrawElements()` path.

## Key Concepts

### Uniform

A uniform is a value supplied by the application to a shader program. It is shared by shader invocations for a draw call.

### Vertex Attribute vs Uniform

Vertex attributes can vary per vertex, such as position and color. Uniforms are shared values, such as one brightness or offset value used by every vertex or fragment.

### Uniform Location

`glGetUniformLocation()` asks a linked shader program where a named uniform lives. A location of `-1` can mean the uniform is missing or optimized away.

### glUniform

`glUniform1f()` uploads one float to the currently active shader program. The program should be active with `glUseProgram()` before setting uniforms.

### Time-Based Animation

`glfwGetTime()` returns elapsed application time. Time can drive values that change every frame.

### Sine Wave

`sin()` produces smooth periodic values. Scaling and offsetting the sine result controls animation range, speed, and amplitude.

## Important API Calls

| Function | Purpose |
|---|---|
| `glGetUniformLocation()` | Retrieve a named uniform location from a linked shader program. |
| `glUseProgram()` | Make the shader program active before setting uniforms and drawing. |
| `glUniform1f()` | Upload one float uniform to the active shader program. |
| `glfwGetTime()` | Read elapsed application time for animation. |
| `glDrawElements()` | Continue drawing the indexed rectangle through the EBO. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `animateIntensity` | Enables sine-based brightness animation | `true`, `false` |
| `animateHorizontalPosition` | Enables sine-based horizontal motion | `true`, `false` |
| `intensitySpeed` | Controls brightness animation speed | `0.5f`, `1.0f`, `3.0f` |
| `movementSpeed` | Controls horizontal motion speed | `0.5f`, `1.0f`, `2.0f` |
| `movementAmplitude` | Controls horizontal travel distance | `0.1f`, `0.35f`, `0.7f` |
| `staticIntensity` | Brightness used when intensity animation is off | `0.2f`, `0.5f`, `1.0f` |
| `staticOffsetX` | Horizontal offset used when motion animation is off | `-0.5f`, `0.0f`, `0.5f` |
| Corner colors | Per-vertex colors still interpolated across the rectangle | RGB/Y, all white, all red |

## Observations

- The same four vertex colors remain per-vertex attributes.
- `uIntensity` affects every fragment in the draw call equally.
- `uOffsetX` affects every vertex in the draw call equally.
- Brightness changes smoothly because intensity is recomputed from time each frame.
- Horizontal motion is smooth and periodic because offset comes from `sin(time * movementSpeed)`.
- Disabling animations makes `staticIntensity` and `staticOffsetX` control the result.

## Remember

- Attributes vary per vertex.
- Uniforms are shared for a draw call.
- Uniforms belong to a linked shader program.
- Query uniform locations once, not every frame.
- Call `glUseProgram()` before `glUniform1f()`.
- Time plus sine creates smooth animation.
- No matrices are needed for this simple uniform offset.
