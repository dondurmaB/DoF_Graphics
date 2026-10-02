# Experiment 01 — Vertex Attributes and Interpolation

## Goal

Understand how one vertex can carry multiple attributes, and how OpenGL interprets those attributes through a VAO. Observe that values produced by the vertex shader are interpolated across the primitive before reaching the fragment shader.

## Pipeline

CPU Vertex Data → VBO → VAO → Vertex Shader → Primitive Assembly → Rasterization → Fragment Shader → Framebuffer

## What Changed

- Vertex data now stores position and color together as `x, y, z, r, g, b`.
- The VBO receives interleaved vertex data: three position floats followed by three color floats.
- The VAO configures two attributes: location 0 for position and location 1 for color.
- The vertex shader passes each vertex color to the fragment shader.
- The fragment shader outputs the interpolated color it receives from rasterization.
- A primitive-mode control allows the same vertex data to be drawn as triangles, lines, or points.

## Key Concepts

### VBO

Stores raw vertex data in GPU-accessible memory. In this experiment, the VBO stores three vertices with position and color data interleaved.

### VAO

Stores the configuration describing how vertex attributes are interpreted. It remembers the enabled attributes, their sizes, strides, offsets, and source buffer bindings.

### Vertex Attribute

A named piece of per-vertex data consumed by the vertex shader. Here, location 0 is `aPos` and location 1 is `aColor`.

### Stride

The byte distance from the start of one vertex to the start of the next. Because each vertex has six floats, the stride is `6 * sizeof(float)`.

### Offset

The byte position where an attribute starts inside one vertex. Color starts after `x, y, z`, so its offset is `3 * sizeof(float)`.

### Rasterizer Interpolation

The vertex shader runs only at vertices, but the fragment shader runs for many fragments. Values passed from the vertex shader are interpolated across the triangle surface.

### Primitive Assembly

Controls how submitted vertices are grouped into primitives. `GL_TRIANGLES`, `GL_LINE_LOOP`, and `GL_POINTS` use the same vertex data but produce different primitives.

## Important API Calls

| Function | Purpose |
|---|---|
| `glBufferData()` | Upload vertex data to the currently bound buffer. |
| `glVertexAttribPointer()` | Describe the layout of one vertex attribute. |
| `glEnableVertexAttribArray()` | Enable a vertex attribute location for rendering. |
| `glDrawArrays()` | Draw vertices using the selected primitive mode. |
| `glGetShaderiv()` | Query shader compilation status. |
| `glGetShaderInfoLog()` | Retrieve GLSL compiler errors. |
| `glGetProgramiv()` | Query shader program link status. |
| `glGetProgramInfoLog()` | Retrieve shader linker errors. |

## Experiment Controls

| Control | Effect | Suggested Values |
|---|---|---|
| `triangleScale` | Changes triangle size by scaling CPU vertex positions before upload | `0.5`, `1.0`, `1.5`, `2.0` |
| `triangleOffsetX` | Moves the triangle left or right in clip-space coordinates | `-0.5`, `0`, `0.5`, `2.0` |
| `triangleOffsetY` | Moves the triangle down or up in clip-space coordinates | `-0.5`, `0.0`, `0.5` |
| Vertex colors | Changes the RGB value assigned to each triangle corner | RGB, all red, white, cyan/magenta/yellow |
| `primitiveMode` | Changes primitive assembly while using the same vertex buffer | `GL_TRIANGLES`, `GL_LINE_LOOP`, `GL_POINTS` |

## Observations

- Colors specified only at vertices become smooth gradients across a filled triangle.
- If all three vertex color controls contain the same RGB value, the triangle appears as one solid color.
- Color values are normally in the `0.0f` to `1.0f` range; larger values are clamped by the framebuffer output.
- Changing `GL_TRIANGLES` to `GL_LINE_LOOP` or `GL_POINTS` changes primitive assembly without changing the VBO data.
- The current code computes scale and offsets on the CPU before uploading the vertices; no matrix transform is used yet.

## Remember

- VBO = vertex data.
- VAO = vertex layout/configuration.
- One vertex can contain multiple attributes.
- `layout(location = N)` must match the VAO attribute index.
- Stride moves from one vertex to the next.
- Offset moves to a field inside one vertex.
- Rasterization interpolates vertex shader outputs.
