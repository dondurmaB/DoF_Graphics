# Experiment 02 — Indexed Drawing with an EBO

## Goal

Learn how multiple triangles can form one surface while reusing shared vertices. This experiment separates vertex data from topology using an index array and an Element Buffer Object.

## Pipeline

CPU Vertex Data  
→ VBO  
→ VAO  
→ EBO / Indices  
→ Vertex Shader  
→ Primitive Assembly  
→ Rasterization  
→ Fragment Shader  
→ Framebuffer

## What Changed

- Replaced one triangle with a rectangle made from two triangles.
- Stored four unique vertices, each with position and color.
- Added six indices to describe the two triangles.
- Added an EBO and uploaded the index array to `GL_ELEMENT_ARRAY_BUFFER`.
- Replaced `glDrawArrays()` with `glDrawElements()`.
- Added `wireframeMode` to reveal the two-triangle structure.

## Key Concepts

### Unique Vertices

The rectangle stores each corner once. Shared corners do not need duplicate position/color data.

### Index

An index is an integer that refers to one vertex in the vertex array. Groups of indices define which vertices form each triangle.

### EBO

An Element Buffer Object stores indices in GPU-accessible memory. It is bound with `GL_ELEMENT_ARRAY_BUFFER`.

### Vertex Reuse

The same vertex can be referenced by more than one triangle. In the default rectangle, vertices 1 and 3 are reused.

### Topology

Vertex data describes points and attributes. Indices describe how those points are connected into primitives.

### VAO and EBO State

The currently bound `GL_ELEMENT_ARRAY_BUFFER` is stored inside the VAO. Bind the EBO while the VAO is active.

### glDrawElements

Draws primitives by reading indices from the currently bound EBO instead of consuming vertices sequentially.

## Important API Calls

| Function | Purpose |
|---|---|
| `glGenBuffers()` | Create buffer object names for VBO/EBO storage. |
| `glBindBuffer()` | Bind either vertex data or index data to a buffer target. |
| `glBufferData()` | Upload vertex or index data to the currently bound buffer. |
| `glDrawElements()` | Draw primitives using indices from the bound EBO. |
| `glPolygonMode()` | Switch between filled and wireframe polygon rendering. |

## Experiment Controls

| Control | Effect | Suggested Values |
|---|---|---|
| `rectangleScale` | Changes rectangle size before vertex upload | `0.5`, `1.0`, `1.5` |
| `rectangleOffsetX` | Moves the rectangle left or right | `-0.5`, `0.0`, `0.5` |
| `rectangleOffsetY` | Moves the rectangle down or up | `-0.5`, `0.0`, `0.5` |
| Corner colors | Changes color interpolation across the two triangles | RGB/Y, all white, all red, cyan/magenta/yellow |
| `indices` | Changes how corners connect into triangles | `0, 1, 3, 1, 2, 3` or `0, 1, 2, 0, 2, 3` |
| `wireframeMode` | Shows the underlying triangle edges | `false`, `true` |

## Observations

- Four unique vertices form a complete rectangle through two triangles.
- Shared corners do not need duplicate vertex data.
- Wireframe mode exposes the diagonal boundary between the two triangles.
- Changing indices changes connectivity without changing vertex positions.
- Vertex data and topology are separate concepts.

## Remember

- VBO = vertex attributes.
- EBO = vertex indices.
- VAO remembers the EBO binding.
- Indices define topology.
- `glDrawElements()` uses indices.
- Four rectangle corners can produce two triangles.
- Wireframe mode reveals primitive structure.
