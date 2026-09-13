# Experiment 05 — First 3D Cube

## Goal

Build the first 3D object using the existing vertex attribute, EBO, and matrix-transform pipeline. Observe that a cube can be made from triangles, and that visibility is incorrect without depth testing.

## Pipeline

CPU Cube Vertices + Indices → VBO + EBO → VAO → Vertex Shader + `uTransform` → Rasterization → Fragment Shader → Framebuffer

## What Changed

- Replaced the transformed 2D rectangle with a cube centered at the local origin.
- Stored 24 face vertices so each cube face can have its own solid color.
- Used 36 indices for 12 triangles: 6 faces × 2 triangles per face.
- Applied a 3D model matrix with scale, X/Y/Z rotations, and translation.
- Added controls for static rotation, animated rotation, uniform scale, and wireframe display.
- Left depth testing disabled intentionally.

## Key Concepts

### Cube Mesh

A cube is not a built-in OpenGL primitive. It is represented as triangles, with each face made from two triangles.

### Face Vertices

This experiment uses 24 vertices instead of 8 shared cube corners. Duplicating face vertices makes it simple to assign one color per face.

### EBO

The EBO stores the index order used to assemble triangles from vertex data. Here it stores 36 indices for the cube.

### Model Matrix

The model matrix moves vertices from local object space into transformed clip-space input for this simple experiment. The shader still applies `uTransform * vec4(aPos, 1.0)`.

### Transform Order

The code builds `T * Rz * Ry * Rx * S`. With column-vector math, vertices experience scale first, then X rotation, Y rotation, Z rotation, and translation.

### No Depth Testing

Depth testing is disabled, so triangles drawn later can overwrite earlier triangles even when they should appear farther away. This is expected for Experiment 05.

### Wireframe Mode

`glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)` shows the triangle edges used to build the cube faces.

## Important API Calls

| Function | Purpose |
|---|---|
| `glBufferData()` | Upload cube vertex data and index data to GPU buffers. |
| `glVertexAttribPointer()` | Describe the position and color layout inside each vertex. |
| `glDrawElements()` | Draw triangles using indices from the EBO. |
| `glUniformMatrix4fv()` | Upload the model transform matrix to the vertex shader. |
| `glPolygonMode()` | Switch between filled faces and wireframe rendering. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `uniformScale` | Changes cube size uniformly | `0.4f`, `0.7f`, `1.0f` |
| `rotationXDegrees` | Rotates the cube around the X axis | `0.0f`, `25.0f`, `60.0f` |
| `rotationYDegrees` | Rotates the cube around the Y axis | `0.0f`, `35.0f`, `60.0f` |
| `rotationZDegrees` | Rotates the cube around the Z axis | `0.0f`, `30.0f`, `45.0f` |
| `animateRotation` | Enables time-based rotation | `false`, `true` |
| `rotationSpeedX` | Controls animated X rotation speed | `0.0f`, `20.0f`, `60.0f` |
| `rotationSpeedY` | Controls animated Y rotation speed | `0.0f`, `35.0f`, `90.0f` |
| `rotationSpeedZ` | Controls animated Z rotation speed | `0.0f`, `20.0f`, `60.0f` |
| `wireframeMode` | Shows filled faces or triangle edges | `false`, `true` |

## Observations

- A cube is visible as colored faces, but only because triangles are drawn in a particular order.
- Without depth testing, some farther faces can appear in front of nearer faces.
- Wireframe mode reveals 12 triangles: two triangles per cube face.
- X, Y, and Z rotations affect the cube around different axes.
- Combined rotations make the object read as 3D even without projection or a camera.

## Remember

- A cube mesh is built from triangles.
- 6 faces × 2 triangles = 12 triangles.
- 12 triangles × 3 indices = 36 indices.
- Duplicated face vertices make per-face color simple.
- `T * Rz * Ry * Rx * S` applies scale first to the vertex.
- No depth test means draw order controls visibility.
- Wireframe mode is useful for inspecting mesh topology.
