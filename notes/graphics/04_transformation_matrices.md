# Experiment 04 — Transformation Matrices

## Goal

Learn how fixed model-space geometry can be translated, rotated, and scaled using a 4x4 matrix. The matrix is built on the CPU with GLM and sent to the vertex shader as a uniform.

## Pipeline

Model-space Vertex  
→ Transformation Matrix  
→ Vertex Shader  
→ Transformed Clip-space Position  
→ Rasterization  
→ Fragment Shader  
→ Framebuffer

## What Changed

- Rectangle vertex positions became fixed model-space coordinates.
- Removed uniform-based horizontal offset from Experiment 03.
- Added a `mat4 uTransform` uniform in the vertex shader.
- Built a `glm::mat4` every frame from translation, rotation, and scale controls.
- Uploaded the matrix with `glUniformMatrix4fv()`.
- Added a transform-order control to compare matrix composition order.

## Key Concepts

### Identity Matrix

The identity matrix leaves geometry unchanged. It is the starting point for composing transforms.

### Translation

Moves geometry by adding position offset through a matrix. With homogeneous coordinates, translation works because positions use `w = 1`.

### Rotation

Rotates geometry around an axis. The rectangle is centered at the local origin, so Z-axis rotation spins it around its center.

### Scaling

Changes size along each axis. Non-uniform scale uses different values for X and Y.

### Homogeneous Coordinates

4D coordinates allow translation to be represented in a 4x4 matrix. Points use `w = 1`; directions would use `w = 0`.

### Matrix Composition

Multiple transforms can be combined into one matrix. The vertex shader applies the final composed matrix to every vertex.

### Transformation Order

Matrix multiplication is not commutative. Scale-rotate-translate and translate-rotate-scale can produce visibly different results.

### Model Space

The VBO stores base object geometry before transformation. The model transform moves that object into its final position.

### Matrix Uniform

A matrix uniform is one shared transform value sent from the CPU to the shader program. It applies to all vertices in the draw call.

## Important API Calls

| Function | Purpose |
|---|---|
| `glm::mat4(1.0f)` | Create an identity 4x4 matrix. |
| `glm::translate()` | Compose a translation transform. |
| `glm::rotate()` | Compose a rotation transform. |
| `glm::scale()` | Compose a scaling transform. |
| `glm::radians()` | Convert degrees to radians for GLM rotation. |
| `glm::value_ptr()` | Provide raw matrix data to OpenGL. |
| `glGetUniformLocation()` | Retrieve the `uTransform` uniform location once. |
| `glUniformMatrix4fv()` | Upload the matrix to the active shader program. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `translationX` | Moves the rectangle horizontally | `-0.5f`, `0.0f`, `0.5f` |
| `translationY` | Moves the rectangle vertically | `-0.5f`, `0.0f`, `0.5f` |
| `scaleX` | Scales width | `0.5f`, `1.0f`, `1.5f` |
| `scaleY` | Scales height | `0.5f`, `1.0f`, `1.5f` |
| `rotationDegrees` | Static rotation when animation is off | `0.0f`, `30.0f`, `45.0f` |
| `animateRotation` | Enables time-based rotation | `false`, `true` |
| `rotationSpeed` | Controls animated rotation speed in degrees/second | `30.0f`, `45.0f`, `90.0f` |
| `transformOrder` | Changes transform composition order | `ScaleRotateTranslate`, `TranslateRotateScale` |
| `animateIntensity` | Preserved brightness animation from Experiment 03 | `false`, `true` |
| `staticIntensity` | Preserved brightness value when intensity animation is off | `0.5f`, `1.0f`, `1.5f` |

## Observations

- Geometry stays fixed in the VBO; transforms change where it appears.
- Translation, rotation, and scale can be combined into one matrix.
- Rotation occurs around the rectangle center because the local origin is at its center.
- Changing transform order changes the final position/shape.
- The vertex shader transforms every vertex with the same `uTransform`.
- The fragment shader color intensity uniform still works, but it is not the focus here.

## Remember

- VBO geometry can stay fixed.
- Model transforms move object geometry mathematically.
- 4x4 matrices support translation with `w = 1`.
- Matrix order matters.
- Uniform matrices are shared by the draw call.
- `glUseProgram()` before `glUniformMatrix4fv()`.
- Camera/view/projection matrices are later concepts.
