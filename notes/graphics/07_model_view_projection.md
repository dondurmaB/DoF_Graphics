# Experiment 07 — Model, View, and Projection

## Goal

Separate object placement, camera view, and perspective projection into explicit matrices. Observe how perspective makes farther objects appear smaller and how camera/projection controls affect clip-space output.

## Pipeline

Local/Object Space
→ Model Matrix
→ World Space
→ View Matrix
→ Camera/View Space
→ Projection Matrix
→ Clip Space
→ Perspective Divide
→ NDC
→ Viewport
→ Screen

## What Changed

- Replaced one combined `uTransform` uniform with `uModel`, `uView`, and `uProjection`.
- Rendered three cube instances from the same VAO/VBO/EBO mesh.
- Built a separate model matrix for each cube.
- Built a static camera view matrix with `glm::lookAt()`.
- Built a framebuffer-aspect perspective projection with `glm::perspective()`.
- Added controls for camera position, FOV, near/far planes, projection mode, and cube positions.

## Key Concepts

### Local/Object Space

The cube vertex data is centered around its own origin. This raw mesh data does not change when cubes move in the scene.

### Model Matrix

Transforms local/object-space vertices into world space. Each cube gets a different model matrix before drawing.

### World Space

The shared coordinate system where cube instances have different positions. The camera also exists in world space.

### View Matrix

Transforms world-space geometry into camera/view space. `glm::lookAt()` builds this from camera position, target, and up direction.

### Camera/View Space

Coordinates relative to the camera. After the view matrix, the scene is expressed from the camera's point of view.

### Projection Matrix

Transforms camera/view-space coordinates into clip space. It defines the visible volume used before clipping and perspective divide.

### Perspective Projection

Perspective projection makes farther objects appear smaller. It creates a clip-space `w` component used by the later perspective divide.

### Field of View

FOV controls how wide the camera view appears. Larger FOV values show more of the scene and make objects look smaller.

### Aspect Ratio

Aspect ratio must match the framebuffer width divided by framebuffer height. Otherwise the scene stretches when the window shape changes.

### Near and Far Planes

Near and far planes define the visible depth interval. Geometry outside this interval is clipped.

### Clip Space

The output space of the vertex shader before perspective division. Clipping happens against this space.

### Perspective Divide

After the vertex shader, OpenGL divides clip-space `x`, `y`, and `z` by `w`. This produces normalized device coordinates.

### Normalized Device Coordinates

NDC is the cube-shaped coordinate range after perspective divide. The viewport maps NDC to screen pixels.

## Important API Calls

| Function | Purpose |
|---|---|
| `glm::lookAt()` | Build the view matrix from camera position, target, and up vector. |
| `glm::perspective()` | Build a perspective projection matrix from FOV, aspect, near, and far. |
| `glm::radians()` | Convert degrees to radians for GLM matrix functions. |
| `glUniformMatrix4fv()` | Upload model, view, and projection matrices to the shader. |
| `glGetUniformLocation()` | Find MVP uniform locations once after shader linking. |
| `glfwGetFramebufferSize()` | Get current framebuffer dimensions for the projection aspect ratio. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `fieldOfViewDegrees` | Changes camera view width | `30.0f`, `45.0f`, `75.0f` |
| `nearPlane` | Clips geometry too close to the camera | `0.1f`, `1.0f`, `7.0f` |
| `farPlane` | Clips geometry too far from the camera | `8.0f`, `20.0f`, `100.0f` |
| `cameraPosition` | Moves the static camera | `glm::vec3(0.0f, 1.0f, 5.0f)`, `glm::vec3(1.5f, 1.0f, 5.0f)` |
| `cameraTarget` | Changes where the camera looks | `glm::vec3(0.0f, 0.0f, -2.0f)`, `glm::vec3(0.0f, 0.0f, -4.0f)` |
| `cameraUp` | Defines camera vertical direction | `glm::vec3(0.0f, 1.0f, 0.0f)` |
| `usePerspectiveProjection` | Enables perspective projection or identity projection | `true`, `false` |
| `cubeAPosition` / `cubeBPosition` / `cubeCPosition` | Moves each cube in world space | change X or Z values |
| `cubeAUniformScale` / `cubeBUniformScale` / `cubeCUniformScale` | Changes each cube size | `0.4f`, `0.7f`, `1.0f` |

## Observations

- Three cubes use the same vertex/index buffers but different model matrices.
- With equal scale, the farther cube appears smaller under perspective projection.
- Increasing FOV shows more of the scene and makes cubes appear smaller.
- Moving the camera changes the view without changing cube vertex data.
- Raising the near plane clips nearby geometry.
- Lowering the far plane clips distant geometry.
- Disabling perspective projection removes normal perspective size behavior.

## Remember

- Model transforms local → world.
- View transforms world → camera/view.
- Projection transforms camera/view → clip.
- Shader order is `P * V * M * vertex`.
- Actual vertex application order is Model → View → Projection.
- Perspective divide happens after the vertex shader.
- FOV controls view width.
- Near/far planes define the projection clipping range.
- Raw projected depth needs careful interpretation later.
