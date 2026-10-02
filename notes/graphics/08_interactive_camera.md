# Experiment 08 — Interactive Camera

## Goal

Replace the static camera with explicit first-person camera state. Use keyboard and mouse input to update camera position and direction, then build the view matrix from that state.

## Pipeline

Keyboard + Mouse
→ Camera State
→ Camera Position / Front
→ View Matrix
→ Vertex Shader
→ Projection
→ Rasterization
→ Depth Test
→ Framebuffer

## What Changed

- Added camera position, front direction, world-up reference, yaw, and pitch.
- Added delta-time based WASD movement.
- Added mouse-look using a GLFW cursor-position callback.
- Clamped pitch to avoid unstable vertical camera flips.
- Built the view matrix from `cameraPosition` and `cameraPosition + cameraFront`.
- Kept perspective projection, depth testing, screenshots, and shared cube geometry.
- Added more cube instances so movement through the scene is easier to see.

## Key Concepts

### Camera Position

The camera's location in world space. Moving with WASD changes this value.

### Camera Front

The normalized direction the camera is looking. Forward and backward movement use this vector.

### World Up

The global vertical reference. It is used with `cameraFront` to compute the camera right vector.

### Camera Right

The strafe direction, computed with `normalize(cross(cameraFront, worldUp))`. A/D movement moves along this axis.

### Camera Basis

The camera basis is built from front, right, and up directions. This experiment does not implement roll, so `worldUp` remains the view up reference.

### Yaw

Horizontal rotation angle. Mouse movement left/right changes yaw.

### Pitch

Vertical rotation angle. Mouse movement up/down changes pitch, clamped near +/-89 degrees.

### Mouse Look

Mouse offsets update yaw and pitch. The first mouse event initializes previous cursor coordinates to prevent a sudden jump.

### Delta Time

Time elapsed since the previous frame. Movement uses `movementSpeed * deltaTime`.

### Frame-Rate-Independent Movement

Using delta time keeps movement distance per second more stable when frame rate changes.

### View Matrix

Represents the world relative to the camera. `glm::lookAt(cameraPosition, cameraPosition + cameraFront, worldUp)` builds it from camera state.

## Important API / Math

| Function / Operation | Purpose |
|---|---|
| `glfwGetTime()` | Measure frame time for delta-time movement. |
| `glfwGetKey()` | Read WASD and Escape key state. |
| `glfwSetCursorPosCallback()` | Register mouse movement handling. |
| `glfwSetInputMode()` | Capture and hide the cursor for mouse-look. |
| `glm::cross()` | Build the camera right vector from front and world-up. |
| `glm::normalize()` | Keep direction vectors unit length. |
| `glm::lookAt()` | Build the view matrix from camera position and direction. |
| `glm::radians()` | Convert yaw/pitch degrees before trigonometry. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `cameraPosition` | Starting camera location | `glm::vec3(0.0f, 0.0f, 5.0f)`, `glm::vec3(0.0f, 1.0f, 8.0f)` |
| `cameraFront` | Starting look direction | `glm::vec3(0.0f, 0.0f, -1.0f)` |
| `worldUp` | Global up reference | `glm::vec3(0.0f, 1.0f, 0.0f)` |
| `yawDegrees` | Starting horizontal angle | `-90.0f`, `-45.0f`, `0.0f` |
| `pitchDegrees` | Starting vertical angle | `0.0f`, `20.0f`, `-20.0f` |
| `movementSpeed` | WASD movement speed | `1.0f`, `2.5f`, `6.0f` |
| `mouseSensitivity` | Mouse-look sensitivity | `0.05f`, `0.1f`, `0.25f` |
| `printCameraState` | Prints camera debug values occasionally | `false`, `true` |
| `fieldOfViewDegrees` | Projection view width | `30.0f`, `45.0f`, `75.0f` |
| `cubeAPosition` ... `cubeEPosition` | Cube world positions | Move X/Y/Z values |

## Observations

- W/S movement follows the current camera-front direction.
- A/D strafing follows the right vector from a cross product.
- Mouse movement changes orientation, not position.
- Moving the mouse upward increases pitch and looks upward.
- Pitch clamping prevents the camera from flipping at vertical extremes.
- The view matrix updates every frame from camera state.
- Depth testing still resolves overlapping cube faces correctly.

## Remember

- Camera position and camera direction are separate.
- `cameraPosition + cameraFront` gives the `lookAt()` target.
- Forward/back uses `cameraFront`.
- Strafe uses `normalize(cross(cameraFront, worldUp))`.
- Yaw controls left/right look.
- Pitch controls up/down look.
- Clamp pitch before +/-90 degrees.
- Delta time makes movement speed less frame-dependent.
