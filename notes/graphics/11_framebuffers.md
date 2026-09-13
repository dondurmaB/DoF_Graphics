# Experiment 11 — Off-Screen Framebuffers and Post-Processing

## Goal

Render the 3D scene into an off-screen framebuffer with a color texture and a depth texture, then present the color texture to the window with a full-screen quad.

## Pipeline

3D Scene
→ Scene Shader
→ Off-Screen FBO
   ├── Color Texture
   └── Depth Texture
→ Full-Screen Quad
→ Screen Shader
→ Default Framebuffer
→ Display

## What Changed

- Added `renderThroughFramebuffer` to switch between direct rendering and the FBO path.
- Created an off-screen scene framebuffer object.
- Attached a scene color texture to `GL_COLOR_ATTACHMENT0`.
- Attached a scene depth texture to `GL_DEPTH_ATTACHMENT`.
- Checked framebuffer completeness after attachment setup and printed the complete framebuffer size.
- Added `screen.vert` and `screen.frag` for the presentation pass.
- Added a separate full-screen quad VAO/VBO.
- Resized off-screen textures when the actual framebuffer size changes.
- Kept the existing camera, MVP matrices, cube mesh, depth testing, and screenshot workflow.

## Key Concepts

### Framebuffer Object

An FBO is an alternative OpenGL rendering destination. Binding it makes subsequent rendering write into its attachments instead of the window.

### Default Framebuffer

Framebuffer `0` is the default/window framebuffer. Binding `0` returns rendering to the visible window.

### Color Attachment

The color attachment receives fragment color output from the scene pass.

### Depth Attachment

The depth attachment receives per-fragment depth values used by depth testing.

### Color Texture

The scene color texture stores RGB output from the off-screen scene pass. The screen pass samples it and writes it to the window.

### Depth Texture

The scene depth texture stores depth output. It is not sampled yet, but later experiments can use it for post-processing and Depth of Field.

### Off-Screen Rendering

The scene is rendered into textures first, before anything is shown in the window.

### Full-Screen Quad

A pair of triangles that covers clip/NDC space from `-1` to `1`. It does not need model, view, or projection matrices.

### Texture Sampling

The screen fragment shader uses `texture(uSceneColor, texCoord)` to read the off-screen color texture.

### Multi-Pass Rendering

Rendering now has two passes: scene rendering into the FBO, then screen presentation to the default framebuffer.

### Post-Processing

Post-processing separates scene rendering from final image presentation. This architecture makes later blur and Depth of Field passes possible.

## Important API Calls

| Function | Purpose |
|---|---|
| `glGenFramebuffers()` | Create a custom framebuffer object. |
| `glBindFramebuffer()` | Switch between the custom FBO and framebuffer `0`. |
| `glFramebufferTexture2D()` | Attach color and depth textures to the FBO. |
| `glCheckFramebufferStatus()` | Verify that the FBO is complete before rendering to it. |
| `glTexImage2D()` | Allocate color and depth texture storage at framebuffer size. |
| `glActiveTexture()` | Select texture unit `0` for screen sampling. |
| `glBindTexture()` | Bind the scene color texture for sampling. |
| `texture()` GLSL | Sample the scene color texture in the screen shader. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `renderThroughFramebuffer` | Uses scene → FBO → screen quad → window when true | `true`, `false` |
| `visualizationMode` | Controls the scene shader output before it reaches the FBO | `Color`, `RawDepth`, `LinearDepth` |
| `fieldOfViewDegrees` | Changes the scene projection | `30.0f`, `45.0f`, `75.0f` |
| `nearPlane` / `farPlane` | Define projection and depth reconstruction values | `0.1f/100.0f`, `1.0f/30.0f` |

## Observations

- With `renderThroughFramebuffer = true`, the visible image should match direct rendering.
- The scene pass needs depth testing because cube fragments overlap in 3D.
- The screen pass disables depth testing because it only draws one finished image-sized quad.
- Resizing the window reallocates the off-screen textures to match the actual framebuffer size.
- The depth texture exists even though the screen shader does not sample it yet.

## Remember

- Framebuffer `0` is the visible default framebuffer.
- A custom FBO allows rendering off-screen.
- Color and depth output can be stored in textures.
- Texture attachments must match framebuffer dimensions.
- Framebuffer completeness should be checked.
- A full-screen quad lets a shader process or present a rendered image.
- This experiment builds the post-processing structure but does not add blur or Depth of Field yet.
