# Experiment 12 — Sampling and Linearizing the Depth Texture

## Goal

Use the depth texture produced by the off-screen scene pass as an input to the post-processing screen shader. Sample raw stored depth, optionally reconstruct linear view depth, and display it from the screen pass.

## Pipeline

3D Scene
→ Scene Pass
→ FBO
   ├── Color Texture
   └── Depth Texture
→ Screen Pass
→ Sample Color + Depth
→ Optional Depth Linearization
→ Default Framebuffer

## What Changed

- Added `ScreenMode` controls for color, raw sampled depth, and linear sampled depth.
- Bound `sceneColorTexture` to texture unit `0`.
- Bound `sceneDepthTexture` to texture unit `1`.
- Added `uSceneDepth` to the screen shader.
- Moved depth visualization from the scene shader into the post-processing screen shader.
- Linearized sampled depth using the same near/far planes as the projection.
- Preserved FBO resizing, Retina framebuffer handling, scene depth testing, and the full-screen quad pass.

## Key Concepts

### Depth Texture

A depth texture stores the depth values written during the scene pass. It is attached to the FBO as `GL_DEPTH_ATTACHMENT`.

### Texture Sampling

The screen shader reads texture values with `texture(sampler, uv)`. The color texture returns RGB scene color; the depth texture returns a single depth value in `.r`.

### Texture Unit

A texture unit is a binding slot used by shaders. Texture objects are bound to texture units before drawing.

### Sampler2D

A `sampler2D` uniform stores the texture-unit index the shader should read from. `uSceneColor` uses unit `0`; `uSceneDepth` uses unit `1`.

### Multiple Input Textures

The screen pass can sample both color and depth at the same UV coordinate. This is the basic input structure needed for later depth-based post-processing.

### Raw Sampled Depth

`texture(uSceneDepth, texCoord).r` retrieves the raw perspective depth stored by the scene pass. It is normally nonlinear and close to `[0, 1]`.

### Linear Sampled Depth

Sampled raw depth can be reconstructed into positive view-depth using the projection near and far planes. This is more useful for depth comparisons and later Depth of Field logic.

### gl_FragCoord.z vs Sampled Depth Texture

`gl_FragCoord.z` is the depth of the fragment currently being processed. `texture(uSceneDepth, texCoord).r` reads a depth value stored during an earlier render pass. They are related raw depth quantities, but they are accessed at different stages.

### Post-Processing

The screen shader is now the natural place to inspect and process rendered textures. Future effects can use scene color and scene depth together.

## Important API / Shader Calls

| Item | Purpose |
|---|---|
| `glActiveTexture()` | Selects the texture unit to bind next. |
| `glBindTexture()` | Binds a texture object to the active texture unit. |
| `sampler2D` | Lets GLSL sample a 2D texture bound to a texture unit. |
| `texture()` | Reads color or depth from a texture at UV coordinates. |
| `uSceneColor` | Samples the scene color texture from texture unit `0`. |
| `uSceneDepth` | Samples the scene depth texture from texture unit `1`. |
| `uNearPlane` | Must match the projection near plane for depth reconstruction. |
| `uFarPlane` | Must match the projection far plane for depth reconstruction. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `screenMode` | Chooses color, raw depth, or linear depth in the screen pass | `Color`, `RawDepth`, `LinearDepth` |
| `renderThroughFramebuffer` | Uses scene → FBO → screen shader → window | `true`, `false` |
| `nearPlane` | Projection and reconstruction near plane | `0.1f`, `0.5f`, `1.0f` |
| `farPlane` | Projection and reconstruction far plane | `30.0f`, `50.0f`, `100.0f` |
| `depthVisualizationMax` | Display-only normalization for linear depth | `10.0f`, `25.0f`, `50.0f` |

## Observations

- Color mode should match the Experiment 11 FBO presentation path.
- Raw depth mode reads the FBO's depth texture and shows nonlinear perspective depth.
- Linear depth mode reconstructs view-depth from the sampled depth texture.
- Moving the camera changes both the color texture and the stored depth texture.
- Changing `depthVisualizationMax` affects only linear-depth display brightness.

## Remember

- Pass 1 writes color and depth into FBO textures.
- Pass 2 samples both textures in the screen shader.
- Texture unit `0` is color.
- Texture unit `1` is depth.
- Raw depth is still nonlinear.
- Linearization requires matching projection near/far values.
- This enables future Depth of Field, but no blur is implemented yet.
