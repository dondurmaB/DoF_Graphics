# Experiment 14 — Basic Depth of Field

Review time: approximately 3–4 minutes.

## First Actual DoF Rendering

Experiment 13 estimated and visualized circle of confusion (CoC). Experiment 14 uses that value to blur scene color: this is the project's first actual depth-of-field rendering. Geometry around the focus distance stays relatively sharp, while nearer and farther geometry blurs.

The two-pass pipeline remains:

```text
Scene → FBO color + depth textures → screen shader
      → linear view depth → signed CoC → disk gather → displayed color
```

The Experiment 13 `linearizeDepth` and `calculateSignedCoCPixels` functions are byte-for-byte unchanged. The color, raw-depth, linear-depth, CoC-magnitude, and signed-CoC diagnostic modes remain available. `BasicDoF` is the new default mode.

## How the Blur Works

The screen shader reads one depth value at the output pixel and uses the magnitude of its CoC as the gather radius:

```text
radiusPixels = min(abs(cocPixels), max(maxBlurRadiusPixels, 0))
```

This deliberately retains Experiment 13's approximate CoC-to-radius convention. The sign still distinguishes near/far defocus in the diagnostic view, but the baseline blur uses only magnitude. The default maximum radius is **12 framebuffer pixels**; `cocVisualizationMaxPixels` affects only the diagnostic display.

If the radius is below **0.5 pixels**, the shader returns the original scene color immediately. This sharp-pixel early exit preserves near-focus detail and skips the neighboring color samples.

Otherwise, the shader performs a **17-tap gather/disk blur**: the center plus 16 neighbors on two staggered eight-sample rings, at approximately half and full radius. It averages all 17 color samples equally. A blurred pixel therefore requires 17 color texture samples and one depth sample; a sharp pixel needs only the center color and depth.

## Framebuffer Pixels → Texel Size

The CPU queries `glfwGetFramebufferSize()` each frame and passes the actual pixel width and height to the screen shader. These dimensions also drive FBO attachment resizing and the viewport, including on Retina/HiDPI displays.

```text
texelSize = 1 / max(vec2(framebufferWidth, framebufferHeight), vec2(1))
sampleUV = texCoord + diskOffset * radiusPixels * texelSize
```

Using both dimensions converts pixel offsets into texture coordinates without stretching the disk on a non-square framebuffer. CoC's sensor-to-pixel conversion continues to use framebuffer height.

Sample coordinates are clamped to a half-texel inset at each edge. The textures also use `GL_CLAMP_TO_EDGE`, preventing colors from wrapping from the opposite screen edge. Clamping can still repeat edge colors.

## Default Controls

These are source controls near the top of `src/main.cpp`; edit and rebuild to change them.

| Control | Default |
|---|---|
| `screenMode` | `ScreenMode::BasicDoF` |
| `focusDistanceMeters` | `10.0f` |
| `focalLengthMillimeters` | `50.0f` |
| `fNumber` | `1.4f` |
| `sensorHeightMillimeters` | `24.0f` |
| `maxBlurRadiusPixels` | `12.0f` |
| `cocVisualizationMaxPixels` | `20.0f` (diagnostics only) |
| `renderThroughFramebuffer` | `true` |

The camera starts at `(0, 0, 5)`, looking along negative Z. Cube centers span roughly 2, 5, 10, 15, and 30 meters of view depth, placing cube C near the default focus distance. WASD moves the camera, mouse movement changes view direction, `P` saves a screenshot, and Escape exits. Focus/aperture controls do not have runtime key bindings.

## Why This Is Only a Baseline

The gather never checks neighboring depths. It mixes foreground and background colors across silhouettes, so haloing and foreground/background bleeding are expected. A sharp foreground pixel may also remain too crisp against a blurred background. Sparse fixed taps can reveal the sample pattern at large radii, and the radius cap limits stronger defocus.

Only the nearest visible surface is stored in the screen-space textures; hidden surfaces cannot contribute correctly. Focal length and sensor size control CoC, while raster projection still uses an independent 45-degree FOV. This is an educational baseline, not a fully consistent physical camera model.

Later ray-traced DoF should improve occlusion correctness by tracing visibility from different aperture positions, allowing contributions that a single screen-space color/depth layer cannot represent.

## Verification

Build passed. Final visual verification was performed manually in the real application outside the sandbox; automated visual capture was blocked by macOS window-service restrictions. The manual result is authoritative. See the [archived verification report](../../experiments/graphics/14_basic_dof/README.md#verification-report) for the recorded checks and artifact caveat.
