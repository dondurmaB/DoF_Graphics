# Experiment 14 — Basic Depth of Field

## Purpose

Archive the first actual DoF rendering: a screen-space, equal-weight 17-tap disk gather driven by the magnitude of Experiment 13's unchanged signed CoC calculation. Near-focus pixels bypass the gather below a 0.5-pixel radius; defocused pixels use a radius capped at 12 framebuffer pixels.

## Snapshot

The following files are exact copies of the final Experiment 14 working versions:

- `src/main.cpp`
- `shaders/basic.vert`
- `shaders/basic.frag`
- `shaders/screen.vert`
- `shaders/screen.frag`

Previous experiment archives are unchanged. No temporary verification harness is included.

## Build and Controls

At the Experiment 14 commit, build and run the active project from the repository root:

```sh
cmake -S . -B build -G Ninja
cmake --build build
./build/DepthResearch
```

The archive is a source snapshot, not a separate CMake target. The root build uses the active `src/` and `shaders/` directories.

Defaults: `BasicDoF`, focus distance 10 m, focal length 50 mm, f/1.4, sensor height 24 mm, maximum blur radius 12 pixels, and CoC diagnostic scale 20 pixels. DoF settings are source controls in `src/main.cpp`; edit and rebuild to compare focus distances or f/1.4 with f/8. WASD and mouse control the camera; `P` captures a frame; Escape exits.

## Verification Report

The user manually runtime-verified BasicDoF in the real application outside the sandbox. That manual verification is authoritative. **Automated visual capture was blocked by macOS window-service restrictions, so final visual verification was performed manually.** No automated screenshot comparison or sandbox visual verification is claimed.

| Check | Result and evidence |
|---|---|
| Build | Passed previously; completion check `cmake --build build` also passed with `ninja: no work to do.` |
| FBO | Completed in the manually verified runtime; the scene path requires `GL_FRAMEBUFFER_COMPLETE` before rendering. |
| BasicDoF | Manually runtime-verified outside the sandbox, as reported by the user. |
| Focused geometry | Remains relatively sharp in the manual verification. |
| Out-of-focus geometry | Visibly blurs in the manual verification. |
| Focus distance | Changing it moves the focused region in the manual verification. |
| Aperture | f/1.4 produces more blur than f/8 in the manual verification. |
| Window resizing | Works in the manual verification; source review confirms attachment resizing uses the current framebuffer dimensions. |
| Pixel dimensions | Source review confirms `glfwGetFramebufferSize()` supplies actual framebuffer pixel dimensions for CoC conversion, blur texel size, and viewport sizing. |
| Screen edges | No screen-edge wrapping in the manual verification; source review confirms half-texel UV clamping and `GL_CLAMP_TO_EDGE`. |
| Experiment 13 CoC | Depth linearization and signed CoC functions compared byte-for-byte with the Experiment 13 archive: unchanged. |

Specific artifact observations were not separately supplied with the manual verification. Haloing and foreground/background bleeding are expected limitations, not claimed automated observations. Other limitations include sparse-tap patterns, repeated edge colors from clamping, the radius cap, and unavailable hidden-surface information. Neighboring sample depths are not checked. Later ray-traced DoF should improve occlusion correctness through aperture-dependent visibility.

## Related Notes

[Experiment 14 notes](../../../notes/graphics/14_basic_dof.md)
