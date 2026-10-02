# Experiment 11 — Off-Screen Framebuffers and Post-Processing

## Purpose

Render the 3D scene into an off-screen framebuffer with a color texture and a depth texture, then present the color texture to the window through a full-screen quad.

## Run

From the repository root:

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/DepthResearch
```

## Related Notes

[Experiment Notes](../../../notes/graphics/11_framebuffers.md)
