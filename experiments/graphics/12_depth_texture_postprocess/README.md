# Experiment 12 — Sampling and Linearizing the Depth Texture

## Purpose

Sample the scene color texture and scene depth texture in the screen/post-processing shader. Display normal color, raw sampled depth, or reconstructed linear depth without adding blur or Depth of Field yet.

## Run

From the repository root:

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/DepthResearch
```

## Related Notes

[Experiment Notes](../../../notes/graphics/12_depth_texture_postprocess.md)
