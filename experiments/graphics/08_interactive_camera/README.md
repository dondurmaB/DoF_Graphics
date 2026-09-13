# Experiment 08 — Interactive Camera

## Purpose

Replace the static camera with explicit first-person camera state. This experiment adds WASD movement, mouse-look, yaw/pitch, and delta-time movement while preserving the MVP and depth-testing pipeline.

## Run

From the repository root:

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/DepthResearch
```

## Related Notes

[Experiment Notes](../../../notes/graphics/08_interactive_camera.md)
