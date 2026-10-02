# Experiment 05 — First 3D Cube

## Purpose

Build the first 3D cube from indexed triangles while preserving the existing shader, color attribute, and matrix-transform pipeline. Depth testing is intentionally left disabled so draw-order visibility problems are easy to observe.

## Run

From the repository root:

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/DepthResearch
```

## Related Notes

[Experiment Notes](../../../notes/graphics/05_first_3d_cube.md)
