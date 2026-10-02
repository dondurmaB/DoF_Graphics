# Experiment 13 — Physical Focus Distance and Circle of Confusion

## Purpose

Calculate and visualize Circle of Confusion values from the sampled scene depth texture. This estimates per-pixel blur radius for future Depth of Field work without applying blur yet.

## Run

From the repository root:

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/DepthResearch
```

## Related Notes

[Experiment Notes](../../../notes/graphics/13_coc_visualization.md)
