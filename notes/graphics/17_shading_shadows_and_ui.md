# Experiment 17: shading, shadows, higher-sample DoF, live UI

## What changed
- `basic.vert/frag`: every object has a world-space normal; diffuse (N.L) + ambient + PCF shadow map.
- `shadow.vert/frag` + shadow FBO: depth-only render from the directional light (orthographic, 2048^2).
- `screen.frag`: BasicDoF gathers `uCoCSampleCount` (default 100) Vogel-spiral taps instead of 16 fixed taps.
- `main.cpp`: Dear ImGui panel (Tab frees the cursor, G hides it); cubeF and cubeG add depth layers.
- `render_dof.py`: BOXES mirrors cubeF/cubeG so Cycles renders the same scene.

## Why
At a 120 px blur radius, 16 taps leave visible rings. Cycles samples the lens aperture with many
paths; the gather now approximates the same integral with more samples. Cycles produces shadows
automatically; the raster pass needed an explicit shadow map so both images share shadows.

## Known limits (still baseline)
Gather is not depth-aware, so sharp foreground/background silhouettes bleed. This is exactly what
the ray-traced comparison and the AI stage are meant to show and fix.
