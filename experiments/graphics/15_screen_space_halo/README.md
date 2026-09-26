# Experiment 15 — Where Screen-Space Depth of Field Breaks

## Purpose

Experiment 14 produced the project's first depth-of-field blur and listed haloing and foreground bleeding as expected limitations. This experiment stops treating them as expected and makes them measurable: a scene built specifically to expose every failure mode of a single-layer screen-space gather, rendered against a reference that has none of them.

The claim being tested is narrow and falsifiable. A single-layer gather has **one** color image and **one** depth image, both rendered from a pinhole at the lens centre, and it chooses its blur radius from the depth of the pixel it is writing. Neighbour depths are never consulted. Two errors follow, and neither can be removed by taking more samples.

## Not a Source Snapshot

Experiments 01-14 archive a copy of the source because the active `src/` moved on. Experiments 15-17 do not. All three methods are one program, `DoFApproaches`, rendering one scene through three different visibility solves; copying it three times would only let the three copies drift apart and destroy the comparison. The code is:

- `src/dof_approaches/scene.cpp` — the shared scene, as analytic primitives
- `shaders/dof_approaches/postprocess.frag` — this experiment's method
- `shaders/dof_approaches/shading.glsl` — surface response shared by all three methods

## The Scene

Four failure modes are staged side by side so one frame exposes all of them. Every placement in `createHaloScene()` is expressed as a depth from the eye, because depth is what the circle of confusion depends on.

| Zone | Placement | What it tests |
|---|---|---|
| A — railing | Depth 3.80 m, **exactly** the focus distance, against a bright checkered wall 37.6 m back | The railing's circle of confusion is exactly zero, so no correct method may spread its colour by even one pixel |
| B — near posts | Depth 1.05 m, 27 px wide, defocus diameter 63 px | A defocus disc wider than the occluder is the condition under which a real aperture sees straight through it |
| C — bokeh lights | Emissive spheres far smaller than their own defocus discs | Where the energy of a defocused highlight goes |
| D — arcade | Columns on the lines x = ±0.22·depth, plus spheres, blocks and wires from 4 m to 31 m | A continuous depth ramp, so the sharp-to-blurred transition reads as a gradient |

The lens is 50 mm on a 24 mm sensor at f/1.0, focused at 3.80 m. That gives a 27° vertical field of view; object sizes are chosen against the resulting visible half-height of 0.24·d, which is why a telephoto DoF scene has to be laid out deliberately rather than filled with objects.

## Run It

```sh
cmake -S . -B build
cmake --build build --target DoFApproaches

BIN=./build/DoFApproaches.app/Contents/MacOS/DoFApproaches

# This experiment: the single-layer gather.
$BIN --method 1 --samples 64 --size 1280x860 --capture output/dof3-m1-gather.png

# A pinhole rendering of the same scene, for reference.
$BIN --method 1 --f-number 22 --size 1280x860 --capture output/dof3-reference-pinhole.png
```

Interactively, `1`/`2`/`3` switch method, `Up`/`Down` halve or double the sample count, `RMB` looks, `WASD`/`QE` move, `[`/`]` and the scroll wheel change focus distance, `-`/`=` change the f-number, and `R` restarts accumulation.

## Findings

Measured against `--method 3 --samples 512`, mean absolute error on a 0..255 range (`tools/compare_dof_methods.py`):

| Method | Overall | Zone A (halo) | Zone B (near occluder) | Defocused highlights |
|---|---|---|---|---|
| 1: gather, 64 taps | **4.946** | **2.925** | **12.641** | **4.014** |
| 2: multi-view, 64 views | 0.682 | 0.558 | 1.364 | 0.626 |
| 3: ray traced, 32 spp | 2.002 | 1.608 | 4.113 | 1.918 |

**Error 1 — it spreads light that cannot spread.** Wall brightness approaching a sharp railing bar, averaged over 170 rows. The bar begins at pixel 1064:

| Distance from bar | Method 1 | Method 2 | Reference | Method 1 error |
|---|---|---|---|---|
| 34 px | 180.7 | 187.4 | 187.4 | −6.7 |
| 18 px | 156.2 | 162.1 | 162.1 | −5.9 |
| 10 px | 144.3 | 162.1 | 162.1 | −17.8 |
| 2 px | 127.1 | 162.1 | 162.2 | **−35.1** |

Methods 2 and 3 agree with each other to 0.1 at every distance; method 1 darkens the wall by up to 35/255 within a band the width of the wall's own blur radius. The railing is in perfect focus, so every one of those darkened pixels is light that the renderer invented.

**Error 2 — the data it needs does not exist.** Zone B is where the gather is worst by a wide margin: 12.641 against 1.364. In `dof3-m1-gather.png` the three near posts are opaque soft-edged bars. In `dof3-m2-multiview-64.png` and the reference they are translucent, with the background plainly visible through them. No filter over a pinhole image can recover what is behind an occluder, because the pinhole image never recorded it.

**Neither error is a sampling error.** Increasing the tap count does not converge the gather:

| Samples | Method 1 (taps) | Method 2 (views) |
|---|---|---|
| 8 | 5.626 | 2.455 |
| 16 | 5.238 | 1.487 |
| 32 | 5.022 | 0.923 |
| 64 | 4.946 | 0.682 |

Eight times the work buys method 1 a 12% improvement toward an asymptote near 4.9, while method 2 improves by 3.6x and keeps going. The gather is a *biased* estimator: it converges, but not to the right answer. That distinction is the whole reason Experiments 16 and 17 exist.

## Verification

| Check | Result |
|---|---|
| Build | `cmake --build build --target DoFApproaches` clean with `-Wall -Wextra -Wpedantic` |
| Camera consistency | Railing bar centres identical across all five captures (1076, 1428, 1780, 2132, 2485), confirming the methods differ only in how they resolve the lens |
| Metrics | `python3 tools/compare_dof_methods.py`, written to `output/dof3-comparison.json` |
| Captures | `output/dof3-m1-gather.png`, `output/dof3-reference-pinhole.png`, `output/dof3-diff-m1_gather-vs-reference.png` |

`output/` is git-ignored; every capture above is reproduced by the commands in this file.

## Related Notes

[Experiment 15 notes](../../../notes/graphics/15_screen_space_halo.md)
