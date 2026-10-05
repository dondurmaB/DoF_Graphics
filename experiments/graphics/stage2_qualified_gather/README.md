# Qualified stage-2 experiment

The production OpenGL gather is evaluated on Mitsuba pinhole RGB/depth against
Mitsuba thin-lens images from the same scene. This is an **opaque diffuse,
fixed-flash café with direct path transport**, not a full-GI or matched-café
material experiment. Read the [meeting account](../../../notes/graphics/stage2_qualified_gather.md)
for the model, measured uncertainty, results and limits, and the
[run guide](../../../renderer/mitsuba/STAGE2.md) for the one-command workflow.

- [Results table](evidence/RESULTS.md), [full numeric results](evidence/results.json),
  [settings](evidence/settings.json), and [verification record](verification.json).
- [f/1.2 comparison](evidence/f1.2/comparison.png): pinhole, actual GL naive,
  actual GL weighted, thin-lens reference.
- [f/22 comparison](evidence/f22/comparison.png).
- [28 mm inspection view](evidence/wide-angle-inspection.png); not a qualified reference.
- Raw float32 EXRs, paired-seed images, authenticated sidecars, planar depth,
  edge masks, linear differences and displayed previews are all in `evidence/`.
- `diagnostics/` retains final execution/tests and earlier **failed** noise
  qualification reports. Earlier full render sets remain in `output/stage2/`.
- [SHA-256 manifest](manifest.json) covers every preserved evidence and diagnostic
  file plus the verification record. README/report text is versioned by Git.

Both stops use 131,072 spp **per independent seed**. Reference A is compared;
B estimates its noise using the equal-N difference divided by sqrt(2). Every
variant/region passed the unchanged 10% combined-noise-to-MAE threshold. Actual
ratios were 0.254%–4.851%. No sample count alone is described as convergence.

From the project root, reauthenticate all stored inputs and repeat the metrics:

```sh
.venv/bin/python renderer/mitsuba/stage2_experiment.py --compare-only --out experiments/graphics/stage2_qualified_gather/evidence
```

Render a new complete experiment without overwriting anything:

```sh
.venv/bin/python renderer/mitsuba/stage2_experiment.py --out output/stage2/new-run
```

The evidence includes the actual EXRs, not only screenshots or summary numbers.
No previous experiment archive was modified. This local experiment does not
archive or remove any of the other working project folders.
