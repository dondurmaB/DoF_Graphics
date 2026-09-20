# Ray-Traced DoF Outputs

Generated Cycles PNGs and their JSON sidecars belong here and are ignored by Git. The Experiment 16 runtime fix produced all four 1200×1200 previews with Blender 5.2.2 LTS, Cycles, 32 samples and Apple M4 Max Metal. Total render-loop time was 8.27 seconds; per-image time and settings are in the JSON sidecars. All four PNGs were inspected for basic scene sanity; no paired BasicDoF comparison is claimed.

See [render commands and matched settings](../../tools/raytraced_reference/README.md). The default run creates `rt_focus5m_f1.4.png`, `rt_focus5m_f2.8.png`, and `rt_focus5m_f8.png`; `--sharp` also creates `rt_sharp.png`. A matching JSON records settings and the device used after a successful render. These names are overwritten on repeated runs.

Cycles is a quality reference. The ≤20 ms/frame target applies to the future AI solution, not to these renders.
