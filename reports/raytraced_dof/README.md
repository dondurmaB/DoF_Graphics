# Ray-Traced DoF Outputs

Generated Cycles PNGs and their JSON sidecars belong here and are ignored by Git. No reference images were produced during Experiment 16 implementation because Blender was not installed.

See [render commands and matched settings](../../tools/raytraced_reference/README.md). The default run creates `rt_focus5m_f1.4.png`, `rt_focus5m_f2.8.png`, and `rt_focus5m_f8.png`; `--sharp` also creates `rt_sharp.png`. A matching JSON records settings and the device used after a successful render. These names are overwritten on repeated runs.

Cycles is a quality reference. The ≤20 ms/frame target applies to the future AI solution, not to these renders.
