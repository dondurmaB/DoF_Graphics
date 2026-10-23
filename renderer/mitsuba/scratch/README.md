# Cluster scratch scripts

One-off probes and dev scripts that only existed on the student-lab cluster (`~/dof_dev/renderer/mitsuba/`). They were copied here unchanged on 2026-10-23 so the cluster can be dropped, and keep their original relative layout (`hpc/`, `scenes/`).

They were written as throwaway checks (several say "delete when done") and import siblings by plain module name or by `parents[1]`, so they are records, not maintained tools. To run one, copy it back to the path it had under `renderer/mitsuba/`.

| Script | What it was for (from its docstring) |
| --- | --- |
| `_freeze_probe.py` | Which scenes `lens_trace` can record, and which scene parts break the recording |
| `_freeze_probe3.py` | Minimal reproduction of the area-emitter recording failure |
| `_freeze_probe4.py` | Which area emitters of a scene break `lens_trace` recording |
| `_validate_fast.py` | Frozen `render_beauty` / `render_gbuffer` vs `mi.render`, same lens, same filter |
| `_filter_compare.py` | Gaussian pixel filter (`mi.render`, dataset path) vs box filter (`lens_trace`), same view |
| `_cocvalid_test.py` | `coc_valid` inside a frozen AOV kernel, tested apart from the pilot modules |
| `hpc/_inventory.py` | List every object in every scene: name, instances, triangles, material, procedural or scanned |
| `hpc/_probe_inst.py`, `_probe_sky.py`, `_probe_sources.py`, `_probe_tex.py`, `_probe_web.py` | Probes of instancing, sky, asset sources, textures and web assets (no docstrings) |
| `scenes/_desk_probe.py` | Per-part bounds and top-section centroid of web models |
| `scenes/_greenhouse_probe.py` | Variant layout of scanned plant sheets |
| `scenes/_night_street_dev.py` | Noise ablation for night_street |
| `scenes/_pool_preview.py` | Tone-map sampler outputs into sharp / DoF PNG pairs |
| `scenes/_rooftop_dev.py` | Render several rooftop views in one job |
| `scenes/_rooftop_probe.py` | Fetch ambientCG facades and tile their colour maps into a contact sheet |
