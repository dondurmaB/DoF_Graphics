# train10k run status

Snapshot taken 2026-10-23 about 14:55 (+04) from the student-lab cluster (`~/dof_dev/dataset/train10k/_queue`) and CIAI. The run was still going when this was written. `plan.json` is the exact plan file the queue was created from on 2026-10-21 09:58; `GIT_STATE.json` is what the cluster recorded when the code was last synced (`f29819e`, dirty).

## Plan

- 10,000 samples = 5,000 poses x 2 lenses per pose, environments `clear` and `overcast`.
- 100 chunks of 50 poses, 10 per scene, in this order: cafe, courtyard, desk, greenhouse, museum, night_street, pool, rooftop, terminal, workshop.

## Progress at the snapshot

| Scene | Chunks done |
| --- | --- |
| cafe | 10 / 10 |
| courtyard | 10 / 10 |
| desk | 10 / 10 |
| greenhouse | 10 / 10 |
| museum | 8 / 10 |
| night_street, pool, rooftop, terminal, workshop | 0 / 50 |

- 48 of 100 chunks done (2,400 poses), 2 in flight, 0 in `failed/`. A scan of the queue logs found no traceback or error.
- Jobs 244159 (ws-l5-005) and 244160 (ws-l4-008) were running. Last log lines: `museum.clear.000300: done rc 0 in 546s`, `shipped 50 poses`. At about 9 minutes per chunk and 2 workers, the remaining 52 chunks need roughly 4 hours, so the run should finish this evening if the jobs keep requeueing.

## Where the data is

- Finished chunks are shipped to CIAI: `/nfs-stor/rui.gao/Researches/dof/dof_dataset/train10k`. It is not in git. On 2026-10-22 it measured 49 GB for 360 poses (about 0.14 GB per pose), so the full run will be roughly 0.7 TB.
- The cluster home keeps only chunks in progress. Do not use the cluster's `/l` (storage migration).
- `sheets/cafe.png` is the only contact sheet that exists so far (made 2026-10-22). Left column is the sharp input, right column the DoF target, and the caption gives focal length, f-number, focus distance and roll.

## Observations from the cafe sheet

- Sharp and DoF columns line up and the blur grows with aperture and focal length as expected.
- Pose 3 of the 6 shown (rows 5 and 6) is mostly bare wall with a diagonal door edge, and pose 2 (rows 3 and 4) is large wall areas with one frame and one door. Those carry little depth structure; worth checking whether wall-dominated views should be rejected before the other scenes finish.
- Wet and snow presets are still excluded because they look wrong (night_street wet renders daylight; courtyard snow and wet look like clear). `dataset/weather_pilot/` holds the pilot sheets from 2026-10-20.
