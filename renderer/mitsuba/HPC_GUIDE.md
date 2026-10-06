# Mitsuba on the MBZUAI cluster: connect, render, inspect locally

For anyone new to this project's HPC setup. Assumes you already have an
account on `login-student-lab.mbzu.ae` and the university VPN up. If you
just want the render CLI flags, see `renderer/mitsuba/README.md` instead —
this doc is about the connect/run/inspect loop itself.

## Why Mitsuba fits this loop

Mitsuba 3 is a plain Python library: `mi.load_file()` / `mi.render()`
behave identically with or without a display attached, so the exact same
scene script runs unmodified on a laptop (`metal_ad_rgb` on Apple GPUs,
`llvm_ad_rgb` on CPU) and on a cluster node (`cuda_ad_rgb`). That is what
lets you develop locally and move the same code to HPC for production spp
with no fork. It also means the interactive viewer is just a Python HTTP
server — no window manager, no native GUI — so it can run on a headless
compute node and be reached over a single SSH-tunneled port.

## 1. Connect

```sh
ssh <username>@login-student-lab.mbzu.ae
```

**Never render or build environments on a login node** (`lo-01`/`lo-02`).
Use it for queries (`sinfo`, `squeue`), job submission (`sbatch`/`srun`),
and lightweight file management. Rendering and environment setup need
an allocation on a compute node.

Partitions:
- `ws-ia` — default, 116 workstations, 1x RTX 5000 Ada each, 48 CPU,
  230 GB RAM, 1-day time limit.
- `gpu` — 4 nodes x 8 RTX 5000 Ada, for anything that needs multiple GPUs.

Get an interactive shell on a compute node (keep this in a `tmux` session
so it survives disconnects — do not reuse a tmux pane that your editor's
Remote-SSH connection also drives, the two will fight over the same
terminal):

```sh
tmux new -s mitsuba
srun --partition=ws-ia --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash
```

Clone or `cd` into your copy of this repo, then set up the environment
once:

```sh
cd DoF_Graphics
conda env update -f environment.yml --prune   # creates/updates dof-graphics, adds mitsuba==3.9.1
conda activate dof-graphics
python -c "import mitsuba; print(mitsuba.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv
```

## 2. Render on HPC

Two modes, same scene:

**Batch render** — produces the full pass set (`sharp`, `dof`, `depth`,
`gbuffer` + previews + `metadata.json`) and exits:

```sh
python renderer/mitsuba/render.py --res 1920x1080 --spp 2048 --dof-spp 4096 \
  --variant cuda_ad_rgb --out output/mitsuba/<name>
```

Or as a SLURM job from the login node (submit only from the login node,
never render on it):

```sh
sbatch renderer/mitsuba/hpc/render_cafe.slurm
sbatch --export=ALL,RENDER_ARGS="--focus-target menu --f-number 1.4" renderer/mitsuba/hpc/render_cafe.slurm
```

On an RTX 5000 Ada, 1920x1080 at 1024 spp takes about 10s per beauty pass.

**Interactive viewer** — a progressive path tracer you drive live from a
browser, for exploring focus/aperture/camera before committing to a batch
render:

```sh
sbatch renderer/mitsuba/hpc/viewer.slurm
```

## 3. Inspect from your own machine

This is the part that actually needs HPC-specific plumbing. Two ways to
get the result in front of your eyes, depending on whether you rendered a
fixed set of stills or want to fly the camera live.

### A. Live viewer over an SSH tunnel

```sh
grep "tunnel:" logs/cafe-viewer-<jobid>.out   # on the login node — prints the node name and a token URL
```

That line gives you the exact two commands to run **on your laptop**:

```sh
ssh -N -L 8765:<node>:8765 login-student-lab.mbzu.ae   # keep this open
open "http://localhost:8765/?t=<token>"
```

The server listens on the compute node's network interface, so the random
per-job token is required — without it you get a 403. Controls: drag to
look, WASD to move, Q/E for down/up, Shift for speed, click to focus,
scroll to change focus distance, `[`/`]` for aperture, `1`-`5` to switch
between DoF / sharp / depth / CoC / traditional views. When you're done,
`scancel <jobid>` on the login node to free the GPU.

If the tunnel drops (laptop sleeps, VPN blips), the job is usually still
running — check with `squeue -u <username>` before resubmitting anything,
then just re-run the `ssh -N -L ...` command.

### B. Pull finished stills back to your laptop

For a batch render, the outputs are small enough to just copy down with
`scp`/`rsync` rather than needing a tunnel:

```sh
rsync -avz login-student-lab.mbzu.ae:"DoF_Graphics/output/mitsuba/<name>/" ./output/mitsuba/<name>/
```

- `*.png` (`sharp.png`, `dof.png`, `depth.png`, `normal.png`) are
  ACES-tonemapped previews — open with anything.
- `*.exr` are linear radiance data, not automatically qualified ground truth — use an
  HDR-aware viewer (e.g. [`tev`](https://github.com/Tom94/tev)), not
  Preview/Finder, or load with `mitsuba.Bitmap` / `imageio` in Python.
- `metadata.json` has the full camera/lens/timing record for that render —
  read this before re-running anything, it tells you exactly what
  produced the images (focus distance, f-number, spp, seconds per pass).

## Troubleshooting

- `squeue -u <username>` before submitting anything, so you know what's
  already running and on which node.
- Don't kill another tmux pane or job that isn't yours — if a pane looks
  like it's running something other than a plain shell (e.g. a VS Code
  Remote-SSH server), it probably is; leave it alone or ask whoever owns
  it.
- `llvm_ad_rgb` (CPU fallback) needs `libLLVM` on the node; if it's
  missing, use `--variant cuda_ad_rgb` explicitly instead of `auto`.
- The PyPI `mitsuba` wheel ships CUDA variants prebuilt for Linux — you
  need an NVIDIA driver on the node, not a full CUDA toolkit install.

## 4. Stage 3: production gather and dataset arrays

Use `stage3/dataset`, which includes the full Mitsuba café and the actual
100-tap GLSL gather. The browser's traditional mode is a separate preview
approximation, not the dataset baseline. See [the dataset contract](../../dataset/README.md)
for tensors, splits and metadata. Targets are **unqualified**: reference noise
is not measured per sample. The render timings in section 2 do not measure this
multi-pass dataset pipeline and must not be used to size its array jobs.

These are instructions for the user to run. The stage-3 EGL backend on an
actual NVIDIA cluster driver is **not verified**. The connection/allocation
procedure above comes from the existing cluster guide; the additions below
have not been executed on the cluster.

### Missing setup details

For SSH keys, on the laptop choose a new filename (do not overwrite an existing
key), then install only its public half using the account's usual login:

```sh
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_mbzuai
cat ~/.ssh/id_ed25519_mbzuai.pub | ssh <username>@login-student-lab.mbzu.ae \
  'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
ssh-add ~/.ssh/id_ed25519_mbzuai
ssh -i ~/.ssh/id_ed25519_mbzuai <username>@login-student-lab.mbzu.ae
```

The last command should reach `lo-01` or `lo-02` without an account-password
prompt; a key-passphrase or institutional MFA prompt may remain. Cluster
acceptance of public keys is **not verified**; follow local account policy if
the key is rejected. Continue with section 1's tmux and `srun` allocation.

Keep the repository at `/home/$USER/DoF_Graphics` and the conda environment under
home; put the persistent dataset at `/l/users/$USER/dof-v1`. Reserve `/temp`
(and `/tmp` for the smoke test) for disposable node-local scratch, not the only
copy of renders. The reported home/parallel quotas are 1 TB/3 TB and the train
dataset planning size is roughly 40 GB; available quota, retention and actual
dataset size are **not verified**. Check the allocated node before a large run:

```sh
df -h "/home/$USER" "/l/users/$USER" /temp
```

Free filesystem space is not a per-user quota report. Follow site quota tools
if the administrator provides them. Clone on the allocated compute node if
needed, then use section 1's environment setup. No sudo is needed:

```sh
git clone --branch stage3/dataset https://github.com/dondurmaB/DoF_Graphics.git "$HOME/DoF_Graphics"
```

If conda is not initialized, use the installed initialization file:

```sh
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f /apps/local/anaconda3/conda_init.sh ]; then
  source /apps/local/anaconda3/conda_init.sh
else
  echo 'Conda initialization path not found; ask the lab administrator.'
fi
```

The current `dataset.slurm` hard-codes the first path and the `dof-graphics`
environment. If only the site initialization path exists, change that one
`source` line in the submit script before submission. Do not copy a Mac conda
environment to Linux: create it from `environment.yml` on the allocated node.

### Preflight on the allocated GPU node

After `conda activate dof-graphics`, `hostname` must identify a compute node,
and `nvidia-smi` must show the allocated NVIDIA GPU. Then, from the repository:

```sh
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); print(mi.__version__, mi.variant())"
export DOF_GL_BACKEND=egl
export DOF_EGL_DEVICE=0
python renderer/mitsuba/gl_context.py
python dataset/verify_pipeline.py
python dataset/build_dataset.py --dry-run --out /tmp/ds-plan
python dataset/build_dataset.py --out /tmp/ds --preset smoke --limit 2 --variant cuda_ad_rgb
```

Use a fresh smoke output path if `/tmp/ds` already contains samples; otherwise
the command skips them and proves no new rendering. Preserve older output.
The explicit EGL index 0 is for the guide's **single-GPU `ws-ia` nodes only**.
CUDA allocation IDs/UUIDs are not generally EGL enumeration indices; do not
reuse this selection on multi-GPU or MIG nodes without verifying the mapping.

Required results, in order:

- CUDA check: `3.9.1 cuda_ad_rgb` (Mac verification uses `metal_ad_rgb`).
- GL check: backend `EGL`, vendor/renderer identifying **NVIDIA**, desktop GL
  3.3 or newer, and `OK: offscreen GL context is usable for the production gather.`
  `llvmpipe`, `softpipe`, or `swrast` is a failure, even if images can be produced.
- Pipeline check: every check `PASS`, including different naive/weighted
  outputs, then `READY: this machine can generate the dataset.`
- Dry run: 2,592 planned samples at the default 12 seeds; no rendering.
- Smoke: `wrote 2, skipped 0, failed 0`; each sample has `sharp.exr`, `dof.exr`,
  `depth.npy`, `coc.npy`, both `gather-*.exr`, four EXR JSON seals and `sample.json`.
  Inspect the EXRs and metadata; an exit code alone is not a visual check.

### Small array first, then measure the full job

Submit from the login node as in section 2. Create `logs/` **before** `sbatch`:
SLURM opens its output file before the script starts. Use a distinct output
directory for each preset/configuration; the builder currently skips an existing
`sample.json` without reauthenticating it and rewrites the top-level plan.
Leave `OVERWRITE` unset: the current script treats **any nonempty value,
including `0`, as permission to overwrite**.

```sh
cd "$HOME/DoF_Graphics"
mkdir -p logs "/l/users/$USER/dof-stage3-trial"
unset OVERWRITE SHARDS SHARD EXTRA_ARGS
export DOF_GL_BACKEND=egl DOF_EGL_DEVICE=0
export PRESET=train OUT="/l/users/$USER/dof-stage3-trial"
export EXTRA_ARGS='--limit 2'
sbatch --array=0-1%1 --time=01:00:00 --export=ALL renderer/mitsuba/hpc/dataset.slurm
```

Expected: `Submitted batch job <id>`, then two shard logs each ending with
`wrote 2, skipped 0, failed 0`. This verifies array execution at the **train**
preset. Four configurations are not a representative timing study: time more
views, lenses, focus targets and scene seeds before scaling. Use the saved
`shard-*.json` value `seconds / written` from fresh runs to include scene setup,
gathers and file I/O. Per-sample `timings_s` currently covers sharp/reference/
G-buffer rendering only; it omits gathers, writing and scene setup.

For `S=2592` samples, `K` shards, conservative measured end-to-end cost `t`
seconds/sample and wall limit `W` seconds, choose K so
`ceil(S/K) * t * 1.5 < W`, allowing separately for unusually expensive scene
initialization. Measure on the actual GPU/storage at the chosen preset, not
from the 16-spp smoke run. For the script's default W=43,200 s (12 hours), a
**conditional** 64-shard submission is below; replace 64 and the last index
together if measurements require a different count. `%4` limits concurrency
to four allocated GPUs, subject to lab policy; it is not the shard count.

```sh
unset EXTRA_ARGS OVERWRITE SHARD
export PRESET=train OUT="/l/users/$USER/dof-v1"
export SHARDS=64
mkdir -p logs "$OUT"
sbatch --array=0-63%4 --time=12:00:00 --export=ALL renderer/mitsuba/hpc/dataset.slurm
```

Keep the output new for the first run. After a timeout, resubmit only affected
indices with the **original SHARDS=64**, same revision/configuration and output,
and no `OVERWRITE`; e.g. `--array=7,19%2`. Otherwise SLURM's smaller task count
changes the modulo partition. Completed samples are skipped; partial samples
without a completion marker are rendered again. Stop the old jobs before
retrying their indices. Do not resume across settings or source changes; the
current filename-based skip is not an integrity check.

### Stage-3 failure checks

- No allocated GPU: check `hostname`, `squeue -u "$USER"`, and `nvidia-smi`;
  obtain section 1's GPU allocation rather than rendering on `lo-01`/`lo-02`.
- EGL failure/software renderer: preserve the exact `gl_context.py` error and
  inspect vendor `libEGL`/`libGL` availability and the allocation/device mapping.
  Do not submit the dataset or install a software fallback to hide the failure.
- Missing CUDA variant: check `which python` and `mi.variants()` inside the
  Linux conda environment. Recreate it from `environment.yml` if wrong; if the
  variant exists but fails, investigate the allocated driver. Do not silently
  switch to CPU for the dataset job.
- Missing SLURM log: create `logs/` in the submission directory before `sbatch`.
- Wall-clock timeout: use the resume procedure above and resize shards from
  observed cost. A killed process may never write its shard summary; inspect
  the log and completed sample markers as well.
