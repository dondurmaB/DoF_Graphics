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

**Never run jobs or heavy work on a login node** (`lo-01`/`lo-02`). Only
read-only queries (`sinfo`, `squeue`) belong there. Everything else needs
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
- `*.exr` are the linear radiance data (the actual ground truth) — use an
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
