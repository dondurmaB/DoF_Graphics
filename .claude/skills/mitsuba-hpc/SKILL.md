---
name: mitsuba-hpc
description: Connect to the MBZUAI student-lab HPC cluster, render the Mitsuba cafe DoF scene (or run the live interactive viewer) on a GPU node, and get the result in front of the user on their own machine — either via an SSH-tunneled live viewer or by syncing finished stills back. Use this whenever the user asks to render with Mitsuba on HPC, start/reopen the viewer, or inspect/view HPC render output locally.
---

# Mitsuba on HPC

Full background and rationale: `renderer/mitsuba/HPC_GUIDE.md`. This skill
is the condensed operational checklist — read the guide if anything below
is unclear or the user asks "why".

## Before doing anything

1. Ask (or recall from context) the user's cluster username and whether
   they already have a tmux session with an allocation. Do not guess a
   username.
2. `ssh <user>@login-student-lab.mbzu.ae 'squeue -u <user>'` to see what's
   already running. **Never cancel or send input to a job or tmux pane you
   didn't start in this conversation without confirming with the user
   first** — a pane that doesn't look like a plain idle shell (e.g. it's
   running a VS Code Remote-SSH / code-server process) is probably backing
   their live editor connection; leave it alone.
3. Submit new work (`sbatch`, read-only `squeue`/`sinfo`) directly from
   the login node over a fresh one-off SSH command. Only use the user's
   existing tmux session on a compute node for interactive work they
   explicitly point you at, and never `Ctrl-C` or kill anything in it
   without asking first.

## Starting a render

Batch render (fire-and-forget, writes full pass set + metadata.json):

```sh
ssh <user>@login-student-lab.mbzu.ae 'cd <repo> && sbatch renderer/mitsuba/hpc/render_cafe.slurm'
```

Pass custom args with `--export=ALL,RENDER_ARGS="--focus-target menu --f-number 1.4"`.

Poll with `squeue -u <user>` / tail `logs/cafe-render-<jobid>.out` until it
finishes, then sync results back (see below).

## Starting the interactive viewer

Submit as its own SLURM job — this always lands on whatever node SLURM
assigns, independent of any allocation the user already holds, so it
never competes with their existing session:

```sh
ssh <user>@login-student-lab.mbzu.ae 'cd <repo> && mkdir -p logs && sbatch renderer/mitsuba/hpc/viewer.slurm'
```

Get the job id from the `sbatch` output, then poll the log until the
tunnel line appears:

```sh
ssh <user>@login-student-lab.mbzu.ae 'cat logs/cafe-viewer-<jobid>.out' | grep "tunnel:"
```

That prints the node name and a token URL. Open the local tunnel
(background it — it must stay open) and verify it:

```sh
ssh -N -L 8765:<node>:8765 login-student-lab.mbzu.ae   # run in background
curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8765/?t=<token>" --max-time 5
```

Hand the user the `http://localhost:8765/?t=<token>` link.

## Reopening a dropped tunnel

If the user says the viewer/link stopped working, don't resubmit the job —
check it's still running first (`squeue -u <user>`), and if so just reopen
the SSH tunnel with the same node/token from before.

## When done

Only `scancel <jobid>` a viewer job if the user asks, or confirms when you
offer. Never `scancel` a job you didn't submit yourself in this
conversation without asking first.

## Getting results to the user's machine

Batch render output (`output/mitsuba/<name>/`) is small enough to `rsync`/
`scp` back directly — no tunnel needed. PNGs are tonemapped previews; EXRs
are the linear ground truth (need an HDR-aware viewer, not Preview);
`metadata.json` has the camera/lens/timing record. See
`renderer/mitsuba/HPC_GUIDE.md` section 3B for the exact commands and for
building a presentable comparison (e.g. publishing a side-by-side artifact
of sharp vs. DoF vs. depth vs. normal with the metadata as stats).
