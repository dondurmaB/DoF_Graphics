"""Dev driver (cluster only): render several rooftop views in one job. Not a scene; deleted when done."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render  # noqa: E402

args = sys.argv[1:]
out = args[args.index("--out") + 1]
for env in args[args.index("--envs") + 1].split(","):
    for v in args[args.index("--views") + 1].split(","):
        render.main(["--scene", "rooftop", "--view", v, "--env", env, "--res", "960x540", "--spp", args[args.index("--spp") + 1],
                     "--out", f"{out}/{env}_{v}", "--passes", "sharp"])
