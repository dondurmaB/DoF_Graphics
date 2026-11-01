"""Time the classic DoF as the real DoFScene GLSL shader (shaders/dof_scene/screen.frag), headless on a GPU.

    python ai/classic_glsl.py LENS_DIR [LENS_DIR ...] --out runs/classic_glsl.json

Runs the shader's own gatherDof() through OpenGL 3.3 on an EGL context (moderngl, no window), on full 1920 x 1080
frames in 32-bit float textures. Two edits to the shader source, both outside gatherDof: depth arrives already
linear in metres (sky = traditional_dof.SKY_DEPTH_M), so linearizeDepth() returns its input, and main() writes the
HDR result instead of the tone-mapped display colour. Per frame: GPU time of one full-screen pass from GL timer
queries (median of --reps after warm-up), and the error against the numpy port (traditional_dof.gather_dof) as a
fraction of the blur that port adds. Needs numpy, moderngl and an EXR reader (OpenEXR or mitsuba); no PyTorch.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "renderer" / "mitsuba"))
import data as D              # noqa: E402
import traditional_dof as T   # noqa: E402

VERT = """#version 330 core
in vec2 pos;
out vec2 texCoord;
void main() { texCoord = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }
"""


def fragment_source():
    src = (REPO / "shaders" / "dof_scene" / "screen.frag").read_text()
    src, n = re.subn(r"float linearizeDepth\(float rawDepth\)\s*\{.*?\n\}",
                     "float linearizeDepth(float rawDepth)\n{\n    return rawDepth;\n}", src, flags=re.S)
    assert n == 1, "linearizeDepth not found"
    head = src[:src.index("void main()")]
    return head + """void main()
{
    vec2 uv = clampUv(texCoord, 1.0 / max(uResolution, vec2(1.0)));
    float linearDepth = texture(uDepth, uv).r;
    fragColor = vec4(gatherDof(uv, linearDepth, signedCoCRadiusPixels(linearDepth)), 1.0);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lens_dirs", nargs="+", type=Path)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="runs/classic_glsl.json")
    args = ap.parse_args()
    import moderngl
    ctx = moderngl.create_context(standalone=True, backend="egl")
    info = {k: ctx.info.get(k) for k in ("GL_RENDERER", "GL_VERSION")}
    print(info, flush=True)
    prog = ctx.program(vertex_shader=VERT, fragment_shader=fragment_source())
    tri = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
    vao = ctx.vertex_array(prog, [(tri, "2f", "pos")])
    rows = []
    for ldir in args.lens_dirs:
        meta = json.loads((ldir / "metadata.json").read_text())["camera"]
        sharp = D.read_exr(ldir.parent / "sharp.exr").astype(np.float32)
        depth = np.load(ldir.parent / "depth.npy").astype(np.float32)
        h, w = depth.shape
        depth_lin = np.where(np.isfinite(depth), depth, T.SKY_DEPTH_M).astype(np.float32)
        # numpy row 0 is the top of the image; a GL texture's row 0 is its bottom. Upload and read back the same
        # way, so the output rows line up with the input rows.
        col = ctx.texture((w, h), 3, np.ascontiguousarray(sharp).tobytes(), dtype="f4")
        dep = ctx.texture((w, h), 1, np.ascontiguousarray(depth_lin).tobytes(), dtype="f4")
        for t in (col, dep):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4, dtype="f4")])
        uniforms = {"uColor": 0, "uDepth": 1, "uFocusDistance": meta["focus_distance_m"],
                    "uFocalLengthMm": meta["focal_length_mm"], "uSensorHeightMm": meta["sensor_height_mm"],
                    "uFNumber": meta["f_number"], "uMaxRadius": T.MAX_RADIUS_PX, "uResolution": (float(w), float(h))}
        for k, v in uniforms.items():
            if k in prog:
                prog[k].value = v
        col.use(0)
        dep.use(1)
        fbo.use()
        ctx.viewport = (0, 0, w, h)
        for _ in range(5):
            vao.render(moderngl.TRIANGLES)
        ctx.finish()
        q = ctx.query(time=True)
        gpu = []
        for _ in range(args.reps):
            with q:
                vao.render(moderngl.TRIANGLES)
            ctx.finish()
            gpu.append(q.elapsed / 1e6)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            vao.render(moderngl.TRIANGLES)
        ctx.finish()
        wall = 1000 * (time.perf_counter() - t0) / args.reps
        out = np.frombuffer(fbo.read(components=4, dtype="f4"), np.float32).reshape(h, w, 4)[..., :3]
        ref = T.gather_dof(sharp, depth, meta["focal_length_mm"] / 1000, meta["sensor_height_mm"] / 1000,
                           meta["f_number"], meta["focus_distance_m"])
        err = float(np.abs(out - ref).mean() / np.abs(ref - sharp).mean())
        r = {"lens_dir": str(ldir), "gpu_ms": float(np.median(gpu)), "wall_ms": wall, "rel_err_vs_numpy": err,
             "f_number": meta["f_number"], "focal_mm": meta["focal_length_mm"]}
        rows.append(r)
        print(json.dumps(r), flush=True)
        for o in (col, dep, fbo):
            o.release()
    print(f"\nGLSL shader on {info['GL_RENDERER']}: median {np.median([r['gpu_ms'] for r in rows]):.2f} ms per "
          f"1080p frame (GPU timer), max error vs numpy {max(r['rel_err_vs_numpy'] for r in rows):.1e}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"gl": info, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
