#!/usr/bin/env python3
"""Interactive progressive viewer for the Mitsuba cafe, served to a browser.

    python renderer/mitsuba/viewer.py            # then open http://localhost:8765

One render thread owns Mitsuba and path-traces continuously, accumulating
samples until the camera or lens changes. Camera parameters are pushed into the
already-compiled kernel as opaque values, so moving or refocusing restarts the
accumulation without recompiling (~50 ms per 4 spp at 800x450 on an M4 Max).

Controls (in the page): drag to look, WASD/QE to move, Shift for speed, click to
focus on the surface under the cursor, scroll to change focus distance, 1-4 to
switch DoF / sharp / depth / CoC views.
"""

from __future__ import annotations

import argparse
import io
import secrets
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render as R  # noqa: E402

RESOLUTIONS = {"640x360": (640, 360), "960x540": (960, 540), "1280x720": (1280, 720)}
MODES = ("dof", "sharp", "depth", "coc", "trad")


class ViewerState:
    """Everything the page can change, guarded by `lock`. The render thread polls it and
    restarts accumulation when `render_key()` changes."""

    def __init__(self, view: str, focus_points: dict):
        self.lock = threading.Lock()
        self.focus_points = {k: (v[0] if isinstance(v[0], list) else v) for k, v in focus_points.items()}
        self.mode = "dof"
        self.lens_mm = 50.0
        self.f_number = 1.8
        self.exposure = 0.0
        self.max_spp = 4096
        self.res = "960x540"
        self.restarts = 0
        self.set_view(view)

    def set_view(self, name: str) -> None:
        v = R.VIEWS[name]
        o, t = np.asarray(v["origin"], float), np.asarray(v["target"], float)
        f = (t - o) / np.linalg.norm(t - o)
        self.position = o
        self.yaw = math.atan2(f[0], -f[2])
        self.pitch = math.asin(f[1])
        self.focus_m = self.depth_of(self.focus_points["teapot"])

    def forward(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array([math.sin(self.yaw) * cp, math.sin(self.pitch), -math.cos(self.yaw) * cp])

    def depth_of(self, point) -> float:
        return float(np.dot(np.asarray(point) - self.position, self.forward()))

    def lens(self) -> R.Lens:
        """The camera actually path traced: a thin lens only in DoF mode, else a pinhole."""
        f_no = self.f_number if self.mode == "dof" else 1e5
        return R.Lens(self.position, self.position + self.forward(), self.lens_mm, 24.0, f_no, self.focus_m)

    def optics(self) -> R.Lens:
        """The lens the user dialled in, for CoC maps and the traditional gather."""
        return R.Lens(self.position, self.position + self.forward(), self.lens_mm, 24.0, self.f_number, self.focus_m)

    def pose_key(self) -> tuple:
        """Everything that changes the depth buffer."""
        return (tuple(np.round(self.position, 6)), round(self.yaw, 6), round(self.pitch, 6), self.lens_mm, self.res)

    def render_key(self) -> tuple:
        """Everything that changes the path-traced image; accumulation restarts only when it does.

        Focus and aperture belong here only in DoF mode. The traditional mode
        blurs a pinhole render, so refocusing it just re-runs the 23 ms gather
        on the samples already accumulated.
        """
        lens = (self.f_number, self.focus_m) if self.mode == "dof" else None
        return self.pose_key() + (self.restarts, self.mode == "dof", lens)

    def public(self) -> dict:
        w, h = RESOLUTIONS[self.res]
        lens = R.Lens(self.position, self.position + self.forward(), self.lens_mm, 24.0, self.f_number, self.focus_m)
        return {"mode": self.mode, "lens_mm": self.lens_mm, "f_number": self.f_number,
                "focus_m": self.focus_m, "exposure": self.exposure, "res": self.res,
                "max_spp": self.max_spp, "position": [round(float(x), 3) for x in self.position],
                "coc_inf_px": lens.metadata(w, h)["coc_diameter_px_at_infinity"],
                "targets": sorted(k for k in self.focus_points if k != "pendants"),
                "views": sorted(R.VIEWS), "resolutions": list(RESOLUTIONS)}


class Renderer(threading.Thread):
    def __init__(self, state: ViewerState, scene_dict: dict, out_dir: Path):
        super().__init__(daemon=True)
        self.state = state
        self.scene_dict = scene_dict
        self.out_dir = out_dir
        self.frame_jpeg = b""
        self.frame_id = 0
        self.stats = {"spp": 0, "spp_per_s": 0.0, "status": "compiling"}
        self.requests: list = []  # (kind, payload, event, result-dict)
        self.req_lock = threading.Lock()

    # Called from HTTP threads; executed on the render thread.
    def call(self, kind: str, payload=None, timeout: float = 30.0):
        ev, box = threading.Event(), {}
        with self.req_lock:
            self.requests.append((kind, payload, ev, box))
        ev.wait(timeout)
        return box.get("result")

    def load(self) -> None:
        """Load the scene. Call on the main thread, before start().

        A scene loaded with mi.load_dict on a worker thread renders without
        its sunlight (measured: mean radiance 0.084 vs 0.172), while one loaded
        on the main thread renders correctly from any thread.
        """
        import drjit as dr
        import mitsuba as mi

        self.mi, self.dr = mi, dr
        w, h = RESOLUTIONS[self.state.res]
        self.scene_dict["integrator"] = {"type": "path", "max_depth": 10, "rr_depth": 5}
        self.scene_dict["sensor"] = self.state.lens().sensor(w, h, 1, thin_lens=True)
        self.scene = mi.load_dict(self.scene_dict)
        self.params = mi.traverse(self.scene)
        self.aov = mi.load_dict({"type": "aov", "aovs": "t:depth,pos:position"})

    def run(self) -> None:
        mi = self.mi
        st = self.state
        w, h = RESOLUTIONS[st.res]
        applied_key, pose_applied, res_applied = None, None, st.res
        accum, spp, t_start, last_push = None, 0, time.time(), 0.0
        shown = None  # (mode, exposure, focus, f-number) of the frame on screen
        depth = None
        seed = 0
        while True:
            self._serve_requests(lambda: depth)
            with st.lock:
                key, pose, mode, max_spp, res = st.render_key(), st.pose_key(), st.mode, st.max_spp, st.res
                lens, optics, exposure = st.lens(), st.optics(), st.exposure
            if res != res_applied:
                w, h = RESOLUTIONS[res]
                self.params["sensor.film.size"] = mi.ScalarVector2u(w, h)
                self.params.update()
                res_applied, applied_key = res, None
            if key != applied_key:
                self._apply_camera(lens, w, h)
                applied_key, accum, spp, shown = key, None, 0, None
                t_start = time.time()
                self.stats["status"] = "rendering"
            if pose != pose_applied:
                depth, pose_applied = None, pose
            view = (mode, exposure, optics.focus_m, optics.f_number)
            if mode in ("depth", "coc"):
                if depth is None:
                    depth = self._render_depth(lens, seed)
                    seed += 1
                if view != shown:
                    self._push(self._depth_view(depth, optics, mode, h), 1, 0.0)
                    shown = view
                time.sleep(0.03)
                continue
            if mode == "trad" and depth is None:
                depth = self._render_depth(lens, seed)
                seed += 1
            if accum is not None and view != shown:
                self._show(accum / spp, mode, depth, optics, exposure, spp, self.stats["spp_per_s"])
                shown = view
            if spp >= max_spp:
                self.stats["status"] = "converged"
                time.sleep(0.03)
                continue
            chunk = 1 if spp < 4 else (4 if spp < 64 else 16)
            img = np.asarray(mi.render(self.scene, spp=chunk, seed=seed), dtype=np.float64)[..., :3]
            seed += 1
            accum = img * chunk if accum is None else accum + img * chunk
            spp += chunk
            now = time.time()
            if now - last_push > 0.12 or spp >= max_spp:
                self._show(accum / spp, mode, depth, optics, exposure, spp, spp / max(now - t_start, 1e-6))
                last_push, shown = now, view

    # --- helpers (render thread only) -----------------------------------
    def _show(self, rgb, mode, depth, optics: R.Lens, exposure, spp, rate) -> None:
        rgb = rgb.astype(np.float32)
        if mode == "trad":
            from traditional_dof import gather_dof_gpu

            rgb = gather_dof_gpu(rgb, depth, optics.lens_m, optics.sensor_m, optics.f_number, optics.focus_m)
        self.latest_rgb = rgb
        self._push(R.tonemap(rgb, exposure), spp, rate)

    def _apply_camera(self, lens: R.Lens, w: int, h: int) -> None:
        mi, dr, p = self.mi, self.dr, self.params
        fov_x = math.degrees(2 * math.atan(math.tan(math.radians(lens.fov_y) / 2) * w / h))
        to_world = mi.ScalarTransform4f().look_at(origin=lens.origin.tolist(), target=lens.target.tolist(),
                                                  up=[0, 1, 0])
        p["sensor.to_world"] = mi.Transform4f(to_world.matrix)
        p["sensor.x_fov"] = mi.Float(fov_x)
        p["sensor.focus_distance"] = mi.Float(lens.focus_m)
        p["sensor.aperture_radius"] = mi.Float(max(lens.aperture_radius, 1e-7))
        for k in ("sensor.x_fov", "sensor.focus_distance", "sensor.aperture_radius", "sensor.to_world"):
            dr.make_opaque(p[k])
        p.update()

    def _render_depth(self, lens: R.Lens, seed: int) -> np.ndarray:
        """Planar depth from a pinhole pass (aperture shrunk just for this call)."""
        mi, dr, p = self.mi, self.dr, self.params
        p["sensor.aperture_radius"] = mi.Float(1e-7)
        dr.make_opaque(p["sensor.aperture_radius"])
        p.update()
        img = np.asarray(mi.render(self.scene, integrator=self.aov, spp=1, seed=seed), dtype=np.float32)
        p["sensor.aperture_radius"] = mi.Float(max(lens.aperture_radius, 1e-7))
        dr.make_opaque(p["sensor.aperture_radius"])
        p.update()
        hit = img[..., 0] > 0
        d = np.einsum("hwc,c->hw", img[..., 1:4] - lens.origin.astype(np.float32), lens.forward.astype(np.float32))
        return np.where(hit, d, np.inf)

    def _depth_view(self, depth, lens: R.Lens, mode: str, h: int) -> np.ndarray:
        if mode == "depth":
            finite = np.where(np.isfinite(depth), depth, 30.0)
            v = 1.0 - np.clip(np.log(np.clip(finite, 0.3, 30) / 0.3) / math.log(30 / 0.3), 0, 1)
            return np.repeat(v[..., None], 3, -1)
        # Signed CoC: near field blue, far field orange, in focus black.
        f, s, a = lens.lens_m, lens.focus_m, 2 * lens.aperture_radius
        d = np.where(np.isfinite(depth), depth, 1e6)
        coc = a * f * (d - s) / (d * (s - f)) / lens.sensor_m * h
        m = np.clip(np.abs(coc) / 40.0, 0, 1) ** 0.6
        out = np.zeros(depth.shape + (3,), np.float32)
        out[coc > 0] = np.array([1.0, 0.55, 0.15]) * m[coc > 0][:, None]
        out[coc < 0] = np.array([0.2, 0.55, 1.0]) * m[coc < 0][:, None]
        out[np.abs(coc) < 1.0] = [0.0, 0.0, 0.0]
        return out

    def _push(self, rgb01: np.ndarray, spp: int, rate: float) -> None:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray((np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=90)
        self.frame_jpeg = buf.getvalue()
        self.frame_id += 1
        self.stats.update(spp=spp, spp_per_s=round(rate, 1))

    def _serve_requests(self, get_depth) -> None:
        with self.req_lock:
            pending, self.requests = self.requests, []
        for kind, payload, ev, box in pending:
            try:
                box["result"] = getattr(self, f"_req_{kind}")(payload, get_depth)
            except Exception as exc:  # report to the page rather than kill the loop
                box["result"] = {"error": str(exc)}
            ev.set()

    def _req_pick(self, payload, get_depth):
        """Planar depth of the surface under a normalized (u, v) image point."""
        st = self.state
        with st.lock:
            lens = st.lens()
        depth = get_depth()
        if depth is None:
            depth = self._render_depth(lens, 991)
        hgt, wid = depth.shape
        x = min(max(int(payload["u"] * wid), 0), wid - 1)
        y = min(max(int(payload["v"] * hgt), 0), hgt - 1)
        win = depth[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        win = win[np.isfinite(win)]
        return {"depth": float(np.median(win))} if win.size else {"depth": None}

    def _req_snapshot(self, payload, get_depth):
        mi = self.mi
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = self.out_dir / stamp
        out.mkdir(parents=True, exist_ok=True)
        rgb = getattr(self, "latest_rgb", None)
        if rgb is None:
            return {"error": "nothing rendered yet"}
        with self.state.lock:
            st = self.state.public()
            mode, lens = self.state.mode, self.state.lens()
        mi.Bitmap(rgb).write(str(out / f"{mode}.exr"))
        (out / f"{mode}.png").write_bytes(self.frame_jpeg)
        h, w = rgb.shape[:2]
        meta = {"viewer_state": st, "spp": self.stats["spp"], "camera": lens.metadata(w, h)}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        return {"saved": str(out)}


def make_handler(state: ViewerState, renderer: Renderer, token: str | None):
    """`token`, when set, must arrive once as ?t=... and is then kept in a cookie.

    On a shared cluster node the server listens beyond localhost, and without
    it any user on the cluster network could drive the renderer.
    """
    page = (HERE / "viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _authorized(self) -> bool:
            if token is None:
                return True
            if f"dof_token={token}" in self.headers.get("Cookie", ""):
                return True
            return self.path.split("?", 1)[-1] == f"t={token}"

        def do_GET(self):
            if not self._authorized():
                return self._send(b"forbidden: open the URL printed by the viewer", "text/plain", 403)
            if token is not None and self.path == f"/?t={token}":
                self.send_response(302)
                self.send_header("Set-Cookie", f"dof_token={token}; Path=/; HttpOnly; SameSite=Strict")
                self.send_header("Location", "/")
                self.end_headers()
                return
            if self.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif self.path.startswith("/frame"):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Id", str(renderer.frame_id))
                body = renderer.frame_jpeg
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state":
                with state.lock:
                    s = state.public()
                s.update(renderer.stats, frame_id=renderer.frame_id)
                self._json(s)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if not self._authorized():
                return self._send(b"forbidden", "text/plain", 403)
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/set":
                with state.lock:
                    if body.get("view") in R.VIEWS:
                        state.set_view(body["view"])
                    if body.get("mode") in MODES:
                        state.mode = body["mode"]
                    if body.get("res") in RESOLUTIONS:
                        state.res = body["res"]
                    if "target" in body and body["target"] in state.focus_points:
                        state.focus_m = max(0.1, state.depth_of(state.focus_points[body["target"]]))
                    for key, lo, hi in (("lens_mm", 16, 200), ("f_number", 0.95, 32), ("focus_m", 0.1, 50),
                                        ("exposure", -6, 6), ("max_spp", 1, 65536)):
                        if key in body:
                            val = min(max(float(body[key]), lo), hi)
                            setattr(state, key, int(val) if key == "max_spp" else val)
                    if "max_spp" in body:
                        state.restarts += 1
                self._json({"ok": True})
            elif self.path == "/move":
                with state.lock:
                    state.yaw += float(body.get("dyaw", 0))
                    state.pitch = min(max(state.pitch + float(body.get("dpitch", 0)), -1.45), 1.45)
                    flat = np.array([math.sin(state.yaw), 0.0, -math.cos(state.yaw)])
                    right = np.array([math.cos(state.yaw), 0.0, math.sin(state.yaw)])
                    step = (float(body.get("forward", 0)) * flat + float(body.get("right", 0)) * right
                            + np.array([0.0, float(body.get("up", 0)), 0.0]))
                    state.position = state.position + step
                    state.position[0] = min(max(state.position[0], -3.6), 3.6)
                    state.position[1] = min(max(state.position[1], 0.15), 3.0)
                    state.position[2] = min(max(state.position[2], -6.8), 3.3)
                self._json({"ok": True})
            elif self.path == "/pick":
                res = renderer.call("pick", body)
                if res and res.get("depth"):
                    with state.lock:
                        state.focus_m = max(0.1, res["depth"])
                self._json(res or {"error": "timeout"})
            elif self.path == "/snapshot":
                self._json(renderer.call("snapshot") or {"error": "timeout"})
            else:
                self._send(b"not found", "text/plain", 404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to listen on; 0.0.0.0 on a cluster node (a token is then required)")
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--view", default="home", choices=sorted(R.VIEWS))
    ap.add_argument("--assets", default=str(HERE / "generated"))
    ap.add_argument("--out", default=str(R.REPO / "output" / "mitsuba" / "viewer"))
    args = ap.parse_args()

    variant = R.pick_variant(args.variant)
    from cafe_scene import build_scene

    scene_dict, focus_points = build_scene(Path(args.assets))
    state = ViewerState(args.view, focus_points)
    renderer = Renderer(state, scene_dict, Path(args.out))
    renderer.load()
    renderer.start()
    token = None if args.host in ("127.0.0.1", "localhost") else secrets.token_urlsafe(16)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, renderer, token))
    suffix = f"/?t={token}" if token else ""
    print(f"variant {variant} | viewer on http://localhost:{args.port}{suffix}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
