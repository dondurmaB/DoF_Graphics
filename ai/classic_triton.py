"""The classic screen-space DoF gather as one hand-written GPU kernel (Triton), for a fair speed comparison.

Same algorithm as renderer/mitsuba/traditional_dof.py gather_dof (the DoFScene shader's gatherDof): per pixel, 64
golden-angle disc taps scaled by the pixel's own CoC radius (capped at 32 px), bilinear samples of colour, depth and
radius, depth-aware weights. Here every pixel's 64 taps run in one fused kernel with no intermediate arrays, the way
a shader would, instead of the 64 whole-image passes of the PyTorch port (ai/speed.py gather_torch).

    from classic_triton import gather_triton
    out = gather_triton(rgb [3, H, W] linear, depth [H, W] metres (inf = sky), focal_m, sensor_m, f_number, focus_m)

`python ai/speed_classic.py ROOT` checks it against the numpy original and times it.
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "renderer" / "mitsuba"))
import traditional_dof as T  # noqa: E402


@triton.jit
def _ss(e0, e1, x):
    t = tl.minimum(tl.maximum((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@triton.jit
def _bil(ptr, i00, i01, i10, i11, w00, w01, w10, w11, mask):
    return (tl.load(ptr + i00, mask=mask, other=0.0) * w00 + tl.load(ptr + i01, mask=mask, other=0.0) * w01
            + tl.load(ptr + i10, mask=mask, other=0.0) * w10 + tl.load(ptr + i11, mask=mask, other=0.0) * w11)


@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w) for b, w in ((128, 4), (256, 4), (256, 8))],
                 key=["H", "W"])
@triton.jit
def _gather_kernel(rgb, depth, radius, disk, out, H, W, MAXR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    HW = H * W
    mask = offs < HW
    py = (offs // W).to(tl.float32)
    px = (offs % W).to(tl.float32)
    d = tl.load(depth + offs, mask=mask, other=1.0)
    r = tl.load(radius + offs, mask=mask, other=0.0)
    blur = tl.minimum(tl.abs(r), MAXR)
    centre_fg = r < 0.0
    active = blur >= 0.5
    acc_r = tl.load(rgb + offs, mask=mask, other=0.0)
    acc_g = tl.load(rgb + HW + offs, mask=mask, other=0.0)
    acc_b = tl.load(rgb + 2 * HW + offs, mask=mask, other=0.0)
    wsum = tl.full([BLOCK], 1.0, tl.float32)
    for i in range(64):                         # a real loop: unrolling 64 taps x 20 loads made compiling crawl
        dx = tl.load(disk + 3 * i)
        dy = tl.load(disk + 3 * i + 1)
        dl = tl.load(disk + 3 * i + 2)
        sx = tl.minimum(tl.maximum(px + dx * blur, 0.0), W - 1.0)
        sy = tl.minimum(tl.maximum(py + dy * blur, 0.0), H - 1.0)
        x0 = sx.to(tl.int32)
        y0 = sy.to(tl.int32)
        x1 = tl.minimum(x0 + 1, W - 1)
        y1 = tl.minimum(y0 + 1, H - 1)
        fx = sx - x0.to(tl.float32)
        fy = sy - y0.to(tl.float32)
        i00, i01, i10, i11 = y0 * W + x0, y0 * W + x1, y1 * W + x0, y1 * W + x1
        w00, w01, w10, w11 = (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy
        s_d = _bil(depth, i00, i01, i10, i11, w00, w01, w10, w11, mask)
        s_r = _bil(radius, i00, i01, i10, i11, w00, w01, w10, w11, mask)
        delta = s_d - d
        similar = 1.0 - _ss(0.05, 1.4, tl.abs(delta))
        keep = tl.where(centre_fg, 1.0 - _ss(-0.05, 0.25, delta), 1.0)
        big_fg = tl.where(s_r < -0.5, _ss(0.2, 1.2, tl.abs(s_r)), 0.0)
        reach = _ss(dl * blur - 0.75, dl * blur + 0.75, tl.abs(s_r))
        wgt = (0.18 + 0.82 * similar) * tl.maximum(keep, big_fg * 0.85) * tl.maximum(reach, similar * 0.65)
        wgt = tl.where(active, wgt, 0.0)
        acc_r += _bil(rgb, i00, i01, i10, i11, w00, w01, w10, w11, mask) * wgt
        acc_g += _bil(rgb + HW, i00, i01, i10, i11, w00, w01, w10, w11, mask) * wgt
        acc_b += _bil(rgb + 2 * HW, i00, i01, i10, i11, w00, w01, w10, w11, mask) * wgt
        wsum += wgt
    tl.store(out + offs, acc_r / wsum, mask=mask)
    tl.store(out + HW + offs, acc_g / wsum, mask=mask)
    tl.store(out + 2 * HW + offs, acc_b / wsum, mask=mask)


_DISK = {}


def gather_triton(rgb, depth, focal_m, sensor_m, f_number, focus_m, max_radius=T.MAX_RADIUS_PX):
    """rgb [3, H, W] fp32 linear, depth [H, W] fp32 metres (inf = sky), both on the GPU -> [3, H, W]."""
    h, w = depth.shape
    dev = depth.device
    if dev not in _DISK:
        _DISK[dev] = torch.tensor([[dx, dy, dl] for (dx, dy), dl in zip(T.DISK, T.DISK_LEN)],
                                  dtype=torch.float32, device=dev).contiguous()
    depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, T.SKY_DEPTH_M)).contiguous()
    aperture = focal_m / f_number
    radius = (0.5 * aperture * focal_m * (depth - focus_m) / (depth * (focus_m - focal_m)) / sensor_m * h).contiguous()
    rgb = rgb.contiguous()
    out = torch.empty_like(rgb)
    grid = lambda meta: (triton.cdiv(h * w, meta["BLOCK"]),)  # noqa: E731
    _gather_kernel[grid](rgb, depth, radius, _DISK[dev], out, h, w, MAXR=float(min(max_radius, 32.0)))
    return out
