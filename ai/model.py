"""Baseline model for the smoke test: a small U-Net that predicts a residual on the sharp image.

Input and output follow ai/DATA_PROTOCOL.md: x [B, 10, H, W] -> y_hat [B, 3, H, W] in model space.
With film=True the 6 camera settings (the protocol's `settings` vector, also x[4:10]) additionally scale
and shift the features after every block, so each level knows where the focus plane is.
This is plumbing for checking the data path end to end, not the research architecture.

Variants for the architecture sweep (`build(arch, ...)`), all residual on the sharp RGB and all with optional FiLM:
  unet        the U-Net above; `skips=False` drops the encoder-decoder skip links (the input residual stays)
  plain       full-resolution stack of residual blocks, no down/up sampling, dilation 1 (narrow view)
  dilated     the same stack with dilations 1, 2, 4, 8, 16 repeated (wide view at full resolution)
  twoarm      a shallow full-resolution detail arm plus a context arm at 1/4 resolution, fused at the end
  twoarm2     experiment_plan.md M1: U-Net arm (fine image + blend mask) and a 1/4-resolution large-blur arm that
              outputs the blurred image itself, mixing light in linear radiance; output = mask blend of the two
"""

import torch.nn.functional as F

import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin=10, cout=3, width=32, depth=4, film=False, n_settings=6, skips=True, extra=0):
        super().__init__()
        self.skips, self.cout = skips, cout
        chans = [width * 2 ** i for i in range(depth + 1)]
        self.inp = block(cin, chans[0])
        self.down = nn.ModuleList(nn.Sequential(nn.Conv2d(chans[i], chans[i], 2, stride=2), block(chans[i], chans[i + 1]))
                                  for i in range(depth))
        self.up = nn.ModuleList(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth)))
        self.fuse = nn.ModuleList(block((2 if skips else 1) * chans[i], chans[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(chans[0], cout + extra, 1)   # extra: raw output maps after the image (e.g. a mask)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)       # starts as the identity: y_hat = sharp
        # one (scale, shift) per channel after each block: inp, down blocks, fuse blocks
        self.film_chans = [chans[0]] + chans[1:] + [chans[i] for i in reversed(range(depth))] if film else []
        if film:
            self.film = nn.Sequential(nn.Linear(n_settings, 128), nn.SiLU(), nn.Linear(128, 2 * sum(self.film_chans)))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)   # starts as no modulation

    def forward(self, x, settings=None):
        mods = []
        if self.film_chans:
            m = self.film(settings if settings is not None else x[:, 4:, 0, 0])
            mods = list(torch.split(m, [2 * c for c in self.film_chans], 1))

        def mod(h):
            if not mods:
                return h
            g, b = mods.pop(0)[:, :, None, None].chunk(2, 1)
            return h * (1 + g) + b

        skips, h = [], mod(self.inp(x))
        for d in self.down:
            skips.append(h)
            h = mod(d(h))
        for u, f in zip(self.up, self.fuse):
            h = mod(f(torch.cat([u(h), skips.pop()], 1) if self.skips else u(h)))
        o = self.out(h)
        if o.shape[1] == self.cout:
            return x[:, :3] + o
        return torch.cat([x[:, :3] + o[:, :self.cout], o[:, self.cout:]], 1)


def film_net(n_settings, chans):
    """MLP from the settings to one (scale, shift) per channel of each listed block; zero-initialised."""
    net = nn.Sequential(nn.Linear(n_settings, 128), nn.SiLU(), nn.Linear(128, 2 * sum(chans)))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def film_mods(net, chans, settings):
    """List of modulators, one per block in order; identity when there is no FiLM net."""
    if net is None:
        return [lambda h: h] * len(chans)
    parts = torch.split(net(settings), [2 * c for c in chans], 1)

    def make(p):
        g, b = p[:, :, None, None].chunk(2, 1)
        return lambda h: h * (1 + g) + b
    return [make(p) for p in parts]


class ResBlock(nn.Module):
    """Two 3x3 convs (dilation d) with GroupNorm and SiLU, plus the input."""

    def __init__(self, c, d=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=d, dilation=d), nn.GroupNorm(8, c), nn.SiLU(),
                                  nn.Conv2d(c, c, 3, padding=d, dilation=d), nn.GroupNorm(8, c))
        self.act = nn.SiLU()

    def forward(self, h):
        return self.act(h + self.body(h))


class PlainNet(nn.Module):
    """Full-resolution residual conv stack: no down- or upsampling. dilations sets each block's dilation."""

    def __init__(self, cin=10, cout=3, width=64, dilations=(1,) * 10, film=False, n_settings=6):
        super().__init__()
        self.inp = block(cin, width)
        self.blocks = nn.ModuleList(ResBlock(width, d) for d in dilations)
        self.out = nn.Conv2d(width, cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.film_chans = [width] * (1 + len(dilations))
        self.film = film_net(n_settings, self.film_chans) if film else None

    def forward(self, x, settings=None):
        mods = film_mods(self.film, self.film_chans, settings if settings is not None else x[:, 4:, 0, 0])
        h = mods[0](self.inp(x))
        for blk, m in zip(self.blocks, mods[1:]):
            h = m(blk(h))
        return x[:, :3] + self.out(h)


class TwoArm(nn.Module):
    """A detail arm at full resolution (small view, keeps edges) and a context arm at 1/4 resolution with
    dilated blocks (wide view: how much blur, what is in front). The context is upsampled and fused with the
    detail features just before the head; there are no other skip links."""

    def __init__(self, cin=10, cout=3, width=32, ctx_width=128, ctx_dilations=(1, 2, 4, 8, 1, 2, 4, 8),
                 film=False, n_settings=6):
        super().__init__()
        self.detail_in = block(cin, width)
        self.detail = nn.ModuleList(ResBlock(width) for _ in range(2))
        self.ctx_in = nn.Sequential(nn.Conv2d(cin, ctx_width // 2, 4, stride=2, padding=1), nn.GroupNorm(8, ctx_width // 2),
                                    nn.SiLU(), nn.Conv2d(ctx_width // 2, ctx_width, 4, stride=2, padding=1),
                                    nn.GroupNorm(8, ctx_width), nn.SiLU())
        self.ctx = nn.ModuleList(ResBlock(ctx_width, d) for d in ctx_dilations)
        self.ctx_out = nn.Conv2d(ctx_width, width, 1)
        self.fuse = nn.ModuleList([block(2 * width, width), ResBlock(width)])
        self.out = nn.Conv2d(width, cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.film_chans = [width] * 3 + [ctx_width] * (1 + len(ctx_dilations)) + [width] * 2
        self.film = film_net(n_settings, self.film_chans) if film else None

    def forward(self, x, settings=None):
        mods = iter(film_mods(self.film, self.film_chans, settings if settings is not None else x[:, 4:, 0, 0]))
        d = next(mods)(self.detail_in(x))
        for blk in self.detail:
            d = next(mods)(blk(d))
        c = next(mods)(self.ctx_in(x))
        for blk in self.ctx:
            c = next(mods)(blk(c))
        c = nn.functional.interpolate(self.ctx_out(c), size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([d, c], 1)
        for f in self.fuse:
            h = next(mods)(f(h))
        return x[:, :3] + self.out(h)


class TwoArmLinear(nn.Module):
    """experiment_plan.md M1. Arm 1 is a U-Net at full resolution that predicts the fine image A (sharp areas,
    edges, spill, small blur) and a blend mask m. Arm 2 works at 1/4 resolution and predicts the heavily blurred
    image B directly: its input is the sharp image box-averaged in linear radiance at 1/4, 1/8, 1/16 and 1/32
    (how far each light spreads, which log space gets wrong, experiment_plan.md E0), plus depth and settings;
    dilated blocks refine the 1/4 image, and it is upsampled in linear radiance. The two are mixed as light:
    y = log(1 + m * B_lin + (1 - m) * A_lin). Pure network: the pooling, up-sampling and exp / log are fixed
    elementwise or resampling steps, no gather. At the start m is about 0.02 and A is the identity.
    After each forward pass `self.aux` holds A, B (model space, full resolution) and m for the per-arm losses."""

    def __init__(self, cin=10, width=16, ctx_width=48, ctx_dilations=(1, 2, 4, 8, 2, 1), pyramid=(8, 16, 32),
                 film=False, n_settings=6):
        super().__init__()
        self.arm1 = UNet(cin, 3, width, film=film, n_settings=n_settings, extra=1)
        with torch.no_grad():
            self.arm1.out.bias[3] = -4.0
        self.pyramid = pyramid
        self.stem = block(3 * (1 + len(pyramid)) + 1 + n_settings, ctx_width)
        self.ctx = nn.ModuleList(ResBlock(ctx_width, d) for d in ctx_dilations)
        self.head = nn.Conv2d(ctx_width, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.film_chans = [ctx_width] * (1 + len(ctx_dilations))
        self.film = film_net(n_settings, self.film_chans) if film else None
        self.aux = None

    def forward(self, x, settings=None):
        s = settings if settings is not None else x[:, 4:, 0, 0]
        a = self.arm1(x, s)
        A, m = a[:, :3], torch.sigmoid(a[:, 3:4].float())
        lin = torch.expm1(x[:, :3].float()).clamp(min=0)
        base = F.avg_pool2d(lin, 4, ceil_mode=True)
        size4 = base.shape[-2:]
        feats = [torch.log1p(base)]
        for k in self.pyramid:
            p = F.avg_pool2d(lin, k, ceil_mode=True)
            feats.append(torch.log1p(F.interpolate(p, size=size4, mode="bilinear", align_corners=False)))
        feats.append(F.avg_pool2d(x[:, 3:4].float(), 4, ceil_mode=True))
        feats.append(s.float()[:, :, None, None].expand(-1, -1, *size4))
        mods = film_mods(self.film, self.film_chans, s)
        h = mods[0](self.stem(torch.cat(feats, 1)))
        for blk, mo in zip(self.ctx, mods[1:]):
            h = mo(blk(h))
        b4 = torch.log1p(base) + self.head(h).float()
        b_lin = F.interpolate(torch.expm1(b4).clamp(min=0), size=x.shape[-2:], mode="bilinear", align_corners=False)
        a_lin = torch.expm1(A.float()).clamp(min=0)
        y = torch.log1p(m * b_lin + (1 - m) * a_lin)
        self.aux = {"A": A.float(), "B": torch.log1p(b_lin), "m": m}
        return y


ARCHS = ("unet", "unet_noskip", "plain", "dilated", "twoarm", "twoarm2")


def build(arch="unet", width=None, film=False):
    """Model for the architecture sweep; width=None takes each variant's default."""
    if arch in ("unet", "unet_noskip"):
        return UNet(width=width or 32, film=film, skips=arch == "unet")
    if arch == "plain":
        return PlainNet(width=width or 64, dilations=(1,) * 10, film=film)
    if arch == "dilated":
        return PlainNet(width=width or 64, dilations=(1, 2, 4, 8, 16) * 2, film=film)
    if arch == "twoarm":
        return TwoArm(width=width or 32, film=film)
    if arch == "twoarm2":
        return TwoArmLinear(width=width or 16, film=film)
    raise ValueError(f"unknown arch {arch!r}; one of {ARCHS}")
