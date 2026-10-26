"""Baseline model for the smoke test: a small U-Net that predicts a residual on the sharp image.

Input and output follow ai/DATA_PROTOCOL.md: x [B, 10, H, W] -> y_hat [B, 3, H, W] in model space.
With film=True the 6 camera settings (the protocol's `settings` vector, also x[4:10]) additionally scale
and shift the features after every block, so each level knows where the focus plane is.
This is plumbing for checking the data path end to end, not the research architecture.
"""

import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin=10, cout=3, width=32, depth=4, film=False, n_settings=6):
        super().__init__()
        chans = [width * 2 ** i for i in range(depth + 1)]
        self.inp = block(cin, chans[0])
        self.down = nn.ModuleList(nn.Sequential(nn.Conv2d(chans[i], chans[i], 2, stride=2), block(chans[i], chans[i + 1]))
                                  for i in range(depth))
        self.up = nn.ModuleList(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth)))
        self.fuse = nn.ModuleList(block(2 * chans[i], chans[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(chans[0], cout, 1)
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
            h = mod(f(torch.cat([u(h), skips.pop()], 1)))
        return x[:, :3] + self.out(h)
