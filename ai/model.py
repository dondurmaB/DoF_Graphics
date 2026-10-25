"""Baseline model for the smoke test: a small U-Net that predicts a residual on the sharp image.

Input and output follow ai/DATA_PROTOCOL.md: x [B, 10, H, W] -> y_hat [B, 3, H, W] in model space.
This is plumbing for checking the data path end to end, not the research architecture.
"""

import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin=10, cout=3, width=32, depth=4):
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

    def forward(self, x):
        skips, h = [], self.inp(x)
        for d in self.down:
            skips.append(h)
            h = d(h)
        for u, f in zip(self.up, self.fuse):
            h = f(torch.cat([u(h), skips.pop()], 1))
        return x[:, :3] + self.out(h)
