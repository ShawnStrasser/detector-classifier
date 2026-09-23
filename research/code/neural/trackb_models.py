"""Track B (stage 15): backbone variants with dropout, state-dict compatible with
`neural.models`.

`nn.Dropout` and the GRU's own inter-layer dropout carry no parameters, so a checkpoint
written by `PairNetB(arch, dropout=p)` loads into the plain `neural.models.PairNet(arch)`
unchanged -- which is what `infer2.py` and `export_gru.py` do.  With `dropout=0.0` the
modules here are numerically identical to the stage-03 ones.

Where the dropout goes:
  * TCN     -- between the two convolutions of every residual block;
  * GRU     -- between the recurrent layers (`nn.GRU(dropout=...)`);
  * both    -- on the 128-d head, after `proj`, before the phase score.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from neural.models import AttnPool  # noqa: F401  (same pooling, unchanged)


class TCNBlockD(nn.Module):
    """`neural.models.TCNBlock` plus a dropout between the two convolutions."""

    def __init__(self, c: int, k: int, d: int, p: float = 0.0):
        super().__init__()
        pad = (k - 1) * d // 2
        self.c1 = nn.Conv1d(c, c, k, padding=pad, dilation=d)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, k, padding=pad, dilation=d)
        self.b2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.b2(self.c2(self.drop(h)))
        return F.gelu(x + h)


class TCND(nn.Module):
    """`neural.models.TCN` with block dropout.  Identical weights at p = 0."""

    def __init__(self, cin: int = 9, c: int = 96, nblk: int = 7, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, c, 7, stride=2, padding=3),
                                  nn.BatchNorm1d(c), nn.GELU(),
                                  nn.MaxPool1d(2))
        self.blocks = nn.Sequential(*[TCNBlockD(c, 5, 2 ** i, dropout)
                                      for i in range(nblk)])
        self.pool = AttnPool(c)
        self.out_dim = self.pool.out_dim

    def forward(self, x):                                  # x [N,9,T]
        h = self.blocks(self.stem(x))
        return self.pool(h.transpose(1, 2))


class GRUAttnD(nn.Module):
    """`neural.models.GRUAttn` with inter-layer dropout.  Identical weights at p = 0."""

    def __init__(self, cin: int = 9, c: int = 64, h: int = 128, nlayer: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, c, 7, stride=4, padding=3),
                                  nn.BatchNorm1d(c), nn.GELU())
        self.rnn = nn.GRU(c, h, num_layers=nlayer, batch_first=True, bidirectional=True,
                          dropout=float(dropout) if nlayer > 1 else 0.0)
        self.pool = AttnPool(2 * h)
        self.out_dim = self.pool.out_dim

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.pool(h)


BACKBONES_B = {"tcn": TCND, "gru": GRUAttnD}


class PairNetB(nn.Module):
    """`neural.models.PairNet` with head dropout.  Same state-dict keys."""

    def __init__(self, arch: str = "tcn", nfunc: int = 3, dropout: float = 0.0, **kw):
        super().__init__()
        self.arch = arch
        self.kw = dict(kw)
        self.backbone = BACKBONES_B[arch](dropout=dropout, **kw)
        e = self.backbone.out_dim
        self.proj = nn.Sequential(nn.Linear(e, 128), nn.GELU(), nn.Dropout(dropout))
        self.phase = nn.Linear(128, 1)
        self.func = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, nfunc))

    def forward(self, x):
        z = self.proj(self.backbone(x))
        return self.phase(z).squeeze(-1), self.func(z), z


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
