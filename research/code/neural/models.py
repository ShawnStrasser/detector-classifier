"""Stage 03: four small pair-scoring backbones, all < 2M params, all CPU-runnable.

Every backbone maps one (detector, candidate phase) pair raster to an embedding;
`PairNet` turns the embedding into (a) a scalar phase logit that is softmaxed across
the signal's candidates and (b) a 3-way function logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    """Learned-query attention pooling, concatenated with mean+max."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.out_dim = 3 * d

    def forward(self, h: torch.Tensor) -> torch.Tensor:   # h [N,T,d]
        w = torch.softmax(self.score(h).float(), dim=1).to(h.dtype)
        return torch.cat([(w * h).sum(1), h.mean(1), h.amax(1)], dim=-1)


# --------------------------------------------------------------------- (a) TCN
class TCNBlock(nn.Module):
    def __init__(self, c: int, k: int, d: int):
        super().__init__()
        p = (k - 1) * d // 2
        self.c1 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, k, padding=p, dilation=d)
        self.b2 = nn.BatchNorm1d(c)

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.gelu(x + h)


class TCN(nn.Module):
    """Dilated 1-D CNN over the 1 s raster."""

    def __init__(self, cin: int = 9, c: int = 96, nblk: int = 7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, c, 7, stride=2, padding=3),
                                  nn.BatchNorm1d(c), nn.GELU(),
                                  nn.MaxPool1d(2))
        self.blocks = nn.Sequential(*[TCNBlock(c, 5, 2 ** i) for i in range(nblk)])
        self.pool = AttnPool(c)
        self.out_dim = self.pool.out_dim

    def forward(self, x):                                  # x [N,9,T]
        h = self.blocks(self.stem(x))
        return self.pool(h.transpose(1, 2))


# --------------------------------------------------------------------- (b) GRU
class GRUAttn(nn.Module):
    """Strided conv stem + 2-layer BiGRU + attention pooling."""

    def __init__(self, cin: int = 9, c: int = 64, h: int = 128, nlayer: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, c, 7, stride=4, padding=3),
                                  nn.BatchNorm1d(c), nn.GELU())
        self.rnn = nn.GRU(c, h, num_layers=nlayer, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * h)
        self.out_dim = self.pool.out_dim

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.pool(h)


# ------------------------------------------------------------- (c) Transformer
class TransEnc(nn.Module):
    """Patch the raster into tokens, then a small pre-norm transformer encoder."""

    def __init__(self, cin: int = 9, d: int = 128, nlayer: int = 4, nhead: int = 4,
                 patch: int = 8, maxtok: int = 512):
        super().__init__()
        self.stem = nn.Conv1d(cin, d, patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, maxtok, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.norm = nn.LayerNorm(d)
        self.pool = AttnPool(d)
        self.out_dim = self.pool.out_dim

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h = h + self.pos[:, :h.shape[1]]
        return self.pool(self.norm(self.enc(h)))


# ------------------------------------------------------- (d) cycle-raster 2-D CNN
class Cyc2D(nn.Module):
    """2-D CNN over the (cycle x time-in-cycle) image, 10 channels."""

    def __init__(self, cin: int = 10, c: int = 64):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1),
                                 nn.BatchNorm2d(o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.GELU())
        self.net = nn.Sequential(blk(cin, c, 1), blk(c, 2 * c, (2, 2)),
                                 blk(2 * c, 2 * c, (2, 2)))
        self.pool = AttnPool(2 * c)
        self.out_dim = self.pool.out_dim

    def forward(self, x):                                  # x [N,10,NCYC,KBIN]
        h = self.net(x)                                    # [N,2c,6,8]
        return self.pool(h.flatten(2).transpose(1, 2))


# ------------------------------------------------- (e) hybrid: dilated CNN + GRU
class ConvGRU(nn.Module):
    """Dilated-conv front end (local waveform) + BiGRU (long-range order)."""

    def __init__(self, cin: int = 9, c: int = 96, nblk: int = 4, h: int = 128,
                 nlayer: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, c, 7, stride=2, padding=3),
                                  nn.BatchNorm1d(c), nn.GELU(), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(*[TCNBlock(c, 5, 2 ** i) for i in range(nblk)])
        self.rnn = nn.GRU(c, h, num_layers=nlayer, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * h)
        self.out_dim = self.pool.out_dim

    def forward(self, x):
        hh = self.blocks(self.stem(x)).transpose(1, 2)
        hh, _ = self.rnn(hh)
        return self.pool(hh)


BACKBONES = {"tcn": TCN, "gru": GRUAttn, "transformer": TransEnc, "cyc2d": Cyc2D,
             "convgru": ConvGRU}
IS_2D = {"cyc2d"}


class PairNet(nn.Module):
    def __init__(self, arch: str = "tcn", nfunc: int = 3, **kw):
        super().__init__()
        self.arch = arch
        self.kw = dict(kw)
        self.backbone = BACKBONES[arch](**kw)
        e = self.backbone.out_dim
        self.proj = nn.Sequential(nn.Linear(e, 128), nn.GELU())
        self.phase = nn.Linear(128, 1)
        self.func = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, nfunc))

    def forward(self, x):
        z = self.proj(self.backbone(x))
        return self.phase(z).squeeze(-1), self.func(z), z


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
