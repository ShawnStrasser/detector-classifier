"""Stage 03: on-the-fly rasterisation of a (signal, time window) into tensors.

Representation (chosen in results/03_neural.md):
  fixed-rate raster, 1 s bins, 30 min window -> T = 1800 steps, 9 channels per
  (detector, candidate phase) PAIR.  The 9 channels are assembled on the GPU from
  three much smaller blocks so the same phase/ signal rasters are shared by every
  detector at that signal:

    detector block  [2, T] : occupancy fraction, ON-onset count
    phase block     [4, T] : p green / yellow / red-clearance / call fractions
    signal block    [3, T] : total #phases green, total #phases called, coordinated flag

  pair channels = [det_occ, det_onrate, p_green, p_yellow, p_redclr, p_call,
                   other_green_count, other_call_share, is_coord]

Nothing here sees a phase number or a detector channel number: the candidate axis is
just an unordered set, exactly as the protocol requires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK  # noqa: E402

SIGDIR = DC_WORK / "neural" / "sig"
T0 = pd.Timestamp("2024-12-02 00:00:00")
BIN_MS = 1000              # 1 s bins
WIN_MS = 30 * 60 * 1000    # 30 min base window
T_STEPS = WIN_MS // BIN_MS  # 1800
N_CH = 9
DATA_END_MS = 72 * 3600 * 1000


# ------------------------------------------------------------------ primitives
def _cover(on: np.ndarray, off: np.ndarray, w0: int, w1: int, T: int, bw: int) -> np.ndarray:
    """Fraction of each bin covered by the (sorted, disjoint) intervals."""
    out = np.zeros(T, dtype=np.float32)
    if on.size == 0:
        return out
    i0 = int(np.searchsorted(off, w0, "right"))
    i1 = int(np.searchsorted(on, w1, "left"))
    if i1 <= i0:
        return out
    a = (on[i0:i1].astype(np.float64) - w0) / bw
    b = (off[i0:i1].astype(np.float64) - w0) / bw
    np.clip(a, 0.0, T, out=a)
    np.clip(b, 0.0, T, out=b)
    cum = np.concatenate(([0.0], np.cumsum(b - a)))
    x = np.arange(T + 1, dtype=np.float64)
    j = np.searchsorted(a, x, "right")          # intervals starting at or before x
    G = np.zeros(T + 1)
    m = j >= 1
    jm = j[m] - 1
    G[m] = cum[jm] + np.minimum(x[m], b[jm]) - a[jm]
    out[:] = np.diff(G)
    return np.clip(out, 0.0, 1.0)


def _onrate(on: np.ndarray, w0: int, w1: int, T: int, bw: int) -> np.ndarray:
    if on.size == 0:
        return np.zeros(T, dtype=np.float32)
    i0 = int(np.searchsorted(on, w0, "left"))
    i1 = int(np.searchsorted(on, w1, "left"))
    if i1 <= i0:
        return np.zeros(T, dtype=np.float32)
    idx = ((on[i0:i1].astype(np.int64) - w0) // bw).astype(np.int64)
    return np.bincount(idx, minlength=T)[:T].astype(np.float32)


# ------------------------------------------------------------------ signal store
class SignalStore:
    """Lazy per-signal loader for the npz interval cache."""

    def __init__(self, sigdir: Path = SIGDIR, max_cached: int = 64):
        self.dir = Path(sigdir)
        self._c: dict[str, dict] = {}
        self.max_cached = max_cached

    def get(self, dev: str) -> dict:
        z = self._c.get(dev)
        if z is None:
            if len(self._c) >= self.max_cached:
                self._c.pop(next(iter(self._c)))
            with np.load(self.dir / f"{dev}.npz") as f:
                z = {k: f[k] for k in f.files}
            z["_chpos"] = {int(c): i for i, c in enumerate(z["det_ch"])}
            self._c[dev] = z
        return z

    def cand(self, dev: str) -> np.ndarray:
        return self.get(dev)["cand"]


NCYC, KBIN = 24, 32
S_STEPS = NCYC * KBIN


def _cycle_index(gon: np.ndarray, w0: int, w1: int, T: int, bw: int):
    """Gather index into the 1 s raster that re-slices it into p's last NCYC cycles."""
    gi = np.zeros(S_STEPS, dtype=np.int32)
    cl = np.zeros(NCYC, dtype=np.float32)
    i0 = int(np.searchsorted(gon, w0, "left"))
    i1 = int(np.searchsorted(gon, w1, "left"))
    st = ((gon[i0:i1].astype(np.int64) - w0) // bw)
    if st.size < 2:                                   # no complete cycle in window
        st = np.linspace(0, T, NCYC + 1).astype(np.int64)
    st = st[-(NCYC + 1):]
    nc = st.size - 1
    frac = (np.arange(KBIN) + 0.5) / KBIN
    for j in range(nc):
        a, b = int(st[j]), int(st[j + 1])
        row = NCYC - nc + j
        gi[row * KBIN:(row + 1) * KBIN] = np.clip(a + (b - a) * frac, 0, T - 1).astype(np.int32)
        cl[row] = np.log1p(max(b - a, 0) * bw / 1000.0) / 5.0
    if nc < NCYC:                                     # pad rows repeat the first cycle
        gi[:(NCYC - nc) * KBIN] = gi[(NCYC - nc) * KBIN:(NCYC - nc + 1) * KBIN].repeat(1)[0]
    return gi, cl


def render(store: SignalStore, dev: str, w0_ms: int, dets: list[int],
           T: int = T_STEPS, bw: int = BIN_MS, cycles: bool = False):
    """Return (det[len(dets),2,T], ph[K,4,T], sig[3,T], nact[len(dets)]) float32."""
    z = store.get(dev)
    w1 = w0_ms + T * bw
    cand = z["cand"]
    K = len(cand)

    ph = np.zeros((K, 4, T), dtype=np.float32)
    for tag, row in (("g", 0), ("y", 1), ("r", 2), ("c", 3)):
        ptr, on, off = z[tag + "_ptr"], z[tag + "_on"], z[tag + "_off"]
        for k in range(K):
            s, e = int(ptr[k]), int(ptr[k + 1])
            if e > s:
                ph[k, row] = _cover(on[s:e], off[s:e], w0_ms, w1, T, bw)

    sig = np.zeros((3, T), dtype=np.float32)
    sig[0] = ph[:, 0].sum(0)                 # number of phases green
    sig[1] = ph[:, 3].sum(0)                 # number of phases with a call
    sig[2] = _cover(z["co_on"], z["co_off"], w0_ms, w1, T, bw)

    D = len(dets)
    det = np.zeros((D, 2, T), dtype=np.float32)
    nact = np.zeros(D, dtype=np.float32)
    chpos, ptr, on, off = z["_chpos"], z["det_ptr"], z["det_on"], z["det_off"]
    for i, ch in enumerate(dets):
        k = chpos.get(int(ch))
        if k is None:
            continue
        s, e = int(ptr[k]), int(ptr[k + 1])
        if e <= s:
            continue
        det[i, 0] = _cover(on[s:e], off[s:e], w0_ms, w1, T, bw)
        det[i, 1] = _onrate(on[s:e], w0_ms, w1, T, bw)
        nact[i] = det[i, 1].sum()
    det[:, 1] = np.clip(det[:, 1], 0, 4) / 2.0        # keep channels O(1)
    if not cycles:
        return det, ph, sig, nact
    gi = np.zeros((K, S_STEPS), dtype=np.int32)
    cl = np.zeros((K, NCYC), dtype=np.float32)
    gptr, gon = z["g_ptr"], z["g_on"]
    for k in range(K):
        s, e = int(gptr[k]), int(gptr[k + 1])
        gi[k], cl[k] = _cycle_index(gon[s:e], w0_ms, w1, T, bw)
    return det, ph, sig, nact, gi, cl


# --------------------------------------------------------------- GPU assembly
def assemble_flat(det, ph, sig, ncand, bi, di, ki) -> torch.Tensor:
    """Build only the pairs we need: [N, 9, T].

    det [B,D,2,T], ph [B,K,4,T], sig [B,3,T]; bi/di/ki are flat pair indices.
    """
    d = det[bi, di]                                   # [N,2,T]
    p = ph[bi, ki]                                    # [N,4,T]
    s = sig[bi]                                       # [N,3,T]
    og = (s[:, 0:1] - p[:, 0:1]) / 2.0                # other phases green (count/2)
    den = (ncand[bi].clamp(min=2) - 1).float().view(-1, 1, 1)
    oc = (s[:, 1:2] - p[:, 3:4]) / den                # share of other phases called
    return torch.cat([d, p, og, oc, s[:, 2:3]], dim=1)


# --------------------------------------------------------------- 2-D variant
def assemble_cycles_flat(det, ph, sig, ncand, gidx, clen, bi, di, ki) -> torch.Tensor:
    """Cycle-aligned images [N,10,NCYC,KBIN], gathered out of the same 1 s raster.

    Each candidate phase has its own gather index (its own begin-green grid), so the
    same detector trace is re-cut once per candidate -- the "time-in-cycle raster"
    representation at no extra rasterisation cost.
    """
    gi = gidx[bi, ki].long()                          # [N,S]
    N, S = gi.shape
    d = torch.gather(det[bi, di], 2, gi[:, None, :].expand(N, 2, S))
    p = torch.gather(ph[bi, ki], 2, gi[:, None, :].expand(N, 4, S))
    s = torch.gather(sig[bi], 2, gi[:, None, :].expand(N, 3, S))
    og = (s[:, 0:1] - p[:, 0:1]) / 2.0
    den = (ncand[bi].clamp(min=2) - 1).float().view(-1, 1, 1)
    oc = (s[:, 1:2] - p[:, 3:4]) / den
    cl = clen[bi, ki][:, None, :, None].expand(N, 1, NCYC, KBIN).reshape(N, 1, S)
    x = torch.cat([d, p, og, oc, s[:, 2:3], cl], dim=1)        # [N,10,S]
    return x.reshape(N, 10, NCYC, KBIN)
