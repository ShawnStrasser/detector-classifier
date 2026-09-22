"""Stage 03: dataset / collate for listwise (detector -> candidate phase) training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CACHE, DC_WORK, FOLDS_CSV, LABELS_DEV, FUNCTIONS  # noqa: E402
from neural.raster import (BIN_MS, NCYC, S_STEPS, SignalStore, T_STEPS,  # noqa: E402
                           WIN_MS, render)  # noqa: F401

DATA_MS = 72 * 3600 * 1000
FUNC_IDX = {f: i for i, f in enumerate(FUNCTIONS)}


# ------------------------------------------------------------------ label prep
def load_signal_table(devices: list[str] | None = None, labeled_only: bool = True,
                      drop_failed: bool = False) -> dict:
    """dev -> dict(dets, y_phase (index into cand), y_func, cand)."""
    cand = pd.read_parquet(CACHE / "signal_meta.parquet")[["DeviceId", "cand_phases"]]
    cand_of = {r.DeviceId: [int(p) for p in sorted(r.cand_phases) if 1 <= int(p) <= 16]
               for r in cand.itertuples()}
    lab = pd.read_parquet(LABELS_DEV)
    if devices is not None:
        lab = lab[lab.DeviceId.isin(devices)]
    if drop_failed:
        h = pd.read_parquet(DC_WORK / "atspm" / "detector_health.parquet")
        if "health_flag" in h.columns:
            bad = set(map(tuple, h.loc[h.health_flag == "failed",
                                       ["DeviceId", "Detector"]].values))
            keep = [t not in bad for t in zip(lab.DeviceId, lab.Detector)]
            lab = lab[np.array(keep)]
    out = {}
    for dev, g in lab.groupby("DeviceId"):
        c = cand_of.get(dev, [])
        if len(c) < 2:
            continue
        pos = {p: i for i, p in enumerate(c)}
        dets, yp, yf = [], [], []
        for r in g.itertuples():
            i = pos.get(int(r.Phase), -1)
            if labeled_only and i < 0:
                continue                       # true phase not a candidate: unlearnable
            dets.append(int(r.Detector)); yp.append(i)
            yf.append(FUNC_IDX.get(r.Function, -1))
        if not dets:
            continue
        out[dev] = dict(dets=np.array(dets, np.int32), y_phase=np.array(yp, np.int64),
                        y_func=np.array(yf, np.int64), cand=np.array(c, np.int64))
    return out


def all_detector_table(devices: list[str]) -> dict:
    """Every labeled detector (incl. ones whose phase is not a candidate) for inference."""
    return load_signal_table(devices, labeled_only=False, drop_failed=False)


# ------------------------------------------------------------------ the dataset
# minute lengths cycled across training batches (protocol: train on a MIX of
# window lengths so one model serves 1 min ... 72 h).  30 min is the deployment
# chunk, so it is over-represented.
TRAIN_MINUTES = (30, 30, 30, 30, 15, 10, 5, 2, 1)


class WinDataset(Dataset):
    def __init__(self, table: dict, devs: list[str], nwin: int = 3, seed: int = 0,
                 max_det: int = 16, cycles: bool = False, fixed=None,
                 bs: int = 1, minutes: tuple = TRAIN_MINUTES, T: int | None = None):
        self.table, self.devs, self.nwin = table, devs, nwin
        self.seed, self.max_det, self.cycles = seed, max_det, cycles
        self.fixed = fixed          # list[(dev, w0_ms, det_subset_idx|None)] for eval
        self.epoch = 0
        self.store = None
        self.bs, self.minutes = bs, minutes
        self.T = T                  # eval: fixed number of 1 s steps
        self.order = np.arange(len(self))

    def set_epoch(self, ep: int):
        self.epoch = ep
        if self.fixed is None:
            self.order = np.random.default_rng((self.seed, ep)).permutation(len(self))

    def __len__(self):
        return len(self.fixed) if self.fixed is not None else len(self.devs) * self.nwin

    def _s(self):
        if self.store is None:
            self.store = SignalStore()
        return self.store

    def __getitem__(self, i):
        st = self._s()
        T = self.T if self.T is not None else T_STEPS
        if self.fixed is not None:
            dev, w0, sub = self.fixed[i]
            rec = self.table[dev]
            idx = np.arange(len(rec["dets"])) if sub is None else sub
        else:
            j = int(self.order[i])
            dev = self.devs[j // self.nwin]
            # every item in one batch must share T -> derive it from the batch index
            mins = self.minutes[(i // self.bs + self.epoch) % len(self.minutes)]
            T = int(mins * 60)
            win = T * 1000
            rng = np.random.default_rng((self.seed, self.epoch, j))
            if rng.random() < 0.5:                      # daytime-biased half
                day = rng.integers(0, 3)
                w0 = int(day * 24 * 3600 * 1000 +
                         rng.integers(6 * 3600 * 1000, 20 * 3600 * 1000 - win))
            else:
                w0 = int(rng.integers(0, DATA_MS - win))
            rec = self.table[dev]
            n = len(rec["dets"])
            idx = (rng.permutation(n)[:self.max_det] if n > self.max_det else np.arange(n))
        dets = rec["dets"][idx].tolist()
        r = render(st, dev, w0, dets, T=T, cycles=self.cycles)
        item = dict(det=r[0], ph=r[1], sig=r[2], nact=r[3],
                    y_phase=rec["y_phase"][idx], y_func=rec["y_func"][idx],
                    ncand=len(rec["cand"]), dev=dev, w0=w0,
                    dets=np.asarray(dets, np.int32), cand=rec["cand"])
        if self.cycles:
            item["gidx"], item["clen"] = r[4], r[5]
        return item


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    D = max(len(b["dets"]) for b in batch)
    K = max(b["ncand"] for b in batch)
    T = batch[0]["det"].shape[-1]
    assert all(b["det"].shape[-1] == T for b in batch), "mixed window lengths in a batch"
    det = np.zeros((B, D, 2, T), np.float32)
    ph = np.zeros((B, K, 4, T), np.float32)
    sig = np.zeros((B, 3, T), np.float32)
    nact = np.zeros((B, D), np.float32)
    yp = np.full((B, D), -1, np.int64)
    yf = np.full((B, D), -1, np.int64)
    ncand = np.zeros(B, np.int64)
    dmask = np.zeros((B, D), bool)
    cyc = "gidx" in batch[0]
    gidx = np.zeros((B, K, S_STEPS), np.int32) if cyc else None
    clen = np.zeros((B, K, NCYC), np.float32) if cyc else None
    meta = []
    for b, it in enumerate(batch):
        d, k = len(it["dets"]), it["ncand"]
        det[b, :d] = it["det"]; ph[b, :k] = it["ph"]; sig[b] = it["sig"]
        nact[b, :d] = it["nact"]; yp[b, :d] = it["y_phase"]; yf[b, :d] = it["y_func"]
        ncand[b] = k; dmask[b, :d] = True
        if cyc:
            gidx[b, :k] = it["gidx"]; clen[b, :k] = it["clen"]
        meta.append((it["dev"], it["w0"], it["dets"], it["cand"]))
    out = dict(det=torch.from_numpy(det), ph=torch.from_numpy(ph),
               sig=torch.from_numpy(sig), nact=torch.from_numpy(nact),
               y_phase=torch.from_numpy(yp), y_func=torch.from_numpy(yf),
               ncand=torch.from_numpy(ncand), dmask=torch.from_numpy(dmask), meta=meta)
    if cyc:
        out["gidx"] = torch.from_numpy(gidx); out["clen"] = torch.from_numpy(clen)
    return out


def folds_df() -> pd.DataFrame:
    return pd.read_csv(FOLDS_CSV)


def devs_of(folds: list[int]) -> list[str]:
    f = folds_df()
    return sorted(f.loc[f.fold.isin(folds), "DeviceId"].tolist())
