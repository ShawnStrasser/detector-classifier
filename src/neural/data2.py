"""Stage 13: dataset for the REFIT GRU (official labels, two data periods).

Differences from the stage-03 `data.py`:

* labels are the OFFICIAL controller timing (`labels_official.parquet`), for every
  labelled channel that actuates in the period -- not only the hand-labelled ones;
* two periods live side by side, keyed `"<period>|<DeviceId>"`:
      dec  375 DEV signals,      Dec-2024, 72 h
      stg  334 NEWTRAIN signals, Sept-2026, 16:15 Fri .. 10:23 Mon
  exactly the pool `src/official/fit_final_v1.py` fits the shipped LightGBM on, with the
  same signal->fold map, so held-out rows line up one-for-one;
* training windows are an explicit per-epoch PLAN (dev, w0, T); a batch sampler groups
  items of equal length and spends a constant time budget per step, so short windows get
  bigger batches instead of idling the GPU;
* the function head is not trained here (phase loss only).

Nothing here sees a phase number or a detector channel number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, DC_WORK, FOLDS_CSV  # noqa: E402
from neural.raster import SignalStore, render  # noqa: E402

NEURAL = DC_WORK / "neural"
OFFICIAL = DC_WORK / "official"
STG_CACHE = OFFICIAL / "stg" / "cache"
H = 3600 * 1000

PERIODS = {
    # lo/hi are the millisecond bounds of real data relative to the period's T0
    "dec": dict(sigdir=NEURAL / "sig", t0="2024-12-02 00:00:00",
                meta=CACHE / "signal_meta.parquet", real="real_dec2024",
                lo=0, hi=72 * H, full="full72"),
    "stg": dict(sigdir=NEURAL / "sig_stg", t0="2026-09-18 00:00:00",
                meta=STG_CACHE / "signal_meta.parquet", real="real_staging",
                lo=int(16.25 * H), hi=int((72 + 10.38) * H), full="full66"),
}

# window lengths cycled across training batches: 5 min .. 60 min, 30 min over-represented
# (30 min is the deployment chunk and the length where the blend pays off).
TRAIN_MINUTES = (5, 10, 15, 20, 30, 30, 30)
# constant compute budget per optimiser step, in signal-minutes
BATCH_MINUTES = 120


# ------------------------------------------------------------------ split maps
def dev_folds() -> dict[str, int]:
    f = pd.read_csv(FOLDS_CSV)
    return dict(zip(f.DeviceId.str.lower(), f.fold.astype(int)))


def newtrain_folds() -> dict[str, int]:
    """The fold map `fit_final_v1.build_pool` drew (numpy default_rng(0) over the sorted
    NEWTRAIN ids that carry official labels).  Cached as a csv the first time."""
    p = OFFICIAL / "newtrain_folds.csv"
    if p.exists():
        d = pd.read_csv(p)
        return dict(zip(d.DeviceId.str.lower(), d.fold.astype(int)))
    q = pd.read_parquet(OFFICIAL / "final_v1" / "oof_bywindow.parquet",
                        columns=["DeviceId", "src"])
    ids = np.array(sorted(q.loc[q.src == "STG", "DeviceId"].unique()))
    fold = np.random.default_rng(0).integers(0, 6, len(ids))
    plain = [s.replace("@stg", "").lower() for s in ids]
    pd.DataFrame({"DeviceId": plain, "fold": fold}).to_csv(p, index=False)
    return dict(zip(plain, fold.tolist()))


def training_signals() -> pd.DataFrame:
    """DeviceId, period, fold -- the 701 signals the shipped LightGBM was fitted on."""
    rows = [(d, "dec", f) for d, f in dev_folds().items()]
    rows += [(d, "stg", f) for d, f in newtrain_folds().items()]
    df = pd.DataFrame(rows, columns=["DeviceId", "period", "fold"])
    df["key"] = df.period + "|" + df.DeviceId
    return df.sort_values("key").reset_index(drop=True)


# ---------------------------------------------------------------------- labels
def official_labels() -> pd.DataFrame:
    o = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    o = o[o.target_type == "phase"].copy()
    o["DeviceId"] = o.DeviceId.str.lower()
    o["Detector"] = o.Detector.astype(int)
    return o[["DeviceId", "Detector", "target_num", "real_dec2024", "real_staging"]] \
        .rename(columns={"target_num": "Phase"})


def _cand_map(period: str) -> dict[str, np.ndarray]:
    m = pd.read_parquet(PERIODS[period]["meta"])[["DeviceId", "cand_phases"]]
    out = {}
    for r in m.itertuples():
        c = r.cand_phases
        if c is None or (isinstance(c, float)) or c is pd.NA:
            c = []
        out[str(r.DeviceId).lower()] = np.array(
            sorted(int(p) for p in c if 1 <= int(p) <= 16), dtype=np.int64)
    return out


def load_table(sigs: pd.DataFrame, labelled_only: bool = True) -> dict:
    """key -> dict(dets, y_phase (index into cand, -1 if the phase never greens), cand,
    period, dev).  `labelled_only` drops channels whose phase is not a candidate."""
    lab = official_labels()
    out: dict[str, dict] = {}
    for period, g in sigs.groupby("period"):
        cmap = _cand_map(period)
        real = PERIODS[period]["real"]
        sub = lab[lab[real]]
        keep = set(g.DeviceId)
        sub = sub[sub.DeviceId.isin(keep)]
        for dev, gg in sub.groupby("DeviceId"):
            cand = cmap.get(dev, np.zeros(0, np.int64))
            if len(cand) < 2:
                continue
            pos = {int(p): i for i, p in enumerate(cand)}
            dets, yp = [], []
            for det, ph in zip(gg.Detector.to_numpy(), gg.Phase.to_numpy()):
                i = pos.get(int(ph), -1)
                if labelled_only and i < 0:
                    continue
                dets.append(int(det)); yp.append(i)
            if not dets:
                continue
            out[f"{period}|{dev}"] = dict(
                dets=np.asarray(dets, np.int32), y_phase=np.asarray(yp, np.int64),
                cand=np.asarray(cand, np.int64), period=period, dev=dev)
    return out


# --------------------------------------------------------------------- dataset
class Stores:
    """One lazy npz store per period."""

    def __init__(self, max_cached: int = 48):
        self._s = {p: SignalStore(PERIODS[p]["sigdir"], max_cached=max_cached)
                   for p in PERIODS}

    def get(self, period: str) -> SignalStore:
        return self._s[period]


def sample_plan(table: dict, keys: list[str], nwin: int, seed: int, epoch: int,
                minutes: tuple = TRAIN_MINUTES) -> list[tuple]:
    """(key, w0_ms, T_steps) for one epoch: `nwin` windows per signal, lengths cycled."""
    rng = np.random.default_rng((seed, epoch))
    plan = []
    for j, k in enumerate(keys):
        per = table[k]["period"]
        lo, hi = PERIODS[per]["lo"], PERIODS[per]["hi"]
        for w in range(nwin):
            mins = int(minutes[(j * nwin + w + epoch) % len(minutes)])
            win = mins * 60 * 1000
            if rng.random() < 0.5:                       # daytime-biased half
                day = int(rng.integers(0, max(1, (hi - lo) // (24 * H) + 1)))
                a = lo + day * 24 * H
                a = max(lo, min(a + 6 * H, hi - win))
                b = max(a + 1, min(a + 14 * H, hi - win))
                w0 = int(rng.integers(a, b))
            else:
                w0 = int(rng.integers(lo, max(lo + 1, hi - win)))
            plan.append((k, w0, mins * 60))
    rng.shuffle(plan)
    return plan


def batches_of(plan: list[tuple], budget_minutes: int = BATCH_MINUTES,
               max_bs: int = 8, shuffle_seed: int | None = None) -> list[list[int]]:
    """Group plan indices of equal length into batches of ~constant compute.

    `max_bs` bounds the number of (detector, candidate) pairs in one step, which is what
    actually bounds GPU memory on the 8 GB card."""
    by_T: dict[int, list[int]] = {}
    for i, (_, _, T) in enumerate(plan):
        by_T.setdefault(T, []).append(i)
    out = []
    for T, idx in by_T.items():
        bs = int(np.clip(round(budget_minutes / (T / 60.0)), 1, max_bs))
        for i in range(0, len(idx), bs):
            out.append(idx[i:i + bs])
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(out)
    return out


class PlanDataset(Dataset):
    def __init__(self, table: dict, plan: list[tuple], stores: Stores | None = None,
                 max_det: int = 16, seed: int = 0):
        self.table, self.plan = table, plan
        self.max_det, self.seed = max_det, seed
        self.stores = stores

    def __len__(self):
        return len(self.plan)

    def _st(self):
        if self.stores is None:
            self.stores = Stores()
        return self.stores

    def __getitem__(self, i):
        key, w0, T = self.plan[i]
        rec = self.table[key]
        n = len(rec["dets"])
        if self.max_det and n > self.max_det:
            idx = np.random.default_rng((self.seed, i, w0)).permutation(n)[:self.max_det]
            idx.sort()
        else:
            idx = np.arange(n)
        dets = rec["dets"][idx].tolist()
        st = self._st().get(rec["period"])
        det, ph, sig, nact = render(st, rec["dev"], int(w0), dets, T=int(T))
        return dict(det=det, ph=ph, sig=sig, nact=nact,
                    y_phase=rec["y_phase"][idx], ncand=len(rec["cand"]),
                    key=key, dev=rec["dev"], w0=int(w0),
                    dets=np.asarray(dets, np.int32), cand=rec["cand"])


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
    ncand = np.zeros(B, np.int64)
    dmask = np.zeros((B, D), bool)
    meta = []
    for b, it in enumerate(batch):
        d, k = len(it["dets"]), it["ncand"]
        det[b, :d] = it["det"]; ph[b, :k] = it["ph"]; sig[b] = it["sig"]
        nact[b, :d] = it["nact"]; yp[b, :d] = it["y_phase"]
        ncand[b] = k; dmask[b, :d] = True
        meta.append((it["key"], it["dev"], it["w0"], it["dets"], it["cand"]))
    return dict(det=torch.from_numpy(det), ph=torch.from_numpy(ph),
                sig=torch.from_numpy(sig), nact=torch.from_numpy(nact),
                y_phase=torch.from_numpy(yp), ncand=torch.from_numpy(ncand),
                dmask=torch.from_numpy(dmask), meta=meta)


class FixedDataset(PlanDataset):
    """Evaluation: every labelled detector of the signal, no subsampling."""

    def __init__(self, table: dict, plan: list[tuple], stores: Stores | None = None):
        super().__init__(table, plan, stores, max_det=0)
