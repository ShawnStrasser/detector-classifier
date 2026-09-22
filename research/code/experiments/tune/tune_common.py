"""Stage 07 (`07_tuning_trees`) -- shared loaders, metrics and significance tests.

Everything here is read-only with respect to earlier stages: feature tables are reused,
never rebuilt, and all artefacts land under `dc_work/tune/`.

Evaluation contract used by every stage-07 experiment
-----------------------------------------------------
* rows come from the beta's **variant-B 22-window mix** (4x5 min, 4x10 min, 4x30 min,
  3x1 h, 2x3 h, 2x6 h, 2x24 h, 72 h);
* a labelled detector-window is **scorable** iff it actuated at least once in the window
  (`det_n_on >= 1`) and its labelled phase is one of that window's candidate phases
  (protocol "unscorable excluded" rule, `>=1 actuation` variant -- the footnote definition
  the champion's .9818 / .9624 numbers are quoted under);
* `acc72`  = top-1 accuracy over scorable rows of window `full72`;
* `acc30`  = mean of the top-1 accuracies of the four 30-minute windows;
* **primary metric = (acc72 + acc30) / 2**;
* secondary: non-standard-detector accuracy at 72 h, coverage/accuracy at p >= 0.9.

Hyper-parameter search NEVER touches fold 0 (the 38-signal 2025 hold-out): searches run
5-fold over folds 1-5 and fold 0 is scored once, afterwards, for every adopted change.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", message=".*ndcg_eval_at.*")
warnings.filterwarnings("ignore", message=".*experimental feature.*")
try:
    from lightgbm.basic import LGBMDeprecationWarning
    warnings.filterwarnings("ignore", category=LGBMDeprecationWarning)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path

from common import DC_WORK, FEATURES, LABELS_DEV, N_FOLDS, REPO  # noqa: E402

TUNE = DC_WORK / "tune"
TUNE.mkdir(parents=True, exist_ok=True)
(TUNE / "logs").mkdir(exist_ok=True)

FUNC_V3 = DC_WORK / "function_v3"
FOLDS_V3 = FUNC_V3 / "folds_v3.csv"

N_JOBS = int(os.environ.get("DC_TUNE_THREADS", "7"))

M30 = ["m30_a", "m30_b", "m30_c", "m30_d"]
FULL = "full72"
# windows used inside the hyper-parameter search (keeps a trial ~40 % of the cost of the
# full 22-window fit while covering every duration band that the metric reads)
SEARCH_WINS = [FULL, "m30_a", "m30_b", "m30_c", "m30_d", "m5_a", "m10_a", "h1_a", "h6_a"]

BASE_FILES = [FEATURES / "pair_features_windows.parquet",
              FEATURES / "pair_features_windows_B.parquet"]
V2_FILES = [FEATURES / "pair_features_v2_extra.parquet",
            FEATURES / "pair_features_v2_extra_B.parquet"]
SIM_FILES = [FEATURES / "det_similarity.parquet",
             FEATURES / "det_similarity_B.parquet"]

PAIR_KEY = ["DeviceId", "Detector", "cand_phase", "win"]
KEY_EXCLUDE = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase", "Function",
               "fold", "y", "cyc", "partner_phase", "health_flag", "std_phase",
               "scorable", "labelled"}

PHASE_CACHE = TUNE / "phase_pairs.parquet"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ pair frame
def build_phase_pairs() -> pd.DataFrame:
    from features_partner import PDIFF_FEATS, add_partner_diffs
    from train_lgbm_v2 import load_health
    df = pd.concat([pd.read_parquet(f) for f in BASE_FILES], ignore_index=True)
    v2 = pd.concat([pd.read_parquet(f) for f in V2_FILES], ignore_index=True)
    df = df.merge(v2, on=PAIR_KEY, how="left")
    del v2
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    folds = pd.read_csv(FOLDS_V3)
    df = df.merge(folds, on="DeviceId", how="inner")
    df = df.merge(pd.read_parquet(LABELS_DEV)[["DeviceId", "Detector", "Phase"]],
                  on=["DeviceId", "Detector"], how="left")
    h = load_health()
    h["Detector"] = h.Detector.astype(df.Detector.dtype)
    df = df.merge(h, on=["DeviceId", "Detector"], how="left")
    df["health_flag"] = df.health_flag.fillna("unknown")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def load_phase_pairs(rebuild: bool = False) -> pd.DataFrame:
    if PHASE_CACHE.exists() and not rebuild:
        return pd.read_parquet(PHASE_CACHE)
    log("building the 22-window pair frame (cached once)")
    df = build_phase_pairs()
    df.to_parquet(PHASE_CACHE, index=False)
    log(f"cached {df.shape} -> {PHASE_CACHE}")
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in KEY_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------- scoring
def add_scorable(df: pd.DataFrame) -> pd.DataFrame:
    """`scorable` = labelled, >=1 actuation in the window, label phase is a candidate."""
    lab = df.Phase.notna()
    has_lab_cand = df.groupby(["DeviceId", "Detector", "win"], sort=False)["y"].transform("max")
    df["scorable"] = (lab & (df.det_n_on >= 1) & (has_lab_cand > 0)).to_numpy()
    df["labelled"] = lab.to_numpy()
    return df


def top1_table(keys: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """One row per scorable (DeviceId, Detector, win): was the top-1 candidate right?"""
    d = keys.copy()
    d["p"] = np.asarray(prob, dtype=np.float64)
    d = d[d.scorable]
    # deterministic tie-break on the lowest phase number, as in src/evaluate.py
    d = d.sort_values(["DeviceId", "Detector", "win", "p", "cand_phase"],
                      ascending=[True, True, True, False, True])
    t = d.groupby(["DeviceId", "Detector", "win"], as_index=False, sort=False).first()
    t["ok"] = (t.cand_phase == t.Phase).astype(np.int8)
    return t


def _nonstd(t: pd.DataFrame) -> pd.Series:
    from common import DEFAULT_PHASE
    std = t.Detector.map(DEFAULT_PHASE)
    return ~((std == t.Phase) & (t.Detector <= 40))


def phase_metrics(t: pd.DataFrame) -> dict:
    """`t` = output of top1_table."""
    f = t[t.win == FULL]
    acc72 = float(f.ok.mean()) if len(f) else np.nan
    per_m30 = [float(t[t.win == w].ok.mean()) for w in M30 if (t.win == w).any()]
    acc30 = float(np.mean(per_m30)) if per_m30 else np.nan
    ns = f[_nonstd(f)]
    out = {"primary": (acc72 + acc30) / 2, "acc72": acc72, "acc30": acc30,
           "acc_allwin": float(t.ok.mean()),
           "nonstd72": float(ns.ok.mean()) if len(ns) else np.nan,
           "n72": int(len(f)), "n_nonstd72": int(len(ns)), "n_all": int(len(t))}
    m = f.p >= 0.9
    out["cov90_72"] = float(m.mean()) if len(f) else np.nan
    out["acc_at_cov90_72"] = float(f[m].ok.mean()) if m.any() else np.nan
    return out


def per_fold_primary(t: pd.DataFrame, folds: pd.DataFrame) -> pd.Series:
    d = t if "fold" in t.columns else t.merge(folds, on="DeviceId", how="left")
    out = {}
    for k, g in d.groupby("fold"):
        f = g[g.win == FULL]
        m30 = [float(g[g.win == w].ok.mean()) for w in M30 if (g.win == w).any()]
        out[int(k)] = (float(f.ok.mean()) + float(np.mean(m30))) / 2
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------- significance
def paired_bootstrap(t_a: pd.DataFrame, t_b: pd.DataFrame, n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.10) -> dict:
    """Bootstrap over SIGNALS of the primary-metric difference (b - a).

    Both tables must cover the same detector-windows; they are aligned on the key so the
    comparison is paired.  Returns the point difference and a 90 % interval.
    """
    key = ["DeviceId", "Detector", "win"]
    m = t_a[key + ["ok"]].merge(t_b[key + ["ok"]], on=key, suffixes=("_a", "_b"))
    m["is72"] = m.win == FULL
    m["is30"] = m.win.isin(M30)
    sub = m[m.is72 | m.is30]
    devs = sub.DeviceId.unique()
    idx = {d: i for i, d in enumerate(devs)}
    dev_i = sub.DeviceId.map(idx).to_numpy()
    oka, okb = sub.ok_a.to_numpy(float), sub.ok_b.to_numpy(float)
    w72 = sub.is72.to_numpy()
    # per-signal sufficient statistics for both halves of the metric
    def _stat(counts_ok_a, counts_ok_b, counts_n, mask):
        pass
    n_dev = len(devs)
    s72_a = np.bincount(dev_i[w72], oka[w72], n_dev)
    s72_b = np.bincount(dev_i[w72], okb[w72], n_dev)
    n72 = np.bincount(dev_i[w72], None, n_dev)
    w30 = ~w72
    s30_a = np.bincount(dev_i[w30], oka[w30], n_dev)
    s30_b = np.bincount(dev_i[w30], okb[w30], n_dev)
    n30 = np.bincount(dev_i[w30], None, n_dev)

    def prim(sa, sb, na, nb):
        return 0.5 * (sa / max(na, 1e-9) + sb / max(nb, 1e-9))

    point = (prim(s72_b.sum(), s30_b.sum(), n72.sum(), n30.sum())
             - prim(s72_a.sum(), s30_a.sum(), n72.sum(), n30.sum()))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_dev, n_dev)
        A = prim(s72_a[pick].sum(), s30_a[pick].sum(), n72[pick].sum(), n30[pick].sum())
        B = prim(s72_b[pick].sum(), s30_b[pick].sum(), n72[pick].sum(), n30[pick].sum())
        diffs[i] = B - A
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(point), "lo90": float(lo), "hi90": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_signals": int(n_dev)}


def fold_agreement(t_a: pd.DataFrame, t_b: pd.DataFrame, folds: pd.DataFrame) -> dict:
    a = per_fold_primary(t_a, folds)
    b = per_fold_primary(t_b, folds)
    d = (b - a).dropna()
    return {"per_fold_diff": {int(k): round(float(v), 5) for k, v in d.items()},
            "n_folds_better": int((d > 0).sum()), "n_folds": int(len(d)),
            "mean": float(d.mean()), "sd_a": float(a.std(ddof=0)),
            "sd_b": float(b.std(ddof=0))}


def verdict(boot: dict, agree: dict) -> str:
    if boot["excludes_zero"] or agree["n_folds_better"] >= 5:
        return "REAL" if boot["diff"] > 0 else "REAL (worse)"
    return "not significant"


def dump(obj, name: str) -> None:
    p = TUNE / name
    json.dump(obj, open(p, "w"), indent=1, default=str)
    log(f"wrote {p}")
