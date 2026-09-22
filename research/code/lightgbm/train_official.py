"""Task 3 - retrain the v2 phase pipeline (ranker -> joint decoder) on OFFICIAL labels.

Label source
------------
PHASE truth = `dc_work/official/labels_official.parquet` (call_phase, else call_overlap)
everywhere.  The hand `detector-configs.csv` phase labels are used only for the
historical reference row.

Data sources
------------
DEC  Dec-2024, 22-window variant-B mix, `dc_work/features/*{,_B}.parquet` (already built)
STG  Sept-2026 staging, same 22-window mix, `dc_work/official/stg/features/*.parquet`

Variants
--------
A  DEV signals, Dec-2024, HAND labels          (historical reference only)
B  DEV signals, Dec-2024, OFFICIAL labels
C  DEV (Dec-2024) + NEWTRAIN (Sept-2026), OFFICIAL labels

Every variant is scored on the SAME evaluation rows: DEV Dec-2024, 6-fold grouped OOF,
against the OFFICIAL labels.

    python src/official/train_official.py --variant B --seeds 0,1,2
    python src/official/train_official.py --curve
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CONCURRENT_PAIRS, DC_WORK, DEFAULT_PHASE, FOLDS_CSV, N_FOLDS  # noqa: E402
import decode_train as dec  # noqa: E402
from features_partner import PDIFF_FEATS, add_partner_diffs  # noqa: E402

OFFICIAL = DC_WORK / "official"
STG = OFFICIAL / "stg"
FEAT = DC_WORK / "features"
SFEAT = STG / "features"
OUT = OFFICIAL / "train"
PAIR_KEY = ["DeviceId", "Detector", "cand_phase", "win"]
# Anything label-derived is a KEY, never a model input.  `cand_is_overlap` is excluded
# here and added back explicitly by the overlap experiment (it is a candidate *type*,
# not an identity).  Delay / extend / switch_phase / additional-call metadata is never an
# input either: it is not available at inference time for another agency.
KEY_EXCLUDE = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase", "Function",
               "fold", "y", "cyc", "partner_phase", "health_flag", "std_phase",
               "scorable", "scorable_hand", "labelled", "src", "n_add_phases",
               "n_add_overlaps", "delay", "extend", "switch_phase", "target_num",
               "target_cand", "cand_is_overlap", "y_hand", "Phase_hand", "ovl_cand",
               "has_both_phase_and_overlap", "call_phase", "call_overlap", "call_ped",
               "real_dec2024", "real_staging", "n_on_dec2024", "n_on_staging",
               "target_type", "correct", "answered"}

FULL_DEC, FULL_STG = "full72", "full66"
M30 = ["m30_a", "m30_b", "m30_c", "m30_d"]

RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                   learning_rate=0.05, num_leaves=63, min_child_samples=40,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=1.0, n_estimators=1200, n_jobs=10, verbose=-1,
                   label_gain=[0, 1])
BIN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_child_samples=60, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=800, n_jobs=10, verbose=-1)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------- loading
def _official_phase() -> pd.DataFrame:
    o = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    o = o[o.target_type == "phase"]
    return o[["DeviceId", "Detector", "target_num", "n_add_phases",
              "additional_call_phases", "delay", "extend", "switch_phase"]].rename(
        columns={"target_num": "Phase"})


def _hand_phase() -> pd.DataFrame:
    h = pd.read_parquet(DC_WORK / "labels_dev.parquet")[["DeviceId", "Detector", "Phase"]]
    h["Detector"] = h.Detector.astype(int)
    return h


def _load_side(base: list[Path], v2: list[Path], sim: list[Path], tag: str,
               keep: set[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([pd.read_parquet(f) for f in base if f.exists()], ignore_index=True)
    ex = pd.concat([pd.read_parquet(f) for f in v2 if f.exists()], ignore_index=True)
    if keep is not None:
        df = df[df.DeviceId.isin(keep)]
        ex = ex[ex.DeviceId.isin(keep)]
    df = df.merge(ex, on=PAIR_KEY, how="left")
    del ex
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    s = pd.concat([pd.read_parquet(f) for f in sim if f.exists()], ignore_index=True)
    if keep is not None:
        s = s[s.DeviceId.isin(keep)]
    df["src"] = tag
    return df, s


def load_dec(keep: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_side([FEAT / "pair_features_windows.parquet",
                       FEAT / "pair_features_windows_B.parquet"],
                      [FEAT / "pair_features_v2_extra.parquet",
                       FEAT / "pair_features_v2_extra_B.parquet"],
                      [FEAT / "det_similarity.parquet",
                       FEAT / "det_similarity_B.parquet"], "DEC", keep)


def load_stg(keep: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, s = _load_side([SFEAT / "pair_features_stg.parquet"],
                       [SFEAT / "pair_features_v2_stg.parquet"],
                       [SFEAT / "det_similarity_stg.parquet"], "STG", keep)
    df["DeviceId"] = df.DeviceId + "@stg"
    s["DeviceId"] = s.DeviceId + "@stg"
    return df, s


def attach_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["Detector"] = lab.Detector.astype(df.Detector.dtype)
    df = df.merge(lab, on=["DeviceId", "Detector"], how="left")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    return df


def add_scorable(df: pd.DataFrame) -> pd.DataFrame:
    lab = df.Phase.notna()
    has = df.groupby(["DeviceId", "Detector", "win"], sort=False)["y"].transform("max")
    df["scorable"] = (lab & (df.det_n_on >= 1) & (has > 0)).to_numpy()
    return df


FORBIDDEN_SUBSTRINGS = ("Phase", "phase_pred", "target", "label", "scorab", "y_hand",
                        "overlap", "delay", "extend", "switch")
_ALLOWED = {"cand_green_share", "green_share", "green_share_per_claim",
            "ext_corr_late", "ext_green_gain"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns
            if c not in KEY_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    bad = [c for c in cols if c not in _ALLOWED
           and any(s in c for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise ValueError(f"label-derived columns would leak into the features: {bad}")
    return cols


# -------------------------------------------------------------------- training
def _groups(df: pd.DataFrame) -> np.ndarray:
    key = (df["win"].astype(str) + "|" + df.DeviceId.astype(str) + "|" +
           df.Detector.astype(str)).to_numpy()
    _, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


def _fit_rank(tr, va, fc, params, ycol="y"):
    P = dict(params)
    n = P.pop("n_estimators")
    m = lgb.LGBMRanker(n_estimators=n, **P)
    m.fit(tr[fc], tr[ycol], group=_groups(tr), eval_set=[(va[fc], va[ycol])],
          eval_group=[_groups(va)],
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    return m


def to_prob(df, s, T: float = 1.0):
    g = (df["win"].astype(str) + "|" + df.DeviceId.astype(str) + "|" +
         df.Detector.astype(str)).to_numpy()
    d = pd.DataFrame({"g": g, "s": np.asarray(s) / T})
    d["s"] = np.exp(d.s - d.groupby("g")["s"].transform("max"))
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def norm_prob(df, s):
    g = (df["win"].astype(str) + "|" + df.DeviceId.astype(str) + "|" +
         df.Detector.astype(str)).to_numpy()
    d = pd.DataFrame({"g": g, "s": np.clip(np.asarray(s), 1e-9, None)})
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def run_oof(ev: pd.DataFrame, extra_tr: pd.DataFrame | None, fc: list[str],
            seed: int, sim_ev: pd.DataFrame, sim_extra: pd.DataFrame | None,
            keep_train: set[str] | None = None, ycol: str = "y",
            labcol: str = "Phase", save_models: Path | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """6-fold grouped OOF over `ev` (the evaluation frame); `extra_tr` rows are added to
    every fold's training set.  Returns (first-stage prob, decoded prob) aligned to `ev`."""
    rp = dict(RANK_PARAMS, bagging_seed=seed, feature_fraction_seed=seed + 100,
              data_random_seed=seed + 200, seed=seed)
    s1 = np.zeros(len(ev))
    lab_ev = ev[labcol].notna()
    for k in range(N_FOLDS):
        te = (ev.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & lab_ev.to_numpy()
        if keep_train is not None:
            base = base & ev.DeviceId.isin(keep_train).to_numpy()
        tr = ev[base & (ev.fold != inner).to_numpy()]
        va = ev[base & (ev.fold == inner).to_numpy()]
        if extra_tr is not None and len(extra_tr):
            tr = pd.concat([tr, extra_tr], ignore_index=True)
        m = _fit_rank(tr, va, fc, rp, ycol)
        s1[te] = m.predict(ev.loc[te, fc])
    p0 = to_prob(ev, s1)

    # ---- second stage (joint per-signal decoder) --------------------------
    pr = ev[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=ev, sim=sim_ev)
    X = X.merge(ev[["DeviceId", "Detector", "win", "cand_phase", labcol, "fold", ycol]]
                .rename(columns={labcol: "_lab", ycol: "_y"}),
                on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    Xe = None
    if extra_tr is not None and len(extra_tr):
        pr2 = extra_tr[["DeviceId", "Detector", "win", "cand_phase"]].copy()
        # first-stage probabilities for the extra training signals: one model fitted on
        # the evaluation side (they are never scored, only used as decoder training rows)
        m_all = _fit_rank(ev[lab_ev & (ev.fold != 0)], ev[lab_ev & (ev.fold == 0)],
                          fc, rp, ycol)
        pr2["p0"] = to_prob(extra_tr, m_all.predict(extra_tr[fc]))
        Xe = dec.assemble(pr2, pairs=extra_tr, sim=sim_extra)
        Xe = Xe.merge(extra_tr[["DeviceId", "Detector", "win", "cand_phase", labcol, ycol]]
                      .rename(columns={labcol: "_lab", ycol: "_y"}),
                      on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
        Xe["fold"] = -1
    s2 = np.zeros(len(X))
    labX = X._lab.notna()
    for k in range(N_FOLDS):
        te = (X.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & labX.to_numpy()
        if keep_train is not None:
            base = base & X.DeviceId.isin(keep_train).to_numpy()
        tr = X[base & (X.fold != inner).to_numpy()]
        va = X[base & (X.fold == inner).to_numpy()]
        if Xe is not None:
            tr = pd.concat([tr, Xe], ignore_index=True)
        P = dict(BIN_PARAMS, bagging_seed=seed, feature_fraction_seed=seed + 100, seed=seed)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols], tr._y, eval_set=[(va[cols], va._y)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
    X["p2"] = norm_prob(X, s2)
    back = ev[["DeviceId", "Detector", "win", "cand_phase"]].merge(
        X[["DeviceId", "Detector", "win", "cand_phase", "p2"]],
        on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    return p0, back.p2.to_numpy()


# --------------------------------------------------------------------- metrics
def top1(ev: pd.DataFrame, prob: np.ndarray, labcol: str = "Phase",
         scol: str = "scorable") -> pd.DataFrame:
    d = ev[["DeviceId", "Detector", "win", "cand_phase", labcol, "fold", scol]].copy()
    d = d.rename(columns={labcol: "Phase", scol: "scorable"})
    d["p"] = np.asarray(prob, dtype=float)
    d = d[d.scorable]
    d = d.sort_values(["DeviceId", "Detector", "win", "p", "cand_phase"],
                      ascending=[True, True, True, False, True])
    t = d.groupby(["DeviceId", "Detector", "win"], as_index=False, sort=False).first()
    t["ok"] = (t.cand_phase == t.Phase).astype(np.int8)
    return t


def metrics(t: pd.DataFrame, full: str = FULL_DEC) -> dict:
    f = t[t.win == full]
    acc72 = float(f.ok.mean()) if len(f) else np.nan
    m30 = [float(t[t.win == w].ok.mean()) for w in M30 if (t.win == w).any()]
    acc30 = float(np.mean(m30)) if m30 else np.nan
    std = f.Detector.map(DEFAULT_PHASE)
    ns = f[~((std == f.Phase) & (f.Detector <= 40))]
    err = f[f.ok == 0]
    nconc = sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                for a, b in zip(err.Phase, err.cand_phase))
    fold0 = t[(t.fold == 0)]
    f0 = fold0[fold0.win == full]
    m300 = [float(fold0[fold0.win == w].ok.mean()) for w in M30 if (fold0.win == w).any()]
    out = {"primary": (acc72 + acc30) / 2, "acc_full": acc72, "acc_m30": acc30,
           "acc_allwin": float(t.ok.mean()), "n_full": int(len(f)), "n_all": int(len(t)),
           "acc_nonstd_full": float(ns.ok.mean()) if len(ns) else np.nan,
           "n_nonstd_full": int(len(ns)), "n_err_full": int(len(err)),
           "n_err_concurrent": int(nconc),
           "acc_fold0_full": float(f0.ok.mean()) if len(f0) else np.nan,
           "acc_fold0_m30": float(np.mean(m300)) if m300 else np.nan,
           "n_fold0_full": int(len(f0))}
    m = f.p >= 0.9
    out["cov90"] = float(m.mean()) if len(f) else np.nan
    out["acc_at_cov90"] = float(f[m].ok.mean()) if m.any() else np.nan
    per = {}
    for k, g in t.groupby("fold"):
        gf = g[g.win == full]
        gm = [float(g[g.win == w].ok.mean()) for w in M30 if (g.win == w).any()]
        per[int(k)] = round((float(gf.ok.mean()) + float(np.mean(gm))) / 2, 5)
    out["per_fold_primary"] = per
    return out
