"""Stage 11 -- detector **function** v4 on the CURRENT hand-maintained config table.

What is new versus stage 06 (`function_v3.py`)
----------------------------------------------
* **Labels.**  `dc_work/data/labels/detector_config_current.parquet` (8,659 channels /
  466 signals, pulled 2026-09-21) replaces the Dec-2024 + Feb-2025 hand files for the
  FUNCTION target; classes come from `function_label_map_v2.csv`
  (Advance / Presence / Count / Yellow_Red / Other).
* **Two periods.**  Every labelled channel is joined to BOTH Dec-2024 (3 days) and
  Sept-2026 (2.75 days) event data; a channel present in both contributes two samples.
  Folds are grouped by SIGNAL, so both periods of a signal land in the same fold.
* **Phase truth = official timing** (`dc_work/official/labels_official.parquet`) for the
  phase pipeline; the function model's phase-relative / sibling features are always built
  on the phase model's **predicted** phase (out-of-fold, as at inference), never the label.

Hold-outs: the 43 TEST signals and the 143 NEWTEST signals are removed in
`load_config_labels()` and asserted absent in every training / evaluation frame.

    python src/function_v4.py --stage labels
    python src/function_v4.py --stage phase      # official-label ranker -> decoder, 6 folds
    python src/function_v4.py --stage frame      # function design matrix, both periods
    python src/function_v4.py --stage models     # variants a / b / c, seeds, paired tests
    python src/function_v4.py --stage ship       # models/final_candidate/function + review

Everything lands in `dc_work/function_v4/`; no earlier output is overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "official"))

# The staging feature tables live under a second DC_WORK root; keep the main one explicit
# so this module is immune to DC_WORK being re-pointed by a helper script.
DCW = Path(os.environ.get("DC_WORK_MAIN") or (Path.home() / "dc_work"))

N_FOLDS = 6
WORK = DCW / "function_v4"
FEAT = DCW / "features"
SFEAT = DCW / "official" / "stg" / "features"
V4_SFEAT = WORK / "feat_stg"
OFFICIAL = DCW / "official"

CFG_LABELS = DCW / "data" / "labels" / "detector_config_current.parquet"
LABEL_MAP = DCW / "data" / "labels" / "function_label_map_v2.csv"
TEST_CFG = REPO / "data" / "splits" / "test_config.csv"
NEWTEST_CSV = OFFICIAL / "newtest_signals.csv"
NEW_SPLIT = OFFICIAL / "new_signal_split.csv"

LABELS_V4 = WORK / "labels_v4.parquet"
FOLDS_V4 = WORK / "folds_v4.csv"
FRAME = WORK / "funcframe_v4.parquet"

CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CLASSES3 = ["Advance", "Presence", "Count"]
PAIR_KEY = ["DeviceId", "Detector", "cand_phase", "win"]
FULL_WIN = {"dec": "full72", "stg": "full66"}
DUR_GROUPS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wgroup(win: pd.Series) -> np.ndarray:
    w = win.astype(str)
    return np.where(w.isin(["full72", "full66"]), "full", w.str.split("_").str[0])


# ===================================================================== labels
def _holdouts() -> tuple[set, set]:
    test = set(pd.read_csv(TEST_CFG).DeviceId.str.lower())
    newtest = set(pd.read_csv(NEWTEST_CSV).DeviceId.str.lower())
    return test, newtest


def load_config_labels() -> pd.DataFrame:
    """The current config table mapped to the 5 model classes, hold-outs removed."""
    cfg = pd.read_parquet(CFG_LABELS)
    cfg["DeviceId"] = cfg.DeviceId.str.lower()
    cfg["Detector"] = cfg.Detector.astype(int)
    cfg["Function"] = cfg.Function.astype(str)
    mp = pd.read_csv(LABEL_MAP)
    m = dict(zip(mp.raw_key.astype(str), mp.std_function))
    cfg["func5"] = cfg.Function.str.strip().str.lower().map(m)
    bad = cfg[cfg.func5.isna()]
    if len(bad):
        raise ValueError(f"unmapped function strings: {bad.Function.unique()[:20]}")
    test, newtest = _holdouts()
    cfg = cfg[~cfg.DeviceId.isin(test | newtest)]
    cfg = cfg[cfg.Detector.between(1, 64)]
    return cfg[["DeviceId", "Detector", "Phase", "Function", "func5"]].rename(
        columns={"Phase": "cfg_phase"}).reset_index(drop=True)


def assert_no_holdout(df: pd.DataFrame, tag: str) -> None:
    test, newtest = _holdouts()
    d = df.DeviceId.astype(str).str.replace("@stg", "", regex=False).str.lower()
    n_t = int(d.isin(test).sum())
    n_n = int(d.isin(newtest).sum())
    if n_t or n_n:
        raise AssertionError(f"{tag}: {n_t} TEST rows and {n_n} NEWTEST rows present")
    log(f"hold-out assertion OK for {tag} ({len(df):,} rows, {d.nunique()} signals)")


def period_signals() -> tuple[set, set]:
    dec = {p.name.split("=", 1)[1].lower()
           for p in os.scandir(DCW / "cache" / "events") if p.is_dir()}
    stg = set(pd.read_csv(DCW / "official" / "stg" / "signals.csv").DeviceId.str.lower())
    return dec, stg


def stage_labels(args) -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    lab = load_config_labels()
    assert_no_holdout(lab, "config labels")
    dec_sig, stg_sig = period_signals()
    lab["has_dec"] = lab.DeviceId.isin(dec_sig)
    lab["has_stg"] = lab.DeviceId.isin(stg_sig)
    lab = lab[lab.has_dec | lab.has_stg].reset_index(drop=True)

    folds = pd.read_csv(DCW / "folds.csv")
    folds["DeviceId"] = folds.DeviceId.str.lower()
    known = dict(zip(folds.DeviceId, folds.fold))
    extra = sorted(set(lab.DeviceId) - set(known))
    rng = np.random.default_rng(0)
    for d, f in zip(extra, rng.integers(1, N_FOLDS, size=len(extra))):
        known[d] = int(f)
    lab["fold"] = lab.DeviceId.map(known).astype(int)
    pd.DataFrame({"DeviceId": sorted(known), "fold": [known[d] for d in sorted(known)]}
                 ).to_csv(FOLDS_V4, index=False)

    nsp = pd.read_csv(NEW_SPLIT)
    nsp["DeviceId"] = nsp.DeviceId.str.lower()
    newtrain = set(nsp[nsp.split == "NEWTRAIN"].DeviceId)
    dev = set(folds.DeviceId)
    lab["split"] = np.where(lab.DeviceId.isin(dev), "DEV",
                            np.where(lab.DeviceId.isin(newtrain), "NEWTRAIN", "OTHER"))
    lab.to_parquet(LABELS_V4, index=False)

    summ = {
        "n_labels": int(len(lab)), "n_signals": int(lab.DeviceId.nunique()),
        "class_counts": lab.func5.value_counts().to_dict(),
        "by_split": lab.groupby(["split", "func5"]).size().unstack(fill_value=0).to_dict(),
        "n_signals_by_split": lab.groupby("split").DeviceId.nunique().to_dict(),
        "period_coverage": {"dec_only": int((lab.has_dec & ~lab.has_stg).sum()),
                            "stg_only": int((~lab.has_dec & lab.has_stg).sum()),
                            "both": int((lab.has_dec & lab.has_stg).sum())},
        "samples_dec": int(lab.has_dec.sum()), "samples_stg": int(lab.has_stg.sum()),
        "n_new_fold_signals": len(extra),
        "yellow_red_signals": int(lab[lab.func5 == "Yellow_Red"].DeviceId.nunique()),
        "class_counts_dec": lab[lab.has_dec].func5.value_counts().to_dict(),
        "class_counts_stg": lab[lab.has_stg].func5.value_counts().to_dict(),
        "fold_sizes": lab.groupby("fold").DeviceId.nunique().to_dict(),
    }
    json.dump(summ, open(WORK / "labels_v4.json", "w"), indent=1, default=str)
    log(json.dumps(summ, indent=1, default=str))
    log(f"labels done in {time.time() - t0:.0f}s -> {LABELS_V4}")


# ============================================================ feature loading
def _files(period: str, kind: str) -> list[Path]:
    if period == "dec":
        return {"base": [FEAT / "pair_features_windows.parquet",
                         FEAT / "pair_features_windows_B.parquet"],
                "v2": [FEAT / "pair_features_v2_extra.parquet",
                       FEAT / "pair_features_v2_extra_B.parquet"],
                "sim": [FEAT / "det_similarity.parquet",
                        FEAT / "det_similarity_B.parquet"],
                "yr": [FEAT / "func_yr_extra.parquet",
                       FEAT / "func_yr_extra_B.parquet"],
                "lag": [FEAT / "det_lag.parquet", FEAT / "det_lag_B.parquet"]}[kind]
    return {"base": [SFEAT / "pair_features_stg.parquet"],
            "v2": [SFEAT / "pair_features_v2_stg.parquet"],
            "sim": [SFEAT / "det_similarity_stg.parquet"],
            "yr": [V4_SFEAT / "func_yr_extra_stg.parquet"],
            "lag": [V4_SFEAT / "det_lag_stg.parquet"]}[kind]


def _read(files, keep: set[str] | None) -> pd.DataFrame:
    flt = [("DeviceId", "in", list(keep))] if keep else None
    parts = [pd.read_parquet(f, filters=flt) for f in files if f.exists()]
    return pd.concat(parts, ignore_index=True)


def load_pairs(period: str, keep: set[str], with_yr: bool) -> tuple[pd.DataFrame,
                                                                   pd.DataFrame]:
    from features_v2 import PDIFF_FEATS, add_partner_diffs
    df = _read(_files(period, "base"), keep)
    v2 = _read(_files(period, "v2"), keep)
    df = df.merge(v2, on=PAIR_KEY, how="left")
    del v2
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    if with_yr:
        df = df.merge(_read(_files(period, "yr"), keep), on=PAIR_KEY, how="left")
    sim = _read(_files(period, "sim"), keep)
    df["period"] = period
    log(f"[{period}] pair frame {df.shape}, {df.DeviceId.nunique()} signals, "
        f"{df.win.nunique()} windows")
    return df, sim


# ====================================================================== phase
KEY_EXCLUDE = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase", "Function",
               "fold", "y", "cyc", "partner_phase", "health_flag", "std_phase",
               "period", "func5", "cfg_phase", "src", "label_phase", "other_phase",
               "prob", "pred_phase", "split", "has_dec", "has_stg"}
FORBIDDEN = ("Phase", "phase_pred", "target", "label", "scorab", "overlap")
_ALLOWED = {"cand_green_share", "green_share", "green_share_per_claim",
            "ext_corr_late", "ext_green_gain"}

RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                   learning_rate=0.05, num_leaves=63, min_child_samples=40,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=1.0, n_estimators=1200, n_jobs=12, verbose=-1,
                   label_gain=[0, 1])
DEC_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_child_samples=60, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=800, n_jobs=12, verbose=-1)


def phase_feature_cols(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    cols = [c for c in a.columns if c in set(b.columns)
            and c not in KEY_EXCLUDE and not c.startswith("yr_")
            and pd.api.types.is_numeric_dtype(a[c])]
    bad = [c for c in cols if c not in _ALLOWED and any(s in c for s in FORBIDDEN)]
    if bad:
        raise ValueError(f"label-derived columns would leak: {bad}")
    return cols


def official_phase() -> pd.DataFrame:
    o = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    o = o[o.target_type == "phase"][["DeviceId", "Detector", "target_num"]]
    o["DeviceId"] = o.DeviceId.str.lower()
    o["Detector"] = o.Detector.astype(int)
    return o.rename(columns={"target_num": "Phase"})


def _groups(df: pd.DataFrame) -> np.ndarray:
    key = (df["win"].astype(str) + "|" + df.DeviceId.astype(str) + "|" +
           df.Detector.astype(str)).to_numpy()
    _, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


def _norm(df: pd.DataFrame, s: np.ndarray, softmax: bool) -> np.ndarray:
    g = (df["win"].astype(str) + "|" + df.DeviceId.astype(str) + "|" +
         df.Detector.astype(str)).to_numpy()
    d = pd.DataFrame({"g": g, "s": np.asarray(s, dtype=float)})
    if softmax:
        d["s"] = np.exp(d.s - d.groupby("g")["s"].transform("max"))
    else:
        d["s"] = np.clip(d.s, 1e-9, None)
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def stage_phase(args) -> None:
    """Official-label ranker -> joint decoder, 6 folds grouped by signal, fitted on
    Dec-2024 only and applied fold-wise to BOTH periods (a Sept-2026 row of signal X is
    scored by the model that never saw signal X)."""
    import decode_v2 as dc
    import lightgbm as lgb
    t0 = time.time()
    lab = pd.read_parquet(LABELS_V4)
    fold_of = dict(zip(lab.DeviceId, lab.fold))
    keep_dec = set(lab[lab.has_dec].DeviceId)
    keep_stg = set(lab[lab.has_stg].DeviceId)
    off = official_phase()

    frames, sims = {}, {}
    for period, keep in (("dec", keep_dec), ("stg", keep_stg)):
        df, sim = load_pairs(period, keep, with_yr=False)
        assert_no_holdout(df, f"phase pairs {period}")
        df["Detector"] = df.Detector.astype(int)
        df = df.merge(off, on=["DeviceId", "Detector"], how="left")
        df["fold"] = df.DeviceId.map(fold_of).astype(int)
        df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
        df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]
                            ).reset_index(drop=True)
        frames[period], sims[period] = df, sim
    fc = phase_feature_cols(frames["dec"], frames["stg"])
    log(f"phase features {len(fc)}; dec {frames['dec'].shape} stg {frames['stg'].shape}")

    # ---- stage 1: pair ranker
    s1 = {p: np.zeros(len(frames[p])) for p in frames}
    dec = frames["dec"]
    lab_dec = dec.Phase.notna().to_numpy()
    for k in range(N_FOLDS):
        inner = (k + 1) % N_FOLDS
        base = lab_dec & (dec.fold != k).to_numpy()
        tr = dec[base & (dec.fold != inner).to_numpy()]
        va = dec[base & (dec.fold == inner).to_numpy()]
        prm = dict(RANK_PARAMS)
        m = lgb.LGBMRanker(n_estimators=prm.pop("n_estimators"), **prm)
        m.fit(tr[fc], tr.y, group=_groups(tr), eval_set=[(va[fc], va.y)],
              eval_group=[_groups(va)],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        for p, f in frames.items():
            te = (f.fold == k).to_numpy()
            if te.any():
                s1[p][te] = m.predict(f.loc[te, fc])
        log(f"  ranker fold {k}: {m.best_iteration_} trees, {time.time() - t0:.0f}s")

    # ---- stage 2: joint per-signal decoder
    out = {}
    X = {}
    for p, f in frames.items():
        pr = f[["DeviceId", "Detector", "win", "cand_phase"]].copy()
        pr["p0"] = _norm(f, s1[p], softmax=True)
        x = dc.assemble(pr, pairs=f, sim=sims[p])
        x = x.merge(f[["DeviceId", "Detector", "win", "cand_phase", "Phase", "fold", "y"]],
                    on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
        X[p] = x.sort_values(["win", "DeviceId", "Detector", "cand_phase"]
                             ).reset_index(drop=True)
    cols = dc.BASE_COLS + dc.SIM_COLS + dc.ADJ_COLS
    s2 = {p: np.zeros(len(X[p])) for p in X}
    xd = X["dec"]
    labX = xd.Phase.notna().to_numpy()
    for k in range(N_FOLDS):
        inner = (k + 1) % N_FOLDS
        base = labX & (xd.fold != k).to_numpy()
        tr = xd[base & (xd.fold != inner).to_numpy()]
        va = xd[base & (xd.fold == inner).to_numpy()]
        prm = dict(DEC_PARAMS)
        m = lgb.LGBMClassifier(n_estimators=prm.pop("n_estimators"), **prm)
        m.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        for p in X:
            te = (X[p].fold == k).to_numpy()
            if te.any():
                s2[p][te] = m.predict_proba(X[p].loc[te, cols])[:, 1]
        log(f"  decoder fold {k}: {time.time() - t0:.0f}s")

    res = {}
    for p in X:
        x = X[p]
        x["prob"] = _norm(x, s2[p], softmax=False)
        o = x[["DeviceId", "Detector", "win", "cand_phase", "prob"]].copy()
        o["Detector"] = o.Detector.astype(np.int16)
        o["cand_phase"] = o.cand_phase.astype(np.int16)
        o.to_parquet(WORK / f"phase_pred_{p}.parquet", index=False)
        out[p] = o
        # sanity: top-1 accuracy against the official labels (scorable rows only)
        t = x[x.Phase.notna()].sort_values(
            ["DeviceId", "Detector", "win", "prob"], ascending=[1, 1, 1, 0])
        has = t.groupby(["DeviceId", "Detector", "win"])["y"].transform("max")
        t = t[has > 0].groupby(["DeviceId", "Detector", "win"], as_index=False).first()
        full = t.win.isin(["full72", "full66"])
        res[p] = {"n": int(len(t)),
                  "acc_full": round(float((t[full].cand_phase == t[full].Phase).mean()), 4),
                  "acc_allwin": round(float((t.cand_phase == t.Phase).mean()), 4)}
        log(f"[{p}] phase top-1 {json.dumps(res[p])}")
    json.dump(res, open(WORK / "phase_v4_results.json", "w"), indent=1)
    log(f"phase done in {time.time() - t0:.0f}s")


# ====================================================================== frame
def stage_frame(args) -> None:
    import function_v3 as fv3
    t0 = time.time()
    lab = pd.read_parquet(LABELS_V4)
    health = None
    try:
        from train_lgbm_v2 import load_health
        health = load_health()
        health["DeviceId"] = health.DeviceId.str.lower()
    except Exception as exc:                                        # pragma: no cover
        log(f"health table unavailable ({exc})")

    parts = []
    for period in ("dec", "stg"):
        keep = set(lab[lab.has_dec if period == "dec" else lab.has_stg].DeviceId)
        df, _ = load_pairs(period, keep, with_yr=True)
        probs = pd.read_parquet(WORK / f"phase_pred_{period}.parquet")
        probs["Detector"] = probs.Detector.astype(df.Detector.dtype)
        probs["cand_phase"] = probs.cand_phase.astype(df.cand_phase.dtype)
        p = df.merge(probs, on=PAIR_KEY, how="inner")
        del df, probs
        i = p.groupby(["DeviceId", "Detector", "win"], sort=False)["prob"].idxmax()
        top = p.loc[i].copy()
        del p
        top = top.rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})
        top = fv3.add_shape_features(top)
        top = fv3.add_sibling_features(top)
        top = top.reset_index(drop=True)
        lg = _read(_files(period, "lag"), keep)
        orig = fv3._cat
        fv3._cat = lambda files, _lg=lg: _lg        # feed this period's lag table
        try:
            top = fv3.add_lag_features(top)
        finally:
            fv3._cat = orig
        del lg
        top["period"] = period
        top["Detector"] = top.Detector.astype(int)
        top = top.merge(lab[["DeviceId", "Detector", "func5", "Function", "cfg_phase",
                             "fold", "split"]], on=["DeviceId", "Detector"], how="left")
        if health is not None and period == "dec":
            h = health.copy()
            h["Detector"] = h.Detector.astype(int)
            top = top.merge(h, on=["DeviceId", "Detector"], how="left")
            top["health_flag"] = top.health_flag.fillna("unknown")
        else:
            top["health_flag"] = "unknown"
        top["wgroup"] = wgroup(top.win)
        for c in top.columns:
            if top[c].dtype == np.float64:
                top[c] = top[c].astype(np.float32)
        assert_no_holdout(top, f"frame {period}")
        log(f"[{period}] frame {top.shape}; labelled rows "
            f"{int(top.func5.notna().sum()):,}")
        parts.append(top)
    common = [c for c in parts[0].columns if c in set(parts[1].columns)]
    dropped = sorted(set(parts[0].columns) ^ set(parts[1].columns))
    if dropped:
        log(f"columns present in only one period, dropped: {dropped}")
    fr = pd.concat([p[common] for p in parts], ignore_index=True)
    fr.to_parquet(FRAME, index=False)
    log(f"frame {fr.shape} -> {FRAME} in {time.time() - t0:.0f}s")


# ===================================================================== models
NON_FEATURES = KEY_EXCLUDE | {"wgroup", "top_prob_x", "sib_n_x"}

FUNC_PARAMS = dict(objective="multiclass", learning_rate=0.05, num_leaves=31,
                   min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=1.0, n_estimators=1200, n_jobs=12,
                   verbose=-1)


def feat_cols(fr: pd.DataFrame) -> list[str]:
    return [c for c in fr.columns
            if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(fr[c])]


def load_frame() -> pd.DataFrame:
    fr = pd.read_parquet(FRAME)
    fr = fr[fr.func5.notna()].reset_index(drop=True)
    fr["is_full"] = fr.wgroup == "full"
    assert_no_holdout(fr, "function frame")
    return fr


def oof_multiclass(fr, y, cols, train_mask, seed=0, params=FUNC_PARAMS,
                   classes=CLASSES5, sample_weight=None):
    """6-fold OOF grouped by signal.  Rows of fold k are scored by a model fitted on
    `train_mask` rows of the other folds (inner fold = early-stopping validation).
    `train_mask` may exclude a whole period -- every row still gets a prediction."""
    import lightgbm as lgb
    P = np.zeros((len(fr), len(classes)))
    folds = fr.fold.to_numpy()
    ok = train_mask & (fr.health_flag != "failed").to_numpy()
    models, iters = [], []
    for k in range(N_FOLDS):
        te = folds == k
        inner = (k + 1) % N_FOLDS
        trm = ok & (folds != k) & (folds != inner)
        vam = ok & (folds != k) & (folds == inner)
        prm = dict(params, num_class=len(classes), seed=seed, bagging_seed=seed + 1,
                   feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
        n = prm.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **prm)
        m.fit(fr.loc[trm, cols], y[trm],
              sample_weight=None if sample_weight is None else sample_weight[trm],
              eval_set=[(fr.loc[vam, cols], y[vam])], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        if te.any():
            P[te] = m.predict_proba(fr.loc[te, cols])
        models.append(m)
        iters.append(int(m.best_iteration_ or n))
    return P, models, iters


def apply_rule(P: np.ndarray, classes=CLASSES5, delta=0.2, smin=0.6) -> np.ndarray:
    """v3's shipped 'advance presence' rule: report Other when p_adv and p_pres are within
    `delta` of each other and together exceed `smin`."""
    pred = np.array(classes)[P.argmax(1)]
    ia, ip = classes.index("Advance"), classes.index("Presence")
    amb = (np.abs(P[:, ia] - P[:, ip]) < delta) & ((P[:, ia] + P[:, ip]) > smin)
    return np.where(amb, "Other", pred)


def prf(yt, yp, classes) -> dict:
    yt, yp = np.asarray(yt), np.asarray(yp)
    out = {}
    for c in classes:
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"n": int((yt == c).sum()), "precision": round(p, 4),
                  "recall": round(r, 4),
                  "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}
    out["macro_f1"] = round(float(np.mean([out[c]["f1"] for c in classes])), 4)
    return out


def prec_at_recall(yt: np.ndarray, score: np.ndarray, target: float = 0.7) -> dict:
    from sklearn.metrics import precision_recall_curve
    if yt.sum() == 0:
        return {}
    pr, rc, th = precision_recall_curve(yt, score)
    ok = rc >= target
    if not ok.any():
        return {"precision": None, "recall": None, "threshold": None}
    i = int(np.argmax(pr[ok]))
    idx = np.flatnonzero(ok)[i]
    return {"precision": round(float(pr[idx]), 4), "recall": round(float(rc[idx]), 4),
            "threshold": round(float(th[min(idx, len(th) - 1)]), 4)}


def score_variant(fr: pd.DataFrame, P: np.ndarray, mask: np.ndarray, tag: str,
                  classes=CLASSES5) -> dict:
    """Full metric block on the rows selected by `mask`."""
    from sklearn.metrics import average_precision_score
    f = fr[mask].reset_index(drop=True)
    Q = P[mask]
    yt = f.func5.to_numpy()
    raw = np.array(classes)[Q.argmax(1)]
    rul = apply_rule(Q, classes)
    conf = Q.max(1)
    wg = f.wgroup.to_numpy()
    apc = np.isin(yt, CLASSES3)
    d = {"tag": tag, "n": int(len(f)), "n_signals": int(f.DeviceId.nunique()),
         "acc5_allwin": round(float((yt == raw).mean()), 4),
         "acc5_allwin_rule": round(float((yt == rul).mean()), 4),
         "accAPC_allwin": round(float((yt[apc] == raw[apc]).mean()), 4)}
    by, byapc, byrule = {}, {}, {}
    for g in DUR_GROUPS:
        m = wg == g
        if not m.any():
            continue
        by[g] = round(float((yt[m] == raw[m]).mean()), 4)
        byrule[g] = round(float((yt[m] == rul[m]).mean()), 4)
        ma = m & apc
        byapc[g] = round(float((yt[ma] == raw[ma]).mean()), 4) if ma.any() else None
    d["acc5_by_duration"] = by
    d["acc5_by_duration_rule"] = byrule
    d["accAPC_by_duration"] = byapc
    d["n_by_duration"] = {g: int((wg == g).sum()) for g in DUR_GROUPS if (wg == g).any()}
    full = wg == "full"
    if full.any():
        d["per_class_full"] = prf(yt[full], raw[full], classes)
        d["per_class_full_rule"] = prf(yt[full], rul[full], classes)
        d["confusion_full"] = pd.crosstab(pd.Series(yt[full], name="true"),
                                          pd.Series(raw[full], name="pred")).to_dict()
        yyr = (yt == "Yellow_Red").astype(int)
        jy = classes.index("Yellow_Red")
        if yyr[full].sum():
            d["yellow_red"] = {
                "n_pos_full": int(yyr[full].sum()),
                "n_signals": int(f[full & (yyr == 1)].DeviceId.nunique()),
                "ap_full": round(float(average_precision_score(yyr[full], Q[full, jy])), 4),
                "ap_allwin": round(float(average_precision_score(yyr, Q[:, jy])), 4),
                "at_recall70_full": prec_at_recall(yyr[full], Q[full, jy], 0.7),
                "at_recall70_allwin": prec_at_recall(yyr, Q[:, jy], 0.7),
                "argmax_full": {
                    "precision": round(float((yt[full][raw[full] == "Yellow_Red"] ==
                                              "Yellow_Red").mean()), 4)
                    if (raw[full] == "Yellow_Red").any() else None,
                    "recall": round(float((raw[full][yyr[full] == 1] ==
                                           "Yellow_Red").mean()), 4)},
                "recall_by_duration": {g: round(float((raw[(wg == g) & (yyr == 1)] ==
                                                       "Yellow_Red").mean()), 4)
                                       for g in DUR_GROUPS
                                       if ((wg == g) & (yyr == 1)).any()}}
        # coverage vs accuracy on the confidence of the reported answer
        cov = []
        for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            m = full & (conf >= th)
            cov.append({"th": th, "coverage": round(float(m.sum() / max(full.sum(), 1)), 4),
                        "acc5": round(float((yt[m] == raw[m]).mean()), 4) if m.any() else None})
        d["coverage_full"] = cov
    d["other"] = {
        "precision": round(float((yt[raw == "Other"] == "Other").mean()), 4)
        if (raw == "Other").any() else None,
        "recall": round(float((raw[yt == "Other"] == "Other").mean()), 4),
        "precision_rule": round(float((yt[rul == "Other"] == "Other").mean()), 4)
        if (rul == "Other").any() else None,
        "recall_rule": round(float((rul[yt == "Other"] == "Other").mean()), 4)}
    per_fold = {}
    for k, g in f.groupby("fold"):
        gm = (g.wgroup == "full").to_numpy()
        yy, pp = g.func5.to_numpy(), raw[f.fold.to_numpy() == k]
        per_fold[int(k)] = round(float((yy[gm] == pp[gm]).mean()), 4) if gm.any() else None
    d["per_fold_full"] = per_fold
    return d


def paired_bootstrap(fr: pd.DataFrame, mask: np.ndarray, ok_a: np.ndarray,
                     ok_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over SIGNALS of (accuracy_b - accuracy_a) on the masked rows."""
    sub = fr[mask]
    sig = sub.DeviceId.str.replace("@stg", "", regex=False).to_numpy()
    a, b = ok_a[mask].astype(float), ok_b[mask].astype(float)
    uniq, inv = np.unique(sig, return_inverse=True)
    n = len(uniq)
    sa = np.bincount(inv, weights=a, minlength=n)
    sb = np.bincount(inv, weights=b, minlength=n)
    cnt = np.bincount(inv, minlength=n)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (sb[idx].sum() - sa[idx].sum()) / cnt[idx].sum()
    return {"delta": round(float(b.mean() - a.mean()), 5),
            "lo90": round(float(np.quantile(diffs, 0.05)), 5),
            "hi90": round(float(np.quantile(diffs, 0.95)), 5),
            "p_better": round(float((diffs > 0).mean()), 3)}


def stage_models(args) -> None:
    t0 = time.time()
    fr = load_frame()
    cols = feat_cols(fr)
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    is_dec = (fr.period == "dec").to_numpy()
    is_stg = ~is_dec
    full = fr.is_full.to_numpy()
    log(f"function frame {fr.shape}, {len(cols)} features, "
        f"{fr.DeviceId.nunique()} signal-periods; dec {is_dec.sum():,} stg {is_stg.sum():,}")
    res = {"n_rows": int(len(fr)), "n_features": len(cols),
           "class_counts_full_dec": fr[full & is_dec].func5.value_counts().to_dict(),
           "class_counts_full_stg": fr[full & is_stg].func5.value_counts().to_dict()}

    # ------------------------------------------------ (b) new labels, Dec-2024 only
    Pb, Mb, itb = oof_multiclass(fr, y, cols, is_dec, seed=0)
    res["B_dec_only"] = {"dec": score_variant(fr, Pb, is_dec, "B/dec-OOF"),
                         "stg": score_variant(fr, Pb, is_stg, "B/cross-period Sept-2026"),
                         "best_iters": itb}
    log("variant B done " + json.dumps({k: res["B_dec_only"][k]["acc5_by_duration"]
                                        for k in ("dec", "stg")}))

    # ------------------------------------------------ (c) new labels, both periods
    Pc, Mc, itc = oof_multiclass(fr, y, cols, np.ones(len(fr), bool), seed=0)
    res["C_both"] = {"dec": score_variant(fr, Pc, is_dec, "C/dec-OOF"),
                     "stg": score_variant(fr, Pc, is_stg, "C/stg-OOF"),
                     "all": score_variant(fr, Pc, np.ones(len(fr), bool), "C/all"),
                     "best_iters": itc}
    log("variant C done " + json.dumps({k: res["C_both"][k]["acc5_by_duration"]
                                        for k in ("dec", "stg")}))

    # ------------------------------------------------ (a) v3 as shipped, new labels
    v3 = pd.read_parquet(DCW / "function_v3" / "function_oof_v3_bywindow.parquet")
    v3["DeviceId"] = v3.DeviceId.str.lower()
    v3["Detector"] = v3.Detector.astype(int)
    pc3 = [f"p_{c.lower()}" for c in CLASSES5]
    v3 = v3[["DeviceId", "Detector", "win"] + pc3].rename(
        columns={c: f"v3_{c}" for c in pc3})
    dd = fr[is_dec].reset_index(drop=True)
    dd["_row"] = np.flatnonzero(is_dec)
    j = dd.merge(v3, on=["DeviceId", "Detector", "win"], how="inner")
    rows = j._row.to_numpy()
    Pa_sub = j[[f"v3_p_{c.lower()}" for c in CLASSES5]].to_numpy()
    common = np.zeros(len(fr), bool)
    common[rows] = True
    Pa = np.zeros_like(Pc)
    Pa[rows] = Pa_sub
    res["A_v3_oof_rescored"] = score_variant(fr, Pa, common, "A/v3 OOF vs new labels")
    res["A_common_rows"] = {"n": int(common.sum()),
                            "n_signals": int(fr[common].DeviceId.nunique())}
    res["B_on_common"] = score_variant(fr, Pb, common, "B on A's rows")
    res["C_on_common"] = score_variant(fr, Pc, common, "C on A's rows")

    # ------------------------------------------------ paired tests
    yt = fr.func5.to_numpy()
    okA = (np.array(CLASSES5)[Pa.argmax(1)] == yt)
    okB = (np.array(CLASSES5)[Pb.argmax(1)] == yt)
    okC = (np.array(CLASSES5)[Pc.argmax(1)] == yt)
    res["paired"] = {
        "A_vs_B_common_full": paired_bootstrap(fr, common & full, okA, okB),
        "A_vs_B_common_all": paired_bootstrap(fr, common, okA, okB),
        "B_vs_C_dec_full": paired_bootstrap(fr, is_dec & full, okB, okC),
        "B_vs_C_dec_all": paired_bootstrap(fr, is_dec, okB, okC),
        "B_vs_C_stg_full": paired_bootstrap(fr, is_stg & full, okB, okC),
        "B_vs_C_stg_all": paired_bootstrap(fr, is_stg, okB, okC)}
    log("paired " + json.dumps(res["paired"], indent=1))

    # ------------------------------------------------ seed repeats (noise floor)
    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    reps = []
    for s in seeds[1:]:
        Ps, _, _ = oof_multiclass(fr, y, cols, np.ones(len(fr), bool), seed=s)
        reps.append({"seed": s,
                     "acc5_full_dec": score_variant(fr, Ps, is_dec & full,
                                                    f"C seed{s}")["acc5_allwin"],
                     "acc5_full_stg": score_variant(fr, Ps, is_stg & full,
                                                    f"C seed{s}")["acc5_allwin"],
                     "acc5_all": float((np.array(CLASSES5)[Ps.argmax(1)] == yt).mean())})
        log(f"seed {s}: {json.dumps(reps[-1])}")
    base = {"seed": seeds[0],
            "acc5_full_dec": res["C_both"]["dec"]["acc5_by_duration"].get("full"),
            "acc5_full_stg": res["C_both"]["stg"]["acc5_by_duration"].get("full"),
            "acc5_all": float(okC.mean())}
    allr = [base] + reps
    res["seed_repeats"] = {
        "runs": allr,
        "sd_acc5_full_dec": round(float(np.std([r["acc5_full_dec"] for r in allr],
                                               ddof=1)), 5) if len(allr) > 1 else None,
        "sd_acc5_all": round(float(np.std([r["acc5_all"] for r in allr], ddof=1)), 5)
        if len(allr) > 1 else None}

    # ------------------------------------------------ advance-presence rule sweep
    ap_raw = fr.Function.str.strip().str.lower().eq("advance presence").to_numpy()
    sweep = []
    for delta in (0.0, 0.15, 0.2, 0.25, 0.3, 0.35):
        pr = apply_rule(Pc, delta=delta) if delta else np.array(CLASSES5)[Pc.argmax(1)]
        sweep.append({
            "delta": delta,
            "acc5_full": round(float((yt[full] == pr[full]).mean()), 4),
            "accAPC_full": round(float((yt[full & np.isin(yt, CLASSES3)] ==
                                        pr[full & np.isin(yt, CLASSES3)]).mean()), 4),
            "other_recall": round(float((pr[yt == "Other"] == "Other").mean()), 4),
            "other_precision": round(float((yt[pr == "Other"] == "Other").mean()), 4),
            "advpres_to_other": round(float((pr[ap_raw] == "Other").mean()), 4)
            if ap_raw.any() else None})
    res["advance_presence_rule"] = sweep
    res["n_advance_presence_rows"] = int(ap_raw.sum())

    # ------------------------------------------------ raw-label breakdown + importance
    predC = np.array(CLASSES5)[Pc.argmax(1)]
    o = fr[full & ~fr.func5.isin(CLASSES3 + ["Yellow_Red"])].copy()
    o["pred"] = predC[full & ~fr.func5.isin(CLASSES3 + ["Yellow_Red"]).to_numpy()]
    res["other_by_raw_label"] = o.groupby("Function").agg(
        n=("Function", "size"),
        to_other=("pred", lambda s: round(float((s == "Other").mean()), 3))
    ).sort_values("n", ascending=False).head(15).reset_index().to_dict("records")
    imp = pd.Series(np.mean([m.booster_.feature_importance("gain") for m in Mc], axis=0),
                    index=cols).sort_values(ascending=False)
    imp.to_csv(WORK / "feature_importance_v4.csv", header=["gain"])
    res["top_features"] = imp.head(25).round(1).to_dict()

    out = fr[["DeviceId", "Detector", "period", "win", "wgroup", "fold", "func5",
              "Function", "split", "health_flag", "det_n_on", "top_prob",
              "pred_phase"]].copy()
    out[[f"p_{c.lower()}" for c in CLASSES5]] = Pc
    out["pred5"] = predC
    out["pred5_rule"] = apply_rule(Pc)
    out[[f"b_{c.lower()}" for c in CLASSES5]] = Pb
    out.to_parquet(WORK / "function_oof_v4_bywindow.parquet", index=False)
    out[out.is_full if "is_full" in out else (out.wgroup == "full")].to_parquet(
        WORK / "function_oof_v4_full.parquet", index=False)
    json.dump(res, open(WORK / "function_v4_results.json", "w"), indent=1, default=str)
    log(f"models done in {time.time() - t0:.0f}s")


# ================================================== Yellow_Red: cost and head
CLASSES4 = ["Advance", "Presence", "Count", "Other"]


def stage_yr(args) -> None:
    """What does carrying the Yellow_Red class cost the three real classes, and is the
    class shippable at precision >= .85 / recall >= .7 with 337 (293 usable) labels?"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    fr = load_frame()
    cols = feat_cols(fr)
    full = fr.is_full.to_numpy()
    yt5 = fr.func5.to_numpy()
    yt4 = np.where(yt5 == "Yellow_Red", "Other", yt5)
    apc = np.isin(yt5, CLASSES3)
    allm = np.ones(len(fr), bool)

    P4, _, it4 = oof_multiclass(fr, pd.Series(yt4).map(
        {c: i for i, c in enumerate(CLASSES4)}).to_numpy(), cols, allm, seed=0,
        classes=CLASSES4)
    pred4 = np.array(CLASSES4)[P4.argmax(1)]
    oof = pd.read_parquet(WORK / "function_oof_v4_bywindow.parquet")
    P5 = oof[[f"p_{c.lower()}" for c in CLASSES5]].to_numpy()
    pred5 = np.array(CLASSES5)[P5.argmax(1)]
    pred5to4 = np.where(pred5 == "Yellow_Red", "Other", pred5)

    def acc(m, a, b):
        return round(float((a[m] == b[m]).mean()), 4)

    res = {"n_yr_labels": int((yt5 == "Yellow_Red").sum()),
           "n_yr_detectors": int(fr[yt5 == "Yellow_Red"][["DeviceId", "Detector"]]
                                 .drop_duplicates().shape[0]),
           "n_yr_signals": int(fr[yt5 == "Yellow_Red"].DeviceId.nunique()),
           "cost_to_APC": {
               "acc_APC_full_5class": acc(full & apc, yt5, pred5),
               "acc_APC_full_4class": acc(full & apc, yt5, pred4),
               "acc_APC_allwin_5class": acc(apc, yt5, pred5),
               "acc_APC_allwin_4class": acc(apc, yt5, pred4),
               "acc4_full_5class": acc(full, yt4, pred5to4),
               "acc4_full_4class": acc(full, yt4, pred4),
               "paired_APC_full": paired_bootstrap(
                   fr, full & apc, yt5 == pred5, yt5 == pred4),
               "best_iters_4class": it4},
           }
    yyr = (yt5 == "Yellow_Red").astype(int)
    jy = CLASSES5.index("Yellow_Red")
    vc = np.isin(yt5, ["Yellow_Red", "Count"])
    for tag, m in (("full", full), ("allwin", allm)):
        if not yyr[m].sum():
            continue
        res[f"head_{tag}"] = {
            "auc": round(float(roc_auc_score(yyr[m], P5[m, jy])), 4),
            "ap": round(float(average_precision_score(yyr[m], P5[m, jy])), 4),
            "auc_vs_count": round(float(roc_auc_score(yyr[m & vc], P5[m & vc, jy])), 4),
            "ap_vs_count": round(float(average_precision_score(yyr[m & vc],
                                                              P5[m & vc, jy])), 4),
            "at_recall70": prec_at_recall(yyr[m], P5[m, jy], 0.7),
            "operating_points": [
                {"th": th, "n_pred": int(((P5[:, jy] >= th) & m).sum()),
                 "precision": round(float(yyr[(P5[:, jy] >= th) & m].mean()), 4)
                 if ((P5[:, jy] >= th) & m).any() else None,
                 "recall": round(float(((P5[m, jy] >= th)[yyr[m] == 1]).mean()), 4)}
                for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]}
    res["by_period"] = {}
    for p in ("dec", "stg"):
        m = full & (fr.period == p).to_numpy()
        if yyr[m].sum():
            res["by_period"][p] = {
                "n_pos": int(yyr[m].sum()),
                "ap": round(float(average_precision_score(yyr[m], P5[m, jy])), 4),
                "at_recall70": prec_at_recall(yyr[m], P5[m, jy], 0.7)}
    res["recall_by_duration_argmax"] = {
        g: round(float((pred5[(fr.wgroup == g).to_numpy() & (yyr == 1)] ==
                        "Yellow_Red").mean()), 4)
        for g in DUR_GROUPS if ((fr.wgroup == g).to_numpy() & (yyr == 1)).any()}
    json.dump(res, open(WORK / "yellow_red_v4.json", "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))
    log(f"yellow_red done in {time.time() - t0:.0f}s")


# ======================================================================= ship
CAND_DIR = REPO / "models" / "final_candidate" / "function"


def stage_ship(args) -> None:
    import lightgbm as lgb
    from lgbm_numpy import NumpyBooster
    t0 = time.time()
    fr = load_frame()
    cols = feat_cols(fr)
    res = json.load(open(WORK / "function_v4_results.json"))
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    ok = (fr.health_flag != "failed").to_numpy()
    prm = dict(FUNC_PARAMS, num_class=len(CLASSES5), seed=0, bagging_seed=1,
               feature_fraction_seed=2, data_random_seed=3)
    prm["n_estimators"] = int(np.mean(res["C_both"]["best_iters"]))
    m = lgb.LGBMClassifier(**prm)
    m.fit(fr.loc[ok, cols], y[ok])
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    txt = CAND_DIR / "function_lgbm_v4.txt"
    m.booster_.save_model(str(txt))

    # ---- exact-reproduction check with the numpy-only backend
    sub = fr.loc[ok, cols].iloc[:4000]
    a = m.booster_.predict(sub)
    b = NumpyBooster(str(txt)).predict(sub)
    maxdiff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    log(f"numpy backend max abs diff = {maxdiff:.3e}")

    yr = res["C_both"]["all"].get("yellow_red", {})
    meta = {
        "classes": CLASSES5, "features": cols,
        "n_estimators": prm["n_estimators"], "params": {k: v for k, v in prm.items()
                                                        if k != "n_estimators"},
        "sibling_features": __import__("function_v3").SIB_FEATS,
        "label_source": "dc_work/data/labels/detector_config_current.parquet "
                        "(pulled 2026-09-21) via function_label_map_v2.csv",
        "trained_on": {"windows": sorted(fr.win.unique().tolist()),
                       "periods": ["Dec-2024 (3 d)", "Sept-2026 staging (2.75 d)"],
                       "n_rows": int(ok.sum()),
                       "n_signals": int(fr.DeviceId.nunique()),
                       "n_detectors": int(fr[["DeviceId", "Detector"]]
                                          .drop_duplicates().shape[0])},
        "excluded_signals": {"TEST": 43, "NEWTEST": 143},
        # v3 shipped this rule ON; re-validated on the current config table it now COSTS
        # 0.16 pt of 5-class and 0.84 pt of Advance/Presence/Count accuracy, because the
        # explicitly trained Other class already absorbs 61 % of 'advance presence'
        # channels on its own.  Shipped OFF; the parameters stay for an operator who
        # prefers Other recall (.613 -> .650) over Other precision (.666 -> .608).
        "advance_presence_rule": {"enabled": False, "delta": 0.2, "sum_min": 0.6,
                                  "note": "report Other when p_advance and p_presence are "
                                          "within delta and together exceed sum_min",
                                  "measured": res.get("advance_presence_rule")},
        "other_threshold_note": "Other is an explicitly trained class; no threshold needed.",
        "yellow_red_operating_point": dict(yr.get("at_recall70_full", {}),
                                           note="p_yellow_red >= threshold gives this "
                                                "precision/recall; plain argmax gives "
                                                "P .89 / R .67 at the long window"),
        "numpy_backend_max_abs_diff": maxdiff,
        "metrics": {"C_dec": res["C_both"]["dec"], "C_stg": res["C_both"]["stg"],
                    "C_all": res["C_both"]["all"]},
    }
    json.dump(meta, open(CAND_DIR / "function_lgbm_v4.json", "w"), indent=1, default=str)
    log(f"wrote {CAND_DIR} ({len(cols)} features, {prm['n_estimators']} trees, "
        f"{txt.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


# ===================================================================== review
def _why(r) -> str:
    def g(k, d="?"):
        v = r.get(k)
        return d if v is None or (isinstance(v, float) and np.isnan(v)) else v
    bits = [f"median ON {g('det_dur_med'):.1f} s" if not np.isnan(r.get("det_dur_med", np.nan)) else "",
            f"occupied in {g('f_occ_red'):.2f} of red" if not np.isnan(r.get("f_occ_red", np.nan)) else "",
            f"queue {g('queue_occ_pre_green'):.1f} s before green" if not np.isnan(r.get("queue_occ_pre_green", np.nan)) else "",
            f"calls the phase {g('call43_fwd_035'):.2f}" if not np.isnan(r.get("call43_fwd_035", np.nan)) else "",
            f"lag to siblings {g('lagsib_best_lag'):+.1f} s" if not np.isnan(r.get("lagsib_best_lag", np.nan)) else "",
            f"fires in yellow/red-clear on {g('yr_hit_yr'):.2f} of cycles" if not np.isnan(r.get("yr_hit_yr", np.nan)) else ""]
    bits = [b for b in bits if b]
    return (f"model {r['pred5']} p={r['conf']:.2f} vs config '{r['Function']}' "
            f"({r['func5']}): " + ", ".join(bits) + ".")


def stage_review(args) -> None:
    t0 = time.time()
    oof = pd.read_parquet(WORK / "function_oof_v4_bywindow.parquet")
    oof = oof[oof.wgroup == "full"].copy()
    pcols = [f"p_{c.lower()}" for c in CLASSES5]
    oof["conf"] = oof[pcols].max(1)
    fr = pd.read_parquet(FRAME, columns=["DeviceId", "Detector", "period", "win",
                                         "det_dur_med", "f_occ_red",
                                         "queue_occ_pre_green", "call43_fwd_035",
                                         "lagsib_best_lag", "yr_hit_yr"])
    fr = fr[fr.win.isin(["full72", "full66"])].drop(columns=["win"])
    oof = oof.merge(fr, on=["DeviceId", "Detector", "period"], how="left")
    # one row per detector: prefer the Sept-2026 sample (contemporaneous with the config)
    oof["ord"] = np.where(oof.period == "stg", 0, 1)
    oof = oof.sort_values(["DeviceId", "Detector", "ord"]).groupby(
        ["DeviceId", "Detector"], as_index=False).first()

    bad = oof[(oof.pred5 != oof.func5) & (oof.conf >= 0.8)].copy()
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")[
        ["DeviceId", "DeviceName", "Detector", "target", "description"]]
    off["DeviceId"] = off.DeviceId.str.lower()
    off["Detector"] = off.Detector.astype(int)
    bad = bad.merge(off, on=["DeviceId", "Detector"], how="left")
    bad["why"] = bad.apply(_why, axis=1)
    out = bad.sort_values("conf", ascending=False).head(300)[
        ["DeviceName", "DeviceId", "Detector", "Function", "func5", "pred5", "conf",
         "target", "period", "det_n_on", "description", "why"]].rename(
        columns={"Function": "config_function_raw", "func5": "config_function_mapped",
                 "pred5": "model_function", "conf": "prob", "target": "official_phase",
                 "description": "channel_description"})
    dest = REPO / "review" / "function_vs_config_disagreements.csv"
    out.to_csv(dest, index=False)
    log(f"{len(bad)} confident disagreements ({len(out)} written) -> {dest}")

    # ---- did the newer config table "fix" the old review list?
    # the full stage-06 list (387 confident function disagreements); the repo copy
    # `review/review_function.csv` is the curated 150-row subset of the same rows.
    old = pd.read_csv(DCW / "function_v3" / "review_function.csv")
    old = old[old.reason.str.startswith("confident function disagreement")].copy()
    old["DeviceId"] = old.DeviceId.str.lower()
    old["Detector"] = old.Detector.astype(int)
    new = load_config_labels()[["DeviceId", "Detector", "func5", "Function"]].rename(
        columns={"func5": "new_func5", "Function": "new_raw"})
    j = old.merge(new, on=["DeviceId", "Detector"], how="left")
    have = j.new_func5.notna()
    summ = {
        "old_confident_function_disagreements": int(len(old)),
        "still_in_current_config": int(have.sum()),
        "new_label_agrees_with_model": int((j.loc[have, "new_func5"] ==
                                            j.loc[have, "pred5"]).sum()),
        "new_label_unchanged": int((j.loc[have, "new_func5"] ==
                                    j.loc[have, "func5"]).sum()),
        "new_label_changed_to_third_class": int(((j.loc[have, "new_func5"] !=
                                                  j.loc[have, "func5"]) &
                                                 (j.loc[have, "new_func5"] !=
                                                  j.loc[have, "pred5"])).sum()),
        "dropped_from_config": int((~have).sum()),
        "n_new_confident_disagreements": int(len(bad)),
        "n_new_disagreement_signals": int(bad.DeviceId.nunique()),
        "new_disagreement_confusion": pd.crosstab(bad.func5, bad.pred5).to_dict(),
    }
    j[have].to_csv(WORK / "old_review_resolved.csv", index=False)
    json.dump(summ, open(WORK / "review_v4.json", "w"), indent=1, default=str)
    log(json.dumps(summ, indent=1, default=str))
    log(f"review done in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["labels", "phase", "frame", "models", "yr", "ship",
                             "review"])
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
