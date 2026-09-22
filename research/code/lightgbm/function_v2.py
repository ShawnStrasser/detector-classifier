"""Stage 04 step 5: detector-function model v2 (Advance / Presence / Count / Other).

Improvements over stage 01:
  * the pair features of the *predicted* phase now include the v2 families
    (occupancy by colour, per-cycle first-ON latency, queue-release timing, discharge burst);
  * explicit occupancy-by-colour ratios and ON-duration shape;
  * **sibling features** -- every other detector that the phase model assigns to the same
    phase at the same signal.  An Advance detector actuates several seconds *before* the
    Presence detector of its own phase, and a Count detector carries only one lane's volume,
    so each detector is described relative to its siblings (difference and within-sibling
    rank of first-ON latency, ON duration, occupancy, volume).  Phase-anonymous: the sibling
    set is defined by the model's own output, not by a phase number;
  * probability calibration (temperature scaling fitted out-of-fold);
  * an explicit "Other" rule = max class probability below a threshold.

    python src/train_lgbm_v2.py --stage function
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:                                   # only needed for training, not for inference
    import lightgbm as lgb
except ImportError:
    lgb = None
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CACHE, FUNCTIONS, MODELS, N_FOLDS, PREDS  # noqa: E402

SIB_FEATS = ["first_on_med", "first_on_le1", "first_on_le3", "dtg_b0", "dtg_h0",
             "det_dur_med", "det_dur_q90", "det_occ_frac", "det_on_per_hour",
             "queue_occ_pre_green", "f_on_red_last10", "release_frac_long",
             "burst_rate_g4", "det_frac_long", "f_on_green", "dur_mean_green"]

FUNC_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05,
                   num_leaves=31, min_child_samples=40, feature_fraction=0.7,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   n_estimators=1200, n_jobs=12, verbose=-1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_sibling_features(top: pd.DataFrame) -> pd.DataFrame:
    """Relative position of a detector among the detectors sharing its predicted phase."""
    key = ["DeviceId", "win", "pred_phase"]
    g = top.groupby(key, sort=False)
    new = {"sib_n": g["Detector"].transform("size").astype("float32")}
    for f in SIB_FEATS:
        if f not in top.columns:
            continue
        med = g[f].transform("median")
        new[f"{f}__sibdiff"] = top[f] - med
        new[f"{f}__sibrank"] = g[f].rank(pct=True, method="average")
        mn = g[f].transform("min")
        new[f"{f}__sibmin"] = top[f] - mn
    return pd.concat([top, pd.DataFrame(new, index=top.index)], axis=1)


def add_shape_features(top: pd.DataFrame) -> pd.DataFrame:
    d = {}
    eps = 1e-6
    if {"f_occ_green", "f_occ_red"} <= set(top.columns):
        d["occ_green_red_ratio"] = top.f_occ_green / (top.f_occ_red + eps)
    if {"dur_mean_green", "dur_mean_red"} <= set(top.columns):
        d["dur_green_red_ratio"] = top.dur_mean_green / (top.dur_mean_red + eps)
    if {"det_dur_q90", "det_dur_med"} <= set(top.columns):
        d["dur_q90_over_med"] = top.det_dur_q90 / (top.det_dur_med + eps)
    if {"det_occ_frac", "det_on_per_hour"} <= set(top.columns):
        d["occ_per_actuation"] = top.det_occ_frac * 3600.0 / (top.det_on_per_hour + eps)
    return pd.concat([top, pd.DataFrame(d, index=top.index)], axis=1)


def _temperature(P: np.ndarray, y: np.ndarray) -> float:
    """1-D temperature scaling on log-probabilities (minimises NLL)."""
    L = np.log(np.clip(P, 1e-9, 1))
    best, bt = np.inf, 1.0
    for T in np.linspace(0.5, 3.0, 26):
        Z = L / T
        Z = Z - Z.max(1, keepdims=True)
        Q = np.exp(Z)
        Q /= Q.sum(1, keepdims=True)
        nll = -np.log(np.clip(Q[np.arange(len(y)), y], 1e-9, 1)).mean()
        if nll < best:
            best, bt = nll, T
    return float(bt)


def _apply_T(P: np.ndarray, T: float) -> np.ndarray:
    Z = np.log(np.clip(P, 1e-9, 1)) / T
    Z = Z - Z.max(1, keepdims=True)
    Q = np.exp(Z)
    return Q / Q.sum(1, keepdims=True)


def _ece(P: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = P.max(1)
    corr = (P.argmax(1) == y).astype(float)
    e, n = 0.0, len(y)
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def build_frame(pairs: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """One row per (detector, window) = the features of the model's own top-1 phase."""
    p = pairs.merge(probs, on=["DeviceId", "Detector", "win", "cand_phase"], how="inner")
    i = p.groupby(["DeviceId", "Detector", "win"], sort=False)["prob"].idxmax()
    top = p.loc[i].copy()
    top = top.rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})
    top = add_shape_features(top)
    top = add_sibling_features(top)
    return top.reset_index(drop=True)


def main_stage() -> None:
    from train_lgbm_v2 import KEY_EXCLUDE, load_pairs_v2
    t0 = time.time()
    pairs = load_pairs_v2(with_v2=True)
    probs = pd.read_parquet(PREDS / "phase_oof_v2_decoded_bywindow.parquet")
    probs["Detector"] = probs.Detector.astype(pairs.Detector.dtype)
    probs["cand_phase"] = probs.cand_phase.astype(pairs.cand_phase.dtype)
    fr = build_frame(pairs, probs)
    del pairs
    fr = fr[fr.Function.notna()].reset_index(drop=True)
    excl = KEY_EXCLUDE | {"pred_phase", "prob"}
    cols = [c for c in fr.columns if c not in excl and pd.api.types.is_numeric_dtype(fr[c])]
    log(f"function frame {fr.shape}, {len(cols)} features")

    fr3 = fr[fr.Function.isin(FUNCTIONS)].reset_index(drop=True)
    y = fr3.Function.map({f: i for i, f in enumerate(FUNCTIONS)}).to_numpy()
    P = np.zeros((len(fr3), 3))
    ok = (fr3.health_flag != "failed").to_numpy()
    models = []
    for k in range(N_FOLDS):
        te = (fr3.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        trm = (~te) & ok & (fr3.fold != inner).to_numpy()
        vam = (~te) & ok & (fr3.fold == inner).to_numpy()
        prm = dict(FUNC_PARAMS)
        n = prm.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **prm)
        m.fit(fr3.loc[trm, cols], y[trm], eval_set=[(fr3.loc[vam, cols], y[vam])],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        P[te] = m.predict_proba(fr3.loc[te, cols])
        models.append(m)

    full = fr3.win == "full72"
    acc = float((P.argmax(1) == y).mean())
    acc72 = float((P[full.to_numpy()].argmax(1) == y[full.to_numpy()]).mean())
    T = _temperature(P, y)
    Pc = _apply_T(P, T)
    res = {"acc_allwin": acc, "acc_full72": acc72,
           "ece_raw": _ece(P, y), "ece_cal": _ece(Pc, y), "temperature": T,
           "n": int(len(fr3))}
    cl = (fr3.health_flag != "failed").to_numpy() & full.to_numpy()
    res["acc_full72_classifiable"] = float((P[cl].argmax(1) == y[cl]).mean())
    # coverage / Other threshold curve (DEV has no Other labels -> reported, not tuned on)
    curve = []
    for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = Pc[full.to_numpy()].max(1) >= th
        curve.append({"threshold": th, "coverage": float(m.mean()),
                      "acc_covered": float((Pc[full.to_numpy()][m].argmax(1) ==
                                            y[full.to_numpy()][m]).mean()) if m.any() else None})
    res["other_curve"] = curve
    conf = pd.crosstab(fr3.loc[full, "Function"],
                       np.array(FUNCTIONS)[P[full.to_numpy()].argmax(1)])
    res["confusion_full72"] = conf.to_dict()
    log(json.dumps({k: v for k, v in res.items() if k != "confusion_full72"}, indent=1))
    print(conf)

    out = fr3[["DeviceId", "Detector", "win"]].copy()
    out[["p_advance", "p_presence", "p_count"]] = Pc
    out[out.win == "full72"].drop(columns=["win"]).to_parquet(
        PREDS / "function_oof_v2.parquet", index=False)
    out.to_parquet(PREDS / "function_oof_v2_bywindow.parquet", index=False)
    imp = pd.DataFrame({"feature": cols,
                        "gain": np.mean([m.booster_.feature_importance("gain")
                                         for m in models], axis=0)})
    imp.sort_values("gain", ascending=False).to_csv(
        PREDS / "feature_importance_function_v2.csv", index=False)
    json.dump(res, open(PREDS / "function_v2_results.json", "w"), indent=1, default=str)

    # final DEV-wide model
    MODELS.mkdir(parents=True, exist_ok=True)
    prm = dict(FUNC_PARAMS)
    prm["n_estimators"] = int(np.mean([m.best_iteration_ or 400 for m in models]))
    fin = lgb.LGBMClassifier(**prm)
    fin.fit(fr3.loc[ok, cols], y[ok])
    fin.booster_.save_model(str(MODELS / "function_lgbm_v2.txt"))
    json.dump({"features": cols, "classes": list(FUNCTIONS), "temperature": T,
               "other_threshold": 0.6},
              open(MODELS / "function_lgbm_v2.json", "w"), indent=1)
    log(f"function v2 done in {time.time()-t0:.0f}s")
