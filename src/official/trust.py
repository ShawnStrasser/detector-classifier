"""Task 5b - does an INFERRED delay/extend improve the trust (confidence) estimate?

A trust model predicts P(this prediction is correct) from label-free quantities.
Three feature sets, same rows, same grouped folds (by signal):

  base      phase_prob, margin, evidence (actuations, candidates, minutes, health)
  + est     the label-free delay / extend estimators from `delay_extend.py`
  + official  the true programmed delay / extend (NOT shippable -- upper bound only)

Reported: ROC-AUC and log-loss, out-of-fold.

    python src/official/trust.py --pred <beta detail parquet>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK  # noqa: E402

OFFICIAL = DC_WORK / "official"
BASE = ["phase_prob", "margin", "log_n_act", "n_candidate_phases", "log_minutes"]
EST = ["dur_q01", "dur_med", "frac_lt03", "frac_lt1", "est_delay_score",
       "short_deficit", "rate_deficit", "est_extend_rel"]
OFF = ["delay", "extend"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15,
              min_child_samples=80, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, n_estimators=300, n_jobs=6, verbose=-1)


def oof(df: pd.DataFrame, cols: list[str], folds: np.ndarray, seed: int = 0) -> np.ndarray:
    p = np.zeros(len(df))
    for k in np.unique(folds):
        te = folds == k
        m = lgb.LGBMClassifier(**dict(PARAMS, seed=seed, bagging_seed=seed))
        m.fit(df.loc[~te, cols], df.loc[~te, "y"])
        p[te] = m.predict_proba(df.loc[te, cols])[:, 1]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--est", default=str(OFFICIAL / "delay_extend_est_stg.parquet"))
    ap.add_argument("--out", default=str(OFFICIAL / "trust_results.json"))
    a = ap.parse_args()
    m = pd.read_parquet(a.pred)
    m = m[(m.target_type == "phase") & m.answered & m.scorable].copy()
    est = pd.read_parquet(a.est)
    est["Detector"] = est.Detector.astype(int)
    m["Detector"] = m.Detector.astype(int)
    m = m.merge(est[["DeviceId", "Detector"] + [c for c in EST if c in est.columns]],
                on=["DeviceId", "Detector"], how="left")
    m["y"] = m.correct.astype(int)
    m["margin"] = m.phase_prob - m.phase_prob  # placeholder if 2nd prob missing
    if "phase_2nd_prob" in m.columns:
        m["margin"] = m.phase_prob.fillna(0) - m.phase_2nd_prob.fillna(0)
    m["log_n_act"] = np.log1p(m.n_actuations)
    m["log_minutes"] = np.log1p(m.get("minutes_of_data", pd.Series(0, index=m.index)))
    for c in BASE + EST + OFF:
        if c not in m.columns:
            m[c] = np.nan
    devs = {d: i % 5 for i, d in enumerate(sorted(m.DeviceId.unique()))}
    folds = m.DeviceId.map(devs).to_numpy()
    res = {"n": int(len(m)), "base_rate": float(m.y.mean())}
    print(f"n={len(m):,}  accuracy(base rate)={m.y.mean():.4f}")
    # raw model probability as a trust score, for reference
    res["phase_prob_alone"] = {"auc": float(roc_auc_score(m.y, m.phase_prob)),
                               "logloss": float(log_loss(m.y, m.phase_prob.clip(1e-6, 1-1e-6)))}
    for name, cols in [("base", BASE), ("base+est", BASE + EST),
                       ("base+official", BASE + OFF),
                       ("base+est+official", BASE + EST + OFF)]:
        p = oof(m, cols, folds)
        res[name] = {"auc": float(roc_auc_score(m.y, p)),
                     "logloss": float(log_loss(m.y, np.clip(p, 1e-6, 1 - 1e-6))),
                     "n_features": len(cols)}
        print(f"{name:20s} AUC {res[name]['auc']:.4f}  logloss {res[name]['logloss']:.4f}")
    print(f"{'phase_prob alone':20s} AUC {res['phase_prob_alone']['auc']:.4f}  "
          f"logloss {res['phase_prob_alone']['logloss']:.4f}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
