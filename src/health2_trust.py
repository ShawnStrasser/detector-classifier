"""Stage 09 (b): does period-level detector health predict WRONG predictions?

Builds a per-(detector, window) table from the shipped beta OOF phase predictions, adds
the `health2` period-level summary, and compares ways of deciding what to trust:

  current   phase top prob (what `predict.py` thresholds today, with its >=5-actuation rule)
  rule      `health2.trust_score(..., rule_only=True)`
  evid      LightGBM on confidence + evidence only (no health)
  +health   the same model plus the health summary columns
  +shuffled the same model with the health columns permuted across detectors (control)

Metrics: accuracy at fixed coverage (90 / 95 %), AUC of error detection, per fold, with
paired fold differences and a bootstrap over signals.

    python src/health2_trust.py [--target phase|function]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DC_WORK, FOLDS_CSV, LABELS_DEV, N_FOLDS, PREDS  # noqa: E402
import features  # noqa: E402
import health2 as h2  # noqa: E402
from health2_features import WINDOWS_H2  # noqa: E402  (registers nothing new)
from function_v3_prep import WINDOWS_SHORT_B  # noqa: E402

OUT = DC_WORK / "health2"
ALL_WINDOWS = {w["win"]: w for w in features.WINDOWS_MIXED + WINDOWS_SHORT_B}

EVID = ["top_prob", "margin", "entropy", "log_n_act", "log_win_hours", "n_cand"]
HEALTH = ["masked_frac", "n_reasons", "h_stuck_on", "h_chatter", "h_flatline",
          "h_fault", "h_level_low", "h_level_high", "h_geh_anomaly", "h_comm_loss",
          "h_flash", "n_bad_intervals", "act_masked_frac"]

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_child_samples=80, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=600, n_jobs=6, verbose=-1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- health per window
def health_by_window(bad: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """`health2.summarise` for each evaluation window."""
    out = []
    for name, w in ALL_WINDOWS.items():
        t0 = w["t0"]
        t1 = t0 + pd.Timedelta(seconds=float(w["secs"]))
        b = bad[(bad["start"] < t1) & (bad["end"] > t0)]
        s = h2.summarise(b, universe, t0, t1, reasons=h2.REASONS)
        s["win"] = name
        n = (b.assign(k=b.Detector.fillna(-1))
             .groupby(["DeviceId", "k"]).size().rename("n_bad_intervals").reset_index())
        n = n[n.k >= 0].rename(columns={"k": "Detector"})
        s = s.merge(n, on=["DeviceId", "Detector"], how="left")
        s["n_bad_intervals"] = s.n_bad_intervals.fillna(0)
        out.append(s)
    h = pd.concat(out, ignore_index=True)
    fcols = [c for c in h.columns if c.startswith("h_fault_")]
    h["h_fault"] = h[fcols].sum(axis=1) if fcols else 0.0
    return h


# ---------------------------------------------------------------- assemble
def build_function_table() -> pd.DataFrame:
    """Same thing for the stage-06 function head (5-class, `function_oof_v3`)."""
    f = pd.read_parquet(DC_WORK / "function_v3" / "function_oof_v3_bywindow.parquet")
    P = f[["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other"]].to_numpy()
    srt = np.sort(P, axis=1)
    d = f[["DeviceId", "Detector", "win", "fold", "det_n_on"]].copy()
    d["top_prob"] = srt[:, -1]
    d["margin"] = srt[:, -1] - srt[:, -2]
    d["entropy"] = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    d["n_cand"] = 5
    d["y"] = (f.pred5 == f.func5).astype(np.int8)
    d["in_cand"] = True
    d["n_actuations"] = d.det_n_on.fillna(0)
    wh = {k: w["secs"] / 3600.0 for k, w in ALL_WINDOWS.items()}
    d["win_hours"] = d.win.map(wh).fillna(1.0)
    d["log_n_act"] = np.log1p(d.n_actuations)
    d["log_win_hours"] = np.log(d.win_hours)
    bad = pd.read_parquet(OUT / "bad_intervals.parquet")
    hw = health_by_window(bad, d[["DeviceId", "Detector"]].drop_duplicates())
    d = d.merge(hw, on=["DeviceId", "Detector", "win"], how="left")
    mk = pd.read_parquet(OUT / "masked_on_counts.parquet")
    d = d.merge(mk[["DeviceId", "Detector", "frac_occ"]]
                .rename(columns={"frac_occ": "act_masked_frac"}),
                on=["DeviceId", "Detector"], how="left")
    for c in HEALTH:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = d[c].fillna(0.0)
    return d


def build_table() -> pd.DataFrame:
    pr = pd.read_parquet(PREDS / "beta" / "phase_oof_beta_decoded_bywindow.parquet")
    pr["prob"] = pr.prob.astype(float)
    pr["prob"] = pr.prob / pr.groupby(["DeviceId", "Detector", "win"])["prob"] \
                             .transform("sum")
    g = pr.sort_values("prob", ascending=False).groupby(["DeviceId", "Detector", "win"],
                                                        sort=False)
    top = g.head(1).rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})
    sec = g.nth(1)[["DeviceId", "Detector", "win", "prob"]] \
        .rename(columns={"prob": "second_prob"})
    ent = (pr.assign(e=-pr.prob * np.log(pr.prob.clip(1e-12)))
             .groupby(["DeviceId", "Detector", "win"], as_index=False)
             .agg(entropy=("e", "sum"), n_cand=("cand_phase", "size")))
    d = top[["DeviceId", "Detector", "win", "pred_phase", "top_prob"]] \
        .merge(sec, on=["DeviceId", "Detector", "win"], how="left") \
        .merge(ent, on=["DeviceId", "Detector", "win"], how="left")
    d["second_prob"] = d.second_prob.fillna(0.0)
    d["margin"] = d.top_prob - d.second_prob

    lab = pd.read_parquet(LABELS_DEV)
    d = d.merge(lab[["DeviceId", "Detector", "Phase"]], on=["DeviceId", "Detector"],
                how="inner")
    cand = pr.groupby(["DeviceId", "Detector", "win"])["cand_phase"].apply(set) \
             .rename("cands").reset_index()
    d = d.merge(cand, on=["DeviceId", "Detector", "win"], how="left")
    d["in_cand"] = [p in c for p, c in zip(d.Phase, d.cands)]
    d = d.drop(columns=["cands"])
    d["y"] = (d.pred_phase == d.Phase).astype(np.int8)

    # evidence
    cols = ["DeviceId", "Detector", "win", "det_n_on", "win_secs"]
    fa = pd.read_parquet(DC_WORK / "features" / "pair_features_windows.parquet",
                         columns=cols).drop_duplicates(["DeviceId", "Detector", "win"])
    fb = pd.read_parquet(DC_WORK / "features" / "pair_features_windows_B.parquet",
                         columns=cols).drop_duplicates(["DeviceId", "Detector", "win"])
    f = pd.concat([fa, fb], ignore_index=True)
    d = d.merge(f, on=["DeviceId", "Detector", "win"], how="left")
    d["n_actuations"] = d.det_n_on.fillna(0)
    d["win_hours"] = d.win_secs.fillna(3600) / 3600.0
    d["log_n_act"] = np.log1p(d.n_actuations)
    d["log_win_hours"] = np.log(d.win_hours)

    # health
    bad = pd.read_parquet(OUT / "bad_intervals.parquet")
    univ = d[["DeviceId", "Detector"]].drop_duplicates()
    hw = health_by_window(bad, univ)
    d = d.merge(hw, on=["DeviceId", "Detector", "win"], how="left")

    # share of the detector's occupied time removed by masking (whole sample)
    mk = pd.read_parquet(OUT / "masked_on_counts.parquet")
    d = d.merge(mk[["DeviceId", "Detector", "frac_occ"]]
                .rename(columns={"frac_occ": "act_masked_frac"}),
                on=["DeviceId", "Detector"], how="left")
    for c in HEALTH:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = d[c].fillna(0.0)

    d = d.merge(pd.read_csv(FOLDS_CSV), on="DeviceId", how="inner")
    return d


# ---------------------------------------------------------------- scoring helpers
def acc_at_coverage(score: np.ndarray, y: np.ndarray, cov: float) -> float:
    n = max(int(round(len(y) * cov)), 1)
    idx = np.argsort(-score, kind="stable")[:n]
    return float(y[idx].mean())


def auc_error(score: np.ndarray, y: np.ndarray) -> float:
    """AUC of ranking errors below correct predictions (1.0 = perfect error detection)."""
    from sklearn.metrics import roc_auc_score
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(1 - y, -score))


def oof_scores(d: pd.DataFrame, cols: list, shuffle_cols: list = None,
               seed: int = 0) -> np.ndarray:
    s = np.zeros(len(d))
    X = d[cols].copy()
    if shuffle_cols:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        for c in shuffle_cols:
            X[c] = X[c].to_numpy()[perm]
    for k in range(N_FOLDS):
        te = (d.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        tr = (~te) & (d.fold != inner).to_numpy()
        va = (~te) & (d.fold == inner).to_numpy()
        P = dict(LGB_PARAMS)
        m = lgb.LGBMClassifier(n_estimators=P.pop("n_estimators"), **P)
        m.fit(X[tr], d.y[tr], eval_set=[(X[va], d.y[va])],
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def evaluate(d: pd.DataFrame, scores: dict) -> pd.DataFrame:
    rows = []
    y = d.y.to_numpy()
    for name, s in scores.items():
        r = {"model": name,
             "acc@100": float(y.mean()),
             "acc@95": acc_at_coverage(s, y, 0.95),
             "acc@90": acc_at_coverage(s, y, 0.90),
             "acc@80": acc_at_coverage(s, y, 0.80),
             "auc_err": auc_error(s, y)}
        per = []
        for k in range(N_FOLDS):
            m = (d.fold == k).to_numpy()
            per.append(acc_at_coverage(s[m], y[m], 0.90))
        r["acc@90_folds"] = per
        r["acc@90_fold_mean"] = float(np.mean(per))
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_delta(d, sa, sb, cov=0.90, n_boot=300, seed=0):
    """Paired bootstrap over SIGNALS of acc@cov(b) - acc@cov(a)."""
    rng = np.random.default_rng(seed)
    devs = d.DeviceId.unique()
    idx_by_dev = {dev: np.where(d.DeviceId.to_numpy() == dev)[0] for dev in devs}
    y = d.y.to_numpy()
    out = []
    for _ in range(n_boot):
        pick = rng.choice(devs, size=len(devs), replace=True)
        ix = np.concatenate([idx_by_dev[p] for p in pick])
        out.append(acc_at_coverage(sb[ix], y[ix], cov) - acc_at_coverage(sa[ix], y[ix], cov))
    return float(np.mean(out)), float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-actuations", type=int, default=5)
    ap.add_argument("--target", default="phase", choices=["phase", "function"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = build_function_table() if a.target == "function" else build_table()
    log(f"table: {len(d):,} detector-windows, {d.DeviceId.nunique()} signals, "
        f"base accuracy {d.y.mean():.4f}")
    d.to_parquet(OUT / f"trust_table_{a.target}.parquet", index=False)

    # protocol: unscorable rows (labelled phase never green in the window) are excluded
    d = d[d.in_cand].reset_index(drop=True)
    log(f"scorable: {len(d):,} rows, accuracy {d.y.mean():.4f}")

    scores = {}
    scores["current (top prob)"] = d.top_prob.to_numpy()
    cur = d.top_prob.to_numpy().copy()
    cur[d.n_actuations.to_numpy() < a.min_actuations] = -1.0
    scores[f"current + min{a.min_actuations} act"] = cur
    scores["rule (health2.trust_score)"] = h2.trust_score(
        d.assign(DeviceId=d.DeviceId, Detector=d.Detector), d, rule_only=True)
    scores["model: evidence"] = oof_scores(d, EVID)
    scores["model: evidence+health"] = oof_scores(d, EVID + HEALTH)
    scores["model: evidence+shuffled health"] = oof_scores(d, EVID + HEALTH,
                                                           shuffle_cols=HEALTH)
    scores["model: health only"] = oof_scores(d, HEALTH)

    res = evaluate(d, scores)
    log("\n" + res.drop(columns=["acc@90_folds"]).to_string(index=False))

    base = scores["model: evidence"]
    deltas = {}
    for name in ("model: evidence+health", "model: evidence+shuffled health"):
        for cov in (0.90, 0.95):
            m, lo, hi = bootstrap_delta(d, base, scores[name], cov)
            nf = sum(acc_at_coverage(scores[name][(d.fold == k).to_numpy()],
                                     d.y.to_numpy()[(d.fold == k).to_numpy()], cov) >
                     acc_at_coverage(base[(d.fold == k).to_numpy()],
                                     d.y.to_numpy()[(d.fold == k).to_numpy()], cov)
                     for k in range(N_FOLDS))
            deltas[f"{name} @{cov:.0%}"] = dict(mean=m, lo=lo, hi=hi, folds_better=nf)
            log(f"delta {name} @{cov:.0%}: {m*100:+.2f} pt ({lo*100:+.2f}, {hi*100:+.2f}), "
                f"{nf}/6 folds better")

    # which health columns does the model actually use?
    P = dict(LGB_PARAMS)
    m = lgb.LGBMClassifier(n_estimators=300, **{k: v for k, v in P.items()
                                                if k != "n_estimators"})
    m.fit(d[EVID + HEALTH], d.y)
    imp = pd.Series(m.booster_.feature_importance("gain"),
                    index=EVID + HEALTH).sort_values(ascending=False)
    log("\nfeature gain:\n" + imp.to_string())

    json.dump({"target": a.target, "results": res.to_dict("records"), "deltas": deltas,
               "importance": imp.to_dict(), "n_rows": int(len(d)),
               "base_acc": float(d.y.mean())},
              open(OUT / f"trust_results_{a.target}.json", "w"), indent=1, default=float)
    np.save(OUT / "trust_scores.npy", np.vstack([scores[k] for k in scores]))
    json.dump(list(scores), open(OUT / "trust_score_names.json", "w"))
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
