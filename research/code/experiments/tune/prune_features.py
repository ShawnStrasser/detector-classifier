"""Stage 07 task 3: how few features does the pair ranker actually need?

Three views, all on the search protocol (folds 1-5, 9-window subset, fold 0 untouched):
  1. **gain importance** averaged over the five fold models;
  2. **permutation importance** measured directly on the stage-07 primary metric --
     one feature at a time is shuffled inside each fold's test rows and the drop in the
     (72 h + 30 min)/2 top-1 accuracy is recorded;
  3. **top-N sweep** and **leave-one-family-out**, to find the smallest set within
     0.05 pt of the full model.

    python src/tune/prune_features.py --stage importance
    python src/tune/prune_features.py --stage sweep
    python src/tune/prune_features.py --stage verify --keep 60
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path

from tune_common import (N_JOBS, SEARCH_WINS, TUNE, add_scorable, dump,  # noqa: E402
                         feature_cols, load_phase_pairs, log, phase_metrics, top1_table)
from tune_phase import (MAX_TREES, SEARCH_FOLDS, compare, fit_score,  # noqa: E402
                        group_sizes, inner_split, norm_prob, run_full_ranker, to_prob)

# plain-English family map for the pair features
FAMILIES = {
    "colour_state": ["f_on_", "f_occ_", "dur_mean_", "dur_g2"],
    "time_hist": ["dtg_", "dtr_", "tog_"],
    "queue_release": ["queue_", "release_", "straddle", "n_long", "burst_"],
    "calls_43_44": ["call43", "call44"],
    "exclusive_green": ["excl_", "pex_", "solo_lift"],
    "green_lift": ["on_lift_", "occ_lift_"],
    "green_extension": ["ext_", "late2_", "late_green"],
    "coordination": ["_coord", "_free", "looks_recall"],
    "phase_context": ["cand_green_share", "n_cycles", "green_mean", "green_sd",
                      "green_med", "cycle_mean", "phase_"],
    "detector_level": ["det_"],
    "partner_geometry": ["cogreen_", "partner_lead_", "__pdiff"],
    "first_on_timing": ["first_on_", "cyc_hit_frac"],
    "within_detector_rank": ["__rank", "__z", "__mgap", "__argmax"],
    "evidence_size": ["win_secs", "log_win_hours", "log_det_n_on"],
}


def family_of(col: str) -> str:
    for fam, pats in FAMILIES.items():
        if any(p in col for p in pats):
            return fam
    return "other"


def search_frame():
    df = add_scorable(load_phase_pairs())
    df = df[df.win.isin(SEARCH_WINS) & df.fold.isin(SEARCH_FOLDS) & df.labelled]
    return df.reset_index(drop=True)


def best_params() -> dict:
    p = json.load(open(TUNE / "best_ranker.json"))["full_params"]
    p["n_estimators"] = MAX_TREES
    p["n_jobs"] = N_JOBS
    return p


def _fit_models(df, cols, params):
    models, tests = [], []
    for k in SEARCH_FOLDS:
        te = (df.fold == k).to_numpy()
        sub = df[~te & df.health_flag.ne("failed").to_numpy()]
        vm = inner_split(sub, seed=1000 + k)
        tr, va = sub[~vm], sub[vm]
        P = dict(params)
        n = P.pop("n_estimators")
        obj = P.get("objective", "lambdarank")
        if obj == "binary":
            m = lgb.LGBMClassifier(n_estimators=n, **P)
            m.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
        else:
            m = lgb.LGBMRanker(n_estimators=n, **P)
            m.fit(tr[cols], tr.y, group=group_sizes(tr), eval_set=[(va[cols], va.y)],
                  eval_group=[group_sizes(va)],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
        models.append(m)
        tests.append(te)
        log(f"  fitted fold {k} ({m.best_iteration_} trees)")
    return models, tests


def _score(df, models, tests, cols, perm_col=None, rng=None):
    P = np.zeros(len(df))
    for m, te in zip(models, tests):
        X = df.loc[te, cols]
        if perm_col is not None:
            X = X.copy()
            X[perm_col] = rng.permutation(X[perm_col].to_numpy())
        s = (m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X))
        P[te] = (norm_prob(df[te], s) if hasattr(m, "predict_proba")
                 else to_prob(df[te], s))
    return phase_metrics(top1_table(df, P))["primary"]


def stage_importance(a) -> None:
    df = search_frame()
    cols = feature_cols(df)
    params = best_params()
    t0 = time.time()
    models, tests = _fit_models(df, cols, params)
    gain = np.mean([m.booster_.feature_importance("gain") for m in models], axis=0)
    base = _score(df, models, tests, cols)
    log(f"baseline primary {base:.5f}; permutation importance over {len(cols)} features")
    rng = np.random.default_rng(0)
    perm = []
    for i, c in enumerate(cols):
        v = _score(df, models, tests, cols, perm_col=c, rng=rng)
        perm.append(base - v)
        if i % 40 == 0:
            log(f"  {i}/{len(cols)} [{time.time()-t0:.0f}s]")
    imp = pd.DataFrame({"feature": cols, "gain": gain, "perm_drop": perm,
                        "family": [family_of(c) for c in cols]})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(TUNE / "feature_importance_ranker_tuned.csv", index=False)
    fam = imp.groupby("family").agg(n=("gain", "size"), gain=("gain", "sum"),
                                    perm=("perm_drop", "sum")).sort_values(
        "perm", ascending=False)
    dump({"baseline_primary": base, "by_family": fam.round(6).to_dict("index"),
          "seconds": time.time() - t0}, "feature_importance_summary.json")
    log(fam.to_string())


def stage_sweep(a) -> None:
    df = search_frame()
    cols = feature_cols(df)
    params = best_params()
    trainable = df.health_flag.ne("failed").to_numpy()
    imp = pd.read_csv(TUNE / "feature_importance_ranker_tuned.csv")
    res = {}

    def run(sub_cols, tag):
        parts = []
        for k in SEARCH_FOLDS:
            te = (df.fold == k).to_numpy()
            s = df[~te & trainable]
            vm = inner_split(s, seed=1000 + k)
            p, _ = fit_score(params, s[~vm], s[vm], df[te], sub_cols)
            parts.append(top1_table(df[te], p))
        m = phase_metrics(pd.concat(parts, ignore_index=True))
        m["n_features"] = len(sub_cols)
        res[tag] = m
        log(f"{tag} ({len(sub_cols)} feat): primary {m['primary']:.5f} "
            f"72h {m['acc72']:.4f} 30min {m['acc30']:.4f} nonstd {m['nonstd72']:.4f}")
        return m

    run(cols, "all")
    # rank by permutation drop, falling back to gain for ties
    order = imp.sort_values(["perm_drop", "gain"], ascending=False).feature.tolist()
    for n in [200, 150, 100, 70, 50, 35, 25, 15, 10]:
        if n >= len(cols):
            continue
        run(order[:n], f"top{n}")
    fam_of = dict(zip(imp.feature, imp.family))
    for fam in sorted(set(imp.family)):
        keep = [c for c in cols if fam_of.get(c) != fam]
        if len(keep) == len(cols):
            continue
        run(keep, f"minus_{fam}")
    dump(res, "feature_pruning.json")


def stage_verify(a) -> None:
    imp = pd.read_csv(TUNE / "feature_importance_ranker_tuned.csv")
    order = imp.sort_values(["perm_drop", "gain"], ascending=False).feature.tolist()
    keep = order[:a.keep]
    params = best_params()
    run_full_ranker(params, f"ranker_pruned{a.keep}", cols=keep)
    json.dump(keep, open(TUNE / f"features_top{a.keep}.json", "w"), indent=1)
    compare("ranker_tuned", f"ranker_pruned{a.keep}", f"pruned{a.keep}_vs_tuned")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--keep", type=int, default=60)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
