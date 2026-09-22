"""Stage 07 task 2: seed bagging (K = 1, 3, 5, 10) of the ranker / decoder / function head.

Bagging is measured on the SEARCH protocol first (folds 1-5, 9-window subset) because K=10
means 10x the fits; only the K that is worth keeping is then re-run over all 22 windows and
all 6 folds.  Model size and CPU inference time are measured for the LightGBM text models
via both `lightgbm` and the numpy-only evaluator `src/lgbm_numpy.py`.

    python src/tune/bagging.py --stage ranker
    python src/tune/bagging.py --stage function
    python src/tune/bagging.py --stage cost
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

from tune_common import (FULL, N_JOBS, SEARCH_WINS, TUNE, add_scorable, dump,  # noqa: E402
                         feature_cols, load_phase_pairs, log, phase_metrics, top1_table)
from tune_phase import (MAX_TREES, SEARCH_FOLDS, fit_score, inner_split)  # noqa: E402

KS = [1, 3, 5, 10]


def stage_ranker(a) -> None:
    best = json.load(open(TUNE / "best_ranker.json"))["full_params"]
    best["n_estimators"] = MAX_TREES
    best["n_jobs"] = N_JOBS
    df = add_scorable(load_phase_pairs())
    df = df[df.win.isin(SEARCH_WINS) & df.fold.isin(SEARCH_FOLDS) & df.labelled]
    df = df.reset_index(drop=True)
    cols = feature_cols(df)
    trainable = df.health_flag.ne("failed").to_numpy()
    # cache the per-seed per-fold probabilities once, then average prefixes
    S = np.zeros((max(KS), len(df)))
    secs = []
    for k in SEARCH_FOLDS:
        te = (df.fold == k).to_numpy()
        sub = df[trainable & ~te & df.fold.isin(SEARCH_FOLDS).to_numpy()]
        vm = inner_split(sub, seed=1000 + k)
        tr, va = sub[~vm], sub[vm]
        for s in range(max(KS)):
            p = dict(best)
            p.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202,
                     data_random_seed=s + 303)
            t0 = time.time()
            pr, _ = fit_score(p, tr, va, df[te], cols)
            secs.append(time.time() - t0)
            S[s, te] = pr
        log(f"  fold {k}: {max(KS)} seeds done")
    res = {}
    for K in KS:
        P = S[:K].mean(0)
        t = top1_table(df, P)
        res[f"K{K}"] = phase_metrics(t)
        t.to_parquet(TUNE / f"top1_bag_ranker_K{K}.parquet", index=False)
        log(f"K={K}: {res[f'K{K}']}")
    res["fit_secs_mean"] = float(np.mean(secs))
    dump(res, "bagging_ranker.json")


def stage_function(a) -> None:
    import tune_function as tf
    meta = json.load(open(TUNE / "best_function.json"))
    p = meta["full_params"]
    p["n_jobs"] = N_JOBS
    fr = tf.load_frame()
    cols = tf.feat_cols(fr)
    fs = fr[fr.win.isin(SEARCH_WINS) & fr.fold.isin(SEARCH_FOLDS)].reset_index(drop=True)
    res = {}
    Ps = []
    for s in range(max(KS)):
        ps = dict(p)
        ps.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202,
                  data_random_seed=s + 303)
        P, _ = tf.oof(ps, fs, cols, SEARCH_FOLDS, meta["class_weight"])
        Ps.append(P)
        log(f"  seed {s} done")
    for K in KS:
        M = np.mean(Ps[:K], axis=0)
        pred = np.array(tf.CLASSES5)[M.argmax(1)]
        m = tf.func_metrics(fs, pred)
        conf = M.max(1)
        for th in (0.7, 0.9):
            sel = conf >= th
            m[f"cov{th}"] = float(sel.mean())
            m[f"acc_at_cov{th}"] = float((fs.func5.to_numpy()[sel] == pred[sel]).mean())
        res[f"K{K}"] = m
        log(f"K={K}: {m}")
    dump(res, "bagging_function.json")


# ------------------------------------------------------------- size / latency
def stage_cost(a) -> None:
    """Model size + CPU inference time for 1 / 3 / 5 / 10 averaged LightGBM text models."""
    from lgbm_numpy import NumpyBooster
    best = json.load(open(TUNE / "best_ranker.json"))["full_params"]
    best["n_estimators"] = MAX_TREES
    best["n_jobs"] = N_JOBS
    df = add_scorable(load_phase_pairs())
    df = df[df.win.isin(SEARCH_WINS) & df.labelled].reset_index(drop=True)
    cols = feature_cols(df)
    tr = df[(df.fold != 0) & df.health_flag.ne("failed")]
    vm = inner_split(tr, seed=1000)
    itr, iva = tr[~vm], tr[vm]
    files, boosters = [], []
    d = TUNE / "bagmodels"
    d.mkdir(exist_ok=True)
    for s in range(max(KS)):
        p = dict(best)
        p.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202)
        P = dict(p)
        n = P.pop("n_estimators")
        obj = P.get("objective")
        m = (lgb.LGBMClassifier(n_estimators=n, **P) if obj == "binary"
             else lgb.LGBMRanker(n_estimators=n, **P))
        if obj == "binary":
            m.fit(itr[cols], itr.y, eval_set=[(iva[cols], iva.y)],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
        else:
            from tune_phase import group_sizes
            m.fit(itr[cols], itr.y, group=group_sizes(itr),
                  eval_set=[(iva[cols], iva.y)], eval_group=[group_sizes(iva)],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
        f = d / f"ranker_seed{s}.txt"
        m.booster_.save_model(str(f))
        files.append(f)
        boosters.append(m.booster_)
        log(f"  seed {s}: {m.best_iteration_} trees, {f.stat().st_size/1e6:.2f} MB")
    X = df[df.win == FULL][cols].iloc[:5000]
    res = {}
    for K in KS:
        size = sum(f.stat().st_size for f in files[:K])
        t0 = time.time()
        for b in boosters[:K]:
            b.predict(X)
        tl = (time.time() - t0) / len(X) * 1e6
        nb = [NumpyBooster(str(f)) for f in files[:K]]
        t0 = time.time()
        for b in nb:
            b.predict(X)
        tn = (time.time() - t0) / len(X) * 1e6
        res[f"K{K}"] = {"model_bytes": int(size), "model_MB": round(size / 1e6, 2),
                        "lightgbm_us_per_row": round(tl, 2),
                        "numpy_us_per_row": round(tn, 2), "n_rows": int(len(X))}
        log(f"K={K}: {res[f'K{K}']}")
    # numpy vs lightgbm agreement (the shipping contract)
    b0 = boosters[0]
    n0 = NumpyBooster(str(files[0]))
    d1 = float(np.abs(np.asarray(b0.predict(X)) - n0.predict(X)).max())
    res["numpy_vs_lightgbm_max_abs_diff_single"] = d1
    avg_l = np.mean([b.predict(X) for b in boosters], axis=0)
    avg_n = np.mean([NumpyBooster(str(f)).predict(X) for f in files], axis=0)
    res["numpy_vs_lightgbm_max_abs_diff_K10_average"] = float(np.abs(avg_l - avg_n).max())
    dump(res, "bagging_cost.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
