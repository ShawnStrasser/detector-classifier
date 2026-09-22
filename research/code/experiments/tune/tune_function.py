"""Stage 07 task 1c: Optuna tuning of the v3 5-class function head.

Reuses `dc_work/function_v3/funcframe_v3.parquet` (built in stage 06) -- no features are
rebuilt.  Search on folds 1-5 only; fold 0 reported afterwards.

    python src/tune/tune_function.py --stage search --trials 80 --timeout 5400
    python src/tune/tune_function.py --stage full
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

from tune_common import (FULL, FUNC_V3, M30, N_JOBS, SEARCH_WINS, TUNE, dump,  # noqa: E402
                         log)

CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CLASSES3 = ["Advance", "Presence", "Count"]
SEARCH_FOLDS = [1, 2, 3, 4, 5]
INNER_FRAC = 0.15
MAX_TREES = 3000
EARLY = 100

NON_FEATURES = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase", "Function",
                "fold", "y", "cyc", "partner_phase", "health_flag", "std_phase",
                "pred_phase", "prob", "func5", "src", "label_phase", "other_phase",
                "wgroup"}


def load_frame() -> pd.DataFrame:
    fr = pd.read_parquet(FUNC_V3 / "funcframe_v3.parquet")
    fr = fr[fr.func5.notna()].reset_index(drop=True)
    fr["wgroup"] = np.where(fr.win == "full72", "full72", fr.win.str.split("_").str[0])
    return fr


def feat_cols(fr: pd.DataFrame) -> list[str]:
    return [c for c in fr.columns
            if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(fr[c])]


def func_metrics(fr: pd.DataFrame, pred: np.ndarray) -> dict:
    y = fr.func5.to_numpy()
    ok = (y == pred)
    full = (fr.win == FULL).to_numpy()
    per30 = [float(ok[(fr.win == w).to_numpy()].mean()) for w in M30
             if (fr.win == w).any()]
    a72 = float(ok[full].mean())
    a30 = float(np.mean(per30)) if per30 else np.nan
    apc = np.isin(y, CLASSES3)
    return {"primary": (a72 + a30) / 2, "acc72": a72, "acc30": a30,
            "acc_allwin": float(ok.mean()),
            "acc_apc72": float(ok[full & apc].mean()),
            "acc_apc_allwin": float(ok[apc].mean()),
            "n72": int(full.sum())}


def inner_split(tr: pd.DataFrame, seed: int) -> np.ndarray:
    devs = np.sort(tr.DeviceId.unique())
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(devs, max(1, int(round(INNER_FRAC * len(devs)))), replace=False))
    return tr.DeviceId.isin(pick).to_numpy()


def class_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    cnt = np.bincount(y, minlength=len(CLASSES5)).astype(float)
    w = cnt.sum() / np.maximum(cnt, 1)
    if mode == "sqrt_balanced":
        w = np.sqrt(w)
    w = w / w.mean()
    return w[y]


def oof(params: dict, fr: pd.DataFrame, cols: list[str], folds: list[int],
        wmode: str = "none", report=None, seeds: list[int] | None = None):
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    ok = (fr.health_flag != "failed").to_numpy()
    pool = fr.fold.isin(folds).to_numpy()
    P = np.zeros((len(fr), len(CLASSES5)))
    seeds = seeds or [0]
    iters = []
    for j, k in enumerate(folds):
        te = (fr.fold == k).to_numpy()
        trall = pool & ~te & ok
        sub = fr[trall]
        vm = inner_split(sub, seed=1000 + k)
        yy = y[trall]
        tr, va = sub[~vm], sub[vm]
        ytr, yva = yy[~vm], yy[vm]
        sw = class_weights(ytr, wmode)
        acc = np.zeros((int(te.sum()), len(CLASSES5)))
        for s in seeds:
            P2 = dict(params)
            P2["num_class"] = len(CLASSES5)
            if len(seeds) > 1:
                P2.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202)
            n = P2.pop("n_estimators", MAX_TREES)
            m = lgb.LGBMClassifier(n_estimators=n, **P2)
            m.fit(tr[cols], ytr, sample_weight=sw, eval_set=[(va[cols], yva)],
                  eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(EARLY, verbose=False),
                             lgb.log_evaluation(0)])
            acc += m.predict_proba(fr.loc[te, cols])
            iters.append(int(m.best_iteration_ or n))
        P[te] = acc / len(seeds)
        if report is not None:
            done = fr.fold.isin(folds[:j + 1]).to_numpy()
            report(j, func_metrics(fr[done], np.array(CLASSES5)[P[done].argmax(1)])["primary"])
    return P, iters


def suggest(trial) -> tuple[dict, str]:
    p = dict(objective="multiclass", n_jobs=N_JOBS, verbose=-1, n_estimators=MAX_TREES,
             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
             num_leaves=trial.suggest_int("num_leaves", 7, 160, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 5, 300, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.2, 1.0),
             bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
             bagging_freq=1,
             lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 20.0, log=True),
             lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 50.0, log=True),
             min_split_gain=trial.suggest_float("min_split_gain", 1e-6, 1.0, log=True),
             max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511]))
    w = trial.suggest_categorical("class_weight", ["none", "sqrt_balanced", "balanced"])
    return p, w


def stage_search(a) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fr = load_frame()
    cols = feat_cols(fr)
    fs = fr[fr.win.isin(SEARCH_WINS) & fr.fold.isin(SEARCH_FOLDS)].reset_index(drop=True)
    log(f"function search frame {fs.shape}, {len(cols)} features, "
        f"{fs.DeviceId.nunique()} signals")
    t0 = time.time()

    def objective(trial):
        p, w = suggest(trial)

        def rep(j, val):
            trial.report(val, j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        P, iters = oof(p, fs, cols, SEARCH_FOLDS, w, report=rep)
        m = func_metrics(fs, np.array(CLASSES5)[P.argmax(1)])
        trial.set_user_attr("iters", iters)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["primary"]

    st = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=13, n_startup_trials=12,
                                           multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    st.enqueue_trial({"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40,
                      "feature_fraction": 0.7, "bagging_fraction": 0.8,
                      "lambda_l1": 1e-4, "lambda_l2": 1.0, "min_split_gain": 1e-6,
                      "max_bin": 255, "class_weight": "none"})
    st.optimize(objective, n_trials=a.trials, timeout=a.timeout,
                callbacks=[lambda s, t: log(
                    f"trial {t.number} {t.state.name} value={t.value} "
                    f"best={s.best_value:.5f} [{time.time()-t0:.0f}s]")])
    pd.DataFrame([dict(number=t.number, value=t.value, state=t.state.name, **t.params,
                       acc72=t.user_attrs.get("acc72"),
                       acc_apc72=t.user_attrs.get("acc_apc72"),
                       iters=t.user_attrs.get("iters")) for t in st.trials]
                 ).to_csv(TUNE / "function_trials.csv", index=False)
    p, w = suggest(optuna.trial.FixedTrial(st.best_params))
    dump({"best_params": st.best_params, "best_value": st.best_value,
          "full_params": p, "class_weight": w, "n_trials": len(st.trials),
          "seconds": time.time() - t0}, "best_function.json")
    log(f"BEST {st.best_value:.5f} {st.best_params}")


# --------------------------------------------------------------- full 6 folds
def _champ_params() -> dict:
    return dict(objective="multiclass", learning_rate=0.05, num_leaves=31,
                min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, n_estimators=MAX_TREES, n_jobs=N_JOBS,
                verbose=-1)


def boot_func(fr, pa, pb, n_boot=2000, seed=0, alpha=0.10) -> dict:
    y = fr.func5.to_numpy()
    full = (fr.win == FULL).to_numpy()
    is30 = fr.win.isin(M30).to_numpy()
    m = full | is30
    dev = fr.DeviceId.to_numpy()[m]
    devs, di = np.unique(dev, return_inverse=True)
    n = len(devs)
    oka = (y[m] == pa[m]).astype(float)
    okb = (y[m] == pb[m]).astype(float)
    w72 = full[m]
    s = {}
    for nm, ok in (("a", oka), ("b", okb)):
        s[nm + "72"] = np.bincount(di[w72], ok[w72], n)
        s[nm + "30"] = np.bincount(di[~w72], ok[~w72], n)
    n72 = np.bincount(di[w72], None, n)
    n30 = np.bincount(di[~w72], None, n)

    def prim(x72, x30, c72, c30):
        return 0.5 * (x72.sum() / max(c72.sum(), 1e-9) + x30.sum() / max(c30.sum(), 1e-9))
    point = (prim(s["b72"], s["b30"], n72, n30) - prim(s["a72"], s["a30"], n72, n30))
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.integers(0, n, n)
        d[i] = (prim(s["b72"][p], s["b30"][p], n72[p], n30[p])
                - prim(s["a72"][p], s["a30"][p], n72[p], n30[p]))
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(point), "lo90": float(lo), "hi90": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def per_fold_func(fr, pred) -> dict:
    out = {}
    for k, g in fr.assign(_p=pred).groupby("fold"):
        out[int(k)] = func_metrics(g, g._p.to_numpy())["primary"]
    return out


def run_full(params, wmode, tag, seeds=None, cols=None):
    fr = load_frame()
    cols = cols or feat_cols(fr)
    P, iters = oof(params, fr, cols, list(range(6)), wmode, seeds=seeds)
    pred = np.array(CLASSES5)[P.argmax(1)]
    m = func_metrics(fr, pred)
    m["per_fold_primary"] = per_fold_func(fr, pred)
    conf = P.max(1)
    for th in (0.7, 0.9):
        sel = conf >= th
        m[f"cov{th}"] = float(sel.mean())
        m[f"acc_at_cov{th}"] = float((fr.func5.to_numpy()[sel] == pred[sel]).mean())
    out = fr[["DeviceId", "Detector", "win", "fold", "func5", "health_flag"]].copy()
    out[[f"p_{c.lower()}" for c in CLASSES5]] = P
    out["pred5"] = pred
    out.to_parquet(TUNE / f"func_{tag}.parquet", index=False)
    json.dump({"metrics": m, "iters": iters}, open(TUNE / f"metrics_func_{tag}.json", "w"),
              indent=1, default=str)
    log(f"{tag}: {m}")
    return fr, pred, P


def stage_full(a) -> None:
    meta = json.load(open(TUNE / "best_function.json"))
    p = meta["full_params"]
    p["n_jobs"] = N_JOBS
    fr, pt, _ = run_full(p, meta["class_weight"], "tuned")
    if not (TUNE / "func_champ.parquet").exists():
        run_full(_champ_params(), "none", "champ")
    pc = pd.read_parquet(TUNE / "func_champ.parquet").pred5.to_numpy()
    b = boot_func(fr, pc, pt)
    pf_a, pf_b = per_fold_func(fr, pc), per_fold_func(fr, pt)
    dsig = {k: pf_b[k] - pf_a[k] for k in pf_a}
    res = {"bootstrap": b, "per_fold_diff": dsig,
           "n_folds_better": int(sum(v > 0 for v in dsig.values())),
           "metrics_champ": func_metrics(fr, pc), "metrics_tuned": func_metrics(fr, pt)}
    res["verdict"] = ("REAL" if b["excludes_zero"] or res["n_folds_better"] >= 5
                      else "not significant")
    dump(res, "compare_function_tuned_vs_champ.json")
    log(f"function tuned vs champ: {b['diff']:+.4f} [{b['lo90']:+.4f},{b['hi90']:+.4f}] "
        f"{res['n_folds_better']}/6 -> {res['verdict']}")


def stage_variants(a) -> None:
    """Separate the two things the search changed: the tree parameters and the class
    weighting.  The weighting lifts the rare classes (Other / Yellow_Red) but can cost
    accuracy on the three classes the user cares most about."""
    meta = json.load(open(TUNE / "best_function.json"))
    p = dict(meta["full_params"])
    p["n_jobs"] = N_JOBS
    fr, _, _ = run_full(p, "none", "tuned_noweight")
    cp = _champ_params()
    run_full(cp, meta["class_weight"], "champ_weighted")
    out = {}
    base = pd.read_parquet(TUNE / "func_champ.parquet").pred5.to_numpy()
    for tag in ("tuned", "tuned_noweight", "champ_weighted", "champ"):
        f = TUNE / f"func_{tag}.parquet"
        if not f.exists():
            continue
        pr = pd.read_parquet(f).pred5.to_numpy()
        m = func_metrics(fr, pr)
        m["per_fold_primary"] = per_fold_func(fr, pr)
        m["vs_champ"] = boot_func(fr, base, pr)
        out[tag] = m
        log(f"{tag}: primary {m['primary']:.4f} acc72 {m['acc72']:.4f} "
            f"APC72 {m['acc_apc72']:.4f} diff {m['vs_champ']['diff']:+.4f}")
    dump(out, "function_variants.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
