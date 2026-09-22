"""Stage 07 task 1: Optuna (TPE) tuning of the LightGBM pair ranker and joint decoder.

    python src/tune/tune_phase.py --stage ranker_search  --trials 80 --timeout 5400
    python src/tune/tune_phase.py --stage ranker_full    --params best_ranker.json
    python src/tune/tune_phase.py --stage decoder_search --trials 80
    python src/tune/tune_phase.py --stage decoder_full

Discipline: the search only ever sees folds 1-5.  Fold 0 (the 38-signal 2025 hold-out) is
scored once, by `ranker_full` / `decoder_full`, over the complete 22-window mix.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from tune_common import (FOLDS_V3, FULL, M30, N_JOBS, SEARCH_WINS, SIM_FILES,  # noqa: E402
                         TUNE, add_scorable, dump, feature_cols, load_phase_pairs, log,
                         paired_bootstrap, per_fold_primary, phase_metrics, top1_table,
                         fold_agreement, verdict)

SEARCH_FOLDS = [1, 2, 3, 4, 5]
INNER_FRAC = 0.15
EARLY = 100
MAX_TREES = 3000


# --------------------------------------------------------------------- helpers
def group_sizes(df: pd.DataFrame) -> np.ndarray:
    key = (df["win"].astype(str) + "|" + df.DeviceId + "|" +
           df.Detector.astype(str)).to_numpy()
    _, idx, cnt = np.unique(key, return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


def to_prob(df: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    g = (df["win"].astype(str) + "|" + df.DeviceId + "|" + df.Detector.astype(str))
    d = pd.DataFrame({"g": g.to_numpy(), "s": np.asarray(s, dtype=np.float64)})
    d["s"] = np.exp(d.s - d.groupby("g")["s"].transform("max"))
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def norm_prob(df: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    g = (df["win"].astype(str) + "|" + df.DeviceId + "|" + df.Detector.astype(str))
    d = pd.DataFrame({"g": g.to_numpy(), "s": np.clip(np.asarray(s, float), 1e-9, None)})
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def inner_split(tr: pd.DataFrame, seed: int) -> np.ndarray:
    """Signal-grouped inner validation mask inside the training folds (fixed per fold)."""
    devs = np.sort(tr.DeviceId.unique())
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(devs, max(1, int(round(INNER_FRAC * len(devs)))), replace=False))
    return tr.DeviceId.isin(pick).to_numpy()


def fit_score(params: dict, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
              cols: list[str]) -> tuple[np.ndarray, int]:
    P = dict(params)
    obj = P.get("objective", "lambdarank")
    n_est = P.pop("n_estimators", MAX_TREES)
    cb = [lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(0)]
    if obj == "binary":
        m = lgb.LGBMClassifier(n_estimators=n_est, **P)
        m.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)], eval_metric="binary_logloss",
              callbacks=cb)
        s = m.predict_proba(te[cols])[:, 1]
        return norm_prob(te, s), int(m.best_iteration_ or n_est)
    m = lgb.LGBMRanker(n_estimators=n_est, **P)
    m.fit(tr[cols], tr.y, group=group_sizes(tr),
          eval_set=[(va[cols], va.y)], eval_group=[group_sizes(va)], callbacks=cb)
    return to_prob(te, m.predict(te[cols])), int(m.best_iteration_ or n_est)


def oof(params: dict, df: pd.DataFrame, cols: list[str], folds: list[int],
        trainable: np.ndarray, report=None) -> tuple[pd.DataFrame, list[int]]:
    """Grouped OOF over `folds`; training pool = the other folds in `folds`."""
    pool = df.fold.isin(folds).to_numpy()
    parts, iters = [], []
    for j, k in enumerate(folds):
        te = (df.fold == k).to_numpy()
        trall = pool & ~te & trainable
        sub = df[trall]
        vmask = inner_split(sub, seed=1000 + k)
        tr, va = sub[~vmask], sub[vmask]
        p, it = fit_score(params, tr, va, df[te], cols)
        iters.append(it)
        t = top1_table(df[te], p)
        parts.append(t)
        if report is not None:
            report(j, pd.concat(parts, ignore_index=True))
    return pd.concat(parts, ignore_index=True), iters


def primary_of(parts: pd.DataFrame) -> float:
    return phase_metrics(parts)["primary"]


# ---------------------------------------------------------------- ranker search
def suggest_ranker(trial) -> dict:
    obj = trial.suggest_categorical("objective", ["lambdarank", "rank_xendcg", "binary"])
    p = dict(objective=obj, n_jobs=N_JOBS, verbose=-1, n_estimators=MAX_TREES,
             learning_rate=trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
             num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 5, 300, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.25, 1.0),
             bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
             bagging_freq=1,
             lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 20.0, log=True),
             lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 50.0, log=True),
             min_split_gain=trial.suggest_float("min_split_gain", 1e-6, 1.0, log=True),
             max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511]))
    if obj == "lambdarank":
        p["label_gain"] = [0, 1]
        p["metric"] = "ndcg"
        p["ndcg_eval_at"] = [1]
        p["lambdarank_truncation_level"] = trial.suggest_int("truncation", 3, 40)
    elif obj == "rank_xendcg":
        p["label_gain"] = [0, 1]
        p["metric"] = "ndcg"
        p["ndcg_eval_at"] = [1]
    return p


def stage_ranker_search(a) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = add_scorable(load_phase_pairs())
    df = df[df.win.isin(SEARCH_WINS) & df.fold.isin(SEARCH_FOLDS) & df.labelled]
    df = df.reset_index(drop=True)
    cols = feature_cols(df)
    trainable = df.health_flag.ne("failed").to_numpy()
    log(f"ranker search frame {df.shape}, {len(cols)} features, "
        f"{df.DeviceId.nunique()} signals, windows {sorted(df.win.unique())}")

    t0 = time.time()

    def objective(trial):
        p = suggest_ranker(trial)

        def rep(j, parts):
            trial.report(primary_of(parts), j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        parts, iters = oof(p, df, cols, SEARCH_FOLDS, trainable, report=rep)
        trial.set_user_attr("iters", iters)
        m = phase_metrics(parts)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["primary"]

    sampler = optuna.samplers.TPESampler(seed=7, n_startup_trials=12,
                                         multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    st = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                             study_name="ranker")
    # seed the search with the champion's own parameters
    st.enqueue_trial({"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 63,
                      "min_child_samples": 40, "feature_fraction": 0.7,
                      "bagging_fraction": 0.8, "lambda_l1": 1e-4, "lambda_l2": 1.0,
                      "min_split_gain": 1e-6, "max_bin": 255, "truncation": 30})
    st.optimize(objective, n_trials=a.trials, timeout=a.timeout,
                callbacks=[lambda s, t: log(
                    f"trial {t.number} {t.state.name} value={t.value} "
                    f"best={s.best_value:.5f} [{time.time()-t0:.0f}s]")])
    rows = [dict(number=t.number, value=t.value, state=t.state.name, **t.params,
                 iters=t.user_attrs.get("iters"),
                 acc72=t.user_attrs.get("acc72"), acc30=t.user_attrs.get("acc30"))
            for t in st.trials]
    pd.DataFrame(rows).to_csv(TUNE / "ranker_trials.csv", index=False)
    best = suggest_ranker(optuna.trial.FixedTrial(st.best_params))
    dump({"best_params": st.best_params, "best_value": st.best_value,
          "full_params": {k: v for k, v in best.items()},
          "n_trials": len(st.trials), "seconds": time.time() - t0,
          "search_windows": SEARCH_WINS, "search_folds": SEARCH_FOLDS},
         "best_ranker.json")
    log(f"BEST {st.best_value:.5f} {st.best_params}")


# ------------------------------------------------------- full 22-window 6-fold
def _champ_ranker_params() -> dict:
    return dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                learning_rate=0.05, num_leaves=63, min_child_samples=40,
                feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, n_estimators=1200, n_jobs=N_JOBS, verbose=-1,
                label_gain=[0, 1])


def run_full_ranker(params: dict, tag: str, seeds: list[int] | None = None,
                    cols: list[str] | None = None) -> pd.DataFrame:
    """6-fold OOF over every window and every active detector (labelled or not).

    Scores of `seeds` models are averaged in probability space when more than one seed is
    given (task 2, seed bagging)."""
    df = add_scorable(load_phase_pairs())
    allcols = feature_cols(df) if cols is None else cols
    trainable = (df.health_flag.ne("failed") & df.labelled).to_numpy()
    seeds = seeds or [0]
    P = np.zeros(len(df))
    iters = []
    for k in range(6):
        te = (df.fold == k).to_numpy()
        sub = df[~te & trainable]
        vmask = inner_split(sub, seed=1000 + k)
        tr, va = sub[~vmask], sub[vmask]
        acc = np.zeros(int(te.sum()))
        for s in seeds:
            p = dict(params)
            if len(seeds) > 1:
                p["seed"] = s
                p["bagging_seed"] = s + 101
                p["feature_fraction_seed"] = s + 202
            pr, it = fit_score(p, tr, va, df[te], allcols)
            acc += pr
            iters.append(it)
        P[te] = acc / len(seeds)
        log(f"  fold {k} done ({len(seeds)} seed(s))")
    out = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    out["p0"] = P
    out.to_parquet(TUNE / f"p0_{tag}.parquet", index=False)
    t = top1_table(df, P)
    t.to_parquet(TUNE / f"top1_{tag}.parquet", index=False)
    json.dump({"metrics": phase_metrics(t), "iters": iters,
               "per_fold_primary": {int(a): float(b) for a, b in
                                    per_fold_primary(t, None).items()}},
              open(TUNE / f"metrics_{tag}.json", "w"), indent=1, default=str)
    log(f"{tag}: {phase_metrics(t)}")
    return t


def compare(tag_a: str, tag_b: str, name: str) -> dict:
    a = pd.read_parquet(TUNE / f"top1_{tag_a}.parquet")
    b = pd.read_parquet(TUNE / f"top1_{tag_b}.parquet")
    folds = pd.read_csv(FOLDS_V3)
    boot = paired_bootstrap(a, b)
    ag = fold_agreement(a, b, folds)
    res = {"a": tag_a, "b": tag_b, "metrics_a": phase_metrics(a),
           "metrics_b": phase_metrics(b), "bootstrap": boot, "folds": ag,
           "verdict": verdict(boot, ag)}
    dump(res, f"compare_{name}.json")
    log(f"{name}: diff {boot['diff']:+.4f} [{boot['lo90']:+.4f},{boot['hi90']:+.4f}] "
        f"{ag['n_folds_better']}/6 folds -> {res['verdict']}")
    return res


def stage_ranker_full(a) -> None:
    best = json.load(open(TUNE / "best_ranker.json"))["full_params"]
    best["n_estimators"] = MAX_TREES
    best["n_jobs"] = N_JOBS
    log(f"full 22-window 6-fold with tuned ranker params: {best}")
    run_full_ranker(best, "ranker_tuned")
    # champion ranker inside the same harness, for a paired comparison
    if not (TUNE / "top1_ranker_champ.parquet").exists():
        run_full_ranker(_champ_ranker_params(), "ranker_champ")
    compare("ranker_champ", "ranker_tuned", "ranker_tuned_vs_champ")
    compare("champion_ranker", "ranker_tuned", "ranker_tuned_vs_stage06")


# ------------------------------------------------------------- decoder search
_DEC_CACHE: dict = {}


def _decoder_inputs():
    """ctx / key / similarity tables are the same for every first stage -- load once."""
    if not _DEC_CACHE:
        pairs = add_scorable(load_phase_pairs())
        _DEC_CACHE["ctx"] = pairs[["DeviceId", "Detector", "win", "cand_phase",
                                   "cand_green_share", "call43_per_cycle", "det_n_on",
                                   "log_win_hours"]].copy()
        _DEC_CACHE["keys"] = pairs[["DeviceId", "Detector", "win", "cand_phase", "Phase",
                                    "fold", "scorable", "labelled", "health_flag"]].copy()
        _DEC_CACHE["sim"] = pd.concat([pd.read_parquet(f) for f in SIM_FILES],
                                      ignore_index=True)
        del pairs
    return _DEC_CACHE


def decoder_frame(p0_file: Path) -> pd.DataFrame:
    import decode_v2 as dc
    C = _decoder_inputs()
    ctx, keys = C["ctx"], C["keys"]
    pr = pd.read_parquet(p0_file)
    pr["Detector"] = pr.Detector.astype(ctx.Detector.dtype)
    pr["cand_phase"] = pr.cand_phase.astype(ctx.cand_phase.dtype)
    X = dc.assemble(pr, pairs=ctx, sim=C["sim"])
    X = X.merge(keys, on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    X["y"] = (X.cand_phase == X.Phase).fillna(False).astype(np.int8)
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    return X


DEC_COLS = None


def dec_cols() -> list[str]:
    import decode_v2 as dc
    return dc.BASE_COLS + dc.SIM_COLS + dc.ADJ_COLS


def suggest_decoder(trial) -> dict:
    obj = trial.suggest_categorical("objective", ["binary", "lambdarank", "rank_xendcg"])
    p = dict(objective=obj, n_jobs=N_JOBS, verbose=-1, n_estimators=MAX_TREES,
             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
             num_leaves=trial.suggest_int("num_leaves", 7, 255, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 5, 500, log=True),
             feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
             bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
             bagging_freq=1,
             lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 20.0, log=True),
             lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 50.0, log=True),
             min_split_gain=trial.suggest_float("min_split_gain", 1e-6, 1.0, log=True),
             max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511]))
    if obj != "binary":
        p["label_gain"] = [0, 1]
        p["metric"] = "ndcg"
        p["ndcg_eval_at"] = [1]
    return p


def stage_decoder_search(a) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    src = TUNE / f"p0_{a.p0}.parquet"
    X = decoder_frame(src)
    X.to_parquet(TUNE / f"decframe_{a.p0}.parquet", index=False)
    cols = dec_cols()
    Xs = X[X.win.isin(SEARCH_WINS) & X.fold.isin(SEARCH_FOLDS)].reset_index(drop=True)
    trainable = (Xs.labelled & Xs.health_flag.ne("failed")).to_numpy()
    log(f"decoder search frame {Xs.shape}, {len(cols)} features (p0 = {a.p0})")
    t0 = time.time()

    def objective(trial):
        p = suggest_decoder(trial)

        def rep(j, parts):
            trial.report(primary_of(parts), j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        parts, iters = oof(p, Xs, cols, SEARCH_FOLDS, trainable, report=rep)
        trial.set_user_attr("iters", iters)
        m = phase_metrics(parts)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["primary"]

    st = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=11, n_startup_trials=12,
                                           multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    st.enqueue_trial({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                      "min_child_samples": 60, "feature_fraction": 0.8,
                      "bagging_fraction": 0.8, "lambda_l1": 1e-4, "lambda_l2": 1.0,
                      "min_split_gain": 1e-6, "max_bin": 255})
    st.optimize(objective, n_trials=a.trials, timeout=a.timeout,
                callbacks=[lambda s, t: log(
                    f"trial {t.number} {t.state.name} value={t.value} "
                    f"best={s.best_value:.5f} [{time.time()-t0:.0f}s]")])
    pd.DataFrame([dict(number=t.number, value=t.value, state=t.state.name, **t.params,
                       acc72=t.user_attrs.get("acc72"), acc30=t.user_attrs.get("acc30"),
                       iters=t.user_attrs.get("iters")) for t in st.trials]
                 ).to_csv(TUNE / "decoder_trials.csv", index=False)
    best = suggest_decoder(optuna.trial.FixedTrial(st.best_params))
    dump({"best_params": st.best_params, "best_value": st.best_value,
          "full_params": best, "p0": a.p0, "n_trials": len(st.trials),
          "seconds": time.time() - t0}, "best_decoder.json")
    log(f"BEST {st.best_value:.5f} {st.best_params}")


def run_full_decoder(params: dict, X: pd.DataFrame, tag: str,
                     seeds: list[int] | None = None) -> pd.DataFrame:
    cols = dec_cols()
    trainable = (X.labelled & X.health_flag.ne("failed")).to_numpy()
    seeds = seeds or [0]
    P = np.zeros(len(X))
    for k in range(6):
        te = (X.fold == k).to_numpy()
        sub = X[~te & trainable]
        vmask = inner_split(sub, seed=1000 + k)
        tr, va = sub[~vmask], sub[vmask]
        acc = np.zeros(int(te.sum()))
        for s in seeds:
            p = dict(params)
            if len(seeds) > 1:
                p["seed"] = s
                p["bagging_seed"] = s + 101
                p["feature_fraction_seed"] = s + 202
            pr, _ = fit_score(p, tr, va, X[te], cols)
            acc += pr
        P[te] = acc / len(seeds)
        log(f"  decoder fold {k} done")
    out = X[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    out["prob"] = P
    out.to_parquet(TUNE / f"p2_{tag}.parquet", index=False)
    t = top1_table(X, P)
    t.to_parquet(TUNE / f"top1_{tag}.parquet", index=False)
    json.dump({"metrics": phase_metrics(t),
               "per_fold_primary": {int(a): float(b) for a, b in
                                    per_fold_primary(t, None).items()}},
              open(TUNE / f"metrics_{tag}.json", "w"), indent=1, default=str)
    log(f"{tag}: {phase_metrics(t)}")
    return t


# ----------------------------------------------------------------- noise floor
def _champ_decoder_params() -> dict:
    import decode_v2 as dc
    p = dict(dc.BIN_PARAMS)
    p["n_estimators"] = MAX_TREES
    p["n_jobs"] = N_JOBS
    return p


def stage_noise_floor(a) -> None:
    """How much does the DECODED metric move when nothing real changes?

    Stage 08 found that the joint decoder's OOF accuracy shifts by ~+-0.2 pt under any
    perturbation of the first stage -- even 82 columns of pure noise "gained" +0.16 pt.
    So the champion pipeline is re-run end to end with S different random seeds (ranker and
    decoder alike) and the run-to-run sd of the decoded metric is the floor that any claimed
    gain has to clear.  The same runs are then averaged in disjoint groups to measure how
    much seed bagging shrinks that floor (K = 1 vs 2 vs 3).
    """
    import decode_v2 as dc
    S = a.seeds
    df = add_scorable(load_phase_pairs())
    res = {"single": {}, "bags": {}}
    tags = []
    for s in range(S):
        tag = f"nf_r{s}"
        if not (TUNE / f"p0_{tag}.parquet").exists():
            p = _champ_ranker_params()
            p.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202,
                     data_random_seed=s + 303, n_estimators=MAX_TREES)
            run_full_ranker(p, tag)
        tags.append(tag)
    for s, tag in enumerate(tags):
        dtag = f"nf_d{s}"
        if not (TUNE / f"top1_{dtag}.parquet").exists():
            X = decoder_frame(TUNE / f"p0_{tag}.parquet")
            dp = _champ_decoder_params()
            dp.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202)
            run_full_decoder(dp, X, dtag)
            del X
        res["single"][dtag] = phase_metrics(
            pd.read_parquet(TUNE / f"top1_{dtag}.parquet"))
    for K in (2, 3):
        if S // K < 2:
            continue
        for g in range(S // K):
            mem = tags[g * K:(g + 1) * K]
            btag = f"nf_bag{K}_{g}"
            if not (TUNE / f"top1_{btag}.parquet").exists():
                acc = None
                key = ["DeviceId", "Detector", "win", "cand_phase"]
                for m in mem:
                    q = pd.read_parquet(TUNE / f"p0_{m}.parquet")
                    acc = q.p0.to_numpy() if acc is None else acc + q.p0.to_numpy()
                bl = q[key].copy()
                bl["p0"] = acc / len(mem)
                f = TUNE / f"p0_{btag}.parquet"
                bl.to_parquet(f, index=False)
                X = decoder_frame(f)
                dp = _champ_decoder_params()
                dp.update(seed=g, bagging_seed=g + 101, feature_fraction_seed=g + 202)
                run_full_decoder(dp, X, btag)
                del X
            res["bags"].setdefault(f"K{K}", {})[btag] = phase_metrics(
                pd.read_parquet(TUNE / f"top1_{btag}.parquet"))
    summ = {}
    for name, group in [("K1", res["single"])] + [(k, v) for k, v in res["bags"].items()]:
        for metric in ("primary", "acc72", "acc30", "nonstd72"):
            vals = [g[metric] for g in group.values()]
            summ.setdefault(name, {})[metric] = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))
                if len(vals) > 1 else None, "n_runs": len(vals),
                "min": float(np.min(vals)), "max": float(np.max(vals))}
    res["summary"] = summ
    dump(res, "noise_floor.json")
    log(json.dumps(summ, indent=1))


def stage_noise_control(a) -> None:
    """The sibling stage's control: add pure-noise columns to the ranker and re-decode."""
    rng = np.random.default_rng(99)
    df = add_scorable(load_phase_pairs())
    cols = feature_cols(df)
    noise = {f"noise_{i}": rng.standard_normal(len(df)).astype(np.float32)
             for i in range(82)}
    df = pd.concat([df, pd.DataFrame(noise, index=df.index)], axis=1)
    run_full_ranker(_champ_ranker_params() | {"n_estimators": MAX_TREES},
                    "ranker_noise82", cols=cols + list(noise))
    X = decoder_frame(TUNE / "p0_ranker_noise82.parquet")
    run_full_decoder(_champ_decoder_params(), X, "decoded_noise82")


def stage_decoder_full(a) -> None:
    meta = json.load(open(TUNE / "best_decoder.json"))
    p = meta["full_params"]
    p["n_estimators"] = MAX_TREES
    p["n_jobs"] = N_JOBS
    X = pd.read_parquet(TUNE / f"decframe_{meta['p0']}.parquet")
    run_full_decoder(p, X, "decoded_tuned")
    compare("champion_decoded", "decoded_tuned", "decoded_tuned_vs_champ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--p0", default="ranker_tuned")
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
