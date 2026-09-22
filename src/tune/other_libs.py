"""Stage 07 task 4: XGBoost / CatBoost / sklearn on the SAME folds and the SAME features.

Protocol (identical for every library, so the table is fair):
  * search: 40 Optuna trials, 5-fold grouped OOF over folds 1-5 only, 9-window subset
    (`SEARCH_WINS`), metric = the stage-07 primary (mean of 72 h and 30 min top-1 accuracy);
  * `--stage full` then refits the winner over all 22 windows and all 6 folds, feeds the
    result through the *unchanged* champion LightGBM joint decoder, and records train
    time, inference time and model size;
  * sklearn ExtraTrees / RandomForest / HistGradientBoosting are reference points only and
    are measured on the search protocol (they are far too slow to refit 6x22 windows).

    python src/tune/other_libs.py --stage search --lib xgb   --trials 40
    python src/tune/other_libs.py --stage search --lib cat   --trials 40
    python src/tune/other_libs.py --stage refs
    python src/tune/other_libs.py --stage full   --lib xgb
    python src/tune/other_libs.py --stage blend
    python src/tune/other_libs.py --stage function --lib xgb|cat
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from tune_common import (FOLDS_V3, FULL, M30, N_JOBS, SEARCH_WINS, TUNE,  # noqa: E402
                         add_scorable, dump, feature_cols, fold_agreement,
                         load_phase_pairs, log, paired_bootstrap, per_fold_primary,
                         phase_metrics, top1_table, verdict)
from tune_phase import (SEARCH_FOLDS, inner_split, norm_prob)  # noqa: E402

MAX_TREES = 3000
EARLY = 100


# ------------------------------------------------------------------- utilities
def qid_of(df: pd.DataFrame) -> np.ndarray:
    key = (df["win"].astype(str) + "|" + df.DeviceId + "|" +
           df.Detector.astype(str)).to_numpy()
    return pd.factorize(key)[0].astype(np.int32)


def _fit_xgb(params, tr, va, te, cols, ytr, yva):
    import xgboost as xgb
    P = dict(params)
    obj = P.pop("_objective")
    P.update(n_estimators=MAX_TREES, early_stopping_rounds=EARLY, tree_method="hist",
             n_jobs=N_JOBS, verbosity=0)
    if obj == "binary:logistic":
        m = xgb.XGBClassifier(objective=obj, eval_metric="logloss", **P)
        m.fit(tr[cols], ytr, eval_set=[(va[cols], yva)], verbose=False)
        s = m.predict_proba(te[cols])[:, 1]
    else:
        m = xgb.XGBRanker(objective=obj, eval_metric="ndcg@1", **P)
        m.fit(tr[cols], ytr, qid=qid_of(tr), eval_set=[(va[cols], yva)],
              eval_qid=[qid_of(va)], verbose=False)
        s = m.predict(te[cols])
        s = s - s.min() + 1e-6
    return norm_prob(te, s), m


def _fit_cat(params, tr, va, te, cols, ytr, yva):
    from catboost import CatBoostClassifier, CatBoostRanker, Pool
    P = dict(params)
    loss = P.pop("_loss")
    P.update(iterations=MAX_TREES, od_type="Iter", od_wait=EARLY, thread_count=N_JOBS,
             verbose=0, allow_writing_files=False)
    if loss == "Logloss":
        m = CatBoostClassifier(loss_function="Logloss", **P)
        m.fit(tr[cols], ytr, eval_set=(va[cols], yva), verbose=0)
        s = m.predict_proba(te[cols])[:, 1]
    else:
        ptr = Pool(tr[cols], ytr, group_id=qid_of(tr))
        pva = Pool(va[cols], yva, group_id=qid_of(va))
        m = CatBoostRanker(loss_function=loss, **P)
        m.fit(ptr, eval_set=pva, verbose=0)
        s = m.predict(te[cols])
        s = s - s.min() + 1e-6
    return norm_prob(te, s), m


FITTERS = {"xgb": _fit_xgb, "cat": _fit_cat}


def oof_lib(lib: str, params: dict, df: pd.DataFrame, cols: list[str],
            folds: list[int], trainable: np.ndarray, report=None):
    pool = df.fold.isin(folds).to_numpy()
    parts, models, secs = [], [], []
    for j, k in enumerate(folds):
        te = (df.fold == k).to_numpy()
        sub = df[pool & ~te & trainable]
        vm = inner_split(sub, seed=1000 + k)
        tr, va = sub[~vm], sub[vm]
        t0 = time.time()
        p, m = FITTERS[lib](params, tr, va, df[te], cols,
                            tr.y.to_numpy(), va.y.to_numpy())
        secs.append(time.time() - t0)
        models.append(m)
        parts.append(top1_table(df[te], p))
        if report is not None:
            report(j, pd.concat(parts, ignore_index=True))
    return pd.concat(parts, ignore_index=True), models, secs


# -------------------------------------------------------------- search spaces
def suggest_xgb(trial) -> dict:
    obj = trial.suggest_categorical("objective",
                                    ["rank:pairwise", "rank:ndcg", "binary:logistic"])
    p = {"_objective": obj,
         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
         "max_depth": trial.suggest_int("max_depth", 3, 12),
         "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 100, log=True),
         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 1.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
         "gamma": trial.suggest_float("gamma", 1e-6, 2.0, log=True),
         "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512])}
    if obj.startswith("rank"):
        p["lambdarank_pair_method"] = "mean"
        p["lambdarank_num_pair_per_sample"] = trial.suggest_int("pairs", 1, 8)
    return p


def suggest_cat(trial) -> dict:
    # QueryCrossEntropy is GPU-only in CatBoost; QuerySoftMax / PairLogit are its CPU
    # listwise / pairwise query losses.
    loss = trial.suggest_categorical(
        "loss", ["YetiRank", "QuerySoftMax", "PairLogit", "Logloss"])
    p = {"_loss": loss,
         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
         "depth": trial.suggest_int("depth", 4, 10),
         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50.0, log=True),
         "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
         "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
         "rsm": trial.suggest_float("rsm", 0.3, 1.0),
         "bootstrap_type": "Bernoulli",
         "subsample": trial.suggest_float("subsample", 0.5, 1.0)}
    return p


SUGGEST = {"xgb": suggest_xgb, "cat": suggest_cat}
ENQUEUE = {"xgb": {"objective": "rank:pairwise", "learning_rate": 0.05, "max_depth": 6,
                   "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.7,
                   "reg_alpha": 1e-4, "reg_lambda": 1.0, "gamma": 1e-6, "max_bin": 256,
                   "pairs": 4},
           "cat": {"loss": "YetiRank", "learning_rate": 0.05, "depth": 6,
                   "l2_leaf_reg": 3.0, "random_strength": 1.0, "border_count": 128,
                   "rsm": 0.7, "subsample": 0.8}}


def stage_search(a) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = add_scorable(load_phase_pairs())
    df = df[df.win.isin(SEARCH_WINS) & df.fold.isin(SEARCH_FOLDS) & df.labelled]
    df = df.reset_index(drop=True)
    cols = feature_cols(df)
    trainable = df.health_flag.ne("failed").to_numpy()
    log(f"{a.lib} search frame {df.shape}, {len(cols)} features")
    t0 = time.time()

    def objective(trial):
        p = SUGGEST[a.lib](trial)

        def rep(j, parts):
            trial.report(phase_metrics(parts)["primary"], j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        try:
            parts, _, secs = oof_lib(a.lib, p, df, cols, SEARCH_FOLDS, trainable, rep)
        except optuna.TrialPruned:
            raise
        except Exception as e:                       # an unsupported option combination
            log(f"  trial failed: {type(e).__name__}: {str(e)[:200]}")
            raise optuna.TrialPruned()
        m = phase_metrics(parts)
        trial.set_user_attr("fit_secs", float(np.mean(secs)))
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["primary"]

    st = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=23, n_startup_trials=10,
                                           multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1))
    st.enqueue_trial(ENQUEUE[a.lib])
    st.optimize(objective, n_trials=a.trials, timeout=a.timeout,
                callbacks=[lambda s, t: log(
                    f"[{a.lib}] trial {t.number} {t.state.name} value={t.value} "
                    f"best={s.best_value:.5f} [{time.time()-t0:.0f}s]")])
    pd.DataFrame([dict(number=t.number, value=t.value, state=t.state.name, **t.params,
                       acc72=t.user_attrs.get("acc72"), acc30=t.user_attrs.get("acc30"),
                       fit_secs=t.user_attrs.get("fit_secs")) for t in st.trials]
                 ).to_csv(TUNE / f"{a.lib}_trials.csv", index=False)
    best = SUGGEST[a.lib](optuna.trial.FixedTrial(st.best_params))
    dump({"best_params": st.best_params, "full_params": best,
          "best_value": st.best_value, "n_trials": len(st.trials),
          "seconds": time.time() - t0}, f"best_{a.lib}.json")
    log(f"[{a.lib}] BEST {st.best_value:.5f} {st.best_params}")


# ------------------------------------------------------- sklearn reference row
def stage_refs(a) -> None:
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    df = add_scorable(load_phase_pairs())
    df = df[df.win.isin(SEARCH_WINS) & df.fold.isin(SEARCH_FOLDS) & df.labelled]
    df = df.reset_index(drop=True)
    cols = feature_cols(df)
    trainable = df.health_flag.ne("failed").to_numpy()
    X = df[cols].to_numpy(dtype=np.float32)
    out = {}
    models = {
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            n_jobs=N_JOBS * 2, random_state=0),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            n_jobs=N_JOBS * 2, random_state=0),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, n_iter_no_change=50,
            validation_fraction=0.12, random_state=0),
    }
    for name, mk in models.items():
        parts, secs = [], []
        for k in SEARCH_FOLDS:
            te = (df.fold == k).to_numpy()
            trm = ~te & trainable
            t0 = time.time()
            m = mk()
            m.fit(np.nan_to_num(X[trm], nan=-999.0) if "Hist" not in name else X[trm],
                  df.y.to_numpy()[trm])
            secs.append(time.time() - t0)
            Xt = X[te] if "Hist" in name else np.nan_to_num(X[te], nan=-999.0)
            s = m.predict_proba(Xt)[:, 1]
            parts.append(top1_table(df[te], norm_prob(df[te], s)))
        t = pd.concat(parts, ignore_index=True)
        out[name] = phase_metrics(t)
        out[name]["fit_secs_mean"] = float(np.mean(secs))
        t.to_parquet(TUNE / f"top1_search_{name}.parquet", index=False)
        log(f"{name}: {out[name]}")
    dump(out, "sklearn_refs.json")


# ------------------------------------------------------ full 22-window + decode
def stage_full(a) -> None:
    import decode_v2 as dc
    from tune_phase import decoder_frame, run_full_decoder
    meta = json.load(open(TUNE / f"best_{a.lib}.json"))
    params = meta["full_params"]
    df = add_scorable(load_phase_pairs())
    cols = feature_cols(df)
    trainable = (df.health_flag.ne("failed") & df.labelled).to_numpy()
    P = np.zeros(len(df))
    secs, sizes = [], []
    for k in range(6):
        te = (df.fold == k).to_numpy()
        sub = df[~te & trainable]
        vm = inner_split(sub, seed=1000 + k)
        tr, va = sub[~vm], sub[vm]
        t0 = time.time()
        p, m = FITTERS[a.lib](params, tr, va, df[te], cols,
                              tr.y.to_numpy(), va.y.to_numpy())
        secs.append(time.time() - t0)
        P[te] = p
        if k == 0:
            f = TUNE / f"model_{a.lib}_fold0.{'json' if a.lib == 'xgb' else 'cbm'}"
            m.save_model(str(f))
            sizes.append(f.stat().st_size)
        log(f"  [{a.lib}] fold {k} in {secs[-1]:.0f}s")
    out = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    out["p0"] = P
    out.to_parquet(TUNE / f"p0_{a.lib}.parquet", index=False)
    t1 = top1_table(df, P)
    t1.to_parquet(TUNE / f"top1_{a.lib}_ranker.parquet", index=False)
    # inference timing on the 72 h slice, single model
    full = (df.win == FULL).to_numpy()
    Xf = df.loc[full, cols]
    t0 = time.time()
    if a.lib == "xgb":
        m.predict(Xf) if hasattr(m, "predict") else None
    else:
        m.predict(Xf)
    inf = time.time() - t0
    res = {"ranker": phase_metrics(t1), "fit_secs_per_fold": secs,
           "model_bytes_fold0": sizes[0] if sizes else None,
           "infer_secs_full72_rows": inf, "n_full72_rows": int(full.sum())}
    log(f"[{a.lib}] ranker {res['ranker']}")
    # feed the unchanged champion decoder
    X = decoder_frame(TUNE / f"p0_{a.lib}.parquet")
    dp = dict(dc.BIN_PARAMS)
    dp["n_jobs"] = N_JOBS
    dp["n_estimators"] = MAX_TREES
    t2 = run_full_decoder(dp, X, f"{a.lib}_decoded")
    res["decoded"] = phase_metrics(t2)
    dump(res, f"lib_{a.lib}_full.json")


def stage_lgbm_decode(a) -> None:
    """Champion-parameter decoder on top of the tuned-LGBM p0 (reference for the table)."""
    import decode_v2 as dc
    from tune_phase import decoder_frame, run_full_decoder
    X = decoder_frame(TUNE / f"p0_{a.p0}.parquet")
    dp = dict(dc.BIN_PARAMS)
    dp["n_jobs"] = N_JOBS
    dp["n_estimators"] = MAX_TREES
    run_full_decoder(dp, X, f"{a.p0}_decoded")


# ------------------------------------------------------------------- blending
def stage_blend(a) -> None:
    import decode_v2 as dc
    from tune_phase import decoder_frame, run_full_decoder
    key = ["DeviceId", "Detector", "win", "cand_phase"]
    df = add_scorable(load_phase_pairs())
    base = df[key + ["Phase", "fold", "scorable"]]
    avail = {n: TUNE / f"p0_{n}.parquet" for n in a.members.split(",")}
    combos = {"lgbm+cat": ["lgbm", "cat"], "lgbm+xgb": ["lgbm", "xgb"],
              "lgbm+cat+xgb": ["lgbm", "cat", "xgb"]}
    res = {}
    for name, mem in combos.items():
        if not all(m in avail for m in mem):
            continue
        acc = None
        for m in mem:
            p = pd.read_parquet(avail[m])
            p["Detector"] = p.Detector.astype(base.Detector.dtype)
            p["cand_phase"] = p.cand_phase.astype(base.cand_phase.dtype)
            p = base[key].merge(p, on=key, how="left").p0.to_numpy()
            acc = p if acc is None else acc + p
        acc = acc / len(mem)
        bl = df[key].copy()
        bl["p0"] = acc
        f = TUNE / f"p0_blend_{name.replace('+', '_')}.parquet"
        bl.to_parquet(f, index=False)
        t = top1_table(df, acc)
        res[name + "_ranker"] = phase_metrics(t)
        X = decoder_frame(f)
        dp = dict(dc.BIN_PARAMS)
        dp["n_jobs"] = N_JOBS
        dp["n_estimators"] = MAX_TREES
        td = run_full_decoder(dp, X, f"blend_{name.replace('+', '_')}_decoded")
        res[name + "_decoded"] = phase_metrics(td)
        log(f"blend {name}: {res[name + '_decoded']}")
    dump(res, "blends.json")


def stage_compare_blend(a) -> None:
    folds = pd.read_csv(FOLDS_V3)
    ref = pd.read_parquet(TUNE / f"top1_{a.ref}.parquet")
    out = {}
    for tag in a.members.split(","):
        b = pd.read_parquet(TUNE / f"top1_{tag}.parquet")
        boot = paired_bootstrap(ref, b)
        ag = fold_agreement(ref, b, folds)
        out[tag] = {"metrics": phase_metrics(b), "bootstrap": boot, "folds": ag,
                    "verdict": verdict(boot, ag)}
        log(f"{tag} vs {a.ref}: {boot['diff']:+.4f} [{boot['lo90']:+.4f},"
            f"{boot['hi90']:+.4f}] {ag['n_folds_better']}/6 -> {out[tag]['verdict']}")
    dump(out, f"compare_vs_{a.ref}.json")


# ------------------------------------------------------------------- function
def stage_function(a) -> None:
    """XGBoost multi:softprob / CatBoost MultiClass on the v3 function frame."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import tune_function as tf
    fr = tf.load_frame()
    cols = tf.feat_cols(fr)
    fs = fr[fr.win.isin(SEARCH_WINS) & fr.fold.isin(SEARCH_FOLDS)].reset_index(drop=True)
    y = fs.func5.map({c: i for i, c in enumerate(tf.CLASSES5)}).to_numpy()
    okm = (fs.health_flag != "failed").to_numpy()
    log(f"[{a.lib}] function search frame {fs.shape}")

    def run(params):
        P = np.zeros((len(fs), len(tf.CLASSES5)))
        secs = []
        for k in SEARCH_FOLDS:
            te = (fs.fold == k).to_numpy()
            trall = ~te & okm
            sub = fs[trall]
            vm = tf.inner_split(sub, seed=1000 + k)
            yy = y[trall]
            t0 = time.time()
            if a.lib == "xgb":
                import xgboost as xgb
                m = xgb.XGBClassifier(objective="multi:softprob", num_class=5,
                                      n_estimators=MAX_TREES, tree_method="hist",
                                      early_stopping_rounds=EARLY, n_jobs=N_JOBS,
                                      eval_metric="mlogloss", verbosity=0, **params)
                m.fit(sub[~vm][cols], yy[~vm],
                      eval_set=[(sub[vm][cols], yy[vm])], verbose=False)
            else:
                from catboost import CatBoostClassifier
                m = CatBoostClassifier(loss_function="MultiClass", iterations=MAX_TREES,
                                       od_type="Iter", od_wait=EARLY, verbose=0,
                                       thread_count=N_JOBS, allow_writing_files=False,
                                       **params)
                m.fit(sub[~vm][cols], yy[~vm], eval_set=(sub[vm][cols], yy[vm]), verbose=0)
            secs.append(time.time() - t0)
            P[te] = m.predict_proba(fs.loc[te, cols])
        return P, float(np.mean(secs))

    def objective(trial):
        if a.lib == "xgb":
            p = {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
                 "max_depth": trial.suggest_int("max_depth", 3, 10),
                 "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 100, log=True),
                 "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                 "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True)}
        else:
            p = {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
                 "depth": trial.suggest_int("depth", 4, 9),
                 "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50.0, log=True),
                 "rsm": trial.suggest_float("rsm", 0.3, 1.0)}
        P, s = run(p)
        m = tf.func_metrics(fs, np.array(tf.CLASSES5)[P.argmax(1)])
        trial.set_user_attr("fit_secs", s)
        for k, v in m.items():
            trial.set_user_attr(k, v)
        return m["primary"]

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=31,
                                                                n_startup_trials=8))
    st.optimize(objective, n_trials=a.trials, timeout=a.timeout,
                callbacks=[lambda s, t: log(
                    f"[{a.lib}-func] trial {t.number} value={t.value} "
                    f"best={s.best_value:.5f}")])
    dump({"best_params": st.best_params, "best_value": st.best_value,
          "acc72": st.best_trial.user_attrs.get("acc72"),
          "acc_apc72": st.best_trial.user_attrs.get("acc_apc72"),
          "fit_secs": st.best_trial.user_attrs.get("fit_secs"),
          "n_trials": len(st.trials)}, f"best_function_{a.lib}.json")
    log(f"[{a.lib}-func] BEST {st.best_value:.5f} {st.best_params}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--lib", default="xgb")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--members", default="lgbm,cat,xgb")
    ap.add_argument("--ref", default="champion_decoded")
    ap.add_argument("--p0", default="ranker_tuned")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
