"""Stage 01 step E: LightGBM phase ranker + function classifier with 6-fold grouped OOF.

    python src/train_lgbm.py --stage main        # tune (small grid), OOF, preds, models
    python src/train_lgbm.py --stage ablations   # cheap ablations + duration curve

All model selection uses out-of-fold predictions grouped by signal (folds.csv).
TEST signals are never loaded here.
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
import evaluate as ev  # noqa: E402
from common import (CACHE, DEFAULT_PHASE, FEATURES, FOLDS_CSV, FUNCTIONS,  # noqa: E402
                    LABELS_DEV, MODELS, N_FOLDS, PREDS)
from features import CALL_FEATS, FEATURE_COLS, win_hours  # noqa: E402

FEAT_FILE = FEATURES / "pair_features_windows.parquet"
GROUP_KEYS = ["DeviceId", "Detector", "win"]
SHORT_WINS = ["m30_a", "m30_b", "m30_c", "m30_d", "h1_a", "h1_b", "h1_c"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- loading
HEALTH_FILE = Path(str(CACHE).replace("cache", "atspm")) / "detector_health.parquet"


def load_health() -> pd.DataFrame:
    """Label-free per-channel health flag from stage 02 (healthy / suspect / failed)."""
    if not HEALTH_FILE.exists():
        return pd.DataFrame(columns=["DeviceId", "Detector", "health_flag"])
    h = pd.read_parquet(HEALTH_FILE, columns=["DeviceId", "Detector", "health_flag"])
    h["Detector"] = h["Detector"].astype(np.int16)
    return h


def load_pairs(feat_file: Path = FEAT_FILE) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(feat_file)
    folds = pd.read_csv(FOLDS_CSV)
    df = df.merge(folds, on="DeviceId", how="inner")        # DEV only, drops TEST
    lab = pd.read_parquet(LABELS_DEV)
    df = df.merge(lab, on=["DeviceId", "Detector"], how="left")
    h = load_health()
    if len(h):
        df = df.merge(h, on=["DeviceId", "Detector"], how="left")
        df["health_flag"] = df.health_flag.fillna("unknown")
    else:
        df["health_flag"] = "unknown"
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    feat_cols = FEATURE_COLS(df)
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    return df, feat_cols


def _groups(df: pd.DataFrame) -> np.ndarray:
    key = df["win"].astype(str) + "|" + df["DeviceId"] + "|" + df["Detector"].astype(str)
    _, idx, cnt = np.unique(key.to_numpy(), return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


# ------------------------------------------------------------------ training
BIN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=1200, n_jobs=10,
                  verbose=-1)
RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                   learning_rate=0.05, num_leaves=63, min_child_samples=40,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=1.0, n_estimators=1200, n_jobs=10, verbose=-1,
                   label_gain=[0, 1])


def fit_one(tr: pd.DataFrame, va: pd.DataFrame, feat_cols, mode: str, params: dict):
    P = dict(params)
    n_est = P.pop("n_estimators", 1000)
    cb = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    if mode == "binary":
        m = lgb.LGBMClassifier(n_estimators=n_est, **P)
        m.fit(tr[feat_cols], tr.y, eval_set=[(va[feat_cols], va.y)],
              eval_metric="binary_logloss", callbacks=cb)
    else:
        m = lgb.LGBMRanker(n_estimators=n_est, **P)
        m.fit(tr[feat_cols], tr.y, group=_groups(tr),
              eval_set=[(va[feat_cols], va.y)], eval_group=[_groups(va)], callbacks=cb)
    return m


def score(m, X, mode: str) -> np.ndarray:
    return m.predict_proba(X)[:, 1] if mode == "binary" else m.predict(X)


def to_prob(df: pd.DataFrame, s: np.ndarray, mode: str) -> np.ndarray:
    """Normalise pair scores to a distribution over each detector's candidates."""
    d = pd.DataFrame({"g": df["win"].astype(str) + "|" + df.DeviceId + "|" + df.Detector.astype(str),
                      "s": s})
    if mode == "rank":
        d["s"] = d.s - d.groupby("g")["s"].transform("max")
        d["s"] = np.exp(d.s)
    else:
        d["s"] = d.s.clip(lower=1e-9)
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def oof_predict(df: pd.DataFrame, feat_cols, mode: str, params: dict,
                seed: int = 0) -> tuple[np.ndarray, list]:
    """6-fold grouped OOF.  Inner early-stopping split = one training fold."""
    s = np.zeros(len(df))
    models = []
    ok = df.health_flag.ne("failed") if "health_flag" in df.columns else pd.Series(True, index=df.index)
    for k in range(N_FOLDS):
        te = df.fold == k
        inner = (k + 1) % N_FOLDS
        if inner == k:
            inner = (k + 2) % N_FOLDS
        tr = df[(~te) & ok & (df.fold != inner)]
        va = df[(~te) & ok & (df.fold == inner)]
        m = fit_one(tr, va, feat_cols, mode, params)
        s[te.to_numpy()] = score(m, df.loc[te, feat_cols], mode)
        models.append(m)
        log(f"  fold {k}: n_tr={len(tr)} best_iter={m.best_iteration_}")
    return s, models


def top1_acc(df: pd.DataFrame, prob: np.ndarray, wins=None) -> float:
    d = df[["DeviceId", "Detector", "win", "cand_phase", "Phase"]].copy()
    d["p"] = prob
    if wins is not None:
        d = d[d.win.isin(wins)]
    d = d[d.Phase.notna()]
    i = d.groupby(GROUP_KEYS)["p"].idxmax()
    t = d.loc[i]
    return float((t.cand_phase == t.Phase).mean())


# ---------------------------------------------------------------- prediction
def phase_pred_file(df: pd.DataFrame, prob: np.ndarray, win: str) -> pd.DataFrame:
    d = df.loc[df.win == win, ["DeviceId", "Detector", "cand_phase"]].copy()
    d["prob"] = prob[(df.win == win).to_numpy()]
    d["prob"] = d.prob / d.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
    d["cand_phase"] = d.cand_phase.astype(int)
    d["Detector"] = d.Detector.astype(int)
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ function
DET_FEATS = ["det_n_on", "det_on_per_hour", "det_occ_frac", "det_dur_med", "det_dur_q90",
             "det_dur_mean", "det_dur_max", "det_frac_short", "det_frac_long",
             "log_det_n_on", "log_win_hours", "win_secs", "n_cand", "n_on_per_cycle"]


def function_frame(df: pd.DataFrame, prob: np.ndarray, feat_cols) -> pd.DataFrame:
    """Detector-level frame = features of the model's own top-1 phase pair."""
    d = df.copy()
    d["_p"] = prob
    i = d.groupby(GROUP_KEYS)["_p"].idxmax()
    top = d.loc[i].copy()
    top["top_prob"] = top["_p"]
    keep = [c for c in feat_cols if c in top.columns]
    cols = (["DeviceId", "Detector", "win", "fold", "Function", "cand_phase", "Phase", "top_prob",
             "health_flag"] + keep)
    return top[cols].reset_index(drop=True)


FUNC_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05,
                   num_leaves=31, min_child_samples=40, feature_fraction=0.7,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   n_estimators=1200, n_jobs=10, verbose=-1)


def oof_function(fr: pd.DataFrame, feat_cols) -> tuple[np.ndarray, list]:
    fr = fr[fr.Function.isin(FUNCTIONS)].reset_index(drop=True)
    y = fr.Function.map({f: i for i, f in enumerate(FUNCTIONS)}).to_numpy()
    P = np.zeros((len(fr), 3))
    models = []
    ok = (fr.health_flag != "failed").to_numpy() if "health_flag" in fr.columns \
        else np.ones(len(fr), bool)
    for k in range(N_FOLDS):
        te = (fr.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        trm = (~te) & ok & (fr.fold != inner).to_numpy()
        vam = (~te) & ok & (fr.fold == inner).to_numpy()
        p = dict(FUNC_PARAMS)
        n_est = p.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n_est, **p)
        m.fit(fr.loc[trm, feat_cols], y[trm],
              eval_set=[(fr.loc[vam, feat_cols], y[vam])], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        P[te] = m.predict_proba(fr.loc[te, feat_cols])
        models.append(m)
    return P, models, fr


# ---------------------------------------------------------------------- main
def stage_main(args) -> None:
    t0 = time.time()
    df, feat_cols = load_pairs()
    log(f"pairs={len(df):,}  features={len(feat_cols)}  windows={df.win.nunique()}  "
        f"signals={df.DeviceId.nunique()}")
    dtr = df[df.Phase.notna()].reset_index(drop=True)
    del df
    log(f"labelled pairs for training: {len(dtr):,} "
        f"({dtr.groupby(GROUP_KEYS).ngroups:,} detector-windows)")

    nfail = int((dtr.health_flag == "failed").sum())
    log(f"health flags: {dtr.groupby(GROUP_KEYS[:2]).health_flag.first().value_counts().to_dict()}"
        f"  ({nfail:,} pair rows flagged failed -> excluded from training, still scored)")

    # ---- modest hyper-parameter search on an inner split (folds 2-5 train, fold 1 valid)
    ok = dtr.health_flag != "failed"
    tune_tr = dtr[ok & ~dtr.fold.isin([0, 1])]
    tune_va = dtr[ok & (dtr.fold == 1)]
    grid = [dict(num_leaves=31, min_child_samples=40, feature_fraction=0.7),
            dict(num_leaves=63, min_child_samples=40, feature_fraction=0.7),
            dict(num_leaves=63, min_child_samples=100, feature_fraction=0.5),
            dict(num_leaves=127, min_child_samples=100, feature_fraction=0.5)]
    best = {}
    for mode, base in (("binary", BIN_PARAMS), ("rank", RANK_PARAMS)):
        rows = []
        for g in grid:
            p = {**base, **g}
            m = fit_one(tune_tr, tune_va, feat_cols, mode, p)
            a = top1_acc(tune_va, to_prob(tune_va, score(m, tune_va[feat_cols], mode), mode))
            rows.append((g, a, m.best_iteration_))
            log(f"  tune[{mode}] {g} -> inner acc {a:.4f} (iter {m.best_iteration_})")
        g, a, _ = max(rows, key=lambda r: r[1])
        best[mode] = {**base, **g}
        log(f"  best[{mode}] = {g}  inner acc {a:.4f}")

    results = {}
    outs = {}
    for mode in ("binary", "rank"):
        log(f"OOF {mode}")
        s, models = oof_predict(dtr, feat_cols, mode, best[mode])
        prob = to_prob(dtr, s, mode)
        outs[mode] = (s, prob, models)
        results[mode] = {"acc_full72": top1_acc(dtr, prob, ["full72"]),
                         "acc_allwin": top1_acc(dtr, prob)}
        log(f"  {mode}: full72 OOF acc {results[mode]['acc_full72']:.4f}, "
            f"all-window mean {results[mode]['acc_allwin']:.4f}")

    champ = max(results, key=lambda m: results[m]["acc_full72"])
    log(f"champion phase model: {champ}")
    s, prob, models = outs[champ]

    PREDS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for mode in ("binary", "rank"):
        pf = phase_pred_file(dtr, outs[mode][1], "full72")
        pf.to_parquet(PREDS / f"phase_lgbm_{mode}_oof.parquet", index=False)
    phase_pred_file(dtr, prob, "full72").to_parquet(PREDS / "phase_oof.parquet", index=False)
    bw = dtr[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    bw["prob"] = prob
    bw.to_parquet(PREDS / "phase_oof_bywindow.parquet", index=False)

    # ---- duration curve (mixed model)
    dur = []
    for w in sorted(dtr.win.unique(), key=win_hours):
        sub = dtr[dtr.win == w]
        acc = top1_acc(dtr, prob, [w])
        dur.append({"win": w, "hours": win_hours(w), "acc": acc,
                    "med_n_on": float(sub.det_n_on.median())})
    results["duration_curve"] = dur
    log("duration curve: " + ", ".join(f"{d['win']}={d['acc']:.3f}" for d in dur))
    # mean over windows of the same nominal duration
    byh: dict = {}
    for d in dur:
        byh.setdefault(d["hours"], []).append(d["acc"])
    results["duration_curve_mean"] = {h: float(np.mean(v)) for h, v in sorted(byh.items())}

    # ---- accuracy vs number of actuations in the window
    d_all = dtr[["DeviceId", "Detector", "win", "cand_phase", "Phase", "det_n_on"]].copy()
    d_all["p"] = prob
    tt = d_all.loc[d_all.groupby(GROUP_KEYS)["p"].idxmax()].copy()
    tt["ok"] = tt.cand_phase == tt.Phase
    bins = [0, 10, 25, 50, 100, 250, 500, 1000, 5000, 10**9]
    tt["nbin"] = pd.cut(tt.det_n_on, bins, right=False)
    acc_n = tt.groupby("nbin", observed=True).agg(n=("ok", "size"), acc=("ok", "mean"))
    results["acc_vs_actuations"] = {str(k): [int(v.n), float(v.acc)]
                                    for k, v in acc_n.iterrows()}
    log("acc vs actuations: " + str(results["acc_vs_actuations"]))

    # ---- feature importance
    imp = pd.DataFrame({"feature": feat_cols,
                        "gain": np.mean([m.booster_.feature_importance("gain")
                                         for m in models], axis=0)})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp.to_csv(PREDS / "feature_importance_phase.csv", index=False)
    results["top20"] = imp.head(20).to_dict("records")

    # ---- final models trained on all DEV (for later TEST inference)
    log("fitting final DEV-wide models")
    itr = dtr[dtr.fold != 0]
    iva = dtr[dtr.fold == 0]
    final = fit_one(itr, iva, feat_cols, champ, best[champ])
    final.booster_.save_model(str(MODELS / "phase_lgbm.txt"))
    json.dump({"mode": champ, "features": feat_cols, "params": best[champ]},
              open(MODELS / "phase_lgbm.json", "w"), indent=1)

    # ---- function model -------------------------------------------------------
    log("function model")
    fr = function_frame(dtr, prob, feat_cols)
    ffeats = [c for c in feat_cols if c in fr.columns] + ["top_prob"]
    Pf, fmodels, frl = oof_function(fr, ffeats)
    fout = frl[["DeviceId", "Detector", "win"]].copy()
    fout[["p_advance", "p_presence", "p_count"]] = Pf
    fout[fout.win == "full72"].drop(columns=["win"]).to_parquet(
        PREDS / "function_oof.parquet", index=False)
    fout.to_parquet(PREDS / "function_oof_bywindow.parquet", index=False)
    pred = np.array(FUNCTIONS)[Pf.argmax(1)]
    results["function"] = {
        "acc_allwin": float((pred == frl.Function).mean()),
        "acc_full72": float((pred[frl.win == "full72"] == frl.Function[frl.win == "full72"]).mean()),
    }
    log(f"  function OOF: full72 {results['function']['acc_full72']:.4f}, "
        f"all-window {results['function']['acc_allwin']:.4f}")
    fimp = pd.DataFrame({"feature": ffeats,
                         "gain": np.mean([m.booster_.feature_importance("gain")
                                          for m in fmodels], axis=0)}
                        ).sort_values("gain", ascending=False)
    fimp.to_csv(PREDS / "feature_importance_function.csv", index=False)
    ffinal = lgb.LGBMClassifier(n_estimators=max(m.best_iteration_ or 300 for m in fmodels),
                                **{k: v for k, v in FUNC_PARAMS.items() if k != "n_estimators"})
    frl3 = frl[frl.Function.isin(FUNCTIONS) & (frl.health_flag != "failed")]
    ffinal.fit(frl3[ffeats], frl3.Function.map({f: i for i, f in enumerate(FUNCTIONS)}))
    ffinal.booster_.save_model(str(MODELS / "function_lgbm.txt"))
    json.dump({"features": ffeats, "classes": list(FUNCTIONS)},
              open(MODELS / "function_lgbm.json", "w"), indent=1)

    # ---- error analysis / suspect labels --------------------------------------
    d72 = dtr[dtr.win == "full72"].copy()
    d72["prob"] = prob[(dtr.win == "full72").to_numpy()]
    i = d72.groupby(["DeviceId", "Detector"])["prob"].idxmax()
    t = d72.loc[i, ["DeviceId", "Detector", "cand_phase", "prob", "Phase", "Function"]].copy()
    t = t.rename(columns={"cand_phase": "predicted", "Phase": "label"})
    t["std_phase"] = t.Detector.map(DEFAULT_PHASE)
    t["correct"] = t.predicted == t.label
    wrong = t[~t.correct].sort_values("prob", ascending=False)
    wrong.head(40).to_csv(PREDS / "suspect_labels_lgbm.csv", index=False)
    review = pd.concat([wrong[wrong.prob >= 0.8], t[t.prob < 0.5]]).drop_duplicates()
    review.to_csv(PREDS / "review_list_lgbm.csv", index=False)
    results["error_analysis"] = {
        "n_wrong": int((~t.correct).sum()),
        "n_confident_wrong_0.8": int(((~t.correct) & (t.prob >= 0.8)).sum()),
        "n_confident_wrong_matching_std": int(((~t.correct) & (t.prob >= 0.8) &
                                               (t.predicted == t.std_phase)).sum()),
        "n_low_conf": int((t.prob < 0.5).sum()),
    }
    json.dump(results, open(PREDS / "train_lgbm_results.json", "w"), indent=1, default=str)
    log(f"stage main done in {time.time()-t0:.0f}s")

    print(ev.run(str(PREDS / "phase_oof.parquet"), str(PREDS / "function_oof.parquet"),
                 title="LGBM OOF (DEV, 72 h window)"))


# ---------------------------------------------------------------- ablations
def stage_ablations(args) -> None:
    t0 = time.time()
    df, feat_cols = load_pairs()
    dtr = df[df.Phase.notna()].reset_index(drop=True)
    res = json.load(open(PREDS / "train_lgbm_results.json"))
    champ = "rank" if res.get("rank", {}).get("acc_full72", 0) >= \
        res.get("binary", {}).get("acc_full72", 0) else "binary"
    base = RANK_PARAMS if champ == "rank" else BIN_PARAMS
    params = {**base, **dict(num_leaves=63, min_child_samples=40, feature_fraction=0.7)}
    out = {}

    def run(name, sub, cols):
        s, models = oof_predict(sub, cols, champ, params)
        p = to_prob(sub, s, champ)
        r = {"acc_full72": top1_acc(sub, p, ["full72"]),
             "acc_allwin": top1_acc(sub, p),
             "acc_short": top1_acc(sub, p, SHORT_WINS),
             "n_features": len(cols)}
        out[name] = r
        log(f"ablation {name}: {r}")
        return p

    def is_call(c: str) -> bool:
        return ("call4" in c) or ("n43" in c) or ("n44" in c) or ("recall" in c)

    run("full", dtr, feat_cols)
    run("no_calls", dtr, [c for c in feat_cols if not is_call(c)])
    run("no_rank_z", dtr, [c for c in feat_cols
                           if not c.endswith(("__rank", "__z", "__mgap", "__argmax"))])

    # 72h-only training vs mixed training, both evaluated on short windows
    d72 = dtr[dtr.win == "full72"].reset_index(drop=True)
    s72, m72 = oof_predict(d72, feat_cols, champ, params)
    out["train72_eval72"] = {"acc_full72": top1_acc(d72, to_prob(d72, s72, champ))}
    # apply the 72 h-trained fold models to short windows (out of fold by signal)
    sshort = np.zeros(len(dtr))
    for k in range(N_FOLDS):
        m = dtr.fold == k
        sshort[m.to_numpy()] = score(m72[k], dtr.loc[m, feat_cols], champ)
    pshort = to_prob(dtr, sshort, champ)
    out["train72_eval_short"] = {w: top1_acc(dtr, pshort, [w])
                                 for w in sorted(dtr.win.unique(), key=win_hours)}
    json.dump(out, open(PREDS / "ablations.json", "w"), indent=1, default=str)
    log(f"ablations done in {time.time()-t0:.0f}s")
    print(json.dumps(out, indent=1, default=str))


# ------------------------------------------------------------- inference API
def predict(device_ids: list[str] | None = None, feat_file: Path = FEAT_FILE,
            win: str = "full72", model_dir: Path = MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase + function predictions for any set of signals, from a pair-feature file.

    Returns (phase_preds, function_preds) in the protocol's prediction-file format.
    Used for TEST only when the orchestrator authorises the final scoring.
    """
    meta = json.load(open(model_dir / "phase_lgbm.json"))
    fmeta = json.load(open(model_dir / "function_lgbm.json"))
    bst = lgb.Booster(model_file=str(model_dir / "phase_lgbm.txt"))
    fbst = lgb.Booster(model_file=str(model_dir / "function_lgbm.txt"))
    df = pd.read_parquet(feat_file)
    df = df[df.win == win]
    if device_ids is not None:
        df = df[df.DeviceId.isin(device_ids)]
    df = df.sort_values(["DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    for c in meta["features"]:
        if c not in df.columns:
            df[c] = np.nan
    s = bst.predict(df[meta["features"]])
    df["prob"] = to_prob(df, np.asarray(s), meta["mode"])
    ph = df[["DeviceId", "Detector", "cand_phase", "prob"]].copy()
    ph["prob"] = ph.prob / ph.groupby(["DeviceId", "Detector"])["prob"].transform("sum")

    i = df.groupby(["DeviceId", "Detector"])["prob"].idxmax()
    top = df.loc[i].copy()
    top["top_prob"] = top["prob"]
    for c in fmeta["features"]:
        if c not in top.columns:
            top[c] = np.nan
    P = fbst.predict(top[fmeta["features"]])
    fn = top[["DeviceId", "Detector"]].copy()
    fn[["p_advance", "p_presence", "p_count"]] = np.asarray(P)
    return ph.reset_index(drop=True), fn.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main", choices=["main", "ablations"])
    a = ap.parse_args()
    (stage_main if a.stage == "main" else stage_ablations)(a)


if __name__ == "__main__":
    main()
