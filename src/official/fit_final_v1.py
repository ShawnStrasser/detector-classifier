"""FINAL model fit -> `models/final_v1/`  (variant C + 3-seed ranker bagging).

Decisions frozen before this script was written (stage 07 + stage 10 + stage 11):
  * phase = LightGBM pair ranker -> joint per-signal decoder,
  * trained on OFFICIAL timing labels,
  * training pool = DEV (375 signals, Dec-2024) + NEWTRAIN (334 signals, Sept-2026),
  * the beta's variant-B 22-window mix (5 min .. full span),
  * **3-seed bagging of the ranker** (stage 07: halves run-to-run noise),
  * no Optuna params, no XGBoost/CatBoost, no contrast features, no health masking,
  * function head = `models/final_candidate/function/function_lgbm_v4.*` (copied in).

The 43 TEST signals and the 143 NEWTEST signals are never loaded here.

    python src/official/fit_final_v1.py --stage oof      # DEV out-of-sample metrics
    python src/official/fit_final_v1.py --stage models   # fit + save models/final_v1/
    python src/official/fit_final_v1.py --stage card     # model_card.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONCURRENT_PAIRS, DC_WORK, DEFAULT_PHASE, FOLDS_CSV, N_FOLDS, REPO  # noqa: E402
import decode_v2 as dec  # noqa: E402
import train_official as T  # noqa: E402
import run_train as R  # noqa: E402

OFFICIAL = DC_WORK / "official"
OUT = OFFICIAL / "final_v1"
MODEL_DIR = REPO / "models" / "final_v1"
FUNC_SRC = REPO / "models" / "final_candidate" / "function"
SEEDS = [0, 1, 2]
FULL_DEC, FULL_STG = "full72", "full66"
M30 = ["m30_a", "m30_b", "m30_c", "m30_d"]
H6 = ["h6_a", "h6_b"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ the pool
def build_pool():
    """DEV Dec-2024 rows + NEWTRAIN Sept-2026 rows, official labels, folds on both."""
    ev, sim = R.build_eval()
    extra, sim_x = R.build_extra("NEWTRAIN")
    nt = np.array(sorted(extra.DeviceId.unique()))
    rng = np.random.default_rng(0)
    fmap = dict(zip(nt, rng.integers(0, N_FOLDS, len(nt))))
    extra = extra.copy()
    extra["fold"] = extra.DeviceId.map(fmap)
    extra["scorable_hand"] = False
    extra["Phase_hand"] = np.nan
    extra = T.add_scorable(extra)                    # <- the column fit_final.py forgot
    comb = pd.concat([ev, extra], ignore_index=True)
    simc = pd.concat([sim, sim_x], ignore_index=True)
    comb = comb.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    comb["scorable"] = comb.scorable.fillna(False).astype(bool)
    fc = [c for c in T.feature_cols(ev) if c in extra.columns]
    log(f"pool {comb.shape}; {comb.DeviceId.nunique()} signals; {len(fc)} features; "
        f"{int(comb.Phase.notna().sum()):,} labelled rows")
    return comb, simc, fc


def _detkey(df: pd.DataFrame) -> np.ndarray:
    return (df["win"].astype(str) + "|" + df.DeviceId.astype(str) + "|" +
            df.Detector.astype(str)).to_numpy()


def bag_prob(df: pd.DataFrame, scores: list[np.ndarray]) -> np.ndarray:
    """Mean of the per-detector-normalised (softmax) probabilities of K ranker seeds.

    This is exactly what `predict.py` does at inference time, so the OOF numbers below
    describe the shipped pipeline and not a training-only variant."""
    return np.mean([T.to_prob(df, s) for s in scores], axis=0)


# ------------------------------------------------------------------ OOF (bagged)
def oof_bagged(comb, simc, fc, seeds=SEEDS):
    """6-fold grouped OOF: K ranker seeds averaged, then the joint decoder."""
    lab = comb.Phase.notna().to_numpy()
    S = np.zeros((len(seeds), len(comb)))
    for k in range(N_FOLDS):
        te = (comb.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & lab
        tr = comb[base & (comb.fold != inner).to_numpy()]
        va = comb[base & (comb.fold == inner).to_numpy()]
        for j, sd in enumerate(seeds):
            rp = dict(T.RANK_PARAMS, bagging_seed=sd, feature_fraction_seed=sd + 100,
                      data_random_seed=sd + 200, seed=sd)
            m = T._fit_rank(tr, va, fc, rp)
            S[j, te] = m.predict(comb.loc[te, fc])
        log(f"  fold {k}: {len(seeds)} ranker seeds done")
    p0 = bag_prob(comb, list(S))
    per_seed = [T.to_prob(comb, S[j]) for j in range(len(seeds))]

    # ---- joint decoder on the bagged first-stage probabilities ---------------
    pr = comb[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[["DeviceId", "Detector", "win", "cand_phase", "Phase", "fold", "y"]],
                on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    s2 = np.zeros(len(X))
    labX = X.Phase.notna().to_numpy()
    for k in range(N_FOLDS):
        te = (X.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & labX
        tr = X[base & (X.fold != inner).to_numpy()]
        va = X[base & (X.fold == inner).to_numpy()]
        P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
        log(f"  decoder fold {k} done")
    X["p2"] = T.norm_prob(X, s2)
    back = comb[["DeviceId", "Detector", "win", "cand_phase"]].merge(
        X[["DeviceId", "Detector", "win", "cand_phase", "p2"]],
        on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    return p0, back.p2.to_numpy(), per_seed


# ------------------------------------------------------------------- metrics
def rich_metrics(comb: pd.DataFrame, prob: np.ndarray, mask: np.ndarray,
                 full: str) -> dict:
    """Top-1 accuracy over scorable detectors at the full / 6 h / 30 min windows."""
    sub = comb[mask].reset_index(drop=True)
    t = T.top1(sub, np.asarray(prob)[mask])
    out = {}
    f = t[t.win == full]
    out["acc_full"] = float(f.ok.mean()) if len(f) else float("nan")
    out["n_full"] = int(len(f))
    for tag, wins in (("h6", H6), ("m30", M30)):
        vals = [float(t[t.win == w].ok.mean()) for w in wins if (t.win == w).any()]
        out[f"acc_{tag}"] = float(np.mean(vals)) if vals else float("nan")
        out[f"acc_{tag}_per_window"] = [round(v, 4) for v in vals]
    out["acc_allwin"] = float(t.ok.mean())
    out["n_allwin"] = int(len(t))
    out["primary"] = (out["acc_full"] + out["acc_m30"]) / 2
    std = f.Detector.map(DEFAULT_PHASE)
    ns = f[~((std == f.Phase) & (f.Detector <= 40))]
    out["acc_nonstd_full"] = float(ns.ok.mean()) if len(ns) else float("nan")
    out["n_nonstd_full"] = int(len(ns))
    err = f[f.ok == 0]
    out["n_err_full"] = int(len(err))
    out["n_err_concurrent"] = int(sum(
        frozenset((int(a), int(b))) in CONCURRENT_PAIRS
        for a, b in zip(err.Phase, err.cand_phase)))
    for th in (0.8, 0.9):
        m = f.p >= th
        out[f"cov{th}"] = float(m.mean()) if len(f) else float("nan")
        out[f"acc_at_cov{th}"] = float(f[m].ok.mean()) if m.any() else float("nan")
    m30 = t[t.win.isin(M30)]
    m = m30.p >= 0.9
    out["cov0.9_m30"] = float(m.mean()) if len(m30) else float("nan")
    out["acc_at_cov0.9_m30"] = float(m30[m].ok.mean()) if m.any() else float("nan")
    f0 = t[(t.fold == 0) & (t.win == full)]
    out["acc_fold0_full"] = float(f0.ok.mean()) if len(f0) else float("nan")
    out["n_fold0_full"] = int(len(f0))
    return out


def stage_oof(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comb, simc, fc = build_pool()
    t0 = time.time()
    p0, p2, per_seed = oof_bagged(comb, simc, fc)
    log(f"bagged OOF done in {time.time()-t0:.0f}s")
    isdec = (comb.src == "DEC").to_numpy()
    isstg = ~isdec
    res = {
        "n_features": len(fc), "seeds": SEEDS,
        "DEV_Dec2024": {"ranker_bag": rich_metrics(comb, p0, isdec, FULL_DEC),
                        "decoded": rich_metrics(comb, p2, isdec, FULL_DEC)},
        "NEWTRAIN_Sept2026": {"decoded": rich_metrics(comb, p2, isstg, FULL_STG)},
    }
    # single-seed ranker spread (how much the bag is smoothing)
    ss = [rich_metrics(comb, ps, isdec, FULL_DEC)["primary"] for ps in per_seed]
    res["ranker_single_seed_primary"] = [round(v, 5) for v in ss]
    res["ranker_single_seed_sd"] = float(np.std(ss, ddof=1))
    res["ranker_bag_primary"] = res["DEV_Dec2024"]["ranker_bag"]["primary"]
    json.dump(res, open(OUT / "oof_metrics.json", "w"), indent=1, default=str)
    log(json.dumps({k: v for k, v in res["DEV_Dec2024"]["decoded"].items()
                    if not k.endswith("per_window")}, indent=1))

    # contract-format OOF (full window of each period) + by-window detail
    full = comb.win.isin([FULL_DEC, FULL_STG]).to_numpy()
    o = comb.loc[full, ["DeviceId", "Detector", "cand_phase"]].copy()
    o["prob"] = p2[full]
    o["DeviceId"] = o.DeviceId.str.replace("@stg", "", regex=False)
    o["prob"] = o.prob / o.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
    o["Detector"] = o.Detector.astype(int)
    o["cand_phase"] = o.cand_phase.astype(int)
    o.to_parquet(DC_WORK / "preds" / "phase_oof_final_v1.parquet", index=False)
    ob = comb[["DeviceId", "Detector", "win", "cand_phase", "src"]].copy()
    ob["prob"] = p2
    ob["p0"] = p0
    ob.to_parquet(OUT / "oof_bywindow.parquet", index=False)
    log("wrote OOF contract files")


# -------------------------------------------------------------------- models
def stage_models(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comb, simc, fc = build_pool()
    lab = comb.Phase.notna().to_numpy()
    tr = comb[lab & (comb.fold != 0).to_numpy()]
    va = comb[lab & (comb.fold == 0).to_numpy()]

    # ---- 3 ranker seeds on the whole pool -----------------------------------
    iters, scores = [], []
    for j, sd in enumerate(SEEDS):
        rp = dict(T.RANK_PARAMS, bagging_seed=sd, feature_fraction_seed=sd + 100,
                  data_random_seed=sd + 200, seed=sd)
        m = T._fit_rank(tr, va, fc, rp)
        m.booster_.save_model(str(MODEL_DIR / f"phase_lgbm_v4_s{j}.txt"))
        iters.append(int(m.best_iteration_ or rp["n_estimators"]))
        scores.append(m.predict(comb[fc]))
        log(f"ranker seed {sd}: {iters[-1]} trees")
    json.dump({"features": fc, "params": {k: v for k, v in T.RANK_PARAMS.items()},
               "n_models": len(SEEDS), "seeds": SEEDS, "best_iterations": iters,
               "objective": "lambdarank",
               "averaging": "mean of per-detector-normalised (softmax) probabilities"},
              open(MODEL_DIR / "phase_lgbm_v4.json", "w"), indent=1)

    # ---- decoder on the bagged in-pool first-stage probabilities -------------
    # (the decoder's own training rows use the OOF probabilities written by --stage oof
    #  when they exist, so the second stage never sees a first stage that memorised them)
    oofp = OUT / "oof_bywindow.parquet"
    pr = comb[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    if oofp.exists():
        q = pd.read_parquet(oofp)[["DeviceId", "Detector", "win", "cand_phase", "p0"]]
        q["Detector"] = q.Detector.astype(comb.Detector.dtype)
        q["cand_phase"] = q.cand_phase.astype(comb.cand_phase.dtype)
        pr = pr.merge(q, on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
        log(f"decoder trains on OOF first-stage probabilities "
            f"({int(pr.p0.notna().sum()):,}/{len(pr):,} rows matched)")
        pr["p0"] = pr.p0.fillna(pd.Series(bag_prob(comb, scores), index=pr.index))
    else:
        pr["p0"] = bag_prob(comb, scores)
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[["DeviceId", "Detector", "win", "cand_phase", "Phase", "fold", "y"]],
                on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    X = X[X.Phase.notna()].sort_values(["win", "DeviceId", "Detector", "cand_phase"]) \
                          .reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0)
    n = P.pop("n_estimators")
    m2 = lgb.LGBMClassifier(n_estimators=n, **P)
    dtr, dva = X[X.fold != 0], X[X.fold == 0]
    m2.fit(dtr[cols], dtr.y, eval_set=[(dva[cols], dva.y)], eval_metric="binary_logloss",
           callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    m2.booster_.save_model(str(MODEL_DIR / "decode_lgbm_v4.txt"))
    json.dump({"features": cols, "champion": "decode_sim_adj_binary", "mode": "binary",
               "temperature": 1.0, "best_iteration": int(m2.best_iteration_ or n),
               "params": dict(P, n_estimators=n)},
              open(MODEL_DIR / "decode_lgbm_v4.json", "w"), indent=1)
    log(f"decoder: {m2.best_iteration_} trees")

    # ---- function head: copied unchanged from stage 11 -----------------------
    for f in ("function_lgbm_v4.txt", "function_lgbm_v4.json"):
        shutil.copy2(FUNC_SRC / f, MODEL_DIR / f)
    log("function head copied from models/final_candidate/function")

    # ---- in-sample check (the same DEV rows the OOF numbers use) ------------
    pr_is = comb[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    pr_is["p0"] = bag_prob(comb, scores)
    p0_is = pr_is.p0.to_numpy()
    Xa = dec.assemble(pr_is, pairs=comb, sim=simc)
    sd2 = np.asarray(m2.predict_proba(Xa[cols])[:, 1])
    Xa["p2"] = T.norm_prob(Xa, sd2)
    back = comb[["DeviceId", "Detector", "win", "cand_phase"]].merge(
        Xa[["DeviceId", "Detector", "win", "cand_phase", "p2"]],
        on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    isdec = (comb.src == "DEC").to_numpy()
    ins = {"ranker_bag": rich_metrics(comb, p0_is, isdec, FULL_DEC),
           "decoded": rich_metrics(comb, back.p2.to_numpy(), isdec, FULL_DEC)}
    json.dump(ins, open(OUT / "insample_metrics.json", "w"), indent=1, default=str)
    log("in-sample (DEV, models trained on these rows): " + json.dumps(
        {k: round(v, 4) for k, v in ins["decoded"].items()
         if k.startswith("acc_") and isinstance(v, float)}))

    # ---- numpy-backend verification ----------------------------------------
    sys.path.insert(0, str(REPO / "src"))
    from lgbm_numpy import NumpyBooster, NumpyBoosterBag
    smp = comb.sample(min(20000, len(comb)), random_state=0)
    files = [str(MODEL_DIR / f"phase_lgbm_v4_s{j}.txt") for j in range(len(SEEDS))]
    a1 = np.mean([lgb.Booster(model_file=f).predict(smp[fc]) for f in files], axis=0)
    b1 = NumpyBoosterBag(files).predict(smp[fc])
    smpX = X.sample(min(20000, len(X)), random_state=0)
    a2 = np.asarray(m2.predict_proba(smpX[cols])[:, 1])
    b2 = NumpyBooster(MODEL_DIR / "decode_lgbm_v4.txt").predict(smpX[cols])
    ver = {"ranker_bag_max_abs_diff": float(np.max(np.abs(a1 - b1))),
           "decoder_max_abs_diff": float(np.max(np.abs(a2 - b2)))}
    json.dump(ver, open(OUT / "numpy_verification.json", "w"), indent=1)
    log("numpy backend: " + json.dumps(ver))
    json.dump({"ranker_best_iterations": iters, "n_features_ranker": len(fc),
               "n_features_decoder": len(cols),
               "decoder_best_iteration": int(m2.best_iteration_ or n),
               "pool_signals": int(comb.DeviceId.nunique()),
               "pool_labelled_rows": int(lab.sum()),
               "windows": sorted(comb.win.unique().tolist())},
              open(OUT / "fit_info.json", "w"), indent=1, default=str)


# ----------------------------------------------------------------- model card
def stage_card(a) -> None:
    oof = json.load(open(OUT / "oof_metrics.json"))
    ins = json.load(open(OUT / "insample_metrics.json"))
    fit = json.load(open(OUT / "fit_info.json"))
    ver = json.load(open(OUT / "numpy_verification.json"))
    rmeta = json.load(open(MODEL_DIR / "phase_lgbm_v4.json"))
    dmeta = json.load(open(MODEL_DIR / "decode_lgbm_v4.json"))
    fmeta = json.load(open(MODEL_DIR / "function_lgbm_v4.json"))
    # stage 10's three single-seed variant-C runs give the run-to-run noise of the
    # decoded pipeline; this stage's own per-seed spread is the first stage's.
    try:
        v = json.load(open(OFFICIAL / "train" / "variants.json"))
        cseeds = [v[k]["decoded"]["primary"] for k in v if k.startswith("C_seed")]
    except Exception:
        cseeds = []
    dev = oof["DEV_Dec2024"]["decoded"]
    card = {
        "name": "final_v1",
        "date": time.strftime("%Y-%m-%d"),
        "what_it_does": ("per vehicle-detector channel: the signal phase it is wired to "
                         "and its function (Advance / Presence / Count / Yellow_Red / "
                         "Other), with a probability for each and a status saying whether "
                         "it could be answered at all"),
        "pipeline": ["LightGBM pair ranker, 3 seeds averaged",
                     "LightGBM joint per-signal decoder",
                     "(no wiring table: the ODOT tie-breaker was measured and dropped)",
                     "LightGBM 5-class function head on the predicted phase",
                     "detector-health status + minimum-evidence rule"],
        "phase_anonymous": ("no phase number and no detector channel number is ever a "
                            "model input; candidates are every phase with a Begin Green in "
                            "the sample. No channel-to-phase wiring table is consulted "
                            "anywhere in the shipped inference path."),
        "data": {
            "training_signals": fit["pool_signals"],
            "DEV_Dec_2024": "375 ODOT signals, 2024-12-02..04 (3 weekdays)",
            "NEWTRAIN_Sept_2026": "334 ODOT signals, 2026-09-18..21 (2.75 days, mostly weekend)",
            "labelled_detector_window_rows": fit["pool_labelled_rows"],
            "windows": fit["windows"],
            "window_note": ("one model for every sample length: trained on a mix of 22 "
                            "windows from 5 minutes to the full span")},
        "label_sources": {
            "phase": "official controller timing (data/detector_plans.parquet: call_phase, "
                     "else call_overlap)",
            "function": fmeta.get("label_source")},
        "held_out_never_used_in_training": {
            "TEST_signals": 43, "NEWTEST_signals": 143,
            "note": "scored exactly once, after this model was frozen"},
        "models": {
            "ranker": {"file": "phase_lgbm_v4_s{0,1,2}.txt", "n_models": rmeta["n_models"],
                       "n_features": fit["n_features_ranker"],
                       "best_iterations": fit["ranker_best_iterations"],
                       "averaging": rmeta["averaging"], "params": rmeta["params"]},
            "decoder": {"file": "decode_lgbm_v4.txt",
                        "n_features": fit["n_features_decoder"],
                        "best_iteration": fit["decoder_best_iteration"],
                        "params": dmeta.get("params")},
            "function": {"file": "function_lgbm_v4.txt", "classes": fmeta["classes"],
                         "n_features": len(fmeta["features"]),
                         "n_estimators": fmeta["n_estimators"]}},
        "feature_lists": {"ranker": rmeta["features"], "decoder": dmeta["features"],
                          "function": fmeta["features"]},
        "thresholds": {
            "min_actuations": 5,
            "min_prob": {"default": 0.0,
                         "alternative_service": "--min-actuations 1 --min-prob 0.9"},
            "low_evidence_flag_below_actuations": 20,
            "auto_accept_phase_prob": 0.9,
            "odot_tiebreak": {"shipped": False,
                              "note": "an optional post-processing tie-breaker using the ODOT "
                                      "standard channel-to-phase table was measured (+0.1 to "
                                      "+0.5 pt at 30 min, ~+0.05 pt at the full window) and "
                                      "REMOVED from the shipped model: it was the only "
                                      "component that read a phase number, and it can hurt at "
                                      "non-standard cabinets. src/tiebreak.py is research code "
                                      "only; src/predict.py does not import it."},
            "advance_presence_rule": fmeta.get("advance_presence_rule"),
            "yellow_red_operating_point": fmeta.get("yellow_red_operating_point")},
        "dev_out_of_sample_metrics": {
            "note": ("6-fold cross-validation grouped by signal on the 375 Dec-2024 "
                     "training signals, official labels, detectors with >=5 actuations "
                     "whose labelled phase turns green in the window"),
            "ranker_only": oof["DEV_Dec2024"]["ranker_bag"],
            "full_pipeline": dev,
            "Sept_2026_training_signals": oof["NEWTRAIN_Sept2026"]["decoded"],
            "seed_noise": {
                "ranker_single_seed_primary": oof["ranker_single_seed_primary"],
                "ranker_single_seed_sd": oof["ranker_single_seed_sd"],
                "decoded_single_seed_primary_stage10": cseeds,
                "decoded_single_seed_sd_stage10": (float(np.std(cseeds, ddof=1))
                                                   if len(cseeds) > 1 else None),
                "note": "3-seed bagging roughly halves this run-to-run spread (stage 07)"},
            "function_5class": fmeta.get("metrics")},
        "in_sample_metrics": {
            "note": "the same DEV rows, scored by the shipped models that were fitted on "
                    "them -- the overfitting check",
            "full_pipeline": ins["decoded"], "ranker_only": ins["ranker_bag"]},
        "numpy_backend_verification": ver,
        "function_numpy_backend_max_abs_diff": fmeta.get("numpy_backend_max_abs_diff"),
        "inference": {
            "entry_point": "src/predict.py",
            "requirements": "numpy, pandas, duckdb (lightgbm optional -- "
                            "src/lgbm_numpy.py evaluates the trees with numpy only)",
            "event_codes_used": [1, 7, 8, 9, 10, 11, 43, 44, 81, 82,
                                 83, 84, 85, 86, 87, 88, 131, 150, 173],
            "event_codes_required": [81, 82, 1],
            "event_codes_strongly_recommended": [43],
            "event_codes_enough_for_a_cheap_pull": [1, 7, 43, 44, 81, 82],
            "output_columns": ["DeviceId", "Detector", "phase_pred", "phase_prob",
                               "phase_2nd", "phase_2nd_prob", "function_pred",
                               "function_prob", "status", "review_flag", "n_actuations",
                               "minutes_of_data", "phase_guess",
                               "phase_guess_prob", "function_guess", "function_guess_prob",
                               "phase_margin", "p_advance", "p_presence", "p_count",
                               "p_yellow_red", "p_other", "n_candidate_phases",
                               "health_flag", "review_reason"]},
        "supersedes": "models/beta_v0 (2026-09-17), kept for reproducibility",
    }
    sizes = {p.name: p.stat().st_size for p in sorted(MODEL_DIR.glob("*.txt"))}
    card["model_files_bytes"] = sizes
    card["total_model_bytes"] = int(sum(sizes.values()))
    json.dump(card, open(MODEL_DIR / "model_card.json", "w"), indent=1, default=str)
    log(f"wrote {MODEL_DIR / 'model_card.json'} "
        f"({card['total_model_bytes']/1e6:.1f} MB of trees)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["oof", "models", "card"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
