"""Stage 07 task 5: freeze whatever stage 07 adopted into `models/beta_v2_candidate/`.

Driven by `dc_work/tune/adopt.json`, written by hand once the experiments have reported:

    {"ranker": {"params": {...}, "features": [...], "seeds": [0,1,2], "n_estimators": 900},
     "decoder": {"params": {...}, "seeds": [0], "n_estimators": 700},
     "function": {"params": {...}, "class_weight": "none", "seeds": [0], "n_estimators": 400}}

Steps: refit each stage on ALL DEV signals with the beta's variant-B 22-window mix, save the
LightGBM text models, verify `src/lgbm_numpy.py` reproduces them exactly (including the
K-model average), and export the OOF predictions in the protocol's contract format.

    python src/tune/ship_v2.py --stage preds     # contract-format OOF files
    python src/tune/ship_v2.py --stage models    # models/beta_v2_candidate/
    python src/tune/ship_v2.py --stage verify    # numpy evaluator == lightgbm
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

from common import PREDS, REPO  # noqa: E402
from tune_common import (FULL, N_JOBS, TUNE, add_scorable, dump, feature_cols,  # noqa: E402
                         load_phase_pairs, log)
from tune_phase import group_sizes, inner_split  # noqa: E402

CAND = REPO / "models" / "beta_v2_candidate"
ADOPT = TUNE / "adopt.json"
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]


def adopt() -> dict:
    return json.load(open(ADOPT))


# ------------------------------------------------------------- contract preds
def stage_preds(a) -> None:
    """Write `phase_oof_v3_tuned*` and `function_oof_v3_tuned*` in contract format."""
    cfg = adopt()
    src = TUNE / f"p2_{cfg['phase_oof_tag']}.parquet"
    pr = pd.read_parquet(src)
    pr["prob"] = pr.prob / pr.groupby(["DeviceId", "Detector", "win"])["prob"].transform("sum")
    pr["Detector"] = pr.Detector.astype(int)
    pr["cand_phase"] = pr.cand_phase.astype(int)
    pr.to_parquet(PREDS / "phase_oof_v3_tuned_bywindow.parquet", index=False)
    pr[pr.win == FULL].drop(columns=["win"]).to_parquet(
        PREDS / "phase_oof_v3_tuned.parquet", index=False)
    log(f"phase OOF -> {PREDS / 'phase_oof_v3_tuned.parquet'} "
        f"({int((pr.win == FULL).sum())} rows at 72 h)")

    f = pd.read_parquet(TUNE / f"func_{cfg['function_oof_tag']}.parquet")
    out = f[["DeviceId", "Detector", "win"]].copy()
    out["p_advance"] = f.p_advance
    out["p_presence"] = f.p_presence
    out["p_count"] = f.p_count
    out["p_other"] = f.p_other + f.p_yellow_red
    out["p_yellow_red"] = f.p_yellow_red
    out["Detector"] = out.Detector.astype(int)
    out.to_parquet(PREDS / "function_oof_v3_tuned_bywindow.parquet", index=False)
    out[out.win == FULL].drop(columns=["win"]).to_parquet(
        PREDS / "function_oof_v3_tuned.parquet", index=False)
    log(f"function OOF -> {PREDS / 'function_oof_v3_tuned.parquet'}")


# -------------------------------------------------------------------- models
def _fit_lgb(params, tr, va, cols, y, ranker: bool, seeds):
    out = []
    for s in seeds:
        P = dict(params)
        P.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202)
        n = P.pop("n_estimators")
        if ranker:
            m = lgb.LGBMRanker(n_estimators=n, **P)
            m.fit(tr[cols], y[0], group=group_sizes(tr),
                  eval_set=[(va[cols], y[1])], eval_group=[group_sizes(va)],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
        else:
            m = lgb.LGBMClassifier(n_estimators=n, **P)
            m.fit(tr[cols], y[0], eval_set=[(va[cols], y[1])],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
        out.append(m)
    return out


def stage_models(a) -> None:
    import decode_v2 as dc
    from tune_phase import decoder_frame
    cfg = adopt()
    CAND.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- stage 1: pair ranker on ALL DEV
    df = add_scorable(load_phase_pairs())
    rcfg = cfg["ranker"]
    cols = rcfg.get("features") or feature_cols(df)
    lab = df[df.labelled & df.health_flag.ne("failed")]
    vm = inner_split(lab, seed=1000)
    tr, va = lab[~vm], lab[vm]
    p = dict(rcfg["params"])
    p["n_jobs"] = N_JOBS
    ms = _fit_lgb(p, tr, va, cols, (tr.y, va.y), p.get("objective") != "binary",
                  rcfg.get("seeds", [0]))
    for i, m in enumerate(ms):
        m.booster_.save_model(str(CAND / f"phase_lgbm_v3_s{i}.txt"))
    json.dump({"features": cols, "params": {k: v for k, v in p.items()},
               "n_models": len(ms), "best_iterations": [int(m.best_iteration_ or 0)
                                                        for m in ms],
               "objective": p.get("objective"),
               "averaging": "mean of per-detector-normalised probabilities"},
              open(CAND / "phase_lgbm_v3.json", "w"), indent=1)
    log(f"ranker saved ({len(ms)} model(s), {len(cols)} features)")

    # ---- stage 2: decoder on the OOF first-stage probabilities
    dcfg = cfg["decoder"]
    X = pd.read_parquet(TUNE / f"decframe_{cfg['decoder_p0']}.parquet")
    Xl = X[X.labelled & X.health_flag.ne("failed")]
    vm = inner_split(Xl, seed=1000)
    dcols = dc.BASE_COLS + dc.SIM_COLS + dc.ADJ_COLS
    dp = dict(dcfg["params"])
    dp["n_jobs"] = N_JOBS
    dms = _fit_lgb(dp, Xl[~vm], Xl[vm], dcols, (Xl[~vm].y, Xl[vm].y),
                   dp.get("objective") != "binary", dcfg.get("seeds", [0]))
    for i, m in enumerate(dms):
        m.booster_.save_model(str(CAND / f"decode_lgbm_v3_s{i}.txt"))
    json.dump({"features": dcols, "params": dp, "n_models": len(dms),
               "mode": "binary" if dp.get("objective") == "binary" else "rank",
               "best_iterations": [int(m.best_iteration_ or 0) for m in dms]},
              open(CAND / "decode_lgbm_v3.json", "w"), indent=1)
    log(f"decoder saved ({len(dms)} model(s))")

    # ---- stage 3: function head.  Stage 07 adopts NO function change, so the stage-06
    # head is copied verbatim rather than refitted (a refit with early stopping on an
    # inner split runs far past stage 06's 131 trees and produces a 50 MB file).
    if cfg["function"].get("copy_from"):
        import shutil
        src = REPO / "models" / cfg["function"]["copy_from"]
        for f in ("function_lgbm_v3.txt", "function_lgbm_v3.json"):
            shutil.copy2(src / f, CAND / f)
        log(f"function head copied unchanged from {src.name}")
        return
    import tune_function as tf
    fcfg = cfg["function"]
    fr = tf.load_frame()
    fcols = fcfg.get("features") or tf.feat_cols(fr)
    ok = (fr.health_flag != "failed").to_numpy()
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    fp = dict(fcfg["params"])
    fp["n_jobs"] = N_JOBS
    fp["num_class"] = 5
    sub = fr[ok]
    ysub = y[ok]
    sw = tf.class_weights(ysub, fcfg.get("class_weight", "none"))
    fms = []
    for s in fcfg.get("seeds", [0]):
        P = dict(fp)
        P.update(seed=s, bagging_seed=s + 101, feature_fraction_seed=s + 202)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(sub[fcols], ysub, sample_weight=sw)
        m.booster_.save_model(str(CAND / f"function_lgbm_v3t_s{len(fms)}.txt"))
        fms.append(m)
    json.dump({"classes": CLASSES5, "features": fcols, "params": fp,
               "class_weight": fcfg.get("class_weight", "none"),
               "sibling_features": tf.__dict__.get("SIB_FEATS"),
               "n_models": len(fms)},
              open(CAND / "function_lgbm_v3t.json", "w"), indent=1)
    log(f"function saved ({len(fms)} model(s), {len(fcols)} features) "
        f"in {time.time()-t0:.0f}s -> {CAND}")


# -------------------------------------------------------------------- verify
def stage_verify(a) -> None:
    """`src/lgbm_numpy.py` must reproduce every saved model (and their average) exactly."""
    from lgbm_numpy import NumpyBooster, predict_average
    df = add_scorable(load_phase_pairs())
    meta = json.load(open(CAND / "phase_lgbm_v3.json"))
    cols = meta["features"]
    X = df[df.win == FULL][cols].iloc[:4000]
    files = sorted(CAND.glob("phase_lgbm_v3_s*.txt"))
    res = {}
    lg = [lgb.Booster(model_file=str(f)) for f in files]
    npb = [NumpyBooster(str(f)) for f in files]
    d = max(float(np.abs(np.asarray(a_.predict(X)) - b.predict(X)).max())
            for a_, b in zip(lg, npb))
    res["phase_max_abs_diff_per_model"] = d
    res["phase_max_abs_diff_average"] = float(np.abs(
        np.mean([np.asarray(m.predict(X)) for m in lg], axis=0)
        - predict_average([str(f) for f in files], X)).max())
    fmeta = json.load(open(CAND / "function_lgbm_v3.json"))
    frcols = fmeta["features"]
    import tune_function as tf
    fr = tf.load_frame()
    Xf = fr[fr.win == FULL][frcols].iloc[:3000]
    ffiles = sorted(CAND.glob("function_lgbm_v3*.txt"))
    lgf = [lgb.Booster(model_file=str(f)) for f in ffiles]
    npf = [NumpyBooster(str(f)) for f in ffiles]
    res["function_max_abs_diff_per_model"] = max(
        float(np.abs(np.asarray(a_.predict(Xf)) - b.predict(Xf)).max())
        for a_, b in zip(lgf, npf))
    res["function_max_abs_diff_average"] = float(np.abs(
        np.mean([np.asarray(m.predict(Xf)) for m in lgf], axis=0)
        - predict_average([str(f) for f in ffiles], Xf)).max())
    dfiles = sorted(CAND.glob("decode_lgbm_v*.txt"))
    Xd = pd.read_parquet(TUNE / f"decframe_{adopt()['decoder_p0']}.parquet")
    dcols = json.load(open(CAND / "decode_lgbm_v2.json"))["features"]
    Xd = Xd[dcols].iloc[:4000]
    lgd = [lgb.Booster(model_file=str(f)) for f in dfiles]
    npd = [NumpyBooster(str(f)) for f in dfiles]
    res["decoder_max_abs_diff_per_model"] = max(
        float(np.abs(np.asarray(a_.predict(Xd)) - b.predict(Xd)).max())
        for a_, b in zip(lgd, npd))
    res["model_files"] = {p.name: p.stat().st_size for p in sorted(CAND.glob("*.txt"))}
    res["total_model_bytes"] = int(sum(res["model_files"].values()))
    dump(res, "numpy_verification.json")
    log(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
