"""Task 6 - fit and save the OFFICIAL-label candidate models.

Training pool: DEV (Dec-2024, 375 signals) + NEWTRAIN (Sept-2026, 335 signals), official
phase labels, the beta's variant-B 22-window mix.  The 43 TEST signals and the 143 NEWTEST
signals are never touched.

Writes
    models/beta_v2_official_candidate/phase_lgbm_v2.{txt,json}
    models/beta_v2_official_candidate/decode_lgbm_v2.{txt,json}
    models/beta_v2_official_candidate/settings.json
    models/beta_v2_official_candidate/function_lgbm_v2.{txt,json}   (copied from beta_v0,
        unchanged -- function labels are being revised, this stage only changes phase)
    dc_work/preds/phase_oof_official_candidate.parquet              (contract format)
    dc_work/preds/phase_oof_official_candidate_bywindow.parquet

and verifies that `src/lgbm_numpy.py` reproduces both boosters bit-for-bit.

    python src/official/fit_final.py
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

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, FOLDS_CSV, N_FOLDS, REPO  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
from lgbm_numpy import NumpyBooster  # noqa: E402
import run_train as R  # noqa: E402

OFFICIAL = DC_WORK / "official"
MODEL_DIR = REPO / "models" / "beta_v2_official_candidate"
SEED = 0


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ev, sim = R.build_eval()                       # DEV, Dec-2024, official labels
    extra, sim_x = R.build_extra("NEWTRAIN")       # NEWTRAIN, Sept-2026, official labels
    # give the new signals folds too, so the combined pool has a proper grouped OOF
    nt = np.array(sorted(extra.DeviceId.unique()))
    rng = np.random.default_rng(0)
    fmap = dict(zip(nt, rng.integers(0, N_FOLDS, len(nt))))
    extra = extra.copy()
    extra["fold"] = extra.DeviceId.map(fmap)
    extra["scorable_hand"] = False
    comb = pd.concat([ev, extra], ignore_index=True)
    simc = pd.concat([sim, sim_x], ignore_index=True)
    comb = comb.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    fc = [c for c in T.feature_cols(ev) if c in extra.columns]
    log(f"combined pool {comb.shape}, {comb.DeviceId.nunique()} signals, {len(fc)} features")

    # ---- OOF over the whole pool (contract files + decoder training rows) ----
    p0, p2 = T.run_oof(comb, None, fc, a.seed, simc, None)
    t = T.top1(comb, p2)
    log("combined-pool OOF: " + json.dumps(
        {k: v for k, v in T.metrics(t, T.FULL_DEC).items() if k != "per_fold_primary"}))
    log("  Sept-2026 half: " + json.dumps(
        {k: v for k, v in T.metrics(t[t.DeviceId.str.endswith("@stg")],
                                    T.FULL_STG).items() if k != "per_fold_primary"}))
    full = comb.win.isin([T.FULL_DEC, T.FULL_STG]).to_numpy()
    o = comb.loc[full, ["DeviceId", "Detector", "cand_phase"]].copy()
    o["prob"] = p2[full]
    o["DeviceId"] = o.DeviceId.str.replace("@stg", "", regex=False)
    o["prob"] = o.prob / o.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
    o["Detector"] = o.Detector.astype(int)
    o["cand_phase"] = o.cand_phase.astype(int)
    o.to_parquet(DC_WORK / "preds" / "phase_oof_official_candidate.parquet", index=False)
    ob = comb[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    ob["prob"] = p2
    ob.to_parquet(DC_WORK / "preds" / "phase_oof_official_candidate_bywindow.parquet",
                  index=False)
    log("wrote contract-format OOF files")

    # ---- final ranker on everything (fold 0 as the early-stopping watch set) --
    rp = dict(T.RANK_PARAMS, bagging_seed=a.seed, feature_fraction_seed=a.seed + 100,
              data_random_seed=a.seed + 200, seed=a.seed)
    lab = comb.Phase.notna()
    m1 = T._fit_rank(comb[lab & (comb.fold != 0)], comb[lab & (comb.fold == 0)], fc, rp)
    m1.booster_.save_model(str(MODEL_DIR / "phase_lgbm_v2.txt"))
    json.dump({"features": fc, "params": {k: v for k, v in rp.items()}},
              open(MODEL_DIR / "phase_lgbm_v2.json", "w"), indent=1)
    log(f"saved ranker ({len(fc)} features, best_iter {m1.best_iteration_})")

    # ---- final decoder on the pool's OOF first-stage probabilities -----------
    pr = comb[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    pr["p0"] = p0
    X = dec.assemble(pr, pairs=comb, sim=simc)
    X = X.merge(comb[["DeviceId", "Detector", "win", "cand_phase", "Phase", "fold", "y"]],
                on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
    X = X[X.Phase.notna()].sort_values(["win", "DeviceId", "Detector", "cand_phase"]) \
                          .reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    P = dict(T.BIN_PARAMS, bagging_seed=a.seed, feature_fraction_seed=a.seed + 100,
             seed=a.seed)
    n = P.pop("n_estimators")
    m2 = lgb.LGBMClassifier(n_estimators=n, **P)
    tr, va = X[X.fold != 0], X[X.fold == 0]
    m2.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)], eval_metric="binary_logloss",
           callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    m2.booster_.save_model(str(MODEL_DIR / "decode_lgbm_v2.txt"))
    json.dump({"features": cols, "champion": "decode_sim_adj_binary", "mode": "binary",
               "temperature": 1.0}, open(MODEL_DIR / "decode_lgbm_v2.json", "w"), indent=1)
    log(f"saved decoder ({len(cols)} features, best_iter {m2.best_iteration_})")

    # ---- function head: unchanged copy of the beta's -------------------------
    for f in ("function_lgbm_v2.txt", "function_lgbm_v2.json"):
        shutil.copy(REPO / "models" / "beta_v0" / f, MODEL_DIR / f)

    # ---- numpy-backend verification -----------------------------------------
    ver = {}
    smp = comb.sample(min(50000, len(comb)), random_state=0)
    a1 = np.asarray(m1.booster_.predict(smp[fc]))
    b1 = NumpyBooster(MODEL_DIR / "phase_lgbm_v2.txt").predict(smp[fc])
    ver["ranker_max_abs_diff"] = float(np.max(np.abs(a1 - b1)))
    smpX = X.sample(min(50000, len(X)), random_state=0)
    a2 = np.asarray(m2.predict_proba(smpX[cols])[:, 1])
    b2 = NumpyBooster(MODEL_DIR / "decode_lgbm_v2.txt").predict(smpX[cols])
    ver["decoder_max_abs_diff"] = float(np.max(np.abs(a2 - b2)))
    log("lgbm_numpy verification: " + json.dumps(ver))

    settings = {
        "name": "beta_v2_official_candidate",
        "label_source": {"phase": "official timing database "
                                  "(data/detector_plans.parquet: call_phase, else call_overlap)",
                         "function": "unchanged from beta_v0 (hand labels); being revised"},
        "training_pool": {
            "DEV_Dec_2024_signals": int(ev.DeviceId.nunique()),
            "NEWTRAIN_Sept_2026_signals": int(extra.DeviceId.nunique()),
            "labelled_detector_windows": int(lab.sum()),
            "windows": sorted(comb.win.unique().tolist())},
        "held_out_never_used": {"TEST_signals": 43, "NEWTEST_signals": 143},
        "ranker_params": {k: v for k, v in rp.items()},
        "decoder_params": dict(P, n_estimators=n),
        "n_features_ranker": len(fc), "n_features_decoder": len(cols),
        "ranker_best_iteration": int(m1.best_iteration_ or rp["n_estimators"]),
        "decoder_best_iteration": int(m2.best_iteration_ or n),
        "numpy_backend_verification": ver,
        "seed": a.seed}
    json.dump(settings, open(MODEL_DIR / "settings.json", "w"), indent=1, default=str)
    log(f"wrote {MODEL_DIR/'settings.json'}")


if __name__ == "__main__":
    main()
