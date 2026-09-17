"""Stage 04 step 2/3: LightGBM pair ranker v2 (+ v2 features) and the joint per-signal decoder.

    python src/train_lgbm_v2.py --stage ranker      # v2 features -> phase_oof_v2.parquet
    python src/train_lgbm_v2.py --stage ablation    # feature-family ablation table
    python src/train_lgbm_v2.py --stage decode      # second-stage joint decoder
    python src/train_lgbm_v2.py --stage function    # function model v2
    python src/train_lgbm_v2.py --stage final       # fit + save the DEV-wide v2 models

Nothing from stage 01 is overwritten; every artefact carries a `_v2` suffix.
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
from common import (CACHE, FEATURES, FOLDS_CSV, FUNCTIONS, LABELS_DEV,  # noqa: E402
                    MODELS, N_FOLDS, PREDS)
from features import win_hours  # noqa: E402
from features_v2 import PDIFF_FEATS, add_partner_diffs  # noqa: E402

BASE_FEAT = FEATURES / "pair_features_windows.parquet"
V2_FEAT = FEATURES / "pair_features_v2_extra.parquet"
SIM_FILE = FEATURES / "det_similarity.parquet"
GROUP_KEYS = ["DeviceId", "Detector", "win"]
SHORT_WINS = ["m30_a", "m30_b", "m30_c", "m30_d", "h1_a", "h1_b", "h1_c"]
KEY_EXCLUDE = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase", "Function",
               "fold", "y", "cyc", "partner_phase", "health_flag", "std_phase"}

RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                   learning_rate=0.05, num_leaves=63, min_child_samples=40,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=1.0, n_estimators=1200, n_jobs=12, verbose=-1,
                   label_gain=[0, 1])


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in KEY_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


# -------------------------------------------------------------------- loading
def load_health() -> pd.DataFrame:
    f = Path(str(CACHE).replace("cache", "atspm")) / "detector_health.parquet"
    h = pd.read_parquet(f, columns=["DeviceId", "Detector", "health_flag"])
    h["Detector"] = h.Detector.astype(np.int16)
    return h


def load_pairs_v2(with_v2: bool = True, dev_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(BASE_FEAT)
    if with_v2:
        v2 = pd.read_parquet(V2_FEAT)
        df = df.merge(v2, on=["DeviceId", "Detector", "cand_phase", "win"], how="left")
        df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                                  "f_on_green", "excl_diff_min",
                                                  "release_frac_long", "call43_fwd_lift"])
    folds = pd.read_csv(FOLDS_CSV)
    if dev_only:
        df = df.merge(folds, on="DeviceId", how="inner")
    else:
        df = df.merge(folds, on="DeviceId", how="left")
    df = df.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    df = df.merge(load_health(), on=["DeviceId", "Detector"], how="left")
    df["health_flag"] = df.health_flag.fillna("unknown")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    return df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)


# ------------------------------------------------------------------ training
def _groups(df: pd.DataFrame) -> np.ndarray:
    key = df["win"].astype(str) + "|" + df["DeviceId"] + "|" + df["Detector"].astype(str)
    _, idx, cnt = np.unique(key.to_numpy(), return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


def fit_one(tr, va, feat_cols, params):
    P = dict(params)
    n_est = P.pop("n_estimators", 1200)
    m = lgb.LGBMRanker(n_estimators=n_est, **P)
    m.fit(tr[feat_cols], tr.y, group=_groups(tr),
          eval_set=[(va[feat_cols], va.y)], eval_group=[_groups(va)],
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    return m


def to_prob(df: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    d = pd.DataFrame({"g": df["win"].astype(str) + "|" + df.DeviceId + "|" +
                      df.Detector.astype(str), "s": s})
    d["s"] = np.exp(d.s - d.groupby("g")["s"].transform("max"))
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def oof_predict(df, feat_cols, params=RANK_PARAMS, train_mask=None):
    """6-fold grouped OOF.  Trains on labelled+healthy rows, scores EVERY row of the fold
    (unlabelled channels included, so the joint decoder can use them as neighbours)."""
    s = np.zeros(len(df))
    models = []
    ok = df.health_flag.ne("failed")
    if train_mask is not None:
        ok = ok & train_mask
    for k in range(N_FOLDS):
        te = df.fold == k
        inner = (k + 1) % N_FOLDS
        tr = df[(~te) & ok & (df.fold != inner)]
        va = df[(~te) & ok & (df.fold == inner)]
        m = fit_one(tr, va, feat_cols, params)
        s[te.to_numpy()] = m.predict(df.loc[te, feat_cols])
        models.append(m)
    return s, models


def top1(df, prob, wins=None, mask=None):
    d = df[["DeviceId", "Detector", "win", "cand_phase", "Phase"]].copy()
    d["p"] = prob
    if wins is not None:
        d = d[d.win.isin(wins)]
    if mask is not None:
        d = d[mask.reindex(d.index).fillna(False)]
    d = d[d.Phase.notna()]
    t = d.loc[d.groupby(GROUP_KEYS)["p"].idxmax()]
    return float((t.cand_phase == t.Phase).mean()), len(t)


def summarise(df, prob, tag) -> dict:
    """Standard metric block for a set of pair probabilities."""
    from common import CONCURRENT_PAIRS, DEFAULT_PHASE
    d = df[["DeviceId", "Detector", "win", "cand_phase", "Phase", "det_n_on"]].copy()
    d["p"] = prob
    t = d.loc[d.groupby(GROUP_KEYS)["p"].idxmax()].copy()
    t["ok"] = t.cand_phase == t.Phase
    t["std_phase"] = t.Detector.map(DEFAULT_PHASE)
    t["is_std"] = (t.std_phase == t.Phase) & (t.Detector <= 40)
    f = t[t.win == "full72"]
    err = f[~f.ok]
    nconc = sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                for a, b in zip(err.Phase, err.cand_phase))
    res = {
        "tag": tag,
        "acc_full72": float(f.ok.mean()),
        "acc_allwin": float(t.ok.mean()),
        "acc_short": float(t[t.win.isin(SHORT_WINS)].ok.mean()),
        "acc_m30": float(t[t.win.str.startswith("m30")].ok.mean()),
        "acc_nonstd_full72": float(f[~f.is_std].ok.mean()),
        "n_nonstd_full72": int((~f.is_std).sum()),
        "n_err_full72": int(len(err)),
        "n_err_concurrent": int(nconc),
    }
    return res


# ------------------------------------------------------------------- stages
def stage_ranker(args) -> None:
    t0 = time.time()
    dtr = load_pairs_v2(with_v2=True)
    fc = feature_cols(dtr)
    log(f"pairs={len(dtr):,} features={len(fc)} (stage01 + v2)")
    s, models = oof_predict(dtr, fc, train_mask=dtr.Phase.notna())
    prob = to_prob(dtr, s)
    lab = dtr.Phase.notna().to_numpy()
    r = summarise(dtr[lab].reset_index(drop=True), prob[lab], "ranker_v2")
    log(json.dumps(r, indent=1))
    PREDS.mkdir(parents=True, exist_ok=True)
    out = dtr.loc[dtr.win == "full72", ["DeviceId", "Detector", "cand_phase"]].copy()
    out["prob"] = prob[(dtr.win == "full72").to_numpy()]
    out["prob"] = out.prob / out.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
    out["cand_phase"] = out.cand_phase.astype(int)
    out["Detector"] = out.Detector.astype(int)
    out.to_parquet(PREDS / "phase_oof_v2.parquet", index=False)
    bw = dtr[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    bw["prob"] = prob
    bw.to_parquet(PREDS / "phase_oof_v2_bywindow.parquet", index=False)
    imp = pd.DataFrame({"feature": fc,
                        "gain": np.mean([m.booster_.feature_importance("gain")
                                         for m in models], axis=0)})
    imp.sort_values("gain", ascending=False).to_csv(
        PREDS / "feature_importance_phase_v2.csv", index=False)
    json.dump(r, open(PREDS / "ranker_v2_results.json", "w"), indent=1)
    log(f"ranker v2 done in {time.time()-t0:.0f}s")


V2_FAMILIES = {
    "partner_excl": ["pex_"],
    "fine_timing": ["dtg_h", "first_on_", "cyc_hit_frac", "tog_h0", "dur_g2", "burst_g2"],
    "release": ["release_lag", "queue_end_frac", "n_long"],
    "calls_v2": ["call43_b", "call43_red"],
    "partner_geom": ["cogreen_", "excl_secs_", "partner_lead_"],
    "pdiff": ["__pdiff"],
}


def stage_ablation(args) -> None:
    t0 = time.time()
    df = load_pairs_v2(with_v2=True)
    dtr = df[df.Phase.notna()].reset_index(drop=True)
    del df
    allf = feature_cols(dtr)
    v2cols = [c for c in allf if any(k in c for k in
                                     sum(V2_FAMILIES.values(), []))]
    base = [c for c in allf if c not in set(v2cols)]
    out = {}
    out["base_stage01"] = summarise(dtr, to_prob(dtr, oof_predict(dtr, base)[0]), "base_stage01")
    log(json.dumps(out["base_stage01"]))
    out["base_plus_all_v2"] = summarise(dtr, to_prob(dtr, oof_predict(dtr, allf)[0]),
                                        "base_plus_all_v2")
    log(json.dumps(out["base_plus_all_v2"]))
    for fam, pats in V2_FAMILIES.items():
        cols = base + [c for c in v2cols if any(p in c for p in pats)]
        out[f"base_plus_{fam}"] = summarise(dtr, to_prob(dtr, oof_predict(dtr, cols)[0]),
                                            f"base_plus_{fam}")
        log(json.dumps(out[f"base_plus_{fam}"]))
    for fam, pats in V2_FAMILIES.items():
        cols = [c for c in allf if not any(p in c for p in pats)]
        out[f"all_minus_{fam}"] = summarise(dtr, to_prob(dtr, oof_predict(dtr, cols)[0]),
                                            f"all_minus_{fam}")
        log(json.dumps(out[f"all_minus_{fam}"]))
    json.dump(out, open(PREDS / "ablations_v2.json", "w"), indent=1)
    log(f"ablation done in {time.time()-t0:.0f}s")


def stage_final(args) -> None:
    """Fit the DEV-wide v2 models that `src/predict.py` loads."""
    import decode_v2 as dc
    t0 = time.time()
    dtr = load_pairs_v2(with_v2=True)
    fc = feature_cols(dtr)
    lab = dtr[dtr.Phase.notna() & dtr.health_flag.ne("failed")]
    itr, iva = lab[lab.fold != 0], lab[lab.fold == 0]
    m = fit_one(itr, iva, fc, RANK_PARAMS)
    MODELS.mkdir(parents=True, exist_ok=True)
    m.booster_.save_model(str(MODELS / "phase_lgbm_v2.txt"))
    json.dump({"features": fc, "params": {k: v for k, v in RANK_PARAMS.items()}},
              open(MODELS / "phase_lgbm_v2.json", "w"), indent=1)
    log(f"saved phase_lgbm_v2 ({len(fc)} features, iter {m.best_iteration_})")

    # second stage on the OOF first-stage probabilities
    meta = json.load(open(MODELS / "decode_v2.json"))
    pr = pd.read_parquet(PREDS / "phase_oof_v2_bywindow.parquet").rename(columns={"prob": "p0"})
    X = dc.assemble(pr)
    X = X.merge(pd.read_csv(FOLDS_CSV), on="DeviceId", how="inner")
    X = X.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    X["y"] = (X.cand_phase == X.Phase).astype(np.int8)
    X = X[X.Phase.notna()].sort_values(["win", "DeviceId", "Detector", "cand_phase"]) \
                          .reset_index(drop=True)
    cols = meta["cols"]
    mode = meta.get("mode", "rank")
    tr, va = X[X.fold != 0], X[X.fold == 0]
    cb = [lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)]
    if mode == "binary":
        P = dict(dc.BIN_PARAMS)
        d2 = lgb.LGBMClassifier(n_estimators=P.pop("n_estimators"), **P)
        d2.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)],
               eval_metric="binary_logloss", callbacks=cb)
    else:
        P = dict(dc.RANK_PARAMS)
        d2 = lgb.LGBMRanker(n_estimators=P.pop("n_estimators"), **P)
        d2.fit(tr[cols], tr.y, group=dc._groups(tr), eval_set=[(va[cols], va.y)],
               eval_group=[dc._groups(va)], callbacks=cb)
    d2.booster_.save_model(str(MODELS / "decode_lgbm_v2.txt"))
    json.dump({"features": cols, "champion": meta["champion"], "mode": mode,
               "temperature": meta.get("temperature", 1.0)},
              open(MODELS / "decode_lgbm_v2.json", "w"), indent=1)
    log(f"saved decode_lgbm_v2 ({len(cols)} features) in {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ranker",
                    choices=["ranker", "ablation", "decode", "function", "final"])
    a = ap.parse_args()
    if a.stage == "ranker":
        stage_ranker(a)
    elif a.stage == "ablation":
        stage_ablation(a)
    elif a.stage == "final":
        stage_final(a)
    else:
        import decode_v2
        decode_v2.run_stage(a.stage)


if __name__ == "__main__":
    main()
