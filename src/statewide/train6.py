"""Stage 05 task 2a: train the **6-code** v2 pipeline (pair ranker -> joint decoder ->
function model) on the Dec-2024 DEV data with the same 6 grouped folds, and report DEV OOF
accuracy next to the full-code model.

Everything is identical to stage 04 except the inputs: only what is computable from event
codes 1, 7, 43, 44, 81, 82 (see `common6.BANNED6`).  Per-fold models are saved so that the
statewide scoring can use, for each DEV signal, the model that never saw it; an all-DEV
model is saved for signals that are in neither DEV nor TEST.

    python src/statewide/train6.py --stage all
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

import decode_v2 as dec  # noqa: E402
import function_v2 as fv2  # noqa: E402
from common import FOLDS_CSV, FUNCTIONS, LABELS_DEV, N_FOLDS  # noqa: E402
from common6 import SW_FEAT, SW_MODELS, SW_PREDS, feature_cols6  # noqa: E402
from features_v2 import PDIFF_FEATS, add_partner_diffs  # noqa: E402
from train_lgbm_v2 import RANK_PARAMS, _groups, fit_one, to_prob  # noqa: E402

PDIFF_ALL = PDIFF_FEATS + ["on_lift_green", "occ_lift_green", "f_on_green",
                           "excl_diff_min", "release_frac_long", "call43_fwd_lift"]
DECODE_COLS = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS     # stage-04 champion
FULL72 = ["full72"]
W30 = ["m30_a", "m30_b", "m30_c", "m30_d"]
W6H = ["h6_a", "h6_b"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- loading
def load_dev() -> pd.DataFrame:
    df = pd.read_parquet(SW_FEAT / "pairs6_dev.parquet")
    df = add_partner_diffs(df, PDIFF_ALL)
    h = pd.read_parquet(SW_FEAT / "health6_dev.parquet",
                        columns=["DeviceId", "Detector", "win", "health_flag"])
    df = df.merge(h, on=["DeviceId", "Detector", "win"], how="left")
    df["health_flag"] = df.health_flag.fillna("failed")
    df = df.merge(pd.read_csv(FOLDS_CSV), on="DeviceId", how="inner")
    df = df.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    return df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)


# ------------------------------------------------------------------ scoring
def top1_table(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    d = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    d["p"] = prob
    d = d.loc[d.groupby(["DeviceId", "Detector", "win"])["p"].idxmax()].copy()
    d["Detector"] = d.Detector.astype(int)
    return d


def score_windows(top: pd.DataFrame, labels: pd.DataFrame, wins: list[str]) -> dict:
    """Accuracy over a set of windows, reported both ways (protocol 'Data hygiene')."""
    t = top[top.win.isin(wins)].copy()
    t["Detector"] = t.Detector.astype(int)
    labels = labels.copy()
    labels["Detector"] = labels.Detector.astype(int)
    universe = labels.merge(pd.DataFrame({"win": wins}), how="cross")
    ev = universe.merge(t, on=["DeviceId", "Detector", "win"], how="left")
    ev["correct"] = (ev.cand_phase == ev.Phase).fillna(False)
    cl = ev.cand_phase.notna()
    return {"acc_all": float(ev.correct.mean()), "n_all": int(len(ev)),
            "acc_classifiable": float(ev[cl].correct.mean()) if cl.any() else np.nan,
            "n_classifiable": int(cl.sum()), "coverage": float(cl.mean())}


# -------------------------------------------------------------------- stages
def stage_phase(df: pd.DataFrame) -> pd.DataFrame:
    fc = feature_cols6(df)
    log(f"first stage: {len(df):,} pair rows, {len(fc)} features (6 codes)")
    ok = df.health_flag.ne("failed") & df.Phase.notna()
    s = np.zeros(len(df))
    SW_MODELS.mkdir(parents=True, exist_ok=True)
    for k in range(N_FOLDS):
        te = (df.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        tr = df[(~te) & ok & (df.fold != inner)]
        va = df[(~te) & ok & (df.fold == inner)]
        m = fit_one(tr, va, fc, RANK_PARAMS)
        s[te] = m.predict(df.loc[te, fc])
        m.booster_.save_model(str(SW_MODELS / f"phase6_fold{k}.txt"))
        log(f"  fold {k}: train {len(tr):,} best_iter {m.best_iteration_}")
    tr = df[ok & (df.fold != 0)]
    va = df[ok & (df.fold == 0)]
    m = fit_one(tr, va, fc, RANK_PARAMS)
    m.booster_.save_model(str(SW_MODELS / "phase6_all.txt"))
    json.dump({"features": fc}, open(SW_MODELS / "phase6.json", "w"), indent=1)
    out = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    out["p0"] = to_prob(df, s)
    out.to_parquet(SW_PREDS / "phase6_oof_dev.parquet", index=False)
    return out


def _decode_frame(pr: pd.DataFrame, pairs: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    X = dec.assemble(pr, pairs=pairs, sim=sim)
    for c in DECODE_COLS:
        if c not in X.columns:
            X[c] = np.nan
    return X


def stage_decode(df: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    sim = pd.read_parquet(SW_FEAT / "sim6_dev.parquet")
    X = _decode_frame(pr, df, sim)
    X = X.merge(pd.read_csv(FOLDS_CSV), on="DeviceId", how="inner")
    X = X.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    X["y"] = (X.cand_phase == X.Phase).astype(np.int8)
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    log(f"second stage frame {X.shape}")
    lab = X.Phase.notna()
    s = np.zeros(len(X))
    for k in range(N_FOLDS):
        te = (X.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        tr, va = X[(~te) & lab & (X.fold != inner)], X[(~te) & lab & (X.fold == inner)]
        P = dict(dec.BIN_PARAMS)
        m = lgb.LGBMClassifier(n_estimators=P.pop("n_estimators"), **P)
        m.fit(tr[DECODE_COLS], tr.y, eval_set=[(va[DECODE_COLS], va.y)],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s[te] = m.predict_proba(X.loc[te, DECODE_COLS])[:, 1]
        m.booster_.save_model(str(SW_MODELS / f"decode6_fold{k}.txt"))
    tr, va = X[lab & (X.fold != 0)], X[lab & (X.fold == 0)]
    P = dict(dec.BIN_PARAMS)
    m = lgb.LGBMClassifier(n_estimators=P.pop("n_estimators"), **P)
    m.fit(tr[DECODE_COLS], tr.y, eval_set=[(va[DECODE_COLS], va.y)],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    m.booster_.save_model(str(SW_MODELS / "decode6_all.txt"))
    json.dump({"features": DECODE_COLS, "mode": "binary"},
              open(SW_MODELS / "decode6.json", "w"), indent=1)
    d = pd.DataFrame({"g": X["win"].astype(str) + "|" + X.DeviceId + "|" +
                      X.Detector.astype(str), "s": np.clip(s, 1e-9, None)})
    X["prob"] = (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()
    out = X[["DeviceId", "Detector", "win", "cand_phase", "prob"]]
    out.to_parquet(SW_PREDS / "phase6_decoded_dev.parquet", index=False)
    return out


def stage_function(df: pd.DataFrame, probs: pd.DataFrame) -> dict:
    fr = fv2.build_frame(df, probs)
    fr = fr[fr.Function.notna()].reset_index(drop=True)
    fr3 = fr[fr.Function.isin(FUNCTIONS)].reset_index(drop=True)
    cols = feature_cols6(fr3)
    cols = [c for c in cols if c not in ("pred_phase", "top_prob_dup")]
    y = fr3.Function.map({f: i for i, f in enumerate(FUNCTIONS)}).to_numpy()
    ok = fr3.health_flag.ne("failed").to_numpy()
    P = np.zeros((len(fr3), 3))
    log(f"function frame {fr3.shape}, {len(cols)} features")
    for k in range(N_FOLDS):
        te = (fr3.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        trm = (~te) & ok & (fr3.fold != inner).to_numpy()
        vam = (~te) & ok & (fr3.fold == inner).to_numpy()
        prm = dict(fv2.FUNC_PARAMS)
        m = lgb.LGBMClassifier(n_estimators=prm.pop("n_estimators"), **prm)
        m.fit(fr3.loc[trm, cols], y[trm], eval_set=[(fr3.loc[vam, cols], y[vam])],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        P[te] = m.predict_proba(fr3.loc[te, cols])
        m.booster_.save_model(str(SW_MODELS / f"function6_fold{k}.txt"))
    T = fv2._temperature(P, y)
    Pc = fv2._apply_T(P, T)
    prm = dict(fv2.FUNC_PARAMS)
    trm = ok & (fr3.fold != 0).to_numpy()
    vam = ok & (fr3.fold == 0).to_numpy()
    m = lgb.LGBMClassifier(n_estimators=prm.pop("n_estimators"), **prm)
    m.fit(fr3.loc[trm, cols], y[trm], eval_set=[(fr3.loc[vam, cols], y[vam])],
          eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    m.booster_.save_model(str(SW_MODELS / "function6_all.txt"))
    json.dump({"features": cols, "classes": list(FUNCTIONS), "temperature": T},
              open(SW_MODELS / "function6.json", "w"), indent=1)
    out = fr3[["DeviceId", "Detector", "win", "Function"]].copy()
    out[["p_advance", "p_presence", "p_count"]] = Pc
    out.to_parquet(SW_PREDS / "function6_oof_dev.parquet", index=False)
    res = {}
    for tag, wins in (("full72", FULL72), ("h6", W6H), ("m30", W30)):
        m_ = fr3.win.isin(wins).to_numpy()
        res[tag] = {"acc": float((Pc[m_].argmax(1) == y[m_]).mean()), "n": int(m_.sum())}
        mc = m_ & ok
        res[tag]["acc_classifiable"] = float((Pc[mc].argmax(1) == y[mc]).mean())
    res["temperature"] = T
    res["confusion_full72"] = pd.crosstab(
        fr3.loc[fr3.win == "full72", "Function"],
        np.array(FUNCTIONS)[Pc[(fr3.win == "full72").to_numpy()].argmax(1)]).to_dict()
    return res


# ------------------------------------------------------------- full-code ref
def fullcode_reference(labels: pd.DataFrame) -> dict:
    """Same metric on the stage-04 full-code decoded OOF predictions."""
    p = pd.read_parquet(Path(r"C:\Users\hwyr67g\dc_work\preds") /
                        "phase_oof_v2_decoded_bywindow.parquet")
    top = p.loc[p.groupby(["DeviceId", "Detector", "win"])["prob"].idxmax()].copy()
    top = top.rename(columns={"prob": "p"})
    top["Detector"] = top.Detector.astype(int)
    return {tag: score_windows(top, labels, wins)
            for tag, wins in (("full72", FULL72), ("h6", W6H), ("m30", W30))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.parse_args()
    t0 = time.time()
    labels = pd.read_parquet(LABELS_DEV)
    df = load_dev()
    pr = stage_phase(df)
    dec_out = stage_decode(df, pr)
    res = {"n_pairs": int(len(df)), "n_features": len(feature_cols6(df))}
    t1 = top1_table(df, pr.p0.to_numpy())
    t2 = dec_out.loc[dec_out.groupby(["DeviceId", "Detector", "win"])["prob"].idxmax()].copy()
    t2["Detector"] = t2.Detector.astype(int)
    for tag, wins in (("full72", FULL72), ("h6", W6H), ("m30", W30)):
        res[f"ranker6_{tag}"] = score_windows(t1, labels, wins)
        res[f"decoded6_{tag}"] = score_windows(t2, labels, wins)
    res["fullcode_ref"] = fullcode_reference(labels)
    log(json.dumps({k: v for k, v in res.items() if k != "fullcode_ref"}, indent=1))
    log(json.dumps(res["fullcode_ref"], indent=1))
    res["function6"] = stage_function(df, dec_out)
    log(json.dumps(res["function6"], indent=1, default=str))
    json.dump(res, open(SW_PREDS / "dev6_results.json", "w"), indent=1, default=str)
    log(f"train6 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
