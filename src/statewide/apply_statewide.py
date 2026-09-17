"""Stage 05 task 2b/3/5: apply the 6-code models to the statewide 2025-02-25 data and
score every labelled detector against `all_configs.csv`.

Signal groups (TEST signals are dropped in `common6.statewide_labels()` and never scored):
  * **unseen** - in neither DEV nor TEST: unseen signal *and* unseen date.  Scored with the
    all-DEV 6-code model.  This is the clean generalisation number.
  * **dev**    - a DEV signal on a new date.  Scored with the fold-appropriate model, i.e.
    the fold model that never saw that signal, so it is still an unseen-signal number.

    python src/statewide/apply_statewide.py
"""
from __future__ import annotations

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
from common import CONCURRENT_PAIRS, DEFAULT_PHASE, FUNCTIONS  # noqa: E402
from common6 import (SW_FEAT, SW_MODELS, SW_PREDS, feature_cols6,  # noqa: E402
                     statewide_labels)
from features_v2 import add_partner_diffs  # noqa: E402
from train6 import DECODE_COLS, PDIFF_ALL  # noqa: E402

FULLWIN = "h6_sw"
M30 = [f"m30_{i:02d}" for i in range(12)]
H1 = [f"h1_{i:02d}" for i in range(6)]
H3 = [f"h3_{i:02d}" for i in range(2)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _predict_by_group(X: pd.DataFrame, cols: list[str], prefix: str,
                      multiclass: bool = False) -> np.ndarray:
    """Row k is scored by `<prefix>_fold{f}.txt` when it belongs to DEV fold f, else by
    `<prefix>_all.txt`.  A DEV signal therefore never meets a model that saw it."""
    out = np.zeros((len(X), 3)) if multiclass else np.zeros(len(X))
    for key, idx in X.groupby(np.where(X.group.eq("dev"), X.fold, -1)).groups.items():
        name = f"{prefix}_all.txt" if key == -1 else f"{prefix}_fold{int(key)}.txt"
        b = lgb.Booster(model_file=str(SW_MODELS / name))
        pos = X.index.get_indexer(idx)
        p = b.predict(X.loc[idx, cols])
        out[pos] = p
    return out


def _norm(df: pd.DataFrame, s: np.ndarray, softmax: bool) -> np.ndarray:
    g = df["win"].astype(str) + "|" + df.DeviceId + "|" + df.Detector.astype(str)
    d = pd.DataFrame({"g": g.to_numpy(), "s": np.asarray(s)})
    if softmax:
        d["s"] = np.exp(d.s - d.groupby("g")["s"].transform("max"))
    else:
        d["s"] = np.clip(d.s, 1e-9, None)
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


# --------------------------------------------------------------------- score
def run_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = statewide_labels()
    grp = lab[["DeviceId", "group", "fold"]].drop_duplicates()
    df = pd.read_parquet(SW_FEAT / "pairs6_sw.parquet")
    df = df[df.DeviceId.isin(set(grp.DeviceId))].reset_index(drop=True)
    df = add_partner_diffs(df, PDIFF_ALL)
    h = pd.read_parquet(SW_FEAT / "health6_sw.parquet",
                        columns=["DeviceId", "Detector", "win", "health_flag"])
    df = df.merge(h, on=["DeviceId", "Detector", "win"], how="left")
    df["health_flag"] = df.health_flag.fillna("failed")
    df = df.merge(grp, on="DeviceId", how="left")
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    meta = json.load(open(SW_MODELS / "phase6.json"))
    for c in meta["features"]:
        if c not in df.columns:
            df[c] = np.nan
    log(f"statewide pairs {df.shape}, {df.DeviceId.nunique()} signals")
    df["p0"] = _norm(df, _predict_by_group(df, meta["features"], "phase6"), softmax=True)

    sim = pd.read_parquet(SW_FEAT / "sim6_sw.parquet")
    X = dec.assemble(df[["DeviceId", "Detector", "win", "cand_phase", "p0"]],
                     pairs=df, sim=sim)
    for c in DECODE_COLS:
        if c not in X.columns:
            X[c] = np.nan
    X = X.merge(grp, on="DeviceId", how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    X["prob"] = _norm(X, _predict_by_group(X, DECODE_COLS, "decode6"), softmax=False)
    probs = X[["DeviceId", "Detector", "win", "cand_phase", "prob"]]
    probs.to_parquet(SW_PREDS / "phase6_statewide.parquet", index=False)

    fmeta = json.load(open(SW_MODELS / "function6.json"))
    fr = fv2.build_frame(df.drop(columns=["p0"]), probs)
    for c in fmeta["features"]:
        if c not in fr.columns:
            fr[c] = np.nan
    fr = fr.reset_index(drop=True)
    P = fv2._apply_T(_predict_by_group(fr, fmeta["features"], "function6", multiclass=True),
                     fmeta["temperature"])
    fn = fr[["DeviceId", "Detector", "win", "pred_phase", "top_prob", "health_flag",
             "group", "fold"]].copy()
    fn[["p_advance", "p_presence", "p_count"]] = P
    fn.to_parquet(SW_PREDS / "function6_statewide.parquet", index=False)
    # keep the 6 h function design matrix: tasks 3 and 4 train statewide-only models on it
    keep = fr.win.eq(FULLWIN)
    cols = ["DeviceId", "Detector", "group", "fold", "health_flag"] + \
           [c for c in feature_cols6(fr) if c != "pred_phase"]
    fr.loc[keep, cols].to_parquet(SW_FEAT / "funcframe6_sw_h6.parquet", index=False)
    log("scored statewide")
    return probs, fn


# -------------------------------------------------------------- evaluation
def phase_eval(probs: pd.DataFrame, lab: pd.DataFrame, wins: list[str],
               tag: str) -> pd.DataFrame:
    top = probs[probs.win.isin(wins)]
    top = top.loc[top.groupby(["DeviceId", "Detector", "win"])["prob"].idxmax()].copy()
    top["Detector"] = top.Detector.astype(int)
    uni = lab.merge(pd.DataFrame({"win": wins}), how="cross")
    ev = uni.merge(top, on=["DeviceId", "Detector", "win"], how="left")
    ev["correct"] = (ev.cand_phase == ev.Phase).fillna(False)
    ev["classifiable"] = ev.cand_phase.notna()
    ev["std_phase"] = ev.Detector.map(DEFAULT_PHASE)
    ev["is_std"] = (ev.std_phase == ev.Phase) & (ev.Detector <= 40)
    ev["tag"] = tag
    return ev


def summarise_phase(ev: pd.DataFrame) -> dict:
    cl = ev.classifiable
    err = ev[cl & ~ev.correct]
    nconc = int(sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                    for a, b in zip(err.Phase, err.cand_phase)))
    d = {"n_signals": int(ev.DeviceId.nunique()), "n_all": int(len(ev)),
         "acc_all": float(ev.correct.mean()),
         "n_classifiable": int(cl.sum()), "coverage": float(cl.mean()),
         "acc_classifiable": float(ev[cl].correct.mean()) if cl.any() else np.nan,
         "n_nonstd": int((cl & ~ev.is_std).sum()),
         "acc_nonstd": float(ev[cl & ~ev.is_std].correct.mean())
         if (cl & ~ev.is_std).any() else np.nan,
         "n_err": int(len(err)), "n_err_concurrent": nconc,
         "err_pairs": err.assign(pair=[f"{int(a)}<->{int(b)}" if
                                       frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                                       else f"{int(a)}->{int(b)}"
                                       for a, b in zip(err.Phase, err.cand_phase)])
                         .pair.value_counts().head(8).to_dict()}
    d["acc_phase_ge9"] = (float(ev[cl & ev.Phase.ge(9)].correct.mean())
                          if (cl & ev.Phase.ge(9)).any() else None)
    d["n_phase_ge9"] = int((cl & ev.Phase.ge(9)).sum())
    cov = []
    for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = cl & (ev.prob.fillna(0) >= th)
        cov.append({"th": th, "coverage": float(m.mean()),
                    "acc": float(ev[m].correct.mean()) if m.any() else None})
    d["coverage_curve"] = cov
    return d


def main() -> None:
    t0 = time.time()
    probs, fn = run_models()
    lab = statewide_labels()
    lab3 = lab[["DeviceId", "Detector", "Phase", "func_std", "Function", "group", "fold"]]
    res = {}
    for grp in ("unseen", "dev"):
        L = lab3[lab3.group == grp]
        res[grp] = {}
        for tag, wins in (("h6", [FULLWIN]), ("m30", M30), ("h1", H1), ("h3", H3)):
            res[grp][tag] = summarise_phase(phase_eval(probs, L, wins, tag))
        # per-sub-window spread
        per = []
        for w in M30:
            e = phase_eval(probs, L, [w], w)
            per.append({"win": w, "acc_classifiable": float(e[e.classifiable].correct.mean()),
                        "coverage": float(e.classifiable.mean())})
        res[grp]["m30_per_window"] = per
    ev6 = pd.concat([phase_eval(probs, lab3[lab3.group == g], [FULLWIN], g)
                     for g in ("unseen", "dev")], ignore_index=True)
    ev6.to_parquet(SW_PREDS / "phase6_eval_h6.parquet", index=False)
    res["dummy_channel_labels"] = int((lab3.Detector > 64).sum())
    json.dump(res, open(SW_PREDS / "statewide_phase_results.json", "w"), indent=1,
              default=str)
    log(json.dumps({g: {k: v for k, v in res[g].items() if k != "m30_per_window"}
                    for g in ("unseen", "dev")}, indent=1, default=str))
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
