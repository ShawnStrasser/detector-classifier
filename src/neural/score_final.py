"""Score the neural predictions against BOTH label sets with the stage-05b/10 rules.

Rules (docs/research_protocol.md + the LightGBM beta):
  * unscorable excluded: the true phase must be a candidate (turns green in the
    window) and the detector must have >= 1 actuation;
  * headline also reported under the beta's minimum-evidence rule (>= 5 actuations).

Caveat printed with the official-label numbers: the neural prediction files only
cover the detectors that carry a HAND label, so the official-label row set is the
intersection, not the 7,197 rows LightGBM variant B is scored on.

    python src/neural/score_final.py --phase preds.parquet [--bywindow bw.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, DC_WORK, FOLDS_CSV, LABELS_DEV, connect  # noqa: E402
from evaluate import load_candidates  # noqa: E402

OFFICIAL = DC_WORK / "official" / "labels_official.parquet"


def top1(pred: pd.DataFrame) -> pd.DataFrame:
    i = pred.groupby(["DeviceId", "Detector"], sort=False)["prob"].idxmax()
    return (pred.loc[i, ["DeviceId", "Detector", "cand_phase", "prob"]]
            .rename(columns={"cand_phase": "pred", "prob": "top_prob"}))


def labels(kind: str) -> pd.DataFrame:
    if kind == "hand":
        lab = pd.read_parquet(LABELS_DEV)[["DeviceId", "Detector", "Phase"]]
    else:
        o = pd.read_parquet(OFFICIAL)
        o = o[(o.call_phase > 0) & o.real_dec2024]
        lab = o[["DeviceId", "Detector", "call_phase"]].rename(
            columns={"call_phase": "Phase"})
    folds = pd.read_csv(FOLDS_CSV)
    lab = lab.merge(folds, on="DeviceId", how="inner")
    cand = load_candidates().rename(columns={"Phase": "Phase"})
    cand["is_cand"] = True
    lab = lab.merge(cand, on=["DeviceId", "Phase"], how="left")
    lab["is_cand"] = lab.is_cand.fillna(False).astype(bool)
    return lab.drop_duplicates(["DeviceId", "Detector"])


def n_on() -> pd.DataFrame:
    con = connect(memory_limit="6GB", threads=6)
    df = con.sql(f"""SELECT DeviceId, Detector::INT AS Detector, n_on
                     FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}')""").df()
    con.close()
    return df


def score(pred: pd.DataFrame, lab: pd.DataFrame, act: pd.DataFrame, tag: str,
          nact_col: pd.DataFrame | None = None) -> dict:
    t = top1(pred)
    e = lab.merge(t, on=["DeviceId", "Detector"], how="inner")
    e = e.merge(act if nact_col is None else nact_col, on=["DeviceId", "Detector"],
                how="left")
    e["n_act"] = e.iloc[:, -1].fillna(0)
    e["hit"] = e.pred == e.Phase
    sc = e[e.is_cand & (e.n_act >= 1)]
    s5 = e[e.is_cand & (e.n_act >= 5)]
    out = dict(tag=tag, n_rows=len(e), n_scorable=len(sc), acc_scorable=sc.hit.mean(),
               n_ge5=len(s5), acc_ge5=s5.hit.mean(),
               n_unscorable_notcand=int((~e.is_cand).sum()),
               n_unscorable_noact=int((e.is_cand & (e.n_act < 1)).sum()))
    pf = sc.groupby("fold").hit.mean()
    out["fold0"] = float(pf.get(0, np.nan))
    out["fold_mean"] = float(pf.mean()); out["fold_sd"] = float(pf.std(ddof=0))
    return out


def show(r: dict) -> None:
    print(f"{r['tag']:34s} scorable {r['acc_scorable']:.4f} (n={r['n_scorable']})  "
          f">=5act {r['acc_ge5']:.4f} (n={r['n_ge5']})  fold0 {r['fold0']:.4f}  "
          f"folds {r['fold_mean']:.4f}+/-{r['fold_sd']:.4f}  "
          f"unscorable: {r['n_unscorable_notcand']} not-candidate + "
          f"{r['n_unscorable_noact']} no-actuation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--bywindow", default="")
    ap.add_argument("--label", default="both", choices=["hand", "official", "both"])
    args = ap.parse_args()

    act = n_on()
    pred = pd.read_parquet(args.phase)
    kinds = ["hand", "official"] if args.label == "both" else [args.label]
    for k in kinds:
        lab = labels(k)
        show(score(pred, lab, act, f"72 h / {k} labels"))
        if args.bywindow:
            bw = pd.read_parquet(args.bywindow)
            for win, g in bw.groupby("win"):
                na = (g.groupby(["DeviceId", "Detector"])["n_act"].first()
                      .reset_index())
                show(score(g, lab, act, f"{win} / {k} labels", nact_col=na))


if __name__ == "__main__":
    main()
