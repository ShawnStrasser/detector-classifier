"""Write `review/official_vs_model_disagreements.csv`.

Combines
  DEV / Dec-2024   out-of-fold predictions of the official-label retrain (variant B)
  NEW / Sept-2026  the frozen beta's predictions on the never-seen signals

and keeps every channel where the model is >= 0.80 confident in a phase other than the
programmed one.  These are candidates for a timing / wiring check, not model errors by
assumption: the user's caveat is that the plan itself is occasionally wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DC_WORK, REPO  # noqa: E402
import disagreements as D  # noqa: E402

OFFICIAL = DC_WORK / "official"
FEAT_COLS = ["DeviceId", "Detector", "cand_phase", "f_on_green", "n_cycles", "det_n_on"]


def from_probs(path: Path) -> pd.DataFrame:
    p = pd.read_parquet(path)
    p = p.sort_values(["DeviceId", "Detector", "prob"], ascending=[True, True, False])
    g = p.groupby(["DeviceId", "Detector"], as_index=False)
    a = g.first().rename(columns={"cand_phase": "phase_pred", "prob": "phase_prob"})
    return a[["DeviceId", "Detector", "phase_pred", "phase_prob"]]


def main() -> None:
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    off["Detector"] = off.Detector.astype(int)
    parts = []

    # ---- DEV, Dec-2024, official-label retrain (out of fold) ---------------
    f = DC_WORK / "preds" / "phase_oof_official_B.parquet"
    if f.exists():
        pred = from_probs(f)
        feat = pd.read_parquet(DC_WORK / "features" / "pair_features_windows.parquet",
                               columns=FEAT_COLS + ["win"])
        feat = feat[feat.win == "full72"].drop(columns=["win"])
        feat["Detector"] = feat.Detector.astype(int)
        nact = feat.groupby(["DeviceId", "Detector"], as_index=False).det_n_on.first()
        pred = pred.merge(nact.rename(columns={"det_n_on": "n_actuations"}),
                          on=["DeviceId", "Detector"], how="left")
        pred["n_actuations"] = pred.n_actuations.fillna(0)
        parts.append(D.collect(pred, feat, off, "DEV Dec-2024 (OOF, official retrain)"))

    # ---- NEW signals, Sept-2026 staging, frozen beta -----------------------
    f = OFFICIAL / "preds_beta" / "beta_NEW_full66_notb.parquet"
    if f.exists():
        pred = pd.read_parquet(f)[["DeviceId", "Detector", "phase_pred", "phase_prob",
                                   "n_actuations"]]
        pred["Detector"] = pred.Detector.astype(int)
        sf = OFFICIAL / "stg" / "features" / "pair_features_stg.parquet"
        feat = pd.read_parquet(sf, columns=FEAT_COLS + ["win"])
        feat = feat[feat.win == "full66"].drop(columns=["win"])
        feat["Detector"] = feat.Detector.astype(int)
        parts.append(D.collect(pred, feat, off, "NEW Sept-2026 (frozen beta)"))

    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out = out.sort_values(["source", "model_prob"], ascending=[True, False])
    p = REPO / "review" / "official_vs_model_disagreements.csv"
    out.to_csv(p, index=False)
    print(f"wrote {p}: {len(out)} rows")
    print(out.groupby("source").size().to_string())
    print(out.head(8).to_string())


if __name__ == "__main__":
    main()
