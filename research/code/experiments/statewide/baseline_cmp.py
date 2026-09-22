"""Stage 05 task 5: the 2025 BiLSTM baseline on the same statewide detectors.

`baseline/inference_results.parquet` holds the old model's per-detector statewide
predictions (same 2025-02-25 pull).  The comparison is restricted to **subset (a)** -- the
signals that are in neither DEV nor TEST -- because the BiLSTM was trained on most of the
others; on those its numbers would be in-sample.

    python src/statewide/baseline_cmp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path

from common import FUNCTIONS  # noqa: E402
from common6 import SW_PREDS, statewide_labels  # noqa: E402

BASE = Path(r"S:\Data_Analysis\Python\detector-classifier\baseline\inference_results.parquet")


def main() -> None:
    lab = statewide_labels()
    lab = lab[lab.group == "unseen"].copy()
    lab["Detector"] = lab.Detector.astype(int)
    b = pd.read_parquet(BASE)[["DeviceId", "Detector", "PredictedPhase",
                               "PhaseConfidence", "PredictedFunction"]]
    b["Detector"] = b.Detector.astype(int)
    ev = lab.merge(b, on=["DeviceId", "Detector"], how="left")

    ph = pd.read_parquet(SW_PREDS / "phase6_statewide.parquet")
    ph = ph[ph.win == "h6_sw"]
    ph = ph.loc[ph.groupby(["DeviceId", "Detector"])["prob"].idxmax()]
    ph["Detector"] = ph.Detector.astype(int)
    ev = ev.merge(ph[["DeviceId", "Detector", "cand_phase", "prob"]],
                  on=["DeviceId", "Detector"], how="left")
    fn = pd.read_parquet(SW_PREDS / "function6_statewide.parquet")
    fn = fn[fn.win == "h6_sw"]
    fn["Detector"] = fn.Detector.astype(int)
    P = fn[["p_advance", "p_presence", "p_count"]].to_numpy(float)
    fn = fn[["DeviceId", "Detector"]].assign(
        lgbm_func=np.array(FUNCTIONS)[P.argmax(1)], lgbm_top=P.max(1))
    ev = ev.merge(fn, on=["DeviceId", "Detector"], how="left")

    ev["bilstm_ok"] = (ev.PredictedPhase == ev.Phase).fillna(False)
    ev["lgbm_ok"] = (ev.cand_phase == ev.Phase).fillna(False)
    both = ev.PredictedPhase.notna() & ev.cand_phase.notna()
    m3 = ev.func_std.isin(FUNCTIONS)
    res = {
        "n_signals": int(ev.DeviceId.nunique()), "n_labelled": int(len(ev)),
        "n_bilstm_pred": int(ev.PredictedPhase.notna().sum()),
        "n_lgbm_pred": int(ev.cand_phase.notna().sum()),
        "phase_all_labelled": {"bilstm": float(ev.bilstm_ok.mean()),
                               "lgbm6": float(ev.lgbm_ok.mean())},
        "phase_both_have_a_prediction": {
            "n": int(both.sum()),
            "bilstm": float(ev[both].bilstm_ok.mean()),
            "lgbm6": float(ev[both].lgbm_ok.mean()),
            "agree": float((ev[both].PredictedPhase == ev[both].cand_phase).mean()),
            "lgbm_right_bilstm_wrong": int((ev[both].lgbm_ok & ~ev[both].bilstm_ok).sum()),
            "bilstm_right_lgbm_wrong": int((ev[both].bilstm_ok & ~ev[both].lgbm_ok).sum())},
        "function_3class_labels_only": {
            "n": int(m3.sum()),
            "bilstm": float((ev[m3].PredictedFunction == ev[m3].func_std).mean()),
            "lgbm6": float((ev[m3].lgbm_func == ev[m3].func_std).mean())},
        "function_on_true_other": {
            "n": int((~m3).sum()),
            "bilstm_sent_to_other": 0.0,
            "note": "the BiLSTM has no Other class at all, so every true-Other detector "
                    "is necessarily mislabelled by it"},
    }
    json.dump(res, open(SW_PREDS / "baseline_cmp.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
