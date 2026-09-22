"""Produce `review/official_vs_model_disagreements.csv`.

The official timing is the truth for training and scoring, but the user's caveat is that a
plan can itself be wrong (a loop physically in the phase-4 approach programmed to call
phase 8).  So wherever the model is CONFIDENT and disagrees with the official label we
emit the channel with a plain-English evidence sentence, as a candidate for a
timing / wiring check in the field.

    python src/official/disagreements.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, REPO  # noqa: E402

OFFICIAL = DC_WORK / "official"
CONF = 0.80


def evidence(feat: pd.DataFrame, dev: str, det: int, off_p: int, mod_p: int,
             n_on: int) -> str:
    f = feat[(feat.DeviceId == dev) & (feat.Detector == det)]
    g = {int(r.cand_phase): r for r in f.itertuples()}
    def pct(p):
        r = g.get(p)
        return None if r is None or pd.isna(r.f_on_green) else 100 * float(r.f_on_green)
    a, b = pct(off_p), pct(mod_p)
    cyc = g.get(off_p).n_cycles if off_p in g else np.nan
    bits = [f"{n_on} actuations"]
    if b is not None:
        bits.append(f"{b:.1f}% of them while the model's phase is green")
    if a is not None:
        bits.append(f"{a:.1f}% while the programmed phase is green")
    if off_p in g and not pd.isna(cyc):
        bits.append(f"the programmed phase ran {int(cyc)} cycles in the window")
    return "; ".join(bits)


def collect(pred: pd.DataFrame, feat: pd.DataFrame, off: pd.DataFrame,
            source: str) -> pd.DataFrame:
    """pred: DeviceId, Detector, phase_pred, phase_prob, phase_2nd, n_actuations."""
    m = pred.merge(off, on=["DeviceId", "Detector"], how="inner")
    m = m[(m.target_type == "phase") & m.phase_pred.notna() &
          (m.phase_pred != m.target_num) & (m.phase_prob >= CONF)].copy()
    if not len(m):
        return pd.DataFrame()
    rows = []
    for r in m.itertuples():
        rows.append({
            "source": source, "DeviceId": r.DeviceId, "Detector": int(r.Detector),
            "official_phase": int(r.target_num), "model_phase": int(r.phase_pred),
            "model_prob": round(float(r.phase_prob), 3),
            "n_actuations": int(r.n_actuations),
            "additional_call_phases": r.additional_call_phases,
            "switch_phase": int(r.switch_phase), "delay": float(r.delay),
            "extend": float(r.extend), "description": r.description,
            "evidence": evidence(feat, r.DeviceId, int(r.Detector),
                                 int(r.target_num), int(r.phase_pred),
                                 int(r.n_actuations)),
            "check": "", "comment": ""})
    return pd.DataFrame(rows).sort_values("model_prob", ascending=False)
