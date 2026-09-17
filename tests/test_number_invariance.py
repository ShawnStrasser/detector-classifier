"""The model must not know phase numbers: renumbering the phases in the raw events must renumber the
predictions and change nothing else. Run: python tests/test_number_invariance.py"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import predict

PHASE_EVENTS = [1, 7, 8, 9, 10, 11, 43, 44]


def test_phase_renumbering_is_followed():
    ev = pd.read_parquet(os.path.join(ROOT, "tests", "data", "sample_events.parquet"))
    base = predict(ev, min_actuations=1)

    is_ph = ev.EventId.isin(PHASE_EVENTS)
    phases = sorted(ev.loc[is_ph, "Parameter"].unique())
    perm = dict(zip(phases, np.random.default_rng(0).permutation(phases)))
    ev2 = ev.copy()
    ev2.loc[is_ph, "Parameter"] = ev2.loc[is_ph, "Parameter"].map(perm)
    scr = predict(ev2, min_actuations=1)

    m = base.merge(scr, on=["DeviceId", "Detector"], suffixes=("", "_s"))
    m = m[m.phase_pred.notna()]
    assert len(m) > 0
    expected = m.phase_pred.astype(int).map(perm)
    assert (expected == m.phase_pred_s.astype(int)).all(), "predictions did not follow the phase renumbering"
    assert np.allclose(m.phase_prob, m.phase_prob_s, atol=1e-6), "probabilities changed under renumbering"


if __name__ == "__main__":
    test_phase_renumbering_is_followed()
    print("PASS test_phase_renumbering_is_followed")
