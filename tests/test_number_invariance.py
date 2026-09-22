"""The model must not know phase numbers: renumbering the phases in the raw events must renumber the
predictions and change nothing else. This must hold for the BLENDED answer too -- the GRU sees the
candidate phases as an unordered set, so the blend cannot smuggle a phase number in.
Run: python tests/test_number_invariance.py"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import DEFAULT_MODEL_DIR, predict
import gru_blend as gb

PHASE_EVENTS = [1, 7, 8, 9, 10, 11, 43, 44]


def test_the_blend_is_actually_on_for_this_sample():
    """Otherwise the invariance test below would silently only cover the trees."""
    cfg = gb.config(DEFAULT_MODEL_DIR)
    assert cfg is not None, "no neural model in the default model folder"
    ev = pd.read_parquet(os.path.join(ROOT, "tests", "data", "sample_events.parquet"))
    minutes = (ev.Timestamp.max() - ev.Timestamp.min()).total_seconds() / 60.0
    assert minutes <= cfg["cutoff_minutes"], "sample is longer than the blend cut-off"


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
    test_the_blend_is_actually_on_for_this_sample()
    print("PASS test_the_blend_is_actually_on_for_this_sample")
    test_phase_renumbering_is_followed()
    print("PASS test_phase_renumbering_is_followed")
