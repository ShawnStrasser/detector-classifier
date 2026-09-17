"""Smoke tests for the shippable inference package.

    pytest tests/test_predict_smoke.py          # or
    python  tests/test_predict_smoke.py

Runs on the bundled 30-minute, single-signal sample in `tests/data/sample_events.parquet`
(raw hi-res events straight out of the controller log, including event codes the pipeline
must filter out and duplicate rows it must drop).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import MIN_ACTUATIONS, OUT_COLS, predict  # noqa: E402

SAMPLE = Path(__file__).resolve().parent / "data" / "sample_events.parquet"


def _events() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE)


def test_sample_exists():
    assert SAMPLE.exists(), f"missing bundled sample {SAMPLE}"
    ev = _events()
    assert set(["DeviceId", "Timestamp", "EventId", "Parameter"]) <= set(ev.columns)
    assert len(ev) > 1000


def test_predict_from_path():
    out = predict(SAMPLE)
    assert len(out) > 0
    assert list(out.columns)[:len(OUT_COLS)] == OUT_COLS
    assert out.DeviceId.nunique() == 1
    assert out.Detector.is_unique
    answered = out[out.phase_pred.notna()]
    assert len(answered) > 0, "no detector got an answer in the sample"
    assert (answered.n_actuations >= MIN_ACTUATIONS).all()
    assert answered.phase_prob.between(0, 1).all()
    assert answered.function_pred.isin(["Advance", "Presence", "Count", "Other"]).all()
    assert not answered.status.str.startswith(("cannot classify", "not enough data")).any()
    dead = out[out.status.str.startswith(("cannot classify", "not enough data"))]
    assert dead.phase_pred.isna().all() and dead.function_pred.isna().all()
    assert dead.review_flag.all()
    assert (out.minutes_of_data > 25).all() and (out.minutes_of_data < 35).all()
    assert not out.tiebreak_applied.any()   # default OFF


def test_minimum_evidence_rule():
    """Thin channels get no answer, but the raw opinion is kept in *_guess."""
    out = predict(SAMPLE)
    thin = out[out.status.str.startswith("not enough data")]
    if len(thin):
        assert (thin.n_actuations < MIN_ACTUATIONS).all()
        assert (thin.n_actuations > 0).all()
        assert thin.phase_guess.notna().all()       # nothing is lost
        assert thin.phase_guess_prob.between(0, 1).all()
        assert thin.status.str.contains(f"need >= {MIN_ACTUATIONS}").all()
    # a strict minimum must answer fewer detectors than a permissive one
    lenient = predict(SAMPLE, min_actuations=1)
    strict = predict(SAMPLE, min_actuations=100)
    assert lenient.phase_pred.notna().sum() >= out.phase_pred.notna().sum()
    assert strict.phase_pred.notna().sum() <= out.phase_pred.notna().sum()
    assert strict.phase_guess.notna().sum() == lenient.phase_pred.notna().sum()


def test_low_evidence_warning():
    out = predict(SAMPLE)
    low = out[out.status.str.startswith("ok - low evidence")]
    assert low.phase_pred.notna().all()
    assert low.review_flag.all()


def test_predict_from_dataframe_lowercase():
    ev = _events().rename(columns={"DeviceId": "device_id", "Timestamp": "timestamp",
                                   "EventId": "event_id", "Parameter": "parameter"})
    a = predict(ev)
    b = predict(SAMPLE)
    pd.testing.assert_frame_equal(a, b)


def test_odot_tiebreak_runs():
    out = predict(SAMPLE, odot_tiebreak=True)
    assert len(out) > 0
    assert out.tiebreak_applied.dtype == bool


def test_one_minute_window_degrades_gracefully():
    ev = _events()
    t0 = ev.Timestamp.min()
    out = predict(ev, start=str(t0), end=str(t0 + pd.Timedelta(minutes=1)))
    assert len(out) > 0
    assert out.status.notna().all()
    assert out.minutes_of_data.max() <= 1.1


def test_no_phase_events():
    """A log with no Begin Green (event 1) must report, not crash."""
    ev = _events()
    ev = ev[~ev.EventId.isin([1])]
    out = predict(ev)
    assert len(out) > 0
    assert out.phase_pred.isna().all()
    assert out.status.str.contains("cannot classify").all()


def test_detector_events_only():
    """Only 81/82 present: no phases, no calls, no faults."""
    ev = _events()
    ev = ev[ev.EventId.isin([81, 82])]
    out = predict(ev)
    assert len(out) > 0
    assert out.phase_pred.isna().all()


def test_no_calls_43_44():
    """43/44 are optional -- results must still come out."""
    ev = _events()
    ev = ev[~ev.EventId.isin([43, 44])]
    out = predict(ev)
    assert out.phase_pred.notna().any()


def test_one_minute_window_answers_little_but_keeps_guesses():
    ev = _events()
    t0 = ev.Timestamp.min()
    out = predict(ev, start=str(t0), end=str(t0 + pd.Timedelta(minutes=1)))
    assert len(out) > 0
    assert out.phase_pred.notna().sum() <= out.phase_guess.notna().sum()


def test_empty_selection():
    out = predict(SAMPLE, start="2030-01-01", end="2030-01-02")
    assert len(out) == 0
    assert list(out.columns)[:len(OUT_COLS)] == OUT_COLS


def test_chunked_matches_unchunked():
    a = predict(SAMPLE)
    b = predict(SAMPLE, chunk_signals=1)
    pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                fails += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("all passed" if not fails else f"{fails} test(s) failed")
    sys.exit(1 if fails else 0)
