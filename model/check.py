"""Health check for the shipped model.  Run it after installing, or after any change.

    python model/check.py

Three checks, on the bundled 30-minute single-signal sample (`sample_events.parquet`,
raw controller events including codes the pipeline must filter out and duplicate rows it
must drop):

1. **Reproduces the stored answers.**  Both the neural pair scores and the final
   per-detector predictions must match the references in `reference/`, which were saved
   when the model was frozen.  This catches a broken install, a wrong model file or a
   library upgrade that changes the numbers.
2. **Phase-number invariance.**  Renumber the phases in the raw log and every prediction
   must follow the renumbering exactly, with identical probabilities.  The model is not
   allowed to know what a phase number means -- that is what lets it work on detectors
   wired against convention, and at other agencies.
3. **No hidden dependencies.**  `predict()` must run in a process where torch, lightgbm,
   scipy and sklearn cannot be imported at all -- only the packages in
   `requirements.txt`.

Exit code 0 = all passed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gru_blend as gb  # noqa: E402
from predict import (DEFAULT_MODEL_DIR, EXTRA_COLS, OUT_COLS,  # noqa: E402
                     PROB_COLS, predict)

SAMPLE = HERE / "sample_events.parquet"
GRU_REF = HERE / "reference" / "gru_reference.parquet"
PRED_REF = HERE / "reference" / "predict_reference.parquet"
KEY = ["DeviceId", "Detector", "cand_phase"]
CLASSES = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
PHASE_EVENTS = [1, 7, 8, 9, 10, 11, 43, 44]


def check_reproduces_reference() -> str:
    import predict as P

    cfg = gb.config(DEFAULT_MODEL_DIR)
    assert cfg is not None, f"no neural model in {DEFAULT_MODEL_DIR}"
    assert cfg["runtime"] == "onnx" and Path(cfg["weights_path"]).exists()

    ev = pd.read_parquet(SAMPLE)
    minutes = (ev.Timestamp.max() - ev.Timestamp.min()).total_seconds() / 60.0
    assert minutes <= cfg["cutoff_minutes"], \
        "sample is longer than the blend cut-off, so the network would not run"

    # -- the neural half, on its own
    con = P._connect(4, "4GB")
    try:
        w0, w1, _ = P.load_events(con, ev)
        P.build_chunk_tables(con)
        got = gb.phase_probs(con, cfg["weights_path"], int(round(w0 * 1000)),
                             int(round(w1 * 1000)))
    finally:
        con.close()
    got = got.sort_values(KEY).reset_index(drop=True)
    ref = pd.read_parquet(GRU_REF).sort_values(KEY).reset_index(drop=True)
    assert len(got) == len(ref) and got[KEY].equals(ref[KEY]), "GRU output shape changed"
    d = np.abs(got.p_gru.to_numpy() - ref.p_gru.to_numpy()).max()
    assert d < 1e-4, f"GRU probabilities drifted by {d:.2e}"
    assert (got.groupby(["DeviceId", "Detector"]).p_gru.idxmax().to_numpy() ==
            ref.groupby(["DeviceId", "Detector"]).p_gru.idxmax().to_numpy()).all(), \
        "the winning candidate changed"

    # -- the whole pipeline
    out = predict(SAMPLE, min_actuations=1)
    assert list(out.columns) == OUT_COLS + EXTRA_COLS, "output columns changed"
    pref = pd.read_parquet(PRED_REF)
    m = out.merge(pref, on=["DeviceId", "Detector"], suffixes=("", "_ref"))
    assert len(m) == len(pref), "a detector appeared or disappeared"
    assert m.phase_pred.fillna(-1).equals(m.phase_pred_ref.fillna(-1)), \
        "phase predictions changed"
    dp = np.abs(m.phase_prob.fillna(-1) - m.phase_prob_ref.fillna(-1)).max()
    assert dp < 1e-6, f"phase probabilities drifted by {dp:.2e}"
    assert m.function_pred.fillna("").equals(m.function_pred_ref.fillna("")), \
        "function predictions changed"

    # -- and the refusal rules still behave
    ans = out[out.function_pred.notna()]
    assert ans.function_pred.isin(CLASSES).all()
    assert (ans[PROB_COLS].sum(1) - 1).abs().max() < 1e-6
    strict = predict(SAMPLE, min_actuations=1, min_prob=0.9)
    kept = strict[strict.phase_pred.notna()]
    assert (kept.phase_prob >= 0.9).all() and len(kept) <= out.phase_pred.notna().sum()
    assert strict.phase_guess.notna().sum() == out.phase_pred.notna().sum(), \
        "a refused detector must still carry the model's opinion in phase_guess"
    return (f"{len(out)} detectors, {int(out.phase_pred.notna().sum())} answered; "
            f"max probability drift {max(d, dp):.1e}")


def check_phase_number_invariance() -> str:
    ev = pd.read_parquet(SAMPLE)
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
    assert (m.phase_pred.astype(int).map(perm) == m.phase_pred_s.astype(int)).all(), \
        "predictions did not follow the phase renumbering"
    assert np.allclose(m.phase_prob, m.phase_prob_s, atol=1e-6), \
        "probabilities changed under renumbering"
    return f"{len(m)} answered detectors followed the renumbering exactly"


def check_no_hidden_dependencies() -> str:
    blocked = ("torch", "lightgbm", "scipy", "sklearn")
    code = (
        "import sys\n"
        f"BLOCK = {blocked!r}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('blocked by check.py: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, r'{HERE}')\n"
        "from predict import predict\n"
        f"out = predict(r'{SAMPLE}', min_actuations=1)\n"
        "assert len(out) and out.phase_pred.notna().any()\n"
        "assert not any(m.split('.')[0] in BLOCK for m in sys.modules)\n"
        "print('ok', len(out))\n"
    )
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
    return "ran with " + ", ".join(blocked) + " blocked"


CHECKS = [("reproduces the stored answers", check_reproduces_reference),
          ("phase-number invariance", check_phase_number_invariance),
          ("no hidden dependencies", check_no_hidden_dependencies)]


def main() -> int:
    if not SAMPLE.exists():
        print(f"FAIL: missing bundled sample {SAMPLE}")
        return 1
    bad = 0
    for name, fn in CHECKS:
        try:
            print(f"PASS  {name}: {fn()}", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
    print("\nall checks passed" if not bad else f"\n{bad} check(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
