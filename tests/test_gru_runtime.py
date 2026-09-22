"""The neural half of `models/final_v2` must run on onnxruntime and reproduce itself.

There is exactly ONE runtime for the network: `src/gru_onnx.py`.  No torch, no numpy
GRU, no optional import, no fallback.  These tests pin that down:

  * the exported graph and the blend settings are present and parse;
  * scoring the bundled 30-minute sample reproduces the stored reference probabilities;
  * the whole pipeline reproduces its stored predictions;
  * above the frozen cut-off the network is not run and the answer is the tree
    pipeline's, unchanged;
  * `predict()` works with torch, lightgbm, scipy and sklearn all blocked.

    python tests/test_gru_runtime.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import DEFAULT_MODEL_DIR, predict  # noqa: E402
import gru_blend as gb  # noqa: E402

SAMPLE = ROOT / "tests" / "data" / "sample_events.parquet"
GRU_REF = ROOT / "tests" / "data" / "gru_reference.parquet"
PRED_REF = ROOT / "tests" / "data" / "predict_v2_reference.parquet"
KEY = ["DeviceId", "Detector", "cand_phase"]


def test_weights_and_settings_present():
    cfg = gb.config(DEFAULT_MODEL_DIR)
    assert cfg is not None, f"no GRU in {DEFAULT_MODEL_DIR}"
    assert 0.0 <= cfg["weight_lightgbm"] <= 1.0
    assert cfg["where"] in ("before", "after")
    assert cfg["cutoff_minutes"] > 0
    assert cfg["runtime"] == "onnx"
    assert Path(cfg["weights_path"]).exists()


def _live_gru() -> pd.DataFrame:
    import predict as P
    ev = pd.read_parquet(SAMPLE)
    con = P._connect(4, "4GB")
    try:
        w0, w1, _ = P.load_events(con, ev)
        P.build_chunk_tables(con)
        cfg = gb.config(DEFAULT_MODEL_DIR)
        g = gb.phase_probs(con, cfg["weights_path"], int(round(w0 * 1000)),
                           int(round(w1 * 1000)))
    finally:
        con.close()
    return g.sort_values(KEY).reset_index(drop=True)


def test_onnx_runtime_reproduces_reference():
    got = _live_gru()
    ref = pd.read_parquet(GRU_REF).sort_values(KEY).reset_index(drop=True)
    assert len(got) == len(ref) and got[KEY].equals(ref[KEY])
    assert np.abs(got.p_gru.to_numpy() - ref.p_gru.to_numpy()).max() < 1e-4
    gi = got.groupby(["DeviceId", "Detector"]).p_gru.idxmax().to_numpy()
    ri = ref.groupby(["DeviceId", "Detector"]).p_gru.idxmax().to_numpy()
    assert (gi == ri).all(), "the winning candidate changed"
    # a probability distribution over the signal's candidate phases
    s = got.groupby(["DeviceId", "Detector"]).p_gru.sum()
    assert np.abs(s.to_numpy() - 1.0).max() < 1e-5


def test_pipeline_reproduces_reference():
    out = predict(SAMPLE, min_actuations=1)
    ref = pd.read_parquet(PRED_REF)
    m = out.merge(ref, on=["DeviceId", "Detector"], suffixes=("", "_ref"))
    assert len(m) == len(ref)
    assert m.phase_pred.fillna(-1).equals(m.phase_pred_ref.fillna(-1))
    assert np.abs(m.phase_prob.fillna(-1) - m.phase_prob_ref.fillna(-1)).max() < 1e-6
    assert m.function_pred.fillna("").equals(m.function_pred_ref.fillna(""))


def test_function_output_is_unchanged_from_final_v1():
    """The function model reads the trees-only phase, so it must be bit-identical."""
    v1 = ROOT / "models" / "final_v1"
    a = predict(SAMPLE, min_actuations=1)
    b = predict(SAMPLE, min_actuations=1, model_dir=v1)
    m = a.merge(b, on=["DeviceId", "Detector"], suffixes=("", "_v1"))
    assert m.function_pred.fillna("").equals(m.function_pred_v1.fillna(""))
    assert np.abs(m.function_prob.fillna(-1) - m.function_prob_v1.fillna(-1)).max() == 0.0
    for c in ("p_advance", "p_presence", "p_count", "p_yellow_red", "p_other"):
        assert np.abs(m[c].fillna(-1) - m[c + "_v1"].fillna(-1)).max() == 0.0


def test_above_the_cutoff_the_network_is_not_run():
    """With the cut-off set to zero the answer must be exactly final_v1's."""
    v1 = ROOT / "models" / "final_v1"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "final_v2_cut0"
        shutil.copytree(DEFAULT_MODEL_DIR, d)
        cfg = json.load(open(d / "blend.json"))
        cfg["cutoff_minutes"] = 0
        json.dump(cfg, open(d / "blend.json", "w"))
        a = predict(SAMPLE, min_actuations=1, model_dir=d)
    b = predict(SAMPLE, min_actuations=1, model_dir=v1)
    pd.testing.assert_frame_equal(a, b)


def test_runs_with_torch_and_lightgbm_blocked():
    """onnxruntime and numpy are allowed -- they are the shipped requirements."""
    code = (
        "import sys\n"
        "BLOCK = ('torch', 'lightgbm', 'scipy', 'sklearn')\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('blocked for test: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, r'{ROOT / 'src'}')\n"
        "from predict import predict\n"
        f"out = predict(r'{SAMPLE}', min_actuations=1)\n"
        "assert len(out) and out.phase_pred.notna().any()\n"
        "assert not any(m.split('.')[0] in BLOCK for m in sys.modules)\n"
        "print('ok', len(out))\n"
    )
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                fails += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("all passed" if not fails else f"{fails} test(s) failed")
    sys.exit(1 if fails else 0)
