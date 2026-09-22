"""The pure-numpy tree evaluator must reproduce LightGBM (including the 3-seed ranker bag),
and predict() must run with lightgbm, scipy and sklearn absent.

    python tests/test_numpy_backend.py
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
MODELS = os.path.join(ROOT, "models", "final_v1")
SAMPLE = os.path.join(ROOT, "tests", "data", "sample_events.parquet")

RANKER_SEEDS = sorted(glob.glob(os.path.join(MODELS, "phase_lgbm_v4_s*.txt")))


def test_boosters_match_lightgbm():
    import lightgbm as lgb
    from lgbm_numpy import NumpyBooster
    rng = np.random.default_rng(0)
    paths = RANKER_SEEDS + [os.path.join(MODELS, n + ".txt")
                            for n in ("decode_lgbm_v4", "function_lgbm_v4")]
    assert len(RANKER_SEEDS) >= 1, "no ranker seed models found"
    for path in paths:
        ref, mine = lgb.Booster(model_file=path), NumpyBooster(path)
        X = rng.normal(0, 3, (3000, mine.n_features))
        X[rng.random(X.shape) < 0.15] = np.nan            # missing values
        X[rng.random(X.shape) < 0.10] = 0.0               # exact zeros
        X[:, ::7] = np.round(X[:, ::7])                   # ties with integer thresholds
        assert np.allclose(ref.predict(X), mine.predict(X), rtol=0, atol=1e-10), path
        assert np.allclose(ref.predict(X, raw_score=True),
                           mine.predict(X, raw_score=True), atol=1e-10), path


def test_bag_averages_models():
    """The shipped 3-seed ranker bag must equal the mean of its parts, both backends."""
    import lightgbm as lgb
    from lgbm_numpy import NumpyBooster, NumpyBoosterBag, predict_average
    rng = np.random.default_rng(1)
    n_feat = NumpyBooster(RANKER_SEEDS[0]).n_features
    X = rng.normal(0, 3, (1200, n_feat))
    X[rng.random(X.shape) < 0.15] = np.nan
    lg = np.mean([lgb.Booster(model_file=p).predict(X) for p in RANKER_SEEDS], axis=0)
    assert np.allclose(NumpyBoosterBag(RANKER_SEEDS).predict(X), lg, rtol=0, atol=1e-10)
    assert np.allclose(predict_average(RANKER_SEEDS, X), lg, rtol=0, atol=1e-10)
    # a trivial K-fold bag of one model is that model
    one = NumpyBooster(RANKER_SEEDS[0])
    assert np.allclose(NumpyBoosterBag([RANKER_SEEDS[0]] * 3).predict(X),
                       one.predict(X), rtol=0, atol=1e-12)


def test_pipeline_identical_with_numpy_backend():
    import predict as P
    ev = pd.read_parquet(SAMPLE)
    P.set_backend("lightgbm"); a = P.predict(ev, min_actuations=1)
    P.set_backend("numpy");    b = P.predict(ev, min_actuations=1)
    P.set_backend("auto")
    assert a.phase_pred.equals(b.phase_pred) and a.function_pred.equals(b.function_pred)
    assert np.allclose(a.phase_prob.fillna(-1), b.phase_prob.fillna(-1), atol=1e-9)
    assert np.allclose(a.function_prob.fillna(-1), b.function_prob.fillna(-1), atol=1e-9)
    for c in P.PROB_COLS:
        assert np.allclose(a[c].fillna(-1), b[c].fillna(-1), atol=1e-9), c


def test_runs_without_lightgbm_and_scipy():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('lightgbm', 'scipy', 'sklearn'):\n"
        "            raise ImportError('blocked for test: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, r'{os.path.join(ROOT, 'src')}')\n"
        "from predict import predict\n"
        f"out = predict(r'{SAMPLE}')\n"
        "assert len(out) and out.phase_pred.notna().any()\n"
        "assert out.function_pred.dropna().isin(['Advance','Presence','Count',"
        "'Yellow_Red','Other']).all()\n"
        "assert not any(m.split('.')[0] in ('lightgbm', 'scipy', 'sklearn') for m in sys.modules)\n"
        "print('ok', len(out))\n"
    )
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]



if __name__ == "__main__":
    for t in (test_boosters_match_lightgbm, test_bag_averages_models,
              test_pipeline_identical_with_numpy_backend,
              test_runs_without_lightgbm_and_scipy,):
        t(); print("PASS", t.__name__)
