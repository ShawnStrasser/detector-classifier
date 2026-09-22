"""The neural half of `models/final_v2`: raw events -> one GRU phase probability per
(detector, candidate phase), ready to be averaged with the LightGBM pipeline's.

The network runs with onnxruntime on the CPU (`src/gru_onnx.py`) -- the one and only
runtime; the input intervals and the raster come from `src/gru_input.py`, straight out of
the de-duplicated `ev` table `src/predict.py` already builds.  Nothing here imports torch,
lightgbm or scipy.

`blend.json` next to the weights carries the three numbers frozen on out-of-fold data:

    {"weight_lightgbm": 0.5, "where": "before", "cutoff_minutes": 120}

`weight_lightgbm` is the weight on the tree pipeline, `1 - weight` on the GRU; `where`
is "after" (average the final per-detector probabilities) or "before" (average the pair
ranker's probabilities and then run the joint decoder on the mixture); above
`cutoff_minutes` of data the GRU is not run at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS_FILE = "gru.onnx"
CONFIG_FILE = "blend.json"
_MODEL_CACHE: dict = {}


def config(model_dir) -> dict | None:
    """The frozen blend settings, or None when this model folder has no GRU."""
    d = Path(model_dir)
    if not (d / WEIGHTS_FILE).exists() or not (d / CONFIG_FILE).exists():
        return None
    cfg = json.load(open(d / CONFIG_FILE))
    cfg["weights_path"] = d / WEIGHTS_FILE
    return cfg


def _model(path: Path, pair_batch: int):
    key = (str(path), pair_batch)
    m = _MODEL_CACHE.get(key)
    if m is None:
        from gru_onnx import GruPhaseModel
        m = _MODEL_CACHE[key] = GruPhaseModel(path, pair_batch=pair_batch)
    return m


def phase_probs(con, weights_path, t0_ms: int, t1_ms: int,
                pair_batch: int = 512, max_chunks: int = 48) -> pd.DataFrame:
    """-> DeviceId, Detector, cand_phase, p_gru (sums to 1 per detector)."""
    import gru_input

    streams = gru_input.build_streams(con, int(t0_ms), int(t1_ms))
    model = _model(Path(weights_path), pair_batch)
    span = int(t1_ms) - int(t0_ms)
    dev_col, det_col, ph_col, p_col = [], [], [], []
    for dev, z in streams.items():
        dets = [int(c) for c in z["det_ch"]]
        cand = np.asarray(z["cand"], dtype=int)
        if not dets or cand.size < 1:
            continue
        p, _ = model.score_signal(z, dets, 0, span, max_chunks=max_chunks)
        D, K = p.shape
        dev_col.append(np.repeat(np.asarray([dev], dtype=object), D * K))
        det_col.append(np.repeat(np.asarray(dets, dtype=np.int64), K))
        ph_col.append(np.tile(cand, D))
        p_col.append(p.reshape(-1))
    if not dev_col:
        return pd.DataFrame({"DeviceId": pd.Series(dtype=object),
                             "Detector": pd.Series(dtype="int64"),
                             "cand_phase": pd.Series(dtype="int64"),
                             "p_gru": pd.Series(dtype="float64")})
    return pd.DataFrame({"DeviceId": np.concatenate(dev_col),
                         "Detector": np.concatenate(det_col),
                         "cand_phase": np.concatenate(ph_col),
                         "p_gru": np.concatenate(p_col).astype(np.float64)})


def mix(df: pd.DataFrame, col: str, gru: pd.DataFrame, weight: float) -> np.ndarray:
    """`weight` * df[col] + (1 - weight) * the GRU probability, renormalised per detector.

    Rows the GRU has no opinion about (a detector or candidate it never saw) keep the
    tree probability untouched."""
    key = ["DeviceId", "Detector", "cand_phase"]
    g = gru.copy()
    g["Detector"] = g.Detector.astype(np.int64)
    g["cand_phase"] = g.cand_phase.astype(np.int64)
    left = df[key].copy()
    left["Detector"] = left.Detector.astype(np.int64)
    left["cand_phase"] = left.cand_phase.astype(np.int64)
    p_nn = left.merge(g, on=key, how="left").p_gru.to_numpy()
    base = np.asarray(df[col], dtype=float)
    # renormalise the neural side over the candidates this frame actually carries
    gid = (df.DeviceId.astype(str) + "|" + df.Detector.astype(str)).to_numpy()
    s = pd.DataFrame({"g": gid, "v": np.where(np.isnan(p_nn), 0.0, p_nn)})
    tot = s.groupby("g")["v"].transform("sum").to_numpy()
    ok = ~np.isnan(p_nn) & (tot > 0)
    p_nn = np.where(ok, np.where(tot > 0, p_nn / np.where(tot > 0, tot, 1.0), 0.0), np.nan)
    out = np.where(np.isnan(p_nn), base, weight * base + (1.0 - weight) * p_nn)
    d = pd.DataFrame({"g": gid, "v": np.clip(out, 1e-12, None)})
    return (d.v / d.groupby("g")["v"].transform("sum")).to_numpy()
