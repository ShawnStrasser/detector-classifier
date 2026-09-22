"""The neural pair scorer of `models/final_v2`, run with onnxruntime on the CPU.

This is the **only** runtime for the network.  There is no PyTorch path, no numpy
fallback and no optional import: `onnxruntime` is a hard requirement of
`requirements-inference.txt`.

    from gru_onnx import GruPhaseModel
    m = GruPhaseModel("models/final_v2/gru.onnx")
    prob, n_act = m.score_signal(streams, dets, t_start_ms, t_end_ms)   # [n_det, n_cand]

`streams` is the interval bundle `src/gru_input.py` builds from the raw events; the
graph is the strided conv stem (batch-norm folded in at export), a 3-layer bidirectional
GRU, attention pooling and two dense layers, exported at opset 17 with the pair and time
axes dynamic, so one session serves every sample length and every signal size.
"""
from __future__ import annotations

import os

import numpy as np

from gru_input import CHUNK_MS, assemble, render, split_range

# a modest office machine; the recurrent products are small, so more threads stop helping
DEFAULT_THREADS = int(os.environ.get("DC_GRU_THREADS", "4"))


class GruPhaseModel:
    def __init__(self, onnx_path, pair_batch: int = 512, threads: int = DEFAULT_THREADS):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), so,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.pair_batch = int(pair_batch)
        self.path = str(onnx_path)

    # ---------------------------------------------------------------- scoring
    def logits(self, x: np.ndarray) -> np.ndarray:
        """[N, 9, T] raster -> [N] pair scores, in batches of `pair_batch`."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty(x.shape[0], dtype=np.float32)
        for i in range(0, x.shape[0], self.pair_batch):
            xb = np.ascontiguousarray(x[i:i + self.pair_batch])
            out[i:i + self.pair_batch] = np.asarray(
                self.session.run(None, {self.input_name: xb})[0]).ravel()
        return out

    def window_logprobs(self, z: dict, dets, w0: int, T: int):
        """One window -> (log-probabilities [D, K], actuations [D])."""
        det, ph, sig, nact = render(z, int(w0), list(dets), int(T))
        D, K = det.shape[0], ph.shape[0]
        if D == 0 or K == 0:
            return np.zeros((D, K), np.float32), nact
        lg = self.logits(assemble(det, ph, sig)).reshape(D, K)
        lg = lg - lg.max(axis=1, keepdims=True)
        return (lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))).astype(np.float32), nact

    def score_signal(self, z: dict, dets, t0_ms: int, t1_ms: int,
                     max_chunks: int = 48):
        """Score an arbitrary period.  Pieces of at most 30 minutes are pooled by the
        mean log-probability (the listwise-correct way to combine windows).

        -> (probabilities [D, K], actuations [D])."""
        pieces = split_range(int(t0_ms), int(t1_ms), CHUNK_MS, max_chunks=max_chunks)
        lp = nact = None
        for a, b in pieces:
            T = max(int((b - a) // 1000), 8)
            l, na = self.window_logprobs(z, dets, a, T)
            lp = l if lp is None else lp + l
            nact = na if nact is None else nact + na
        if lp is None:
            n = len(list(dets))
            return np.zeros((n, len(z["cand"])), np.float32), np.zeros(n, np.float32)
        lp = lp / len(pieces)
        p = np.exp(lp - lp.max(axis=1, keepdims=True))
        return (p / p.sum(axis=1, keepdims=True)).astype(np.float32), nact
