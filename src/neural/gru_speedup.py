"""RESEARCH ONLY -- run the shipped GRU's forward pass with torch (GPU) instead of numpy.

This exists for one reason: scoring 143 locked signals through `src/predict.py` at four
60-minute anchors takes hours on the CPU, and the study machine has an idle GPU.  It is
**not** part of the shipped model and `src/predict.py` never imports it: the shipped
runtime is onnxruntime and only onnxruntime (`src/gru_onnx.py`).

It rebuilds the network from the training checkpoint the shipped `gru.onnx` was exported
from, so the scores are the shipped model's, computed a different way.  `verify()` checks
that against the shipped onnxruntime session on real data, and `enable()` then
monkey-patches `gru_onnx.GruPhaseModel.logits`.

    python src/neural/gru_speedup.py            # verify against the numpy runtime
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gru_input as GI  # noqa: E402
import gru_onnx as G  # noqa: E402

CKPT = Path.home() / "dc_work" / "models" / "gru2" / "gru2_final.pt"
NPZ_DEFAULT = Path(__file__).resolve().parents[2] / "models" / "final_v2" / "gru.onnx"


def build(ckpt=CKPT):
    """The shipped sub-graph as a torch module (the same one `export_gru.py` exports)."""
    from neural.export_gru import PhaseOnly, load_ckpt
    return PhaseOnly(load_ckpt(str(ckpt))).eval()


_PATCHED = {}


def enable(device: str = "cuda", batch: int = 256) -> str:
    """Make `GruPhaseModel.logits` use torch on `device`.  Returns the device used."""
    if not torch.cuda.is_available():
        device = "cpu"
    orig = G.GruPhaseModel.logits

    def logits(self, x):                                    # noqa: ANN001
        net = _PATCHED.get(id(self))
        if net is None:
            net = _PATCHED[id(self)] = build().to(device).eval()
        out = np.empty(x.shape[0], dtype=np.float32)
        with torch.no_grad():
            for i in range(0, x.shape[0], batch):
                xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).to(device)
                out[i:i + batch] = net(xb).float().cpu().numpy()
        return out

    G.GruPhaseModel.logits = logits
    G.GruPhaseModel._onnx_logits = orig
    return device


def verify(onnx_path=NPZ_DEFAULT, n_signals: int = 3, minutes: int = 30) -> dict:
    """Max absolute probability difference and argmax agreement on real rasters."""
    sigdir = Path.home() / "dc_work" / "neural" / "sig"
    m = G.GruPhaseModel(onnx_path)
    net = build()
    files = sorted(sigdir.glob("*.npz"))
    files = [files[i] for i in np.random.default_rng(0).permutation(len(files))[:n_signals]]
    dmax, same, npairs = 0.0, True, 0
    for p in files:
        with np.load(p) as f:
            z = {k: f[k] for k in f.files}
        dets = [int(c) for c in z["det_ch"]]
        if not dets or not len(z["cand"]):
            continue
        det, ph, sig, _ = GI.render(z, 12 * 3600 * 1000, dets, minutes * 60)
        x = GI.assemble(det, ph, sig)
        D, K = det.shape[0], ph.shape[0]
        ln = m.logits(x).reshape(D, K)
        with torch.no_grad():
            lt = net(torch.from_numpy(x)).numpy().reshape(D, K)
        pn = np.exp(ln - ln.max(1, keepdims=True)); pn /= pn.sum(1, keepdims=True)
        pt = np.exp(lt - lt.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)
        dmax = max(dmax, float(np.abs(pn - pt).max()))
        same &= bool((pn.argmax(1) == pt.argmax(1)).all())
        npairs += x.shape[0]
    return {"max_abs_prob_diff": dmax, "argmax_identical": same, "n_pairs": npairs}


if __name__ == "__main__":
    print(verify())
