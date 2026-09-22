"""Stage 13: export the trained GRU pair scorer to the shipped ONNX runtime.

The conv stem's batch-norm is folded into the convolution (eval-mode statistics), the
function head is dropped -- only the phase score is shipped -- and the graph is exported
at opset 17 with the pair axis and the time axis dynamic, so one file serves every sample
length and every signal size.

    python src/neural/export_gru.py --ckpt gru2_final --out models/final_v2/gru.onnx

There is exactly one runtime for this network (`src/gru_onnx.py`, onnxruntime on the
CPU).  A pure-numpy implementation was written and measured first; onnxruntime turned out
2-4x faster for the same numbers, so the numpy version was deleted rather than kept as a
second path (see results/13_gru_blend.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK, REPO  # noqa: E402
from neural.models import PairNet  # noqa: E402

MODELDIR = DC_WORK / "models" / "gru2"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ckpt(name: str) -> PairNet:
    p = Path(name)
    if not p.exists():
        p = MODELDIR / f"{name}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    m = PairNet(d["arch"], **d.get("kw", {}))
    m.load_state_dict(d["state"])
    return m.eval()


def fold_bn(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d) -> None:
    """Fold the eval-mode batch-norm statistics into the convolution, in place."""
    g = bn.weight.detach()
    b = bn.bias.detach()
    mu = bn.running_mean.detach()
    var = bn.running_var.detach()
    s = g / torch.sqrt(var + bn.eps)
    with torch.no_grad():
        cb = conv.bias.detach() if conv.bias is not None else torch.zeros_like(mu)
        conv.weight.copy_(conv.weight * s[:, None, None])
        conv.bias.copy_((cb - mu) * s + b)


class PhaseOnly(torch.nn.Module):
    """The shipped sub-graph: raster -> one score per (detector, candidate phase) pair."""

    def __init__(self, m: PairNet):
        super().__init__()
        bb = m.backbone
        fold_bn(bb.stem[0], bb.stem[1])
        self.stem = bb.stem[0]
        self.act = torch.nn.GELU()
        self.rnn = bb.rnn
        self.attn = bb.pool.score
        self.proj = m.proj
        self.phase = m.phase

    def forward(self, x):                                   # x [pairs, 9, steps]
        h = self.act(self.stem(x)).transpose(1, 2)
        h, _ = self.rnn(h)
        a = torch.softmax(self.attn(h), dim=1)
        pooled = torch.cat([(a * h).sum(1), h.mean(1), h.amax(1)], dim=-1)
        return self.phase(self.proj(pooled)).squeeze(-1)


def to_onnx(model: PairNet, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    m = PhaseOnly(model).eval()
    x = torch.zeros(4, 9, 1800)
    torch.onnx.export(m, (x,), str(out), input_names=["x"], output_names=["score"],
                      dynamic_axes={"x": {0: "pairs", 2: "steps"}, "score": {0: "pairs"}},
                      opset_version=17, dynamo=False)
    log(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
    # the exported graph must agree with the model it came from
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for n, T in ((7, 613), (3, 300), (19, 1800)):
        xb = rng.random((n, 9, T)).astype(np.float32)
        with torch.no_grad():
            a = m(torch.from_numpy(xb)).numpy()
        b = np.asarray(sess.run(None, {"x": xb})[0]).ravel()
        worst = max(worst, float(np.abs(a - b).max()))
    log(f"onnx vs torch, max abs score difference over 3 shapes: {worst:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gru2_final")
    ap.add_argument("--out", default=str(REPO / "models" / "final_v2" / "gru.onnx"))
    a = ap.parse_args()
    to_onnx(load_ckpt(a.ckpt), Path(a.out))


if __name__ == "__main__":
    main()
