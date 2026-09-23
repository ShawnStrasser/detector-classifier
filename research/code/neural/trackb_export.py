"""Track B (stage 15): export a Track B checkpoint to ONNX, prove it, and time it.

Same contract as `export_gru.py`: the shipped sub-graph is the raster -> one score per
(detector, candidate phase) pair, the function head is dropped, opset 17, the pair axis
and the time axis dynamic, so one file serves every sample length and every signal size.
Unlike `export_gru.py` the batch-norms are left in the graph: the TCN has fourteen of
them inside the residual blocks, and onnxruntime's own Conv+BatchNormalization fusion
(`ORT_ENABLE_ALL`, which `model/gru_onnx.py` already switches on) folds every one of them
at session-build time, so folding them by hand would buy nothing.

Three things are measured and written to `%DC_WORK%/trackB/export/<tag>.json`:

  * agreement with the PyTorch module the file came from, on random rasters AND on real
    signal rasters rendered by the production `model/gru_input.py` -- max absolute
    probability difference and argmax agreement;
  * onnxruntime CPU cost per signal per 30-minute window, four threads, at the pair
    counts of a real signal (the bench in `13_gru_blend.md` §3);
  * the same for the shipped `model/weights/gru.onnx`, as the reference.

    python research/code/neural/trackb_export.py --ckpt tb_tcn_f0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, REPO  # noqa: E402
import gru_input as GI  # noqa: E402
from neural.models import PairNet  # noqa: E402

TRACKB = DC_WORK / "trackB"
MODELDIR = TRACKB / "models"
OUTDIR = TRACKB / "models"
EXPDIR = TRACKB / "export"
SIGDIR = DC_WORK / "neural" / "sig"
SHIPPED = REPO / "model" / "weights" / "gru.onnx"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class PhaseOnly(nn.Module):
    """The shipped sub-graph for any Track B backbone."""

    def __init__(self, m: PairNet):
        super().__init__()
        self.backbone = m.backbone
        self.proj = m.proj
        self.phase = m.phase

    def forward(self, x):                                   # x [pairs, 9, steps]
        return self.phase(self.proj(self.backbone(x))).squeeze(-1)


def load(tag: str) -> tuple[PairNet, dict]:
    p = Path(tag)
    if not p.exists():
        p = MODELDIR / f"{tag}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    m = PairNet(d["arch"], **d.get("kw", {}))
    m.load_state_dict(d["state"])
    return m.eval(), d


def export(model: PairNet, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    m = PhaseOnly(model).eval()
    x = torch.zeros(4, 9, 1800)
    torch.onnx.export(m, (x,), str(out), input_names=["x"], output_names=["score"],
                      dynamic_axes={"x": {0: "pairs", 2: "steps"},
                                    "score": {0: "pairs"}},
                      opset_version=17, dynamo=False)
    log(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")


def session(path: Path, threads: int = 4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def verify(model: PairNet, out: Path, n_signals: int = 4, minutes: int = 30) -> dict:
    """Probability agreement between the ONNX file and the torch module it came from."""
    sess = session(out)
    name = sess.get_inputs()[0].name
    net = PhaseOnly(model).eval()
    rng = np.random.default_rng(0)
    worst_rand = 0.0
    for n, T in ((7, 613), (3, 300), (19, 1800), (5, 7200)):
        xb = rng.random((n, 9, T)).astype(np.float32)
        with torch.no_grad():
            a = net(torch.from_numpy(xb)).numpy()
        b = np.asarray(sess.run(None, {name: xb})[0]).ravel()
        worst_rand = max(worst_rand, float(np.abs(a - b).max()))

    files = sorted(SIGDIR.glob("*.npz"))
    files = [files[i] for i in rng.permutation(len(files))[:n_signals]]
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
        lo = np.asarray(sess.run(None, {name: x})[0]).ravel().reshape(D, K)
        with torch.no_grad():
            lt = net(torch.from_numpy(x)).numpy().reshape(D, K)
        po = np.exp(lo - lo.max(1, keepdims=True)); po /= po.sum(1, keepdims=True)
        pt = np.exp(lt - lt.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)
        dmax = max(dmax, float(np.abs(po - pt).max()))
        same &= bool((po.argmax(1) == pt.argmax(1)).all())
        npairs += x.shape[0]
    return {"max_abs_score_diff_random": worst_rand,
            "max_abs_prob_diff_real": dmax,
            "argmax_identical": same, "n_real_pairs": npairs, "n_signals": len(files)}


def bench(path: Path, pairs: int = 120, T: int = 1800, reps: int = 5,
          threads: int = 4) -> dict:
    sess = session(path, threads)
    name = sess.get_inputs()[0].name
    rng = np.random.default_rng(1)
    x = rng.random((pairs, 9, T)).astype(np.float32)
    sess.run(None, {name: x[:8]})                      # warm up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(0, pairs, 512):
            sess.run(None, {name: np.ascontiguousarray(x[i:i + 512])})
        ts.append(time.perf_counter() - t0)
    return {"pairs": pairs, "steps": T, "threads": threads,
            "seconds_per_signal_window": round(float(np.median(ts)), 4),
            "all": [round(t, 4) for t in ts]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pairs", type=int, default=120)
    a = ap.parse_args()
    EXPDIR.mkdir(parents=True, exist_ok=True)
    tag = a.tag or Path(a.ckpt).stem
    out = Path(a.out) if a.out else OUTDIR / f"{tag}.onnx"
    model, d = load(a.ckpt)
    log(f"{a.ckpt}: arch={d['arch']} epoch={d.get('epoch')} "
        f"inner-val={d.get('es_score')} dropout={d.get('dropout', 0.0)}")
    export(model, out)
    res = {"ckpt": a.ckpt, "arch": d["arch"], "onnx": str(out),
           "params": sum(p.numel() for p in model.parameters()),
           "verify": verify(model, out)}
    res["bench_trackb"] = bench(out, a.pairs)
    if SHIPPED.exists():
        res["bench_shipped_gru"] = bench(SHIPPED, a.pairs)
    json.dump(res, open(EXPDIR / f"{tag}.json", "w"), indent=1, default=str)
    log(json.dumps(res["verify"]))
    log("bench " + json.dumps({k: v for k, v in res.items() if k.startswith("bench")},
                              default=str))
    log(f"wrote {EXPDIR / f'{tag}.json'}")


if __name__ == "__main__":
    main()
