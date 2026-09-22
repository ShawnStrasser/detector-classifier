"""Stage 13: how fast is the shipped ONNX runtime, and is it exact?

Runs the exported graph on real rasters and reports wall-clock and peak resident memory
for 1 signal x 30 min, 1 signal x 2 h and 20 signals x 30 min, plus the maximum absolute
probability difference against the training-time PyTorch model and whether the argmax
ever differs.  `--torch` adds the PyTorch-CPU arm for comparison.

    python src/neural/bench_gru.py --onnx models/final_v2/gru.onnx --ckpt gru2_final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK, REPO  # noqa: E402
import gru_input as GI  # noqa: E402

SIGDIR = DC_WORK / "neural" / "sig"
OUT = DC_WORK / "official" / "blend_v2"
W0 = 12 * 3600 * 1000          # Tuesday midday of the Dec-2024 period
THREADS = 4                    # a modest office machine


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def rasters(n: int, minutes: int):
    files = sorted(SIGDIR.glob("*.npz"))
    files = [files[i] for i in np.random.default_rng(0).permutation(len(files))[:n]]
    out, pairs = [], []
    for p in files:
        with np.load(p) as f:
            z = {k: f[k] for k in f.files}
        dets = [int(c) for c in z["det_ch"]]
        if not dets or not len(z["cand"]):
            continue
        for a, b in GI.split_range(W0, W0 + minutes * 60 * 1000, GI.CHUNK_MS):
            det, ph, sig, _ = GI.render(z, a, dets, (b - a) // 1000)
            x = GI.assemble(det, ph, sig)
            out.append((x, det.shape[0], ph.shape[0]))
            pairs.append(x.shape[0])
    return out, float(np.mean(pairs))


def timed(fn, xs, repeat: int = 1):
    base = rss_mb()
    peak = base
    t0 = time.perf_counter()
    for _ in range(repeat):
        for x in xs:
            fn(x)
            peak = max(peak, rss_mb())
    return (time.perf_counter() - t0) / repeat, peak - base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(REPO / "models" / "final_v2" / "gru.onnx"))
    ap.add_argument("--ckpt", default="gru2_final")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--torch", action="store_true", help="add the PyTorch-CPU arm")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    res = {"threads": a.threads, "onnx_bytes": Path(a.onnx).stat().st_size}

    from gru_onnx import GruPhaseModel
    m = GruPhaseModel(a.onnx, threads=a.threads)
    import onnxruntime as ort
    res["onnxruntime_version"] = ort.__version__

    # ---- numerical equality against the PyTorch model it was exported from --
    import torch
    from neural.export_gru import PhaseOnly, load_ckpt
    torch.set_num_threads(a.threads)
    ref = PhaseOnly(load_ckpt(a.ckpt)).eval()
    diffs, same = [], True
    for x, D, K in rasters(3, 30)[0]:
        with torch.no_grad():
            lt = ref(torch.from_numpy(x)).numpy().reshape(D, K)
        lo = m.logits(x).reshape(D, K)
        pt = np.exp(lt - lt.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)
        po = np.exp(lo - lo.max(1, keepdims=True)); po /= po.sum(1, keepdims=True)
        diffs.append(float(np.abs(pt - po).max()))
        same &= bool((pt.argmax(1) == po.argmax(1)).all())
    res["onnx_vs_torch"] = {"max_abs_prob_diff": max(diffs), "argmax_identical": same,
                            "n_pair_blocks": len(diffs)}
    log(f"onnx vs torch: max |dp| = {max(diffs):.2e}, argmax identical = {same}")

    # ---- speed and memory ---------------------------------------------------
    res["speed"] = {}
    for name, nsig, mins in (("1 signal x 30 min", 1, 30), ("1 signal x 2 h", 1, 120),
                             ("20 signals x 30 min", 20, 30)):
        blocks, pairs = rasters(nsig, mins)
        xs = [x for x, _, _ in blocks]
        t, mem = timed(m.logits, xs)
        row = {"n_signals": nsig, "minutes": mins, "n_windows": len(xs),
               "mean_pairs_per_window": round(pairs, 1),
               "onnx_secs": round(t, 3), "onnx_secs_per_signal": round(t / nsig, 3),
               "onnx_extra_rss_mb": round(mem, 1)}
        if a.torch:
            def run_t(x, r=ref):
                with torch.no_grad():
                    return r(torch.from_numpy(x)).numpy()
            t2, mem2 = timed(run_t, xs)
            row.update(torch_secs=round(t2, 3),
                       torch_secs_per_signal=round(t2 / nsig, 3),
                       torch_extra_rss_mb=round(mem2, 1))
        res["speed"][name] = row
        log(f"{name}: onnx {row['onnx_secs']:.2f}s (+{row['onnx_extra_rss_mb']:.0f} MB)"
            + (f", torch {row['torch_secs']:.2f}s" if a.torch else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"runtime_bench_onnx_t{a.threads}.json", "w"), indent=1,
              default=str)
    log(f"wrote {OUT / f'runtime_bench_onnx_t{a.threads}.json'}")


if __name__ == "__main__":
    main()
