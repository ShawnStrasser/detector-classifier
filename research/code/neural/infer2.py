"""Stage 13: held-out (and in-pool) GRU predictions in the protocol contract format.

The windows are EXACTLY the 22 the shipped LightGBM pipeline's out-of-fold file uses
(`dc_work/official/final_v1/oof_bywindow.parquet`), so the two models can be blended and
scored on identical rows:

    m5_a..d  m10_a..d  m30_a..d  h1_a..c  h3_a..b  h6_a..b  h24_a..b  full72 / full66

A window longer than 30 minutes is cut into 30-minute pieces and the pieces are pooled by
the mean log-probability -- the rule `src/gru_onnx.py` also uses at inference, so these
numbers describe the shipped runtime.  Very long windows are sub-sampled to at most
`--max-chunks` pieces (stage-03 practice).

    python src/neural/infer2.py --fold 0                 # that fold's held-out signals
    python src/neural/infer2.py --ckpt gru2_final --all   # every training signal
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402
from neural import data2 as D2  # noqa: E402
from neural.models import PairNet  # noqa: E402
from neural.train2 import forward_batch  # noqa: E402

MODELDIR = DC_WORK / "models" / "gru2"
OUT = DC_WORK / "preds" / "gru2"
CHUNK_MS = 30 * 60 * 1000
H = 3600 * 1000

# (name, start timestamp, seconds) per period -- copied from src/features.py,
# src/function_v3_prep.py and src/official/windows_stg.py
WINDOWS = {
    "dec": [
        ("m5_a", "2024-12-02 07:45:00", 300), ("m5_b", "2024-12-03 12:20:00", 300),
        ("m5_c", "2024-12-03 22:10:00", 300), ("m5_d", "2024-12-04 17:05:00", 300),
        ("m10_a", "2024-12-02 08:05:00", 600), ("m10_b", "2024-12-03 13:00:00", 600),
        ("m10_c", "2024-12-04 02:30:00", 600), ("m10_d", "2024-12-04 17:20:00", 600),
        ("m30_a", "2024-12-02 07:30:00", 1800), ("m30_b", "2024-12-03 12:00:00", 1800),
        ("m30_c", "2024-12-03 21:30:00", 1800), ("m30_d", "2024-12-04 16:45:00", 1800),
        ("h1_a", "2024-12-02 17:00:00", 3600), ("h1_b", "2024-12-03 02:00:00", 3600),
        ("h1_c", "2024-12-04 09:00:00", 3600),
        ("h3_a", "2024-12-02 06:00:00", 10800), ("h3_b", "2024-12-03 14:00:00", 10800),
        ("h6_a", "2024-12-03 06:00:00", 21600), ("h6_b", "2024-12-04 12:00:00", 21600),
        ("h24_a", "2024-12-02 00:00:00", 86400), ("h24_b", "2024-12-04 00:00:00", 86400),
        ("full72", "2024-12-02 00:00:00", 259200),
    ],
    "stg": [
        ("m5_a", "2026-09-21 07:45:00", 300), ("m5_b", "2026-09-19 12:20:00", 300),
        ("m5_c", "2026-09-19 22:10:00", 300), ("m5_d", "2026-09-18 17:05:00", 300),
        ("m10_a", "2026-09-21 08:05:00", 600), ("m10_b", "2026-09-19 13:00:00", 600),
        ("m10_c", "2026-09-20 02:30:00", 600), ("m10_d", "2026-09-18 17:20:00", 600),
        ("m30_a", "2026-09-21 07:30:00", 1800), ("m30_b", "2026-09-19 12:00:00", 1800),
        ("m30_c", "2026-09-19 21:30:00", 1800), ("m30_d", "2026-09-18 17:00:00", 1800),
        ("h1_a", "2026-09-20 17:00:00", 3600), ("h1_b", "2026-09-20 02:00:00", 3600),
        ("h1_c", "2026-09-21 09:00:00", 3600),
        ("h3_a", "2026-09-21 06:00:00", 10800), ("h3_b", "2026-09-19 14:00:00", 10800),
        ("h6_a", "2026-09-20 06:00:00", 21600), ("h6_b", "2026-09-19 12:00:00", 21600),
        ("h24_a", "2026-09-19 00:00:00", 86400), ("h24_b", "2026-09-20 00:00:00", 86400),
        ("full66", "2026-09-18 16:15:00", 66 * 3600),
    ],
}
T0 = {"dec": pd.Timestamp("2024-12-02 00:00:00"), "stg": pd.Timestamp("2026-09-18 00:00:00")}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pieces(period: str, start: str, secs: int, max_chunks: int) -> list[tuple[int, int]]:
    a = int((pd.Timestamp(start) - T0[period]).total_seconds() * 1000)
    lo, hi = D2.PERIODS[period]["lo"], D2.PERIODS[period]["hi"]
    a = max(lo, a)
    b = min(hi, a + secs * 1000)
    total = max(b - a, 60_000)
    if total <= CHUNK_MS:
        return [(a, b)]
    n = total // CHUNK_MS
    out = [(a + i * CHUNK_MS, a + (i + 1) * CHUNK_MS) for i in range(int(n))]
    if total - n * CHUNK_MS >= 60_000:
        out.append((a + int(n) * CHUNK_MS, b))
    if max_chunks and len(out) > max_chunks:
        idx = np.linspace(0, len(out) - 1, max_chunks).round().astype(int)
        out = [out[i] for i in idx]
    return out


def load_model(ckpt: str, device: str) -> PairNet:
    p = Path(ckpt)
    if not p.exists():
        p = MODELDIR / f"{ckpt}.pt"
    d = torch.load(p, map_location=device, weights_only=False)
    m = PairNet(d["arch"], **d.get("kw", {})).to(device).eval()
    m.load_state_dict(d["state"])
    log(f"{p.name}: epoch {d.get('epoch')}, inner-val {d.get('es_score')}")
    return m


@torch.no_grad()
def score_windows(model, table: dict, keys: list[str], device: str, stores,
                  max_chunks: int, budget: int = 180, max_bs: int = 6,
                  chunk: int = 1024) -> pd.DataFrame:
    """-> DeviceId, Detector, cand_phase, win, prob, n_act (one row per pair per window)."""
    acc: dict = {}
    for period in ("dec", "stg"):
        pk = [k for k in keys if table[k]["period"] == period]
        if not pk:
            continue
        for name, start, secs in WINDOWS[period]:
            chunks = pieces(period, start, secs, max_chunks)
            plan = [(k, a, (b - a) // 1000) for k in pk for a, b in chunks]
            ds = D2.FixedDataset(table, plan, stores)
            for bidx in D2.batches_of(plan, budget_minutes=budget, max_bs=max_bs):
                batch = D2.collate([ds[i] for i in bidx])
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device == "cuda")):
                    logit = forward_batch(model, batch, device, chunk=chunk)
                lp = F.log_softmax(logit, dim=2).float().cpu().numpy()
                na = batch["nact"].numpy()
                for b, (key, dev, w0, dets, cand) in enumerate(batch["meta"]):
                    K = len(cand)
                    for j, d in enumerate(dets):
                        e = acc.get((key, name, int(d)))
                        if e is None:
                            e = acc[(key, name, int(d))] = [cand, np.zeros(K), 0, 0.0]
                        e[1] += lp[b, j, :K]
                        e[2] += 1
                        e[3] += float(na[b, j])
            log(f"  {period} {name}: {len(chunks)} piece(s) x {len(pk)} signals")
    rows = []
    for (key, win, det), (cand, s, n, na) in acc.items():
        v = s / max(n, 1)
        p = np.exp(v - v.max())
        p /= p.sum()
        dev = key.split("|", 1)[1]
        suffix = "@stg" if key.startswith("stg|") else ""
        for c, pv in zip(cand, p):
            rows.append((dev + suffix, int(det), int(c), win, float(pv), float(na)))
    return pd.DataFrame(rows, columns=["DeviceId", "Detector", "cand_phase", "win",
                                       "prob", "n_act"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--all", action="store_true", help="score every training signal")
    ap.add_argument("--max-chunks", type=int, default=32)
    ap.add_argument("--max-bs", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=180)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = a.ckpt or f"gru2_f{a.fold}"
    tag = a.tag or (f"gru2_all" if a.all else f"gru2_oof_f{a.fold}")
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{tag}_bywindow.parquet"
    if dest.exists():
        log(f"{dest.name} exists -- skipping")
        return
    sigs = D2.training_signals()
    table = D2.load_table(sigs, labelled_only=False)
    keys = [k for k in (sigs.key if a.all else sigs.loc[sigs.fold == a.fold, "key"])
            if k in table]
    log(f"{tag}: {len(keys)} signals, "
        f"{sum(len(table[k]['dets']) for k in keys):,} labelled channels")
    model = load_model(ckpt, device)
    t0 = time.time()
    df = score_windows(model, table, keys, device, D2.Stores(), a.max_chunks,
                       a.budget, a.max_bs, a.chunk)
    df.to_parquet(dest, index=False)
    log(f"wrote {dest} ({len(df):,} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
