"""Run the 2025 baseline BiLSTM (baseline/model5.pth) on the 38 fold-0 (valid) signals.

Faithful re-implementation of `baseline/inference.ipynb`::DetectorInferenceProcessor:
  * events 1,7,43,44,81,82 with Parameter <= 64
  * 20 channels: PhaseGreen1-9, PhaseCall1-9, DetectorState, Delta (seconds)
  * synthetic "all phases off / no call" events at the first timestamp of the chunk
  * per-detector stream = that detector's 81/82 events merged with ALL of the signal's phase events,
    sorted by (Timestamp, EventId), states forward-filled, Delta = diff within the stream,
    rows with any null dropped (i.e. everything before the detector's first event)
  * non-overlapping windows of 1000 rows, keep a window iff sum(DetectorState) >= 10 and max(Delta) <= 100 s
  * prediction = mean of softmax over the kept windows

Speed-up vs the notebook (mathematically identical): the 18 phase-state columns are forward-filled once
per signal-day on the phase-event stream, then merged into each detector's stream; detector events sort
after same-timestamp phase events because 81/82 > 1/7/43/44, so the ffill result is unchanged.

Outputs long-format phase probs + function probs per (DeviceId, Detector) for two windows of data:
  full   = Dec 2-4 2024, at most 20 windows per detector per day (<= 60 total, spread over the 3 days)
  6h     = Dec 3 2024 09:00-15:00 only (mirrors the Feb-2025 statewide run), no cap
"""
import argparse
import glob
import os
import time

import numpy as np
import polars as pl
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_old import LSTM  # noqa: E402

WORK = r"C:\Users\hwyr67g\dc_work"
CACHE = WORK + r"\cache\fold0_events"
WEIGHTS = r"S:\Data_Analysis\Python\detector-classifier\baseline\model5.pth"

SEQ = 1000
MIN_ACT = 10
MAX_DELTA = 100
FEATS = [f"PhaseGreen{p}" for p in range(1, 10)] + [f"PhaseCall{p}" for p in range(1, 10)] + \
        ["DetectorState", "Delta"]
# pd.get_dummies(target['Function']) in baseline/dataprep.py sorts alphabetically over
# {Advance, Count, Presence} -> this is the function head's class order.
FUNCTION_LABELS = ["Advance", "Count", "Presence"]


def phase_state_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Phase events (+ synthetic zero-state events) with the 18 forward-filled phase columns."""
    min_ts = df["Timestamp"].min()
    phase_events = df.filter(pl.col("EventId").is_in([1, 7, 43, 44])).select(["Timestamp", "EventId", "Parameter"])
    init = pl.DataFrame({
        "Timestamp": [min_ts] * 18,
        "EventId": [7] * 9 + [44] * 9,
        "Parameter": list(range(1, 10)) * 2,
    }, schema={"Timestamp": phase_events.schema["Timestamp"], "EventId": pl.Int32, "Parameter": pl.Int32})
    phase_events = pl.concat([phase_events.cast({"EventId": pl.Int32, "Parameter": pl.Int32}), init])

    exprs = []
    for p in range(1, 10):
        exprs.append(
            pl.when((pl.col("Parameter") == p) & (pl.col("EventId") == 1)).then(1.0)
            .when((pl.col("Parameter") == p) & (pl.col("EventId") == 7)).then(0.0)
            .otherwise(None).cast(pl.Float32).alias(f"PhaseGreen{p}"))
        exprs.append(
            pl.when((pl.col("Parameter") == p) & (pl.col("EventId") == 43)).then(1.0)
            .when((pl.col("Parameter") == p) & (pl.col("EventId") == 44)).then(0.0)
            .otherwise(None).cast(pl.Float32).alias(f"PhaseCall{p}"))

    ph = phase_events.sort(["Timestamp", "EventId"]).with_columns(exprs)
    ph = ph.with_columns([pl.col(c).forward_fill() for c in FEATS[:18]])
    return ph.select(["Timestamp", "EventId"] + FEATS[:18]).with_columns(
        pl.lit(None, dtype=pl.Float32).alias("DetectorState"))


def detector_windows(ph: pl.DataFrame, det_ev: pl.DataFrame):
    """Return (n_rows, features array of shape (n_win, 1000, 20), valid mask) for one detector."""
    det = det_ev.select([
        "Timestamp",
        pl.col("EventId").cast(pl.Int32),
        *[pl.lit(None, dtype=pl.Float32).alias(c) for c in FEATS[:18]],
        pl.when(pl.col("EventId") == 82).then(1.0).when(pl.col("EventId") == 81).then(0.0)
        .otherwise(None).cast(pl.Float32).alias("DetectorState"),
    ])
    s = pl.concat([ph, det]).sort(["Timestamp", "EventId"])
    s = s.with_columns([pl.col(c).forward_fill() for c in FEATS[:18] + ["DetectorState"]])
    s = s.with_columns(
        (pl.col("Timestamp").diff() / pl.duration(nanoseconds=1_000_000_000)).cast(pl.Float32).alias("Delta")
    ).select(FEATS).drop_nulls()

    n = s.height
    nwin = n // SEQ
    if nwin == 0:
        return n, None
    arr = s.head(nwin * SEQ).to_numpy().astype(np.float32).reshape(nwin, SEQ, 20)
    act = arr[:, :, 18].sum(axis=1)
    mx = arr[:, :, 19].max(axis=1)
    keep = (act >= MIN_ACT) & (mx <= MAX_DELTA)
    return n, (arr, keep)


def pick(idx: np.ndarray, cap: int) -> np.ndarray:
    if cap is None or len(idx) <= cap:
        return idx
    sel = np.linspace(0, len(idx) - 1, cap).round().astype(int)
    return idx[np.unique(sel)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "6h"], default="full")
    ap.add_argument("--cap-per-day", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTM()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.to(dev).eval()

    devices = sorted(os.path.basename(d).split("=")[1]
                     for d in glob.glob(os.path.join(CACHE, "DeviceId=*")))
    days = [3] if args.mode == "6h" else [2, 3, 4]
    cap = None if args.mode == "6h" else args.cap_per_day

    rows = []           # (DeviceId, Detector, n_windows, phase_probs(9), func_probs(3))
    t0 = time.time()
    for di, device in enumerate(devices):
        per_det_windows = {}
        per_det_rows = {}
        for day in days:
            files = glob.glob(os.path.join(CACHE, f"DeviceId={device}", f"day={day}", "*.parquet"))
            if not files:
                continue
            df = pl.read_parquet(files).select(["Timestamp", "EventId", "Parameter"])
            if args.mode == "6h":
                df = df.filter((pl.col("Timestamp") >= pl.datetime(2024, 12, 3, 9, 0, 0)) &
                               (pl.col("Timestamp") < pl.datetime(2024, 12, 3, 15, 0, 0)))
            if df.height == 0:
                continue
            ph = phase_state_frame(df)
            det_ev = df.filter(pl.col("EventId").is_in([81, 82]))
            for (d,), grp in det_ev.group_by(["Parameter"], maintain_order=True):
                n, res = detector_windows(ph, grp)
                per_det_rows[d] = per_det_rows.get(d, 0) + n
                if res is None:
                    continue
                arr, keep = res
                idx = pick(np.flatnonzero(keep), cap)
                if len(idx):
                    per_det_windows.setdefault(d, []).append(arr[idx])
            del ph, df, det_ev

        for d in sorted(set(list(per_det_rows.keys()) + list(per_det_windows.keys()))):
            chunks = per_det_windows.get(d)
            if not chunks:
                rows.append((device, int(d), 0, np.full(9, 1.0 / 9), np.full(3, 1.0 / 3)))
                continue
            X = np.concatenate(chunks, axis=0)
            pp, fp = [], []
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    xb = torch.from_numpy(X[i:i + 128]).to(dev)
                    pl_, fl_ = model(xb)
                    pp.append(torch.softmax(pl_, dim=1).cpu().numpy())
                    fp.append(torch.softmax(fl_, dim=1).cpu().numpy())
            pp = np.concatenate(pp).mean(axis=0)
            fp = np.concatenate(fp).mean(axis=0)
            rows.append((device, int(d), len(X), pp.astype(np.float64), fp.astype(np.float64)))
        print(f"[{di + 1}/{len(devices)}] {device} dets={len(per_det_rows)} "
              f"t={time.time() - t0:.0f}s", flush=True)

    dets = pl.DataFrame({
        "DeviceId": [r[0] for r in rows],
        "Detector": [r[1] for r in rows],
        "n_windows": [r[2] for r in rows],
        **{f"p{p}": [float(r[3][p - 1]) for r in rows] for p in range(1, 10)},
        **{f"f_{FUNCTION_LABELS[i].lower()}": [float(r[4][i]) for r in rows] for i in range(3)},
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dets.write_parquet(args.out)
    print("wrote", args.out, dets.shape)


if __name__ == "__main__":
    main()
