"""Stage 03: architecture selection on fold 5 (the early-stopping fold).

Fold 0 is kept as the held-out number, so the choice between architectures and the
choice of window-pooling rule are both made here, never on fold 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK  # noqa: E402
from neural.data import devs_of, load_signal_table  # noqa: E402
from neural.infer import acc_of, load_model, log, predict_windows, sub_windows  # noqa: E402

SHORT = [("m30_a", "2024-12-02 07:30:00", 1800), ("m30_b", "2024-12-03 12:00:00", 1800),
         ("m30_c", "2024-12-03 21:30:00", 1800), ("m30_d", "2024-12-04 16:45:00", 1800)]
LONG = [("full72", "2024-12-02 00:00:00", 259200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["tcn", "gru", "transformer", "cyc2d"])
    ap.add_argument("--fold", type=int, default=5)
    ap.add_argument("--aggs", nargs="+", default=["prob"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=640)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    devs = devs_of([args.fold])
    table = load_signal_table(devs, labeled_only=False, drop_failed=False)
    devs = [d for d in devs if d in table]
    rows = []
    for c in args.ckpts:
        model, arch = load_model(c, device)
        for agg in args.aggs:
            short = []
            for _, t, s in SHORT:
                a = predict_windows(model, arch, table, devs, sub_windows(t, s), device,
                                    args.workers, args.bs, args.chunk, agg=agg)
                short.append(acc_of(a, table))
            a72 = acc_of(predict_windows(model, arch, table, devs,
                                         sub_windows(*LONG[0][1:]), device, args.workers,
                                         args.bs, args.chunk, agg=agg), table)
            m30_all = sum(x[0] for x in short) / len(short)
            m30_cov = sum(x[1] for x in short) / len(short)
            rows.append(dict(ckpt=c, arch=arch, agg=agg, m30_all=round(m30_all, 4),
                             m30_cov=round(m30_cov, 4), h72_all=round(a72[0], 4),
                             h72_cov=round(a72[1], 4),
                             mean=round((m30_all + a72[0]) / 2, 4)))
            log(str(rows[-1]))
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    print(df.to_string(index=False))
    df.to_csv(DC_WORK / "neural" / f"selection_fold{args.fold}.csv", index=False)


if __name__ == "__main__":
    main()
