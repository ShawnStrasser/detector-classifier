"""Track B (stage 15): held-out predictions for a Track B checkpoint.

Thin wrapper around `neural/infer2.py`: same 22 windows, same 30-minute pieces pooled by
mean log-probability, same output contract, so the rows line up one-for-one with the
stage-13 GRU files and with the shipped LightGBM out-of-fold table.  Only the
directories change (`%DC_WORK%/trackB/{models,preds}`).

The checkpoint is loaded with the plain `neural.models.PairNet` -- dropout carries no
parameters, so a Track B state dict is a shipping-shaped state dict.

    python research/code/neural/trackb_infer.py --ckpt tb_b1_tcn_f0 --fold 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402
from neural import data2 as D2  # noqa: E402
from neural import infer2 as I2  # noqa: E402

TRACKB = DC_WORK / "trackB"
MODELDIR = TRACKB / "models"
OUT = TRACKB / "preds"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=32)
    ap.add_argument("--max-bs", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=180)
    a = ap.parse_args()

    I2.MODELDIR = MODELDIR                       # so `--ckpt <tag>` resolves in trackB
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or (f"{a.ckpt}_all" if a.all else f"{a.ckpt}_oof")
    dest = OUT / f"{tag}_bywindow.parquet"
    if dest.exists():
        I2.log(f"{dest.name} exists -- skipping")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sigs = D2.training_signals()
    table = D2.load_table(sigs, labelled_only=False)
    keys = [k for k in (sigs.key if a.all else sigs.loc[sigs.fold == a.fold, "key"])
            if k in table]
    I2.log(f"{tag}: {len(keys)} signals, "
           f"{sum(len(table[k]['dets']) for k in keys):,} labelled channels")
    model = I2.load_model(a.ckpt, device)
    t0 = time.time()
    df = I2.score_windows(model, table, keys, device, D2.Stores(), a.max_chunks,
                          a.budget, a.max_bs, a.chunk)
    df.to_parquet(dest, index=False)
    I2.log(f"wrote {dest} ({len(df):,} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
