"""Stage 03: stitch the per-fold OOF prediction parts into one contract file."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PREDS  # noqa: E402


def main(arch: str = "gru", out: str = "neural_best_oof"):
    for kind, keys in (("phase", ["DeviceId", "Detector", "cand_phase"]),
                       ("function", ["DeviceId", "Detector"]),
                       ("bywindow", ["DeviceId", "Detector", "cand_phase", "win"]),
                       ("bywindowfn", ["DeviceId", "Detector", "win"])):
        parts = sorted(PREDS.glob(f"oofpart_{arch}_f*_{kind}.parquet"))
        if not parts:
            print(f"no {kind} parts"); continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        assert not df.duplicated(keys).any(), f"{kind}: duplicate keys across folds"
        df.to_parquet(PREDS / f"{out}_{kind}.parquet", index=False)
        print(f"{out}_{kind}.parquet: {len(df):,} rows, "
              f"{df[['DeviceId','Detector']].drop_duplicates().shape[0]:,} detectors, "
              f"{len(parts)} folds")


if __name__ == "__main__":
    main(*sys.argv[1:])
