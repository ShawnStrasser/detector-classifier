"""Stage 07 step 0: rebuild the pair frame cache and score the CHAMPION baseline.

The champion's out-of-fold predictions over the full variant-B 22-window mix already exist
(stage 06): `dc_work/function_v3/phase_stage1_v3.parquet` (ranker) and
`phase_oof_v3_bywindow.parquet` (ranker -> joint decoder).  They are scored here with the
stage-07 harness so every later experiment has a paired reference.

    python src/tune/baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from tune_common import (FOLDS_V3, FUNC_V3, TUNE, add_scorable, dump,  # noqa: E402
                         load_phase_pairs, log, per_fold_primary, phase_metrics,
                         top1_table)

KEYS = ["DeviceId", "Detector", "win", "cand_phase", "Phase", "fold", "scorable"]


def keys_frame() -> pd.DataFrame:
    df = load_phase_pairs()
    df = add_scorable(df)
    return df[KEYS].copy()


def main() -> None:
    k = keys_frame()
    log(f"pair frame keys {k.shape}; scorable rows {int(k.scorable.sum()):,}")
    folds = pd.read_csv(FOLDS_V3)
    out = {}
    for tag, f, col in (("ranker", FUNC_V3 / "phase_stage1_v3.parquet", "p0"),
                        ("decoded", FUNC_V3 / "phase_oof_v3_bywindow.parquet", "prob")):
        pr = pd.read_parquet(f)
        pr["Detector"] = pr.Detector.astype(k.Detector.dtype)
        pr["cand_phase"] = pr.cand_phase.astype(k.cand_phase.dtype)
        m = k.merge(pr, on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
        t = top1_table(m, m[col].to_numpy())
        out[tag] = phase_metrics(t)
        out[tag]["per_fold_primary"] = {int(a): round(float(b), 5) for a, b
                                        in per_fold_primary(t, folds).items()}
        log(f"{tag}: {out[tag]}")
        t.to_parquet(TUNE / f"top1_champion_{tag}.parquet", index=False)
    dump(out, "baseline_champion.json")


if __name__ == "__main__":
    main()
