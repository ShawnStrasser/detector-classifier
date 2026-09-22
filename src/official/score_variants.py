"""Recompute the full metric block for the saved variant OOF files (seed 0).

`run_train.py --step variants` writes `train/oof_bywindow_{A,B,C}.parquet`; this rescores
them all against the OFFICIAL labels on identical rows so A / B / C are directly
comparable in one table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DC_WORK  # noqa: E402
import run_train as R  # noqa: E402
import train_official as T  # noqa: E402

OUT = DC_WORK / "official" / "train"
pd.set_option("display.width", 240)


def main() -> None:
    ev, _ = R.build_eval()
    key = ["DeviceId", "Detector", "win", "cand_phase"]
    res = {}
    for v in ("A", "B", "C"):
        f = OUT / f"oof_bywindow_{v}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f)
        m = ev[key].merge(p, on=key, how="left")
        t = T.top1(ev, m.prob.to_numpy())
        r = T.metrics(t, T.FULL_DEC)
        res[v] = r
        print(f"{v}: " + json.dumps({k: (round(x, 4) if isinstance(x, float) else x)
                                     for k, x in r.items() if k != "per_fold_primary"}))
    json.dump(res, open(OUT / "variants_seed0_detail.json", "w"), indent=1, default=str)
    d = pd.DataFrame(res).T[["primary", "acc_full", "acc_m30", "acc_nonstd_full",
                             "acc_fold0_full", "acc_fold0_m30", "cov90", "acc_at_cov90",
                             "n_err_full", "n_err_concurrent", "n_full", "n_nonstd_full"]]
    print()
    print(d.round(4).to_string())


if __name__ == "__main__":
    main()
