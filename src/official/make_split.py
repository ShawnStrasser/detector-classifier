"""Task 2c - split the never-seen NEW signals once, by signal, seed 0: 70 % NEWTRAIN / 30 % NEWTEST.

NEWTEST is locked away exactly like the original 43 TEST signals: it is never trained or
tuned on.  Written to `dc_work/official/new_signal_split.csv` and
`dc_work/official/newtest_signals.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK  # noqa: E402

OFFICIAL = DC_WORK / "official"


def main() -> None:
    sig = pd.read_csv(OFFICIAL / "stg" / "signals.csv")
    new = np.array(sorted(sig[sig.group == "NEW"].DeviceId))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(new))
    n_tr = int(round(0.70 * len(new)))
    split = np.array(["NEWTEST"] * len(new), dtype=object)
    split[perm[:n_tr]] = "NEWTRAIN"
    out = pd.DataFrame({"DeviceId": new, "split": split}).sort_values("DeviceId")
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    nd = off[off.real_staging].groupby("DeviceId").size().rename("n_detectors")
    out = out.merge(nd, on="DeviceId", how="left")
    out.to_csv(OFFICIAL / "new_signal_split.csv", index=False)
    out[out.split == "NEWTEST"][["DeviceId", "n_detectors"]].to_csv(
        OFFICIAL / "newtest_signals.csv", index=False)
    print(out.groupby("split").agg(signals=("DeviceId", "size"),
                                   detectors=("n_detectors", "sum")).to_string())
    print(f"wrote {OFFICIAL/'new_signal_split.csv'} and {OFFICIAL/'newtest_signals.csv'}")


if __name__ == "__main__":
    main()
