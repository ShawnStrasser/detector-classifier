"""Counts of every split, for the report's hold-out statement."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, FOLDS_CSV  # noqa: E402

OFFICIAL = DC_WORK / "official"
pd.set_option("display.width", 220)


def main() -> None:
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    off["Detector"] = off.Detector.astype(int)
    folds = pd.read_csv(FOLDS_CSV)
    test = pd.read_parquet(DC_WORK / "labels_test.parquet")
    sp = pd.read_csv(OFFICIAL / "new_signal_split.csv")
    sig = pd.read_csv(OFFICIAL / "stg" / "signals.csv")
    hand = pd.concat([pd.read_parquet(DC_WORK / "labels_dev.parquet"), test],
                     ignore_index=True)
    hand["Detector"] = hand.Detector.astype(int)

    groups = {
        "DEV (train/CV, Dec-2024)": set(folds.DeviceId),
        "  of which fold 0 (2025 hold-out)": set(folds[folds.fold == 0].DeviceId),
        "TEST (locked, never used)": set(test.DeviceId),
        "NEWTRAIN (may train, Sept-2026)": set(sp[sp.split == "NEWTRAIN"].DeviceId),
        "NEWTEST (locked, never used)": set(sp[sp.split == "NEWTEST"].DeviceId),
    }
    rows = []
    for name, ids in groups.items():
        o = off[off.DeviceId.isin(ids)]
        rows.append({
            "split": name, "signals": len(ids),
            "signals_with_official_plan": o.DeviceId.nunique(),
            "programmed_channels": len(o),
            "real_Dec2024": int(o.real_dec2024.sum()),
            "real_Sept2026": int(o.real_staging.sum()),
            "hand_phase_labels": int(hand.DeviceId.isin(ids).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nstaging cache signals: {len(sig)} "
          f"(DEV {int((sig.group=='DEV').sum())}, NEW {int((sig.group=='NEW').sum())}); "
          "TEST excluded from the cache entirely")
    print(f"official plan rows total {len(off):,} on {off.DeviceId.nunique()} signals; "
          f"real in either dataset {int((off.real_dec2024|off.real_staging).sum()):,}")


if __name__ == "__main__":
    main()
