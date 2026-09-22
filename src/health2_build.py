"""Stage 09 driver: build `bad_intervals` + per-detector health summary for the cache.

Generic logic lives in `src/health2.py`; this file only wires it to the project's
event cache and writes artefacts under `$DC_WORK/health2/`.

(A sibling package `src/health2/` would shadow `src/health2.py` on import, hence the
flat `health2_*.py` naming.)

    python src/health2_build.py                # all 375 DEV signals
    python src/health2_build.py --limit 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DC_WORK, FOLDS_CSV, connect  # noqa: E402
import health2 as h2  # noqa: E402

OUT = DC_WORK / "health2"
T0, T1 = pd.Timestamp("2024-12-02"), pd.Timestamp("2024-12-05")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--stuck-on-s", type=float, default=h2.DEFAULTS.stuck_on_s)
    ap.add_argument("--out", default="bad_intervals.parquet")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    devs = pd.read_csv(FOLDS_CSV).DeviceId.tolist()
    if a.limit:
        devs = devs[:a.limit]
    params = h2.HealthParams(stuck_on_s=a.stuck_on_s)

    con = connect(memory_limit="6GB", threads=6)
    parts, summ = [], []
    t_start = time.time()
    for i in range(0, len(devs), a.chunk):
        grp = devs[i:i + a.chunk]
        lst = ",".join("'" + d + "'" for d in grp)
        con.execute(f"""CREATE OR REPLACE TEMP VIEW ev AS
            SELECT DeviceId, Timestamp, EventId, Parameter
            FROM read_parquet('{(CACHE/'events'/'**'/'*.parquet').as_posix()}',
                              hive_partitioning=true)
            WHERE DeviceId IN ({lst})""")
        bad = h2.find_bad_intervals(con, "ev", params=params)
        univ = con.sql("SELECT DISTINCT DeviceId, Detector FROM _h2_iv").df()
        parts.append(bad)
        summ.append(h2.summarise(bad, univ, T0, T1))
        print(f"  {i+len(grp)}/{len(devs)} signals  {time.time()-t_start:.0f}s  "
              f"{sum(len(p) for p in parts)} intervals", flush=True)

    bad = pd.concat(parts, ignore_index=True)
    bad.to_parquet(OUT / a.out, index=False)
    s = pd.concat(summ, ignore_index=True).fillna(0.0)
    s.to_parquet(OUT / "detector_health2.parquet", index=False)

    print(f"\n{len(bad):,} bad intervals, {len(s):,} detectors, "
          f"{time.time()-t_start:.0f}s")
    bad["_k"] = bad.DeviceId.astype(str) + "|" + bad.Detector.astype(str)
    print(bad.groupby("reason").agg(n=("secs", "size"), hours=("secs", lambda x: x.sum()/3600),
                                    n_det=("_k", "nunique"),
                                    n_sig=("DeviceId", "nunique")).to_string())
    print("\nper-detector masked share (of 72 h):")
    print(s.masked_frac.describe(percentiles=[.5, .9, .99]).to_string())
    print(f"detectors with any masked time: {(s.masked_frac > 0).sum():,} "
          f"({(s.masked_frac > 0).mean():.1%})")


if __name__ == "__main__":
    main()
