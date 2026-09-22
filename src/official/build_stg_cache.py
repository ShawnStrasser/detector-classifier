"""Task 2/3 step 0 - build a parallel DC_WORK root for the Sept-2026 STAGING data.

Creates `dc_work/official/stg/` with exactly the layout `src/common.py` expects, so
`features.py`, `features_v2.py`, `cross_detector.py` and the training code can be driven
against the staging data by setting `DC_WORK=...\\dc_work\\official\\stg` -- nothing in the
Dec-2024 work dir is touched.

    stg/cache/events/DeviceId=<guid>/s<n>_<uuid>.parquet
    stg/cache/{det_intervals,phase_cycles,green_state,coord_state,signal_meta,detector_meta}.parquet
    stg/signals.csv          DeviceId, group (DEV / NEW)

The 43 TEST signals are EXCLUDED from the cache entirely.

    python src/official/build_stg_cache.py --step all
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ALLOWED_EVENTS, DC_WORK, MAX_DETECTOR_CHANNEL, REPO, connect  # noqa: E402

STG = DC_WORK / "official" / "stg"
SCACHE = STG / "cache"
STAGING = (DC_WORK / "data" / "staging" / "date=*" / "part_*.parquet").as_posix()
PLANS = (REPO / "data" / "detector_plans.parquet").as_posix()
EV_LIST = ",".join(str(e) for e in ALLOWED_EVENTS)
DATES = ["2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def signal_list(con) -> pd.DataFrame:
    dev = set(pd.read_csv(DC_WORK / "folds.csv").DeviceId)
    test = set(pd.read_parquet(DC_WORK / "labels_test.parquet").DeviceId)
    plans = set(con.sql(f"SELECT DISTINCT DeviceId FROM read_parquet('{PLANS}')").df().DeviceId)
    stg = set(con.sql(
        f"SELECT DeviceId FROM read_parquet('{STAGING}') WHERE EventId=82 "
        f"AND Parameter BETWEEN 1 AND {MAX_DETECTOR_CHANNEL} GROUP BY 1 HAVING count(*) >= 100"
    ).df().DeviceId)
    keep = (stg & plans) - test
    out = pd.DataFrame({"DeviceId": sorted(keep)})
    out["group"] = ["DEV" if d in dev else "NEW" for d in out.DeviceId]
    log(f"staging={len(stg)} plans={len(plans)} keep={len(keep)} "
        f"(DEV {int((out.group=='DEV').sum())}, NEW {int((out.group=='NEW').sum())}); "
        f"TEST excluded ({len(test)})")
    return out


def build_events(con, devs: list[str]) -> None:
    out = (SCACHE / "events").as_posix()
    if (SCACHE / "events").exists():
        shutil.rmtree(SCACHE / "events")
        log("removed existing stg cache/events")
    con.execute("SET preserve_insertion_order=true")
    con.execute("CREATE OR REPLACE TEMP TABLE keep AS SELECT unnest(?::VARCHAR[]) AS DeviceId",
                [devs])
    for i, d in enumerate(DATES):
        src = (DC_WORK / "data" / "staging" / f"date={d}" / "part_*.parquet").as_posix()
        t0 = time.time()
        con.execute(f"""
            COPY (
              SELECT DISTINCT DeviceId, Timestamp,
                     EventId::UTINYINT AS EventId, Parameter::UTINYINT AS Parameter
              FROM read_parquet('{src}') e
              WHERE e.EventId IN ({EV_LIST})
                AND NOT (e.EventId IN (81,82) AND e.Parameter > {MAX_DETECTOR_CHANNEL})
                AND e.DeviceId IN (SELECT DeviceId FROM keep)
              ORDER BY DeviceId, Timestamp
            ) TO '{out}'
            (FORMAT parquet, PARTITION_BY (DeviceId), APPEND,
             FILENAME_PATTERN 's{i}_{{uuid}}', COMPRESSION zstd, ROW_GROUP_SIZE 200000)
        """)
        log(f"{d} -> stg cache/events in {time.time()-t0:.0f}s")
    con.execute("SET preserve_insertion_order=false")
    n = con.sql(f"SELECT count(*) c, count(DISTINCT DeviceId) d FROM "
                f"read_parquet('{out}/**/*.parquet', hive_partitioning=true)").fetchone()
    log(f"stg cache/events: {n[0]:,} rows, {n[1]} devices")


def build_derived(con) -> None:
    import build_cache as bc
    # build_cache's derived SQL is written against its module-level CACHE constant;
    # re-point it at the staging cache for the duration of this call.
    orig = bc.CACHE
    bc.CACHE = SCACHE
    try:
        bc.build_derived(con)
    finally:
        bc.CACHE = orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all", "signals", "events", "derived"])
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    for d in (SCACHE, STG / "features", STG / "preds", STG / "models", STG / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    con = connect(threads=a.threads)
    con.execute(f"SET temp_directory='{(STG/'tmp').as_posix()}'")
    sig_path = STG / "signals.csv"
    if a.step in ("all", "signals") or not sig_path.exists():
        sig = signal_list(con)
        sig.to_csv(sig_path, index=False)
        log(f"wrote {sig_path}")
    sig = pd.read_csv(sig_path)
    if a.step in ("all", "events"):
        build_events(con, sig.DeviceId.tolist())
    if a.step in ("all", "derived"):
        build_derived(con)
    log("done")


if __name__ == "__main__":
    main()
