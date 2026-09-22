"""Task 4 step 0 - does the Parameter of overlap events 61-66 line up with `call_overlap`?

ODOT overlaps are NUMBERS.  This checks, on the Dec-2024 raw data (the only source on disk
with events 61-66 at the time of writing), that the overlap numbers that appear as event
Parameters are the same numbers the timing database uses in `call_overlap` /
`additional_call_overlaps`, with no offset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, RAW_DAYS, connect  # noqa: E402

OFFICIAL = DC_WORK / "official"
pd.set_option("display.width", 220)


def main() -> None:
    con = connect(threads=8)
    src = RAW_DAYS[3].as_posix()
    ev = con.sql(f"""SELECT DeviceId, EventId, Parameter, count(*) n
                     FROM read_parquet('{src}')
                     WHERE EventId BETWEEN 61 AND 66 GROUP BY 1,2,3""").df()
    print("=== overlap event Parameter range")
    print(ev.groupby("EventId").Parameter.agg(["min", "max", "nunique"]).to_string())
    print("\n=== overlap numbers seen per event code (61=begin green)")
    print(ev[ev.EventId == 61].Parameter.value_counts().sort_index().to_string())

    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    used = off[off.call_overlap > 0]
    print("\n=== call_overlap numbers in the timing database")
    print(used.call_overlap.value_counts().sort_index().to_string())

    # per-signal set comparison
    evs = (ev[ev.EventId == 61].groupby("DeviceId").Parameter
           .apply(lambda s: set(int(x) for x in s)).rename("event_overlaps"))
    pls = (used.groupby("DeviceId").call_overlap
           .apply(lambda s: set(int(x) for x in s)).rename("plan_overlaps"))
    j = pd.concat([evs, pls], axis=1).dropna()
    j["plan_subset_of_events"] = [p <= e for e, p in zip(j.event_overlaps, j.plan_overlaps)]
    j["missing"] = [sorted(p - e) for e, p in zip(j.event_overlaps, j.plan_overlaps)]
    print(f"\n=== signals with both a call_overlap and overlap events: {len(j)}")
    print(f"plan overlap numbers that also run as events: "
          f"{int(j.plan_subset_of_events.sum())}/{len(j)}")
    bad = j[~j.plan_subset_of_events]
    if len(bad):
        print(bad.head(15).to_string())
    # offset test: would +/-1 explain the mismatches?
    for off_k in (-1, 1):
        ok = sum(set(x + off_k for x in p) <= e
                 for e, p in zip(j.event_overlaps, j.plan_overlaps))
        print(f"if plan numbers were shifted by {off_k:+d}: {ok}/{len(j)} would match")


if __name__ == "__main__":
    main()
