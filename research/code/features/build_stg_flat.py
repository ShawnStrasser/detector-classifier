"""Second view of the staging event cache that KEEPS `DeviceId` as a column.

`src/predict.py` takes raw events with a DeviceId column, so it cannot read the
hive-partitioned cache (where DeviceId lives only in the path).  This writes
`stg/cache/events_flat/dev_part=<guid>/*.parquet` -- partitioned on a duplicate of the
key, so each file still carries DeviceId and a per-signal glob is a single small read.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, connect  # noqa: E402

STG = DC_WORK / "official" / "stg"
SRC = (STG / "cache" / "events" / "**" / "*.parquet").as_posix()
OUT = STG / "cache" / "events_flat"


def main() -> None:
    con = connect(threads=10)
    con.execute(f"SET temp_directory='{(STG/'tmp').as_posix()}'")
    if OUT.exists():
        shutil.rmtree(OUT)
    t0 = time.time()
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""
        COPY (SELECT DeviceId, Timestamp, EventId, Parameter, DeviceId AS dev_part
              FROM read_parquet('{SRC}', hive_partitioning=true)
              ORDER BY DeviceId, Timestamp)
        TO '{OUT.as_posix()}'
        (FORMAT parquet, PARTITION_BY (dev_part), OVERWRITE_OR_IGNORE,
         FILENAME_PATTERN 'ev_{{uuid}}', COMPRESSION zstd, ROW_GROUP_SIZE 200000)""")
    print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
