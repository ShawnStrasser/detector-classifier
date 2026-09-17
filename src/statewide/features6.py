"""Stage 05: build the v1+v2 pair features and the cross-detector similarity graph
**from event codes 1, 7, 43, 44, 81, 82 only**.

Instead of reading the stage-01 derived cache (which was built from the full code set),
this module rebuilds the same per-chunk temp tables straight from raw events, using only
the six codes.  `features.apply_window`, `features.build_window`, `features_v2.build_window`
and `cross_detector.build_window` are then reused unchanged, so the 6-code model differs
from the full model **only** in what the six codes cannot express.

Cycle table from six codes
--------------------------
green_start = event 1, green end = event 7 (green termination).  `yellow_start` is set to
the green-termination time and `red_start` is NULL, so `features.py`'s
`ge = coalesce(yellow_start, red_start, next_green)` and `rs = coalesce(red_start,
yellow_start)` both collapse onto the green-termination instant: the green bitmask timeline
is identical to the full-code one and everything after green is simply "red".

    python src/statewide/features6.py --source dev        # Dec-2024 DEV, mixed windows
    python src/statewide/features6.py --source statewide  # 2025-02-25, 6 h + sub-windows
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import cross_detector as cdt  # noqa: E402
import features as f1  # noqa: E402
import features_v2 as f2  # noqa: E402
from common import DC_WORK, MAX_DETECTOR_CHANNEL, connect  # noqa: E402
from common6 import (CODES6, DEV_EVENTS_GLOB, SW_EVENTS_GLOB, SW_FEAT,  # noqa: E402
                     WINDOWS_SW, signal_groups)
from features import WINDOWS_MIXED, log  # noqa: E402

EV_LIST = ",".join(str(c) for c in CODES6)


# ------------------------------------------------------------- chunk loading
def load_chunk6(con, devs: list[str], glob: str) -> None:
    """Build the same temp tables `features.load_chunk` makes, from six event codes."""
    dev_list = ",".join("'" + d + "'" for d in devs)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ev AS
        SELECT DISTINCT DeviceId, Timestamp, EventId::UTINYINT AS EventId,
                        Parameter::UTINYINT AS Parameter
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE DeviceId IN ({dev_list}) AND EventId IN ({EV_LIST})
          AND NOT (EventId IN (81,82) AND Parameter > {MAX_DETECTOR_CHANNEL})""")
    con.execute("""CREATE OR REPLACE TEMP TABLE devmap AS
        SELECT DeviceId, row_number() OVER (ORDER BY DeviceId)::SMALLINT AS dev
        FROM (SELECT DISTINCT DeviceId FROM ev)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE onev_all AS
        WITH e AS (SELECT DeviceId, Parameter::SMALLINT AS det, Timestamp AS ts, EventId
                   FROM ev WHERE EventId IN (81,82)),
             d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
                   WINDOW w AS (PARTITION BY DeviceId, det
                                ORDER BY ts, CASE WHEN EventId=82 THEN 0 ELSE 1 END))
        SELECT m.dev, d.det, epoch_ms(d.ts)/1000.0 AS t_on, epoch_ms(d.nts)/1000.0 AS t_off,
               (epoch_ms(d.nts - d.ts)/1000.0)::FLOAT AS dur
        FROM d JOIN devmap m USING (DeviceId)
        WHERE d.EventId = 82 AND d.nev = 81 AND d.nts IS NOT NULL""")
    # cycles from 1 (begin green) and 7 (green termination) only
    con.execute("""CREATE OR REPLACE TEMP TABLE cyc_raw AS
        WITH g AS (
          SELECT DeviceId, Parameter::SMALLINT AS p, Timestamp AS t, EventId,
                 SUM(CASE WHEN EventId=1 THEN 1 ELSE 0 END) OVER (
                     PARTITION BY DeviceId, Parameter
                     ORDER BY t, CASE EventId WHEN 1 THEN 0 ELSE 1 END
                     ROWS UNBOUNDED PRECEDING) AS cyc
          FROM ev WHERE EventId IN (1,7)
        ), c AS (
          SELECT DeviceId, p, cyc,
                 min(t) FILTER (EventId=1) AS green_start,
                 min(t) FILTER (EventId=7) AS green_end
          FROM g WHERE cyc > 0 AND p BETWEEN 1 AND 16 GROUP BY 1,2,3
        )
        SELECT DeviceId, p, cyc, green_start, green_end AS yellow_start,
               NULL::TIMESTAMP AS red_start,
               LEAD(green_start) OVER (PARTITION BY DeviceId, p ORDER BY green_start)
                   AS next_green
        FROM c""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cyc_all AS
        SELECT m.dev, c.p, c.cyc::INT AS cyc,
               epoch_ms(c.green_start)/1000.0 AS gs,
               epoch_ms(coalesce(c.yellow_start, c.next_green))/1000.0 AS ge,
               epoch_ms(coalesce(c.red_start, c.yellow_start))/1000.0 AS rs,
               epoch_ms(c.next_green)/1000.0 AS ng,
               (epoch_ms(c.yellow_start - c.green_start)/1000.0)::FLOAT AS green_secs
        FROM cyc_raw c JOIN devmap m USING (DeviceId)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE gs_all AS
        WITH iv AS (SELECT DeviceId, p, green_start AS t0,
                           coalesce(yellow_start, next_green) AS t1
                    FROM cyc_raw WHERE coalesce(yellow_start, next_green) IS NOT NULL),
             ch AS (SELECT DeviceId, t0 AS t,  (1::BIGINT << (p-1)) AS d FROM iv
                    UNION ALL
                    SELECT DeviceId, t1 AS t, -(1::BIGINT << (p-1)) AS d FROM iv),
             agg AS (SELECT DeviceId, t, sum(d) AS d FROM ch GROUP BY 1,2),
             run AS (SELECT DeviceId, t,
                            sum(d) OVER (PARTITION BY DeviceId ORDER BY t
                                         ROWS UNBOUNDED PRECEDING) AS mask FROM agg)
        SELECT m.dev, epoch_ms(r.t)/1000.0 AS t0,
               epoch_ms(LEAD(r.t) OVER (PARTITION BY r.DeviceId ORDER BY r.t))/1000.0 AS t1,
               r.mask
        FROM run r JOIN devmap m USING (DeviceId)""")
    con.execute("DELETE FROM gs_all WHERE t1 IS NULL")
    # no 131/150 in the six codes -> no coordination information at all
    con.execute("CREATE OR REPLACE TEMP TABLE coordiv AS "
                "SELECT 0::SMALLINT AS dev, 0.0::DOUBLE AS t0, false AS is_coord LIMIT 0")
    con.execute("""CREATE OR REPLACE TEMP TABLE calls_all AS
        SELECT m.dev, e.Parameter::SMALLINT AS p, e.EventId::SMALLINT AS ev,
               epoch_ms(e.Timestamp)/1000.0 AS t
        FROM ev e JOIN devmap m USING (DeviceId)
        WHERE e.EventId IN (43,44) AND e.Parameter BETWEEN 1 AND 16""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cand AS
        SELECT DISTINCT m.dev, e.Parameter::SMALLINT AS p
        FROM ev e JOIN devmap m USING (DeviceId)
        WHERE e.EventId = 1 AND e.Parameter BETWEEN 1 AND 16""")


# -------------------------------------------------------------------- health
HEALTH_CONST = dict(max_day_gap_s=0.0, unmatched_on_rate=0.0, day_ratio=1.0,
                    n_fault_events=0)


def health6(con, w0: float, w1: float) -> pd.DataFrame:
    """Health flags from the six codes (no 83-88 fault events available)."""
    from health import flag_detectors
    secs = w1 - w0
    days = max(secs / 86400.0, 1e-6)
    h = con.sql(f"""
        WITH o AS (SELECT * FROM onev_all WHERE t_on >= {w0} AND t_on < {w1}),
        a AS (
          SELECT dev, det, count(*) AS n_on, sum(dur) AS occ, max(dur) AS longest_on_s,
                 max(cnt) AS max_on_per_min
          FROM (SELECT dev, det, dur, count(*) OVER (PARTITION BY dev, det,
                       (t_on/60)::BIGINT) AS cnt FROM o) GROUP BY 1,2)
        SELECT d.DeviceId, a.det AS Detector, a.n_on,
               a.n_on / {days} AS on_per_day, a.occ / {secs} AS frac_time_on,
               a.longest_on_s, a.max_on_per_min
        FROM a JOIN devmap d USING (dev)""").df()
    for c, v in HEALTH_CONST.items():
        h[c] = v
    return flag_detectors(h)


# --------------------------------------------------------------------- build
def build(devs: list[str], glob: str, windows: list[dict], tag: str,
          chunk: int = 8, threads: int = 10) -> None:
    con = connect(threads=threads)
    parts, sims, healths, t0 = [], [], [], time.time()
    nch = (len(devs) + chunk - 1) // chunk
    for i in range(0, len(devs), chunk):
        ch = devs[i:i + chunk]
        t1 = time.time()
        load_chunk6(con, ch, glob)
        for w in windows:
            w0 = (w["t0"] - pd.Timestamp("1970-01-01")).total_seconds()
            w1 = w0 + w["secs"]
            f1.apply_window(con, w0, w1)
            base = f1.build_window(con, w["win"], w["secs"])
            if not len(base):
                continue
            extra = f2.build_window(con, w["win"], w["secs"])
            if len(extra):
                base = base.merge(extra, on=["DeviceId", "Detector", "cand_phase", "win"],
                                  how="left")
            parts.append(base)
            s = cdt.build_window(con, w["win"], w["secs"])
            if len(s):
                sims.append(s)
            hh = health6(con, w0, w1)
            hh["win"] = w["win"]
            healths.append(hh)
        log(f"[{tag}] chunk {i//chunk+1}/{nch} {time.time()-t1:.1f}s "
            f"elapsed={time.time()-t0:.0f}s")
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = f1.finalise(df)
    df.to_parquet(SW_FEAT / f"pairs6_{tag}.parquet", index=False)
    sim = pd.concat(sims, ignore_index=True)
    sim["Detector"] = sim.Detector.astype(np.int16)
    sim["other"] = sim.other.astype(np.int16)
    sim["phi"] = sim.phi.astype(np.float32)
    sim.to_parquet(SW_FEAT / f"sim6_{tag}.parquet", index=False)
    hl = pd.concat(healths, ignore_index=True)
    hl["Detector"] = hl.Detector.astype(np.int16)
    hl.to_parquet(SW_FEAT / f"health6_{tag}.parquet", index=False)
    log(f"[{tag}] pairs {df.shape}, sim {sim.shape}, health {hl.shape}, "
        f"{time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["dev", "statewide"])
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.source == "dev":
        devs = sorted(pd.read_csv(DC_WORK / "folds.csv").DeviceId.unique())
        glob, wins, tag = DEV_EVENTS_GLOB, WINDOWS_MIXED, "dev"
    else:
        g = signal_groups()
        devs = sorted(g[g.group != "test"].DeviceId.unique())   # never touch TEST
        glob, wins, tag = SW_EVENTS_GLOB, WINDOWS_SW, "sw"
    if a.limit:
        devs = devs[:a.limit]
    log(f"{tag}: {len(devs)} signals x {len(wins)} windows (6 codes)")
    build(devs, glob, wins, tag, a.chunk, a.threads)


if __name__ == "__main__":
    main()
