"""Task 4 - caches in which OVERLAPS are extra candidates.

An overlap is modelled as just another candidate whose green intervals come from the
overlap events, so the existing phase-anonymous feature code can score it unchanged:

    event 61 (overlap begin green)          -> 1  (begin green)
    event 63 (overlap begin yellow)         -> 8  (begin yellow)
    event 64 (overlap begin red clearance)  -> 10 (begin red clearance)
    event 65 (overlap off)                  -> 11 (end red clearance)
    Parameter  o  (an ODOT overlap NUMBER)  -> pseudo-phase 16 + o

Overlaps have no call events (43/44), so every call feature is legitimately empty for an
overlap candidate.  The model never sees the number; it only sees the extra flag
`cand_is_overlap`, which is allowed (it is a candidate *type*, not an identity).

    python src/official/overlap_cache.py --source dec  --root <dir>
    python src/official/overlap_cache.py --source stg  --root <dir>
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ALLOWED_EVENTS, DC_WORK, MAX_DETECTOR_CHANNEL, connect  # noqa: E402

EV_LIST = ",".join(str(e) for e in ALLOWED_EVENTS)
OVL_MAP = {61: 1, 63: 8, 64: 10, 65: 11}
OVL_OFFSET = 16          # pseudo-phase = OVL_OFFSET + overlap number
MAX_OVL = 16


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def is_overlap(cand: int) -> bool:
    return cand > OVL_OFFSET


def overlap_number(cand: int) -> int:
    return cand - OVL_OFFSET


# --------------------------------------------------------------------- events
def build_events(con, srcs: list[str], ovl_srcs: list[str], devs: list[str],
                 out_dir: Path) -> None:
    out = (out_dir / "events").as_posix()
    if (out_dir / "events").exists():
        shutil.rmtree(out_dir / "events")
    con.execute("SET preserve_insertion_order=true")
    con.execute("CREATE OR REPLACE TEMP TABLE keep AS SELECT unnest(?::VARCHAR[]) AS DeviceId",
                [devs])
    case = " ".join(f"WHEN {k} THEN {v}" for k, v in OVL_MAP.items())
    for i, src in enumerate(srcs + ovl_srcs):
        ovl = i >= len(srcs)
        if ovl:
            sel = (f"SELECT DISTINCT DeviceId, Timestamp, "
                   f"(CASE EventId {case} END)::UTINYINT AS EventId, "
                   f"({OVL_OFFSET} + Parameter)::UTINYINT AS Parameter "
                   f"FROM read_parquet('{src}') "
                   f"WHERE EventId IN ({','.join(str(k) for k in OVL_MAP)}) "
                   f"AND Parameter BETWEEN 1 AND {MAX_OVL} "
                   f"AND DeviceId IN (SELECT DeviceId FROM keep)")
        else:
            sel = (f"SELECT DISTINCT DeviceId, Timestamp, EventId::UTINYINT AS EventId, "
                   f"Parameter::UTINYINT AS Parameter FROM read_parquet('{src}') "
                   f"WHERE EventId IN ({EV_LIST}) "
                   f"AND NOT (EventId IN (81,82) AND Parameter > {MAX_DETECTOR_CHANNEL}) "
                   f"AND DeviceId IN (SELECT DeviceId FROM keep)")
        t0 = time.time()
        con.execute(f"""COPY ({sel} ORDER BY DeviceId, Timestamp) TO '{out}'
            (FORMAT parquet, PARTITION_BY (DeviceId), APPEND,
             FILENAME_PATTERN 'p{i}_{{uuid}}', COMPRESSION zstd, ROW_GROUP_SIZE 200000)""")
        log(f"  part {i} ({'overlap' if ovl else 'base'}) {time.time()-t0:.0f}s")
    con.execute("SET preserve_insertion_order=false")
    n = con.sql(f"SELECT count(*) c, count(DISTINCT DeviceId) d FROM "
                f"read_parquet('{out}/**/*.parquet', hive_partitioning=true)").fetchone()
    log(f"{out_dir.name} cache/events: {n[0]:,} rows, {n[1]} devices")


# -------------------------------------------------------------------- derived
def build_derived(con, cache: Path) -> None:
    """build_cache.build_derived with the phase range widened to 1..32 (pseudo-phases)."""
    ev = (cache / "events" / "**" / "*.parquet").as_posix()
    con.execute(f"CREATE OR REPLACE VIEW ev AS SELECT * FROM read_parquet('{ev}', "
                "hive_partitioning=true)")
    P = lambda n: (cache / f"{n}.parquet").as_posix()  # noqa: E731
    con.execute(f"""COPY (
      WITH e AS (SELECT DeviceId, Parameter::UTINYINT AS Detector, Timestamp AS ts, EventId
                 FROM ev WHERE EventId IN (81,82)),
           d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
                 WINDOW w AS (PARTITION BY DeviceId, Detector
                              ORDER BY ts, CASE WHEN EventId=82 THEN 0 ELSE 1 END))
      SELECT DeviceId, Detector, ts AS t_on, nts AS t_off, epoch_ms(nts-ts)/1000.0 AS dur
      FROM d WHERE EventId=82 AND nev=81 AND nts IS NOT NULL
      ORDER BY DeviceId, Detector, ts) TO '{P('det_intervals')}'
      (FORMAT parquet, COMPRESSION zstd)""")
    log("  det_intervals")
    con.execute(f"""COPY (
      WITH g AS (SELECT DeviceId, Parameter::UTINYINT AS Phase, Timestamp AS t, EventId,
                        SUM(CASE WHEN EventId=1 THEN 1 ELSE 0 END) OVER (
                          PARTITION BY DeviceId, Parameter
                          ORDER BY t, CASE EventId WHEN 1 THEN 0 WHEN 8 THEN 1
                                                   WHEN 10 THEN 2 ELSE 3 END
                          ROWS UNBOUNDED PRECEDING) AS cyc
                 FROM ev WHERE EventId IN (1,8,10,11)),
           c AS (SELECT DeviceId, Phase, cyc,
                        min(t) FILTER (EventId=1)  AS green_start,
                        min(t) FILTER (EventId=8)  AS yellow_start,
                        min(t) FILTER (EventId=10) AS red_start,
                        min(t) FILTER (EventId=11) AS redclr_end
                 FROM g WHERE cyc>0 GROUP BY 1,2,3)
      SELECT DeviceId, Phase, cyc, green_start, yellow_start, red_start, redclr_end,
             LEAD(green_start) OVER (PARTITION BY DeviceId, Phase ORDER BY green_start)
               AS next_green,
             epoch_ms(coalesce(yellow_start, red_start) - green_start)/1000.0 AS green_secs
      FROM c ORDER BY DeviceId, Phase, green_start) TO '{P('phase_cycles')}'
      (FORMAT parquet, COMPRESSION zstd)""")
    log("  phase_cycles")
    con.execute(f"""COPY (
      WITH iv AS (SELECT DeviceId, Phase, green_start AS t0,
                         coalesce(yellow_start, red_start, next_green) AS t1
                  FROM read_parquet('{P('phase_cycles')}') WHERE Phase BETWEEN 1 AND 32),
           ch AS (SELECT DeviceId, t0 AS t,  (1::BIGINT << (Phase-1)) AS d FROM iv WHERE t1 IS NOT NULL
                  UNION ALL
                  SELECT DeviceId, t1 AS t, -(1::BIGINT << (Phase-1)) AS d FROM iv WHERE t1 IS NOT NULL),
           agg AS (SELECT DeviceId, t, sum(d) AS d FROM ch GROUP BY 1,2),
           run AS (SELECT DeviceId, t, sum(d) OVER (PARTITION BY DeviceId ORDER BY t
                                                    ROWS UNBOUNDED PRECEDING) AS mask FROM agg)
      SELECT DeviceId, t AS t_start,
             LEAD(t) OVER (PARTITION BY DeviceId ORDER BY t) AS t_end, mask
      FROM run ORDER BY DeviceId, t) TO '{P('green_state')}'
      (FORMAT parquet, COMPRESSION zstd)""")
    log("  green_state")
    con.execute(f"""COPY (
      WITH p AS (SELECT DeviceId, Timestamp AS t, Parameter AS pattern FROM ev WHERE EventId=131),
           q AS (SELECT DeviceId, t, pattern,
                        LEAD(t) OVER (PARTITION BY DeviceId ORDER BY t) AS t_end FROM p)
      SELECT DeviceId, t AS t_start, t_end, pattern, (pattern BETWEEN 1 AND 253) AS is_coord
      FROM q ORDER BY DeviceId, t) TO '{P('coord_state')}' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"""COPY (
      WITH span AS (SELECT DeviceId, min(Timestamp) t0, max(Timestamp) t1, count(*) n_events
                    FROM ev GROUP BY 1),
           cand AS (SELECT DeviceId, Parameter::UTINYINT AS Phase, count(*) AS n_green
                    FROM ev WHERE EventId=1 GROUP BY 1,2),
           candagg AS (SELECT DeviceId, count(*) AS n_cand,
                              list(Phase ORDER BY Phase) AS cand_phases FROM cand GROUP BY 1),
           coordph AS (SELECT DeviceId, list(DISTINCT Parameter) AS coord_phases
                       FROM ev WHERE EventId=150 AND Parameter BETWEEN 1 AND 16 GROUP BY 1),
           coordfrac AS (SELECT DeviceId,
                  sum(CASE WHEN is_coord THEN epoch_ms(coalesce(t_end,t_start)-t_start)/1000.0
                           ELSE 0 END)
                  / nullif(sum(epoch_ms(coalesce(t_end,t_start)-t_start)/1000.0),0) AS coord_frac
                  FROM read_parquet('{P('coord_state')}') GROUP BY 1),
           dets AS (SELECT DeviceId, count(DISTINCT Parameter) AS n_det_channels
                    FROM ev WHERE EventId=82 GROUP BY 1)
      SELECT s.DeviceId, s.t0, s.t1, s.n_events, epoch_ms(s.t1-s.t0)/1000.0 AS span_secs,
             c.n_cand, c.cand_phases, cp.coord_phases,
             coalesce(cf.coord_frac,0.0) AS coord_frac, d.n_det_channels
      FROM span s LEFT JOIN candagg c USING (DeviceId) LEFT JOIN coordph cp USING (DeviceId)
                  LEFT JOIN coordfrac cf USING (DeviceId) LEFT JOIN dets d USING (DeviceId)
      ORDER BY s.DeviceId) TO '{P('signal_meta')}' (FORMAT parquet, COMPRESSION zstd)""")
    log("  signal_meta")
    con.execute(f"""COPY (
      WITH iv AS (SELECT * FROM read_parquet('{P('det_intervals')}')),
           sm AS (SELECT DeviceId, span_secs FROM read_parquet('{P('signal_meta')}')),
           agg AS (SELECT DeviceId, Detector, count(*) AS n_on, sum(dur) AS occ_secs,
                          avg(dur) AS dur_mean, median(dur) AS dur_med,
                          quantile_cont(dur,0.1) AS dur_q10, quantile_cont(dur,0.9) AS dur_q90,
                          quantile_cont(dur,0.99) AS dur_q99, max(dur) AS dur_max,
                          avg(CASE WHEN dur<0.15 THEN 1 ELSE 0 END) AS frac_dur_lt015,
                          avg(CASE WHEN dur<0.5 THEN 1 ELSE 0 END) AS frac_dur_lt05,
                          avg(CASE WHEN dur>5 THEN 1 ELSE 0 END) AS frac_dur_gt5,
                          avg(CASE WHEN dur>30 THEN 1 ELSE 0 END) AS frac_dur_gt30,
                          count(DISTINCT date_trunc('hour', t_on)) AS n_hours_active,
                          min(t_on) AS first_on, max(t_on) AS last_on
                   FROM iv GROUP BY 1,2),
           flt AS (SELECT DeviceId, Parameter::UTINYINT AS Detector,
                          count(*) FILTER (EventId IN (84,85,86,87,88)) AS n_fault_events,
                          count(*) FILTER (EventId=83) AS n_restore_events
                   FROM ev WHERE EventId BETWEEN 83 AND 88
                     AND Parameter <= {MAX_DETECTOR_CHANNEL} GROUP BY 1,2)
      SELECT a.*, sm.span_secs, a.n_on/(sm.span_secs/3600.0) AS on_per_hour,
             a.occ_secs/sm.span_secs AS occ_frac,
             coalesce(f.n_fault_events,0) AS n_fault_events,
             coalesce(f.n_restore_events,0) AS n_restore_events,
             (coalesce(f.n_fault_events,0)>0 OR a.dur_max>900 OR a.frac_dur_lt015>0.5
              OR a.n_on<20 OR a.n_hours_active<6) AS unhealthy
      FROM agg a JOIN sm USING (DeviceId) LEFT JOIN flt f USING (DeviceId, Detector)
      ORDER BY DeviceId, Detector) TO '{P('detector_meta')}' (FORMAT parquet, COMPRESSION zstd)""")
    log("  detector_meta")


# ------------------------------------------------- patched features.load_chunk
def patch_load_chunk(cache: Path) -> None:
    """features.load_chunk, widened to pseudo-phases 1..32 and pointed at `cache`."""
    import features as f1
    import features_v2 as f2

    def load_chunk(con, devs):
        dev_list = ",".join("'" + d + "'" for d in devs)
        con.execute("CREATE OR REPLACE TEMP TABLE devmap AS "
                    "SELECT DeviceId, row_number() OVER (ORDER BY DeviceId)::SMALLINT AS dev "
                    f"FROM (SELECT unnest([{dev_list}]) AS DeviceId)")
        IN = f"DeviceId IN ({dev_list})"
        p = lambda n: (cache / f"{n}.parquet").as_posix()  # noqa: E731
        con.execute(f"""CREATE OR REPLACE TEMP TABLE onev_all AS
            SELECT m.dev, i.Detector::SMALLINT AS det,
                   epoch_ms(i.t_on)/1000.0 AS t_on, epoch_ms(i.t_off)/1000.0 AS t_off,
                   i.dur::FLOAT AS dur
            FROM read_parquet('{p('det_intervals')}') i
            JOIN devmap m USING (DeviceId) WHERE i.{IN}""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc_all AS
            SELECT m.dev, c.Phase::SMALLINT AS p, c.cyc::INT AS cyc,
                   epoch_ms(c.green_start)/1000.0 AS gs,
                   epoch_ms(coalesce(c.yellow_start, c.red_start, c.next_green))/1000.0 AS ge,
                   epoch_ms(coalesce(c.red_start, c.yellow_start))/1000.0 AS rs,
                   epoch_ms(c.next_green)/1000.0 AS ng, c.green_secs::FLOAT AS green_secs
            FROM read_parquet('{p('phase_cycles')}') c
            JOIN devmap m USING (DeviceId) WHERE c.Phase BETWEEN 1 AND 32 AND c.{IN}""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE gs_all AS
            SELECT m.dev, epoch_ms(g.t_start)/1000.0 AS t0,
                   epoch_ms(g.t_end)/1000.0 AS t1, g.mask
            FROM read_parquet('{p('green_state')}') g
            JOIN devmap m USING (DeviceId) WHERE g.t_end IS NOT NULL AND g.{IN}""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE coordiv AS
            SELECT m.dev, epoch_ms(c.t_start)/1000.0 AS t0, c.is_coord
            FROM read_parquet('{p('coord_state')}') c
            JOIN devmap m USING (DeviceId) WHERE c.{IN}""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE calls_all AS
            SELECT m.dev, e.Parameter::SMALLINT AS p, e.EventId::SMALLINT AS ev,
                   epoch_ms(e.Timestamp)/1000.0 AS t
            FROM read_parquet('{(cache/'events'/'**'/'*.parquet').as_posix()}',
                              hive_partitioning=true) e
            JOIN devmap m USING (DeviceId)
            WHERE e.EventId IN (43,44) AND e.Parameter BETWEEN 1 AND 16 AND e.{IN}""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cand AS
            SELECT m.dev, unnest(s.cand_phases)::SMALLINT AS p
            FROM read_parquet('{p('signal_meta')}') s
            JOIN devmap m USING (DeviceId) WHERE s.{IN}""")

    f1.load_chunk = load_chunk
    f2.load_chunk = load_chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["dec", "stg"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--signals", default="")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--step", default="all", choices=["all", "events", "derived"])
    a = ap.parse_args()
    root = Path(a.root)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    con = connect(threads=a.threads)
    con.execute(f"SET temp_directory='{(root/'tmp').as_posix()}'")
    if a.source == "dec":
        srcs = [(DC_WORK / "data" / "raw" / f"Train_Dec_{d}_2024.parquet").as_posix()
                for d in (2, 3, 4)]
        ovl = list(srcs)
    else:
        srcs = [(DC_WORK / "data" / "staging" / f"date={d}" / "part_*.parquet").as_posix()
                for d in ("2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21")]
        ovl = [(DC_WORK / "data" / "staging_other" / f"date={d}" / "part_*.parquet").as_posix()
               for d in ("2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21")]
    devs = [s.strip() for s in Path(a.signals).read_text().split()
            if s.strip()] if a.signals else None
    if devs is None:
        raise SystemExit("--signals FILE (one DeviceId per line) is required")
    log(f"{len(devs)} signals -> {root}")
    if a.step in ("all", "events"):
        build_events(con, srcs, ovl, devs, cache)
    if a.step in ("all", "derived"):
        build_derived(con, cache)
    log("done")


if __name__ == "__main__":
    main()
