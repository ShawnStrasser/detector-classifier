"""Stage 01 step A+B: folds, label tables and the compact event cache.

Usage
-----
    python src/build_cache.py --step all
    python src/build_cache.py --step labels|events|derived|check

Outputs (all under dc_work\\, see src/README.md for the schema):
    folds.csv, labels_dev.parquet, labels_test.parquet
    cache/events/DeviceId=<guid>/d{2,3,4}.parquet   -- filtered raw events
    cache/det_intervals.parquet                     -- detector ON intervals
    cache/phase_cycles.parquet                      -- phase colour cycles
    cache/green_state.parquet                       -- per-signal green bitmask timeline
    cache/coord_state.parquet                       -- coordination pattern intervals
    cache/detector_meta.parquet                     -- per detector channel stats + health
    cache/signal_meta.parquet                       -- per signal stats
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALLOWED_EVENTS, CACHE, DC_WORK, FOLDS_CSV, LABELS_DEV,  # noqa: E402
                    LABELS_TEST, MAX_DETECTOR_CHANNEL, N_FOLDS, RAW, RAW_DAYS,
                    SPLITS, connect)

EV_LIST = ",".join(str(e) for e in ALLOWED_EVENTS)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- labels
def build_labels(con) -> None:
    cfg = con.sql(f"SELECT * FROM read_csv('{(RAW / 'detector-configs.csv').as_posix()}')").df()
    test_dev = con.sql(
        f"SELECT DISTINCT DeviceId FROM read_csv('{(SPLITS / 'test_config.csv').as_posix()}')"
    ).df()["DeviceId"].tolist()
    train_dev = con.sql(
        f"SELECT column1 AS DeviceId FROM read_csv('{(SPLITS / 'device_id_train.csv').as_posix()}',"
        " header=false, skip=1)").df()["DeviceId"].tolist()
    valid_dev = con.sql(
        f"SELECT column1 AS DeviceId FROM read_csv('{(SPLITS / 'device_id_valid.csv').as_posix()}',"
        " header=false, skip=1)").df()["DeviceId"].tolist()

    test_dev, train_dev, valid_dev = set(test_dev), list(dict.fromkeys(train_dev)), list(dict.fromkeys(valid_dev))
    assert not (set(train_dev) & set(valid_dev)), "train/valid overlap"
    assert not (set(train_dev) & test_dev) and not (set(valid_dev) & test_dev), "test leaks into dev"
    log(f"devices: train={len(train_dev)} valid={len(valid_dev)} test={len(test_dev)} "
        f"labeled_in_cfg={cfg.DeviceId.nunique()}")

    # ---- folds.csv: fold 0 == device_id_valid.csv, folds 1..5 == random split of train, seed 0
    rng = np.random.default_rng(0)
    order = np.array(sorted(train_dev))            # deterministic base order
    perm = rng.permutation(len(order))
    fold = np.empty(len(order), dtype=int)
    # even-sized contiguous blocks of the permutation -> folds 1..5
    blocks = np.array_split(perm, N_FOLDS - 1)
    for i, b in enumerate(blocks):
        fold[b] = i + 1
    folds = pd.concat([
        pd.DataFrame({"DeviceId": sorted(valid_dev), "fold": 0}),
        pd.DataFrame({"DeviceId": order, "fold": fold}),
    ], ignore_index=True).sort_values(["fold", "DeviceId"]).reset_index(drop=True)
    folds.to_csv(FOLDS_CSV, index=False)
    log(f"wrote {FOLDS_CSV}  {len(folds)} devices, fold sizes: "
        f"{folds.fold.value_counts().sort_index().to_dict()}")

    dev_ids = set(folds.DeviceId)
    lab = cfg[["DeviceId", "Detector", "Phase", "Function"]].copy()
    lab["Detector"] = lab["Detector"].astype("int32")
    lab["Phase"] = lab["Phase"].astype("int32")
    dup = lab.duplicated(["DeviceId", "Detector"]).sum()
    if dup:
        log(f"WARNING: {dup} duplicate (DeviceId,Detector) label rows -> keeping first")
        lab = lab.drop_duplicates(["DeviceId", "Detector"], keep="first")
    ld = lab[lab.DeviceId.isin(dev_ids)].reset_index(drop=True)
    lt = lab[lab.DeviceId.isin(test_dev)].reset_index(drop=True)
    assert len(ld) + len(lt) == len(lab), "labels not partitioned by dev/test"
    ld.to_parquet(LABELS_DEV, index=False)
    lt.to_parquet(LABELS_TEST, index=False)
    log(f"wrote {LABELS_DEV} ({len(ld)} rows, {ld.DeviceId.nunique()} signals) and "
        f"{LABELS_TEST} ({len(lt)} rows, {lt.DeviceId.nunique()} signals)")
    log("dev function counts: " + str(ld.Function.value_counts().to_dict()))


# --------------------------------------------------------------------- events
def build_events(con, days=(2, 3, 4)) -> None:
    out = (CACHE / "events").as_posix()
    if (CACHE / "events").exists():          # APPEND mode -> must start from empty
        shutil.rmtree(CACHE / "events")
        log("removed existing cache/events")
    con.execute("SET preserve_insertion_order=true")
    for d in days:
        src = RAW_DAYS[d].as_posix()
        t0 = time.time()
        con.execute(f"""
            COPY (
              SELECT DISTINCT
                     DeviceId, Timestamp, EventId::UTINYINT AS EventId, Parameter::UTINYINT AS Parameter
              FROM read_parquet('{src}')
              WHERE EventId IN ({EV_LIST})
                AND NOT (EventId IN (81,82) AND Parameter > {MAX_DETECTOR_CHANNEL})
              ORDER BY DeviceId, Timestamp
            ) TO '{out}'
            (FORMAT parquet, PARTITION_BY (DeviceId), APPEND,
             FILENAME_PATTERN 'd{d}_{{uuid}}', COMPRESSION zstd, ROW_GROUP_SIZE 200000)
        """)
        log(f"day {d} -> cache/events in {time.time() - t0:.0f}s")
    con.execute("SET preserve_insertion_order=false")
    n = con.sql(f"SELECT count(*) c, count(DISTINCT DeviceId) d FROM read_parquet('{out}/**/*.parquet', hive_partitioning=true)").fetchone()
    log(f"cache/events: {n[0]:,} rows, {n[1]} devices")


# -------------------------------------------------------------------- derived
def build_derived(con) -> None:
    ev = (CACHE / "events" / "**" / "*.parquet").as_posix()
    con.execute(f"CREATE OR REPLACE VIEW ev AS SELECT * FROM read_parquet('{ev}', hive_partitioning=true)")

    # ---- detector ON intervals -------------------------------------------------
    t0 = time.time()
    con.execute(f"""
        COPY (
          WITH e AS (
            SELECT DeviceId, Parameter::UTINYINT AS Detector, Timestamp AS ts, EventId
            FROM ev WHERE EventId IN (81,82)
          ), d AS (
            SELECT *,
                   LEAD(ts)      OVER w AS nts,
                   LEAD(EventId) OVER w AS nev
            FROM e
            WINDOW w AS (PARTITION BY DeviceId, Detector
                         ORDER BY ts, CASE WHEN EventId=82 THEN 0 ELSE 1 END)
          )
          SELECT DeviceId, Detector, ts AS t_on, nts AS t_off,
                 epoch_ms(nts - ts)/1000.0 AS dur
          FROM d WHERE EventId = 82 AND nev = 81 AND nts IS NOT NULL
          ORDER BY DeviceId, Detector, ts
        ) TO '{(CACHE / 'det_intervals.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    log(f"det_intervals in {time.time()-t0:.0f}s")

    # ---- phase colour cycles ---------------------------------------------------
    t0 = time.time()
    con.execute(f"""
        COPY (
          WITH g AS (
            SELECT DeviceId, Parameter::UTINYINT AS Phase, Timestamp AS t, EventId,
                   SUM(CASE WHEN EventId=1 THEN 1 ELSE 0 END) OVER (
                       PARTITION BY DeviceId, Parameter
                       ORDER BY t, CASE EventId WHEN 1 THEN 0 WHEN 8 THEN 1 WHEN 10 THEN 2 ELSE 3 END
                       ROWS UNBOUNDED PRECEDING) AS cyc
            FROM ev WHERE EventId IN (1,8,10,11)
          ), c AS (
            SELECT DeviceId, Phase, cyc,
                   min(t) FILTER (EventId=1)  AS green_start,
                   min(t) FILTER (EventId=8)  AS yellow_start,
                   min(t) FILTER (EventId=10) AS red_start,
                   min(t) FILTER (EventId=11) AS redclr_end
            FROM g WHERE cyc > 0 GROUP BY 1,2,3
          )
          SELECT DeviceId, Phase, cyc, green_start,
                 yellow_start, red_start, redclr_end,
                 LEAD(green_start) OVER (PARTITION BY DeviceId, Phase ORDER BY green_start) AS next_green,
                 epoch_ms(coalesce(yellow_start, red_start) - green_start)/1000.0 AS green_secs
          FROM c
          ORDER BY DeviceId, Phase, green_start
        ) TO '{(CACHE / 'phase_cycles.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    log(f"phase_cycles in {time.time()-t0:.0f}s")

    # ---- green bitmask timeline ------------------------------------------------
    # bit (Phase-1) set while that phase is green.  mask = running sum of +/- bit.
    t0 = time.time()
    con.execute(f"""
        COPY (
          WITH iv AS (
            SELECT DeviceId, Phase, green_start AS t0,
                   coalesce(yellow_start, red_start, next_green) AS t1
            FROM read_parquet('{(CACHE / 'phase_cycles.parquet').as_posix()}')
            WHERE Phase BETWEEN 1 AND 16
          ), ch AS (
            SELECT DeviceId, t0 AS t,  (1::BIGINT << (Phase-1)) AS d FROM iv WHERE t1 IS NOT NULL
            UNION ALL
            SELECT DeviceId, t1 AS t, -(1::BIGINT << (Phase-1)) AS d FROM iv WHERE t1 IS NOT NULL
          ), agg AS (
            SELECT DeviceId, t, sum(d) AS d FROM ch GROUP BY 1,2
          ), run AS (
            SELECT DeviceId, t,
                   sum(d) OVER (PARTITION BY DeviceId ORDER BY t ROWS UNBOUNDED PRECEDING) AS mask
            FROM agg
          )
          SELECT DeviceId, t AS t_start,
                 LEAD(t) OVER (PARTITION BY DeviceId ORDER BY t) AS t_end, mask
          FROM run
          ORDER BY DeviceId, t
        ) TO '{(CACHE / 'green_state.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    log(f"green_state in {time.time()-t0:.0f}s")

    # ---- coordination pattern intervals ---------------------------------------
    con.execute(f"""
        COPY (
          WITH p AS (
            SELECT DeviceId, Timestamp AS t, Parameter AS pattern
            FROM ev WHERE EventId = 131
          ), q AS (
            SELECT DeviceId, t, pattern,
                   LEAD(t) OVER (PARTITION BY DeviceId ORDER BY t) AS t_end
            FROM p
          )
          SELECT DeviceId, t AS t_start, t_end, pattern,
                 (pattern BETWEEN 1 AND 253) AS is_coord
          FROM q ORDER BY DeviceId, t
        ) TO '{(CACHE / 'coord_state.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)

    # ---- per-signal meta -------------------------------------------------------
    con.execute(f"""
        COPY (
          WITH span AS (
            SELECT DeviceId, min(Timestamp) t0, max(Timestamp) t1, count(*) n_events
            FROM ev GROUP BY 1
          ), cand AS (
            SELECT DeviceId, Parameter::UTINYINT AS Phase, count(*) AS n_green
            FROM ev WHERE EventId = 1 GROUP BY 1,2
          ), candagg AS (
            SELECT DeviceId, count(*) AS n_cand, list(Phase ORDER BY Phase) AS cand_phases
            FROM cand GROUP BY 1
          ), coordph AS (
            SELECT DeviceId, list(DISTINCT Parameter) AS coord_phases
            FROM ev WHERE EventId = 150 AND Parameter BETWEEN 1 AND 16 GROUP BY 1
          ), coordfrac AS (
            SELECT DeviceId,
                   sum(CASE WHEN is_coord THEN epoch_ms(coalesce(t_end,t_start)-t_start)/1000.0 ELSE 0 END)
                     / nullif(sum(epoch_ms(coalesce(t_end,t_start)-t_start)/1000.0),0) AS coord_frac
            FROM read_parquet('{(CACHE / 'coord_state.parquet').as_posix()}') GROUP BY 1
          ), dets AS (
            SELECT DeviceId, count(DISTINCT Parameter) AS n_det_channels
            FROM ev WHERE EventId = 82 GROUP BY 1
          )
          SELECT s.DeviceId, s.t0, s.t1, s.n_events,
                 epoch_ms(s.t1-s.t0)/1000.0 AS span_secs,
                 c.n_cand, c.cand_phases, cp.coord_phases,
                 coalesce(cf.coord_frac, 0.0) AS coord_frac, d.n_det_channels
          FROM span s LEFT JOIN candagg c USING (DeviceId)
                      LEFT JOIN coordph cp USING (DeviceId)
                      LEFT JOIN coordfrac cf USING (DeviceId)
                      LEFT JOIN dets d USING (DeviceId)
          ORDER BY s.DeviceId
        ) TO '{(CACHE / 'signal_meta.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)

    # ---- per-detector meta + health -------------------------------------------
    t0 = time.time()
    con.execute(f"""
        COPY (
          WITH iv AS (
            SELECT * FROM read_parquet('{(CACHE / 'det_intervals.parquet').as_posix()}')
          ), sm AS (
            SELECT DeviceId, span_secs FROM read_parquet('{(CACHE / 'signal_meta.parquet').as_posix()}')
          ), agg AS (
            SELECT DeviceId, Detector,
                   count(*) AS n_on,
                   sum(dur) AS occ_secs,
                   avg(dur) AS dur_mean,
                   median(dur) AS dur_med,
                   quantile_cont(dur, 0.1) AS dur_q10,
                   quantile_cont(dur, 0.9) AS dur_q90,
                   quantile_cont(dur, 0.99) AS dur_q99,
                   max(dur) AS dur_max,
                   avg(CASE WHEN dur < 0.15 THEN 1 ELSE 0 END) AS frac_dur_lt015,
                   avg(CASE WHEN dur < 0.5  THEN 1 ELSE 0 END) AS frac_dur_lt05,
                   avg(CASE WHEN dur > 5    THEN 1 ELSE 0 END) AS frac_dur_gt5,
                   avg(CASE WHEN dur > 30   THEN 1 ELSE 0 END) AS frac_dur_gt30,
                   count(DISTINCT date_trunc('hour', t_on)) AS n_hours_active,
                   min(t_on) AS first_on, max(t_on) AS last_on
            FROM iv GROUP BY 1,2
          ), flt AS (
            SELECT DeviceId, Parameter::UTINYINT AS Detector,
                   count(*) FILTER (EventId IN (84,85,86,87,88)) AS n_fault_events,
                   count(*) FILTER (EventId = 83) AS n_restore_events
            FROM ev WHERE EventId BETWEEN 83 AND 88 AND Parameter <= {MAX_DETECTOR_CHANNEL}
            GROUP BY 1,2
          )
          SELECT a.*, sm.span_secs,
                 a.n_on / (sm.span_secs/3600.0) AS on_per_hour,
                 a.occ_secs / sm.span_secs AS occ_frac,
                 coalesce(f.n_fault_events,0) AS n_fault_events,
                 coalesce(f.n_restore_events,0) AS n_restore_events,
                 (coalesce(f.n_fault_events,0) > 0
                  OR a.dur_max > 900
                  OR a.frac_dur_lt015 > 0.5
                  OR a.n_on < 20
                  OR a.n_hours_active < 6) AS unhealthy
          FROM agg a JOIN sm USING (DeviceId) LEFT JOIN flt f USING (DeviceId, Detector)
          ORDER BY DeviceId, Detector
        ) TO '{(CACHE / 'detector_meta.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    log(f"meta tables in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------- checks
def check(con) -> None:
    p = lambda s: print(con.sql(s).df().to_string(), flush=True)
    log("--- cache row counts")
    for t in ["det_intervals", "phase_cycles", "green_state", "coord_state",
              "signal_meta", "detector_meta"]:
        n = con.sql(f"SELECT count(*) c FROM read_parquet('{(CACHE / (t + '.parquet')).as_posix()}')").fetchone()[0]
        print(f"  {t:16s} {n:>12,}")
    log("--- candidates per signal")
    p(f"SELECT n_cand, count(*) c FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}') GROUP BY 1 ORDER BY 1")
    log("--- label coverage: labeled detectors that have events")
    p(f"""SELECT count(*) n_lab,
                 count(*) FILTER (dm.Detector IS NOT NULL) n_with_events,
                 count(*) FILTER (dm.Detector IS NULL) n_no_events
          FROM read_parquet('{LABELS_DEV.as_posix()}') l
          LEFT JOIN read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}') dm USING (DeviceId, Detector)""")
    log("--- labeled phase present among candidates?")
    p(f"""WITH cand AS (SELECT DeviceId, unnest(cand_phases) AS Phase
                        FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}'))
          SELECT count(*) n, count(*) FILTER (c.Phase IS NOT NULL) n_in_cand
          FROM read_parquet('{LABELS_DEV.as_posix()}') l
          LEFT JOIN cand c ON c.DeviceId=l.DeviceId AND c.Phase=l.Phase""")
    log("--- active detector channels vs labeled")
    p(f"""SELECT count(*) n_active,
                 count(*) FILTER (l.Detector IS NOT NULL) n_labeled
          FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}') dm
          LEFT JOIN read_parquet('{LABELS_DEV.as_posix()}') l USING (DeviceId, Detector)""")
    log("--- unhealthy detectors")
    p(f"SELECT unhealthy, count(*) c FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}') GROUP BY 1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["all", "labels", "events", "derived", "check"])
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    for d in (CACHE, DC_WORK / "features", DC_WORK / "preds", DC_WORK / "models", DC_WORK / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    con = connect(threads=a.threads)
    if a.step in ("all", "labels"):
        build_labels(con)
    if a.step in ("all", "events"):
        build_events(con)
    if a.step in ("all", "derived"):
        build_derived(con)
    if a.step in ("all", "check"):
        check(con)
    log("done")


if __name__ == "__main__":
    main()
