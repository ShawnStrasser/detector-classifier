"""Per-detector health table from raw hi-res events (label-free).

Input : dc_work/atspm/events_all.parquet  (EventId 81,82,83-88; Parameter<=64)
Output: dc_work/atspm/detector_health.parquet
"""
import duckdb, time, sys, os

sys.path.insert(0, r"S:\Data_Analysis\Python\detector-classifier\src")
from health import HEALTH_FLAG_SQL  # single source of truth for the flag rules

SKIP_PASSES = '--assemble-only' in sys.argv
WORK = 'C:/Users/hwyr67g/dc_work'
# atspm's loader does SELECT DISTINCT -> exact duplicate events are dropped. We do the same.
EV = f"(SELECT DISTINCT DeviceId, Timestamp, EventId, Parameter FROM read_parquet('{WORK}/atspm/events_all.parquet'))"
OUT = f'{WORK}/atspm'

con = duckdb.connect()
con.execute("SET memory_limit='6GB'; SET threads=6; SET temp_directory='C:/Users/hwyr67g/dc_work/tmp'; SET preserve_insertion_order=false;")


def step(name, sql):
    if SKIP_PASSES and name not in ('device_span', 'assemble'):
        print(f'{name}: skipped')
        return
    t0 = time.time()
    con.execute(sql)
    print(f'{name}: {time.time()-t0:.1f}s', flush=True)


# ---- device-level observed span (denominator for fraction-of-time-on) ----
step('device_span', f"""
CREATE OR REPLACE TABLE device_span AS
SELECT DeviceId,
       min(Timestamp) AS t_min, max(Timestamp) AS t_max,
       epoch_ms(max(Timestamp) - min(Timestamp))/1000.0 AS span_s
FROM {EV} GROUP BY DeviceId;
""")

# ---- Pass A: ON/OFF pairing ----
# order ts, EventId DESC => 82 (ON) before 81 (OFF) at identical timestamps,
# so a zero-duration actuation still pairs.
step('pairing', f"""
COPY (
WITH ev AS (
    SELECT DeviceId, Parameter AS Detector, Timestamp AS ts, EventId
    FROM {EV} WHERE EventId IN (81,82)
),
p AS (
    SELECT DeviceId, Detector, ts, EventId,
           lead(ts)      OVER w AS next_ts,
           lead(EventId) OVER w AS next_ev,
           lag(EventId)  OVER w AS prev_ev
    FROM ev
    WINDOW w AS (PARTITION BY DeviceId, Detector ORDER BY ts, EventId DESC)
),
d AS (
    SELECT *, CASE WHEN EventId=82 AND next_ev=81
                   THEN epoch_ms(next_ts - ts)/1000.0 END AS dur_s
    FROM p
)
SELECT DeviceId, Detector,
    count(*) FILTER (EventId=82)                                   AS n_on,
    count(*) FILTER (EventId=81)                                   AS n_off,
    count(*) FILTER (EventId=82 AND next_ev=82)                    AS n_on_unmatched,
    count(*) FILTER (EventId=81 AND (prev_ev IS NULL OR prev_ev=81)) AS n_off_unmatched,
    count(*) FILTER (dur_s IS NOT NULL)                            AS n_paired,
    coalesce(sum(dur_s),0)                                         AS sum_on_s,
    coalesce(max(dur_s),0)                                         AS max_on_s,
    coalesce(median(dur_s),0)                                      AS med_on_s,
    count(*) FILTER (dur_s < 0.1)                                  AS n_short_on,
    min(ts) AS first_ts, max(ts) AS last_ts
FROM d GROUP BY DeviceId, Detector
) TO '{OUT}/_pairs.parquet' (FORMAT PARQUET);
""")

# ---- Pass B/C: per-minute -> per-day actuation counts ----
step('per_day', f"""
COPY (
WITH per_min AS (
    SELECT DeviceId, Parameter AS Detector, date_trunc('minute', Timestamp) AS m, count(*) AS c
    FROM {EV} WHERE EventId=82 GROUP BY 1,2,3
),
per_day AS (
    SELECT DeviceId, Detector, m::DATE AS d, sum(c) AS c_day, max(c) AS max_min
    FROM per_min GROUP BY 1,2,3
)
SELECT DeviceId, Detector,
    max(max_min)                       AS max_on_per_min,
    sum(c_day) FILTER (d='2024-12-02') AS n_on_d1,
    sum(c_day) FILTER (d='2024-12-03') AS n_on_d2,
    sum(c_day) FILTER (d='2024-12-04') AS n_on_d3,
    min(c_day) AS min_day, max(c_day) AS max_day, count(*) AS n_days
FROM per_day GROUP BY 1,2
) TO '{OUT}/_days.parquet' (FORMAT PARQUET);
""")

# ---- Pass D: longest daytime (06:00-20:00) gap with zero actuations ----
step('day_gap', f"""
COPY (
WITH acts AS (
    SELECT DeviceId, Parameter AS Detector, Timestamp AS ts
    FROM {EV} WHERE EventId=82 AND hour(Timestamp) BETWEEN 6 AND 19
),
dets AS (SELECT DISTINCT DeviceId, Detector FROM acts),
days AS (SELECT UNNEST(['2024-12-02','2024-12-03','2024-12-04']) AS d),
sent AS (
    SELECT DeviceId, Detector, (d || ' 06:00:00')::TIMESTAMP AS ts FROM dets, days
    UNION ALL
    SELECT DeviceId, Detector, (d || ' 20:00:00')::TIMESTAMP AS ts FROM dets, days
),
allts AS (SELECT * FROM acts UNION ALL SELECT * FROM sent),
g AS (
    SELECT DeviceId, Detector, ts,
           lag(ts) OVER (PARTITION BY DeviceId, Detector ORDER BY ts) AS prev_ts
    FROM allts
)
SELECT DeviceId, Detector,
       max(CASE WHEN prev_ts::DATE = ts::DATE THEN epoch_ms(ts - prev_ts)/1000.0 END) AS max_day_gap_s
FROM g GROUP BY 1,2
) TO '{OUT}/_gaps.parquet' (FORMAT PARQUET);
""")

# ---- Pass E: detector fault events 83-88 ----
step('faults', f"""
COPY (
SELECT DeviceId, Parameter AS Detector,
    count(*)                    AS n_fault_events,
    count(*) FILTER (EventId=83) AS n_fault_restore,
    count(*) FILTER (EventId=84) AS n_fault_84_maxpresence,
    count(*) FILTER (EventId=85) AS n_fault_85_erratic,
    count(*) FILTER (EventId=86) AS n_fault_86_lowcount,
    count(*) FILTER (EventId=87) AS n_fault_87_open,
    count(*) FILTER (EventId=88) AS n_fault_88_shorted
FROM {EV} WHERE EventId BETWEEN 83 AND 88 GROUP BY 1,2
) TO '{OUT}/_faults.parquet' (FORMAT PARQUET);
""")

# ---- assemble + flag ----
step('assemble', f"""
CREATE OR REPLACE TABLE health AS
WITH cfg AS (
    SELECT DISTINCT DeviceId, Detector FROM '{WORK}/data/raw/detector-configs.csv' WHERE Detector <= 64
),
keys AS (
    SELECT DeviceId, Detector FROM '{OUT}/_pairs.parquet'
    UNION
    SELECT DeviceId, Detector FROM cfg
),
j AS (
SELECT k.DeviceId, k.Detector,
    coalesce(p.n_on,0) AS n_on, coalesce(p.n_off,0) AS n_off,
    coalesce(p.n_on_unmatched,0) AS n_on_unmatched,
    coalesce(p.n_off_unmatched,0) AS n_off_unmatched,
    coalesce(p.n_paired,0) AS n_paired,
    coalesce(p.sum_on_s,0) AS sum_on_s,
    coalesce(p.max_on_s,0) AS longest_on_s,
    coalesce(p.med_on_s,0) AS median_on_s,
    coalesce(p.n_short_on,0) AS n_short_on,
    coalesce(dd.max_on_per_min,0) AS max_on_per_min,
    coalesce(dd.n_on_d1,0) AS n_on_d1, coalesce(dd.n_on_d2,0) AS n_on_d2, coalesce(dd.n_on_d3,0) AS n_on_d3,
    coalesce(dd.min_day,0) AS min_day_on, coalesce(dd.max_day,0) AS max_day_on,
    g.max_day_gap_s,
    coalesce(f.n_fault_events,0) AS n_fault_events,
    coalesce(f.n_fault_84_maxpresence,0) AS n_fault_84_maxpresence,
    coalesce(f.n_fault_85_erratic,0) AS n_fault_85_erratic,
    coalesce(f.n_fault_86_lowcount,0) AS n_fault_86_lowcount,
    coalesce(f.n_fault_87_open,0) AS n_fault_87_open,
    coalesce(f.n_fault_88_shorted,0) AS n_fault_88_shorted,
    coalesce(s.span_s, 259200) AS span_s,
    (p.DeviceId IS NOT NULL) AS has_events,
    (c.DeviceId IS NOT NULL) AS in_config
FROM keys k
LEFT JOIN '{OUT}/_pairs.parquet'  p USING (DeviceId, Detector)
LEFT JOIN '{OUT}/_days.parquet'  dd USING (DeviceId, Detector)
LEFT JOIN '{OUT}/_gaps.parquet'   g USING (DeviceId, Detector)
LEFT JOIN '{OUT}/_faults.parquet' f USING (DeviceId, Detector)
LEFT JOIN device_span s USING (DeviceId)
LEFT JOIN cfg c USING (DeviceId, Detector)
),
m AS (
SELECT *,
    CASE WHEN n_on>0 THEN n_on_unmatched::DOUBLE/n_on ELSE NULL END AS unmatched_on_rate,
    CASE WHEN n_off>0 THEN n_off_unmatched::DOUBLE/n_off ELSE NULL END AS unmatched_off_rate,
    least(sum_on_s/nullif(span_s,0), 1.0) AS frac_time_on,
    CASE WHEN n_paired>0 THEN n_short_on::DOUBLE/n_paired ELSE NULL END AS chatter_rate,
    n_on/nullif(span_s,0)*86400 AS on_per_day,
    CASE WHEN max_day_on>0 THEN min_day_on::DOUBLE/max_day_on ELSE NULL END AS day_ratio
FROM j
)
SELECT *,
""" + HEALTH_FLAG_SQL + """
FROM m;
""")

con.execute(f"COPY health TO '{OUT}/detector_health.parquet' (FORMAT PARQUET)")
print(con.sql("SELECT health_flag, count(*) FROM health GROUP BY 1 ORDER BY 2 DESC").df().to_string())
print(con.sql("SELECT count(*) AS rows, count(DISTINCT DeviceId) AS devs FROM health").df().to_string())
