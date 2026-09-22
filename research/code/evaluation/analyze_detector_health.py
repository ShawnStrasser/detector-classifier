"""Quantify: how much of the labeled DEV population is suspect/failed, and how much
more error-prone those detectors are for the old baseline (fold 0). DEV ONLY - TEST is never touched."""
import os
from pathlib import Path

import duckdb
import pandas as pd

pd.set_option('display.width', 200)
W = (os.environ.get('DC_WORK') or str(Path.home() / 'dc_work')).replace('\\', '/')
c = duckdb.connect()
c.execute(f"SET memory_limit='6GB'; SET threads=6; SET temp_directory='{W}/tmp';")

c.execute(f"""
CREATE VIEW h AS SELECT * FROM '{W}/atspm/detector_health.parquet';
CREATE VIEW folds AS SELECT * FROM '{W}/folds.csv';
CREATE TABLE test_dev AS SELECT DISTINCT DeviceId FROM '{W}/data/splits/test_config.csv';
CREATE VIEW cfg AS SELECT DISTINCT DeviceId, Detector, Phase, Function FROM '{W}/data/raw/detector-configs.csv';
-- labeled DEV detectors only
CREATE TABLE lab AS
  SELECT cfg.*, f.fold FROM cfg JOIN folds f USING (DeviceId)
  WHERE cfg.DeviceId NOT IN (SELECT DeviceId FROM test_dev);
CREATE VIEW preds AS SELECT * FROM '{W}/preds/old_baseline_fold0_phase.parquet';
""")
assert c.sql("SELECT count(*) FROM lab JOIN test_dev USING(DeviceId)").fetchone()[0] == 0

print('=== labeled DEV detectors ===')
print(c.sql("SELECT count(*) n_det, count(DISTINCT DeviceId) n_sig FROM lab").df().to_string(index=False))

print('\n=== 1. health mix of labeled DEV detectors ===')
print(c.sql("""
SELECT coalesce(h.health_flag,'no_row') AS flag, count(*) n,
       round(100.0*count(*)/sum(count(*)) OVER (),1) pct
FROM lab l LEFT JOIN h USING (DeviceId, Detector) GROUP BY 1 ORDER BY n DESC
""").df().to_string(index=False))

print('\n=== 2. configured detectors with NO events (DEV) ===')
print(c.sql("""
SELECT count(*) FILTER (h.n_on = 0 OR h.DeviceId IS NULL) AS no_events_dev,
       count(*) FILTER ((h.n_on = 0 OR h.DeviceId IS NULL) AND l.Detector <= 64) AS no_events_dev_ch_le64,
       count(*) AS labeled_dev
FROM lab l LEFT JOIN h USING (DeviceId, Detector)
""").df().to_string(index=False))
print(c.sql(f"""
SELECT count(*) AS no_events_ALL_labeled_signals FROM cfg
LEFT JOIN h USING (DeviceId, Detector) WHERE h.DeviceId IS NULL OR h.n_on = 0
""").df().to_string(index=False))

print('\n=== 3. old baseline fold-0 accuracy by health flag ===')
c.execute("""
CREATE TABLE top1 AS
SELECT DeviceId, Detector, arg_max(cand_phase, prob) AS pred_phase, max(prob) AS top_prob
FROM preds GROUP BY 1,2;
CREATE TABLE scored AS
SELECT l.DeviceId, l.Detector, l.Phase AS true_phase, t.pred_phase, t.top_prob,
       (t.pred_phase = l.Phase)::int AS correct,
       coalesce(h.health_flag,'no_row') AS flag, h.*
FROM lab l
LEFT JOIN top1 t USING (DeviceId, Detector)
LEFT JOIN h USING (DeviceId, Detector)
WHERE l.fold = 0;
""")
print(c.sql("""
SELECT flag, count(*) n, sum(pred_phase IS NULL)::int no_pred,
       round(100.0*avg(coalesce(correct,0)),2) acc_pct,
       round(100.0*(1-avg(coalesce(correct,0))),2) err_pct
FROM scored GROUP BY 1 ORDER BY n DESC
""").df().to_string(index=False))
print(c.sql("""
SELECT CASE WHEN flag='healthy' THEN 'healthy' ELSE 'suspect+failed+norow' END grp,
       count(*) n, round(100.0*avg(coalesce(correct,0)),2) acc_pct
FROM scored GROUP BY 1
""").df().to_string(index=False))

print('\n=== 4. fold-0 error rate per individual criterion ===')
crit = {
 'no_events'          : "n_on = 0 OR n_on IS NULL",
 'stuck_on_.9'        : "frac_time_on >= 0.90",
 'longest_on_>=4h'    : "longest_on_s >= 14400",
 'longest_on_>=1h'    : "longest_on_s >= 3600",
 'daygap_>=6h'        : "max_day_gap_s >= 21600",
 'daygap_>=3h'        : "max_day_gap_s >= 10800",
 'on_per_day_<20'     : "on_per_day < 20",
 'on_per_day_<100'    : "on_per_day < 100",
 'unmatched_on_>.25'  : "unmatched_on_rate > 0.25",
 'unmatched_on_>.05'  : "unmatched_on_rate > 0.05",
 'fault_evt_83_88'    : "n_fault_events > 0",
 'day_ratio_<.25'     : "day_ratio < 0.25",
 'max_on_per_min>=120': "max_on_per_min >= 120",
 'chatter_>.5'        : "chatter_rate > 0.5",
}
rows = []
for name, expr in crit.items():
    r = c.sql(f"""SELECT count(*) FILTER ({expr}) n_hit,
        round(100.0*avg(coalesce(correct,0)) FILTER ({expr}),2) acc_hit,
        round(100.0*avg(coalesce(correct,0)) FILTER (NOT ({expr}) OR ({expr}) IS NULL),2) acc_miss
        FROM scored""").fetchone()
    rows.append((name, r[0], r[1], r[2]))
print(pd.DataFrame(rows, columns=['criterion', 'n_fold0', 'acc_if_hit', 'acc_if_not']).to_string(index=False))

print('\n=== 5. fold-0 accuracy: how much error sits in bad detectors ===')
print(c.sql("""
SELECT flag, count(*) n, sum(1-coalesce(correct,0))::int n_err,
       round(100.0*sum(1-coalesce(correct,0))/sum(sum(1-coalesce(correct,0))) OVER (),1) share_of_all_errors
FROM scored GROUP BY 1 ORDER BY n_err DESC
""").df().to_string(index=False))

print('\n=== 6. DEV-wide health mix by function label ===')
print(c.sql("""
SELECT l.Function, coalesce(h.health_flag,'no_row') flag, count(*) n
FROM lab l LEFT JOIN h USING (DeviceId, Detector) GROUP BY 1,2 ORDER BY 1,3 DESC
""").df().to_string(index=False))
