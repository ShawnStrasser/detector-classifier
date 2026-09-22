"""Detector-health SQL and helpers used only while building the research tables.

The flag *rules* live with the model (`model/health.py`); this module carries the DuckDB
versions of them, the bad-period masking used by the stage-09 experiment, and the reader
for the cached `dc_work/atspm/detector_health.parquet` table.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402
from health import flag_detectors, status_for_user  # noqa: E402,F401

DEFAULT_HEALTH_PATH = str(DC_WORK / "atspm" / "detector_health.parquet")

# The same rules as one DuckDB CASE expression (used by build_health.py so that the
# parquet ships with a flag column identical to what flag_detectors() produces).
HEALTH_FLAG_SQL = """
  CASE
    WHEN coalesce(n_on,0) = 0                     THEN 'failed'
    WHEN coalesce(on_per_day,0) < 20              THEN 'failed'
    WHEN coalesce(frac_time_on,0) >= 0.90         THEN 'failed'
    WHEN coalesce(max_on_per_min,0) >= 600        THEN 'failed'
    WHEN coalesce(on_per_day,0) < 100             THEN 'suspect'
    WHEN coalesce(frac_time_on,0) >= 0.50         THEN 'suspect'
    WHEN coalesce(longest_on_s,0) >= 3600         THEN 'suspect'
    WHEN coalesce(max_day_gap_s,0) >= 21600       THEN 'suspect'
    WHEN coalesce(n_fault_events,0) > 0           THEN 'suspect'
    WHEN coalesce(unmatched_on_rate,0) > 0.25     THEN 'suspect'
    WHEN coalesce(day_ratio,1.0) < 0.10           THEN 'suspect'
    WHEN coalesce(max_on_per_min,0) >= 120        THEN 'suspect'
    ELSE 'healthy'
  END AS health_flag,
  CASE
    WHEN coalesce(n_on,0) = 0                     THEN 'no_events'
    WHEN coalesce(on_per_day,0) < 20              THEN 'near_zero_volume'
    WHEN coalesce(frac_time_on,0) >= 0.90         THEN 'stuck_on'
    WHEN coalesce(max_on_per_min,0) >= 600        THEN 'chatter_storm'
    WHEN coalesce(on_per_day,0) < 100             THEN 'low_volume'
    WHEN coalesce(frac_time_on,0) >= 0.50         THEN 'high_occupancy'
    WHEN coalesce(longest_on_s,0) >= 3600         THEN 'long_stuck_interval'
    WHEN coalesce(max_day_gap_s,0) >= 21600       THEN 'long_daytime_outage'
    WHEN coalesce(n_fault_events,0) > 0           THEN 'controller_fault_events'
    WHEN coalesce(unmatched_on_rate,0) > 0.25     THEN 'broken_on_off_stream'
    WHEN coalesce(day_ratio,1.0) < 0.10           THEN 'day_to_day_instability'
    WHEN coalesce(max_on_per_min,0) >= 120        THEN 'chatter'
    ELSE 'ok'
  END AS health_reason
"""



def load_health(path: str = DEFAULT_HEALTH_PATH) -> pd.DataFrame:
    """Read the per-detector health table."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - build it first.")
    return pd.read_parquet(path)


def apply_to_predictions(preds: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
    """Attach status to a long-format phase prediction file.

    `preds`: DeviceId, Detector, cand_phase, prob.  Rows belonging to failed detectors keep
    their probabilities (for auditing) but get status='failed' and a NULLed top-1, so the
    scoring harness / UI can report "cannot classify" instead of a meaningless phase.
    """
    if "health_flag" not in health.columns:
        health = flag_detectors(health)
    h = health[["DeviceId", "Detector", "health_flag", "health_reason"]]
    out = preds.merge(h, on=["DeviceId", "Detector"], how="left")
    out["health_flag"] = out["health_flag"].fillna("failed")
    out["health_reason"] = out["health_reason"].fillna("no_events")
    out["status"] = [status_for_user(f, r)
                     for f, r in zip(out["health_flag"], out["health_reason"])]
    return out


# ---------------------------------------------------------------------------
# Masking bad *periods* (for feature building from raw events).
# ---------------------------------------------------------------------------
MASK_INTERVALS_SQL = """
-- Intervals during which a (DeviceId, Detector) should be ignored when building features.
-- `raw` must expose DeviceId, Timestamp, EventId, Parameter (deduplicated).
WITH fault AS (   -- 84-88 = fault declared, 83 = detector restored (per channel)
    SELECT DeviceId, Parameter AS Detector, Timestamp AS mask_start,
           coalesce(lead(Timestamp) OVER (PARTITION BY DeviceId, Parameter ORDER BY Timestamp),
                    Timestamp + INTERVAL 1 HOUR) AS mask_end,
           'fault_' || EventId AS reason
    FROM raw WHERE EventId BETWEEN 83 AND 88
    QUALIFY EventId <> 83
),
flash AS (        -- 173 = flash on, 174 = flash off; whole signal is meaningless in flash
    SELECT DeviceId, NULL::SMALLINT AS Detector, Timestamp AS mask_start,
           coalesce(lead(Timestamp) OVER (PARTITION BY DeviceId ORDER BY Timestamp),
                    Timestamp + INTERVAL 1 HOUR) AS mask_end,
           'flash' AS reason
    FROM raw WHERE EventId IN (173, 174)
    QUALIFY EventId = 173
),
stuck AS (        -- a single ON longer than {stuck_on_s} s is not a real vehicle call
    SELECT DeviceId, Detector, mask_start, mask_end, 'stuck_on' AS reason FROM (
        SELECT DeviceId, Parameter AS Detector, Timestamp AS mask_start,
               lead(Timestamp) OVER w AS mask_end,
               EventId, lead(EventId) OVER w AS next_ev
        FROM raw WHERE EventId IN (81, 82)
        WINDOW w AS (PARTITION BY DeviceId, Parameter ORDER BY Timestamp, EventId DESC)
    ) WHERE EventId = 82 AND next_ev = 81
      AND epoch_ms(mask_end - mask_start)/1000.0 >= {stuck_on_s}
)
SELECT * FROM fault UNION ALL SELECT * FROM flash UNION ALL SELECT * FROM stuck
"""


def mask_intervals_sql(stuck_on_s: int = 1800) -> str:
    """DuckDB SQL returning (DeviceId, Detector, mask_start, mask_end, reason).

    Detector IS NULL means the interval masks the whole signal.  Feed it a deduplicated
    `raw` table/view (see note on duplicates in results/02_atspm_health.md).
    """
    return MASK_INTERVALS_SQL.format(stuck_on_s=stuck_on_s)


# Deduplication of raw hi-res events: ~2% of 81/82 rows are exact duplicates and they
# account for 99% of apparent unmatched OFF events.  Always apply this before pairing.
DEDUP_SQL = ("SELECT DISTINCT DeviceId, Timestamp, EventId, Parameter FROM {src} "
             "WHERE EventId NOT IN (81, 82) OR Parameter <= 64")
