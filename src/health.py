"""Detector health: flagging bad channels and masking bad periods.

Stage 02 (`results/02_atspm_health.md`) found that, for phase classification, the only
detector pathology that actually costs accuracy is *absence of actuations*.  Everything
else (stuck-on, chatter, unmatched ON/OFF, controller fault events, day-to-day drift) is
either very rare or does not measurably hurt the old baseline once volume is adequate.
So the rules below are deliberately asymmetric: `failed` is a small, high-precision set
that must NOT be classified, `suspect` is a wider advisory set that should still be
classified but reported with lower confidence / sent to manual review.

Typical use in a model stage:

    from health import load_health, flag_detectors, CLASSIFIABLE
    h = load_health()                       # dc_work/atspm/detector_health.parquet
    h = flag_detectors(h)                   # adds health_flag, health_reason
    feats = feats.merge(h[['DeviceId','Detector','health_flag']], how='left')
    train = feats[feats.health_flag.isin(CLASSIFIABLE)]      # drop 'failed' from TRAINING
    # at inference keep every row, but override the output for 'failed' ones:
    out['status'] = h.health_flag.map(status_for_user)

The health table itself is built by `dc_work/atspm/build_health.py` (pure DuckDB, ~2 min
for all 421 signals / 3 days) and is label-free, so it is available for unlabeled channels.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Training-time artefact only; `src/predict.py` computes health from the events it is given.
DEFAULT_HEALTH_PATH = str(Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
                          / "atspm" / "detector_health.parquet")

# ---------------------------------------------------------------------------
# Rules.  Ordered; first match wins, so the reason column is the *primary* reason.
# Each entry: (reason, predicate on the health dataframe)
# ---------------------------------------------------------------------------
FAILED_RULES = [
    # No 82 event in the whole window.  Fold-0 accuracy on these: 6% (mean top prob 0.11).
    ("no_events", lambda d: d["n_on"].fillna(0) == 0),
    # < 20 actuations/day is not a working traffic detector.  Fold-0 accuracy: 21%.
    ("near_zero_volume", lambda d: d["on_per_day"].fillna(0) < 20),
    # Permanently occupied: the ON/OFF pattern carries no phase information.
    ("stuck_on", lambda d: d["frac_time_on"].fillna(0) >= 0.90),
    # >10 Hz toggling for at least one minute: electrical chatter.
    ("chatter_storm", lambda d: d["max_on_per_min"].fillna(0) >= 600),
]

SUSPECT_RULES = [
    # 20-100 actuations/day: classifiable but thin (fold-0 accuracy 43-90%).
    ("low_volume", lambda d: d["on_per_day"].fillna(0) < 100),
    ("high_occupancy", lambda d: d["frac_time_on"].fillna(0) >= 0.50),
    ("long_stuck_interval", lambda d: d["longest_on_s"].fillna(0) >= 3600),
    ("long_daytime_outage", lambda d: d["max_day_gap_s"].fillna(0) >= 21600),
    ("controller_fault_events", lambda d: d["n_fault_events"].fillna(0) > 0),
    ("broken_on_off_stream", lambda d: d["unmatched_on_rate"].fillna(0) > 0.25),
    ("day_to_day_instability", lambda d: d["day_ratio"].fillna(1.0) < 0.10),
    ("chatter", lambda d: d["max_on_per_min"].fillna(0) >= 120),
]

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

CLASSIFIABLE = ("healthy", "suspect")

USER_STATUS = {
    "no_events": "cannot classify: detector produced no actuations in the analysis window "
                 "(channel not wired, card removed, or detector failed)",
    "near_zero_volume": "cannot classify: detector produced almost no actuations "
                        "(<20/day) - likely failed",
    "stuck_on": "cannot classify: detector stuck ON for most of the analysis window "
                "- likely failed (shorted loop / card fault)",
    "chatter_storm": "cannot classify: detector chattering (>10 actuations/second) "
                     "- likely failed (open loop / noise)",
}


# ---------------------------------------------------------------------------
def load_health(path: str = DEFAULT_HEALTH_PATH) -> pd.DataFrame:
    """Read the per-detector health table."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run dc_work/atspm/build_health.py first.")
    return pd.read_parquet(path)


def flag_detectors(df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
    """Add `health_flag` ('healthy'/'suspect'/'failed') and `health_reason` columns.

    `df` must carry the metric columns produced by build_health.py.  Rows are evaluated
    independently, so this works for labeled and unlabeled channels alike.
    """
    df = df.copy()
    if not overwrite and "health_flag" in df.columns:
        return df
    flag = pd.Series("healthy", index=df.index, dtype=object)
    reason = pd.Series("ok", index=df.index, dtype=object)
    for level, rules in (("suspect", SUSPECT_RULES), ("failed", FAILED_RULES)):
        # apply suspect first, then failed, so failed wins; within a level the first
        # matching rule supplies the reason.
        for name, pred in reversed(rules):
            hit = pred(df).fillna(False).to_numpy()
            flag[hit] = level
            reason[hit] = name
    df["health_flag"] = flag
    df["health_reason"] = reason
    return df


def is_classifiable(df: pd.DataFrame) -> pd.Series:
    """True where the detector should get a phase prediction at all."""
    if "health_flag" not in df.columns:
        df = flag_detectors(df)
    return df["health_flag"].isin(CLASSIFIABLE)


def status_for_user(flag: str, reason: str = "") -> str:
    """One-line, end-user facing status string for a detector."""
    if flag == "failed":
        return USER_STATUS.get(reason, "cannot classify: detector failed")
    if flag == "suspect":
        return f"low confidence: detector data quality issue ({reason})"
    return "ok"


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
