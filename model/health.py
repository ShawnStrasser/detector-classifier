"""Detector health: which channels can be classified at all, and what to tell the user.

The research (stage 02) found that, for phase classification, the only detector pathology
that actually costs accuracy is *absence of actuations*.  Everything else (stuck-on,
chatter, unmatched ON/OFF, controller fault events, day-to-day drift) is either very rare
or does not measurably hurt once volume is adequate.  So the rules below are deliberately
asymmetric: `failed` is a small, high-precision set that must NOT be classified,
`suspect` is a wider advisory set that is still classified but reported with lower
confidence and flagged for review.

`predict.py` computes the metric columns these rules read from the events it is given;
the research-only SQL versions live in `research/code/features/health_sql.py`.
"""
from __future__ import annotations

import pandas as pd

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


def status_for_user(flag: str, reason: str = "") -> str:
    """One-line, end-user facing status string for a detector."""
    if flag == "failed":
        return USER_STATUS.get(reason, "cannot classify: detector failed")
    if flag == "suspect":
        return f"low confidence: detector data quality issue ({reason})"
    return "ok"
