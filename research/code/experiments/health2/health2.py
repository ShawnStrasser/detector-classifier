"""Period-level detector health: find bad *intervals*, mask them, score trust.

Stage 02 (`results/02_atspm_health.md`) flagged bad *channels* over the whole sample.
This module flags bad *periods* inside the sample, in the spirit of ODOT's production
ATSPM detector-health job but without its multi-week history: everything here is
within-sample, label-free and config-free, so it runs on 30 minutes of hi-res events
just as well as on 3 days.

Three public entry points
-------------------------
``find_bad_intervals(con, ...)``  -> DataFrame(DeviceId, Detector, start, end, reason, ...)
                                    ``Detector IS NULL`` means "the whole signal".
``mask_intervals(...)``           -> remove those periods from a detector-interval table
                                    (SQL predicate or pandas filter) before feature building.
``trust_score(...)``              -> P(top-1 prediction is correct) from model confidence +
                                    evidence + health, for coverage control / review routing.

Detections (all within-sample)
------------------------------
=========================  ======================================================
reason                     rule
=========================  ======================================================
``comm_loss``              signal-level: no event of ANY kind for >= ``comm_gap_s``
``flash``                  signal-level: event 173, or event 131 pattern 255
``fault_84..88``           detector fault declared (84-88) until restore (83),
                           capped at ``fault_max_s``
``stuck_on``               one continuous ON longer than ``max(stuck_on_s,
                           stuck_cycles * this signal's median cycle length)``
``chatter``                >= ``chatter_per_min`` ONs in a clock minute with a
                           median ON shorter than ``chatter_max_dur_s``
``flatline``               detector silent for >= ``flat_gap_s`` while the rest of
                           the signal keeps actuating (>= ``flat_signal_frac`` of
                           the gap's minutes have activity on other channels) and
                           the detector is otherwise busy (>= ``flat_min_rate_hr``)
``level_low``/``level_high``  >= 2 days of data: an (day, hour-of-day) cell whose
                           actuation count is < 1/``level_ratio`` or >
                           ``level_ratio`` x the median of the same hour on the
                           detector's other days
=========================  ======================================================

``comm_loss`` is deliberately separated from ``flatline``: a silent detector while the
whole signal is silent is a communications / logging outage, not a broken detector, and
masking it would be wrong (there is nothing to mask) while *flagging* the detector would
be a false alarm.

Usage
-----
    import duckdb, health2
    con = duckdb.connect(); con.execute("SET threads=6; SET memory_limit='6GB'")
    con.execute("CREATE TEMP VIEW ev AS SELECT DISTINCT * FROM read_parquet(...)")
    bad = health2.find_bad_intervals(con, ev="ev")
    summ = health2.summarise(bad, universe, t0, t1)
    # feature building: drop masked ONs
    con.execute("CREATE TEMP TABLE bad AS SELECT * FROM bad_df")
    con.execute(f"CREATE TEMP TABLE onev_all AS SELECT * FROM iv WHERE {health2.MASK_PREDICATE}")

Integration into `src/predict.py` (specification, not applied -- see
`results/09_detector_health.md` for why masking is OFF by default)
------------------------------------------------------------------
1. `import health2` next to `from health import flag_detectors, status_for_user`; add
   `mask_bad_periods: bool = False` and `trust_threshold: float = 0.80` to `predict()`
   and to the CLI (`--mask-bad-periods`, `--trust-threshold`).
2. In `predict()`, right after `build_chunk_tables(con)` (which creates `onev_all`):

       bad = health2.find_bad_intervals(con, ev="ev", t0=..., t1=...)
       if mask_bad_periods and len(bad):
           con.register("_bad_src", bad_as_epoch_seconds(bad))   # t0/t1 like onev_all
           con.execute("CREATE OR REPLACE TABLE _bad AS SELECT * FROM _bad_src")
           con.execute("DELETE FROM onev_all o WHERE EXISTS (SELECT 1 FROM _bad b "
                       "JOIN devmap m USING (DeviceId) WHERE m.dev = o.dev AND "
                       "(b.det IS NULL OR b.det = o.det) AND o.t_on < b.t1 AND o.t_off > b.t0)")

   Nothing downstream changes: `health_frame`, `build_features` and the models all read
   `onev_all`.  Masking must therefore happen BEFORE `health_frame(con, ...)`, or the
   stage-02 channel flags will be computed on the cleaned stream (which is usually what
   you want -- say so in the docstring).
3. `hf = health_frame(...)` gains the period summary:

       summ = health2.summarise(bad, univ[["DeviceId","Detector"]], t_start, t_end)
       hf = hf.merge(summ, on=["DeviceId","Detector"], how="left")

4. In `_assemble`, after the existing `n_actuations` / `health_flag` branches and before
   the `phase_prob < LOW_CONF` note, add two things:

   (a) the period messages, from `health2.describe_intervals(bad)`, e.g.
       "detector stuck on 02:10-05:40 - period ignored" (or "- period included" when
       `mask_bad_periods` is False).  Append at most 2-3 lines per detector and set
       `review_flag` only for `stuck_on`, `chatter` and `flatline`; `comm_loss` and
       `flash` are signal-level and belong in a per-signal note, not a per-detector one.
   (b) a `trust` column:

           res["trust"] = health2.trust_score(res.rename(columns={"phase_prob":"top_prob",
                              "phase_margin":"margin", "minutes_of_data":"win_hours"}),
                              health=summ, model=load_trust_model(model_dir))
           note = health2.trust_status(trust, reasons, masked_frac, trust_threshold)
           # -> "prediction low-trust (P(correct) ~ 0.62); detector health: stuck_on;
           #     18% of the window ignored"

   `trust` replaces the bare `phase_prob < 0.5` test as the review trigger and is the
   column a downstream service should threshold for coverage control.
5. New output columns: `trust`, `masked_frac`, `health_periods` (the joined messages).
   `OUT_COLS` / `EXTRA_COLS` and the module docstring's column list must list them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthParams:
    """Thresholds.  Defaults are calibrated on ODOT Dec-2024 hi-res logs (DEV only)."""
    # signal level
    comm_gap_s: float = 300.0        # no events at all -> logging / comms outage
    # detector faults (events 84-88 -> 83)
    fault_max_s: float = 3600.0
    # stuck ON
    stuck_on_s: float = 300.0
    stuck_cycles: float = 3.0        # ... and at least this many signal cycles
    # chatter
    chatter_per_min: int = 120       # >= 2 Hz sustained for a clock minute
    chatter_max_dur_s: float = 0.5   # ... with short ONs (else it is real traffic)
    # flatline (dead detector while the signal lives)
    flat_gap_s: float = 1800.0
    flat_signal_frac: float = 0.90   # share of the gap's minutes in which the signal logged
    flat_min_rate_hr: float = 10.0   # detector must be this busy overall to be checked
    flat_expected_n: int = 30        # expected ONs in the gap from the signal's own demand
    # level shift (needs >= 2 days)
    level_ratio: float = 5.0
    level_min_count: int = 30        # reference median must be at least this many ONs
    # GEH volume anomaly (the ODOT ATSPM production method, ported to a short sample)
    geh_bin_min: int = 15
    geh_threshold: float = 6.0       # ODOT `entity_threshold`
    geh_peer_z: float = 3.0          # ODOT `group_threshold`, peer group = this signal
    geh_min_median: float = 3.0      # skip channels whose typical bin is near zero
    # bookkeeping
    min_interval_s: float = 1.0      # drop shorter intervals


DEFAULTS = HealthParams()

REASONS = ("comm_loss", "flash", "fault_84", "fault_85", "fault_86", "fault_87",
           "fault_88", "stuck_on", "chatter", "flatline", "level_low", "level_high",
           "geh_anomaly")

#: reasons that mask a detector's own events (the rest mask the whole signal)
DETECTOR_REASONS = ("fault_84", "fault_85", "fault_86", "fault_87", "fault_88",
                    "stuck_on", "chatter", "flatline", "level_low", "level_high",
                    "geh_anomaly")

#: default masking set.  `geh_anomaly` and `level_*` are *advisory*: they say the volume
#: was unusual, not that the individual actuations were wrong, so masking them throws
#: away real vehicles.  They are trust-model inputs, not mask reasons, unless asked for.
MASK_REASONS = ("comm_loss", "flash", "fault_84", "fault_85", "fault_86", "fault_87",
                "fault_88", "stuck_on", "chatter", "flatline")

#: SQL predicate for "this detector-ON is not inside a bad interval".
#: expects a detector-interval table aliased `i` (DeviceId, Detector, t_on, t_off)
#: and a bad-interval table `bad` (DeviceId, Detector, start, end).
MASK_PREDICATE = """NOT EXISTS (
    SELECT 1 FROM bad b
    WHERE b.DeviceId = i.DeviceId
      AND (b.Detector IS NULL OR b.Detector = i.Detector)
      AND i.t_on < b."end" AND i.t_off > b."start")"""


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
_PREP_SQL = """
-- Every allowed code is kept: the "is this signal logging at all?" test must see the
-- phase-colour and call events too, otherwise a coordinated signal sitting in a long
-- phase-2 green looks like a communications outage.
CREATE OR REPLACE TEMP TABLE _h2_ev AS
SELECT DeviceId, Timestamp AS ts, EventId::SMALLINT AS EventId, Parameter::SMALLINT AS Parameter
FROM ({ev}) WHERE EventId IN (1,7,8,9,10,11,43,44,81,82,83,84,85,86,87,88,131,150,173)
  AND (EventId NOT IN (81,82,83,84,85,86,87,88) OR Parameter <= {max_ch})
  {trange};

-- ON/OFF pairing.  ORDER BY ts, EventId DESC puts 82 (ON) before 81 (OFF) at an
-- identical timestamp so a zero-length actuation still pairs.
CREATE OR REPLACE TEMP TABLE _h2_iv AS
WITH p AS (
  SELECT DeviceId, Parameter AS Detector, ts, EventId,
         lead(ts)      OVER w AS next_ts,
         lead(EventId) OVER w AS next_ev
  FROM _h2_ev WHERE EventId IN (81,82)
  WINDOW w AS (PARTITION BY DeviceId, Parameter ORDER BY ts, EventId DESC)
)
SELECT DeviceId, Detector, ts AS t_on, next_ts AS t_off,
       epoch_ms(next_ts - ts)/1000.0 AS dur
FROM p WHERE EventId = 82 AND next_ev = 81;

-- per-signal observed span and median cycle length (median gap between
-- consecutive begin-greens of the phase that runs most often)
CREATE OR REPLACE TEMP TABLE _h2_sig AS
WITH span AS (SELECT DeviceId, min(ts) AS t0, max(ts) AS t1 FROM _h2_ev GROUP BY 1),
g AS (
  SELECT DeviceId, Parameter AS p, ts,
         epoch_ms(ts - lag(ts) OVER (PARTITION BY DeviceId, Parameter ORDER BY ts))/1000.0 AS d
  FROM _h2_ev WHERE EventId = 1 AND Parameter BETWEEN 1 AND 16
),
cyc AS (
  SELECT DeviceId, median(d) AS cycle_s
  FROM g WHERE d BETWEEN 20 AND 600 GROUP BY 1
)
SELECT s.DeviceId, s.t0, s.t1, coalesce(c.cycle_s, 120.0) AS cycle_s
FROM span s LEFT JOIN cyc c USING (DeviceId);
"""

# --- signal-level: communications / logging outage --------------------------
_COMM_SQL = """
SELECT DeviceId, NULL::SMALLINT AS Detector, prev_ts AS "start", ts AS "end",
       'comm_loss' AS reason
FROM (SELECT DeviceId, ts, lag(ts) OVER (PARTITION BY DeviceId ORDER BY ts) AS prev_ts
      FROM (SELECT DISTINCT DeviceId, ts FROM _h2_ev))
WHERE prev_ts IS NOT NULL AND epoch_ms(ts - prev_ts)/1000.0 >= {comm_gap_s}
"""

# --- signal-level: flash ----------------------------------------------------
# 173 = unit flash status, 131 pattern 255 = flash.  Mask until the next
# coordination-pattern/flash event on that signal (capped at fault_max_s).
_FLASH_SQL = """
SELECT DeviceId, NULL::SMALLINT AS Detector, ts AS "start",
       least(coalesce(nxt, ts + to_seconds({fault_max_s}::INT)),
             ts + to_seconds({fault_max_s}::INT)) AS "end",
       'flash' AS reason
FROM (
  SELECT DeviceId, ts, EventId, Parameter,
         lead(ts) OVER (PARTITION BY DeviceId ORDER BY ts) AS nxt
  FROM _h2_ev WHERE EventId IN (131, 173)
) WHERE (EventId = 131 AND Parameter = 255) OR EventId = 173
"""

# --- detector faults 84-88 -> 83 -------------------------------------------
_FAULT_SQL = """
SELECT DeviceId, Detector, "start",
       least(coalesce("end", "start" + to_seconds({fault_max_s}::INT)),
             "start" + to_seconds({fault_max_s}::INT)) AS "end", reason
FROM (
  SELECT DeviceId, Parameter AS Detector, ts AS "start", EventId,
         lead(ts) OVER (PARTITION BY DeviceId, Parameter ORDER BY ts) AS "end",
         'fault_' || EventId::VARCHAR AS reason
  FROM _h2_ev WHERE EventId BETWEEN 83 AND 88
) WHERE EventId <> 83
"""

# --- stuck ON ---------------------------------------------------------------
_STUCK_SQL = """
SELECT i.DeviceId, i.Detector, i.t_on AS "start", i.t_off AS "end", 'stuck_on' AS reason
FROM _h2_iv i JOIN _h2_sig s USING (DeviceId)
WHERE i.dur >= greatest({stuck_on_s}, {stuck_cycles} * s.cycle_s)
"""

# --- chatter ----------------------------------------------------------------
_CHATTER_SQL = """
WITH m AS (
  SELECT DeviceId, Detector, date_trunc('minute', t_on) AS m0,
         count(*) AS n, median(dur) AS med_dur
  FROM _h2_iv GROUP BY 1,2,3
  HAVING count(*) >= {chatter_per_min} AND median(dur) < {chatter_max_dur_s}
),
-- merge consecutive chattering minutes into one interval (gaps-and-islands)
g AS (
  SELECT *, date_diff('minute',
            lag(m0) OVER (PARTITION BY DeviceId, Detector ORDER BY m0), m0) AS d FROM m
),
isl AS (
  SELECT *, sum(CASE WHEN d = 1 THEN 0 ELSE 1 END)
            OVER (PARTITION BY DeviceId, Detector ORDER BY m0
                  ROWS UNBOUNDED PRECEDING) AS grp FROM g
)
SELECT DeviceId, Detector, min(m0) AS "start",
       max(m0) + INTERVAL 1 MINUTE AS "end", 'chatter' AS reason
FROM isl GROUP BY 1,2,grp
"""

# --- flatline (dead detector while the signal is alive) ---------------------
# Candidate gaps come from the detector's own actuation stream (few rows); each is then
# checked against a per-minute, per-signal activity table.
_FLAT_SQL = """
WITH act AS (   -- detectors worth checking at all
  SELECT i.DeviceId, i.Detector, count(*) AS n_on,
         min(i.t_on) AS first_on, max(i.t_on) AS last_on
  FROM _h2_iv i GROUP BY 1,2
),
rate AS (
  SELECT a.*, s.t0, s.t1,
         a.n_on / nullif(epoch_ms(s.t1 - s.t0)/3600000.0, 0) AS on_per_hr
  FROM act a JOIN _h2_sig s USING (DeviceId)
  WHERE a.n_on >= 10
),
gaps AS (   -- interior gaps
  SELECT r.DeviceId, r.Detector, p.prev_t AS "start", p.t_on AS "end"
  FROM (SELECT DeviceId, Detector, t_on,
               lag(t_on) OVER (PARTITION BY DeviceId, Detector ORDER BY t_on) AS prev_t
        FROM _h2_iv) p
  JOIN rate r USING (DeviceId, Detector)
  WHERE p.prev_t IS NOT NULL
    AND epoch_ms(p.t_on - p.prev_t)/1000.0 >= {flat_gap_s}
    AND r.on_per_hr >= {flat_min_rate_hr}
  UNION ALL   -- leading gap
  SELECT DeviceId, Detector, t0, first_on FROM rate
  WHERE epoch_ms(first_on - t0)/1000.0 >= {flat_gap_s} AND on_per_hr >= {flat_min_rate_hr}
  UNION ALL   -- trailing gap
  SELECT DeviceId, Detector, last_on, t1 FROM rate
  WHERE epoch_ms(t1 - last_on)/1000.0 >= {flat_gap_s} AND on_per_hr >= {flat_min_rate_hr}
),
evmin AS (SELECT DeviceId, date_trunc('minute', ts) AS m0, count(*) AS n_ev
          FROM _h2_ev GROUP BY 1,2),
onmin AS (SELECT DeviceId, date_trunc('minute', t_on) AS m0, count(*) AS n_on
          FROM _h2_iv GROUP BY 1,2),
minagg AS (SELECT e.DeviceId, e.m0, coalesce(o.n_on, 0) AS n_on
           FROM evmin e LEFT JOIN onmin o USING (DeviceId, m0)),
share AS (   -- this channel's share of the signal's actuations over the whole sample
  SELECT r.DeviceId, r.Detector,
         r.n_on::DOUBLE / nullif(sum(r.n_on) OVER (PARTITION BY r.DeviceId), 0) AS sh
  FROM rate r
),
chk AS (
  SELECT g.DeviceId, g.Detector, g."start", g."end",
         epoch_ms(g."end" - g."start")/60000.0 AS gap_min,
         count(*) AS up_min, sum(m.n_on) AS oth_on
  FROM gaps g JOIN minagg m
    ON m.DeviceId = g.DeviceId AND m.m0 >= g."start" AND m.m0 < g."end"
  GROUP BY 1,2,3,4
)
-- Demand-normalised: how many ONs would this channel have produced during the gap,
-- given what the REST of the signal actually saw?  Being silent at 3 am is normal;
-- being silent while your own share of the signal's traffic implies 30+ actuations
-- is not.  This replaces a clock-time rule and needs no history.
SELECT c.DeviceId, c.Detector, c."start", c."end", 'flatline' AS reason
FROM chk c JOIN share s USING (DeviceId, Detector)
WHERE c.gap_min > 0
  AND c.up_min / c.gap_min >= {flat_signal_frac}                 -- comms were up
  AND s.sh < 0.95
  AND c.oth_on * s.sh / (1.0 - s.sh) >= {flat_expected_n}
"""

# --- level shift vs the detector's own other days ---------------------------
_LEVEL_SQL = """
WITH h AS (
  SELECT DeviceId, Detector, t_on::DATE AS d, hour(t_on) AS hh, count(*) AS n
  FROM _h2_iv GROUP BY 1,2,3,4
),
full_hours AS (   -- only hours fully inside the observed span for that signal
  SELECT h.* FROM h JOIN _h2_sig s USING (DeviceId)
  WHERE (h.d + to_hours(h.hh)) >= s.t0 AND (h.d + to_hours(h.hh) + INTERVAL 1 HOUR) <= s.t1
),
ref AS (   -- the same hour-of-day on this detector's OTHER days
  SELECT DeviceId, Detector, hh, count(*) AS n_days, median(n) AS med_n
  FROM full_hours GROUP BY 1,2,3
)
SELECT f.DeviceId, f.Detector, (f.d + to_hours(f.hh)) AS "start",
       (f.d + to_hours(f.hh) + INTERVAL 1 HOUR) AS "end",
       CASE WHEN f.n * {level_ratio} < r.med_n THEN 'level_low' ELSE 'level_high' END AS reason
FROM full_hours f JOIN ref r USING (DeviceId, Detector, hh)
WHERE r.n_days >= 2 AND r.med_n >= {level_min_count}
  AND (f.n * {level_ratio} < r.med_n OR f.n > r.med_n * {level_ratio})
"""

# --- GEH volume anomaly: the ODOT ATSPM production method, ported -----------
# ODOT's `detector_health` job (atspm + traffic_anomaly) is a volume-anomaly detector on
# 15-min counts of event 82: zero-filled bins, `decompose` with a STATIC per-detector
# median (rolling_window_enable=False) plus time-of-day / day-of-week seasonal medians,
# then a signed GEH residual with under-counting amplified (`log_adjust_negative`),
# flagged when |GEH| > 6 AND the residual is > 3 SD from its PEER GROUP at the same
# timestamp.  The peer gate is what stops a snowstorm or a regional comms outage from
# flagging every detector at once.
# Ported here to a short sample: the peer group becomes "the other channels of this
# signal" (finer than ODOT's corridor `group_name`), and the seasonal term is only used
# when there are enough days to estimate it -- with 3 days there are not, so the
# baseline is the detector's own median 15-min count over the sample.
_GEH_SQL = """
WITH grid AS (
  SELECT DeviceId, unnest(generate_series(
           time_bucket(INTERVAL {geh_bin_min} MINUTES, t0), t1,
           INTERVAL {geh_bin_min} MINUTES)) AS b
  FROM _h2_sig
),
dets AS (SELECT DISTINCT DeviceId, Detector FROM _h2_iv),
cnt AS (
  SELECT DeviceId, Detector, time_bucket(INTERVAL {geh_bin_min} MINUTES, t_on) AS b,
         count(*) AS n FROM _h2_iv GROUP BY 1,2,3
),
full_bins AS (   -- zero-fill: a detector that went silent must be a 0, not a missing row
  SELECT d.DeviceId, d.Detector, g.b, coalesce(c.n, 0)::DOUBLE AS n
  FROM dets d JOIN grid g USING (DeviceId)
  LEFT JOIN cnt c ON c.DeviceId = d.DeviceId AND c.Detector = d.Detector AND c.b = g.b
),
med AS (SELECT DeviceId, Detector, median(n) AS med FROM full_bins GROUP BY 1,2),
pred AS (
  SELECT f.DeviceId, f.Detector, f.b, f.n, greatest(m.med, 0.0) AS p
  FROM full_bins f JOIN med m USING (DeviceId, Detector) WHERE m.med >= {geh_min_median}
),
res AS (
  SELECT *, sign(p - n) * sqrt(2 * (p - n) * (p - n) / (p + n + 1e-8)) AS geh
  FROM pred
),
adj AS (   -- log_adjust_negative: amplify under-counts (a dead detector) over over-counts
  SELECT *, geh * greatest((-ln(n / (p + 1e-8) + 0.1) + 2) / 2, 1.0) AS resid FROM res
),
z AS (   -- peer gate: how does this residual compare with the signal's other channels now?
  SELECT *, (resid - avg(resid) OVER w) / nullif(stddev_samp(resid) OVER w, 0) AS zz,
         count(*) OVER w AS n_peers
  FROM adj WINDOW w AS (PARTITION BY DeviceId, b)
),
flag AS (
  SELECT DeviceId, Detector, b FROM z
  WHERE abs(resid) > {geh_threshold} AND (n_peers < 4 OR zz > {geh_peer_z})
),
g AS (SELECT *, date_diff('minute', lag(b) OVER w, b) AS d FROM flag
      WINDOW w AS (PARTITION BY DeviceId, Detector ORDER BY b)),
isl AS (SELECT *, sum(CASE WHEN d = {geh_bin_min} THEN 0 ELSE 1 END)
                  OVER (PARTITION BY DeviceId, Detector ORDER BY b ROWS UNBOUNDED PRECEDING) AS grp
        FROM g)
SELECT DeviceId, Detector, min(b) AS "start",
       max(b) + INTERVAL {geh_bin_min} MINUTES AS "end", 'geh_anomaly' AS reason
FROM isl GROUP BY 1,2,grp
"""

_PARTS = {"comm_loss": _COMM_SQL, "flash": _FLASH_SQL, "fault": _FAULT_SQL,
          "stuck_on": _STUCK_SQL, "chatter": _CHATTER_SQL, "flatline": _FLAT_SQL,
          "level": _LEVEL_SQL, "geh": _GEH_SQL}


# ---------------------------------------------------------------------------
def prepare(con, ev: str = "ev", t0=None, t1=None, max_channel: int = 64) -> None:
    """Build the temp tables `_h2_ev`, `_h2_iv`, `_h2_sig` from a raw event source.

    `ev` is a table/view name or a parenthesised SELECT giving
    (DeviceId, Timestamp, EventId, Parameter).  Duplicate rows are removed here.
    """
    src = ev if ev.strip().startswith("(") else f"SELECT * FROM {ev}"
    src = f"SELECT DISTINCT DeviceId, Timestamp, EventId, Parameter FROM ({src})"
    tr = ""
    if t0 is not None:
        tr += f" AND Timestamp >= TIMESTAMP '{pd.Timestamp(t0)}'"
    if t1 is not None:
        tr += f" AND Timestamp < TIMESTAMP '{pd.Timestamp(t1)}'"
    for stmt in _PREP_SQL.format(ev=src, max_ch=max_channel, trange=tr).split(";"):
        if stmt.strip():
            con.execute(stmt)


def find_bad_intervals(con, ev: str = "ev", params: HealthParams = DEFAULTS,
                       parts: Optional[tuple] = None, prepared: bool = False,
                       t0=None, t1=None) -> pd.DataFrame:
    """Return every period in which a detector (or a whole signal) misbehaves.

    Columns: DeviceId, Detector (Int16, NULL = whole signal), start, end, reason, secs.
    Overlapping intervals of different reasons are kept separate; use `merge_intervals`
    to get the union per detector.
    """
    if not prepared:
        prepare(con, ev, t0, t1)
    out = []
    for name, sql in _PARTS.items():
        if parts is not None and name not in parts:
            continue
        df = con.sql(sql.format(**asdict(params))).df()
        if len(df):
            out.append(df)
    if not out:
        return pd.DataFrame({"DeviceId": pd.Series(dtype=object),
                             "Detector": pd.Series(dtype="Int16"),
                             "start": pd.Series(dtype="datetime64[us]"),
                             "end": pd.Series(dtype="datetime64[us]"),
                             "reason": pd.Series(dtype=object),
                             "secs": pd.Series(dtype=float)})
    bad = pd.concat(out, ignore_index=True)
    bad["Detector"] = bad["Detector"].astype("Int16")
    bad["secs"] = (bad["end"] - bad["start"]).dt.total_seconds()
    bad = bad[bad.secs >= params.min_interval_s].reset_index(drop=True)
    return bad.sort_values(["DeviceId", "Detector", "start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def merge_intervals(bad: pd.DataFrame, by=("DeviceId", "Detector")) -> pd.DataFrame:
    """Union of overlapping intervals per group (keeps the first reason of each run)."""
    if not len(bad):
        return bad
    b = bad.sort_values(list(by) + ["start"]).copy()
    key = b[list(by)].astype(str).agg("|".join, axis=1)
    newgrp = (key != key.shift()).to_numpy()
    prev_end = b["end"].shift()
    starts = (newgrp | (b["start"].to_numpy() > prev_end.to_numpy()))
    b["_g"] = np.cumsum(starts)
    out = b.groupby("_g").agg(**{c: (c, "first") for c in by},
                              start=("start", "min"), end=("end", "max"),
                              reason=("reason", lambda s: ",".join(sorted(set(s)))))
    out["secs"] = (out["end"] - out["start"]).dt.total_seconds()
    return out.reset_index(drop=True)


def mask_intervals(iv: pd.DataFrame, bad: pd.DataFrame,
                   on: str = "t_on", off: str = "t_off") -> pd.DataFrame:
    """Drop detector-ON intervals that overlap a bad interval (pandas path).

    `iv`: DeviceId, Detector, t_on, t_off.  Signal-level bad intervals
    (`Detector` NULL) apply to every detector of that signal.
    """
    if not len(bad) or not len(iv):
        return iv
    keep = np.ones(len(iv), dtype=bool)
    iv_on, iv_off = iv[on].to_numpy(), iv[off].to_numpy()
    for dev, gb in bad.groupby("DeviceId", sort=False):
        sel_dev = (iv.DeviceId.to_numpy() == dev)
        if not sel_dev.any():
            continue
        for det, gd in gb.groupby(gb.Detector.fillna(-1).astype(int), sort=False):
            sel = sel_dev if det == -1 else (sel_dev & (iv.Detector.to_numpy() == det))
            if not sel.any():
                continue
            for s, e in zip(gd["start"].to_numpy(), gd["end"].to_numpy()):
                keep &= ~(sel & (iv_on < e) & (iv_off > s))
    return iv[keep]


def mask_sql(bad_table: str = "bad", iv_alias: str = "i") -> str:
    """SQL predicate form of `mask_intervals` (preferred for feature building)."""
    return MASK_PREDICATE.replace(" bad b", f" {bad_table} b").replace("i.", f"{iv_alias}.")


# ---------------------------------------------------------------------------
def summarise(bad: pd.DataFrame, universe: pd.DataFrame,
              t0, t1, reasons: tuple = DETECTOR_REASONS) -> pd.DataFrame:
    """Per-detector health summary over the window [t0, t1).

    `universe`: DeviceId, Detector (every channel to report on; usually every channel
    with an 81/82 event, plus configured-but-silent ones).
    Returns one row per detector with `masked_secs`, `masked_frac`, `n_reasons`,
    `health_reasons` (comma separated) and one 0/1 column per reason.
    """
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    win = max((t1 - t0).total_seconds(), 1.0)
    out = universe[["DeviceId", "Detector"]].drop_duplicates().copy()
    for r in reasons:
        out[f"h_{r}"] = 0.0
    out["masked_secs"] = 0.0
    out["health_reasons"] = ""
    if not len(bad):
        out["masked_frac"] = 0.0
        out["n_reasons"] = 0
        return out

    b = bad.copy()
    b["start"] = b["start"].clip(lower=t0)
    b["end"] = b["end"].clip(upper=t1)
    b["secs"] = (b["end"] - b["start"]).dt.total_seconds()
    b = b[b.secs > 0]
    # signal-level rows apply to every detector of the signal
    sig = b[b.Detector.isna()]
    det = b[b.Detector.notna()]
    if len(sig):
        exp = out[["DeviceId", "Detector"]].merge(
            sig.drop(columns=["Detector"]), on="DeviceId", how="inner")
        det = pd.concat([det, exp], ignore_index=True)
    if len(det):
        det["Detector"] = det["Detector"].astype("int64")
        per = det.groupby(["DeviceId", "Detector", "reason"], as_index=False).secs.sum()
        wide = per.pivot_table(index=["DeviceId", "Detector"], columns="reason",
                               values="secs", fill_value=0.0).reset_index()
        tot = merge_intervals(det).groupby(["DeviceId", "Detector"], as_index=False).secs.sum()
        tot = tot.rename(columns={"secs": "masked_secs_"})
        out = out.drop(columns=["masked_secs"]).merge(tot, on=["DeviceId", "Detector"],
                                                      how="left")
        out["masked_secs"] = out.pop("masked_secs_").fillna(0.0)
        for r in wide.columns:
            if r in ("DeviceId", "Detector"):
                continue
            col = f"h_{r}"
            m = wide[["DeviceId", "Detector", r]].rename(columns={r: "_v"})
            out = out.merge(m, on=["DeviceId", "Detector"], how="left")
            out[col] = out.pop("_v").fillna(0.0) if col not in out.columns else \
                out[col].fillna(0.0) + out.pop("_v").fillna(0.0)
    rcols = [c for c in out.columns if c.startswith("h_")]
    out["n_reasons"] = (out[rcols] > 0).sum(axis=1)
    out["health_reasons"] = [",".join(c[2:] for c in rcols if row[c] > 0)
                             for _, row in out[rcols].iterrows()]
    out["masked_frac"] = (out["masked_secs"] / win).clip(0, 1)
    for c in rcols:
        out[c] = (out[c] / win).clip(0, 1)
    return out


# ---------------------------------------------------------------------------
# Trust / confidence
# ---------------------------------------------------------------------------
#: inputs the trust model may use.  Deliberately small, phase-anonymous and
#: label-free so it can be fitted once and shipped with the model.
TRUST_FEATURES = ["top_prob", "margin", "log_n_act", "log_win_hours", "n_cand",
                  "entropy", "masked_frac", "n_reasons",
                  "h_stuck_on", "h_chatter", "h_flatline", "h_fault",
                  "h_level_low", "h_level_high", "h_comm_loss"]


def trust_inputs(pred: pd.DataFrame, health: Optional[pd.DataFrame] = None,
                 probs: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Assemble the trust-model design matrix.

    `pred` needs: top_prob, second_prob (or margin), n_actuations, win_hours, n_cand
    and optionally an `entropy` column.  `health` is the output of `summarise`.
    """
    d = pd.DataFrame(index=pred.index)
    d["top_prob"] = pred["top_prob"].astype(float)
    d["margin"] = (pred["margin"] if "margin" in pred
                   else pred["top_prob"] - pred.get("second_prob", 0.0)).astype(float)
    d["log_n_act"] = np.log1p(pred["n_actuations"].fillna(0).astype(float))
    d["log_win_hours"] = np.log1p(pred["win_hours"].astype(float))
    d["n_cand"] = pred["n_cand"].astype(float)
    d["entropy"] = pred["entropy"].astype(float) if "entropy" in pred else np.nan
    if health is not None:
        h = health.set_index(["DeviceId", "Detector"])
        idx = pd.MultiIndex.from_arrays([pred.DeviceId, pred.Detector])
        for c in ("masked_frac", "n_reasons"):
            d[c] = h[c].reindex(idx).to_numpy() if c in h else 0.0
        for r in ("stuck_on", "chatter", "flatline", "level_low", "level_high",
                  "comm_loss"):
            col = f"h_{r}"
            d[col] = h[col].reindex(idx).to_numpy() if col in h else 0.0
        fcols = [c for c in h.columns if c.startswith("h_fault_")]
        d["h_fault"] = (h[fcols].sum(axis=1).reindex(idx).to_numpy() if fcols else 0.0)
    else:
        for c in TRUST_FEATURES[6:]:
            d[c] = 0.0
    return d[[c for c in TRUST_FEATURES if c in d.columns]].fillna(0.0)


def trust_score(pred: pd.DataFrame, health: Optional[pd.DataFrame] = None,
                model=None, rule_only: bool = False) -> np.ndarray:
    """P(the top-1 prediction is correct), in [0, 1].

    With `model` (a fitted LightGBM/sklearn binary classifier over `TRUST_FEATURES`)
    this is the calibrated probability.  Without one it falls back to a transparent
    rule: the model's own top probability, multiplied down by thin evidence and by
    health problems.  The rule is what `predict.py` can ship before a trust model is
    trained; the fitted model is strictly better where labels exist.
    """
    X = trust_inputs(pred, health)
    if model is not None and not rule_only:
        p = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") \
            else np.asarray(model.predict(X))
        return np.clip(p, 0.0, 1.0)
    def _col(name, default=0.0):
        return (X[name].to_numpy(dtype=float) if name in X.columns
                else np.full(len(X), default))

    p = X["top_prob"].to_numpy(dtype=float).copy()
    n = np.expm1(_col("log_n_act"))
    p *= np.clip(0.80 + 0.20 * np.log1p(n) / np.log(30.0), 0.0, 1.0)   # evidence
    p *= 1.0 - 0.30 * np.clip(_col("masked_frac"), 0, 1)                # masked time
    p *= 1.0 - 0.10 * np.clip(_col("n_reasons"), 0, 3) / 3.0            # how many faults
    return np.clip(p, 0.0, 1.0)


def trust_status(score: float, reasons: str = "", masked_frac: float = 0.0,
                 threshold: float = 0.80) -> str:
    """One-line user-facing note for a low-trust prediction."""
    if score >= threshold:
        return ""
    bits = [f"prediction low-trust (P(correct) ~ {score:.2f})"]
    if reasons:
        bits.append(f"detector health: {reasons}")
    if masked_frac > 0.01:
        bits.append(f"{masked_frac*100:.0f}% of the window ignored")
    return "; ".join(bits)


def describe_intervals(bad: pd.DataFrame, max_rows: int = 6) -> dict:
    """(DeviceId, Detector) -> human-readable status lines, e.g.
    'detector stuck on 02:10-05:40 - period ignored'."""
    out: dict = {}
    if not len(bad):
        return out
    verb = {"stuck_on": "detector stuck on", "chatter": "detector chattering",
            "flatline": "detector silent while the signal was active",
            "comm_loss": "no data from this signal", "flash": "signal in flash",
            "level_low": "actuation count far below this detector's other days",
            "level_high": "actuation count far above this detector's other days"}
    for (dev, det), g in bad.groupby(["DeviceId", bad.Detector.fillna(-1)], sort=False):
        lines = []
        for r in g.sort_values("start").head(max_rows).itertuples():
            v = verb.get(r.reason, f"detector fault event {r.reason[-2:]}")
            lines.append(f"{v} {pd.Timestamp(r.start):%H:%M}-{pd.Timestamp(r.end):%H:%M}"
                         f" - period ignored")
        if len(g) > max_rows:
            lines.append(f"(+{len(g)-max_rows} more periods)")
        out[(dev, None if det == -1 else int(det))] = lines
    return out
