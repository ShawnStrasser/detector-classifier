"""Stage 06 step 1 -- extra *function* features: yellow / red-clearance behaviour and
pairwise actuation lag.  Both are phase-anonymous (channel numbers and phase numbers are
join keys only).

1. `func_yr_extra.parquet` -- one row per (DeviceId, Detector, cand_phase, win).
   A **Yellow_Red** detector is a downstream, non-calling counter placed to see the vehicles
   that enter on yellow / during red clearance.  Stage 04/05 features split the cycle into
   green / yellow / red only, and "red" lumps the red-clearance interval together with the
   long wait.  Events 10 and 11 (begin / end red clearance) let us cut that apart:

        green      [green_start, yellow_start)
        yellow     [yellow_start, red_start)          red_start = BEGIN red clearance (ev 10)
        redclear   [red_start,   redclr_end)          redclr_end = END red clearance (ev 11)
        red rest   [redclr_end,  next_green)

   Every feature is a share, a per-cycle count or a *rate lift* against the same detector's
   green rate, so it is duration invariant.

2. `det_lag.parquet` -- one row per (DeviceId, win, Detector, other): the signed lag from
   each ON of `Detector` to the nearest ON of `other` (median / quartiles), and the share of
   ONs that have a partner within 0.3 / 1 / 3 s.  An upstream **Advance** loop fires ~2-6 s
   before the stop-bar **Presence** loop of the same approach (setback / free-flow speed);
   a per-lane **Count** loop sits at the same longitudinal position as its neighbour
   (lag ~ 0) but sees different vehicles; a **Yellow_Red** loop trails the stop-bar loop.

    python src/features_v3.py --windows mixed --out func_yr_extra.parquet
    python src/features_v3.py --windows shortb --lag-out det_lag_B.parquet ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, FEATURES, connect  # noqa: E402
from features import WINDOW_SETS, load_chunk, log  # noqa: E402

LAG_MAX = 8.0          # seconds either side
LAG_BUCKET = 8.0       # bucket width for the equi-join
MAX_ON_PER_DET = 4000  # sub-sample cap per detector per window (bounds the join)
TOPK_LAG = 12          # neighbours kept per detector


# --------------------------------------------------------------- cycle table
def load_cycles(con, devs: list[str]) -> None:
    """Full four-colour cycle table (features.load_chunk drops redclr_end)."""
    dev_list = ",".join("'" + d + "'" for d in devs)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc4_all AS
        SELECT m.dev, c.Phase::SMALLINT AS p, c.cyc::INT AS cyc,
               epoch_ms(c.green_start)/1000.0 AS gs,
               epoch_ms(coalesce(c.yellow_start, c.red_start, c.next_green))/1000.0 AS ge,
               epoch_ms(coalesce(c.red_start, c.yellow_start, c.next_green))/1000.0 AS rs,
               epoch_ms(coalesce(c.redclr_end, c.red_start, c.yellow_start,
                                 c.next_green))/1000.0 AS rce,
               epoch_ms(c.next_green)/1000.0 AS ng
        FROM read_parquet('{(CACHE/'phase_cycles.parquet').as_posix()}') c
        JOIN devmap m USING (DeviceId)
        WHERE c.Phase BETWEEN 1 AND 16 AND c.DeviceId IN ({dev_list})""")


def window_cycles(con, w0: float, w1: float) -> None:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc4 AS SELECT * FROM cyc4_all
        WHERE gs >= {w0 - 900} AND gs < {w1}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc4w AS
        SELECT *, greatest(ge - gs, 0) AS g_secs, greatest(rs - ge, 0) AS y_secs,
               greatest(rce - rs, 0) AS rc_secs,
               greatest(least(ng, {w1}) - rce, 0) AS rr_secs
        FROM cyc4_all WHERE gs >= {w0} AND gs < {w1}""")


SQL_YR = f"""
WITH oc AS (
  SELECT o.dev, o.det, o.t_on, o.dur, c.p
  FROM onev o JOIN cand c USING (dev)
), j AS (
  SELECT oc.dev, oc.det, oc.p, oc.dur, y.cyc,
         (CASE WHEN oc.t_on <  y.ge  THEN 0
               WHEN oc.t_on <  y.rs  THEN 1
               WHEN oc.t_on <  y.rce THEN 2
               ELSE 3 END)::UTINYINT AS st4,
         (oc.t_on - y.ge)::FLOAT  AS d_ge,     -- time since begin yellow
         (oc.t_on - y.rce)::FLOAT AS d_rce,    -- time since end of red clearance
         (y.ge - oc.t_on)::FLOAT  AS to_ge     -- time left of green
  FROM oc ASOF LEFT JOIN cyc4 y ON oc.dev = y.dev AND oc.p = y.p AND oc.t_on >= y.gs
), agg AS (
  SELECT dev, det, p, count(*) AS n_tot,
    count(*) FILTER (st4=0) AS n_g,
    count(*) FILTER (st4=1) AS n_y,
    count(*) FILTER (st4=2) AS n_rc,
    count(*) FILTER (st4=3) AS n_rr,
    count(*) FILTER (st4=3 AND d_rce < 2)  AS n_rr2,
    count(*) FILTER (st4=3 AND d_rce < 5)  AS n_rr5,
    sum(dur) FILTER (st4=1) AS occ_y,
    sum(dur) FILTER (st4=2) AS occ_rc,
    sum(dur) AS occ_tot,
    avg(dur) FILTER (st4=1) AS dur_y,
    avg(dur) FILTER (st4=2 OR st4=1) AS dur_yr,
    avg(dur) FILTER (st4=0) AS dur_g,
    median(d_ge) FILTER (d_ge >= 0 AND d_ge < 12) AS yr_lag_med,
    count(DISTINCT cyc) FILTER (st4=1) AS ncyc_y,
    count(DISTINCT cyc) FILTER (st4=2) AS ncyc_rc,
    count(DISTINCT cyc) FILTER (st4=1 OR st4=2) AS ncyc_yr,
    count(DISTINCT cyc) FILTER (st4=0) AS ncyc_g,
    median(to_ge) FILTER (st4=0) AS med_to_green_end
  FROM j GROUP BY 1,2,3
), sec AS (
  SELECT dev, p, count(*) AS n_cyc, sum(g_secs) AS g_secs, sum(y_secs) AS y_secs,
         sum(rc_secs) AS rc_secs, sum(rr_secs) AS rr_secs
  FROM cyc4w GROUP BY 1,2
)
SELECT a.dev, a.det, a.p,
  a.n_y  / nullif(a.n_tot,0)::DOUBLE                       AS yr_f_on_yellow,
  a.n_rc / nullif(a.n_tot,0)::DOUBLE                       AS yr_f_on_redclr,
  (a.n_y + a.n_rc) / nullif(a.n_tot,0)::DOUBLE             AS yr_f_on_yr,
  a.n_rr2 / nullif(a.n_tot,0)::DOUBLE                      AS yr_f_on_red2,
  a.n_rr5 / nullif(a.n_tot,0)::DOUBLE                      AS yr_f_on_red5,
  (a.n_y + a.n_rc + a.n_rr5) / nullif(a.n_tot,0)::DOUBLE   AS yr_f_on_yr5,
  a.occ_y  / nullif(a.occ_tot,0)                           AS yr_f_occ_yellow,
  a.occ_rc / nullif(a.occ_tot,0)                           AS yr_f_occ_redclr,
  (a.n_y / nullif(s.y_secs,0)) /
      nullif(a.n_g / nullif(s.g_secs,0), 0)                AS yr_lift_yellow,
  (a.n_rc / nullif(s.rc_secs,0)) /
      nullif(a.n_g / nullif(s.g_secs,0), 0)                AS yr_lift_redclr,
  ((a.n_y + a.n_rc) / nullif(s.y_secs + s.rc_secs,0)) /
      nullif(a.n_g / nullif(s.g_secs,0), 0)                AS yr_lift_yr,
  (a.n_rr / nullif(s.rr_secs,0)) /
      nullif(a.n_g / nullif(s.g_secs,0), 0)                AS yr_lift_redrest,
  (a.n_y + a.n_rc) / nullif(s.n_cyc,0)::DOUBLE             AS yr_on_per_cycle_yr,
  a.ncyc_y  / nullif(s.n_cyc,0)::DOUBLE                    AS yr_hit_yellow,
  a.ncyc_rc / nullif(s.n_cyc,0)::DOUBLE                    AS yr_hit_redclr,
  a.ncyc_yr / nullif(s.n_cyc,0)::DOUBLE                    AS yr_hit_yr,
  a.ncyc_yr / nullif(a.ncyc_g,0)::DOUBLE                   AS yr_hit_yr_over_g,
  a.yr_lag_med                                             AS yr_lag_med,
  a.dur_y                                                  AS yr_dur_yellow,
  a.dur_yr / nullif(a.dur_g,0)                             AS yr_dur_yr_over_g,
  a.med_to_green_end                                       AS yr_med_to_green_end,
  s.y_secs / nullif(s.g_secs,0)                            AS yr_ctx_yellow_share,
  s.rc_secs / nullif(s.g_secs,0)                           AS yr_ctx_redclr_share
FROM agg a LEFT JOIN sec s ON a.dev = s.dev AND a.p = s.p
"""


# cross-correlogram: lags binned at LAG_BIN s over [-LAG_MAX, +LAG_MAX], the peak of a
# 3-bin smoothed histogram, chance-corrected (uniform expectation = n_m / NBIN per bin).
LAG_BIN = 0.5
NBIN = int(2 * LAG_MAX / LAG_BIN)

SQL_LAG = f"""
WITH sub AS (
  SELECT dev, det, t_on FROM (
    SELECT dev, det, t_on,
           row_number() OVER (PARTITION BY dev, det ORDER BY t_on) AS rn,
           count(*) OVER (PARTITION BY dev, det) AS n
    FROM onev)
  WHERE n <= {MAX_ON_PER_DET}
     OR rn % ((n / {MAX_ON_PER_DET})::BIGINT + 1) = 0
), na AS (
  SELECT dev, det, count(*) AS n_a FROM sub GROUP BY 1,2
), a AS (
  SELECT dev, det, t_on,
         unnest([(t_on/{LAG_BUCKET})::BIGINT - 1, (t_on/{LAG_BUCKET})::BIGINT,
                 (t_on/{LAG_BUCKET})::BIGINT + 1]) AS bk
  FROM sub
), b AS (
  SELECT dev, det, t_on, (t_on/{LAG_BUCKET})::BIGINT AS bk FROM sub
), h AS (
  SELECT a.dev, a.det, b.det AS oth,
         floor((b.t_on - a.t_on) / {LAG_BIN})::INT AS lb, count(*) AS c
  FROM a JOIN b ON a.dev = b.dev AND a.bk = b.bk AND a.det <> b.det
  WHERE abs(b.t_on - a.t_on) <= {LAG_MAX}
  GROUP BY 1,2,3,4
), h3 AS (
  SELECT *, sum(c) OVER (PARTITION BY dev, det, oth ORDER BY lb
                         RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS c3
  FROM h
), tot AS (
  SELECT dev, det, oth, sum(c) AS n_m,
         sum(c) FILTER (lb IN (-1, 0))        AS n05,
         sum(c) FILTER (lb BETWEEN -2 AND 1)  AS n1,
         sum(c) FILTER (lb >= 0)              AS n_after
  FROM h GROUP BY 1,2,3
), pk AS (
  SELECT dev, det, oth, lb AS pk_lb, c3 AS pk_c3 FROM h3
  -- `lb` last: a perfectly symmetric correlogram must not pick its sign at random
  QUALIFY row_number() OVER (PARTITION BY dev, det, oth
                             ORDER BY c3 DESC, abs(lb), lb) = 1
)
SELECT t.dev, t.det, t.oth,
       (pk.pk_lb + 0.5) * {LAG_BIN}                                   AS lag_peak,
       (pk.pk_c3 - 3.0 * t.n_m / {NBIN}) / nullif(na.n_a, 0)::DOUBLE  AS lag_peak_excess,
       (pk.pk_c3 - 3.0 * t.n_m / {NBIN}) /
           nullif(sqrt(3.0 * t.n_m / {NBIN}), 0)                      AS lag_peak_z,
       (t.n05 - 2.0 * t.n_m / {NBIN}) / nullif(na.n_a, 0)::DOUBLE     AS coinc_05_excess,
       (t.n1  - 4.0 * t.n_m / {NBIN}) / nullif(na.n_a, 0)::DOUBLE     AS coinc_1_excess,
       t.n05 / nullif(na.n_a, 0)::DOUBLE                              AS coinc_05,
       t.n1  / nullif(na.n_a, 0)::DOUBLE                              AS coinc_1,
       t.n_after / nullif(t.n_m, 0)::DOUBLE                           AS frac_after,
       na.n_a, nb.n_a AS n_b, t.n_m
FROM tot t JOIN pk  ON t.dev = pk.dev AND t.det = pk.det AND t.oth = pk.oth
           JOIN na  ON t.dev = na.dev AND t.det = na.det
           JOIN na nb ON t.dev = nb.dev AND t.oth = nb.det
QUALIFY row_number() OVER (PARTITION BY t.dev, t.det
                           ORDER BY (pk.pk_c3 - 3.0 * t.n_m / {NBIN}) /
                                    nullif(na.n_a, 0)::DOUBLE DESC,
                                    t.oth) <= {TOPK_LAG}
"""


def build(con, win: str, devmap: pd.DataFrame, sql: str) -> pd.DataFrame:
    df = con.sql(sql).df()
    if not len(df):
        return df
    df = df.merge(devmap, on="dev", how="left").drop(columns=["dev"])
    df["win"] = win
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--windows", default="mixed")
    ap.add_argument("--out", default="func_yr_extra.parquet")
    ap.add_argument("--lag-out", default="det_lag.parquet")
    ap.add_argument("--what", default="both", choices=["both", "yr", "lag"])
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    import function_v3_prep  # noqa: F401  (registers the 'shortb' window set)
    FEATURES.mkdir(parents=True, exist_ok=True)
    con = connect(threads=a.threads)
    devs = con.sql(f"SELECT DeviceId FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}')"
                   " ORDER BY DeviceId").df()["DeviceId"].tolist()
    if a.limit:
        devs = devs[:a.limit]
    wins = WINDOW_SETS[a.windows]
    log(f"features_v3 [{a.what}]: {len(devs)} signals x {len(wins)} windows")
    yr_parts, lag_parts, t0 = [], [], time.time()
    nch = (len(devs) + a.chunk - 1) // a.chunk
    for i in range(0, len(devs), a.chunk):
        ch = devs[i:i + a.chunk]
        t1 = time.time()
        load_chunk(con, ch)
        load_cycles(con, ch)
        dm = con.sql("SELECT * FROM devmap").df()
        for w in wins:
            w0 = (w["t0"] - pd.Timestamp("1970-01-01")).total_seconds()
            w1 = w0 + w["secs"]
            con.execute(f"CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM onev_all "
                        f"WHERE t_on >= {w0} AND t_on < {w1}")
            window_cycles(con, w0, w1)
            if a.what in ("both", "yr"):
                r = build(con, w["win"], dm, SQL_YR)
                if len(r):
                    yr_parts.append(r)
            if a.what in ("both", "lag"):
                r = build(con, w["win"], dm, SQL_LAG)
                if len(r):
                    lag_parts.append(r)
        log(f"chunk {i//a.chunk+1}/{nch} {time.time()-t1:.1f}s elapsed={time.time()-t0:.0f}s")

    if yr_parts:
        df = pd.concat(yr_parts, ignore_index=True)
        df = df.rename(columns={"det": "Detector", "p": "cand_phase"})
        df = df.replace([np.inf, -np.inf], np.nan)
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        df["Detector"] = df.Detector.astype(np.int16)
        df["cand_phase"] = df.cand_phase.astype(np.int16)
        df.to_parquet(FEATURES / a.out, index=False)
        log(f"wrote {FEATURES / a.out}: {len(df):,} rows x {df.shape[1]} cols")
    if lag_parts:
        dl = pd.concat(lag_parts, ignore_index=True)
        dl = dl.rename(columns={"det": "Detector", "oth": "other"})
        dl = dl.replace([np.inf, -np.inf], np.nan)
        for c in dl.columns:
            if dl[c].dtype == np.float64:
                dl[c] = dl[c].astype(np.float32)
        dl["Detector"] = dl.Detector.astype(np.int16)
        dl["other"] = dl.other.astype(np.int16)
        dl.to_parquet(FEATURES / a.lag_out, index=False)
        log(f"wrote {FEATURES / a.lag_out}: {len(dl):,} rows")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
