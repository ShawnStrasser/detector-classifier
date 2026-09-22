"""Stage 08 step 1 -- **demand-contrast** raw counters.

Idea (from an outside rule-based researcher): a detector's occupancy-by-colour behaviour
*changes with demand*, and that change is what separates the functions.  A stop-bar
Presence loop is occupied through red at 3 am and at 5 pm alike; an Advance loop is
pulse-like off-peak but gets sat on in the peak, and only in the peak, once the queue
reaches it; a Count loop's ON time is nearly invariant; a Yellow_Red loop is never queued
at any demand.  Our existing features pool the whole window, so none of this is visible.

This module emits, for every (DeviceId, Detector, cand_phase, win, demand level), the raw
*numerators and denominators* of the key occupancy / timing features, so that
`contrast_features.py` can form shrunk low/high values, differences, ratios and slopes.

Demand levels, two definitions, both computed in the same pass:
  * `q` -- **data driven**: the window is cut into 15 min bins; the bins of one signal are
    ranked by total actuation count over all of that signal's detectors and split into
    terciles low / mid / high.  Agency-, day- and timezone-agnostic.
  * `c` -- **clock**: overnight 00-05 and evening 19-23 = low, midday 10-14 = mid,
    06:30-09 and 15:30-18:30 = high; the shoulders are dropped.

Occupancy is computed twice: exactly (from the ON/OFF interval) and **snapped to 15 s
bins** (`*_b` columns) -- the `atspm` split-failure style "the bin is occupied if the
detector was on at any point in it".  That is the cheap test of whether binned occupancy
beats exact-interval occupancy (stage 08 task 4).

Phase-anonymous: DeviceId / Detector / cand_phase / win / lvl are keys only.

    python src/contrast/build_contrast.py --out contrast_raw.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CACHE, DC_WORK, connect  # noqa: E402
from build_features import load_chunk  # noqa: E402

WORK = DC_WORK / "contrast"
WORK.mkdir(parents=True, exist_ok=True)

BIN = 900.0        # demand bin, seconds (15 min)
SNAP = 15.0        # occupancy quantisation for the "binned occupancy" variant

# only windows of >= 24 h get contrast features (the "extended" model)
LONG_WINDOWS = [
    {"win": "h24_a", "t0": "2024-12-02 00:00:00", "secs": 24 * 3600},
    {"win": "h24_b", "t0": "2024-12-04 00:00:00", "secs": 24 * 3600},
    {"win": "full72", "t0": "2024-12-02 00:00:00", "secs": 72 * 3600},
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ demand bins
def make_bins(con, w0: float, w1: float) -> None:
    """One row per (dev, bin): volume, tercile level `lq`, clock level `lc`."""
    nb = int(round((w1 - w0) / BIN))
    con.execute(f"""CREATE OR REPLACE TEMP TABLE binv AS
    WITH d AS (SELECT DISTINCT dev FROM onev_w),
         allb AS (SELECT d.dev, r.b::INT AS b FROM d, range(0,{nb}) r(b)),
         v AS (SELECT dev, ((t_on - {w0})/{BIN})::INT AS b, count(*) AS v
               FROM onev_w GROUP BY 1,2)
    SELECT a.dev, a.b, coalesce(v.v,0)::DOUBLE AS vol,
           (ntile(3) OVER (PARTITION BY a.dev ORDER BY coalesce(v.v,0), a.b) - 1)::TINYINT AS lq,
           (CASE
              WHEN ((a.b*{BIN}+{w0})::BIGINT % 86400) <  5*3600 THEN 0
              WHEN ((a.b*{BIN}+{w0})::BIGINT % 86400) >= 19*3600
               AND ((a.b*{BIN}+{w0})::BIGINT % 86400) <  23*3600 THEN 0
              WHEN ((a.b*{BIN}+{w0})::BIGINT % 86400) >= 10*3600
               AND ((a.b*{BIN}+{w0})::BIGINT % 86400) <  14*3600 THEN 1
              WHEN ((a.b*{BIN}+{w0})::BIGINT % 86400) >= 23400
               AND ((a.b*{BIN}+{w0})::BIGINT % 86400) <   9*3600 THEN 2
              WHEN ((a.b*{BIN}+{w0})::BIGINT % 86400) >= 55800
               AND ((a.b*{BIN}+{w0})::BIGINT % 86400) <  66600 THEN 2
              ELSE NULL END)::TINYINT AS lc
    FROM allb a LEFT JOIN v ON v.dev=a.dev AND v.b=a.b""")


def prepare_window(con, w0: float, w1: float) -> None:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE onev_w AS
        SELECT dev, det, t_on, t_off, coalesce(dur,0.0) AS dur,
               (ceil(t_off/{SNAP})*{SNAP} - floor(t_on/{SNAP})*{SNAP}) AS durb
        FROM onev_all WHERE t_on >= {w0} AND t_on < {w1}""")
    make_bins(con, w0, w1)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE onev AS
        SELECT o.*, b.lq, b.lc FROM onev_w o
        JOIN binv b ON b.dev=o.dev AND b.b = ((o.t_on - {w0})/{BIN})::INT""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc AS SELECT * FROM cyc_all
        WHERE gs >= {w0 - 900} AND gs < {w1}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cycw AS
        SELECT c.*, b.lq, b.lc FROM cyc_all c
        JOIN binv b ON b.dev=c.dev AND b.b = ((c.gs - {w0})/{BIN})::INT
        WHERE c.gs >= {w0} AND c.gs < {w1}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE callsw AS
        SELECT * FROM calls_all WHERE ev=43 AND t >= {w0} AND t < {w1} + 10""")
    con.execute("""CREATE OR REPLACE TEMP TABLE j AS
    SELECT oc.dev, oc.det, oc.p, oc.lq, oc.lc, oc.dur, oc.durb,
           (oc.t_on - y.gs)::FLOAT AS dtg, (y.ng - oc.t_on)::FLOAT AS tog,
           (CASE WHEN oc.t_on < y.ge THEN 0 WHEN oc.t_on < y.rs THEN 1
                 ELSE 2 END)::UTINYINT AS st,
           -- properly CLIPPED overlap of the ON interval with this cycle's green / red,
           -- and the same divided by the cycle's own green / red length (per-cycle
           -- occupancy, the statistic the atspm split-failure query aggregates)
           greatest(least(oc.t_off, y.ge) - greatest(oc.t_on, y.gs), 0)::FLOAT AS og,
           greatest(least(oc.t_off, least(y.ng, y.gs+600)) -
                    greatest(oc.t_on, y.ge), 0)::FLOAT AS orr,
           greatest(y.ge - y.gs, 0)::FLOAT AS gsec,
           greatest(least(y.ng, y.gs+600) - y.ge, 0)::FLOAT AS rsec
    FROM (SELECT o.dev,o.det,o.t_on,o.t_off,o.dur,o.durb,o.lq,o.lc,c.p
          FROM onev o JOIN cand c USING (dev)) oc
    ASOF LEFT JOIN cyc y ON oc.dev=y.dev AND oc.p=y.p AND oc.t_on >= y.gs""")


# --------------------------------------------------------------- aggregations
def _sql_pair(L: str) -> str:
    return f"""
SELECT dev, det, p, {L} AS lvl,
  count(*)                                             AS n_on_pair,
  count(*) FILTER (st=0)                               AS n_g,
  count(*) FILTER (st=2)                               AS n_r,
  sum(dur)  FILTER (st=0)                              AS occ_g,
  sum(dur)  FILTER (st=2)                              AS occ_r,
  sum(durb) FILTER (st=0)                              AS occ_gb,
  sum(durb) FILTER (st=2)                              AS occ_rb,
  count(*) FILTER (st=0 AND dtg < 2)                   AS n_dtg2,
  count(*) FILTER (st=0 AND dtg < 4)                   AS n_g4,
  count(*) FILTER (st=2 AND tog <= 10)                 AS n_red10,
  sum(dur) FILTER (st=2 AND tog <= 8)                  AS q_sum,
  count(*) FILTER (st=2 AND tog <= 8)                  AS q_n,
  sum(durb) FILTER (st=2 AND tog <= 8)                 AS q_sumb,
  sum(og)                                              AS occ_gc,
  sum(orr)                                             AS occ_rc,
  sum(og  / nullif(gsec,0))                            AS mg_sum,
  sum(orr / nullif(rsec,0))                            AS mr_sum
FROM j WHERE {L} IS NOT NULL GROUP BY 1,2,3,4"""


def _sql_det(L: str) -> str:
    return f"""
SELECT dev, det, {L} AS lvl,
  count(*) AS d_n, sum(dur) AS d_occ, sum(durb) AS d_occb,
  quantile_cont(dur,0.5)  AS d_med,  quantile_cont(dur,0.9)  AS d_q90,
  quantile_cont(durb,0.5) AS d_medb, quantile_cont(durb,0.9) AS d_q90b,
  count(*) FILTER (dur > 5)   AS d_nlong,
  count(*) FILTER (dur < 0.5) AS d_nshort
FROM onev WHERE {L} IS NOT NULL GROUP BY 1,2,3"""


def _sql_cyc(L: str) -> str:
    return f"""
SELECT dev, p, {L} AS lvl, count(*) AS c_ncyc,
  sum(greatest(ge-gs,0)) AS c_gsecs,
  sum(greatest(least(ng,gs+600)-ge,0)) AS c_rsecs
FROM cycw WHERE {L} IS NOT NULL GROUP BY 1,2,3"""


def _sql_sig(L: str) -> str:
    return f"""
SELECT dev, {L} AS lvl, count(*)*{BIN} AS s_secs, sum(vol) AS s_vol
FROM binv WHERE {L} IS NOT NULL GROUP BY 1,2"""


def _sql_str(L: str) -> str:
    """Long occupancies that end just after this phase's begin green = the queue it released,
    and the share of ONs immediately followed by a phase call (43) on this phase."""
    return f"""
WITH lo AS (SELECT o.dev,o.det,o.t_on,o.t_off,o.{L} AS lvl, c.p
            FROM onev o JOIN cand c USING (dev) WHERE o.dur > 1.5 AND o.{L} IS NOT NULL),
     j2 AS (SELECT lo.*, y.gs FROM lo ASOF LEFT JOIN cyc y
            ON lo.dev=y.dev AND lo.p=y.p AND lo.t_off >= y.gs)
SELECT dev, det, p, lvl,
  count(*) AS n_long15,
  count(*) FILTER (gs IS NOT NULL AND gs > t_on AND t_off - gs <= 6) AS n_str
FROM j2 GROUP BY 1,2,3,4"""


def _sql_call(L: str) -> str:
    return f"""
WITH oc AS (SELECT o.dev,o.det,o.t_on,o.{L} AS lvl, c.p
            FROM onev o JOIN cand c USING (dev) WHERE o.{L} IS NOT NULL),
     a AS (SELECT oc.*, k.t AS t43 FROM oc ASOF LEFT JOIN callsw k
           ON oc.dev=k.dev AND oc.p=k.p AND oc.t_on <= k.t)
SELECT dev, det, p, lvl, count(*) AS n_call_den,
       count(*) FILTER (t43 - t_on <= 0.35) AS n_c43
FROM a GROUP BY 1,2,3,4"""


def build_one(con, tag: str) -> pd.DataFrame:
    L = "lq" if tag == "q" else "lc"
    a = con.sql(_sql_pair(L)).df()
    if not len(a):
        return a
    d = con.sql(_sql_det(L)).df()
    c = con.sql(_sql_cyc(L)).df()
    s = con.sql(_sql_sig(L)).df()
    st = con.sql(_sql_str(L)).df()
    ca = con.sql(_sql_call(L)).df()
    out = a.merge(d, on=["dev", "det", "lvl"], how="left")
    out = out.merge(c, on=["dev", "p", "lvl"], how="left")
    out = out.merge(s, on=["dev", "lvl"], how="left")
    out = out.merge(st, on=["dev", "det", "p", "lvl"], how="left")
    out = out.merge(ca, on=["dev", "det", "p", "lvl"], how="left")
    out["defn"] = tag
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="contrast_raw.parquet")
    a = ap.parse_args()
    con = connect(memory_limit="6GB", threads=a.threads)
    folds = pd.read_csv(DC_WORK / "function_v3" / "folds_v3.csv")
    have = {p.name.split("=", 1)[1] for p in (CACHE / "events").iterdir() if p.is_dir()}
    devs = sorted(set(folds.DeviceId) & have)
    if a.limit:
        devs = devs[:a.limit]
    log(f"{len(devs)} DEV signals x {len(LONG_WINDOWS)} long windows")
    parts, t0 = [], time.time()
    nch = (len(devs) + a.chunk - 1) // a.chunk
    for i in range(0, len(devs), a.chunk):
        ch = devs[i:i + a.chunk]
        t1 = time.time()
        load_chunk(con, ch)
        devmap = con.sql("SELECT * FROM devmap").df()
        for w in LONG_WINDOWS:
            w0 = (pd.Timestamp(w["t0"]) - pd.Timestamp("1970-01-01")).total_seconds()
            prepare_window(con, w0, w0 + w["secs"])
            for tag in ("q", "c"):
                r = build_one(con, tag)
                if len(r):
                    r["win"] = w["win"]
                    parts.append(r.merge(devmap, on="dev", how="left").drop(columns=["dev"]))
        log(f"chunk {i//a.chunk+1}/{nch} {time.time()-t1:.1f}s elapsed={time.time()-t0:.0f}s")
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df.rename(columns={"det": "Detector", "p": "cand_phase"})
    df["Detector"] = df.Detector.astype(np.int16)
    df["cand_phase"] = df.cand_phase.astype(np.int16)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    out = WORK / a.out
    df.to_parquet(out, index=False)
    log(f"wrote {out}: {len(df):,} rows x {df.shape[1]} cols in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
