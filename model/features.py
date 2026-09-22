"""Phase-anonymous detector/phase pair features -- the model's main input.

One row per (DeviceId, Detector, cand_phase, win) for every *active* detector channel
and every candidate phase of that signal: how the detector behaves relative to that
phase's green, yellow and red, whether a phase call follows its actuations, how it
behaves when that phase is green and its usual partner is not.

No phase number, detector channel number or label-derived quantity is ever a feature;
`DeviceId`, `Detector`, `cand_phase`, `win` are keys only.  All features are rates /
shares / lifts (duration invariant); the amount of evidence is exposed explicitly
through `win_secs`, `det_n_on`, `det_on_per_hour`, `n_cycles`, so a 30-minute and a
3-day sample look the same to the model.

`predict.py` calls `apply_window` -> `build_window` -> `finalise` on temp tables it
builds itself from raw events.  The research driver that writes the training tables
from the event cache is `research/code/features/build_features.py`.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

KEYS = ["DeviceId", "Detector", "cand_phase", "win"]

# features that get a within-(signal, detector, window) rank / z-score companion
RANK_FEATS = [
    "on_lift_green", "occ_lift_green", "f_on_green", "f_occ_green",
    "release_frac", "release_frac_long", "straddle_frac",
    "call43_fwd_lift", "call43_rev_frac", "call44_fwd_lift", "call43_rev_onnow",
    "excl_lift_min", "excl_lift_mean", "excl_diff_min", "excl_diff_mean",
    "excl_partner_diff", "queue_occ_pre_green", "burst_rate_g4", "ext_corr_late",
    "f_on_red_last10", "dtg_b0", "dtr_b4", "tog_b0", "late_green_rate",
    "on_lift_green_coord", "on_lift_green_free", "solo_lift", "ext_green_gain",
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def apply_window(con, w0: float, w1: float) -> None:
    """Restrict the chunk tables to [w0, w1) and rebuild the mask aggregates."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM onev_all "
                f"WHERE t_on >= {w0} AND t_on < {w1}")
    # cycles: keep a lookback so an ON early in the window finds its green
    con.execute(f"CREATE OR REPLACE TEMP TABLE cyc AS SELECT * FROM cyc_all "
                f"WHERE gs >= {w0 - 900} AND gs < {w1}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE cyc_win AS SELECT * FROM cyc_all "
                f"WHERE gs >= {w0} AND gs < {w1}")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE gs AS
        SELECT dev, greatest(t0,{w0}) AS t0, least(t1,{w1}) AS t1, mask
        FROM gs_all WHERE t1 > {w0} AND t0 < {w1}""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE calls AS SELECT * FROM calls_all "
                f"WHERE t >= {w0} AND t < {w1} + 10")
    con.execute("""CREATE OR REPLACE TEMP TABLE onmask AS
        WITH a AS (
          SELECT o.dev, o.det, o.t_on, o.dur, coalesce(g.mask, 0) AS mask
          FROM onev o ASOF LEFT JOIN gs g ON o.dev = g.dev AND o.t_on >= g.t0
        )
        SELECT a.dev, a.det, a.mask, coalesce(c.is_coord, false) AS is_coord,
               count(*) AS n_on, sum(a.dur) AS occ
        FROM a ASOF LEFT JOIN coordiv c ON a.dev = c.dev AND a.t_on >= c.t0
        GROUP BY 1,2,3,4""")
    con.execute("""CREATE OR REPLACE TEMP TABLE masktime AS
        SELECT g.dev, g.mask, coalesce(c.is_coord, false) AS is_coord, sum(g.t1 - g.t0) AS secs
        FROM gs g ASOF LEFT JOIN coordiv c ON g.dev = c.dev AND g.t0 >= c.t0
        GROUP BY 1,2,3""")


# --------------------------------------------------------------------- SQL
SQL_STATE = """
CREATE OR REPLACE TEMP TABLE j AS
SELECT oc.dev, oc.det, oc.p, oc.dur, y.cyc, y.green_secs,
       (oc.t_on - y.gs)::FLOAT AS dtg,
       (oc.t_on - y.rs)::FLOAT AS dtr,
       (y.ng - oc.t_on)::FLOAT AS tog,
       (y.ge - oc.t_on)::FLOAT AS to_end_green,
       (CASE WHEN oc.t_on < y.ge THEN 0 WHEN oc.t_on < y.rs THEN 1 ELSE 2 END)::UTINYINT AS st
FROM (SELECT o.dev, o.det, o.t_on, o.dur, c.p FROM onev o JOIN cand c USING (dev)) oc
ASOF LEFT JOIN cyc y ON oc.dev = y.dev AND oc.p = y.p AND oc.t_on >= y.gs
"""

SQL_STATE_AGG = """
SELECT dev, det, p,
  count(*)                                         AS n_on_pair,
  avg(CASE WHEN st=0 THEN 1 ELSE 0 END)            AS f_on_green,
  avg(CASE WHEN st=1 THEN 1 ELSE 0 END)            AS f_on_yellow,
  avg(CASE WHEN st=2 THEN 1 ELSE 0 END)            AS f_on_red,
  sum(CASE WHEN st=0 THEN dur ELSE 0 END)/nullif(sum(dur),0) AS f_occ_green,
  sum(CASE WHEN st=1 THEN dur ELSE 0 END)/nullif(sum(dur),0) AS f_occ_yellow,
  sum(CASE WHEN st=2 THEN dur ELSE 0 END)/nullif(sum(dur),0) AS f_occ_red,
  avg(dur) FILTER (st=0)                           AS dur_mean_green,
  avg(dur) FILTER (st=2)                           AS dur_mean_red,
  avg(CASE WHEN dtg <  2 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b0,
  avg(CASE WHEN dtg >= 2 AND dtg <  5 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b1,
  avg(CASE WHEN dtg >= 5 AND dtg < 10 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b2,
  avg(CASE WHEN dtg >=10 AND dtg < 20 THEN 1 ELSE 0 END) FILTER (st=0) AS dtg_b3,
  avg(CASE WHEN dtg >=20 THEN 1 ELSE 0 END)              FILTER (st=0) AS dtg_b4,
  avg(CASE WHEN dtr <  5 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b0,
  avg(CASE WHEN dtr >= 5 AND dtr < 15 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b1,
  avg(CASE WHEN dtr >=15 AND dtr < 30 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b2,
  avg(CASE WHEN dtr >=30 AND dtr < 60 THEN 1 ELSE 0 END) FILTER (st=2) AS dtr_b3,
  avg(CASE WHEN dtr >=60 THEN 1 ELSE 0 END)              FILTER (st=2) AS dtr_b4,
  avg(CASE WHEN tog <  3 THEN 1 ELSE 0 END) FILTER (st=2) AS tog_b0,
  avg(CASE WHEN tog >= 3 AND tog < 10 THEN 1 ELSE 0 END) FILTER (st=2) AS tog_b1,
  avg(CASE WHEN tog >=10 AND tog < 30 THEN 1 ELSE 0 END) FILTER (st=2) AS tog_b2,
  avg(CASE WHEN tog >=30 THEN 1 ELSE 0 END)              FILTER (st=2) AS tog_b3,
  avg(dur) FILTER (st=2 AND tog <= 8)              AS queue_occ_pre_green,
  avg(CASE WHEN st=2 AND tog <= 10 THEN 1 ELSE 0 END) AS f_on_red_last10,
  count(*) FILTER (st=0 AND dtg < 4)               AS n_on_g4,
  count(*) FILTER (st=0 AND dtg < 8)               AS n_on_g8,
  count(*) FILTER (st=0 AND to_end_green <= 3)     AS n_on_late_green,
  avg(dur) FILTER (st=0 AND dtg < 4)               AS dur_mean_g4
FROM j GROUP BY 1,2,3
"""

SQL_CYC = """
WITH ca AS (
  SELECT dev, det, p, cyc, any_value(green_secs) AS g,
         count(*) FILTER (dtg >= 3) AS n_late,
         max(CASE WHEN to_end_green <= 2 THEN 1 ELSE 0 END) AS late2
  FROM j WHERE st = 0 AND green_secs IS NOT NULL GROUP BY 1,2,3,4
), tot AS (
  SELECT dev, p, count(*) AS n_cyc, sum(green_secs) AS sg, sum(green_secs*green_secs) AS sgg
  FROM cyc_win WHERE green_secs IS NOT NULL GROUP BY 1,2
), agg AS (
  SELECT dev, det, p, count(*) AS n_cyc_act,
         sum(n_late) AS sn, sum(n_late*n_late) AS snn, sum(n_late*g) AS sng,
         sum(CASE WHEN late2=1 THEN g ELSE 0 END) AS sg_late2, sum(late2) AS n_late2
  FROM ca GROUP BY 1,2,3
)
SELECT a.dev, a.det, a.p,
       a.n_cyc_act / nullif(t.n_cyc,0)::DOUBLE AS cyc_active_frac,
       (t.n_cyc*a.sng - a.sn*t.sg) /
         nullif(sqrt(greatest(t.n_cyc*a.snn - a.sn*a.sn,0)) *
                sqrt(greatest(t.n_cyc*t.sgg - t.sg*t.sg,0)), 0) AS ext_corr_late,
       (a.sg_late2/nullif(a.n_late2,0)) - (t.sg/nullif(t.n_cyc,0)) AS ext_green_gain,
       a.n_late2 / nullif(t.n_cyc,0)::DOUBLE AS late2_frac
FROM agg a JOIN tot t ON t.dev=a.dev AND t.p=a.p
"""

SQL_RELEASE = """
WITH lo AS (
  SELECT o.dev, o.det, o.t_on, o.t_off, o.dur, c.p
  FROM onev o JOIN cand c USING (dev) WHERE o.dur > 1.5
), j2 AS (
  SELECT lo.*, y.gs FROM lo ASOF LEFT JOIN cyc y
    ON lo.dev=y.dev AND lo.p=y.p AND lo.t_off >= y.gs
)
SELECT dev, det, p,
  avg(CASE WHEN gs IS NOT NULL AND gs > t_on AND t_off - gs <= 6 THEN 1 ELSE 0 END) AS straddle_frac,
  avg(CASE WHEN gs IS NOT NULL AND gs > t_on AND t_off - gs <= 6 THEN 1 ELSE 0 END)
      FILTER (dur > 5)                                                              AS release_frac_long,
  avg(CASE WHEN gs IS NOT NULL AND gs > t_on THEN exp(-greatest(t_off-gs,0)/3.0) ELSE 0 END) AS release_frac,
  count(*) FILTER (dur > 5)                                                         AS n_long_on,
  count(*)                                                                          AS n_on15
FROM j2 GROUP BY 1,2,3
"""

SQL_CALL_FWD = """
WITH oc AS (SELECT o.dev, o.det, o.t_on, c.p FROM onev o JOIN cand c USING (dev)),
   c43 AS (SELECT dev, p, t FROM calls WHERE ev = 43),
   c44 AS (SELECT dev, p, t FROM calls WHERE ev = 44),
a AS (SELECT oc.*, k.t AS t43 FROM oc ASOF LEFT JOIN c43 k
        ON oc.dev=k.dev AND oc.p=k.p AND oc.t_on <= k.t),
b AS (SELECT a.*, k.t AS t44 FROM a ASOF LEFT JOIN c44 k
        ON a.dev=k.dev AND a.p=k.p AND a.t_on <= k.t)
SELECT dev, det, p,
  avg(CASE WHEN t43 - t_on <= 0.35 THEN 1 ELSE 0 END) AS call43_fwd_035,
  avg(CASE WHEN t43 - t_on <= 1.0  THEN 1 ELSE 0 END) AS call43_fwd_1,
  avg(CASE WHEN t44 - t_on <= 0.35 THEN 1 ELSE 0 END) AS call44_fwd_035,
  avg(CASE WHEN t44 - t_on <= 1.0  THEN 1 ELSE 0 END) AS call44_fwd_1
FROM b GROUP BY 1,2,3
"""

SQL_CALL_REV = """
WITH cd AS (
  SELECT k.dev, k.p, k.ev, k.t, d.det
  FROM calls k JOIN (SELECT DISTINCT dev, det FROM onev) d USING (dev)
), a AS (
  SELECT cd.*, o.t_on, o.t_off FROM cd ASOF LEFT JOIN onev o
    ON cd.dev=o.dev AND cd.det=o.det AND cd.t >= o.t_on
)
SELECT dev, det, p,
  avg(CASE WHEN t - t_on <= 0.35 THEN 1 ELSE 0 END) FILTER (ev=43) AS call43_rev_035,
  avg(CASE WHEN t - t_on <= 1.0  THEN 1 ELSE 0 END) FILTER (ev=43) AS call43_rev_frac,
  avg(CASE WHEN t_off >= t THEN 1 ELSE 0 END)       FILTER (ev=43) AS call43_rev_onnow,
  avg(CASE WHEN t - t_on <= 1.0  THEN 1 ELSE 0 END) FILTER (ev=44) AS call44_rev_frac,
  count(*) FILTER (ev=43)                                          AS n_call43
FROM a GROUP BY 1,2,3
"""

SQL_CTX = """
WITH c AS (
  SELECT dev, p, count(*) AS n_cycles, avg(green_secs) AS green_mean,
         stddev_samp(green_secs) AS green_sd, median(green_secs) AS green_med,
         sum(green_secs) AS green_total,
         avg(ng - gs) FILTER (ng IS NOT NULL AND ng - gs < 600) AS cycle_mean
  FROM cyc_win WHERE green_secs IS NOT NULL GROUP BY 1,2
), k AS (
  SELECT dev, p, count(*) FILTER (ev=43) AS n43, count(*) FILTER (ev=44) AS n44
  FROM calls GROUP BY 1,2
), tot AS (SELECT dev, sum(green_total) AS tg, max(n_cycles) AS mx FROM c GROUP BY 1)
SELECT c.dev, c.p, c.n_cycles, c.green_mean, c.green_sd, c.green_med, c.cycle_mean,
       c.green_total / nullif(t.tg,0) AS cand_green_share,
       c.n_cycles / nullif(t.mx,0)::DOUBLE AS cand_cycle_ratio,
       coalesce(k.n43,0) / nullif(c.n_cycles,0)::DOUBLE AS call43_per_cycle,
       coalesce(k.n44,0) / nullif(c.n_cycles,0)::DOUBLE AS call44_per_cycle,
       (coalesce(k.n43,0) / nullif(c.n_cycles,0)::DOUBLE < 0.25)::INT AS looks_recall,
       coalesce(k.n43,0) AS n43_win
FROM c JOIN tot t USING (dev) LEFT JOIN k USING (dev, p)
"""


# ------------------------------------------------------- mask-based features
def _mask_features(con) -> pd.DataFrame:
    om = con.sql("SELECT * FROM onmask").df()
    mt = con.sql("SELECT * FROM masktime").df()
    cand = con.sql("SELECT * FROM cand").df()
    rows = []
    for dev, cd in cand.groupby("dev"):
        phases = sorted(int(x) for x in cd.p.unique())
        mtd = mt[mt.dev == dev]
        omd = om[om.dev == dev]
        if not len(mtd) or not len(omd):
            continue
        masks = mtd["mask"].to_numpy(dtype=np.int64)
        secs = mtd["secs"].to_numpy(dtype=float)
        is_c = mtd["is_coord"].to_numpy(dtype=bool)
        T = secs.sum()
        if T <= 0:
            continue
        Tc, Tf = secs[is_c].sum(), secs[~is_c].sum()
        bits = {p: ((masks >> (p - 1)) & 1).astype(bool) for p in phases}
        t_anyg = secs[masks != 0].sum()
        for det, g in omd.groupby("det"):
            gm = g["mask"].to_numpy(dtype=np.int64)
            gc = g["is_coord"].to_numpy(dtype=bool)
            gn = g["n_on"].to_numpy(dtype=float)
            go = g["occ"].to_numpy(dtype=float)
            N, O = gn.sum(), go.sum()
            if N == 0:
                continue
            gbits = {p: ((gm >> (p - 1)) & 1).astype(bool) for p in phases}
            rate, orate = N / T, O / T
            nc, nf = gn[gc].sum(), gn[~gc].sum()
            anyg_lift = (gn[gm != 0].sum() / t_anyg) / rate if t_anyg > 5 else np.nan
            base = {}
            for p in phases:
                bp, gp = bits[p], gbits[p]
                tp = secs[bp].sum()
                solo_m = bp & (masks == (1 << (p - 1)))
                gsolo_m = gp & (gm == (1 << (p - 1)))
                solo_t = secs[solo_m].sum()
                tpc = secs[bp & is_c].sum()
                tpf = secs[bp & ~is_c].sum()
                base[p] = dict(
                    green_share=tp / T,
                    on_lift_green=(gn[gp].sum() / tp) / rate if tp > 5 else np.nan,
                    occ_lift_green=(go[gp].sum() / tp) / orate if tp > 5 and orate > 0 else np.nan,
                    on_lift_green_coord=(gn[gp & gc].sum() / tpc) / (nc / Tc)
                    if tpc > 60 and Tc > 60 and nc > 0 else np.nan,
                    on_lift_green_free=(gn[gp & ~gc].sum() / tpf) / (nf / Tf)
                    if tpf > 60 and Tf > 60 and nf > 0 else np.nan,
                    solo_lift=(gn[gsolo_m].sum() / solo_t) / rate if solo_t > 30 else np.nan,
                    solo_share=solo_t / tp if tp > 0 else np.nan,
                    anygreen_lift=anyg_lift,
                )
            for p in phases:
                lp, lq, diffs, cot = [], [], [], []
                for q in phases:
                    if q == p:
                        continue
                    m_pq = bits[p] & ~bits[q]
                    m_qp = ~bits[p] & bits[q]
                    t_pq, t_qp = secs[m_pq].sum(), secs[m_qp].sum()
                    if t_pq < 120 or t_qp < 120:
                        continue
                    l_pq = (gn[gbits[p] & ~gbits[q]].sum() / t_pq) / rate
                    l_qp = (gn[~gbits[p] & gbits[q]].sum() / t_qp) / rate
                    lp.append(l_pq); lq.append(l_qp)
                    diffs.append(np.log1p(l_pq) - np.log1p(l_qp))
                    cot.append(secs[bits[p] & bits[q]].sum())
                r = dict(dev=dev, det=det, p=p, **base[p])
                if diffs:
                    r.update(excl_lift_min=float(np.min(lp)), excl_lift_mean=float(np.mean(lp)),
                             excl_lift_max=float(np.max(lp)),
                             excl_lift_other_max=float(np.max(lq)),
                             excl_lift_other_mean=float(np.mean(lq)),
                             excl_diff_min=float(np.min(diffs)),
                             excl_diff_mean=float(np.mean(diffs)),
                             excl_partner_diff=float(diffs[int(np.argmax(cot))]),
                             n_excl_pairs=len(diffs))
                else:
                    r.update(excl_lift_min=np.nan, excl_lift_mean=np.nan, excl_lift_max=np.nan,
                             excl_lift_other_max=np.nan, excl_lift_other_mean=np.nan,
                             excl_diff_min=np.nan, excl_diff_mean=np.nan,
                             excl_partner_diff=np.nan, n_excl_pairs=0)
                rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- assembly
def build_window(con, win: str, secs: float) -> pd.DataFrame:
    con.execute(SQL_STATE)
    a = con.sql(SQL_STATE_AGG).df()
    b = con.sql(SQL_CYC).df()
    con.execute("DROP TABLE IF EXISTS j")
    c = con.sql(SQL_RELEASE).df()
    d = con.sql(SQL_CALL_FWD).df()
    e = con.sql(SQL_CALL_REV).df()
    ctx = con.sql(SQL_CTX).df()
    m = _mask_features(con)
    dm = con.sql("SELECT dev, det, count(*) n, sum(dur) o, "
                 "quantile_cont(dur,0.5) dq50, quantile_cont(dur,0.9) dq90, "
                 "avg(CASE WHEN dur<0.5 THEN 1 ELSE 0 END) fshort, "
                 "avg(CASE WHEN dur>5 THEN 1 ELSE 0 END) flong, avg(dur) dmean, "
                 "max(dur) dmax FROM onev GROUP BY 1,2").df()
    if not len(m):
        return pd.DataFrame()
    out = m.merge(a, on=["dev", "det", "p"], how="outer")
    for t in (b, c, d, e):
        out = out.merge(t, on=["dev", "det", "p"], how="left")
    out = out.merge(ctx, on=["dev", "p"], how="left")
    out = out.merge(dm.rename(columns={"n": "det_n_on", "o": "det_occ", "dq50": "det_dur_med",
                                       "dq90": "det_dur_q90", "fshort": "det_frac_short",
                                       "flong": "det_frac_long", "dmean": "det_dur_mean",
                                       "dmax": "det_dur_max"}),
                    on=["dev", "det"], how="left")
    devmap = con.sql("SELECT * FROM devmap").df()
    out = out.merge(devmap, on="dev", how="left").drop(columns=["dev"])
    out = out.rename(columns={"det": "Detector", "p": "cand_phase"})
    out["win"] = win
    out["win_secs"] = secs
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    hrs = df["win_secs"] / 3600.0
    gs = df["green_share"].replace(0, np.nan)
    df["det_on_per_hour"] = df["det_n_on"] / hrs
    df["det_occ_frac"] = df["det_occ"] / df["win_secs"]
    df["log_det_n_on"] = np.log1p(df["det_n_on"])
    df["log_win_hours"] = np.log(hrs)
    df["n_on_per_cycle"] = df["det_n_on"] / df["n_cycles"].clip(lower=1)
    df["burst_rate_g4"] = df["n_on_g4"] / (4.0 * df["n_cycles"].clip(lower=1))
    df["burst_rate_g8"] = df["n_on_g8"] / (8.0 * df["n_cycles"].clip(lower=1))
    df["late_green_rate"] = df["n_on_late_green"] / (3.0 * df["n_cycles"].clip(lower=1))
    df["burst_lift_g4"] = df["burst_rate_g4"] / (df["det_n_on"] / df["win_secs"]).replace(0, np.nan)
    df["f_on_green_over_share"] = df["f_on_green"] / gs
    df["f_occ_green_over_share"] = df["f_occ_green"] / gs
    df["dur_ratio_red_green"] = df["dur_mean_red"] / df["dur_mean_green"].replace(0, np.nan)
    df["long_on_share"] = df["n_long_on"] / df["det_n_on"].clip(lower=1)
    df["call43_rev_per_on"] = df["n_call43"] / df["det_n_on"].clip(lower=1)
    # chance-corrected forward-call lift: P(call <=0.35 s after ON) / (call rate * 0.35)
    call_rate = (df["n43_win"] / df["win_secs"]).replace(0, np.nan)
    df["call43_fwd_lift"] = df["call43_fwd_035"] / (0.35 * call_rate)
    df["call44_fwd_lift"] = df["call44_fwd_035"] / (0.35 * call_rate)
    df["n43_per_hour"] = df["n43_win"] / hrs
    df["n_cycles_per_hour"] = df["n_cycles"] / hrs
    df.drop(columns=["n_on_g4", "n_on_g8", "n_on_late_green", "n_long_on",
                     "n_call43", "n43_win", "det_occ", "green_total", "n_on15"],
            inplace=True, errors="ignore")
    return df


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["DeviceId", "win", "Detector"])
    df["n_cand"] = g["cand_phase"].transform("size").astype("float32")
    new = {}
    for f in RANK_FEATS:
        if f not in df.columns:
            continue
        v = df[f]
        new[f"{f}__rank"] = g[f].rank(pct=True, method="average")
        mu, sd = g[f].transform("mean"), g[f].transform("std")
        new[f"{f}__z"] = (v - mu) / sd.replace(0, np.nan)
        mx = g[f].transform("max")
        new[f"{f}__mgap"] = v - mx
        new[f"{f}__argmax"] = (v >= mx).astype("float32")
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = add_derived(df)
    df = add_rank_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
    df["Detector"] = df["Detector"].astype(np.int16)
    df["cand_phase"] = df["cand_phase"].astype(np.int16)
    return df
