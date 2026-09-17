"""Detector phase + function inference: raw controller events in, one row per detector out.

Command line
------------
    python src/predict.py --events events.parquet --out preds.csv
    python src/predict.py --events day.parquet --device-ids <guid>,<guid> \
        --start "2024-12-03 08:00:00" --end "2024-12-03 08:30:00" --out preds.csv
    python src/predict.py --events events.csv --out preds.csv --odot-tiebreak

Python
------
    from predict import predict
    out = predict(events_df_or_path, start=None, end=None, odot_tiebreak=False)

`events` is a pandas DataFrame, a file path or a glob (parquet / csv) with the columns
`DeviceId, Timestamp, EventId, Parameter` (the lowercase / underscore spellings
`device_id, timestamp, event_id, parameter` are accepted too).  Any number of signals and
any duration from a few minutes to days.  Everything else is rebuilt here: the allowed
event codes are selected, `Parameter > 64` on 81/82 dropped, exact duplicate rows removed,
then ON intervals, phase colour cycles, the green bitmask timeline, coordination state, the
pair features, the cross-detector similarity graph, the phase ranker, the joint decoder and
the function model.  No channel->phase table is used unless `odot_tiebreak=True`.

Models are loaded from the repo folder `models/beta_v0/` by default.

Output columns
--------------
DeviceId, Detector, phase_pred, phase_prob, phase_2nd, phase_2nd_prob,
function_pred, function_prob, status, review_flag, n_actuations, minutes_of_data,
tiebreak_applied, and the extras phase_margin, p_advance, p_presence, p_count,
n_candidate_phases, health_flag, review_reason.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_detector as cd  # noqa: E402
import decode_v2 as dec  # noqa: E402
import features as f1  # noqa: E402
import features_v2 as f2  # noqa: E402
import function_v2 as fv2  # noqa: E402
from common import ALLOWED_EVENTS, FUNCTIONS, MAX_DETECTOR_CHANNEL  # noqa: E402
from health import flag_detectors, status_for_user  # noqa: E402
from tiebreak import apply_odot_tiebreak  # noqa: E402

# repo-relative model folder -- no absolute user paths anywhere in the inference path
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "beta_v0"

EV_LIST = ",".join(str(e) for e in ALLOWED_EVENTS)
WIN = "infer"
HEALTH_COLS = dict(max_day_gap_s=0.0, unmatched_on_rate=0.0, day_ratio=1.0)

ALIASES = {"deviceid": "DeviceId", "device_id": "DeviceId", "devid": "DeviceId",
           "timestamp": "Timestamp", "time_stamp": "Timestamp", "ts": "Timestamp",
           "eventid": "EventId", "event_id": "EventId", "eventcode": "EventId",
           "event_code": "EventId", "parameter": "Parameter", "param": "Parameter",
           "eventparam": "Parameter", "event_param": "Parameter"}
NEEDED = ["DeviceId", "Timestamp", "EventId", "Parameter"]

OUT_COLS = ["DeviceId", "Detector", "phase_pred", "phase_prob", "phase_2nd",
            "phase_2nd_prob", "function_pred", "function_prob", "status", "review_flag",
            "n_actuations", "minutes_of_data", "tiebreak_applied"]
EXTRA_COLS = ["phase_margin", "p_advance", "p_presence", "p_count",
              "n_candidate_phases", "health_flag", "review_reason"]

# a channel with fewer actuations than this still gets a prediction but is flagged
LOW_ACTUATIONS = 10
LOW_MINUTES = 10.0
LOW_CONF = 0.5

_VERBOSE = False


def log(m):
    if _VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ connection
def _connect(threads: int = 4, memory: str = "4GB") -> duckdb.DuckDBPyConnection:
    """Small, self-contained DuckDB connection (system temp dir, capped resources)."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET threads={max(1, int(threads))}")
    tmp = Path(tempfile.gettempdir()) / "duckdb_detector_classifier"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


# ------------------------------------------------------------- event ingestion
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if str(c) in NEEDED:
            continue
        if key in ALIASES:
            ren[c] = ALIASES[key]
    out = df.rename(columns=ren) if ren else df
    missing = [c for c in NEEDED if c not in out.columns]
    if missing:
        raise ValueError(
            f"events are missing the column(s) {missing}; need "
            "DeviceId, Timestamp, EventId, Parameter (lowercase/underscore spellings ok)")
    return out[NEEDED]


def _source(con, events) -> str:
    """Return a DuckDB relation name/expression for `events` (DataFrame, path or glob)."""
    if isinstance(events, pd.DataFrame):
        con.register("src_events_df", _normalise_columns(events))
        return "src_events_df"
    p = str(events).replace("\\", "/")
    if p.lower().endswith((".csv", ".csv.gz", ".txt", ".tsv")):
        rel = f"read_csv('{p}', header=true, union_by_name=true)"
    else:
        rel = f"read_parquet('{p}', union_by_name=true)"
    cols = [c[0] for c in con.sql(f"SELECT * FROM {rel} LIMIT 0").description]
    ren, have = {}, set(cols)
    for c in cols:
        key = str(c).strip().lower().replace(" ", "_")
        if c not in NEEDED and key in ALIASES and ALIASES[key] not in have:
            ren[c] = ALIASES[key]
    missing = [c for c in NEEDED if c not in have and c not in ren.values()]
    if missing:
        raise ValueError(f"{events}: missing column(s) {missing}")
    sel = ", ".join(f'"{c}" AS {ren.get(c, c)}' for c in cols if ren.get(c, c) in NEEDED)
    return f"(SELECT {sel} FROM {rel})"


def load_events(con, events, device_ids=None, start=None, end=None):
    """Build the deduplicated, protocol-filtered TEMP TABLE `ev`.  Returns (w0, w1, info)."""
    rel = _source(con, events)
    where = [f"EventId IN ({EV_LIST})",
             f"NOT (EventId IN (81,82) AND Parameter > {MAX_DETECTOR_CHANNEL})"]
    if device_ids:
        ids = ",".join("'" + str(d).replace("'", "''") + "'" for d in device_ids)
        where.append(f"DeviceId IN ({ids})")
    if start:
        where.append(f"Timestamp >= TIMESTAMP '{start}'")
    if end:
        where.append(f"Timestamp < TIMESTAMP '{end}'")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ev AS
        SELECT DISTINCT CAST(DeviceId AS VARCHAR) AS DeviceId,
                        CAST(Timestamp AS TIMESTAMP) AS Timestamp,
                        CAST(EventId AS USMALLINT) AS EventId,
                        CAST(Parameter AS USMALLINT) AS Parameter
        FROM (SELECT CAST(DeviceId AS VARCHAR) AS DeviceId,
                     CAST(Timestamp AS TIMESTAMP) AS Timestamp,
                     CAST(EventId AS INTEGER) AS EventId,
                     CAST(Parameter AS INTEGER) AS Parameter
              FROM {rel}) WHERE {' AND '.join(where)}""")
    n, nd, t0, t1 = con.sql(
        "SELECT count(*), count(DISTINCT DeviceId), min(Timestamp), max(Timestamp) "
        "FROM ev").fetchone()
    if not n:
        return None, None, {"n_events": 0}
    w0 = (pd.Timestamp(t0) - pd.Timestamp("1970-01-01")).total_seconds()
    w1 = (pd.Timestamp(t1) - pd.Timestamp("1970-01-01")).total_seconds() + 1.0
    log(f"{n:,} events, {nd} signals, span {pd.Timestamp(t0)} .. {pd.Timestamp(t1)}")
    return w0, w1, {"n_events": int(n), "n_signals": int(nd), "t0": t0, "t1": t1}


def build_chunk_tables(con) -> None:
    """The temp tables `features.load_chunk` creates, rebuilt from the in-memory events.

    Colour cycles use 1 / 8 / 10 with 7 (green termination) and 9 (end yellow) as
    fall-backs, so controllers that log only 1 + 7 still get a usable green/red split.
    """
    con.execute("""CREATE OR REPLACE TEMP TABLE devmap AS
        SELECT DeviceId, row_number() OVER (ORDER BY DeviceId)::SMALLINT AS dev
        FROM (SELECT DISTINCT DeviceId FROM ev)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE onev_all AS
        WITH e AS (SELECT DeviceId, Parameter::SMALLINT AS det, Timestamp AS ts, EventId
                   FROM ev WHERE EventId IN (81,82)),
             d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
                   WINDOW w AS (PARTITION BY DeviceId, det
                                ORDER BY ts, CASE WHEN EventId=82 THEN 0 ELSE 1 END))
        SELECT m.dev, d.det, epoch_ms(d.ts)/1000.0 AS t_on, epoch_ms(d.nts)/1000.0 AS t_off,
               (epoch_ms(d.nts - d.ts)/1000.0)::FLOAT AS dur
        FROM d JOIN devmap m USING (DeviceId)
        WHERE d.EventId = 82 AND d.nev = 81 AND d.nts IS NOT NULL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cyc_raw AS
        WITH g AS (
          SELECT DeviceId, Parameter::SMALLINT AS p, Timestamp AS t, EventId,
                 SUM(CASE WHEN EventId=1 THEN 1 ELSE 0 END) OVER (
                     PARTITION BY DeviceId, Parameter
                     ORDER BY t, CASE EventId WHEN 1 THEN 0 WHEN 7 THEN 1 WHEN 8 THEN 2
                                              WHEN 9 THEN 3 WHEN 10 THEN 4 ELSE 5 END
                     ROWS UNBOUNDED PRECEDING) AS cyc
          FROM ev WHERE EventId IN (1,7,8,9,10,11)
        ), c AS (
          SELECT DeviceId, p, cyc,
                 min(t) FILTER (EventId=1)  AS green_start,
                 min(t) FILTER (EventId=8)  AS yellow_ev,
                 min(t) FILTER (EventId=7)  AS green_term,
                 min(t) FILTER (EventId=10) AS red_ev,
                 min(t) FILTER (EventId=9)  AS yellow_end
          FROM g WHERE cyc > 0 AND p BETWEEN 1 AND 16 GROUP BY 1,2,3
        )
        SELECT DeviceId, p, cyc, green_start,
               coalesce(yellow_ev, green_term) AS yellow_start,
               coalesce(red_ev, yellow_end) AS red_start,
               LEAD(green_start) OVER (PARTITION BY DeviceId, p ORDER BY green_start)
                   AS next_green
        FROM c WHERE green_start IS NOT NULL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cyc_all AS
        SELECT m.dev, c.p, c.cyc::INT AS cyc,
               epoch_ms(c.green_start)/1000.0 AS gs,
               epoch_ms(coalesce(c.yellow_start, c.red_start, c.next_green))/1000.0 AS ge,
               epoch_ms(coalesce(c.red_start, c.yellow_start))/1000.0 AS rs,
               epoch_ms(c.next_green)/1000.0 AS ng,
               (epoch_ms(coalesce(c.yellow_start, c.red_start) - c.green_start)/1000.0)::FLOAT
                   AS green_secs
        FROM cyc_raw c JOIN devmap m USING (DeviceId)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE gs_all AS
        WITH iv AS (SELECT DeviceId, p, green_start AS t0,
                           coalesce(yellow_start, red_start, next_green) AS t1
                    FROM cyc_raw WHERE coalesce(yellow_start, red_start, next_green) IS NOT NULL),
             ch AS (SELECT DeviceId, t0 AS t,  (1::BIGINT << (p-1)) AS d FROM iv
                    UNION ALL
                    SELECT DeviceId, t1 AS t, -(1::BIGINT << (p-1)) AS d FROM iv),
             agg AS (SELECT DeviceId, t, sum(d) AS d FROM ch GROUP BY 1,2),
             run AS (SELECT DeviceId, t,
                            sum(d) OVER (PARTITION BY DeviceId ORDER BY t
                                         ROWS UNBOUNDED PRECEDING) AS mask FROM agg)
        SELECT m.dev, epoch_ms(r.t)/1000.0 AS t0,
               epoch_ms(LEAD(r.t) OVER (PARTITION BY r.DeviceId ORDER BY r.t))/1000.0 AS t1,
               r.mask
        FROM run r JOIN devmap m USING (DeviceId)""")
    con.execute("DELETE FROM gs_all WHERE t1 IS NULL")
    con.execute("""CREATE OR REPLACE TEMP TABLE coordiv AS
        SELECT m.dev, epoch_ms(e.Timestamp)/1000.0 AS t0,
               (e.Parameter BETWEEN 1 AND 253) AS is_coord
        FROM ev e JOIN devmap m USING (DeviceId) WHERE e.EventId = 131""")
    con.execute("""CREATE OR REPLACE TEMP TABLE calls_all AS
        SELECT m.dev, e.Parameter::SMALLINT AS p, e.EventId::SMALLINT AS ev,
               epoch_ms(e.Timestamp)/1000.0 AS t
        FROM ev e JOIN devmap m USING (DeviceId)
        WHERE e.EventId IN (43,44) AND e.Parameter BETWEEN 1 AND 16""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cand AS
        SELECT DISTINCT m.dev, e.Parameter::SMALLINT AS p
        FROM ev e JOIN devmap m USING (DeviceId)
        WHERE e.EventId = 1 AND e.Parameter BETWEEN 1 AND 16""")


# ------------------------------------------------------------- per-signal facts
def detector_universe(con) -> pd.DataFrame:
    """Every detector channel that appears on an 81/82 event, with its actuation count."""
    return con.sql("""
        SELECT DeviceId, Parameter::INT AS Detector,
               count(*) FILTER (EventId = 82)::BIGINT AS n_actuations
        FROM ev WHERE EventId IN (81,82) GROUP BY 1,2 ORDER BY 1,2""").df()


def signal_facts(con) -> pd.DataFrame:
    """Per signal: span, candidate phases, whether colour-termination / call events exist."""
    return con.sql("""
        SELECT DeviceId,
               (epoch_ms(max(Timestamp) - min(Timestamp))/60000.0)::DOUBLE AS minutes_of_data,
               count(*) FILTER (EventId = 1 AND Parameter BETWEEN 1 AND 16) AS n_green,
               count(DISTINCT CASE WHEN EventId = 1 AND Parameter BETWEEN 1 AND 16
                                   THEN Parameter END) AS n_candidate_phases,
               count(*) FILTER (EventId IN (7,8,9,10)) AS n_green_end,
               count(*) FILTER (EventId IN (43,44)) AS n_calls
        FROM ev GROUP BY 1""").df()


def health_frame(con, win_secs: float) -> pd.DataFrame:
    days = max(win_secs / 86400.0, 1e-6)
    h = con.sql(f"""
        WITH a AS (
          SELECT dev, det, count(*) AS n_on, sum(dur) AS occ, max(dur) AS longest_on_s,
                 max(cnt) AS max_on_per_min
          FROM (SELECT dev, det, dur, count(*) OVER (PARTITION BY dev, det,
                       (t_on/60)::BIGINT) AS cnt FROM onev_all) GROUP BY 1,2
        ), fl AS (
          SELECT m.dev, e.Parameter::SMALLINT AS det,
                 count(*) FILTER (e.EventId BETWEEN 84 AND 88) AS n_fault_events
          FROM ev e JOIN devmap m USING (DeviceId)
          WHERE e.EventId BETWEEN 83 AND 88 GROUP BY 1,2
        )
        SELECT d.DeviceId, a.det::INT AS Detector, a.n_on,
               a.n_on / {days} AS on_per_day, a.occ / {max(win_secs, 1.0)} AS frac_time_on,
               a.longest_on_s, a.max_on_per_min,
               coalesce(fl.n_fault_events, 0) AS n_fault_events
        FROM a JOIN devmap d USING (dev) LEFT JOIN fl ON fl.dev=a.dev AND fl.det=a.det
    """).df()
    if not len(h):
        h = pd.DataFrame(columns=["DeviceId", "Detector", "n_on", "on_per_day",
                                  "frac_time_on", "longest_on_s", "max_on_per_min",
                                  "n_fault_events"])
    for c, v in HEALTH_COLS.items():
        h[c] = v
    return flag_detectors(h)


# ------------------------------------------------------------------- features
EMPTY_SIM = pd.DataFrame({"DeviceId": pd.Series(dtype=object),
                          "Detector": pd.Series(dtype="int16"),
                          "other": pd.Series(dtype="int16"),
                          "phi": pd.Series(dtype="float32"),
                          "n_common": pd.Series(dtype="int64"),
                          "win": pd.Series(dtype=object)})


def build_features(con, w0: float, w1: float):
    secs = max(w1 - w0, 1.0)
    f1.apply_window(con, w0, w1)
    base = f1.build_window(con, WIN, secs)
    if not len(base):
        return None, None
    base = f1.finalise(base)
    extra = f2.build_window(con, WIN, secs)
    sim = cd.build_window(con, WIN, secs)
    if sim is None or not len(sim):
        sim = EMPTY_SIM.copy()
    df = base
    if extra is not None and len(extra):
        df = base.merge(extra, on=["DeviceId", "Detector", "cand_phase", "win"], how="left")
    df = f2.add_partner_diffs(df, f2.PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                                    "f_on_green", "excl_diff_min",
                                                    "release_frac_long", "call43_fwd_lift"])
    return df, sim


# --------------------------------------------------------------------- scoring
def _softmax_by_detector(df: pd.DataFrame, s: np.ndarray, T: float = 1.0) -> np.ndarray:
    g = df.DeviceId.astype(str) + "|" + df.Detector.astype(str)
    d = pd.DataFrame({"g": g.to_numpy(), "s": np.asarray(s) / T})
    d["s"] = np.exp(d.s - d.groupby("g")["s"].transform("max"))
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def _normalise_by_detector(df: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    g = df.DeviceId.astype(str) + "|" + df.Detector.astype(str)
    d = pd.DataFrame({"g": g.to_numpy(), "s": np.clip(np.asarray(s), 1e-9, None)})
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


def score(df: pd.DataFrame, sim: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    meta = json.load(open(model_dir / "phase_lgbm_v2.json"))
    bst = lgb.Booster(model_file=str(model_dir / "phase_lgbm_v2.txt"))
    for c in meta["features"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values(["DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    df["p0"] = _softmax_by_detector(df, np.asarray(bst.predict(df[meta["features"]])))

    dmeta = json.load(open(model_dir / "decode_lgbm_v2.json"))
    dbst = lgb.Booster(model_file=str(model_dir / "decode_lgbm_v2.txt"))
    X = dec.assemble(df[["DeviceId", "Detector", "win", "cand_phase", "p0"]], pairs=df, sim=sim)
    for c in dmeta["features"]:
        if c not in X.columns:
            X[c] = np.nan
    sd = np.asarray(dbst.predict(X[dmeta["features"]]))
    X["prob"] = (_normalise_by_detector(X, sd) if dmeta.get("mode") == "binary"
                 else _softmax_by_detector(X, sd, dmeta.get("temperature", 1.0)))
    return df.merge(X[["DeviceId", "Detector", "win", "cand_phase", "prob"]],
                    on=["DeviceId", "Detector", "win", "cand_phase"], how="left")


def score_function(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    meta = json.load(open(model_dir / "function_lgbm_v2.json"))
    bst = lgb.Booster(model_file=str(model_dir / "function_lgbm_v2.txt"))
    top = fv2.build_frame(df.drop(columns=["p0", "prob"], errors="ignore"),
                          df[["DeviceId", "Detector", "win", "cand_phase", "prob"]])
    if not len(top):
        return pd.DataFrame(columns=["DeviceId", "Detector", "p_advance", "p_presence",
                                     "p_count", "function_pred", "function_prob"])
    for c in meta["features"]:
        if c not in top.columns:
            top[c] = np.nan
    P = fv2._apply_T(np.asarray(bst.predict(top[meta["features"]])), meta["temperature"])
    out = top[["DeviceId", "Detector"]].copy()
    out[["p_advance", "p_presence", "p_count"]] = P
    out["function_prob"] = P.max(1)
    out["function_pred"] = np.where(P.max(1) < meta["other_threshold"], "Other",
                                    np.array(FUNCTIONS)[P.argmax(1)])
    return out


# ----------------------------------------------------------------- assembly
def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=OUT_COLS + EXTRA_COLS)


def _assemble(univ, facts, hf, ph, fn, switched, model_note: str) -> pd.DataFrame:
    res = univ.merge(facts[["DeviceId", "minutes_of_data", "n_candidate_phases",
                            "n_green_end", "n_calls"]], on="DeviceId", how="left")
    if ph is not None and len(ph):
        p = ph.sort_values(["DeviceId", "Detector", "prob"], ascending=[True, True, False])
        g = p.groupby(["DeviceId", "Detector"], sort=False)
        a = g.head(1).rename(columns={"cand_phase": "phase_pred", "prob": "phase_prob"})
        b = g.nth(1).rename(columns={"cand_phase": "phase_2nd", "prob": "phase_2nd_prob"})
        res = res.merge(a[["DeviceId", "Detector", "phase_pred", "phase_prob"]],
                        on=["DeviceId", "Detector"], how="left")
        res = res.merge(b[["DeviceId", "Detector", "phase_2nd", "phase_2nd_prob"]],
                        on=["DeviceId", "Detector"], how="left")
    else:
        for c in ("phase_pred", "phase_prob", "phase_2nd", "phase_2nd_prob"):
            res[c] = np.nan
    res["phase_margin"] = res.phase_prob.fillna(0) - res.phase_2nd_prob.fillna(0)

    if fn is not None and len(fn):
        res = res.merge(fn, on=["DeviceId", "Detector"], how="left")
    else:
        for c in ("p_advance", "p_presence", "p_count", "function_prob"):
            res[c] = np.nan
        res["function_pred"] = pd.NA

    if hf is not None and len(hf):
        res = res.merge(hf[["DeviceId", "Detector", "health_flag", "health_reason"]],
                        on=["DeviceId", "Detector"], how="left")
    else:
        res["health_flag"] = pd.NA
        res["health_reason"] = pd.NA
    res["health_flag"] = res.health_flag.fillna("failed")
    res["health_reason"] = res.health_reason.fillna("no_events")
    res.loc[res.n_actuations.fillna(0) == 0, ["health_flag", "health_reason"]] = \
        ["failed", "no_events"]

    # ---- status ------------------------------------------------------------
    status, review, reason = [], [], []
    for r in res.itertuples():
        notes = []
        if (r.n_actuations or 0) == 0:
            st = "cannot classify: no actuations"
            rv, rs = True, "no actuations"
        elif r.health_flag == "failed":
            st = status_for_user("failed", r.health_reason)
            rv, rs = True, r.health_reason
        elif not (r.n_candidate_phases or 0):
            st = ("cannot classify: no phase begin-green (event 1) records in this window")
            rv, rs = True, "no phase events"
        elif pd.isna(r.phase_pred):
            st = "cannot classify: no usable detector/phase evidence in this window"
            rv, rs = True, "no prediction"
        else:
            rs = ""
            if (r.n_actuations or 0) < LOW_ACTUATIONS:
                n = int(r.n_actuations)
                notes.append(f"low evidence: only {n} actuation" + ("" if n == 1 else "s"))
                rs = "few actuations"
            if (r.minutes_of_data or 0) < LOW_MINUTES:
                notes.append(f"low evidence: only {r.minutes_of_data:.0f} min of data")
                rs = rs or "short sample"
            if not (r.n_green_end or 0):
                notes.append("reduced accuracy: no green-termination events (7/8/9/10)")
                rs = rs or "no colour-state events"
            if not (r.n_calls or 0):
                notes.append("reduced accuracy: no phase call events (43/44)")
                rs = rs or "no 43/44 events"
            if r.health_flag == "suspect":
                notes.append(f"low confidence: detector data quality ({r.health_reason})")
                rs = rs or r.health_reason
            if (r.phase_prob or 0) < LOW_CONF:
                notes.append("low confidence: phase probability below 0.5")
                rs = "low confidence phase"
            st = "; ".join(notes) if notes else "ok"
            rv = bool(notes)
        status.append(st)
        review.append(rv)
        reason.append(rs)
    res["status"] = status
    res["review_flag"] = review
    res["review_reason"] = reason
    # a "cannot classify" channel never carries a phase / function answer
    dead = res.status.str.startswith("cannot classify")
    res.loc[dead, ["phase_pred", "phase_prob", "phase_2nd", "phase_2nd_prob",
                   "function_pred", "function_prob", "p_advance", "p_presence",
                   "p_count", "phase_margin"]] = np.nan
    if model_note:
        res["status"] = res.status + "; " + model_note

    swk = set(switched.DeviceId.astype(str) + "|" + switched.Detector.astype(str)) \
        if switched is not None and len(switched) else set()
    res["tiebreak_applied"] = (res.DeviceId.astype(str) + "|" +
                               res.Detector.astype(str)).isin(swk)
    res.loc[res.phase_pred.isna(), "tiebreak_applied"] = False
    for c in ("phase_pred", "phase_2nd"):
        res[c] = res[c].astype("Int64")
    res["n_actuations"] = res.n_actuations.fillna(0).astype("Int64")
    res["minutes_of_data"] = res.minutes_of_data.astype(float).round(2)
    return res[OUT_COLS + EXTRA_COLS].sort_values(["DeviceId", "Detector"]).reset_index(
        drop=True)


# ---------------------------------------------------------------- entry points
def predict(events, start=None, end=None, odot_tiebreak: bool = False,
            device_ids=None, model_dir=None, threads: int = 4, memory: str = "4GB",
            chunk_signals: int | None = None, verbose: bool = False) -> pd.DataFrame:
    """Raw hi-res events -> one row per detector channel.  Never raises on thin data.

    Parameters
    ----------
    events        pandas DataFrame, or a path / glob to parquet or csv, with columns
                  DeviceId, Timestamp, EventId, Parameter (lowercase variants accepted).
    start, end    optional timestamp strings; `end` is exclusive.
    odot_tiebreak turn on the ODOT standard-wiring tie-breaker (post-processing only).
    chunk_signals process the signals in groups of this many to bound peak memory.
    """
    global _VERBOSE
    _VERBOSE = verbose
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not (model_dir / "phase_lgbm_v2.txt").exists():
        raise FileNotFoundError(f"no models in {model_dir}")

    if chunk_signals:
        ids = device_ids or list_signals(events, start, end, threads, memory)
        if len(ids) > chunk_signals:
            parts = [predict(events, start, end, odot_tiebreak, ids[i:i + chunk_signals],
                             model_dir, threads, memory, None, verbose)
                     for i in range(0, len(ids), chunk_signals)]
            parts = [p for p in parts if len(p)]
            return (pd.concat(parts, ignore_index=True) if parts else _empty_result())

    con = _connect(threads, memory)
    try:
        w0, w1, info = load_events(con, events, device_ids, start, end)
        if not info["n_events"]:
            return _empty_result()
        build_chunk_tables(con)
        univ = detector_universe(con)
        facts = signal_facts(con)
        if not len(univ):
            return _empty_result()
        hf = health_frame(con, w1 - w0)
        note = ""
        try:
            df, sim = build_features(con, w0, w1)
        except Exception as exc:                                  # pragma: no cover
            df, sim = None, None
            note = f"feature build failed ({type(exc).__name__})"
        ph = fn = None
        switched = pd.DataFrame(columns=["DeviceId", "Detector"])
        if df is not None and len(df):
            log(f"features {df.shape}, similarity {sim.shape}")
            df = score(df, sim, model_dir)
            ph = df[["DeviceId", "Detector", "cand_phase", "prob"]].copy()
            s = ph.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
            ph["prob"] = ph.prob / s.replace(0, np.nan)
            ph = ph.dropna(subset=["prob"])
            ph, switched = apply_odot_tiebreak(ph, enabled=odot_tiebreak, return_flags=True)
            try:
                fn = score_function(df, model_dir)
            except Exception as exc:                              # pragma: no cover
                note = (note + "; " if note else "") + \
                    f"function model unavailable ({type(exc).__name__})"
        return _assemble(univ, facts, hf, ph, fn, switched, note)
    finally:
        con.close()


def list_signals(events, start=None, end=None, threads: int = 2,
                 memory: str = "2GB") -> list[str]:
    """DeviceIds present in `events` (after the time filter), sorted."""
    con = _connect(threads, memory)
    try:
        rel = _source(con, events)
        where = ["1=1"]
        if start:
            where.append(f"Timestamp >= TIMESTAMP '{start}'")
        if end:
            where.append(f"Timestamp < TIMESTAMP '{end}'")
        return [r[0] for r in con.sql(
            f"SELECT DISTINCT CAST(DeviceId AS VARCHAR) FROM {rel} "
            f"WHERE {' AND '.join(where)} ORDER BY 1").fetchall()]
    finally:
        con.close()


def run(events: str, out: str, device_ids=None, start=None, end=None,
        odot_tiebreak: bool = False, model_dir=None, threads: int = 4,
        memory: str = "4GB", chunk_signals: int | None = None) -> pd.DataFrame:
    """CLI helper: predict and write a CSV."""
    t0 = time.time()
    res = predict(events, start, end, odot_tiebreak, device_ids, model_dir, threads,
                  memory, chunk_signals, verbose=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"wrote {out}: {len(res)} detectors, "
          f"{res.DeviceId.nunique() if len(res) else 0} signals, "
          f"{time.time()-t0:.1f}s", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, help="parquet/csv path or glob")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--device-ids", default=None, help="comma separated DeviceId filter")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None, help="exclusive")
    ap.add_argument("--odot-tiebreak", action="store_true",
                    help="post-process close concurrent-pair ties with the ODOT standard "
                         "wiring table at signals that look standard-wired (default off)")
    ap.add_argument("--models", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="4GB")
    ap.add_argument("--chunk-signals", type=int, default=None,
                    help="process this many signals at a time to bound memory")
    a = ap.parse_args()
    ids = [s.strip() for s in a.device_ids.split(",")] if a.device_ids else None
    run(a.events, a.out, ids, a.start, a.end, a.odot_tiebreak, Path(a.models),
        a.threads, a.memory, a.chunk_signals)


if __name__ == "__main__":
    main()
