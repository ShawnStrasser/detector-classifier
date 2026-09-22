"""Detector phase + function inference: raw controller events in, one row per detector out.

FINAL MODEL (`models/final_v1`, 2026-09-21).  See `docs/FINAL_REPORT.md`.

Command line
------------
    python src/predict.py --events events.parquet --out preds.csv
    python src/predict.py --events day.parquet --device-ids <guid>,<guid> \
        --start "2026-09-21 08:00:00" --end "2026-09-21 08:30:00" --out preds.csv
    python src/predict.py --events events.parquet --out preds.csv \
        --min-actuations 1 --min-prob 0.9        # the confidence-based service

Python
------
    from predict import predict
    out = predict(events_df_or_path, start=None, end=None)

`events` is a pandas DataFrame, a file path or a glob (parquet / csv) with the columns
`DeviceId, Timestamp, EventId, Parameter` (the lowercase / underscore spellings
`device_id, timestamp, event_id, parameter` are accepted too).  Any number of signals and
any duration from a few minutes to days.  Everything else is rebuilt here: the allowed
event codes are selected, `Parameter > 64` on 81/82 dropped, exact duplicate rows removed,
then ON intervals, phase colour cycles (four colours, so the red-clearance interval a
Yellow_Red detector is defined by can be measured), the green bitmask timeline,
coordination state, the pair features, the cross-detector similarity and actuation-lag
graphs, the 3-seed phase ranker, the joint decoder and the 5-class function model.
No channel->phase table is used anywhere.

Models are loaded from the repo folder `models/final_v1/` by default:
    phase_lgbm_v4_s{0,1,2}.txt   pair ranker, 3 seeds, averaged (stage 07)
    decode_lgbm_v4.txt           joint per-signal decoder
    function_lgbm_v4.txt         5-class function head
This replaces the beta of 2026-09-17 (see the git history).

Minimum evidence -- two ways to refuse
--------------------------------------
* `min_actuations` (default 5): a detector with fewer than this many ON events in the
  sample gets no answer.  `phase_pred` / `function_pred` are blank and `status` says
  "not enough data: N actuations in sample (need >= MIN)".
* `min_prob` (default 0.0 = off): a detector whose top phase probability is below this
  gets no answer ("not confident enough: phase probability 0.62 (need >= 0.90)").

Both rules are applied; the model's raw opinion is always kept in `phase_guess` /
`function_guess`, so nothing is lost either way.  The default is the actuation rule alone.
The measured alternative (`--min-actuations 1 --min-prob 0.9`) answers *more* detectors at
*higher* accuracy, because a short sample can still be conclusive -- see the final report.

Output columns
--------------
DeviceId, Detector, phase_pred, phase_prob, phase_2nd, phase_2nd_prob,
function_pred, function_prob, status, review_flag, n_actuations, minutes_of_data,
and the extras phase_guess, phase_guess_prob, function_guess,
function_guess_prob, phase_margin, p_advance, p_presence, p_count, p_yellow_red,
p_other, n_candidate_phases, health_flag, review_reason.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import os

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_detector as cd  # noqa: E402
import decode_v2 as dec  # noqa: E402
import features as f1  # noqa: E402
import features_v2 as f2  # noqa: E402
import features_v3 as f3  # noqa: E402
import function_v3 as fv3  # noqa: E402
from common import ALLOWED_EVENTS, MAX_DETECTOR_CHANNEL  # noqa: E402
from health import flag_detectors, status_for_user  # noqa: E402

# repo-relative model folder -- no absolute user paths anywhere in the inference path
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "final_v1"

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
            "n_actuations", "minutes_of_data"]
EXTRA_COLS = ["phase_guess", "phase_guess_prob", "function_guess", "function_guess_prob",
              "phase_margin", "p_advance", "p_presence", "p_count", "p_yellow_red",
              "p_other", "n_candidate_phases", "health_flag", "review_reason"]

# Below this many detector ON events in the sample we refuse to answer (see module docstring).
MIN_ACTUATIONS = 5
# Below this top-phase probability we refuse to answer.  0.0 = rule disabled (the default).
MIN_PROB = 0.0
# answered, but thin enough to flag for review
LOW_ACTUATIONS = 20
LOW_MINUTES = 10.0
LOW_CONF = 0.5

_VERBOSE = False
_LAG: pd.DataFrame | None = None        # detector-pair lag table, build_features -> function


# ---- model backend: "lightgbm" (needs lightgbm + scipy) or "numpy" (pure numpy, identical output) ----
_BACKEND = os.environ.get("DC_BOOSTER", "auto").lower()


def set_backend(name: str) -> None:
    """'auto' (lightgbm if installed, else numpy), 'lightgbm' or 'numpy'."""
    global _BACKEND
    if name not in ("auto", "lightgbm", "numpy"):
        raise ValueError(name)
    _BACKEND = name


def _load_booster(path):
    if _BACKEND in ("auto", "lightgbm"):
        try:
            import lightgbm as lgb
            return lgb.Booster(model_file=str(path))
        except ImportError:
            if _BACKEND == "lightgbm":
                raise
    from lgbm_numpy import NumpyBooster
    return NumpyBooster(path)


def _bag_files(model_dir: Path, stem: str, meta: dict) -> list[Path]:
    """The K seed models of a bagged stage, or the single model if it is not bagged."""
    n = int(meta.get("n_models", 1))
    files = [model_dir / f"{stem}_s{i}.txt" for i in range(n)]
    files = [f for f in files if f.exists()]
    if not files and (model_dir / f"{stem}.txt").exists():
        files = [model_dir / f"{stem}.txt"]
    if not files:
        raise FileNotFoundError(f"no {stem} model files in {model_dir}")
    return files


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
    A second, four-colour table (`cyc4_*`) additionally keeps **end red clearance
    (event 11)** apart, which is what the Yellow_Red function features are measured on.
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
                 min(t) FILTER (EventId=9)  AS yellow_end,
                 min(t) FILTER (EventId=11) AS redclr_ev
          FROM g WHERE cyc > 0 AND p BETWEEN 1 AND 16 GROUP BY 1,2,3
        )
        SELECT DeviceId, p, cyc, green_start,
               coalesce(yellow_ev, green_term) AS yellow_start,
               coalesce(red_ev, yellow_end) AS red_start,
               redclr_ev AS redclr_end,
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
    con.execute("""CREATE OR REPLACE TEMP TABLE cyc4_all AS
        SELECT m.dev, c.p, c.cyc::INT AS cyc,
               epoch_ms(c.green_start)/1000.0 AS gs,
               epoch_ms(coalesce(c.yellow_start, c.red_start, c.next_green))/1000.0 AS ge,
               epoch_ms(coalesce(c.red_start, c.yellow_start, c.next_green))/1000.0 AS rs,
               epoch_ms(coalesce(c.redclr_end, c.red_start, c.yellow_start,
                                 c.next_green))/1000.0 AS rce,
               epoch_ms(c.next_green)/1000.0 AS ng
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
    """Pair features (stage 01 + v2 + v3 yellow/red-clearance), similarity and lag graphs."""
    global _LAG
    _LAG = None
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
    # ---- v3: yellow / red-clearance pair features + detector-pair actuation lags ----
    dm = con.sql("SELECT * FROM devmap").df()
    f3.window_cycles(con, w0, w1)
    try:
        yr = f3.build(con, WIN, dm, f3.SQL_YR).rename(
            columns={"det": "Detector", "p": "cand_phase"})
        yr["Detector"] = yr.Detector.astype(df.Detector.dtype)
        yr["cand_phase"] = yr.cand_phase.astype(df.cand_phase.dtype)
        df = df.merge(yr, on=["DeviceId", "Detector", "cand_phase", "win"], how="left")
    except Exception as exc:                                       # pragma: no cover
        log(f"yr features unavailable ({type(exc).__name__}: {exc})")
    try:
        lg = f3.build(con, WIN, dm, f3.SQL_LAG).rename(
            columns={"det": "Detector", "oth": "other"})
        lg["Detector"] = lg.Detector.astype(df.Detector.dtype)
        lg["other"] = lg.other.astype(df.Detector.dtype)
        _LAG = lg
    except Exception as exc:                                       # pragma: no cover
        log(f"lag features unavailable ({type(exc).__name__}: {exc})")
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
    """Stage 1: the seed-bagged pair ranker.  Stage 2: the joint per-signal decoder.

    The K ranker seeds are averaged **after** each one is turned into a per-detector
    probability -- exactly the averaging the training code measured (stage 07).
    """
    meta = json.load(open(model_dir / "phase_lgbm_v4.json"))
    files = _bag_files(model_dir, "phase_lgbm_v4", meta)
    for c in meta["features"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values(["DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    X = df[meta["features"]]
    ps = [_softmax_by_detector(df, np.asarray(_load_booster(f).predict(X))) for f in files]
    df["p0"] = np.mean(ps, axis=0)
    log(f"ranker: {len(files)} seed model(s) averaged")

    dmeta = json.load(open(model_dir / "decode_lgbm_v4.json"))
    dbst = _load_booster(model_dir / "decode_lgbm_v4.txt")
    Xd = dec.assemble(df[["DeviceId", "Detector", "win", "cand_phase", "p0"]],
                      pairs=df, sim=sim)
    for c in dmeta["features"]:
        if c not in Xd.columns:
            Xd[c] = np.nan
    sd = np.asarray(dbst.predict(Xd[dmeta["features"]]))
    Xd["prob"] = (_normalise_by_detector(Xd, sd) if dmeta.get("mode") == "binary"
                  else _softmax_by_detector(Xd, sd, dmeta.get("temperature", 1.0)))
    return df.merge(Xd[["DeviceId", "Detector", "win", "cand_phase", "prob"]],
                    on=["DeviceId", "Detector", "win", "cand_phase"], how="left")


def _function_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The function design matrix: the pair features of the detector's **predicted** phase,
    plus shape, sibling-relative and cross-detector lag aggregates.  Identical code path to
    training (`src/function_v4.py --stage frame`); no label and no phase number is used."""
    pairs = df.drop(columns=["p0", "prob"], errors="ignore")
    probs = df[["DeviceId", "Detector", "win", "cand_phase", "prob"]]
    p = pairs.merge(probs, on=["DeviceId", "Detector", "win", "cand_phase"], how="inner")
    i = p.groupby(["DeviceId", "Detector", "win"], sort=False)["prob"].idxmax()
    top = p.loc[i].copy().rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})
    top = fv3.add_shape_features(top)
    top = fv3.add_sibling_features(top)
    top = top.reset_index(drop=True)
    if _LAG is not None and len(_LAG):
        old = fv3._cat
        fv3._cat = lambda files: _LAG           # feed the in-memory lag table
        try:
            top = fv3.add_lag_features(top)
        finally:
            fv3._cat = old
    # the training frame carried a flag for the full-span window group (>= ~66 h);
    # at inference the sample's own length decides it.
    if "win_secs" in top.columns:
        top["is_full"] = (top.win_secs.astype(float) >= 48 * 3600.0)
    return top


def score_function(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """The 5-class head: Advance / Presence / Count / Yellow_Red / Other."""
    meta = json.load(open(model_dir / "function_lgbm_v4.json"))
    bst = _load_booster(model_dir / "function_lgbm_v4.txt")
    classes = meta["classes"]
    pcols = [f"p_{c.lower()}" for c in classes]
    top = _function_frame(df)
    if not len(top):
        return pd.DataFrame(columns=["DeviceId", "Detector", "function_pred",
                                     "function_prob"] + pcols)
    for c in meta["features"]:
        if c not in top.columns:
            top[c] = np.nan
    Q = np.asarray(bst.predict(top[meta["features"]]))
    out = top[["DeviceId", "Detector"]].copy()
    out[pcols] = Q
    pred = np.array(classes)[Q.argmax(1)]
    # v3 shipped a rule that forced ambiguous Advance/Presence detectors to Other.  With an
    # explicitly trained Other class it now costs accuracy, so it ships disabled; the
    # parameters stay in the json for an operator who prefers Other recall to precision.
    r = meta.get("advance_presence_rule", {})
    if r.get("enabled"):
        ia, ip = classes.index("Advance"), classes.index("Presence")
        amb = (np.abs(Q[:, ia] - Q[:, ip]) < r.get("delta", 0.2)) & \
              ((Q[:, ia] + Q[:, ip]) > r.get("sum_min", 0.6))
        pred = np.where(amb, "Other", pred)
    out["function_pred"] = pred
    out["function_prob"] = Q.max(1)
    return out


# ----------------------------------------------------------------- assembly
PROB_COLS = ["p_advance", "p_presence", "p_count", "p_yellow_red", "p_other"]


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=OUT_COLS + EXTRA_COLS)


def _assemble(univ, facts, hf, ph, fn, model_note: str,
              min_actuations: int = MIN_ACTUATIONS,
              min_prob: float = MIN_PROB) -> pd.DataFrame:
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
    for c in PROB_COLS + ["function_prob"]:
        if c not in res.columns:
            res[c] = np.nan
    if "function_pred" not in res.columns:
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

    # the model's raw opinion is always kept, even where we refuse to answer
    res["phase_guess"] = res.phase_pred
    res["phase_guess_prob"] = res.phase_prob
    res["function_guess"] = res.function_pred
    res["function_guess_prob"] = res.function_prob

    # ---- status ------------------------------------------------------------
    status, review, reason = [], [], []
    for r in res.itertuples():
        notes = []
        if (r.n_actuations or 0) == 0:
            st = "cannot classify: no actuations"
            rv, rs = True, "no actuations"
        elif r.health_flag == "failed" and r.health_reason != "near_zero_volume":
            st = status_for_user("failed", r.health_reason)
            rv, rs = True, r.health_reason
        elif not (r.n_candidate_phases or 0):
            st = ("cannot classify: no phase begin-green (event 1) records in this window")
            rv, rs = True, "no phase events"
        elif pd.isna(r.phase_pred):
            st = "cannot classify: no usable detector/phase evidence in this window"
            rv, rs = True, "no prediction"
        elif (r.n_actuations or 0) < min_actuations:
            n = int(r.n_actuations)
            st = (f"not enough data: {n} actuation" + ("" if n == 1 else "s") +
                  f" in sample (need >= {min_actuations})")
            rv, rs = True, "not enough data"
        elif min_prob > 0 and (r.phase_prob or 0) < min_prob:
            st = (f"not confident enough: phase probability "
                  f"{float(r.phase_prob or 0):.2f} (need >= {min_prob:.2f})")
            rv, rs = True, "not confident enough"
        else:
            rs = ""
            if (r.n_actuations or 0) < LOW_ACTUATIONS:
                n = int(r.n_actuations)
                notes.append(f"ok - low evidence ({n} actuation" +
                             ("" if n == 1 else "s") + ")")
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
            if r.health_flag in ("suspect", "failed"):
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
    # refused channels carry no answer -- only a guess
    dead = (res.status.str.startswith("cannot classify") |
            res.status.str.startswith("not enough data") |
            res.status.str.startswith("not confident enough"))
    res.loc[dead, ["phase_pred", "phase_prob", "phase_2nd", "phase_2nd_prob",
                   "function_pred", "function_prob", "phase_margin"] + PROB_COLS] = np.nan
    if model_note:
        res["status"] = res.status + "; " + model_note

    for c in ("phase_pred", "phase_2nd", "phase_guess"):
        res[c] = res[c].astype("Int64")
    res["n_actuations"] = res.n_actuations.fillna(0).astype("Int64")
    res["minutes_of_data"] = res.minutes_of_data.astype(float).round(2)
    return res[OUT_COLS + EXTRA_COLS].sort_values(["DeviceId", "Detector"]).reset_index(
        drop=True)


# ---------------------------------------------------------------- entry points
def predict(events, start=None, end=None,
            device_ids=None, model_dir=None, threads: int = 4, memory: str = "4GB",
            chunk_signals: int | None = None, verbose: bool = False,
            min_actuations: int = MIN_ACTUATIONS,
            min_prob: float = MIN_PROB) -> pd.DataFrame:
    """Raw hi-res events -> one row per detector channel.  Never raises on thin data.

    Parameters
    ----------
    events         pandas DataFrame, or a path / glob to parquet or csv, with columns
                   DeviceId, Timestamp, EventId, Parameter (lowercase variants accepted).
    start, end     optional timestamp strings; `end` is exclusive.
    chunk_signals  process the signals in groups of this many to bound peak memory.
    min_actuations below this many detector ON events no answer is given.
    min_prob       below this top-phase probability no answer is given (0 = off).
                   The model's raw opinion is kept in phase_guess / function_guess.
    """
    global _VERBOSE
    _VERBOSE = verbose
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not (model_dir / "phase_lgbm_v4.json").exists():
        raise FileNotFoundError(f"no models in {model_dir}")

    if chunk_signals:
        ids = device_ids or list_signals(events, start, end, threads, memory)
        if len(ids) > chunk_signals:
            parts = [predict(events, start, end, ids[i:i + chunk_signals],
                             model_dir, threads, memory, None, verbose, min_actuations,
                             min_prob)
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
        if df is not None and len(df):
            log(f"features {df.shape}, similarity {sim.shape}")
            df = score(df, sim, model_dir)
            ph = df[["DeviceId", "Detector", "cand_phase", "prob"]].copy()
            s = ph.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
            ph["prob"] = ph.prob / s.replace(0, np.nan)
            ph = ph.dropna(subset=["prob"])
            try:
                fn = score_function(df, model_dir)
            except Exception as exc:                              # pragma: no cover
                note = (note + "; " if note else "") + \
                    f"function model unavailable ({type(exc).__name__})"
        return _assemble(univ, facts, hf, ph, fn, note, min_actuations, min_prob)
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
        model_dir=None, threads: int = 4,
        memory: str = "4GB", chunk_signals: int | None = None,
        min_actuations: int = MIN_ACTUATIONS,
        min_prob: float = MIN_PROB) -> pd.DataFrame:
    """CLI helper: predict and write a CSV."""
    t0 = time.time()
    res = predict(events, start, end, device_ids, model_dir, threads,
                  memory, chunk_signals, True, min_actuations, min_prob)
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
    ap.add_argument("--models", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="4GB")
    ap.add_argument("--chunk-signals", type=int, default=None,
                    help="process this many signals at a time to bound memory")
    ap.add_argument("--no-lightgbm", action="store_true",
                    help="score with the pure-numpy tree evaluator (no lightgbm / scipy needed)")
    ap.add_argument("--min-actuations", type=int, default=MIN_ACTUATIONS,
                    help="below this many detector ON events in the sample, report "
                         "'not enough data' instead of an answer (default %(default)s; "
                         "use 1 to always answer)")
    ap.add_argument("--min-prob", type=float, default=MIN_PROB,
                    help="below this top-phase probability, report 'not confident enough' "
                         "instead of an answer (default %(default)s = off; the measured "
                         "alternative service is --min-actuations 1 --min-prob 0.9)")
    a = ap.parse_args()
    if a.no_lightgbm:
        set_backend("numpy")
    ids = [s.strip() for s in a.device_ids.split(",")] if a.device_ids else None
    run(a.events, a.out, ids, a.start, a.end, Path(a.models),
        a.threads, a.memory, a.chunk_signals, a.min_actuations, a.min_prob)


if __name__ == "__main__":
    main()
