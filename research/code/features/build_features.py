"""Build the training feature tables from the `dc_work` event cache.

The feature *definitions* ship with the model (`model/features.py`,
`model/features_partner.py`, `model/features_yellowred.py`, `model/similarity.py`) --
this file is the research-only half: the fixed time windows the study trained on, the
loader that pulls a chunk of signals out of the cache, and the command line that writes
the parquet tables under `%DC_WORK%/features/`.

    python research/code/features/build_features.py --what base --windows mixed \
        --out pair_features_windows.parquet
    python ... --what extra --windows mixed --out pair_features_v2_extra.parquet
    python ... --what sim   --windows mixed --out det_similarity.parquet
    python ... --what yrlag --windows mixed --out func_yr_extra.parquet \
        --lag-out det_lag.parquet

Every feature is a rate / share / lift, so it is duration-invariant; the amount of
evidence is exposed through `win_secs`, `det_n_on`, `det_on_per_hour`, `n_cycles`.
`DeviceId, Detector, cand_phase, win` are keys, never model inputs (`FEATURE_COLS`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CACHE, FEATURES, connect  # noqa: E402
import features  # noqa: E402
import features_partner  # noqa: E402
import features_yellowred as fyr  # noqa: E402
import similarity  # noqa: E402
from features import RANK_FEATS, finalise, log  # noqa: E402


# ------------------------------------------------------------------- windows
# (name, start timestamp, duration seconds).  Data span = 2024-12-02 .. 2024-12-04.
def _w(name, start, secs):
    return {"win": name, "t0": pd.Timestamp(start), "secs": float(secs)}


WINDOWS_FULL = [_w("full72", "2024-12-02 00:00:00", 72 * 3600)]

WINDOWS_MIXED = [
    # 30 min - AM peak, midday, evening off-peak, PM peak
    _w("m30_a", "2024-12-02 07:30:00", 1800),
    _w("m30_b", "2024-12-03 12:00:00", 1800),
    _w("m30_c", "2024-12-03 21:30:00", 1800),
    _w("m30_d", "2024-12-04 16:45:00", 1800),
    # 1 h - PM peak, deep night, mid-morning
    _w("h1_a", "2024-12-02 17:00:00", 3600),
    _w("h1_b", "2024-12-03 02:00:00", 3600),
    _w("h1_c", "2024-12-04 09:00:00", 3600),
    # 3 h
    _w("h3_a", "2024-12-02 06:00:00", 3 * 3600),
    _w("h3_b", "2024-12-03 14:00:00", 3 * 3600),
    # 6 h
    _w("h6_a", "2024-12-03 06:00:00", 6 * 3600),
    _w("h6_b", "2024-12-04 12:00:00", 6 * 3600),
    # 24 h
    _w("h24_a", "2024-12-02 00:00:00", 24 * 3600),
    _w("h24_b", "2024-12-04 00:00:00", 24 * 3600),
    # 72 h
    _w("full72", "2024-12-02 00:00:00", 72 * 3600),
]

# 5 and 10 minute anchors, added for the mixed-length training set (stage 05b/06)
WINDOWS_SHORT_B = [
    _w("m5_a", "2024-12-02 07:45:00", 300),
    _w("m5_b", "2024-12-03 12:20:00", 300),
    _w("m5_c", "2024-12-03 22:10:00", 300),
    _w("m5_d", "2024-12-04 17:05:00", 300),
    _w("m10_a", "2024-12-02 08:05:00", 600),
    _w("m10_b", "2024-12-03 13:00:00", 600),
    _w("m10_c", "2024-12-04 02:30:00", 600),
    _w("m10_d", "2024-12-04 17:20:00", 600),
]

WINDOW_SETS = {"full": WINDOWS_FULL, "mixed": WINDOWS_MIXED,
               "shortb": WINDOWS_SHORT_B,
               "mixedb": WINDOWS_MIXED + WINDOWS_SHORT_B}

# duration label used in the accuracy-vs-duration curve
DURATION_OF = {"m5": 5 / 60.0, "m10": 10 / 60.0, "m30": 0.5, "h1": 1.0, "h3": 3.0,
               "h6": 6.0, "h24": 24.0, "full72": 72.0}


def win_hours(win: str) -> float:
    return DURATION_OF["full72" if win == "full72" else win.split("_")[0]]


# the call-event feature family, used by the stage-01 ablations
CALL_FEATS = [c for c in RANK_FEATS if "call4" in c] + [
    "call43_fwd_035", "call43_fwd_1", "call44_fwd_035", "call44_fwd_1",
    "call43_rev_035", "call44_rev_frac", "call43_rev_per_on",
    "call43_per_cycle", "call44_per_cycle", "looks_recall", "n_call43"]

FEATURE_EXCLUDE = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase",
                   "Function", "fold", "y", "cyc"}


def FEATURE_COLS(df: pd.DataFrame) -> list[str]:
    """The legal model inputs of a feature frame: no key, no label-derived column."""
    return [c for c in df.columns
            if c not in FEATURE_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------- chunk loading
def load_chunk(con, devs: list[str]) -> None:
    """Load the full 3-day slice for a handful of signals into narrow temp tables.

    These are the same temp tables `model/predict.py` builds straight from raw events;
    here they come from the derived cache instead, which is far faster for training."""
    dev_list = ",".join("'" + d + "'" for d in devs)
    con.execute("CREATE OR REPLACE TEMP TABLE devmap AS "
                "SELECT DeviceId, row_number() OVER (ORDER BY DeviceId)::SMALLINT AS dev "
                f"FROM (SELECT unnest([{dev_list}]) AS DeviceId)")
    # explicit IN-lists so parquet zone maps / hive partitions are pruned
    IN = f"DeviceId IN ({dev_list})"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE onev_all AS
        SELECT m.dev, i.Detector::SMALLINT AS det,
               epoch_ms(i.t_on)/1000.0 AS t_on, epoch_ms(i.t_off)/1000.0 AS t_off,
               i.dur::FLOAT AS dur
        FROM read_parquet('{(CACHE/'det_intervals.parquet').as_posix()}') i
        JOIN devmap m USING (DeviceId) WHERE i.{IN}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cyc_all AS
        SELECT m.dev, c.Phase::SMALLINT AS p, c.cyc::INT AS cyc,
               epoch_ms(c.green_start)/1000.0 AS gs,
               epoch_ms(coalesce(c.yellow_start, c.red_start, c.next_green))/1000.0 AS ge,
               epoch_ms(coalesce(c.red_start, c.yellow_start))/1000.0 AS rs,
               epoch_ms(c.next_green)/1000.0 AS ng, c.green_secs::FLOAT AS green_secs
        FROM read_parquet('{(CACHE/'phase_cycles.parquet').as_posix()}') c
        JOIN devmap m USING (DeviceId) WHERE c.Phase BETWEEN 1 AND 16 AND c.{IN}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE gs_all AS
        SELECT m.dev, epoch_ms(g.t_start)/1000.0 AS t0, epoch_ms(g.t_end)/1000.0 AS t1, g.mask
        FROM read_parquet('{(CACHE/'green_state.parquet').as_posix()}') g
        JOIN devmap m USING (DeviceId) WHERE g.t_end IS NOT NULL AND g.{IN}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE coordiv AS
        SELECT m.dev, epoch_ms(c.t_start)/1000.0 AS t0, c.is_coord
        FROM read_parquet('{(CACHE/'coord_state.parquet').as_posix()}') c
        JOIN devmap m USING (DeviceId) WHERE c.{IN}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE calls_all AS
        SELECT m.dev, e.Parameter::SMALLINT AS p, e.EventId::SMALLINT AS ev,
               epoch_ms(e.Timestamp)/1000.0 AS t
        FROM read_parquet('{(CACHE/'events'/'**'/'*.parquet').as_posix()}', hive_partitioning=true) e
        JOIN devmap m USING (DeviceId)
        WHERE e.EventId IN (43,44) AND e.Parameter BETWEEN 1 AND 16 AND e.{IN}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE cand AS
        SELECT m.dev, unnest(s.cand_phases)::SMALLINT AS p
        FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}') s
        JOIN devmap m USING (DeviceId) WHERE s.{IN}""")


def load_cycles(con, devs: list[str]) -> None:
    """Full four-colour cycle table (`load_chunk` drops `redclr_end`)."""
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


def _epoch(w) -> float:
    return (w["t0"] - pd.Timestamp("1970-01-01")).total_seconds()


def _signals(con, limit: int) -> list[str]:
    devs = con.sql(f"SELECT DeviceId FROM "
                   f"read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}')"
                   " ORDER BY DeviceId").df()["DeviceId"].tolist()
    return devs[:limit] if limit else devs


def _shrink(df: pd.DataFrame, int16=("Detector", "cand_phase")) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    for c in int16:
        if c in df.columns:
            df[c] = df[c].astype(np.int16)
    return df


# --------------------------------------------------------------- the builders
def build(what: str, out: str, lag_out: str, windows: str, threads: int,
          chunk: int, limit: int) -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    con = connect(threads=threads)
    devs = _signals(con, limit)
    wins = WINDOW_SETS[windows]
    log(f"{what}: {len(devs)} signals x {len(wins)} windows, chunk={chunk}")
    parts, lag_parts, t0 = [], [], time.time()
    nch = (len(devs) + chunk - 1) // chunk
    for i in range(0, len(devs), chunk):
        ch, t1 = devs[i:i + chunk], time.time()
        load_chunk(con, ch)
        if what == "yrlag":
            load_cycles(con, ch)
            dm = con.sql("SELECT * FROM devmap").df()
        for w in wins:
            w0, w1 = _epoch(w), _epoch(w) + w["secs"]
            if what == "yrlag":
                con.execute("CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM "
                            f"onev_all WHERE t_on >= {w0} AND t_on < {w1}")
                fyr.window_cycles(con, w0, w1)
                r = fyr.build(con, w["win"], dm, fyr.SQL_YR)
                if len(r):
                    parts.append(r)
                r = fyr.build(con, w["win"], dm, fyr.SQL_LAG)
                if len(r):
                    lag_parts.append(r)
                continue
            features.apply_window(con, w0, w1)
            if what == "base":
                r = features.build_window(con, w["win"], w["secs"])
            elif what == "extra":
                r = features_partner.build_window(con, w["win"], w["secs"])
            else:
                r = similarity.build_window(con, w["win"], w["secs"])
            if len(r):
                parts.append(r)
        log(f"chunk {i//chunk+1}/{nch} {time.time()-t1:.1f}s "
            f"elapsed={time.time()-t0:.0f}s")

    df = pd.concat(parts, ignore_index=True)
    del parts
    if what == "base":
        df = finalise(df)
    elif what == "yrlag":
        df = _shrink(df.rename(columns={"det": "Detector", "p": "cand_phase"}))
    elif what == "sim":
        df = _shrink(df, int16=("Detector", "other"))
        df["phi"] = df.phi.astype(np.float32)
    else:
        df = _shrink(df)
    df.to_parquet(FEATURES / out, index=False)
    log(f"wrote {FEATURES / out}: {len(df):,} rows x {df.shape[1]} cols, "
        f"{time.time()-t0:.0f}s")
    if lag_parts:
        dl = _shrink(pd.concat(lag_parts, ignore_index=True).rename(
            columns={"det": "Detector", "oth": "other"}), int16=("Detector", "other"))
        dl.to_parquet(FEATURES / lag_out, index=False)
        log(f"wrote {FEATURES / lag_out}: {len(dl):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="base",
                    choices=["base", "extra", "sim", "yrlag"],
                    help="base = model/features.py, extra = features_partner.py, "
                         "sim = similarity.py, yrlag = features_yellowred.py")
    ap.add_argument("--windows", default="mixed")
    ap.add_argument("--out", default="pair_features_windows.parquet")
    ap.add_argument("--lag-out", default="det_lag.parquet")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    build(a.what, a.out, a.lag_out, a.windows, a.threads, a.chunk, a.limit)


if __name__ == "__main__":
    main()
