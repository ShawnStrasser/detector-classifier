"""Stage 11 step 0 -- build the v3 *function* extra features (yellow/red-clearance and
pairwise actuation lag) for the Sept-2026 STAGING data.

Stage 10 built the base pair features, the v2 extras and the detector-similarity table for
the staging cache (`dc_work/official/stg/features/*`), but not `features_v3.py`'s two
tables, which the function model needs.  This script runs exactly `features_v3.SQL_YR` /
`SQL_LAG` against the staging cache, over the same 22-window variant-B mix
(`src/official/windows_stg.py`), restricted to the signals that carry FUNCTION labels.

Nothing under `dc_work/features/` or `dc_work/official/stg/features/` is touched: the two
tables land in `dc_work/function_v4/feat_stg/`.

    python src/function_v4_prep.py --what both
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
_HOME_WORK = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
STG = _HOME_WORK / "official" / "stg"
OUT_DIR = _HOME_WORK / "function_v4" / "feat_stg"

# every module below resolves its paths from DC_WORK at import time
os.environ["DC_WORK"] = str(STG)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "official"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import windows_stg  # noqa: E402,F401  (registers stgmixed / stgshortb / stgall)
import features  # noqa: E402
import features_v3 as f3  # noqa: E402
from common import CACHE, connect  # noqa: E402
from features import load_chunk  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def target_signals() -> list[str]:
    """Signals that (a) have staging data and (b) carry at least one function label and
    (c) are neither TEST nor NEWTEST (both hold-outs are removed here, before any read)."""
    cfg = pd.read_parquet(_HOME_WORK / "data" / "labels" / "detector_config_current.parquet")
    lab = set(cfg.DeviceId.str.lower())
    test = set(pd.read_csv(REPO_ROOT / "data" / "splits" / "test_config.csv")
               .DeviceId.str.lower())
    newtest = set(pd.read_csv(_HOME_WORK / "official" / "newtest_signals.csv")
                  .DeviceId.str.lower())
    have = set(pd.read_csv(STG / "signals.csv").DeviceId.str.lower())
    keep = (lab & have) - test - newtest
    assert not (keep & test) and not (keep & newtest)
    log(f"label signals {len(lab)}, staging {len(have)}, TEST {len(test)} and "
        f"NEWTEST {len(newtest)} removed -> {len(keep)} signals")
    return sorted(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="both", choices=["both", "yr", "lag"])
    ap.add_argument("--windows", default="stgall")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    devs = target_signals()
    if a.limit:
        devs = devs[:a.limit]
    con = connect(memory_limit="12GB", threads=a.threads)
    con.execute(f"SET temp_directory='{(STG / 'tmp').as_posix()}'")
    wins = features.WINDOW_SETS[a.windows]
    log(f"staging features_v3 [{a.what}]: {len(devs)} signals x {len(wins)} windows "
        f"(cache {CACHE})")

    yr_parts, lag_parts, t0 = [], [], time.time()
    nch = (len(devs) + a.chunk - 1) // a.chunk
    for i in range(0, len(devs), a.chunk):
        ch = devs[i:i + a.chunk]
        t1 = time.time()
        load_chunk(con, ch)
        f3.load_cycles(con, ch)
        dm = con.sql("SELECT * FROM devmap").df()
        for w in wins:
            w0 = (w["t0"] - pd.Timestamp("1970-01-01")).total_seconds()
            w1 = w0 + w["secs"]
            con.execute("CREATE OR REPLACE TEMP TABLE onev AS SELECT * FROM onev_all "
                        f"WHERE t_on >= {w0} AND t_on < {w1}")
            f3.window_cycles(con, w0, w1)
            if a.what in ("both", "yr"):
                r = f3.build(con, w["win"], dm, f3.SQL_YR)
                if len(r):
                    yr_parts.append(r)
            if a.what in ("both", "lag"):
                r = f3.build(con, w["win"], dm, f3.SQL_LAG)
                if len(r):
                    lag_parts.append(r)
        log(f"chunk {i // a.chunk + 1}/{nch} {time.time() - t1:.1f}s "
            f"elapsed={time.time() - t0:.0f}s")

    def _write(parts, ren, path):
        df = pd.concat(parts, ignore_index=True).rename(columns=ren)
        df = df.replace([np.inf, -np.inf], np.nan)
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        for c in ("Detector", "cand_phase", "other"):
            if c in df.columns:
                df[c] = df[c].astype(np.int16)
        df.to_parquet(path, index=False)
        log(f"wrote {path}: {len(df):,} rows x {df.shape[1]} cols")

    if yr_parts:
        _write(yr_parts, {"det": "Detector", "p": "cand_phase"},
               OUT_DIR / "func_yr_extra_stg.parquet")
    if lag_parts:
        _write(lag_parts, {"det": "Detector", "oth": "other"},
               OUT_DIR / "det_lag_stg.parquet")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
