"""Stage 09: rebuild the pair features with the bad periods MASKED OUT.

The only change to the feature pipeline is a filter on the detector-interval table
(`onev_all`) that `features.load_chunk` builds: every ON that overlaps a bad interval
from `health2.find_bad_intervals` is deleted before any window is cut.  Nothing else in
`features.py` / `features_v2.py` is touched, so the masked table is directly comparable
with `pair_features_windows.parquet`.

    python src/health2_features.py --windows h2            # base + v2 extras, masked
    python src/health2_features.py --windows h2 --no-mask  # control: same code, no mask
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, DC_WORK, FOLDS_CSV, connect  # noqa: E402
import features  # noqa: E402
import features_v2 as f2  # noqa: E402
import health2 as h2  # noqa: E402
from features import _w  # noqa: E402

OUT = DC_WORK / "health2"

# 30 min (peak + off-peak), 1 h (night), 6 h and the full 72 h: the protocol's
# "at least 30 min / 6 h / 72 h", sampled at different times of day.
WINDOWS_H2 = [
    _w("m30_a", "2024-12-02 07:30:00", 1800),
    _w("m30_c", "2024-12-03 21:30:00", 1800),
    _w("m30_d", "2024-12-04 16:45:00", 1800),
    _w("h1_b", "2024-12-03 02:00:00", 3600),
    _w("h6_a", "2024-12-03 06:00:00", 6 * 3600),
    _w("h6_b", "2024-12-04 12:00:00", 6 * 3600),
    _w("full72", "2024-12-02 00:00:00", 72 * 3600),
]
features.WINDOW_SETS["h2"] = WINDOWS_H2

_ORIG_LOAD_CHUNK = features.load_chunk
_BAD_READY = False


def _install_mask(con, bad: pd.DataFrame) -> None:
    """Register the bad-interval table once, as epoch seconds (what `onev_all` uses)."""
    global _BAD_READY
    b = bad[["DeviceId", "Detector", "start", "end"]].copy()
    b["t0"] = (b["start"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    b["t1"] = (b["end"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    con.register("_bad_src", b[["DeviceId", "Detector", "t0", "t1"]])
    con.execute("CREATE OR REPLACE TABLE _bad AS SELECT DeviceId, "
                "Detector::SMALLINT AS det, t0, t1 FROM _bad_src")
    _BAD_READY = True


def masked_load_chunk(con, devs):
    """`features.load_chunk`, then delete every masked ON from `onev_all`."""
    _ORIG_LOAD_CHUNK(con, devs)
    if not _BAD_READY:
        return
    con.execute("""DELETE FROM onev_all o WHERE EXISTS (
        SELECT 1 FROM _bad b JOIN devmap m USING (DeviceId)
        WHERE m.dev = o.dev AND (b.det IS NULL OR b.det = o.det)
          AND o.t_on < b.t1 AND o.t_off > b.t0)""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="h2")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--placebo", action="store_true",
                    help="control: keep each bad interval's detector and duration but "
                         "move it to a random time in the sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasons", default=",".join(h2.MASK_REASONS))
    ap.add_argument("--tag", default="masked")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    devs = pd.read_csv(FOLDS_CSV).DeviceId.sort_values().tolist()
    if a.limit:
        devs = devs[:a.limit]
    wins = features.WINDOW_SETS[a.windows]
    con = connect(memory_limit="6GB", threads=a.threads)

    if not a.no_mask:
        bad = pd.read_parquet(OUT / "bad_intervals.parquet")
        bad = bad[bad.reason.isin(a.reasons.split(","))]
        if a.placebo:
            rng = np.random.default_rng(a.seed)
            t0s = pd.Timestamp("2024-12-02")
            span = 72 * 3600 - bad.secs.clip(upper=72 * 3600 - 1)
            off = rng.random(len(bad)) * span.to_numpy()
            bad = bad.assign(
                start=t0s + pd.to_timedelta(off, unit="s"),
                end=t0s + pd.to_timedelta(off + bad.secs.to_numpy(), unit="s"))
        _install_mask(con, bad)
        print(f"masking {len(bad):,} {'PLACEBO ' if a.placebo else ''}"
              f"bad intervals ({a.reasons})")
    features.load_chunk = masked_load_chunk

    base, extra, t0 = [], [], time.time()
    nch = (len(devs) + a.chunk - 1) // a.chunk
    for i in range(0, len(devs), a.chunk):
        ch = devs[i:i + a.chunk]
        features.load_chunk(con, ch)
        for w in wins:
            w0 = (w["t0"] - pd.Timestamp("1970-01-01")).total_seconds()
            features.apply_window(con, w0, w0 + w["secs"])
            r = features.build_window(con, w["win"], w["secs"])
            if len(r):
                base.append(r)
            r2 = f2.build_window(con, w["win"], w["secs"])
            if r2 is not None and len(r2):
                extra.append(r2)
        if (i // a.chunk) % 5 == 0:
            print(f"  chunk {i//a.chunk+1}/{nch} elapsed={time.time()-t0:.0f}s", flush=True)

    df = features.finalise(pd.concat(base, ignore_index=True))
    df.to_parquet(OUT / f"pair_features_{a.tag}.parquet", index=False)
    ex = pd.concat(extra, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for c in ex.columns:
        if ex[c].dtype == np.float64:
            ex[c] = ex[c].astype(np.float32)
    ex["Detector"] = ex["Detector"].astype(np.int16)
    ex["cand_phase"] = ex["cand_phase"].astype(np.int16)
    ex.to_parquet(OUT / f"pair_features_v2_{a.tag}.parquet", index=False)
    print(f"wrote {len(df):,} base rows / {len(ex):,} v2 rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
