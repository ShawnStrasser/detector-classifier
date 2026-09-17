"""Stage 04 step 3a: number-free cross-detector structure for joint per-signal decoding.

For every (signal, window) we measure how similarly two detector channels actuate *in time*:
detectors on the same approach see the same vehicles / the same platoon within the same
second, whereas the opposing approach of a concurrent pair (the 2 <-> 6 problem) is
independent even though both are green at the same time.  This is the only signal that can
separate two phases whose greens always coincide.

Similarity = phi coefficient over `BIN_SECS`-second bins of "detector had an actuation":

    phi(a,b) = (n_ab*N - n_a*n_b) / sqrt(n_a (N-n_a) n_b (N-n_b))

Output `dc_work/features/det_similarity.parquet`:
    DeviceId, win, Detector, other, phi, n_common   (top-K neighbours per detector)

No phase number and no channel->phase knowledge is used; channel *adjacency* is left to the
decoder, which reports results with and without it.

    python src/cross_detector.py --windows mixed
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
from features import WINDOW_SETS, log  # noqa: E402

BIN_SECS = 2.0
TOPK = 8

SQL_SIM = f"""
WITH b AS (
  SELECT DISTINCT dev, det, (t_on / {BIN_SECS})::BIGINT AS bin FROM onev
), na AS (
  SELECT dev, det, count(*) AS n FROM b GROUP BY 1,2
), pa AS (
  SELECT x.dev, x.det AS a, y.det AS c, count(*) AS n_ab
  FROM b x JOIN b y ON x.dev = y.dev AND x.bin = y.bin AND x.det < y.det
  GROUP BY 1,2,3
)
SELECT p.dev, p.a, p.c, p.n_ab, ka.n AS na, kc.n AS nc
FROM pa p JOIN na ka ON ka.dev=p.dev AND ka.det=p.a
          JOIN na kc ON kc.dev=p.dev AND kc.det=p.c
WHERE p.n_ab >= 3
"""


def build_window(con, win: str, secs: float) -> pd.DataFrame:
    d = con.sql(SQL_SIM).df()
    if not len(d):
        return pd.DataFrame()
    N = max(secs / BIN_SECS, 2.0)
    na, nc, nab = d.na.to_numpy(float), d.nc.to_numpy(float), d.n_ab.to_numpy(float)
    denom = np.sqrt(np.clip(na * (N - na) * nc * (N - nc), 1e-9, None))
    d["phi"] = (nab * N - na * nc) / denom
    devmap = con.sql("SELECT * FROM devmap").df()
    d = d.merge(devmap, on="dev", how="left")
    # symmetrise
    x = d[["DeviceId", "a", "c", "phi", "n_ab"]].rename(columns={"a": "Detector", "c": "other"})
    y = d[["DeviceId", "c", "a", "phi", "n_ab"]].rename(columns={"c": "Detector", "a": "other"})
    out = pd.concat([x, y], ignore_index=True).rename(columns={"n_ab": "n_common"})
    out = out.sort_values(["DeviceId", "Detector", "phi"], ascending=[True, True, False])
    out = out.groupby(["DeviceId", "Detector"], sort=False).head(TOPK)
    out["win"] = win
    return out


def main() -> None:
    from features import apply_window, load_chunk
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--windows", default="mixed", choices=list(WINDOW_SETS))
    ap.add_argument("--out", default="det_similarity.parquet")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    con = connect(threads=a.threads)
    devs = con.sql(f"SELECT DeviceId FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}')"
                   " ORDER BY DeviceId").df()["DeviceId"].tolist()
    if a.limit:
        devs = devs[:a.limit]
    wins = WINDOW_SETS[a.windows]
    log(f"similarity: {len(devs)} signals x {len(wins)} windows, bin={BIN_SECS}s top{TOPK}")
    parts, t0 = [], time.time()
    nch = (len(devs) + a.chunk - 1) // a.chunk
    for i in range(0, len(devs), a.chunk):
        t1 = time.time()
        load_chunk(con, devs[i:i + a.chunk])
        for w in wins:
            w0 = (w["t0"] - pd.Timestamp("1970-01-01")).total_seconds()
            apply_window(con, w0, w0 + w["secs"])
            r = build_window(con, w["win"], w["secs"])
            if len(r):
                parts.append(r)
        log(f"chunk {i//a.chunk+1}/{nch} {time.time()-t1:.1f}s elapsed={time.time()-t0:.0f}s")
    df = pd.concat(parts, ignore_index=True)
    df["Detector"] = df.Detector.astype(np.int16)
    df["other"] = df.other.astype(np.int16)
    df["phi"] = df.phi.astype(np.float32)
    out = FEATURES / a.out
    df.to_parquet(out, index=False)
    log(f"wrote {out}: {len(df):,} rows, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
