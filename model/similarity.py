"""Number-free cross-detector structure, the input to joint per-signal decoding.

For every (signal, window) we measure how similarly two detector channels actuate *in time*:
detectors on the same approach see the same vehicles / the same platoon within the same
second, whereas the opposing approach of a concurrent pair (the 2 <-> 6 problem) is
independent even though both are green at the same time.  This is the only signal that can
separate two phases whose greens always coincide.

Similarity = phi coefficient over `BIN_SECS`-second bins of "detector had an actuation":

    phi(a,b) = (n_ab*N - n_a*n_b) / sqrt(n_a (N-n_a) n_b (N-n_b))

Output: DeviceId, win, Detector, other, phi, n_common (top-K neighbours per detector).

No phase number and no channel->phase knowledge is used; channel *adjacency* is left to
the decoder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import log  # noqa: F401  (used by the research build driver)

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
