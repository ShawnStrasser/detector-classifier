"""The GRU's input side: raw events -> interval streams -> the 1 s, 9-channel raster.

The neural half of `models/final_v2` reads the same raw events as the trees, but it needs
them as intervals on a millisecond timeline rather than as engineered features.  The
definitions here are exactly the ones the network was TRAINED on (`src/build_cache.py`
-> `src/neural/ncache2.py`), and they are computed from the de-duplicated `ev` table
directly so they cannot drift apart from it:

    detector ON   event 82 -> the next 81 on that channel
    green         event 1  -> event 8, else event 10, else the next event 1
    yellow        event 8  -> event 10, else event 11
    red clearance event 10 -> event 11
    call          event 43 -> the next 44 on that phase
    coordinated   event 131 with a pattern of 1..253, until the next 131
    candidates    every phase with a Begin Green (event 1) in the sample

`render` then turns those intervals into the blocks the network reads, and `assemble`
expands them into one raster per (detector, candidate phase) pair.  A sample longer than
`CHUNK_MS` is cut by `split_range` into pieces of at most 30 minutes -- the window the
network was trained on -- and the pieces are pooled by the mean log-probability.

Only numpy, pandas and the caller's DuckDB connection are used here -- no torch, no
lightgbm, no onnxruntime (the network itself lives in `src/gru_onnx.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_MS = 1000                      # 1 s raster
CHUNK_MS = 30 * 60 * 1000          # a sample is scored in pieces of at most 30 minutes
N_CH = 9

SQL_DET = """
WITH e AS (SELECT DeviceId, Parameter::INT AS ch, Timestamp AS ts, EventId
           FROM ev WHERE EventId IN (81,82)),
     d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
           WINDOW w AS (PARTITION BY DeviceId, ch
                        ORDER BY ts, CASE WHEN EventId=82 THEN 0 ELSE 1 END))
SELECT DeviceId, ch, epoch_ms(ts)/1000.0 AS a, epoch_ms(nts)/1000.0 AS b
FROM d WHERE EventId = 82 AND nev = 81 AND nts IS NOT NULL
ORDER BY DeviceId, ch, a"""

SQL_CYC = """
WITH g AS (
  SELECT DeviceId, Parameter::INT AS p, Timestamp AS t, EventId,
         SUM(CASE WHEN EventId=1 THEN 1 ELSE 0 END) OVER (
             PARTITION BY DeviceId, Parameter
             ORDER BY t, CASE EventId WHEN 1 THEN 0 WHEN 8 THEN 1 WHEN 10 THEN 2 ELSE 3 END
             ROWS UNBOUNDED PRECEDING) AS cyc
  FROM ev WHERE EventId IN (1,8,10,11) AND Parameter BETWEEN 1 AND 16
), c AS (
  SELECT DeviceId, p, cyc,
         min(t) FILTER (EventId=1)  AS green_start,
         min(t) FILTER (EventId=8)  AS yellow_start,
         min(t) FILTER (EventId=10) AS red_start,
         min(t) FILTER (EventId=11) AS redclr_end
  FROM g WHERE cyc > 0 GROUP BY 1,2,3
), n AS (
  SELECT *, LEAD(green_start) OVER (PARTITION BY DeviceId, p ORDER BY green_start)
              AS next_green
  FROM c WHERE green_start IS NOT NULL
)
SELECT DeviceId, p,
       epoch_ms(green_start)/1000.0 AS g0,
       epoch_ms(coalesce(yellow_start, red_start, next_green))/1000.0 AS g1,
       epoch_ms(yellow_start)/1000.0 AS y0,
       epoch_ms(coalesce(red_start, redclr_end))/1000.0 AS y1,
       epoch_ms(red_start)/1000.0 AS r0,
       epoch_ms(redclr_end)/1000.0 AS r1
FROM n ORDER BY DeviceId, p, g0"""

SQL_CALL = """
WITH e AS (SELECT DeviceId, Parameter::INT AS p, Timestamp AS ts, EventId
           FROM ev WHERE EventId IN (43,44) AND Parameter BETWEEN 1 AND 16),
     d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
           WINDOW w AS (PARTITION BY DeviceId, p
                        ORDER BY ts, CASE WHEN EventId=43 THEN 0 ELSE 1 END))
SELECT DeviceId, p, epoch_ms(ts)/1000.0 AS a, epoch_ms(nts)/1000.0 AS b
FROM d WHERE EventId = 43 AND nev = 44 AND nts IS NOT NULL
ORDER BY DeviceId, p, a"""

SQL_COORD = """
SELECT DeviceId, epoch_ms(Timestamp)/1000.0 AS a,
       epoch_ms(LEAD(Timestamp) OVER (PARTITION BY DeviceId ORDER BY Timestamp))/1000.0 AS b,
       (Parameter BETWEEN 1 AND 253) AS is_coord
FROM ev WHERE EventId = 131 ORDER BY DeviceId, Timestamp"""

SQL_CAND = """
SELECT DISTINCT DeviceId, Parameter::INT AS p FROM ev
WHERE EventId = 1 AND Parameter BETWEEN 1 AND 16 ORDER BY DeviceId, p"""


def _ptr(keys: np.ndarray, order) -> np.ndarray:
    ptr = np.zeros(len(order) + 1, dtype=np.int64)
    pos = {k: i for i, k in enumerate(order)}
    cnt = np.zeros(len(order), dtype=np.int64)
    if len(keys):
        u, c = np.unique(keys, return_counts=True)
        for k, n in zip(u, c):
            if k in pos:
                cnt[pos[k]] = n
    ptr[1:] = np.cumsum(cnt)
    return ptr.astype(np.int32)


def _ms(x, t0_ms: int, span: int) -> np.ndarray:
    v = np.rint(np.asarray(x, dtype=np.float64) * 1000.0 - t0_ms)
    return np.clip(v, -1000, span + 1000).astype(np.int32)


def build_streams(con, t0_ms: int, t1_ms: int) -> dict:
    """-> {DeviceId: interval bundle for `render` below}.

    `t0_ms` / `t1_ms` are epoch milliseconds bracketing the sample; every stored time is
    an int32 offset from `t0_ms`, so a window is cut with integer arithmetic only.
    """
    span = int(t1_ms - t0_ms)
    cands = con.sql(SQL_CAND).df()
    dets = con.sql(SQL_DET).df()
    cyc = con.sql(SQL_CYC).df()
    calls = con.sql(SQL_CALL).df()
    coord = con.sql(SQL_COORD).df()

    gc = {k: v for k, v in cands.groupby("DeviceId", sort=False)}
    gd = {k: v for k, v in dets.groupby("DeviceId", sort=False)}
    gy = {k: v for k, v in cyc.groupby("DeviceId", sort=False)}
    gl = {k: v for k, v in calls.groupby("DeviceId", sort=False)}
    go = {k: v for k, v in coord.groupby("DeviceId", sort=False)}

    out = {}
    for dev in sorted(set(cands.DeviceId) | set(dets.DeviceId)):
        c = gc.get(dev)
        cand = (np.sort(c.p.astype(int).unique()) if c is not None
                else np.zeros(0, dtype=np.int64))
        z = {"cand": cand.astype(np.int16)}
        d = gd.get(dev)
        if d is None or not len(d):
            z.update(det_ch=np.zeros(0, np.int16), det_ptr=np.zeros(1, np.int32),
                     det_on=np.zeros(0, np.int32), det_off=np.zeros(0, np.int32))
        else:
            ch = np.sort(d.ch.astype(int).unique())
            z["det_ch"] = ch.astype(np.int16)
            z["det_ptr"] = _ptr(d.ch.astype(int).to_numpy(), list(ch))
            z["det_on"] = _ms(d.a, t0_ms, span)
            z["det_off"] = _ms(d.b, t0_ms, span)
        p, l = gy.get(dev), gl.get(dev)
        for tag, src, c0, c1 in (("g", p, "g0", "g1"), ("y", p, "y0", "y1"),
                                 ("r", p, "r0", "r1"), ("c", l, "a", "b")):
            ons, offs, ptr = [], [], [0]
            pv = None if src is None else src.p.astype(int).to_numpy()
            for ph in cand:
                if src is None:
                    ptr.append(ptr[-1])
                    continue
                s = src[pv == int(ph)]
                a = pd.to_numeric(s[c0], errors="coerce").to_numpy(dtype=np.float64)
                b = pd.to_numeric(s[c1], errors="coerce").to_numpy(dtype=np.float64)
                ok = np.isfinite(a) & np.isfinite(b) & (b > a)
                a, b = a[ok], b[ok]
                o = np.argsort(a, kind="stable")
                ons.append(a[o]); offs.append(b[o])
                ptr.append(ptr[-1] + int(len(a)))
            z[tag + "_on"] = _ms(np.concatenate(ons) if ons else np.zeros(0), t0_ms, span)
            z[tag + "_off"] = _ms(np.concatenate(offs) if offs else np.zeros(0), t0_ms, span)
            z[tag + "_ptr"] = np.array(ptr, dtype=np.int32)
        co = go.get(dev)
        if co is not None and len(co):
            k = co[co.is_coord.fillna(False).astype(bool)]
            a = _ms(k.a, t0_ms, span)
            b = _ms(k.b.fillna((t1_ms + 1000.0) / 1000.0), t0_ms, span)
            m = b > a
            z["co_on"], z["co_off"] = a[m], b[m]
        else:
            z["co_on"] = np.zeros(0, np.int32)
            z["co_off"] = np.zeros(0, np.int32)
        out[dev] = z
    return out


# ------------------------------------------------------------- raster pieces
def cover(on: np.ndarray, off: np.ndarray, w0: int, w1: int, T: int,
          bw: int = BIN_MS) -> np.ndarray:
    """Fraction of each 1 s bin covered by the sorted, disjoint intervals [on, off)."""
    out = np.zeros(T, dtype=np.float32)
    if on.size == 0:
        return out
    i0 = int(np.searchsorted(off, w0, "right"))
    i1 = int(np.searchsorted(on, w1, "left"))
    if i1 <= i0:
        return out
    a = (on[i0:i1].astype(np.float64) - w0) / bw
    b = (off[i0:i1].astype(np.float64) - w0) / bw
    np.clip(a, 0.0, T, out=a)
    np.clip(b, 0.0, T, out=b)
    cum = np.concatenate(([0.0], np.cumsum(b - a)))
    x = np.arange(T + 1, dtype=np.float64)
    j = np.searchsorted(a, x, "right")
    G = np.zeros(T + 1)
    m = j >= 1
    jm = j[m] - 1
    G[m] = cum[jm] + np.minimum(x[m], b[jm]) - a[jm]
    out[:] = np.diff(G)
    return np.clip(out, 0.0, 1.0)


def onrate(on: np.ndarray, w0: int, w1: int, T: int, bw: int = BIN_MS) -> np.ndarray:
    """Number of detector ON onsets in each bin."""
    if on.size == 0:
        return np.zeros(T, dtype=np.float32)
    i0 = int(np.searchsorted(on, w0, "left"))
    i1 = int(np.searchsorted(on, w1, "left"))
    if i1 <= i0:
        return np.zeros(T, dtype=np.float32)
    idx = ((on[i0:i1].astype(np.int64) - w0) // bw).astype(np.int64)
    return np.bincount(idx, minlength=T)[:T].astype(np.float32)


def stream_arrays() -> str:
    """The interval bundle `render` expects (all int32 milliseconds from a common T0):

        cand    int   [K]     candidate phases (every phase with a Begin Green)
        det_ch  int   [D]     detector channels present
        det_ptr int   [D+1]   offsets into det_on / det_off
        det_on, det_off       detector ON intervals (82 -> next 81)
        g_ptr, g_on, g_off    per candidate phase: green intervals
        y_ptr, y_on, y_off    per candidate phase: yellow intervals
        r_ptr, r_on, r_off    per candidate phase: red-clearance intervals
        c_ptr, c_on, c_off    per candidate phase: call intervals (43 -> next 44)
        co_on, co_off         intervals during which the signal is coordinated
    """
    return stream_arrays.__doc__


def render(z: dict, w0: int, dets, T: int, bw: int = BIN_MS):
    """-> det [D,2,T], ph [K,4,T], sig [3,T] float32 blocks (shared across pairs)."""
    w1 = w0 + T * bw
    K = len(z["cand"])
    ph = np.zeros((K, 4, T), dtype=np.float32)
    for tag, row in (("g", 0), ("y", 1), ("r", 2), ("c", 3)):
        ptr, on, off = z[tag + "_ptr"], z[tag + "_on"], z[tag + "_off"]
        for k in range(K):
            s, e = int(ptr[k]), int(ptr[k + 1])
            if e > s:
                ph[k, row] = cover(on[s:e], off[s:e], w0, w1, T, bw)
    sig = np.zeros((3, T), dtype=np.float32)
    sig[0] = ph[:, 0].sum(0)
    sig[1] = ph[:, 3].sum(0)
    sig[2] = cover(z["co_on"], z["co_off"], w0, w1, T, bw)

    chpos = z.get("_chpos")
    if chpos is None:
        chpos = {int(c): i for i, c in enumerate(z["det_ch"])}
        z["_chpos"] = chpos
    D = len(dets)
    det = np.zeros((D, 2, T), dtype=np.float32)
    nact = np.zeros(D, dtype=np.float32)
    ptr, on, off = z["det_ptr"], z["det_on"], z["det_off"]
    for i, ch in enumerate(dets):
        k = chpos.get(int(ch))
        if k is None:
            continue
        s, e = int(ptr[k]), int(ptr[k + 1])
        if e <= s:
            continue
        det[i, 0] = cover(on[s:e], off[s:e], w0, w1, T, bw)
        det[i, 1] = onrate(on[s:e], w0, w1, T, bw)
        nact[i] = det[i, 1].sum()
    det[:, 1] = np.clip(det[:, 1], 0, 4) / 2.0
    return det, ph, sig, nact


def assemble(det: np.ndarray, ph: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """All (detector, candidate) pairs of one signal -> [D*K, 9, T], pair index d*K + k."""
    D, _, T = det.shape
    K = ph.shape[0]
    x = np.empty((D, K, N_CH, T), dtype=np.float32)
    x[:, :, 0:2] = det[:, None, :, :]
    x[:, :, 2:6] = ph[None, :, :, :]
    x[:, :, 6] = (sig[0][None, None] - ph[:, 0][None]) / 2.0
    den = float(max(K, 2) - 1)
    x[:, :, 7] = (sig[1][None, None] - ph[:, 3][None]) / den
    x[:, :, 8] = sig[2][None, None]
    return x.reshape(D * K, N_CH, T)


def split_range(t0_ms: int, t1_ms: int, chunk_ms: int = CHUNK_MS,
                max_chunks: int = 0) -> list[tuple[int, int]]:
    """Cut [t0, t1) into pieces of at most 30 minutes (the network's native window).

    A period shorter than that stays one short piece -- the network is recurrent and
    accepts any number of 1 s steps."""
    total = max(int(t1_ms) - int(t0_ms), 1000)
    if total <= chunk_ms:
        return [(int(t0_ms), int(t1_ms))]
    n = total // chunk_ms
    rem = total - n * chunk_ms
    out = [(int(t0_ms) + i * chunk_ms, int(t0_ms) + (i + 1) * chunk_ms) for i in range(n)]
    if rem >= 60_000:
        out.append((int(t0_ms) + n * chunk_ms, int(t1_ms)))
    if max_chunks and len(out) > max_chunks:
        idx = np.linspace(0, len(out) - 1, max_chunks).round().astype(int)
        out = [out[i] for i in idx]
    return out
