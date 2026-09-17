"""Stage 03 (neural): compact per-signal interval cache for on-the-fly rasterisation.

Reads the stage-01 cache (already SELECT DISTINCT-deduped, Parameter<=64, allowed
codes only) and writes one small .npz per signal into DC_WORK/neural/sig/.

Everything is stored as int32 milliseconds since T0 = 2024-12-02 00:00:00 so that
a window can be cut with pure integer arithmetic and no float precision loss.

Arrays per signal
-----------------
cand          int8   [K]      candidate phases (every phase with an event 1)
det_ch        int16  [D]      detector channels present
det_ptr       int32  [D+1]    offsets into det_on/det_off
det_on/off    int32  [Nd]     detector ON intervals (82 -> next 81)
g_ptr,g_on/off  int32         per candidate phase: green intervals
y_ptr,y_on/off  int32         per candidate phase: yellow intervals
r_ptr,r_on/off  int32         per candidate phase: red-clearance intervals
c_ptr,c_on/off  int32         per candidate phase: call intervals (43 -> next 44)
co_on/co_off  int32  [Nc]     intervals during which the signal is coordinated
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, DC_WORK, connect  # noqa: E402

NEURAL = DC_WORK / "neural"
SIGDIR = NEURAL / "sig"
T0 = pd.Timestamp("2024-12-02 00:00:00")
SPAN_MS = 72 * 3600 * 1000


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _ptr_arrays(keys: np.ndarray, order: list) -> np.ndarray:
    """offsets into a table sorted by `keys`, for the given ordered key list."""
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


def build(devices: list[str] | None = None, chunk: int = 30) -> None:
    SIGDIR.mkdir(parents=True, exist_ok=True)
    con = connect(memory_limit="8GB", threads=8)

    meta = con.sql(f"""SELECT DeviceId, cand_phases
                       FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}')""").df()
    if devices is not None:
        meta = meta[meta.DeviceId.isin(devices)]
    devs = sorted(meta.DeviceId.tolist())
    cand_of = {r.DeviceId: sorted(int(p) for p in r.cand_phases if 1 <= int(p) <= 16)
               for r in meta.itertuples()}
    log(f"{len(devs)} signals")

    di = (CACHE / "det_intervals.parquet").as_posix()
    pc = (CACHE / "phase_cycles.parquet").as_posix()
    cs = (CACHE / "coord_state.parquet").as_posix()
    ev = (CACHE / "events" / "**" / "*.parquet").as_posix()

    for i in range(0, len(devs), chunk):
        part = devs[i:i + chunk]
        inl = ",".join("'" + d + "'" for d in part)
        t0 = time.time()

        D = con.sql(f"""SELECT DeviceId, Detector::INT AS ch,
                               (epoch_ms(t_on)  - {int(T0.value//10**6)})::BIGINT AS a,
                               (epoch_ms(t_off) - {int(T0.value//10**6)})::BIGINT AS b
                        FROM read_parquet('{di}') WHERE DeviceId IN ({inl})
                        ORDER BY DeviceId, ch, a""").df()
        P = con.sql(f"""SELECT DeviceId, Phase::INT AS ph,
                               (epoch_ms(green_start)-{int(T0.value//10**6)})::BIGINT AS g0,
                               (epoch_ms(coalesce(yellow_start,red_start,next_green))-{int(T0.value//10**6)})::BIGINT AS g1,
                               (epoch_ms(yellow_start)-{int(T0.value//10**6)})::BIGINT AS y0,
                               (epoch_ms(coalesce(red_start,redclr_end))-{int(T0.value//10**6)})::BIGINT AS y1,
                               (epoch_ms(red_start)-{int(T0.value//10**6)})::BIGINT AS r0,
                               (epoch_ms(redclr_end)-{int(T0.value//10**6)})::BIGINT AS r1
                        FROM read_parquet('{pc}') WHERE DeviceId IN ({inl}) AND Phase BETWEEN 1 AND 16
                        ORDER BY DeviceId, ph, g0""").df()
        # calls: 43 -> next 44 on the same phase
        C = con.sql(f"""
            WITH e AS (SELECT DeviceId, Parameter::INT AS ph, Timestamp AS ts, EventId
                       FROM read_parquet('{ev}', hive_partitioning=true)
                       WHERE DeviceId IN ({inl}) AND EventId IN (43,44) AND Parameter BETWEEN 1 AND 16),
                 d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev
                       FROM e WINDOW w AS (PARTITION BY DeviceId, ph
                                           ORDER BY ts, CASE WHEN EventId=43 THEN 0 ELSE 1 END))
            SELECT DeviceId, ph,
                   (epoch_ms(ts) -{int(T0.value//10**6)})::BIGINT AS a,
                   (epoch_ms(nts)-{int(T0.value//10**6)})::BIGINT AS b
            FROM d WHERE EventId=43 AND nev=44 AND nts IS NOT NULL
            ORDER BY DeviceId, ph, a""").df()
        CO = con.sql(f"""SELECT DeviceId,
                                (epoch_ms(t_start)-{int(T0.value//10**6)})::BIGINT AS a,
                                (epoch_ms(coalesce(t_end, TIMESTAMP '2024-12-05'))-{int(T0.value//10**6)})::BIGINT AS b
                         FROM read_parquet('{cs}') WHERE DeviceId IN ({inl}) AND is_coord
                         ORDER BY DeviceId, a""").df()

        gD = {k: v for k, v in D.groupby("DeviceId", sort=False)}
        gP = {k: v for k, v in P.groupby("DeviceId", sort=False)}
        gC = {k: v for k, v in C.groupby("DeviceId", sort=False)}
        gCO = {k: v for k, v in CO.groupby("DeviceId", sort=False)}

        for dev in part:
            cand = np.array(cand_of.get(dev, []), dtype=np.int8)
            out = {"cand": cand}
            d = gD.get(dev)
            if d is None or len(d) == 0:
                out.update(det_ch=np.zeros(0, np.int16), det_ptr=np.zeros(1, np.int32),
                           det_on=np.zeros(0, np.int32), det_off=np.zeros(0, np.int32))
            else:
                chs = np.sort(d.ch.unique())
                out["det_ch"] = chs.astype(np.int16)
                out["det_ptr"] = _ptr_arrays(d.ch.values, list(chs))
                out["det_on"] = d.a.values.astype(np.int32)
                out["det_off"] = d.b.values.astype(np.int32)
            p = gP.get(dev)
            c = gC.get(dev)
            for tag, src, c0, c1 in (("g", p, "g0", "g1"), ("y", p, "y0", "y1"),
                                     ("r", p, "r0", "r1"), ("c", c, "a", "b")):
                ons, offs, ptr = [], [], [0]
                for ph in cand:
                    if src is None:
                        ptr.append(ptr[-1]); continue
                    s = src[src.ph == int(ph)]
                    a = pd.to_numeric(s[c0], errors="coerce").to_numpy(dtype=np.float64)
                    b = pd.to_numeric(s[c1], errors="coerce").to_numpy(dtype=np.float64)
                    ok = np.isfinite(a) & np.isfinite(b) & (b > a)
                    a, b = a[ok], b[ok]
                    o = np.argsort(a, kind="stable")
                    ons.append(a[o]); offs.append(b[o]); ptr.append(ptr[-1] + len(a))
                out[tag + "_on"] = (np.concatenate(ons) if ons else np.zeros(0)).astype(np.int32)
                out[tag + "_off"] = (np.concatenate(offs) if offs else np.zeros(0)).astype(np.int32)
                out[tag + "_ptr"] = np.array(ptr, dtype=np.int32)
            co = gCO.get(dev)
            out["co_on"] = (co.a.values if co is not None else np.zeros(0)).astype(np.int32)
            out["co_off"] = (co.b.values if co is not None else np.zeros(0)).astype(np.int32)
            np.savez(SIGDIR / f"{dev}.npz", **out)
        log(f"chunk {i//chunk}: {len(part)} signals in {time.time()-t0:.0f}s")
    con.close()
    log("done")


if __name__ == "__main__":
    build()
