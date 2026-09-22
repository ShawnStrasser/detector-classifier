"""Stage 13: period-aware interval cache for the refit GRU.

Same arrays and the same integer-millisecond encoding as `ncache.py`, but the source
cache, the time origin and the output folder are parameters, so the Sept-2026 staging
data gets its own cache next to the Dec-2024 one:

    dec   DC_WORK/cache                 T0 = 2024-12-02 00:00   ->  DC_WORK/neural/sig
    stg   DC_WORK/official/stg/cache    T0 = 2026-09-18 00:00   ->  DC_WORK/neural/sig_stg

The Dec-2024 cache is the stage-03 one and is only *checked* here, never rebuilt.

    python src/neural/ncache2.py --period stg
    python src/neural/ncache2.py --period dec --check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, DC_WORK, connect  # noqa: E402

NEURAL = DC_WORK / "neural"
STG_CACHE = DC_WORK / "official" / "stg" / "cache"

PERIODS = {
    "dec": dict(cache=CACHE, sigdir=NEURAL / "sig",
                t0="2024-12-02 00:00:00", end="2024-12-05 00:00:00"),
    "stg": dict(cache=STG_CACHE, sigdir=NEURAL / "sig_stg",
                t0="2026-09-18 00:00:00", end="2026-09-21 11:00:00"),
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _ptr_arrays(keys: np.ndarray, order: list) -> np.ndarray:
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


def build(period: str, devices: list[str] | None = None, chunk: int = 30,
          overwrite: bool = False) -> None:
    spec = PERIODS[period]
    cache, sigdir = Path(spec["cache"]), Path(spec["sigdir"])
    T0 = pd.Timestamp(spec["t0"])
    t0ms = int(T0.value // 10 ** 6)
    endts = spec["end"]
    sigdir.mkdir(parents=True, exist_ok=True)
    con = connect(memory_limit="8GB", threads=8)

    meta = con.sql(f"""SELECT DeviceId, cand_phases
                       FROM read_parquet('{(cache/'signal_meta.parquet').as_posix()}')""").df()
    if devices is not None:
        meta = meta[meta.DeviceId.isin(set(devices))]
    devs = sorted(meta.DeviceId.tolist())
    cand_of = {r.DeviceId: sorted(int(p) for p in (r.cand_phases
                                                   if r.cand_phases is not None
                                                   and not isinstance(r.cand_phases, float)
                                                   and r.cand_phases is not pd.NA else [])
                                  if 1 <= int(p) <= 16)
               for r in meta.itertuples()}
    if not overwrite:
        devs = [d for d in devs if not (sigdir / f"{d}.npz").exists()]
    log(f"period={period}: {len(devs)} signals to build -> {sigdir}")
    if not devs:
        con.close()
        return

    di = (cache / "det_intervals.parquet").as_posix()
    pc = (cache / "phase_cycles.parquet").as_posix()
    cs = (cache / "coord_state.parquet").as_posix()
    ev = (cache / "events" / "**" / "*.parquet").as_posix()

    for i in range(0, len(devs), chunk):
        part = devs[i:i + chunk]
        inl = ",".join("'" + d + "'" for d in part)
        t_start = time.time()

        D = con.sql(f"""SELECT DeviceId, Detector::INT AS ch,
                               (epoch_ms(t_on)  - {t0ms})::BIGINT AS a,
                               (epoch_ms(t_off) - {t0ms})::BIGINT AS b
                        FROM read_parquet('{di}') WHERE DeviceId IN ({inl})
                        ORDER BY DeviceId, ch, a""").df()
        P = con.sql(f"""SELECT DeviceId, Phase::INT AS ph,
                               (epoch_ms(green_start)-{t0ms})::BIGINT AS g0,
                               (epoch_ms(coalesce(yellow_start,red_start,next_green))-{t0ms})::BIGINT AS g1,
                               (epoch_ms(yellow_start)-{t0ms})::BIGINT AS y0,
                               (epoch_ms(coalesce(red_start,redclr_end))-{t0ms})::BIGINT AS y1,
                               (epoch_ms(red_start)-{t0ms})::BIGINT AS r0,
                               (epoch_ms(redclr_end)-{t0ms})::BIGINT AS r1
                        FROM read_parquet('{pc}') WHERE DeviceId IN ({inl})
                          AND Phase BETWEEN 1 AND 16
                        ORDER BY DeviceId, ph, g0""").df()
        C = con.sql(f"""
            WITH e AS (SELECT DeviceId, Parameter::INT AS ph, Timestamp AS ts, EventId
                       FROM read_parquet('{ev}', hive_partitioning=true)
                       WHERE DeviceId IN ({inl}) AND EventId IN (43,44)
                         AND Parameter BETWEEN 1 AND 16),
                 d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev
                       FROM e WINDOW w AS (PARTITION BY DeviceId, ph
                                           ORDER BY ts, CASE WHEN EventId=43 THEN 0 ELSE 1 END))
            SELECT DeviceId, ph, (epoch_ms(ts) -{t0ms})::BIGINT AS a,
                   (epoch_ms(nts)-{t0ms})::BIGINT AS b
            FROM d WHERE EventId=43 AND nev=44 AND nts IS NOT NULL
            ORDER BY DeviceId, ph, a""").df()
        CO = con.sql(f"""SELECT DeviceId,
                                (epoch_ms(t_start)-{t0ms})::BIGINT AS a,
                                (epoch_ms(coalesce(t_end, TIMESTAMP '{endts}'))-{t0ms})::BIGINT AS b
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
                        ptr.append(ptr[-1])
                        continue
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
            np.savez(sigdir / f"{dev}.npz", **out)
        log(f"chunk {i//chunk}: {len(part)} signals in {time.time()-t_start:.0f}s")
    con.close()
    log("build done")


# ------------------------------------------------------------------- integrity
def check(period: str) -> dict:
    """Every stream must reach the end of the log.

    A silent truncation of the 43/44 call stream at the end of Dec 3 once cost ~14
    accuracy points on Dec-4 windows; this is the guard that catches it.
    """
    spec = PERIODS[period]
    sigdir = Path(spec["sigdir"])
    files = sorted(sigdir.glob("*.npz"))
    trunc, nocall, empty = [], [], []
    for p in files:
        with np.load(p) as f:
            g, c = f["g_on"], f["c_on"]
        if not len(g):
            empty.append(p.stem)
            continue
        if not len(c):
            nocall.append(p.stem)
            continue
        gap_h = (int(g.max()) - int(c.max())) / 3.6e6
        if gap_h > 6.0:
            trunc.append((p.stem, round(gap_h, 1)))
    res = {"period": period, "n_signals": len(files), "n_truncated": len(trunc),
           "n_no_call_events_at_all": len(nocall), "n_no_green": len(empty),
           "truncated": trunc[:20]}
    log(f"check {period}: {len(files)} signals, {len(trunc)} truncated call streams, "
        f"{len(nocall)} signals with no 43/44 at all, {len(empty)} with no green")
    assert not trunc, f"truncated call stream in {len(trunc)} signals: {trunc[:5]}"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="stg", choices=sorted(PERIODS))
    ap.add_argument("--devices", default="", help="csv file with a DeviceId column")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--check", action="store_true", help="only run the integrity check")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if not a.check:
        devs = None
        if a.devices:
            devs = sorted(pd.read_csv(a.devices).DeviceId.astype(str).str.lower().unique())
        build(a.period, devs, a.chunk, a.overwrite)
    check(a.period)


if __name__ == "__main__":
    main()
