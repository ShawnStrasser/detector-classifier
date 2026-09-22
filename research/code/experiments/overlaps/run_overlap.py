"""Task 4 - overlaps as extra candidates ("phase OR overlap", one best label).

Steps
-----
signals   which signals actually run overlaps (event 61) and how many official
          `call_overlap` labels they carry
cache     build the pseudo-phase cache (see `overlap_cache.py`)
greens    how often an overlap's green is ~identical to a phase's green (Jaccard on
          green time) -- if it is, no timing-only model can tell them apart
features  pair / v2 / similarity features over that cache (overlaps included)
train     ranker (+ decoder) with and without overlap candidates, grouped CV by signal

    python src/official/run_overlap.py --step signals --source dec
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, FOLDS_CSV, RAW_DAYS, connect  # noqa: E402
import overlap_cache as OC  # noqa: E402

OFFICIAL = DC_WORK / "official"
pd.set_option("display.width", 220)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def root_for(source: str) -> Path:
    return OFFICIAL / ("ovl_dec" if source == "dec" else "ovl_stg")


# ------------------------------------------------------------------- signals
def step_signals(source: str) -> pd.DataFrame:
    con = connect(threads=8)
    root = root_for(source)
    root.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{(root/'tmp').as_posix()}'")
    (root / "tmp").mkdir(exist_ok=True)
    if source == "dec":
        src = ",".join(f"'{RAW_DAYS[d].as_posix()}'" for d in (3,))
        allowed = set(pd.read_csv(FOLDS_CSV).DeviceId)          # DEV only, never TEST
    else:
        src = ",".join(
            f"'{(DC_WORK/'data'/'staging_other'/f'date={d}'/'part_*.parquet').as_posix()}'"
            for d in ("2026-09-19",))
        sig = pd.read_csv(OFFICIAL / "stg" / "signals.csv")
        sp = pd.read_csv(OFFICIAL / "new_signal_split.csv")
        allowed = set(sig[sig.group == "DEV"].DeviceId) | \
            set(sp[sp.split == "NEWTRAIN"].DeviceId)            # never TEST / NEWTEST
    ov = con.sql(f"""SELECT DeviceId, count(DISTINCT Parameter) n_ovl, count(*) n_ev
                     FROM read_parquet([{src}])
                     WHERE EventId = 61 AND Parameter BETWEEN 1 AND 16
                     GROUP BY 1 HAVING count(*) >= 50""").df()
    con.close()
    ov = ov[ov.DeviceId.isin(allowed)]
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    real = off.real_dec2024 if source == "dec" else off.real_staging
    lab = off[real & (off.target_type == "overlap")].groupby("DeviceId").size() \
        .rename("n_overlap_labels")
    both = off[real & off.has_both_phase_and_overlap].groupby("DeviceId").size() \
        .rename("n_both_labels")
    ov = ov.merge(lab, on="DeviceId", how="left").merge(both, on="DeviceId", how="left")
    ov[["n_overlap_labels", "n_both_labels"]] = ov[["n_overlap_labels", "n_both_labels"]].fillna(0)
    ov = ov.sort_values("DeviceId")
    (root / "signals.txt").write_text("\n".join(ov.DeviceId))
    ov.to_csv(root / "signals.csv", index=False)
    log(f"{source}: {len(ov)} signals run overlaps; "
        f"{int((ov.n_overlap_labels > 0).sum())} carry an overlap-only official label "
        f"({int(ov.n_overlap_labels.sum())} channels), "
        f"{int(ov.n_both_labels.sum())} channels have a phase AND an overlap")
    return ov


# -------------------------------------------------------------- green overlap
def step_greens(source: str) -> pd.DataFrame:
    """Jaccard of green time between every overlap and every phase at the same signal."""
    root = root_for(source)
    con = connect(threads=8)
    con.execute(f"SET temp_directory='{(root/'tmp').as_posix()}'")
    cyc = (root / "cache" / "phase_cycles.parquet").as_posix()
    iv = con.sql(f"""SELECT DeviceId, Phase, epoch_ms(green_start)/1000.0 AS t0,
                            epoch_ms(coalesce(yellow_start, red_start, next_green))/1000.0 AS t1
                     FROM read_parquet('{cyc}')
                     WHERE coalesce(yellow_start, red_start, next_green) IS NOT NULL""").df()
    con.close()
    rows = []
    for dev, g in iv.groupby("DeviceId"):
        ps = {int(p): sub[["t0", "t1"]].to_numpy() for p, sub in g.groupby("Phase")}
        ovls = [p for p in ps if p > OC.OVL_OFFSET]
        phs = [p for p in ps if p <= OC.OVL_OFFSET]
        if not ovls or not phs:
            continue
        # event-point union: measure on a merged grid
        for o in ovls:
            ao = ps[o]
            lo = float((ao[:, 1] - ao[:, 0]).sum())
            best, bp = -1.0, None
            for p in phs:
                ap = ps[p]
                inter = _overlap_secs(ao, ap)
                lp = float((ap[:, 1] - ap[:, 0]).sum())
                j = inter / max(lo + lp - inter, 1e-9)
                if j > best:
                    best, bp = j, p
            rows.append({"DeviceId": dev, "overlap": o - OC.OVL_OFFSET,
                         "best_phase": bp, "jaccard": best, "green_secs": lo})
    d = pd.DataFrame(rows)
    d.to_csv(root / "overlap_green_similarity.csv", index=False)
    log(f"{len(d)} (signal, overlap) pairs")
    if len(d):
        print(d.jaccard.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
        for th in (0.95, 0.9, 0.8, 0.5):
            print(f"  share of overlaps whose green is >= {th} Jaccard-identical to a "
                  f"phase's green: {float((d.jaccard >= th).mean()):.3f}")
    return d


def _overlap_secs(a: np.ndarray, b: np.ndarray) -> float:
    """Total intersection seconds between two sorted interval sets."""
    a = a[np.argsort(a[:, 0])]
    b = b[np.argsort(b[:, 0])]
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i, 0], b[j, 0])
        hi = min(a[i, 1], b[j, 1])
        if hi > lo:
            tot += hi - lo
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["signals", "greens"])
    ap.add_argument("--source", default="dec", choices=["dec", "stg"])
    a = ap.parse_args()
    if a.step == "signals":
        step_signals(a.source)
    elif a.step == "greens":
        step_greens(a.source)


if __name__ == "__main__":
    main()
