"""Stage 08 step 2 -- turn the raw per-demand-level counters into model features.

For every base quantity we hold a numerator and a denominator per demand level, so the
level value can be **shrunk** toward the detector's own pooled value:

    v_l = (num_l + k * pooled) / (den_l + k),      pooled = sum_l num_l / sum_l den_l

`k` is expressed in denominator units (seconds, actuations, cycles), so a level with
little evidence simply returns the pooled value and its contrast collapses to 0 instead of
injecting noise.  Emitted per base feature f:

    f_lo, f_hi      value in the low / high demand period (shrunk)
    f_d             hi - lo
    f_lr            log((hi+eps)/(lo+eps))
    f_sl            evidence-weighted slope of f against log(signal actuation rate)

plus context (`_ctx_*`): evidence counts and how far apart the two demand levels actually
are.  Three variants are written side by side:

    cq_   data-driven demand terciles, exact-interval occupancy   (the candidate)
    cc_   clock periods,               exact-interval occupancy
    bq_   data-driven demand terciles, 15 s **binned** occupancy  (task-4 control)

    python src/contrast/contrast_features.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402

WORK = DC_WORK / "contrast"
KEY = ["DeviceId", "Detector", "cand_phase", "win"]
EPS = 1e-4

# name -> (numerator, denominator, k)   k is in denominator units
SPECS_EXACT = {
    "occred":   ("occ_r", "c_rsecs", 300.0),      # occupancy fraction of red
    "occgrn":   ("occ_g", "c_gsecs", 300.0),      # occupancy fraction of green
    "occdet":   ("d_occ", "s_secs", 600.0),       # occupancy fraction of the period
    "onrate":   ("_on3600", "s_secs", 600.0),     # actuations per hour
    "fongrn":   ("n_g", "n_on_pair", 25.0),
    "fonred10": ("n_red10", "n_on_pair", 25.0),
    "dtgb0":    ("n_dtg2", "n_g", 15.0),
    "burst":    ("n_g4", "_cyc4", 40.0),
    "queueocc": ("q_sum", "q_n", 10.0),           # mean ON length just before green
    "straddle": ("n_str", "n_long15", 10.0),
    "call43":   ("n_c43", "n_call_den", 25.0),
    "fraclong": ("d_nlong", "d_n", 25.0),
    "fracshort": ("d_nshort", "d_n", 25.0),
    "durmed":   ("_medn", "d_n", 25.0),           # median ON duration
    "durq90":   ("_q90n", "d_n", 25.0),
}
SPECS_BIN = {
    "occred":   ("occ_rb", "c_rsecs", 300.0),
    "occgrn":   ("occ_gb", "c_gsecs", 300.0),
    "occdet":   ("d_occb", "s_secs", 600.0),
    "queueocc": ("q_sumb", "q_n", 10.0),
    "durmed":   ("_medbn", "d_n", 25.0),
    "durq90":   ("_q90bn", "d_n", 25.0),
}
# per-cycle occupancy, clipped to the colour interval, averaged over the cycles of the
# period -- what `atspm`'s split-failure query computes before time-bucketing
SPECS_CYC = {
    "occred":   ("mr_sum", "c_ncyc", 10.0),
    "occgrn":   ("mg_sum", "c_ncyc", 10.0),
    "occredmi": ("occ_rc", "c_rsecs", 300.0),   # same clipping, micro (time-weighted)
    "occgrnmi": ("occ_gc", "c_gsecs", 300.0),
}
CTX_COLS = ["c_rsecs", "c_gsecs", "c_ncyc", "s_secs", "s_vol"]
NUM_COLS = ["n_on_pair", "n_g", "n_r", "occ_g", "occ_r", "occ_gb", "occ_rb", "n_dtg2",
            "n_g4", "n_red10", "q_sum", "q_n", "q_sumb", "d_n", "d_occ", "d_occb",
            "d_nlong", "d_nshort", "n_long15", "n_str", "n_call_den", "n_c43",
            "_on3600", "_medn", "_q90n", "_medbn", "_q90bn",
            "occ_gc", "occ_rc", "mg_sum", "mr_sum"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["_on3600"] = d.d_n * 3600.0
    # medians are not sums: carry them as n-weighted totals so the same shrinkage applies
    for src, dst in (("d_med", "_medn"), ("d_q90", "_q90n"),
                     ("d_medb", "_medbn"), ("d_q90b", "_q90bn")):
        d[dst] = d[src].fillna(0.0) * d.d_n
    return d


def _grid(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (key, lvl) for lvl 0/1/2, in key-major order.  Missing levels get zero
    numerators and the level's own context denominators, so a detector that is silent in
    the low-demand period is *known* to be silent rather than unknown."""
    ctx = (d.groupby(["DeviceId", "win", "cand_phase", "lvl"], sort=False)[CTX_COLS]
           .max().reset_index())
    keys = d[KEY].drop_duplicates().reset_index(drop=True)
    lv = pd.DataFrame({"lvl": np.array([0, 1, 2], dtype=np.int8)})
    g = keys.merge(lv, how="cross")                      # key-major, lvl 0,1,2
    cols = KEY + ["lvl"] + [c for c in NUM_COLS if c in d.columns]
    g = g.merge(d[cols], on=KEY + ["lvl"], how="left")
    g = g.merge(ctx, on=["DeviceId", "win", "cand_phase", "lvl"], how="left")
    for c in NUM_COLS + CTX_COLS:
        if c in g.columns:
            g[c] = g[c].fillna(0.0)
    g["_cyc4"] = g.c_ncyc * 4.0
    assert (g.lvl.to_numpy()[:6] == [0, 1, 2, 0, 1, 2]).all()
    return g


def build_block(d: pd.DataFrame, specs: dict, prefix: str) -> pd.DataFrame:
    g = _grid(_prep(d))
    n = len(g) // 3

    def M(name: str) -> np.ndarray:
        return g[name].to_numpy(dtype=np.float64).reshape(n, 3)

    rate = M("s_vol") / np.where(M("s_secs") > 0, M("s_secs"), np.nan)
    X = np.log(np.nan_to_num(rate, nan=0.0) + 1e-5)
    out = {}
    for name, (num, den, k) in specs.items():
        A, B = M(num), M(den)
        tot_n, tot_d = A.sum(1), B.sum(1)
        pooled = np.where(tot_d > 0, tot_n / np.maximum(tot_d, 1e-9), np.nan)
        V = (A + k * pooled[:, None]) / (B + k)
        lo, hi = V[:, 0], V[:, 2]
        out[f"{prefix}{name}_lo"] = lo
        out[f"{prefix}{name}_hi"] = hi
        out[f"{prefix}{name}_d"] = hi - lo
        out[f"{prefix}{name}_lr"] = np.log((hi + EPS) / (lo + EPS))
        w = B / np.maximum(B.sum(1, keepdims=True), 1e-9)
        xb = (w * X).sum(1)
        yb = (w * np.nan_to_num(V)).sum(1)
        dx = X - xb[:, None]
        sxx = (w * dx * dx).sum(1)
        sxy = (w * dx * (np.nan_to_num(V) - yb[:, None])).sum(1)
        out[f"{prefix}{name}_sl"] = np.where(sxx > 1e-9, sxy / np.maximum(sxx, 1e-9), np.nan)
    for src, tag in (("n_on_pair", "non"), ("c_ncyc", "ncyc")):
        P = M(src)
        out[f"{prefix}ctx_{tag}_lo"] = np.log1p(P[:, 0])
        out[f"{prefix}ctx_{tag}_hi"] = np.log1p(P[:, 2])
    R = np.nan_to_num(rate, nan=0.0)
    out[f"{prefix}ctx_demand_lr"] = np.log((R[:, 2] + 1e-5) / (R[:, 0] + 1e-5))
    S = M("s_secs")
    out[f"{prefix}ctx_losecs"] = S[:, 0]
    out[f"{prefix}ctx_hisecs"] = S[:, 2]
    base = g.iloc[0::3][KEY].reset_index(drop=True)
    return pd.concat([base, pd.DataFrame(out)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="contrast_raw.parquet")
    ap.add_argument("--out", default="contrast_feats.parquet")
    a = ap.parse_args()
    t0 = time.time()
    raw = pd.read_parquet(WORK / a.raw)
    log(f"raw {raw.shape}")
    blocks = []
    for defn, specs, prefix in (("q", SPECS_EXACT, "cq_"),
                                ("c", SPECS_EXACT, "cc_"),
                                ("q", SPECS_BIN, "bq_"),
                                ("q", SPECS_CYC, "pq_")):
        b = build_block(raw[raw.defn == defn], specs, prefix)
        log(f"{prefix}: {b.shape}")
        blocks.append(b.set_index(KEY))
    out = pd.concat(blocks, axis=1).reset_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    out.to_parquet(WORK / a.out, index=False)
    log(f"wrote {WORK / a.out}: {out.shape} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
