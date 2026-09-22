"""Joint per-signal decoding: the second stage, phase-anonymous.

The pair ranker scores each (detector, candidate phase) independently.  A signal,
however, is a *joint* object: detectors on the same approach actuate together, channels
are usually wired in blocks, and every phase that registers vehicle calls normally owns
at least one detector.  This stage was trained to use those constraints, and it is what
separates two phases whose greens always coincide (the 2 <-> 6 problem).

Second-stage inputs per (detector, candidate c) -- all number-free:
  * first-stage probability, its logit, rank within the detector, gap to the detector's
    best candidate
  * **similarity neighbours** (`similarity.py`): phi-weighted mean / max of the *other*
    detectors' probability for c, and the phi-weighted share of neighbours whose top-1
    is c
  * **channel-adjacency neighbours** (|channel difference| <= 2): same three aggregates.
    Adjacency is a wiring convention, not a phase number.
  * **signal-level occupancy of c**: how many other detectors claim c, the best other
    probability for c, whether c registers vehicle calls but is claimed by nobody
  * context: number of candidates, detector actuation count, window length

The training code is `research/code/lightgbm/decode_train.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_KEYS = ["DeviceId", "Detector", "win"]


# ------------------------------------------------------------- neighbour maths
def _agg_neighbours(pr: pd.DataFrame, nb: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """pr: DeviceId,win,Detector,cand_phase,p0,is_top1.  nb: DeviceId,win,Detector,other,w."""
    j = nb.merge(pr.rename(columns={"Detector": "other", "p0": "np0", "is_top1": "ntop1"}),
                 on=["DeviceId", "win", "other"], how="inner")
    j["wp"] = j.w * j.np0
    j["wt"] = j.w * j.ntop1
    g = j.groupby(["DeviceId", "win", "Detector", "cand_phase"], sort=False)
    out = g.agg(**{f"{prefix}_wsum": ("w", "sum"),
                   f"{prefix}_wp": ("wp", "sum"),
                   f"{prefix}_wt": ("wt", "sum"),
                   f"{prefix}_max": ("np0", "max"),
                   f"{prefix}_n": ("np0", "size")}).reset_index()
    ws = out[f"{prefix}_wsum"].replace(0, np.nan)
    out[f"{prefix}_mean"] = out[f"{prefix}_wp"] / ws
    out[f"{prefix}_top1"] = out[f"{prefix}_wt"] / ws
    return out.drop(columns=[f"{prefix}_wp", f"{prefix}_wt"])


def build_second_stage(pr: pd.DataFrame, ctx: pd.DataFrame,
                       sim: pd.DataFrame) -> pd.DataFrame:
    """pr = first-stage probabilities for ALL active detectors (labelled or not)."""
    pr = pr.copy()
    g = pr.groupby(["DeviceId", "win", "Detector"], sort=False)["p0"]
    pr["p0_max"] = g.transform("max")
    pr["is_top1"] = (pr.p0 >= pr.p0_max).astype(np.float32)
    pr["p0_gap"] = pr.p0 - pr.p0_max
    pr["p0_rank"] = g.rank(ascending=False, method="first")
    pr["n_cand"] = g.transform("size")
    pr["logit0"] = np.log(np.clip(pr.p0, 1e-6, 1 - 1e-6) /
                          (1 - np.clip(pr.p0, 1e-6, 1 - 1e-6)))

    slim = pr[["DeviceId", "win", "Detector", "cand_phase", "p0", "is_top1"]]

    # --- similarity neighbours
    sim = sim.copy()
    sim["w"] = sim.phi.clip(lower=0) ** 2
    sim = sim[sim.w > 0]
    simf = _agg_neighbours(slim, sim[["DeviceId", "win", "Detector", "other", "w"]], "sim")

    # --- channel-adjacency neighbours (|dch| <= 2)
    dets = slim[["DeviceId", "win", "Detector"]].drop_duplicates()
    adj = dets.merge(dets.rename(columns={"Detector": "other"}), on=["DeviceId", "win"])
    d = (adj.Detector - adj.other).abs()
    adj = adj[(d >= 1) & (d <= 2)].copy()
    adj["w"] = np.where((adj.Detector - adj.other).abs() == 1, 1.0, 0.5)
    adjf = _agg_neighbours(slim, adj, "adj")

    # --- signal-level occupancy of the candidate
    sig = slim.groupby(["DeviceId", "win", "cand_phase"], sort=False).agg(
        sig_sum=("p0", "sum"), sig_max=("p0", "max"), sig_claims=("is_top1", "sum"),
        sig_ndet=("p0", "size")).reset_index()

    out = pr.merge(simf, on=["DeviceId", "win", "Detector", "cand_phase"], how="left")
    out = out.merge(adjf, on=["DeviceId", "win", "Detector", "cand_phase"], how="left")
    out = out.merge(sig, on=["DeviceId", "win", "cand_phase"], how="left")
    # leave-one-out versions so a detector never reads its own probability back
    out["sig_sum_o"] = (out.sig_sum - out.p0) / np.maximum(out.sig_ndet - 1, 1)
    out["sig_claims_o"] = out.sig_claims - out.is_top1
    out["sig_max_o"] = np.where(out.p0 >= out.sig_max, np.nan, out.sig_max)
    out = out.merge(ctx, on=["DeviceId", "win", "cand_phase"], how="left")
    # "a phase with vehicle calls should own at least one detector"
    out["calls_unclaimed"] = ((out.call43_per_cycle.fillna(0) > 0.25) &
                              (out.sig_claims_o < 0.5)).astype(np.float32)
    out["green_share_per_claim"] = out.cand_green_share / (out.sig_claims_o + 1.0)
    return out.drop(columns=["sig_sum", "sig_max", "sig_claims"])


CTX_COLS = ["DeviceId", "Detector", "win", "cand_phase", "cand_green_share",
            "call43_per_cycle", "det_n_on", "log_win_hours"]


def assemble(pr: pd.DataFrame, pairs: pd.DataFrame,
             sim: pd.DataFrame) -> pd.DataFrame:
    """Second-stage design matrix from first-stage probabilities `pr` (DeviceId,
    Detector, win, cand_phase, p0).  `pairs` supplies the context columns, `sim` the
    detector-similarity graph."""
    ctx = pairs[CTX_COLS]
    ctxp = ctx.groupby(["DeviceId", "win", "cand_phase"], as_index=False).agg(
        cand_green_share=("cand_green_share", "first"),
        call43_per_cycle=("call43_per_cycle", "first"))
    det = ctx.groupby(["DeviceId", "win", "Detector"], as_index=False).agg(
        det_n_on=("det_n_on", "first"), log_win_hours=("log_win_hours", "first"))
    X = build_second_stage(pr, ctxp, sim)
    return X.merge(det, on=["DeviceId", "win", "Detector"], how="left")
