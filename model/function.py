"""Feature builders for the detector-function head (Advance / Presence / Count /
Yellow_Red / Other).

The function model describes a detector against the *other* detectors that the phase
model has just put on the same phase, so every feature here is relative:

* **shape** -- occupancy in green vs red, mean ON duration in green vs red, the spread of
  the detector's own ON durations, occupancy per actuation.  Presence sits occupied
  through the whole red, Advance almost never does; Count has quarter-second ONs.
* **sibling-relative** -- the same quantities minus the median over the detectors that
  share the predicted phase, plus their rank inside that group.  This is what makes a
  "long" occupancy long *for this approach*.
* **actuation lag** -- the signed lag from each ON of this channel to the nearest ON of
  every other channel.  A positive peak means the other channel fires afterwards, i.e.
  this one is upstream (Advance); a peak at zero with a high coincidence means the two
  see the same vehicles at the same place.

No phase number and no channel number is ever an input.  The training code is
`research/code/lightgbm/function_v4.py`.
"""
from __future__ import annotations

import pandas as pd

# sibling-relative features (stage 04 list + the yellow/red-clearance family)
SIB_FEATS = ["first_on_med", "first_on_le1", "first_on_le3", "dtg_b0", "dtg_h0",
             "det_dur_med", "det_dur_q90", "det_occ_frac", "det_on_per_hour",
             "queue_occ_pre_green", "f_on_red_last10", "release_frac_long",
             "burst_rate_g4", "det_frac_long", "f_on_green", "dur_mean_green",
             "yr_f_on_yr", "yr_lift_yr", "yr_hit_yr", "yr_lag_med",
             "yr_f_on_redclr", "yr_med_to_green_end", "call43_fwd_035"]


def add_shape_features(top: pd.DataFrame) -> pd.DataFrame:
    d, eps = {}, 1e-6
    if {"f_occ_green", "f_occ_red"} <= set(top.columns):
        d["occ_green_red_ratio"] = top.f_occ_green / (top.f_occ_red + eps)
    if {"dur_mean_green", "dur_mean_red"} <= set(top.columns):
        d["dur_green_red_ratio"] = top.dur_mean_green / (top.dur_mean_red + eps)
    if {"det_dur_q90", "det_dur_med"} <= set(top.columns):
        d["dur_q90_over_med"] = top.det_dur_q90 / (top.det_dur_med + eps)
    if {"det_occ_frac", "det_on_per_hour"} <= set(top.columns):
        d["occ_per_actuation"] = top.det_occ_frac * 3600.0 / (top.det_on_per_hour + eps)
    # Presence sits occupied through red, Advance almost never does
    if {"f_occ_red", "f_on_red"} <= set(top.columns):
        d["occ_red_per_on_red"] = top.f_occ_red / (top.f_on_red + eps)
    return pd.concat([top, pd.DataFrame(d, index=top.index)], axis=1)


def add_sibling_features(top: pd.DataFrame) -> pd.DataFrame:
    key = ["DeviceId", "win", "pred_phase"]
    g = top.groupby(key, sort=False)
    new = {"sib_n": g["Detector"].transform("size").astype("float32")}
    for f in SIB_FEATS:
        if f not in top.columns:
            continue
        med = g[f].transform("median")
        new[f"{f}__sibdiff"] = top[f] - med
        new[f"{f}__sibrank"] = g[f].rank(pct=True, method="average")
        new[f"{f}__sibmin"] = top[f] - g[f].transform("min")
    return pd.concat([top, pd.DataFrame(new, index=top.index)], axis=1)


def add_lag_features(top: pd.DataFrame, lag: pd.DataFrame) -> pd.DataFrame:
    """Pairwise actuation-lag structure, aggregated over (a) the detectors the phase model
    puts on the *same* phase ("siblings") and (b) every other channel of the signal.

    `lag` is the table `features_yellowred.SQL_LAG` builds.  Sign convention:
    `lag_peak > 0` means `other` fires AFTER this detector, i.e. this detector is the
    upstream one (Advance).  `coinc_05_excess` near its maximum with `lag_peak ~ 0` means
    the two channels see the *same* vehicles at the same place."""
    keep = top[["DeviceId", "win", "Detector", "pred_phase"]]
    j = lag.merge(keep, on=["DeviceId", "win", "Detector"], how="inner")
    j = j.merge(keep.rename(columns={"Detector": "other",
                                     "pred_phase": "other_phase"}),
                on=["DeviceId", "win", "other"], how="left")
    j["vol_ratio"] = j.n_b / j.n_a.clip(lower=1)
    out = top
    for tag, sub in (("lagany", j), ("lagsib", j[j.other_phase == j.pred_phase])):
        if not len(sub):
            continue
        s = sub.sort_values(["lag_peak_excess", "other"], na_position="first",
                            kind="stable")
        g = s.groupby(["DeviceId", "win", "Detector"], sort=False)
        best = g.tail(1).set_index(["DeviceId", "win", "Detector"])
        agg = g.agg(**{f"{tag}_n": ("lag_peak", "size"),
                       f"{tag}_excess_max": ("lag_peak_excess", "max"),
                       f"{tag}_excess_med": ("lag_peak_excess", "median"),
                       f"{tag}_lag_med": ("lag_peak", "median"),
                       f"{tag}_lead_frac": ("lag_peak", lambda x: float((x > 1).mean())),
                       f"{tag}_follow_frac": ("lag_peak", lambda x: float((x < -1).mean())),
                       f"{tag}_twin_coinc": ("coinc_05_excess", "max"),
                       f"{tag}_volratio_med": ("vol_ratio", "median")})
        agg[f"{tag}_best_lag"] = best["lag_peak"]
        agg[f"{tag}_best_coinc"] = best["coinc_05_excess"]
        agg[f"{tag}_best_volratio"] = best["vol_ratio"]
        agg[f"{tag}_best_frac_after"] = best["frac_after"]
        # the lag of the channel this one is most nearly a duplicate of
        t = s.sort_values(["coinc_05_excess", "other"], na_position="first",
                          kind="stable").groupby(
            ["DeviceId", "win", "Detector"], sort=False).tail(1).set_index(
            ["DeviceId", "win", "Detector"])
        agg[f"{tag}_twinlag"] = t["lag_peak"]
        agg[f"{tag}_twin_volratio"] = t["vol_ratio"]
        out = out.merge(agg.reset_index(), on=["DeviceId", "win", "Detector"], how="left")
    return out
