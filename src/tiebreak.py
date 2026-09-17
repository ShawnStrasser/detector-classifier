"""Stage 04 step 4: OPTIONAL ODOT standard-wiring tie-breaker (post-processing, default OFF).

The standard channel -> phase table (`docs/standard_detector_mapping.md`) is never a model
input (research_protocol.md).  It may only be consulted *after* the model has produced
probabilities, and only in the narrow situation where it carries information the data cannot:

  * the detector's top-2 candidates are within `margin` of each other, AND
  * those two candidates are a concurrent / opposing pair (2-6, 4-8, 1-5, ...), i.e. the case
    the behaviour features genuinely cannot separate, AND
  * the signal *looks* standard-wired -- `standardness` = the fraction of that signal's
    confident predictions that agree with the table (computed from the model's own output,
    so no label is used), AND
  * the channel is <= 40 and its standard phase is one of the two candidates.

Then the two probabilities are swapped so the standard phase wins.

    from tiebreak import apply_odot_tiebreak
    out = apply_odot_tiebreak(phase_probs, enabled=True)      # default is enabled=False
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONCURRENT_PAIRS, DEFAULT_PHASE  # noqa: E402

DEFAULTS = dict(margin=0.25, standardness_min=0.85, conf_for_standardness=0.80,
                concurrent_only=True)


def signal_standardness(phase_probs_df: pd.DataFrame,
                        conf: float = 0.80) -> pd.DataFrame:
    """Per signal: agreement between *confident* model predictions and the standard table.

    Uses candidate phase numbers only at this post-processing step, and never a label.
    Returns DeviceId, standardness, n_conf.
    """
    p = phase_probs_df.sort_values(["DeviceId", "Detector", "prob"],
                                   ascending=[True, True, False])
    top = p.groupby(["DeviceId", "Detector"], as_index=False).first()
    top = top[(top.prob >= conf) & (top.Detector <= 40)].copy()
    top["std_phase"] = top.Detector.map(DEFAULT_PHASE)
    top["agree"] = (top.cand_phase == top.std_phase).astype(float)
    out = top.groupby("DeviceId").agg(standardness=("agree", "mean"),
                                      n_conf=("agree", "size")).reset_index()
    return out


def apply_odot_tiebreak(phase_probs_df: pd.DataFrame, enabled: bool = False,
                        margin: float = DEFAULTS["margin"],
                        standardness_min: float = DEFAULTS["standardness_min"],
                        conf_for_standardness: float = DEFAULTS["conf_for_standardness"],
                        concurrent_only: bool = DEFAULTS["concurrent_only"],
                        return_flags: bool = False):
    """Post-process a protocol phase-prediction frame (DeviceId, Detector, cand_phase, prob).

    With `enabled=False` (the default) the frame is returned unchanged.
    """
    df = phase_probs_df.copy()
    if not enabled:
        return (df, pd.DataFrame(columns=["DeviceId", "Detector"])) if return_flags else df

    std = signal_standardness(df, conf_for_standardness)
    ok_sig = set(std.loc[std.standardness >= standardness_min, "DeviceId"])

    p = df.sort_values(["DeviceId", "Detector", "prob"], ascending=[True, True, False])
    g = p.groupby(["DeviceId", "Detector"], sort=False)
    first = g.head(1)[["DeviceId", "Detector", "cand_phase", "prob"]].rename(
        columns={"cand_phase": "c1", "prob": "q1"})
    second = g.nth(1)[["DeviceId", "Detector", "cand_phase", "prob"]].rename(
        columns={"cand_phase": "c2", "prob": "q2"})
    t = first.merge(second, on=["DeviceId", "Detector"], how="left")
    t["std_phase"] = t.Detector.map(DEFAULT_PHASE)
    t["margin"] = t.q1 - t.q2.fillna(0.0)
    pair_ok = [frozenset((int(a), int(b))) in CONCURRENT_PAIRS if pd.notna(b) else False
               for a, b in zip(t.c1, t.c2)]
    t["pair_ok"] = pair_ok if concurrent_only else True
    swap = (t.DeviceId.isin(ok_sig) & (t.margin < margin) & t.pair_ok &
            t.c2.notna() & t.std_phase.notna() &
            (t.std_phase == t.c2) & (t.std_phase != t.c1))
    sw = t[swap]
    if not len(sw):
        return (df, sw) if return_flags else df

    key = df.DeviceId + "|" + df.Detector.astype(str)
    swk = set(sw.DeviceId + "|" + sw.Detector.astype(str))
    m = key.isin(swk)
    lut = {f"{d}|{det}": (c1, c2, q1, q2) for d, det, c1, c2, q1, q2 in
           zip(sw.DeviceId, sw.Detector, sw.c1, sw.c2, sw.q1, sw.q2)}
    newp = df.prob.to_numpy(dtype=float).copy()
    idx = np.flatnonzero(m.to_numpy())
    kk = key.to_numpy()
    cc = df.cand_phase.to_numpy()
    for i in idx:
        c1, c2, q1, q2 = lut[kk[i]]
        if cc[i] == c1:
            newp[i] = q2
        elif cc[i] == c2:
            newp[i] = q1
    df["prob"] = newp
    sw = sw.assign(reason="odot_tiebreak: close concurrent pair at a standard-wired signal")
    return (df, sw) if return_flags else df


# --------------------------------------------------------------- tuning / eval
def evaluate_tiebreak(phase_probs_df: pd.DataFrame, labels: pd.DataFrame,
                      **kw) -> dict:
    """How often the tie-breaker helps and how often it HURTS (esp. on non-standard labels)."""
    out, sw = apply_odot_tiebreak(phase_probs_df, enabled=True, return_flags=True, **kw)
    if not len(sw):
        return dict(n_switched=0, helped=0, hurt=0, neutral=0, delta_n=0,
                    hurt_nonstandard=0, **kw)
    m = sw.merge(labels, on=["DeviceId", "Detector"], how="inner")
    helped = int(((m.c1 != m.Phase) & (m.c2 == m.Phase)).sum())
    hurt = int(((m.c1 == m.Phase) & (m.c2 != m.Phase)).sum())
    neutral = int(len(m) - helped - hurt)
    m["is_nonstd"] = m.Phase != m.Detector.map(DEFAULT_PHASE)
    hurt_ns = int(((m.c1 == m.Phase) & m.is_nonstd).sum())
    return dict(n_switched=int(len(m)), helped=helped, hurt=hurt, neutral=neutral,
                delta_n=helped - hurt, hurt_nonstandard=hurt_ns, **kw)


def tune(phase_probs_df: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame,
         margins=(0.05, 0.1, 0.2, 0.3, 0.5), stds=(0.7, 0.8, 0.9, 1.0)) -> pd.DataFrame:
    """Nested: choose (margin, standardness) on 5 folds, report the gain on the held-out one."""
    rows = []
    for k in sorted(folds.fold.unique()):
        inner_dev = set(folds.loc[folds.fold != k, "DeviceId"])
        outer_dev = set(folds.loc[folds.fold == k, "DeviceId"])
        pi = phase_probs_df[phase_probs_df.DeviceId.isin(inner_dev)]
        li = labels[labels.DeviceId.isin(inner_dev)]
        best, bg = None, -10**9
        for mg in margins:
            for st in stds:
                r = evaluate_tiebreak(pi, li, margin=mg, standardness_min=st)
                if r["delta_n"] > bg:
                    bg, best = r["delta_n"], (mg, st)
        po = phase_probs_df[phase_probs_df.DeviceId.isin(outer_dev)]
        lo = labels[labels.DeviceId.isin(outer_dev)]
        r = evaluate_tiebreak(po, lo, margin=best[0], standardness_min=best[1])
        r["fold"] = k
        r["chosen_margin"], r["chosen_std"] = best
        rows.append(r)
    return pd.DataFrame(rows)
