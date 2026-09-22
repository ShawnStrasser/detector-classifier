"""Scoring against the OFFICIAL labels (`dc_work/official/labels_official.parquet`).

Used by task 2 (frozen beta on never-seen signals) and task 3 (retrains).
Implements the protocol's rules:

* a detector is **unscorable** if it had zero actuations in the window, or if its official
  phase never turns green in that window (not a candidate);
* the **headline** is accuracy over detectors that got an ANSWER (beta's >= 5 actuation
  rule) and are scorable; "all real channels" (unanswered = wrong) is the footnote;
* non-standard = the official label differs from the ODOT standard wiring table, or the
  channel number is > 40;
* errors are grouped into concurrent pairs vs other.

Label source is stated in every table: PHASE = official timing database.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CONCURRENT_PAIRS, DC_WORK, DEFAULT_PHASE  # noqa: E402

OFFICIAL = DC_WORK / "official"
STG = OFFICIAL / "stg"
COV = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]


def load_official() -> pd.DataFrame:
    o = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    o["Detector"] = o.Detector.astype(int)
    return o


def candidates_in_window(cache: Path, t0: str, t1: str, threads: int = 8) -> pd.DataFrame:
    """(DeviceId, Phase) for every phase with a Begin Green inside [t0, t1)."""
    from common import connect
    con = connect(threads=threads)
    g = (cache / "events" / "**" / "*.parquet").as_posix()
    df = con.sql(f"""SELECT DeviceId, Parameter::INT AS cand
                     FROM read_parquet('{g}', hive_partitioning=true)
                     WHERE EventId = 1 AND Parameter BETWEEN 1 AND 16
                       AND Timestamp >= TIMESTAMP '{t0}' AND Timestamp < TIMESTAMP '{t1}'
                     GROUP BY 1,2""").df()
    con.close()
    return df


def prepare(pred: pd.DataFrame, off: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """One row per official channel that exists in the prediction universe."""
    p = pred[["DeviceId", "Detector", "phase_pred", "phase_prob", "phase_guess",
              "phase_guess_prob", "phase_2nd", "n_actuations", "status",
              "n_candidate_phases"]].copy()
    p["Detector"] = p.Detector.astype(int)
    m = off.merge(p, on=["DeviceId", "Detector"], how="inner")
    m = m[m.target_type.isin(["phase", "overlap"])].copy()
    m["n_actuations"] = m.n_actuations.fillna(0).astype(int)
    m["answered"] = m.phase_pred.notna()
    cs = cand.assign(_c=1)
    m = m.merge(cs, left_on=["DeviceId", "target_num"], right_on=["DeviceId", "cand"],
                how="left")
    m["label_is_candidate"] = m._c.notna() & (m.target_type == "phase")
    m = m.drop(columns=["_c", "cand"])
    m["unscorable_no_act"] = m.n_actuations < 1
    m["unscorable_not_green"] = (~m.label_is_candidate) & (m.target_type == "phase")
    m["scorable"] = ~(m.unscorable_no_act | m.unscorable_not_green)
    m["correct"] = ((m.target_type == "phase") &
                    (m.phase_pred == m.target_num).fillna(False).astype(bool))
    # lenient: any of the channel's called phases (primary + additional)
    def _add(s):
        return [int(t) for t in str(s or "").replace(";", ",").split(",")
                if t.strip().isdigit()]
    m["correct_any_called"] = [
        bool(c) or (t == "phase" and not pd.isna(pp) and int(pp) in _add(a))
        for c, t, pp, a in zip(m.correct, m.target_type, m.phase_pred,
                               m.additional_call_phases)]
    std = m.Detector.map(DEFAULT_PHASE)
    m["std_phase"] = std
    m["is_standard"] = (m.target_type == "phase") & (std == m.target_num) & (m.Detector <= 40)
    m["pair_kind"] = [
        "correct" if c else ("no_answer" if not a else
                             ("concurrent" if frozenset((int(t), int(pp))) in CONCURRENT_PAIRS
                              else "other"))
        for c, a, t, pp in zip(m.correct, m.answered, m.target_num,
                               m.phase_pred.fillna(-1))]
    m["delay_bin"] = pd.cut(m.delay, [-.1, .001, 5.001, 100],
                            labels=["none", "small (<=5 s)", "large (>5 s)"])
    m["extend_bin"] = pd.cut(m.extend, [-.1, .001, 2.001, 100],
                             labels=["none", "small (<=2 s)", "large (>2 s)"])
    return m


def summary(m: pd.DataFrame, tag: str) -> dict:
    ph = m[m.target_type == "phase"]
    ans = ph[ph.answered & ph.scorable]
    real = ph[ph.n_actuations > 0]
    out = {
        "tag": tag,
        "n_official_channels": int(len(m)),
        "n_phase_targets": int(len(ph)),
        "n_overlap_targets": int((m.target_type == "overlap").sum()),
        "n_real": int(len(real)),
        "n_answered_scorable": int(len(ans)),
        "n_unscorable_no_act": int(ph.unscorable_no_act.sum()),
        "n_unscorable_not_green": int(ph.unscorable_not_green.sum()),
        "n_not_answered_real": int((real.answered == False).sum()),  # noqa: E712
        "acc": float(ans.correct.mean()) if len(ans) else np.nan,
        "acc_any_called": float(ans.correct_any_called.mean()) if len(ans) else np.nan,
        "coverage_of_real": float(len(ans) / max(len(real), 1)),
        "acc_all_real": float(real.correct.mean()) if len(real) else np.nan,
    }
    ns = ans[~ans.is_standard]
    out["acc_nonstd"] = float(ns.correct.mean()) if len(ns) else np.nan
    out["n_nonstd"] = int(len(ns))
    st = ans[ans.is_standard]
    out["acc_std"] = float(st.correct.mean()) if len(st) else np.nan
    out["n_std"] = int(len(st))
    err = ans[~ans.correct]
    out["n_err"] = int(len(err))
    out["n_err_concurrent"] = int((err.pair_kind == "concurrent").sum())
    # overlap-target channels (the beta can only answer phases)
    ov = m[(m.target_type == "overlap") & (m.n_actuations > 0)]
    out["n_overlap_real"] = int(len(ov))
    return out


def table_coverage(m: pd.DataFrame) -> pd.DataFrame:
    ans = m[(m.target_type == "phase") & m.answered & m.scorable]
    rows = []
    for t in COV:
        k = ans[ans.phase_prob >= t]
        rows.append({"min_prob": t, "share_kept": len(k) / max(len(ans), 1),
                     "accuracy": float(k.correct.mean()) if len(k) else np.nan,
                     "n": len(k)})
    return pd.DataFrame(rows)


def table_by(m: pd.DataFrame, col: str) -> pd.DataFrame:
    ans = m[(m.target_type == "phase") & m.answered & m.scorable]
    g = ans.groupby(col, observed=True).agg(n=("correct", "size"), acc=("correct", "mean"))
    return g.reset_index()


def table_additional(m: pd.DataFrame) -> pd.DataFrame:
    ans = m[(m.target_type == "phase") & m.answered & m.scorable]
    rows = []
    for name, sub in [("no additional call phases", ans[ans.n_add_phases == 0]),
                      ("has additional call phases", ans[ans.n_add_phases > 0])]:
        rows.append({"group": name, "n": len(sub),
                     "acc (primary only)": float(sub.correct.mean()) if len(sub) else np.nan,
                     "acc (any called phase)": float(sub.correct_any_called.mean())
                     if len(sub) else np.nan})
    return pd.DataFrame(rows)
