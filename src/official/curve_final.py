"""Accuracy vs amount of data, for the FINAL model on the locked NEWTEST signals.

Same design as the beta's chart: 26 log-spaced sample lengths from 1 minute to the full
span, each measured from the SAME ~10 anchor start times (nested windows, so sampling
noise is shared between points instead of being redrawn), averaged over anchors.  Every
point is a full end-to-end `src/predict.py` run on raw controller events.

    python src/official/curve_final.py --stage events    # one sorted event file
    python src/official/curve_final.py --stage run       # the 26 x 10 predictions
    python src/official/curve_final.py --stage json      # docs/img/final_accuracy_vs_minutes.json

Then `python src/plot_final_accuracy.py` draws the png.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

from common import DC_WORK, REPO  # noqa: E402
import score_final as SF  # noqa: E402

WORK = DC_WORK / "preds" / "final_test_DO_NOT_USE" / "curve"
EVENTS = WORK / "_events_newtest_sorted.parquet"
T0, T1 = pd.Timestamp("2026-09-18 16:15:00"), pd.Timestamp("2026-09-21 10:23:00")
SPAN_MIN = (T1 - T0).total_seconds() / 60.0            # ~ 3968 min = 66.1 h
N_DUR = 26
MIN_ACT = 5

# 10 anchors spread over the span: Friday PM peak, Saturday, Sunday night, Monday AM peak.
ANCHORS = ["2026-09-18 17:00:00", "2026-09-18 21:00:00", "2026-09-19 08:00:00",
           "2026-09-19 12:00:00", "2026-09-19 17:30:00", "2026-09-20 02:00:00",
           "2026-09-20 10:00:00", "2026-09-20 16:00:00", "2026-09-21 02:00:00",
           "2026-09-21 07:30:00"]
DURATIONS = sorted({int(round(v)) for v in
                    np.unique(np.round(np.logspace(0, np.log10(SPAN_MIN), N_DUR)))})


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stage_events(a) -> None:
    """One timestamp-sorted parquet of the 143 NEWTEST signals, so short windows prune."""
    import duckdb
    WORK.mkdir(parents=True, exist_ok=True)
    ids = ",".join("'" + d + "'" for d in SF.newtest_ids())
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=10")
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    src = (SF.STG_CACHE / "**" / "*.parquet").as_posix()
    t0 = time.time()
    con.execute(f"""COPY (
        SELECT lower(DeviceId) AS DeviceId, Timestamp, EventId, Parameter
        FROM read_parquet('{src}', hive_partitioning=true)
        WHERE lower(DeviceId) IN ({ids}) ORDER BY Timestamp
    ) TO '{EVENTS.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 400000)""")
    n = con.sql(f"SELECT count(*), count(DISTINCT DeviceId), min(Timestamp), "
                f"max(Timestamp) FROM read_parquet('{EVENTS.as_posix()}')").fetchone()
    # which labelled channels physically exist (>=1 detector event in the whole span)
    live = con.sql(f"""SELECT DISTINCT DeviceId, CAST(Parameter AS INT) AS Detector
                       FROM read_parquet('{EVENTS.as_posix()}')
                       WHERE EventId IN (81,82)""").df()
    con.close()
    live.to_parquet(WORK / "live_channels.parquet", index=False)
    log(f"{n[0]:,} events / {n[1]} signals, {n[2]} .. {n[3]} in {time.time()-t0:.0f}s; "
        f"{len(live):,} live channels")


def stage_run(a) -> None:
    sys.path.insert(0, str(REPO / "src"))
    import predict as P
    WORK.mkdir(parents=True, exist_ok=True)
    todo = [(d, i) for d in DURATIONS for i in range(len(ANCHORS))]
    for d, i in todo:
        dest = WORK / f"m{d}_a{i}.parquet"
        if dest.exists():
            continue
        start = pd.Timestamp(ANCHORS[i])
        if start + pd.Timedelta(minutes=d) > T1:       # slide earlier, keep the length
            start = T1 - pd.Timedelta(minutes=d)
        end = start + pd.Timedelta(minutes=d)
        t0 = time.time()
        kw = dict(threads=8, memory="8GB", chunk_signals=12, min_actuations=1)
        if "odot_tiebreak" in inspect.signature(P.predict).parameters:
            kw["odot_tiebreak"] = True     # as the original run; since removed from predict.py
        out = P.predict(EVENTS.as_posix(), start=str(start), end=str(end), **kw)
        out["minutes"] = d
        out["anchor"] = i
        out["win_start"] = str(start)
        out.to_parquet(dest, index=False)
        log(f"{d} min anchor {i}: {len(out)} detectors, {time.time()-t0:.0f}s")


def _greens() -> pd.DataFrame:
    """Every Begin Green (event 1) in the span: DeviceId, Timestamp, cand_phase."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'"); con.execute("SET threads=8")
    g = con.sql(f"""SELECT DeviceId, Timestamp, CAST(Parameter AS INT) AS cand_phase
                    FROM read_parquet('{EVENTS.as_posix()}')
                    WHERE EventId = 1 AND Parameter BETWEEN 1 AND 16""").df()
    con.close()
    return g


def stage_json(a) -> None:
    lab = SF.phase_labels()
    flab = SF.function_labels()
    live = pd.read_parquet(WORK / "live_channels.parquet")
    live["Detector"] = live.Detector.astype(int)
    lab = lab.merge(live, on=["DeviceId", "Detector"], how="inner")
    n_lab = len(lab)
    greens = _greens()
    log(f"{n_lab:,} labelled live channels; {len(greens):,} begin-green events")
    raw = []
    for d in DURATIONS:
        rows = []
        for i in range(len(ANCHORS)):
            f = WORK / f"m{d}_a{i}.parquet"
            if not f.exists():
                continue
            pr = pd.read_parquet(f)
            pr["DeviceId"] = pr.DeviceId.str.lower()
            pr["Detector"] = pr.Detector.astype(int)
            pr = SF.untiebreak(pr)
            start = pd.Timestamp(pr.win_start.iloc[0])
            end = start + pd.Timedelta(minutes=int(d))
            gw = greens[(greens.Timestamp >= start) & (greens.Timestamp < end)]
            ck = set(gw.DeviceId + "|" + gw.cand_phase.astype(str))
            m = pr.merge(lab, on=["DeviceId", "Detector"], how="inner")
            ans = (m.phase_pred.notna() & (m.n_actuations >= MIN_ACT)).to_numpy()
            has_cand = (m.DeviceId + "|" + m.Phase.astype(int).astype(str)).isin(ck).to_numpy()
            s = m[ans & has_cand]
            ok = (s.phase_pred.astype(float) == s.Phase.astype(float)).to_numpy()
            okt = (s.phase_pred_tb.astype(float) == s.Phase.astype(float)).to_numpy()
            fm = pr.merge(flab, on=["DeviceId", "Detector"], how="inner")
            fa = fm[fm.function_pred.notna() & (fm.n_actuations >= MIN_ACT)]
            rows.append(dict(
                n_answered=int(ans.sum()),
                n_no_actuations=int((m.n_actuations == 0).sum()),
                n_not_enough=int(((m.n_actuations > 0) &
                                  (m.n_actuations < MIN_ACT)).sum()),
                n_label_never_green=int((ans & ~has_cand).sum()),
                acc_phase=float(ok.mean()) if len(s) else np.nan,
                acc_phase_tiebreak=float(okt.mean()) if len(s) else np.nan,
                acc_function=float((fa.function_pred == fa.func5).mean()) if len(fa) else np.nan,
                n_scored=int(len(s)), n_func=int(len(fa)),
                answered_share=float(ans.sum()) / n_lab))
        if not rows:
            continue
        r = pd.DataFrame(rows)
        raw.append(dict(minutes=int(d), n_anchors=int(len(r)),
                        **{k: round(float(np.nanmean(r[k])), 6) for k in
                           ("acc_phase", "acc_phase_tiebreak", "acc_function",
                            "answered_share")},
                        **{k: round(float(r[k].mean()), 1) for k in
                           ("n_answered", "n_no_actuations", "n_not_enough",
                            "n_label_never_green", "n_scored", "n_func")}))
        log(f"{d:5d} min: phase {raw[-1]['acc_phase']:.4f} "
            f"function {raw[-1]['acc_function']:.4f} "
            f"answered {raw[-1]['answered_share']:.3f}")
    out = {
        "chart": "accuracy and answer rate vs how much data is in the sample",
        "generated": time.strftime("%Y-%m-%d"),
        "model": "models/final_v1 (src/predict.py, end to end from raw events)",
        "evaluation": ("the 143 NEWTEST signals -- locked away from every training and "
                       "tuning run of the study and scored exactly once. Phase truth = the "
                       "official controller timing; function truth = the current config "
                       "table (5 classes)."),
        "sampling": (f"{len(DURATIONS)} log-spaced sample lengths from 1 min to "
                     f"{int(SPAN_MIN)} min, each measured from the same {len(ANCHORS)} "
                     "anchor start times (nested windows; an anchor slides earlier when "
                     "the window would run past the end of the data), averaged over anchors."),
        "minimum_evidence_rule": {"min_actuations": MIN_ACT,
                                  "meaning": "a detector with fewer than this many ON "
                                             "events in the sample gets no answer"},
        "accuracy_definition": ("share of answered detectors whose predicted phase matches "
                                "the official timing; answered detectors whose labelled "
                                "phase never turns green inside the window are left out and "
                                "counted separately (n_label_never_green). Function is the "
                                "5-class accuracy on the answered detectors that carry a "
                                "config-table function."),
        "n_labelled_detectors": int(n_lab),
        "n_signals": int(lab.DeviceId.nunique()),
        "anchors": ANCHORS,
        "by_duration_raw": raw,
    }
    img = REPO / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(img / "final_accuracy_vs_minutes.json", "w"), indent=1)
    log(f"wrote {img / 'final_accuracy_vs_minutes.json'} ({len(raw)} points)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["events", "run", "json"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
