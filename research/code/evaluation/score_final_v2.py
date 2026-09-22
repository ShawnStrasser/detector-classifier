"""THE SECOND (and last) look at the locked test signals -- final_v1 vs final_v2.

The 43 TEST and 143 NEWTEST signals were opened once before, to score `models/final_v1`
(`results/12_final_model.md`).  This is a SECOND look at the same exam signals, taken
only after the GRU weights, the blend weight, the blend location and the duration cutoff
were frozen on out-of-fold data.  Nothing is changed afterwards.

    NEWTEST 143 signals, Sept-2026: 5 / 15 / 30 / 60 min (4 start times each),
                                    6 h (2 start times) and the full 66 h span
    TEST     43 signals, Sept-2026 and Dec-2024: 30 min (4 start times) and full

Start times are the first four of `src/official/curve_final.py`'s anchor list (Friday PM
peak, Saturday midday, Sunday morning, Monday AM peak), so the short-window numbers share
their sampling luck with the published accuracy-vs-data chart.  Both models are run over
exactly the same rows through `src/predict.py`, end to end from raw events.

    python src/official/score_final_v2.py --stage predict --sets NEWTEST
    python src/official/score_final_v2.py --stage score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path

from common import CONCURRENT_PAIRS, DC_WORK, DEFAULT_PHASE, REPO  # noqa: E402
import score_final as SF  # noqa: E402

OUTDIR = DC_WORK / "preds" / "final_test_DO_NOT_USE" / "v2"
MODELS = {"v2": REPO / "model" / "weights"}   # final_v1 lives in the git history
MIN_ACT = 5

ANCHORS_STG = ["2026-09-18 17:00:00", "2026-09-19 12:00:00",
               "2026-09-20 10:00:00", "2026-09-21 07:30:00"]
ANCHORS_DEC = ["2024-12-02 17:00:00", "2024-12-03 12:00:00",
               "2024-12-04 09:00:00", "2024-12-03 07:30:00"]
END_STG = pd.Timestamp("2026-09-21 10:23:00")
END_DEC = pd.Timestamp("2024-12-05 00:00:00")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def windows(setname: str) -> list[tuple[str, str | None, float | None]]:
    """(tag, start, minutes); start None = the whole span."""
    dec = setname == "TESTDEC"
    anchors = ANCHORS_DEC if dec else ANCHORS_STG
    end = END_DEC if dec else END_STG
    out = []
    durs = [5, 15, 30, 60] if setname == "NEWTEST" else [30]
    for d in durs:
        for i, a in enumerate(anchors):
            s = pd.Timestamp(a)
            if s + pd.Timedelta(minutes=d) > end:
                s = end - pd.Timedelta(minutes=d)
            out.append((f"m{d}_a{i}", str(s), float(d)))
    if setname == "NEWTEST":
        for i in (0, 3):
            s = pd.Timestamp(anchors[i])
            if s + pd.Timedelta(hours=6) > end:
                s = end - pd.Timedelta(hours=6)
            out.append((f"h6_a{i}", str(s), 360.0))
    out.append(("full", None, None))
    return out


def run_predictions(setname: str, which: str, only: str | None = None,
                    gpu: bool = False) -> None:
    import predict as P
    if gpu:
        # research-only accelerator: the SAME exported weights, evaluated with torch on
        # the GPU instead of numpy, so 143 signals x 60 minutes finishes in minutes.
        # Verified equal to the shipped numpy runtime (max |dp| < 1e-6, same argmax).
        from neural.gru_speedup import enable, verify
        v = verify()
        dev = enable()
        log(f"GRU forward pass on {dev}; vs the shipped onnx runtime {v}")
    ids, glob, _ = SF.set_spec(setname)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, start, mins in windows(setname):
        if only and tag != only:
            continue
        dest = OUTDIR / f"{setname}_{which}_{tag}.parquet"
        if dest.exists():
            continue
        end = None if start is None else str(pd.Timestamp(start) +
                                             pd.Timedelta(minutes=mins))
        t0 = time.time()
        out = P.predict(glob, start=start, end=end, device_ids=ids,
                        model_dir=MODELS[which], threads=8, memory="8GB",
                        chunk_signals=12, min_actuations=1)
        out["win"] = tag
        out.to_parquet(dest, index=False)
        log(f"{dest.name}: {len(out)} detectors, {out.DeviceId.nunique()} signals, "
            f"{time.time()-t0:.0f}s")


# ---------------------------------------------------------------------- score
def candidates(setname: str) -> pd.DataFrame:
    import duckdb
    ids, glob, _ = SF.set_spec(setname)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=10")
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    idl = ",".join("'" + d + "'" for d in ids)
    rows = []
    for tag, start, mins in windows(setname):
        w = ["EventId = 1", "Parameter BETWEEN 1 AND 16",
             f"lower(CAST(DeviceId AS VARCHAR)) IN ({idl})"]
        if start is not None:
            end = str(pd.Timestamp(start) + pd.Timedelta(minutes=mins))
            w += [f"Timestamp >= TIMESTAMP '{start}'", f"Timestamp < TIMESTAMP '{end}'"]
        d = con.sql(f"""SELECT DISTINCT lower(CAST(DeviceId AS VARCHAR)) AS DeviceId,
                               CAST(Parameter AS INT) AS cand_phase
                        FROM read_parquet('{glob}', union_by_name=true)
                        WHERE {' AND '.join(w)}""").df()
        d["win"] = tag
        rows.append(d)
    con.close()
    return pd.concat(rows, ignore_index=True)


def block(pred: pd.DataFrame, lab: pd.DataFrame, cand: pd.DataFrame,
          min_act: int = MIN_ACT) -> dict:
    d = pred.merge(lab, on=["DeviceId", "Detector"], how="inner").copy()
    ck = set(cand.DeviceId.astype(str) + "|" + cand.win.astype(str) + "|" +
             cand.cand_phase.astype(int).astype(str))
    d["has_cand"] = (d.DeviceId.astype(str) + "|" + d.win.astype(str) + "|" +
                     d.Phase.astype(int).astype(str)).isin(ck)
    answered = d.phase_pred.notna() & (d.n_actuations >= min_act)
    s = d[answered & d.has_cand]
    ok = (s.phase_pred.astype(float) == s.Phase.astype(float)).to_numpy()
    std = s.Detector.map(DEFAULT_PHASE)
    ns = (~((std == s.Phase) & (s.Detector <= 40))).to_numpy()
    err = s[~ok]
    out = {"n_labelled": int(len(d)), "n_scored": int(len(s)),
           "share_answered": float(answered.mean()) if len(d) else np.nan,
           "accuracy": float(ok.mean()) if len(s) else np.nan,
           "n_errors": int((~ok).sum()),
           "n_errors_concurrent": int(sum(
               frozenset((int(a), int(b))) in CONCURRENT_PAIRS
               for a, b in zip(err.Phase, err.phase_pred))),
           "acc_nonstandard": float(ok[ns].mean()) if ns.any() else np.nan,
           "n_nonstandard": int(ns.sum())}
    for th in (0.8, 0.9):
        k = (s.phase_prob >= th).to_numpy()
        out[f"cov{th}"] = float(k.mean()) if len(s) else np.nan
        out[f"acc_at_cov{th}"] = float(ok[k].mean()) if k.any() else np.nan
    return out


def score_set(setname: str) -> dict:
    lab, flab = SF.phase_labels(), SF.function_labels()
    cand = candidates(setname)
    res = {}
    for which in MODELS:
        files = {t: OUTDIR / f"{setname}_{which}_{t}.parquet"
                 for t, _, _ in windows(setname)}
        files = {t: f for t, f in files.items() if f.exists()}
        if not files:
            continue
        pr = pd.concat([pd.read_parquet(f) for f in files.values()], ignore_index=True)
        pr["DeviceId"] = pr.DeviceId.str.lower()
        pr["Detector"] = pr.Detector.astype(int)
        b = {t: block(pr[pr.win == t], lab, cand) for t in files}
        for fam in ("m5", "m15", "m30", "m60", "h6"):
            keys = [t for t in files if t.startswith(fam + "_")]
            if keys:
                b[f"MEAN_{fam}"] = {k: float(np.mean([b[t][k] for t in keys]))
                                    for k in b[keys[0]]
                                    if isinstance(b[keys[0]][k], (int, float))}
        b["function"] = SF.function_block(pr[pr.win == "full"], flab, MIN_ACT)
        res[which] = b
    return res


def stage_predict(a) -> None:
    for s in a.sets.split(","):
        for which in a.models.split(","):
            run_predictions(s, which, a.win, a.gpu)


def stage_score(a) -> None:
    out = {}
    for s in a.sets.split(","):
        try:
            out[s] = score_set(s)
        except Exception as exc:
            log(f"{s}: {type(exc).__name__}: {exc}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTDIR / "final_scores_v2.json", "w"), indent=1, default=str)
    for s, r in out.items():
        for w in ("MEAN_m5", "MEAN_m15", "MEAN_m30", "MEAN_m60", "MEAN_h6", "full"):
            row = []
            for which in MODELS:
                if which in r and w in r[which]:
                    row.append(f"{which} {r[which][w]['accuracy']:.4f} "
                               f"(n={r[which][w]['n_scored']:.0f})")
            if row:
                log(f"{s:8s} {w:9s} " + "  ".join(row))
    log(f"wrote {OUTDIR / 'final_scores_v2.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["predict", "score"])
    ap.add_argument("--sets", default="NEWTEST,TESTSTG,TESTDEC")
    ap.add_argument("--models", default="v1,v2")
    ap.add_argument("--win", default=None)
    ap.add_argument("--gpu", action="store_true",
                    help="run the GRU forward pass with torch on the GPU (research "
                         "only; identical numbers, see src/neural/gru_speedup.py)")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
