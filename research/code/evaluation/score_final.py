"""THE SINGLE FINAL SCORING of the study -- the locked test signals, used exactly once.

Three evaluation sets, all scored end to end through the shipped `src/predict.py`
(raw controller events in, one row per detector out -- no cached feature tables):

    TESTDEC   43 TEST signals   (data/splits/test_config.csv)      Dec-2024 data
    NEWTEST  143 NEWTEST signals (dc_work/official/newtest_signals.csv)  Sept-2026 data
    TESTSTG   43 TEST signals                                      Sept-2026 data

For every set the FINAL model (`models/final_v1`) and the OLD BETA (`models/beta_v0`,
through `src/predict_beta_v0.py`) are run on **exactly the same rows**, so the before/after
comparison is paired.  Phase truth = the official controller timing; function truth = the
current hand-maintained config table.

These runs were made while `predict.py` still carried the optional ODOT tie-breaker, with
`odot_tiebreak=True`, so both columns could be read from one pass: the tie-breaker only ever
exchanges the top two candidates' probabilities, so the OFF answer is recovered exactly by
swapping the flagged detectors back.  It has since been dropped from the shipped model, so a
re-run yields the OFF column only -- which is the shipped behaviour and the headline number
either way.

    python src/official/score_final.py --stage predict --set NEWTEST
    python src/official/score_final.py --stage score
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
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path

from common import CONCURRENT_PAIRS, DC_WORK, DEFAULT_PHASE, REPO  # noqa: E402

OUTDIR = DC_WORK / "preds" / "final_test_DO_NOT_USE"
OFFICIAL = DC_WORK / "official"
STG_CACHE = OFFICIAL / "stg" / "cache" / "events"
DEC_CACHE = DC_WORK / "cache" / "events"
STG_RAW = DC_WORK / "data" / "staging"
TEST_CFG = REPO / "data" / "splits" / "test_config.csv"
NEWTEST_CSV = OFFICIAL / "newtest_signals.csv"
CFG_LABELS = DC_WORK / "data" / "labels" / "detector_config_current.parquet"
LABEL_MAP = DC_WORK / "data" / "labels" / "function_label_map_v2.csv"
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
MIN_ACT = 5

# Window anchors.  Dec-2024 = 3 weekdays (Mon 2nd .. Wed 4th).
# Sept-2026 staging = Fri 18th 16:15 .. Mon 21st 10:23 (two thirds weekend).
WINDOWS = {
    "DEC": [("full", None, None),
            ("h6_a", "2024-12-03 06:00:00", 6), ("h6_b", "2024-12-03 12:00:00", 6),
            ("h6_c", "2024-12-02 18:00:00", 6),
            ("m30_a", "2024-12-03 07:30:00", 0.5), ("m30_b", "2024-12-03 12:00:00", 0.5),
            ("m30_c", "2024-12-03 21:30:00", 0.5), ("m30_d", "2024-12-04 17:00:00", 0.5)],
    "STG": [("full", None, None),
            ("h6_a", "2026-09-20 06:00:00", 6), ("h6_b", "2026-09-19 12:00:00", 6),
            ("h6_c", "2026-09-21 04:00:00", 6),
            ("m30_a", "2026-09-21 07:30:00", 0.5), ("m30_b", "2026-09-19 12:00:00", 0.5),
            ("m30_c", "2026-09-19 21:30:00", 0.5), ("m30_d", "2026-09-18 17:00:00", 0.5)],
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- the sets
def test_ids() -> list[str]:
    return sorted(pd.read_csv(TEST_CFG).DeviceId.str.lower().unique())


def newtest_ids() -> list[str]:
    return sorted(pd.read_csv(NEWTEST_CSV).DeviceId.str.lower().unique())


def set_spec(name: str) -> tuple[list[str], str, str]:
    """(device ids, events glob, window family)."""
    if name == "TESTDEC":
        return test_ids(), (DEC_CACHE / "**" / "*.parquet").as_posix(), "DEC"
    if name == "NEWTEST":
        return newtest_ids(), (STG_CACHE / "**" / "*.parquet").as_posix(), "STG"
    if name == "TESTSTG":
        p = OUTDIR / "_events_test_stg.parquet"
        if not p.exists():
            extract_test_staging(p)
        return test_ids(), p.as_posix(), "STG"
    raise ValueError(name)


def extract_test_staging(dest: Path) -> None:
    """Pull the 43 TEST signals' Sept-2026 events out of the raw staging dump once.

    Stage 10 deliberately left them out of the staging cache, so this is the only place
    they are read -- inside the final-scoring step, after everything is frozen."""
    import duckdb
    dest.parent.mkdir(parents=True, exist_ok=True)
    ids = ",".join("'" + d + "'" for d in test_ids())
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=10")
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    src = (STG_RAW / "**" / "*.parquet").as_posix()
    t0 = time.time()
    con.execute(f"""COPY (
        SELECT DISTINCT lower(CAST(DeviceId AS VARCHAR)) AS DeviceId,
               CAST(Timestamp AS TIMESTAMP) AS Timestamp,
               CAST(EventId AS USMALLINT) AS EventId,
               CAST(Parameter AS USMALLINT) AS Parameter
        FROM read_parquet('{src}', union_by_name=true)
        WHERE lower(CAST(DeviceId AS VARCHAR)) IN ({ids})
    ) TO '{dest.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    n = con.sql(f"SELECT count(*), count(DISTINCT DeviceId) FROM "
                f"read_parquet('{dest.as_posix()}')").fetchone()
    con.close()
    log(f"extracted {n[0]:,} events / {n[1]} TEST signals to {dest.name} "
        f"in {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ prediction
def _predict_module(which: str):
    if which == "final":
        import predict as P
        return P
    try:
        import predict_beta_v0 as P                 # the beta's own frozen entry point
    except ImportError as exc:                      # pragma: no cover
        raise SystemExit(
            "the beta arm needs src/predict_beta_v0.py, which was removed after the final "
            "scoring; restore it from git history to re-run this comparison. The predictions "
            "it produced are already in dc_work/preds/final_test_DO_NOT_USE/.") from exc
    return P


def run_predictions(setname: str, which: str, only_win: str | None = None) -> None:
    ids, glob, fam = set_spec(setname)
    P = _predict_module(which)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for tag, start, hours in WINDOWS[fam]:
        if only_win and tag != only_win:
            continue
        dest = OUTDIR / f"{setname}_{which}_{tag}.parquet"
        if dest.exists():
            log(f"skip {dest.name} (exists)")
            continue
        end = None
        if start is not None:
            end = str(pd.Timestamp(start) + pd.Timedelta(hours=hours))
        t0 = time.time()
        kw = dict(device_ids=ids, threads=8, memory="8GB", chunk_signals=12,
                  min_actuations=1)
        if "odot_tiebreak" in inspect.signature(P.predict).parameters:
            kw["odot_tiebreak"] = True      # the original run; see the module docstring
        out = P.predict(glob, start=start, end=end, **kw)
        out["win"] = tag
        out.to_parquet(dest, index=False)
        log(f"{dest.name}: {len(out)} detectors, {out.DeviceId.nunique()} signals, "
            f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------- the truth
def phase_labels() -> pd.DataFrame:
    o = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    o = o[o.target_type == "phase"].copy()
    o["DeviceId"] = o.DeviceId.str.lower()
    o["Detector"] = o.Detector.astype(int)
    return o[["DeviceId", "Detector", "target_num"]].rename(columns={"target_num": "Phase"})


def function_labels() -> pd.DataFrame:
    """The current config table mapped to the 5 classes -- hold-outs KEPT (this is the exam)."""
    cfg = pd.read_parquet(CFG_LABELS)
    cfg["DeviceId"] = cfg.DeviceId.str.lower()
    cfg["Detector"] = cfg.Detector.astype(int)
    mp = pd.read_csv(LABEL_MAP)
    m = dict(zip(mp.raw_key.astype(str), mp.std_function))
    cfg["func5"] = cfg.Function.astype(str).str.strip().str.lower().map(m)
    cfg = cfg[cfg.func5.notna() & cfg.Detector.between(1, 64)]
    return cfg[["DeviceId", "Detector", "func5", "Function"]].reset_index(drop=True)


def candidates(setname: str, only: list[str] | None = None) -> pd.DataFrame:
    """(DeviceId, win, cand_phase): every phase with a Begin Green inside each window."""
    import duckdb
    ids, glob, fam = set_spec(setname)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=10")
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    idl = ",".join("'" + d + "'" for d in ids)
    rows = []
    for tag, start, hours in WINDOWS[fam]:
        if only and tag not in only:
            continue
        w = [f"EventId = 1", "Parameter BETWEEN 1 AND 16",
             f"lower(CAST(DeviceId AS VARCHAR)) IN ({idl})"]
        if start is not None:
            end = str(pd.Timestamp(start) + pd.Timedelta(hours=hours))
            w += [f"Timestamp >= TIMESTAMP '{start}'", f"Timestamp < TIMESTAMP '{end}'"]
        d = con.sql(f"""SELECT DISTINCT lower(CAST(DeviceId AS VARCHAR)) AS DeviceId,
                               CAST(Parameter AS INT) AS cand_phase
                        FROM read_parquet('{glob}', union_by_name=true)
                        WHERE {' AND '.join(w)}""").df()
        d["win"] = tag
        rows.append(d)
    con.close()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------- score
def untiebreak(df: pd.DataFrame) -> pd.DataFrame:
    """Recover the tie-breaker-OFF answer: the flagged detectors' top two are swapped back.

    Predictions made by the current `predict.py` have no `tiebreak_applied` column at all
    (the tie-breaker was removed); then the two columns are simply the same."""
    d = df.copy()
    m = (d.tiebreak_applied.fillna(False).astype(bool) if "tiebreak_applied" in d.columns
         else pd.Series(False, index=d.index))
    d["phase_pred_tb"] = d.phase_pred
    d["phase_prob_tb"] = d.phase_prob
    # the tie-breaker exchanged the two candidates' probabilities, so the pre-swap winner is
    # the runner-up carrying the winner's probability.
    d["phase_pred"] = np.where(m, d.phase_2nd, d.phase_pred)
    return d


def phase_block(pred: pd.DataFrame, lab: pd.DataFrame, cand: pd.DataFrame,
                col: str, pcol: str, min_act: int, min_prob: float = 0.0) -> dict:
    d = pred.merge(lab, on=["DeviceId", "Detector"], how="inner").copy()
    n_lab = int(len(d))
    ckeys = set(cand.DeviceId.astype(str) + "|" + cand.win.astype(str) + "|" +
                cand.cand_phase.astype(int).astype(str))
    d["has_cand"] = (d.DeviceId.astype(str) + "|" + d.win.astype(str) + "|" +
                     d.Phase.astype(int).astype(str)).isin(ckeys)
    answered = d[col].notna() & (d.n_actuations >= min_act) & (d[pcol].fillna(0) >= min_prob)
    scorable = answered & d.has_cand
    s = d[scorable]
    ok = (s[col].astype(float) == s.Phase.astype(float)).to_numpy()
    std = s.Detector.map(DEFAULT_PHASE)
    ns = (~((std == s.Phase) & (s.Detector <= 40))).to_numpy()
    err = s[~ok]
    nconc = int(sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                    for a, b in zip(err.Phase, err[col])))
    out = {"n_labelled": n_lab, "n_answered": int(answered.sum()),
           "n_scored": int(len(s)),
           "share_answered": float(answered.mean()) if n_lab else np.nan,
           "n_no_actuations": int((d.n_actuations == 0).sum()),
           "n_below_min_actuations": int(((d.n_actuations > 0) &
                                          (d.n_actuations < min_act)).sum()),
           "n_label_never_green": int((answered & ~d.has_cand).sum()),
           "accuracy": float(ok.mean()) if len(s) else np.nan,
           "n_errors": int((~ok).sum()),
           "n_errors_concurrent": nconc,
           "acc_nonstandard": float(ok[ns].mean()) if ns.any() else np.nan,
           "n_nonstandard": int(ns.sum()),
           "acc_all_labelled": float((d[col].astype(float) ==
                                      d.Phase.astype(float)).mean()) if n_lab else np.nan}
    for th in (0.8, 0.9):
        k = (s[pcol] >= th).to_numpy()
        out[f"cov{th}"] = float(k.mean()) if len(s) else np.nan
        out[f"acc_at_cov{th}"] = float(ok[k].mean()) if k.any() else np.nan
        # coverage of every labelled detector, not just the answered ones
        k2 = (d[pcol].fillna(0) >= th) & d.has_cand
        out[f"cov{th}_of_labelled"] = float(k2.mean()) if n_lab else np.nan
    return out


def function_block(pred: pd.DataFrame, flab: pd.DataFrame, min_act: int) -> dict:
    d = pred.merge(flab, on=["DeviceId", "Detector"], how="inner")
    d = d[d.function_pred.notna() & (d.n_actuations >= min_act)]
    if not len(d):
        return {"n": 0}
    y, p = d.func5.to_numpy(), d.function_pred.to_numpy()
    out = {"n": int(len(d)), "acc_5class": float((y == p).mean())}
    apc = np.isin(y, ["Advance", "Presence", "Count"])
    out["n_apc"] = int(apc.sum())
    out["acc_apc"] = float((y[apc] == p[apc]).mean()) if apc.any() else np.nan
    for c in CLASSES5:
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn_ = int(((y == c) & (p != c)).sum())
        out[f"{c}_precision"] = tp / (tp + fp) if tp + fp else np.nan
        out[f"{c}_recall"] = tp / (tp + fn_) if tp + fn_ else np.nan
        out[f"{c}_n"] = tp + fn_
    for th in (0.7, 0.8, 0.9):
        k = d.function_prob >= th
        out[f"cov{th}"] = float(k.mean())
        out[f"acc_at_cov{th}"] = float((y[k.to_numpy()] == p[k.to_numpy()]).mean()) \
            if k.any() else np.nan
    return out


def score_set(setname: str) -> dict:
    lab, flab = phase_labels(), function_labels()
    wins = [w for w, _, _ in WINDOWS[set_spec(setname)[2]]]
    cand = candidates(setname)
    res = {}
    for which in ("final", "beta"):
        files = {w: OUTDIR / f"{setname}_{which}_{w}.parquet" for w in wins}
        files = {w: f for w, f in files.items() if f.exists()}
        if not files:
            continue
        pr = pd.concat([pd.read_parquet(f) for f in files.values()], ignore_index=True)
        pr["DeviceId"] = pr.DeviceId.str.lower()
        pr["Detector"] = pr.Detector.astype(int)
        pr = untiebreak(pr)
        block = {}
        for w in files:
            sub = pr[pr.win == w]
            block[w] = {
                "phase_tb_off": phase_block(sub, lab, cand, "phase_pred", "phase_prob",
                                            MIN_ACT),
                "phase_tb_on": phase_block(sub, lab, cand, "phase_pred_tb",
                                           "phase_prob_tb", MIN_ACT),
                "phase_minprob09": phase_block(sub, lab, cand, "phase_pred", "phase_prob",
                                               1, 0.9),
                "function": function_block(sub, flab, MIN_ACT),
            }
        # means over the 6 h and 30 min anchors
        for fam, keys in (("h6", [w for w in files if w.startswith("h6")]),
                          ("m30", [w for w in files if w.startswith("m30")])):
            if not keys:
                continue
            block[f"MEAN_{fam}"] = {
                sec: {k: float(np.mean([block[w][sec][k] for w in keys]))
                      for k in block[keys[0]][sec]
                      if isinstance(block[keys[0]][sec][k], (int, float))}
                for sec in ("phase_tb_off", "phase_tb_on", "phase_minprob09", "function")}
        res[which] = block
    return res


def stage_predict(a) -> None:
    for s in a.sets.split(","):
        for which in a.models.split(","):
            run_predictions(s, which, a.win)


def stage_score(a) -> None:
    out = {}
    for s in a.sets.split(","):
        try:
            out[s] = score_set(s)
        except Exception as exc:
            log(f"{s}: {type(exc).__name__}: {exc}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTDIR / "final_scores.json", "w"), indent=1, default=str)
    for s, r in out.items():
        for which, b in r.items():
            for w in ("full", "MEAN_h6", "MEAN_m30"):
                if w not in b:
                    continue
                p = b[w]["phase_tb_off"]
                pt = b[w]["phase_tb_on"]
                f = b[w]["function"]
                log(f"{s:8s} {which:5s} {w:9s} phase {p['accuracy']:.4f} "
                    f"(tb {pt['accuracy']:.4f}) n={p['n_scored']} "
                    f"ans={p['share_answered']:.3f} nonstd={p['acc_nonstandard']:.4f} "
                    f"cov.9={p['cov0.9']:.3f}/{p['acc_at_cov0.9']:.4f} "
                    f"func5={f.get('acc_5class', float('nan')):.4f} "
                    f"apc={f.get('acc_apc', float('nan')):.4f}")
    log(f"wrote {OUTDIR / 'final_scores.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["predict", "score", "extract"])
    ap.add_argument("--sets", default="TESTDEC,NEWTEST,TESTSTG")
    ap.add_argument("--models", default="final,beta")
    ap.add_argument("--win", default=None)
    a = ap.parse_args()
    if a.stage == "extract":
        extract_test_staging(OUTDIR / "_events_test_stg.parquet")
        return
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
