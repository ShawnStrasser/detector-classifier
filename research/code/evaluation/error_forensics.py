"""Stage 04 step 1: forensics on the stage-01 LightGBM OOF errors.

Categorises every classifiable phase error and tests the "whole-signal renumbering"
hypothesis (are a signal's labels a permutation of the model's predictions?).

    python src/error_forensics.py            # full report -> preds/error_forensics_v2.json
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import (CACHE, CONCURRENT_PAIRS, DEFAULT_PHASE, FOLDS_CSV,  # noqa: E402
                    LABELS_DEV, PREDS, connect)

OUT_JSON = PREDS / "error_forensics_v2.json"


def load_all() -> pd.DataFrame:
    """One row per labelled DEV detector: label, top-1/top-2 prediction, context."""
    pred = pd.read_parquet(PREDS / "phase_oof.parquet")
    lab = pd.read_parquet(LABELS_DEV)
    folds = pd.read_csv(FOLDS_CSV)
    con = connect(threads=6)
    meta = con.sql(f"""SELECT DeviceId, Detector::INT AS Detector, n_on, occ_frac
                       FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}')""").df()
    cand = con.sql(f"""SELECT DeviceId, unnest(cand_phases)::INT AS cand_phase
                       FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}')""").df()
    con.close()
    health = pd.read_parquet(Path(str(CACHE).replace("cache", "atspm")) / "detector_health.parquet",
                             columns=["DeviceId", "Detector", "health_flag"])
    health["Detector"] = health.Detector.astype(int)

    pred = pred.sort_values(["DeviceId", "Detector", "prob"], ascending=[True, True, False])
    g = pred.groupby(["DeviceId", "Detector"])
    top = g.head(1).rename(columns={"cand_phase": "pred1", "prob": "p1"})
    second = g.nth(1).rename(columns={"cand_phase": "pred2", "prob": "p2"})
    t = top[["DeviceId", "Detector", "pred1", "p1"]].merge(
        second[["DeviceId", "Detector", "pred2", "p2"]], on=["DeviceId", "Detector"], how="left")

    ev = lab.merge(t, on=["DeviceId", "Detector"], how="left")
    ev = ev.merge(folds, on="DeviceId", how="left")
    ev = ev.merge(meta, on=["DeviceId", "Detector"], how="left")
    ev = ev.merge(health, on=["DeviceId", "Detector"], how="left")
    cand_set = cand.groupby("DeviceId")["cand_phase"].apply(set)
    ev["cands"] = ev.DeviceId.map(cand_set)
    ev["label_is_cand"] = [isinstance(c, set) and (p in c) for c, p in zip(ev.cands, ev.Phase)]
    ev["classifiable"] = ev.n_on.fillna(0) >= 1
    ev["correct"] = ev.pred1 == ev.Phase
    ev["std_phase"] = ev.Detector.map(DEFAULT_PHASE)
    ev["margin"] = ev.p1 - ev.p2.fillna(0.0)
    return ev


def categorise(err: pd.DataFrame) -> pd.Series:
    cats = []
    for r in err.itertuples():
        if not r.label_is_cand:
            cats.append("label_not_candidate")
        elif frozenset((int(r.Phase), int(r.pred1))) in CONCURRENT_PAIRS:
            cats.append("concurrent_pair")
        elif (int(r.Phase) % 2) == (int(r.pred1) % 2) and abs(int(r.Phase) - int(r.pred1)) in (2, 4):
            cats.append("same_ring")
        elif r.n_on < 50:
            cats.append("low_volume")
        else:
            cats.append("other")
        # low volume overrides only when not already special
    c = pd.Series(cats, index=err.index)
    # refine: any error on a detector with <50 actuations is flagged low volume too
    return c


# ------------------------------------------------------- renumbering analysis
def renumber_analysis(ev: pd.DataFrame, min_conf: float = 0.7) -> pd.DataFrame:
    """Per signal: is the label set a *permutation* of the model's confident predictions?

    We look for a bijection pi over the signal's phases such that label = pi(pred) for
    most confident detectors.  A signal where pi != identity but explains >= 80 % of the
    confident detectors (and fixes >= 2 errors) is flagged as suspected renumbering.
    """
    rows = []
    for dev, g in ev[ev.classifiable & ev.pred1.notna()].groupby("DeviceId"):
        gc = g[g.p1 >= min_conf]
        if len(gc) < 4:
            continue
        n_err = int((~gc.correct).sum())
        if n_err < 2:
            continue
        # greedy best map pred -> label from the confusion counts
        conf = pd.crosstab(gc.pred1.astype(int), gc.Phase.astype(int))
        preds = list(conf.index)
        labs = list(conf.columns)
        best, best_hit = None, -1
        # exhaustive only when small; else greedy
        if len(preds) <= 7 and len(labs) <= 7:
            for perm in permutations(labs, min(len(preds), len(labs))):
                hit = sum(conf.loc[p, q] for p, q in zip(preds, perm) if q in conf.columns)
                if hit > best_hit:
                    best_hit, best = hit, dict(zip(preds, perm))
        else:
            best = {p: conf.loc[p].idxmax() for p in preds}
            best_hit = sum(conf.loc[p, q] for p, q in best.items())
        ident = sum(conf.loc[p, p] for p in preds if p in conf.columns)
        rows.append(dict(DeviceId=dev, n_conf=len(gc), n_err_conf=n_err,
                         identity_hits=int(ident), perm_hits=int(best_hit),
                         gain=int(best_hit - ident),
                         perm={int(k): int(v) for k, v in best.items()},
                         is_identity=all(k == v for k, v in best.items())))
    r = pd.DataFrame(rows)
    if len(r):
        r["perm_frac"] = r.perm_hits / r.n_conf
        r = r.sort_values("gain", ascending=False)
    return r


def main() -> None:
    ev = load_all()
    cl = ev[ev.classifiable].copy()
    err = cl[~cl.correct].copy()
    err["category"] = categorise(err)
    err["low_vol"] = err.n_on < 50

    res = {
        "n_labelled": int(len(ev)),
        "n_classifiable": int(len(cl)),
        "acc_all": float(ev.correct.mean()),
        "acc_classifiable": float(cl.correct.mean()),
        "n_errors_classifiable": int(len(err)),
        "categories": err.category.value_counts().to_dict(),
        "n_low_volume_errors": int(err.low_vol.sum()),
        "confident_errors_0.8": int((err.p1 >= 0.8).sum()),
        "confident_errors_0.9": int((err.p1 >= 0.9).sum()),
        "confident_err_matching_std": int(((err.p1 >= 0.8) &
                                           (err.pred1 == err.std_phase)).sum()),
        "label_matching_std_among_conf_err": int(((err.p1 >= 0.8) &
                                                  (err.Phase == err.std_phase)).sum()),
    }
    # concurrent pair table
    err["pair"] = [f"{min(int(a),int(b))}<->{max(int(a),int(b))}" if
                   frozenset((int(a), int(b))) in CONCURRENT_PAIRS else
                   f"{int(a)}->{int(b)}" for a, b in zip(err.Phase, err.pred1)]
    res["top_pairs"] = err.pair.value_counts().head(15).to_dict()

    # is the label the model's 2nd choice?
    res["label_is_top2"] = int((err.pred2 == err.Phase).sum())
    res["label_rank_beyond2"] = int(len(err) - (err.pred2 == err.Phase).sum())

    # per-signal error concentration
    per_sig = err.groupby("DeviceId").size().sort_values(ascending=False)
    res["signals_with_errors"] = int(len(per_sig))
    res["signals_with_3plus_errors"] = int((per_sig >= 3).sum())
    res["errors_in_those_signals"] = int(per_sig[per_sig >= 3].sum())

    ren = renumber_analysis(ev)
    susp = ren[(~ren.is_identity) & (ren.gain >= 2)] if len(ren) else ren
    res["n_signals_suspected_renumber"] = int(len(susp))
    res["renumber_signals"] = susp.head(20).to_dict("records") if len(susp) else []

    # label-noise ceiling: assume every confident (p>=T) disagreement is a label error
    for T in (0.7, 0.8, 0.9, 0.95):
        n_conf_err = int((err.p1 >= T).sum())
        res[f"ceiling_if_conf{T}_are_label_errors"] = float(
            (cl.correct.sum() + n_conf_err) / len(cl))

    err_out = err[["DeviceId", "Detector", "Phase", "pred1", "p1", "pred2", "p2",
                   "std_phase", "n_on", "health_flag", "fold", "category", "margin"]]
    err_out.sort_values("p1", ascending=False).to_csv(PREDS / "errors_stage01_categorised.csv",
                                                      index=False)
    if len(ren):
        ren.to_csv(PREDS / "renumber_candidates.csv", index=False)
    json.dump(res, open(OUT_JSON, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
