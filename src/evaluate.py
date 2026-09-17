"""Scoring harness for detector -> phase / function predictions.

Implements the prediction-file contract of docs/research_protocol.md.

CLI
---
    python src/evaluate.py --phase preds.parquet [--function f.parquet] [--folds 0]
    python src/evaluate.py --selftest            # reference + sanity predictors

Prediction files
----------------
phase    : parquet with DeviceId, Detector, cand_phase (int), prob (float, sums to 1 per detector)
function : parquet with DeviceId, Detector, p_advance, p_presence, p_count [, p_other]

Rules enforced here
-------------------
* TEST devices are refused unless --final-test is passed.
* HEADLINE accuracy excludes **unscorable** detectors (user decision 2026-09-17):
  (a) zero actuations in the window, or (b) the labeled phase never turns green in the
  window, so it is not a candidate.  Their counts are reported separately and (b) goes on
  the manual-review list.  The old "all labeled detectors" figure (unscorable = wrong) is
  still computed and printed as a footnote.
* Function label space is Advance / Presence / Count / Other; any label outside the
  three maps to Other.  A prediction is Other when max(p_advance,p_presence,p_count)
  < --other-threshold (or when p_other is the argmax, if that column is supplied).

Importable:  evaluate_phase(), evaluate_function(), report(), standard_lookup(),
uniform_predictor().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, CONCURRENT_PAIRS, DEFAULT_PHASE, FOLDS_CSV,  # noqa: E402
                    FUNCTIONS, LABELS_DEV, LABELS_TEST, connect)

COV_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


# ------------------------------------------------------------------ data load
def load_folds() -> pd.DataFrame:
    return pd.read_csv(FOLDS_CSV)


def load_labels(split: str = "dev") -> pd.DataFrame:
    return pd.read_parquet(LABELS_DEV if split == "dev" else LABELS_TEST)


def load_candidates() -> pd.DataFrame:
    """(DeviceId, Phase) for every phase with a Begin Green in that signal's log."""
    con = connect(threads=4)
    df = con.sql(f"""SELECT DeviceId, unnest(cand_phases)::INT AS Phase
                     FROM read_parquet('{(CACHE/'signal_meta.parquet').as_posix()}')""").df()
    con.close()
    return df


def load_active_detectors() -> pd.DataFrame:
    con = connect(threads=4)
    df = con.sql(f"""SELECT DeviceId, Detector::INT AS Detector, n_on, occ_frac, unhealthy
                     FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}')""").df()
    con.close()
    return df


# ---------------------------------------------------------- reference models
def standard_lookup(devices: list[str] | None = None,
                    candidates: pd.DataFrame | None = None,
                    detectors: pd.DataFrame | None = None) -> pd.DataFrame:
    """ODOT standard-wiring lookup as a phase prediction file.

    NOT a model input -- reference line only (protocol says 92.6% on training labels).
    Channel <= 40 -> all mass on the standard phase if it is a candidate, else uniform.
    Channel  > 40 -> uniform over candidates.
    """
    candidates = load_candidates() if candidates is None else candidates
    detectors = load_active_detectors() if detectors is None else detectors
    if devices is not None:
        candidates = candidates[candidates.DeviceId.isin(devices)]
        detectors = detectors[detectors.DeviceId.isin(devices)]
    pairs = detectors[["DeviceId", "Detector"]].merge(candidates, on="DeviceId", how="inner")
    pairs = pairs.rename(columns={"Phase": "cand_phase"})
    std = pairs.Detector.map(DEFAULT_PHASE)
    hit = (std == pairs.cand_phase).astype(float)
    pairs["hit"] = hit
    grp = pairs.groupby(["DeviceId", "Detector"])["hit"].transform("sum")
    n = pairs.groupby(["DeviceId", "Detector"])["hit"].transform("size")
    pairs["prob"] = np.where(grp > 0, pairs.hit / grp.replace(0, 1), 1.0 / n)
    return pairs[["DeviceId", "Detector", "cand_phase", "prob"]]


def uniform_predictor(devices: list[str] | None = None,
                      candidates: pd.DataFrame | None = None,
                      detectors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trivial sanity predictor: uniform over the signal's candidate phases."""
    candidates = load_candidates() if candidates is None else candidates
    detectors = load_active_detectors() if detectors is None else detectors
    if devices is not None:
        candidates = candidates[candidates.DeviceId.isin(devices)]
        detectors = detectors[detectors.DeviceId.isin(devices)]
    pairs = detectors[["DeviceId", "Detector"]].merge(candidates, on="DeviceId", how="inner")
    pairs = pairs.rename(columns={"Phase": "cand_phase"})
    pairs["prob"] = 1.0 / pairs.groupby(["DeviceId", "Detector"])["cand_phase"].transform("size")
    return pairs[["DeviceId", "Detector", "cand_phase", "prob"]]


def prior_function_predictor(labels: pd.DataFrame) -> pd.DataFrame:
    """Sanity predictor for function: the global DEV class prior for everyone."""
    pri = labels.Function.value_counts(normalize=True)
    out = labels[["DeviceId", "Detector"]].copy()
    for f in FUNCTIONS:
        out[f"p_{f.lower()}"] = float(pri.get(f, 0.0))
    return out


# -------------------------------------------------------------------- phase
def _check_no_test(df: pd.DataFrame, final_test: bool) -> None:
    test_dev = set(pd.read_parquet(LABELS_TEST).DeviceId.unique())
    bad = test_dev & set(df.DeviceId.unique())
    if bad and not final_test:
        raise SystemExit(
            f"REFUSING TO SCORE: prediction file contains {len(bad)} TEST devices. "
            "Pass --final-test only when the orchestrator asks for the single final scoring.")


def evaluate_phase(pred: pd.DataFrame, labels: pd.DataFrame,
                   folds: pd.DataFrame | None = None,
                   candidates: pd.DataFrame | None = None,
                   detectors: pd.DataFrame | None = None) -> dict:
    """Per-detector top-1 evaluation.  Returns a dict of tables / scalars."""
    req = {"DeviceId", "Detector", "cand_phase", "prob"}
    missing = req - set(pred.columns)
    if missing:
        raise ValueError(f"phase prediction file missing columns: {missing}")
    pred = pred.copy()
    pred["Detector"] = pred["Detector"].astype(int)
    pred["cand_phase"] = pred["cand_phase"].astype(int)

    # normalisation check
    s = pred.groupby(["DeviceId", "Detector"])["prob"].sum()
    off = float((s - 1.0).abs().max()) if len(s) else 0.0

    # top-1 per detector (deterministic tie-break on the lowest phase number)
    pr = pred.sort_values(["DeviceId", "Detector", "prob", "cand_phase"],
                          ascending=[True, True, False, True])
    top = pr.groupby(["DeviceId", "Detector"], as_index=False).first()
    top = top.rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})

    ev = labels.merge(top, on=["DeviceId", "Detector"], how="left")
    ev["has_pred"] = ev.pred_phase.notna()
    ev["correct"] = (ev.pred_phase == ev.Phase).fillna(False)
    ev["top_prob"] = ev.top_prob.fillna(0.0)

    candidates = load_candidates() if candidates is None else candidates
    detectors = load_active_detectors() if detectors is None else detectors
    cand_set = candidates.assign(_c=1)
    ev = ev.merge(cand_set.rename(columns={"Phase": "Phase"}), on=["DeviceId", "Phase"], how="left")
    ev["label_is_candidate"] = ev["_c"].notna()
    ev = ev.drop(columns=["_c"])
    ev = ev.merge(detectors[["DeviceId", "Detector", "n_on"]], on=["DeviceId", "Detector"], how="left")
    ev["has_events"] = ev.n_on.notna()

    if folds is not None:
        ev = ev.merge(folds, on="DeviceId", how="left")
    else:
        ev["fold"] = -1

    std = ev.Detector.map(DEFAULT_PHASE)
    ev["std_phase"] = std
    ev["is_standard"] = (std == ev.Phase) & (ev.Detector <= 40)
    ev["pair_kind"] = [
        "correct" if c else
        ("no_pred" if not hp else
         ("concurrent" if frozenset((int(t), int(p))) in CONCURRENT_PAIRS else "other"))
        for c, hp, t, p in zip(ev.correct, ev.has_pred, ev.Phase,
                               ev.pred_phase.fillna(-1))]

    # --- unscorable (protocol 2026-09-17): impossible given the sample -------------
    ev["unscorable_no_actuations"] = (~ev.has_events) | (ev.n_on.fillna(0) < 1)
    ev["unscorable_label_not_green"] = ~ev.label_is_candidate
    ev["scorable"] = ~(ev.unscorable_no_actuations | ev.unscorable_label_not_green)
    sc = ev.scorable

    res: dict = {"eval": ev, "prob_sum_max_dev": off, "n": len(ev)}
    # HEADLINE: unscorable detectors excluded
    res["n_scorable"] = int(sc.sum())
    res["n_unscorable"] = int((~sc).sum())
    res["n_unscorable_no_actuations"] = int(ev.unscorable_no_actuations.sum())
    res["n_unscorable_label_not_green"] = int((ev.unscorable_label_not_green
                                               & ~ev.unscorable_no_actuations).sum())
    res["acc_scorable"] = float(ev[sc].correct.mean()) if sc.any() else float("nan")
    f0s = ev[sc & (ev.fold == 0)]
    res["acc_scorable_fold0"] = float(f0s.correct.mean()) if len(f0s) else float("nan")
    res["n_scorable_fold0"] = len(f0s)
    pfs = ev[sc & (ev.fold >= 0)].groupby("fold")["correct"].agg(["mean", "size"])
    res["per_fold_scorable"] = pfs
    res["fold_mean_scorable"] = float(pfs["mean"].mean()) if len(pfs) else float("nan")
    res["fold_sd_scorable"] = float(pfs["mean"].std(ddof=0)) if len(pfs) else float("nan")
    nss = sc & (~ev.is_standard)
    res["acc_scorable_nonstd"] = float(ev[nss].correct.mean()) if nss.any() else float("nan")
    res["n_scorable_nonstd"] = int(nss.sum())
    covs = []
    for t in COV_THRESHOLDS:
        m = sc & (ev.top_prob >= t)
        covs.append({"threshold": t, "coverage": float(m.sum() / max(int(sc.sum()), 1)),
                     "acc_covered": float(ev[m].correct.mean()) if m.any() else float("nan")})
    res["coverage_scorable"] = pd.DataFrame(covs)

    # FOOTNOTE: the old "all labeled detectors" definition (unscorable = wrong)
    res["acc_overall"] = float(ev.correct.mean())
    f0 = ev[ev.fold == 0]
    res["acc_fold0"] = float(f0.correct.mean()) if len(f0) else float("nan")
    res["n_fold0"] = len(f0)
    per_fold = ev[ev.fold >= 0].groupby("fold")["correct"].agg(["mean", "size"])
    res["per_fold"] = per_fold
    res["fold_mean"] = float(per_fold["mean"].mean()) if len(per_fold) else float("nan")
    res["fold_sd"] = float(per_fold["mean"].std(ddof=0)) if len(per_fold) else float("nan")

    # classifiable = at least one detector actuation in the window (stage-02 hygiene rule);
    # every labelled detector is still scored, the unclassifiable ones just count as wrong.
    cl = ev.has_events & (ev.n_on.fillna(0) >= 1)
    ev["classifiable"] = cl
    res["n_classifiable"] = int(cl.sum())
    res["coverage_classifiable"] = float(cl.mean())
    res["acc_classifiable"] = float(ev[cl].correct.mean()) if cl.any() else float("nan")
    f0c = ev[cl & (ev.fold == 0)]
    res["acc_classifiable_fold0"] = float(f0c.correct.mean()) if len(f0c) else float("nan")
    nsc = cl & (~ev.is_standard)
    res["acc_classifiable_nonstd"] = float(ev[nsc].correct.mean()) if nsc.any() else float("nan")
    res["n_classifiable_nonstd"] = int(nsc.sum())

    res["acc_standard"] = float(ev[ev.is_standard].correct.mean()) if ev.is_standard.any() else float("nan")
    res["n_standard"] = int(ev.is_standard.sum())
    ns = ~ev.is_standard
    res["acc_nonstandard"] = float(ev[ns].correct.mean()) if ns.any() else float("nan")
    res["n_nonstandard"] = int(ns.sum())
    res["acc_chan_gt40"] = float(ev[ev.Detector > 40].correct.mean()) if (ev.Detector > 40).any() else float("nan")
    res["n_chan_gt40"] = int((ev.Detector > 40).sum())

    err = ev[~ev.correct]
    res["err_breakdown"] = err.pair_kind.value_counts()
    res["err_pairs"] = (err[err.pair_kind == "concurrent"]
                        .assign(pair=lambda d: [f"{min(a,b)}<->{max(a,b)}"
                                                for a, b in zip(d.Phase.astype(int),
                                                                d.pred_phase.astype(int))])
                        .pair.value_counts())
    res["err_pairs_other"] = (err[err.pair_kind == "other"]
                              .assign(pair=lambda d: [f"{int(a)}->{int(b)}"
                                                      for a, b in zip(d.Phase, d.pred_phase)])
                              .pair.value_counts().head(10))

    res["n_no_pred"] = int((~ev.has_pred).sum())
    res["n_no_events"] = int((~ev.has_events).sum())
    res["n_label_not_candidate"] = int((~ev.label_is_candidate).sum())
    res["unscorable"] = ev.loc[(~ev.has_events) | (~ev.label_is_candidate),
                               ["DeviceId", "Detector", "Phase", "has_events",
                                "label_is_candidate"]]

    cov = []
    for t in COV_THRESHOLDS:
        m = ev.top_prob >= t
        cov.append({"threshold": t, "coverage": float(m.mean()),
                    "acc_covered": float(ev[m].correct.mean()) if m.any() else float("nan"),
                    "acc_abstain_wrong": float((ev.correct & m).mean())})
    res["coverage"] = pd.DataFrame(cov)

    # accuracy by detector volume decile
    vol = ev.copy()
    vol["n_on"] = vol.n_on.fillna(0)
    try:
        vol["vol_bin"] = pd.qcut(vol.n_on, 5, labels=False, duplicates="drop")
        res["acc_by_volume"] = vol.groupby("vol_bin").agg(
            n=("correct", "size"), acc=("correct", "mean"), med_n_on=("n_on", "median"))
    except ValueError:
        res["acc_by_volume"] = pd.DataFrame()
    return res


# ----------------------------------------------------------------- function
def evaluate_function(pred: pd.DataFrame, labels: pd.DataFrame,
                      folds: pd.DataFrame | None = None,
                      other_threshold: float = 0.0) -> dict:
    need = {"DeviceId", "Detector", "p_advance", "p_presence", "p_count"}
    missing = need - set(pred.columns)
    if missing:
        raise ValueError(f"function prediction file missing columns: {missing}")
    pred = pred.copy()
    pred["Detector"] = pred["Detector"].astype(int)
    ev = labels.merge(pred, on=["DeviceId", "Detector"], how="left")
    P = ev[["p_advance", "p_presence", "p_count"]].to_numpy(dtype=float)
    P = np.nan_to_num(P, nan=0.0)
    idx = P.argmax(axis=1)
    top3 = P.max(axis=1)
    pred3 = np.array(FUNCTIONS)[idx]
    ev["pred3"] = pred3
    ev["top_prob"] = top3
    has_pred = ev.p_advance.notna().to_numpy()

    if "p_other" in ev.columns:
        po = np.nan_to_num(ev["p_other"].to_numpy(dtype=float), nan=0.0)
        is_other = po > top3
    else:
        is_other = top3 < other_threshold
    ev["pred4"] = np.where(is_other | (~has_pred), "Other", pred3)
    ev["true4"] = np.where(ev.Function.isin(FUNCTIONS), ev.Function, "Other")

    if folds is not None:
        ev = ev.merge(folds, on="DeviceId", how="left")
    else:
        ev["fold"] = -1

    dets = load_active_detectors()
    ev = ev.merge(dets[["DeviceId", "Detector", "n_on"]], on=["DeviceId", "Detector"], how="left")
    ev["classifiable"] = ev.n_on.fillna(0) >= 1

    res: dict = {"eval": ev, "n": len(ev), "n_no_pred": int((~has_pred).sum()),
                 "n_classifiable": int(ev.classifiable.sum()),
                 "coverage_classifiable": float(ev.classifiable.mean())}
    m3 = ev.true4.isin(FUNCTIONS)
    ev3 = ev[m3]
    res["acc3"] = float((ev3.pred3 == ev3.Function).mean()) if len(ev3) else float("nan")
    res["n3"] = int(m3.sum())
    res["macro_f1_3"] = _macro_f1(ev3.Function, ev3.pred3, FUNCTIONS) if len(ev3) else float("nan")
    res["confusion3"] = pd.crosstab(ev3.Function, ev3.pred3) if len(ev3) else pd.DataFrame()
    f0 = ev3[ev3.fold == 0]
    res["acc3_fold0"] = float((f0.Function == f0.pred3).mean()) if len(f0) else float("nan")
    pf = ev3.assign(ok=ev3.Function == ev3.pred3)
    pf = pf[pf.fold >= 0].groupby("fold")["ok"].mean()
    res["fold_mean3"] = float(pf.mean()) if len(pf) else float("nan")
    res["fold_sd3"] = float(pf.std(ddof=0)) if len(pf) else float("nan")
    ev3c = ev3[ev3.classifiable]
    res["acc3_classifiable"] = float((ev3c.Function == ev3c.pred3).mean()) if len(ev3c) else float("nan")
    res["n3_classifiable"] = len(ev3c)
    # unscorable for function = zero actuations in the window (headline excludes them)
    res["acc3_scorable"] = res["acc3_classifiable"]
    res["n3_scorable"] = res["n3_classifiable"]
    res["n3_unscorable"] = int(len(ev3) - len(ev3c))
    pfs = ev3c.assign(ok=ev3c.Function == ev3c.pred3)
    pfs = pfs[pfs.fold >= 0].groupby("fold")["ok"].mean()
    res["fold_mean3_scorable"] = float(pfs.mean()) if len(pfs) else float("nan")
    res["fold_sd3_scorable"] = float(pfs.std(ddof=0)) if len(pfs) else float("nan")
    f0c = ev3c[ev3c.fold == 0]
    res["acc3_scorable_fold0"] = float((f0c.Function == f0c.pred3).mean()) if len(f0c) else float("nan")

    res["has_other_labels"] = bool((~m3).any())
    if res["has_other_labels"]:
        res["acc4"] = float((ev.pred4 == ev.true4).mean())
        res["macro_f1_4"] = _macro_f1(ev.true4, ev.pred4, list(FUNCTIONS) + ["Other"])
        res["confusion4"] = pd.crosstab(ev.true4, ev.pred4)
    res["other_threshold"] = other_threshold
    res["n_pred_other"] = int((ev.pred4 == "Other").sum())
    return res


def _macro_f1(y_true, y_pred, classes) -> float:
    f1s = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


# ------------------------------------------------------------------- report
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    d = df.copy()
    if not isinstance(d.index, pd.RangeIndex):
        d = d.reset_index()
    cols = list(d.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def report(phase_res: dict | None = None, func_res: dict | None = None,
           title: str = "Evaluation") -> str:
    L: list[str] = [f"## {title}", ""]
    if phase_res is not None:
        r = phase_res
        L += ["### Phase (per detector, top-1)", "",
              f"- labeled detectors: **{r['n']}**; **scorable {r['n_scorable']}**, "
              f"unscorable {r['n_unscorable']} "
              f"(no actuations {r['n_unscorable_no_actuations']}, "
              f"labeled phase never green {r['n_unscorable_label_not_green']})",
              f"- max |sum(prob)-1| per detector: {r['prob_sum_max_dev']:.2e}",
              "",
              "**HEADLINE (unscorable excluded)**", "",
              "| metric | value | n |", "|---|---|---|",
              f"| accuracy | **{r['acc_scorable']:.4f}** | {r['n_scorable']} |",
              f"| fold 0 (2025 hold-out) | **{r['acc_scorable_fold0']:.4f}** | {r['n_scorable_fold0']} |",
              f"| per-fold mean +/- sd | {r['fold_mean_scorable']:.4f} +/- {r['fold_sd_scorable']:.4f} | "
              f"{len(r['per_fold_scorable'])} folds |",
              f"| NON-standard detectors | **{r['acc_scorable_nonstd']:.4f}** | {r['n_scorable_nonstd']} |",
              "", "**Coverage vs accuracy (scorable only)**", "",
              _md_table(r["coverage_scorable"]), "",
              "**Footnote - old 'all labeled detectors' definition** (unscorable counted as wrong; "
              f"no prediction row: {r['n_no_pred']}, no events: {r['n_no_events']}, "
              f"true phase not a candidate: {r['n_label_not_candidate']})", "",
              "| metric | value | n |", "|---|---|---|",
              f"| overall accuracy | **{r['acc_overall']:.4f}** | {r['n']} |",
              f"| fold 0 (2025 hold-out) | **{r['acc_fold0']:.4f}** | {r['n_fold0']} |",
              f"| per-fold mean +/- sd | {r['fold_mean']:.4f} +/- {r['fold_sd']:.4f} | {len(r['per_fold'])} folds |",
              f"| standard-wired detectors | {r['acc_standard']:.4f} | {r['n_standard']} |",
              f"| NON-standard detectors | **{r['acc_nonstandard']:.4f}** | {r['n_nonstandard']} |",
              f"| channel > 40 | {r['acc_chan_gt40']:.4f} | {r['n_chan_gt40']} |",
              "",
              f"**Classifiable only** (>=1 actuation in the window; coverage "
              f"{r['coverage_classifiable']:.4f} = {r['n_classifiable']}/{r['n']})",
              "",
              "| metric | value | n |", "|---|---|---|",
              f"| accuracy, classifiable | **{r['acc_classifiable']:.4f}** | {r['n_classifiable']} |",
              f"| fold 0, classifiable | {r['acc_classifiable_fold0']:.4f} | |",
              f"| NON-standard, classifiable | {r['acc_classifiable_nonstd']:.4f} | {r['n_classifiable_nonstd']} |",
              "", "**Per fold**", "", _md_table(r["per_fold"]), "",
              "**Error breakdown**", "", _md_table(r["err_breakdown"].to_frame("n"), "{:.0f}"), ""]
        if len(r["err_pairs"]):
            L += ["**Concurrent-pair errors**", "", _md_table(r["err_pairs"].to_frame("n"), "{:.0f}"), ""]
        if len(r["err_pairs_other"]):
            L += ["**Other errors (top 10, true->pred)**", "",
                  _md_table(r["err_pairs_other"].to_frame("n"), "{:.0f}"), ""]
        L += ["**Coverage vs accuracy** (threshold on top prob)", "",
              _md_table(r["coverage"]), ""]
        if len(r["acc_by_volume"]):
            L += ["**Accuracy by detector volume quintile**", "",
                  _md_table(r["acc_by_volume"]), ""]
    if func_res is not None:
        f = func_res
        L += ["### Function (per detector)", "",
              f"| metric | value | n |", "|---|---|---|",
              f"| 3-class accuracy, SCORABLE (headline) | **{f['acc3_scorable']:.4f}** | "
              f"{f['n3_scorable']} (unscorable {f['n3_unscorable']}) |",
              f"| fold 0, scorable | {f['acc3_scorable_fold0']:.4f} | |",
              f"| per-fold mean +/- sd, scorable | {f['fold_mean3_scorable']:.4f} +/- "
              f"{f['fold_sd3_scorable']:.4f} | |",
              f"| 3-class macro-F1 (all labeled) | {f['macro_f1_3']:.4f} | |",
              f"| footnote: all labeled detectors | {f['acc3']:.4f} | {f['n3']} |",
              f"| footnote: fold 0, all labeled | {f['acc3_fold0']:.4f} | |",
              "", "**Confusion (rows = true, cols = predicted)**", "",
              _md_table(f["confusion3"], "{:.0f}"), ""]
        if f.get("has_other_labels"):
            L += [f"4-class (Other threshold {f['other_threshold']:.2f}): "
                  f"accuracy **{f['acc4']:.4f}**, macro-F1 {f['macro_f1_4']:.4f}", "",
                  _md_table(f["confusion4"], "{:.0f}"), ""]
        else:
            L += [f"(no Other labels in this split; predictions flagged Other: {f['n_pred_other']} "
                  f"at threshold {f['other_threshold']:.2f})", ""]
    return "\n".join(L)


def run(phase_path=None, function_path=None, folds_filter=None, split="dev",
        other_threshold=0.0, final_test=False, title="Evaluation") -> str:
    labels = load_labels(split)
    folds = load_folds()
    if split == "test":
        if not final_test:
            raise SystemExit("REFUSING: --split test requires --final-test")
        folds = pd.DataFrame({"DeviceId": labels.DeviceId.unique(), "fold": -1})
    if folds_filter:
        keep = set(folds[folds.fold.isin(folds_filter)].DeviceId)
        labels = labels[labels.DeviceId.isin(keep)]
    pr = fr = None
    if phase_path is not None:
        pred = pd.read_parquet(phase_path)
        _check_no_test(pred, final_test)
        pr = evaluate_phase(pred, labels, folds)
    if function_path is not None:
        predf = pd.read_parquet(function_path)
        _check_no_test(predf, final_test)
        fr = evaluate_function(predf, labels, folds, other_threshold)
    return report(pr, fr, title)


def selftest() -> None:
    labels = load_labels("dev")
    folds = load_folds()
    cand = load_candidates()
    dets = load_active_detectors()
    devs = folds.DeviceId.tolist()
    print(report(evaluate_phase(standard_lookup(devs, cand, dets), labels, folds),
                 evaluate_function(prior_function_predictor(labels), labels, folds, 0.0),
                 title="REFERENCE: ODOT standard-wiring lookup (phase) + class prior (function)"))
    print()
    print(report(evaluate_phase(uniform_predictor(devs, cand, dets), labels, folds),
                 title="SANITY: uniform over candidate phases"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", help="phase prediction parquet")
    ap.add_argument("--function", help="function prediction parquet")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="restrict scoring to these folds, e.g. --folds 0")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--other-threshold", type=float, default=0.0,
                    help="max 3-class prob below which the function prediction is Other")
    ap.add_argument("--final-test", action="store_true",
                    help="permit scoring TEST devices (orchestrator-authorised final run only)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--title", default="Evaluation")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.phase and not a.function:
        ap.error("give --phase and/or --function (or --selftest)")
    print(run(a.phase, a.function, a.folds, a.split, a.other_threshold, a.final_test, a.title))


if __name__ == "__main__":
    main()
