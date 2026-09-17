"""Stage 04 step 6: v2 accuracy-vs-duration / vs-actuation curves and the review lists.

    python src/report_v2.py

Writes
    dc_work/preds/curves_v2.json          accuracy vs window length and vs actuation count
                                          for ranker v2, +joint decoding, +ODOT tie-breaker
    dc_work/preds/review_list_v2.csv      (a) confident model-vs-label disagreements
                                          (b) stubborn low-confidence detectors
                                          (c) detectors at signals with a suspected
                                              whole-signal label permutation
    dc_work/preds/tiebreak_v2.json        tuned tie-breaker gain / harm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, CONCURRENT_PAIRS, DEFAULT_PHASE, FOLDS_CSV,  # noqa: E402
                    LABELS_DEV, PREDS, connect)
from features import win_hours  # noqa: E402
from tiebreak import apply_odot_tiebreak, evaluate_tiebreak, tune  # noqa: E402

STAGES = {
    "stage01": PREDS / "phase_oof_bywindow.parquet",
    "ranker_v2": PREDS / "phase_oof_v2_bywindow.parquet",
    "decoded_v2": PREDS / "phase_oof_v2_decoded_bywindow.parquet",
}


def load_ctx():
    lab = pd.read_parquet(LABELS_DEV)
    folds = pd.read_csv(FOLDS_CSV)
    con = connect(threads=4)
    meta = con.sql(f"""SELECT DeviceId, Detector::INT AS Detector, n_on
                       FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}')""").df()
    con.close()
    health = pd.read_parquet(Path(str(CACHE).replace("cache", "atspm")) /
                             "detector_health.parquet",
                             columns=["DeviceId", "Detector", "health_flag", "health_reason"])
    health["Detector"] = health.Detector.astype(int)
    return lab, folds, meta, health


def top1_frame(bw: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    bw = bw.copy()
    bw["Detector"] = bw.Detector.astype(int)
    bw["cand_phase"] = bw.cand_phase.astype(int)
    d = bw.merge(lab, on=["DeviceId", "Detector"], how="inner")
    t = d.loc[d.groupby(["DeviceId", "Detector", "win"])["prob"].idxmax()].copy()
    t["ok"] = t.cand_phase == t.Phase
    return t


def curves(t: pd.DataFrame) -> dict:
    byw = t.groupby("win").ok.agg(["mean", "size"])
    byh: dict = {}
    for w, r in byw.iterrows():
        byh.setdefault(win_hours(w), []).append(float(r["mean"]))
    dur = {str(h): float(np.mean(v)) for h, v in sorted(byh.items())}
    tt = t.merge(pd.read_parquet(PREDS / "det_n_on_tmp.parquet"), how="left",
                 on=["DeviceId", "Detector", "win"]) if False else t
    return {"by_window": {w: [float(r["mean"]), int(r["size"])] for w, r in byw.iterrows()},
            "by_duration": dur}


SHORT = ["m30_a", "m30_b", "m30_c", "m30_d", "h1_a", "h1_b", "h1_c"]
MARGINS = (0.05, 0.1, 0.25, 0.4, 0.5)
STDS = (0.7, 0.8, 0.85, 0.9, 1.0)


def tune_windows(bw: pd.DataFrame, lab: pd.DataFrame, folds: pd.DataFrame,
                 wins: list[str] | None = None) -> pd.DataFrame:
    """Nested CV for the tie-breaker over the SHORT windows, where close concurrent ties
    actually occur (at 72 h the decoder leaves almost none)."""
    wins = SHORT if wins is None else wins
    sub = {w: g.drop(columns=["win"]) for w, g in bw[bw.win.isin(wins)].groupby("win")}
    rows = []
    for k in sorted(folds.fold.unique()):
        inn = set(folds.loc[folds.fold != k, "DeviceId"])
        outd = set(folds.loc[folds.fold == k, "DeviceId"])
        best, bg = (0.25, 0.85), -10 ** 9
        for m in MARGINS:
            for s in STDS:
                tot = sum(evaluate_tiebreak(g[g.DeviceId.isin(inn)], lab, margin=m,
                                            standardness_min=s)["delta_n"]
                          for g in sub.values())
                if tot > bg:
                    bg, best = tot, (m, s)
        agg = {"n_switched": 0, "helped": 0, "hurt": 0, "hurt_nonstandard": 0}
        for g in sub.values():
            r = evaluate_tiebreak(g[g.DeviceId.isin(outd)], lab, margin=best[0],
                                  standardness_min=best[1])
            for key in agg:
                agg[key] += r[key]
        agg.update(fold=int(k), chosen_margin=best[0], chosen_std=best[1],
                   delta_n=agg["helped"] - agg["hurt"])
        rows.append(agg)
    return pd.DataFrame(rows)


def main() -> None:
    lab, folds, meta, health = load_ctx()
    nact = pd.read_parquet(Path(str(CACHE).replace("cache", "features")) /
                           "pair_features_windows.parquet",
                           columns=["DeviceId", "Detector", "win", "det_n_on"]).drop_duplicates(
        ["DeviceId", "Detector", "win"])
    nact["Detector"] = nact.Detector.astype(int)

    out = {}
    tops = {}
    for name, path in STAGES.items():
        if not Path(path).exists():
            continue
        t = top1_frame(pd.read_parquet(path), lab)
        tops[name] = t
        c = curves(t)
        tn = t.merge(nact, on=["DeviceId", "Detector", "win"], how="left")
        bins = [0, 10, 25, 50, 100, 250, 500, 1000, 5000, 10**9]
        tn["nbin"] = pd.cut(tn.det_n_on, bins, right=False)
        c["by_actuations"] = {str(k): [int(v["size"]), float(v["mean"])] for k, v in
                              tn.groupby("nbin", observed=True).ok.agg(["mean", "size"]).iterrows()}
        out[name] = c

    # ---- ODOT tie-breaker, tuned with nested CV on DEV OOF ---------------------
    best = "decoded_v2" if "decoded_v2" in tops else "ranker_v2"
    bw = pd.read_parquet(STAGES[best])
    bw["Detector"] = bw.Detector.astype(int)
    tb = tune_windows(bw, lab, folds)
    per_win = []
    for w, g in bw.groupby("win"):
        for m in (0.1, 0.25, 0.5):
            r = evaluate_tiebreak(g.drop(columns=["win"]), lab, margin=m,
                                  standardness_min=0.85)
            r["win"] = w
            per_win.append(r)
    json.dump({"nested_cv_short_windows": tb.to_dict("records"), "per_window": per_win},
              open(PREDS / "tiebreak_v2.json", "w"), indent=1, default=str)
    print(tb.to_string(index=False))

    mg = float(pd.Series([r["chosen_margin"] for r in tb.to_dict("records")]).median())
    st = float(pd.Series([r["chosen_std"] for r in tb.to_dict("records")]).median())
    parts = []
    for w, g in bw.groupby("win"):
        parts.append(apply_odot_tiebreak(g.drop(columns=["win"]), enabled=True,
                                         margin=mg, standardness_min=st).assign(win=w))
    tbw = top1_frame(pd.concat(parts, ignore_index=True), lab)
    out["decoded_v2_tiebreak"] = curves(tbw)
    out["tiebreak_setting"] = {"margin": mg, "standardness_min": st}
    json.dump(out, open(PREDS / "curves_v2.json", "w"), indent=1, default=str)
    print(json.dumps({k: v.get("by_duration") for k, v in out.items()
                      if isinstance(v, dict) and "by_duration" in v}, indent=1))

    # ---- review list ----------------------------------------------------------
    t = tops[best]
    t72 = t[t.win == "full72"].copy()
    p72 = pd.read_parquet(STAGES[best].as_posix().replace("_bywindow", ""))
    p72["Detector"] = p72.Detector.astype(int)
    p72 = p72.sort_values(["DeviceId", "Detector", "prob"], ascending=[True, True, False])
    g = p72.groupby(["DeviceId", "Detector"], sort=False)
    a = g.head(1).rename(columns={"cand_phase": "pred1", "prob": "p1"})
    b = g.nth(1).rename(columns={"cand_phase": "pred2", "prob": "p2"})
    r = lab.merge(a[["DeviceId", "Detector", "pred1", "p1"]], on=["DeviceId", "Detector"],
                  how="left")
    r = r.merge(b[["DeviceId", "Detector", "pred2", "p2"]], on=["DeviceId", "Detector"],
                how="left")
    r = r.merge(meta, on=["DeviceId", "Detector"], how="left")
    r = r.merge(health, on=["DeviceId", "Detector"], how="left")
    r = r.merge(folds, on="DeviceId", how="left")
    r["std_phase"] = r.Detector.map(DEFAULT_PHASE)
    r["correct"] = r.pred1 == r.Phase
    r["n_on"] = r.n_on.fillna(0)

    # (c) signals with a suspected whole-signal permutation of the labels
    ren = []
    for dev, gg in r[(r.n_on > 0) & r.pred1.notna()].groupby("DeviceId"):
        gc = gg[gg.p1 >= 0.7]
        if len(gc) < 4 or (~gc.correct).sum() < 2:
            continue
        conf = pd.crosstab(gc.pred1.astype(int), gc.Phase.astype(int))
        mp = {int(p): int(conf.loc[p].idxmax()) for p in conf.index}
        hits = sum(conf.loc[p, q] for p, q in mp.items() if q in conf.columns)
        ident = sum(conf.loc[p, p] for p in conf.index if p in conf.columns)
        if hits - ident >= 2 and len(set(mp.values())) == len(mp):
            ren.append({"DeviceId": dev, "perm": mp, "gain": int(hits - ident),
                        "n_conf": int(len(gc))})
    ren_dev = {d["DeviceId"]: d for d in ren}

    rows = []
    for x in r.itertuples():
        reasons = []
        if (not x.correct) and pd.notna(x.p1) and x.p1 >= 0.8 and x.n_on > 0:
            reasons.append("confident disagreement with the label - check the wiring/config")
        if pd.isna(x.p1) or x.n_on == 0:
            reasons.append("no actuations - cannot classify")
        elif x.p1 < 0.5:
            reasons.append("stubborn low-confidence detector")
        if x.DeviceId in ren_dev:
            reasons.append(f"signal may be renumbered {ren_dev[x.DeviceId]['perm']}")
        if not reasons:
            continue
        rows.append(dict(DeviceId=x.DeviceId, Detector=x.Detector, label=x.Phase,
                         label_function=x.Function, pred1=x.pred1, p1=x.p1,
                         pred2=x.pred2, p2=x.p2, std_wiring_phase=x.std_phase,
                         health_flag=x.health_flag, health_reason=x.health_reason,
                         n_actuations=int(x.n_on), fold=x.fold,
                         reason="; ".join(reasons)))
    rl = pd.DataFrame(rows).sort_values(["reason", "p1"], ascending=[True, False])
    rl.to_csv(PREDS / "review_list_v2.csv", index=False)
    print(f"review_list_v2.csv: {len(rl)} rows "
          f"({rl.reason.str.contains('confident').sum()} confident disagreements, "
          f"{rl.reason.str.contains('stubborn').sum()} stubborn, "
          f"{len(ren)} suspected-renumbering signals)")
    json.dump(ren, open(PREDS / "renumber_signals_v2.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
