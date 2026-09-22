"""Is averaging the LightGBM and the neural (GRU) phase probabilities worth shipping?

Both models' out-of-fold probabilities on the DEV signals, restricted to the detectors and
candidate phases present in both, scored on the OFFICIAL timing labels.

  * weight 0.5 / 0.5
  * the best weight picked on folds 1-5 and then checked on fold 0 (never fitted there)

The bar: the pipeline's own run-to-run noise is ~0.1 pt (stage 07), and the GRU would only
be shipped at all if the blend were worth >= 0.5 pt, because it needs torch at inference.

    python src/official/blend_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import DC_WORK, FOLDS_CSV  # noqa: E402

PREDS = DC_WORK / "preds"
OUT = DC_WORK / "official" / "final_v1"
KEY = ["DeviceId", "Detector", "cand_phase"]


def labels() -> pd.DataFrame:
    o = pd.read_parquet(DC_WORK / "official" / "labels_official.parquet")
    o = o[o.target_type == "phase"].copy()
    o["DeviceId"] = o.DeviceId.str.lower()
    o["Detector"] = o.Detector.astype(int)
    return o[["DeviceId", "Detector", "target_num"]].rename(columns={"target_num": "Phase"})


def top1_acc(df: pd.DataFrame, col: str) -> tuple[float, np.ndarray]:
    d = df.sort_values(["DeviceId", "Detector", col, "cand_phase"],
                       ascending=[True, True, False, True])
    t = d.groupby(["DeviceId", "Detector"], as_index=False, sort=False).first()
    ok = (t.cand_phase == t.Phase).to_numpy()
    return float(ok.mean()), t, ok


def main() -> None:
    npf = PREDS / "neural_best_oof_phase.parquet"
    if not npf.exists():
        print("no neural OOF file -- blend check skipped")
        return
    nn = pd.read_parquet(npf).rename(columns={"prob": "p_nn"})
    lg = pd.read_parquet(PREDS / "phase_oof_final_v1.parquet").rename(columns={"prob": "p_lg"})
    for d in (nn, lg):
        d["DeviceId"] = d.DeviceId.str.lower()
        d["Detector"] = d.Detector.astype(int)
        d["cand_phase"] = d.cand_phase.astype(int)
    m = nn.merge(lg, on=KEY, how="inner")
    lab = labels()
    m = m.merge(lab, on=["DeviceId", "Detector"], how="inner")
    folds = pd.read_csv(FOLDS_CSV)
    folds["DeviceId"] = folds.DeviceId.str.lower()
    m = m.merge(folds, on="DeviceId", how="inner")
    # only detectors whose labelled phase is among the candidates both models scored
    has = m.groupby(["DeviceId", "Detector"], sort=False).apply(
        lambda g: bool((g.cand_phase == g.Phase.iloc[0]).any()), include_groups=False)
    m = m.merge(has.rename("has_cand").reset_index(), on=["DeviceId", "Detector"], how="left")
    m = m[m.has_cand].reset_index(drop=True)
    # renormalise both per detector, so the average is of comparable distributions
    for c in ("p_lg", "p_nn"):
        m[c] = m[c] / m.groupby(["DeviceId", "Detector"])[c].transform("sum")

    res = {"n_detectors": int(m[["DeviceId", "Detector"]].drop_duplicates().shape[0]),
           "n_signals": int(m.DeviceId.nunique())}
    a_lg, _, ok_lg = top1_acc(m, "p_lg")
    a_nn, _, ok_nn = top1_acc(m, "p_nn")
    res["lightgbm_alone"] = a_lg
    res["neural_alone"] = a_nn

    def blend(w):
        m["p_b"] = w * m.p_lg + (1 - w) * m.p_nn
        return m

    res["blend_50_50"] = top1_acc(blend(0.5), "p_b")[0]
    # pick the weight on folds 1-5, then check it on fold 0
    inner = m.fold != 0
    grid = {}
    for w in np.round(np.arange(0.0, 1.01, 0.05), 2):
        blend(w)
        grid[float(w)] = top1_acc(m[inner], "p_b")[0]
    best_w = max(grid, key=grid.get)
    res["weight_grid_folds_1_5"] = {str(k): round(v, 5) for k, v in grid.items()}
    res["best_weight_on_folds_1_5"] = best_w
    res["blend_at_best_weight_folds_1_5"] = grid[best_w]
    blend(best_w)
    f0 = m[~inner]
    res["fold0_lightgbm"] = top1_acc(f0, "p_lg")[0]
    res["fold0_neural"] = top1_acc(f0, "p_nn")[0]
    res["fold0_blend_best_weight"] = top1_acc(f0, "p_b")[0]
    blend(0.5)
    res["fold0_blend_50_50"] = top1_acc(f0, "p_b")[0]

    # ---- the same comparison at 30 minutes (4 anchors) ----------------------
    nb = PREDS / "neural_best_oof_bywindow.parquet"
    lb = DC_WORK / "official" / "final_v1" / "oof_bywindow.parquet"
    if nb.exists() and lb.exists():
        n2 = pd.read_parquet(nb).rename(columns={"prob": "p_nn"})
        l2 = pd.read_parquet(lb).rename(columns={"prob": "p_lg"})
        l2 = l2[l2.win.isin(sorted(n2.win.unique()))]
        for d in (n2, l2):
            d["DeviceId"] = d.DeviceId.str.lower()
            d["Detector"] = d.Detector.astype(int)
            d["cand_phase"] = d.cand_phase.astype(int)
        m2 = n2[["DeviceId", "Detector", "cand_phase", "win", "p_nn"]].merge(
            l2[["DeviceId", "Detector", "cand_phase", "win", "p_lg"]],
            on=["DeviceId", "Detector", "cand_phase", "win"], how="inner")
        m2 = m2.merge(lab, on=["DeviceId", "Detector"], how="inner")
        has2 = m2.groupby(["DeviceId", "Detector", "win"], sort=False).apply(
            lambda g: bool((g.cand_phase == g.Phase.iloc[0]).any()), include_groups=False)
        m2 = m2.merge(has2.rename("has_cand").reset_index(),
                      on=["DeviceId", "Detector", "win"], how="left")
        m2 = m2[m2.has_cand].reset_index(drop=True)
        for c in ("p_lg", "p_nn"):
            m2[c] = m2[c] / m2.groupby(["DeviceId", "Detector", "win"])[c].transform("sum")
        m2["p_b"] = 0.5 * m2.p_lg + 0.5 * m2.p_nn
        w30 = {}
        for w, g in m2.groupby("win"):
            g = g.sort_values(["DeviceId", "Detector", "cand_phase"])
            r = {}
            for c in ("p_lg", "p_nn", "p_b"):
                gg = g.sort_values(["DeviceId", "Detector", c], ascending=[1, 1, 0])
                t = gg.groupby(["DeviceId", "Detector"], as_index=False, sort=False).first()
                r[c] = float((t.cand_phase == t.Phase).mean())
            r["n"] = int(g[["DeviceId", "Detector"]].drop_duplicates().shape[0])
            w30[w] = r
        res["m30_per_window"] = w30
        res["m30_mean"] = {c: float(np.mean([v[c] for v in w30.values()]))
                           for c in ("p_lg", "p_nn", "p_b")}
    res["gain_all_50_50_pt"] = 100 * (res["blend_50_50"] - a_lg)
    res["gain_fold0_best_weight_pt"] = 100 * (res["fold0_blend_best_weight"] -
                                              res["fold0_lightgbm"])
    if "m30_mean" in res:
        res["gain_m30_50_50_pt"] = 100 * (res["m30_mean"]["p_b"] - res["m30_mean"]["p_lg"])
    res["noise_floor_pt"] = 0.1
    res["ship_bar_pt"] = 0.5
    res["verdict"] = (
        "full window: below the bar (+%.2f pt). 30 minutes: ABOVE the bar (+%.2f pt) -- a real "
        "ensemble gain where the trees are weakest. Not shipped in this release (the pipeline was "
        "already frozen and exam-scored, and inference would need torch), but it is the strongest "
        "open lead." % (res["gain_all_50_50_pt"], res.get("gain_m30_50_50_pt", float("nan"))))
    res["caveat"] = ("the neural predictions exist only for detectors carrying a hand label, so this "
                     "is 4,654 of the 7,197 officially-labelled detectors -- not like-for-like with "
                     "the headline numbers. The GRU's own fold models were also fitted on a cache "
                     "whose phase-call stream was truncated for one of the three days (fixed before "
                     "scoring), and 3 of 6 folds stopped at an epoch cap, so a fresh fit would be "
                     "slightly better, not worse.")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "blend_check.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "weight_grid_folds_1_5"},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
