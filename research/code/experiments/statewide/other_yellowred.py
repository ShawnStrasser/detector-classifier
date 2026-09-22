"""Stage 05 tasks 3 and 4.

Task 3 -- "Other".  The DEV-trained 3-class function model has no Other class; the protocol
calls an output Other when max(p_advance,p_presence,p_count) < threshold.  The statewide
labels are the only place where Other actually exists, so this is where the rule can be
measured.  Two approaches are compared on exactly the same detectors:
  (A) DEV-trained 3-class model + max-probability threshold;
  (B) a 4-class model with an explicitly trained Other class, grouped 5-fold CV by signal
      over the statewide labels (TEST signals excluded throughout).

Task 4 -- Yellow_Red feasibility.  285 labels at 60 signals, statewide only.  One-vs-rest
LightGBM with grouped CV by signal: ROC-AUC, average precision, precision/recall.

    python src/statewide/other_yellowred.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path

from common import FUNCTIONS  # noqa: E402
from common6 import SW_FEAT, SW_PREDS, statewide_labels  # noqa: E402

FUNC4 = ["Advance", "Presence", "Count", "Other"]
NFOLD = 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_frame() -> pd.DataFrame:
    fr = pd.read_parquet(SW_FEAT / "funcframe6_sw_h6.parquet")
    lab = statewide_labels()
    fr["Detector"] = fr.Detector.astype(int)
    lab["Detector"] = lab.Detector.astype(int)
    fr = fr.merge(lab[["DeviceId", "Detector", "func_std", "Function"]],
                  on=["DeviceId", "Detector"], how="inner")
    return fr.reset_index(drop=True)


def feat_cols(fr: pd.DataFrame) -> list[str]:
    skip = {"DeviceId", "Detector", "group", "fold", "health_flag", "func_std", "Function",
            "y", "y4", "y_yr"}
    return [c for c in fr.columns if c not in skip and pd.api.types.is_numeric_dtype(fr[c])]


def grouped_oof(fr: pd.DataFrame, y: np.ndarray, cols: list[str], params: dict,
                n_class: int) -> np.ndarray:
    gk = GroupKFold(n_splits=NFOLD)
    out = np.zeros((len(fr), n_class)) if n_class > 1 else np.zeros(len(fr))
    for tr, te in gk.split(fr, y, groups=fr.DeviceId):
        P = dict(params)
        m = lgb.LGBMClassifier(n_estimators=P.pop("n_estimators"), **P)
        m.fit(fr.iloc[tr][cols], y[tr])
        p = m.predict_proba(fr.iloc[te][cols])
        out[te] = p if n_class > 1 else p[:, 1]
    return out


MC_PARAMS = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=31,
                 min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
                 bagging_freq=1, lambda_l2=1.0, n_estimators=400, n_jobs=12, verbose=-1)
BIN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=500, n_jobs=12, verbose=-1,
                  is_unbalance=True)


# ------------------------------------------------------------------- task 3
def task3(fr: pd.DataFrame) -> dict:
    pr = pd.read_parquet(SW_PREDS / "function6_statewide.parquet")
    pr = pr[pr.win == "h6_sw"][["DeviceId", "Detector", "p_advance", "p_presence",
                                "p_count"]]
    pr["Detector"] = pr.Detector.astype(int)
    d = fr.merge(pr, on=["DeviceId", "Detector"], how="left")
    # protocol: anything outside the three trained classes is true class Other
    d["true4"] = np.where(d.func_std.isin(FUNCTIONS), d.func_std, "Other")
    P = d[["p_advance", "p_presence", "p_count"]].to_numpy(float)
    ok = ~np.isnan(P).any(1)
    top = np.where(ok, np.nan_to_num(P).max(1), 0.0)
    pred3 = np.array(FUNCTIONS)[np.nan_to_num(P).argmax(1)]
    res: dict = {"n": int(len(d)), "n_scored": int(ok.sum()),
                 "n_true_other": int((d.true4 == "Other").sum()),
                 "class_counts": d.true4.value_counts().to_dict()}
    curve = []
    for th in (0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        p4 = np.where((top < th) | (~ok), "Other", pred3)
        acc = float((p4 == d.true4.to_numpy()).mean())
        m3 = d.true4.isin(FUNCTIONS).to_numpy()
        curve.append({
            "threshold": th, "acc4": acc,
            "acc4_unseen": float((p4[(d.group == "unseen").to_numpy()] ==
                                  d.true4[d.group == "unseen"].to_numpy()).mean()),
            "recall_other": float((p4[~m3] == "Other").mean()),
            "precision_other": float((d.true4.to_numpy()[p4 == "Other"] == "Other").mean())
            if (p4 == "Other").any() else None,
            "acc_on_3class": float((p4[m3] == d.true4.to_numpy()[m3]).mean()),
            "macro_f1": _macro_f1(d.true4.to_numpy(), p4, FUNC4)})
    res["threshold_curve"] = curve
    best = max(curve, key=lambda r: r["macro_f1"])
    res["recommended_threshold"] = best["threshold"]
    # per raw label string, at the recommended threshold
    th = best["threshold"]
    p4 = np.where((top < th) | (~ok), "Other", pred3)
    d = d.assign(pred4=p4, topp=top)
    oth = d[d.true4 == "Other"]
    by = oth.groupby("Function").agg(
        n=("Function", "size"),
        sent_to_other=("pred4", lambda s: float((s == "Other").mean())),
        confident_wrong=("topp", lambda s: float((s >= 0.8).mean())))
    by["top_wrong_class"] = oth[oth.pred4 != "Other"].groupby("Function")["pred4"].agg(
        lambda s: s.value_counts().index[0] if len(s) else "")
    res["other_by_raw_label"] = by.sort_values("n", ascending=False).reset_index().to_dict("records")
    res["confusion4_at_best"] = pd.crosstab(d.true4, d.pred4).to_dict()

    # ---- (B) explicitly trained Other class, grouped CV on statewide labels
    d4 = d[d.true4.notna()].reset_index(drop=True)
    cols = feat_cols(d4)
    y = d4.true4.map({c: i for i, c in enumerate(FUNC4)}).to_numpy()
    Q = grouped_oof(d4, y, cols, MC_PARAMS, 4)
    pred = np.array(FUNC4)[Q.argmax(1)]
    res["trained_other"] = {
        "n": int(len(d4)), "n_features": len(cols),
        "acc4": float((pred == d4.true4.to_numpy()).mean()),
        "acc4_unseen_signals": float((pred[(d4.group == "unseen").to_numpy()] ==
                                      d4.true4[d4.group == "unseen"].to_numpy()).mean()),
        "macro_f1": _macro_f1(d4.true4.to_numpy(), pred, FUNC4),
        "recall_other": float((pred[d4.true4 == "Other"] == "Other").mean()),
        "precision_other": float((d4.true4.to_numpy()[pred == "Other"] == "Other").mean()),
        "acc_on_3class": float((pred[d4.true4.isin(FUNCTIONS).to_numpy()] ==
                                d4.true4.to_numpy()[d4.true4.isin(FUNCTIONS).to_numpy()]).mean()),
        "confusion": pd.crosstab(d4.true4, pred).to_dict(),
        "other_recall_by_raw_label": d4[d4.true4 == "Other"].assign(pr=pred[(d4.true4 == "Other").to_numpy()])
            .groupby("Function").agg(n=("pr", "size"),
                                     sent_to_other=("pr", lambda s: float((s == "Other").mean())))
            .sort_values("n", ascending=False).reset_index().to_dict("records")}
    return res


def _macro_f1(yt, yp, classes) -> float:
    f1 = []
    yt, yp = np.asarray(yt), np.asarray(yp)
    for c in classes:
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1)) if f1 else float("nan")


# ------------------------------------------------------------------- task 4
def task4(fr: pd.DataFrame) -> dict:
    d = fr.reset_index(drop=True)
    y = (d.func_std == "Yellow_Red").astype(int).to_numpy()
    cols = feat_cols(d)
    p = grouped_oof(d, y, cols, BIN_PARAMS, 1)
    res = {"n": int(len(d)), "n_pos": int(y.sum()),
           "n_pos_signals": int(d[y == 1].DeviceId.nunique()),
           "auc": float(roc_auc_score(y, p)),
           "ap": float(average_precision_score(y, p)),
           "base_rate": float(y.mean())}
    pts = []
    for th in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        m = p >= th
        pts.append({"th": th, "n_pred": int(m.sum()),
                    "precision": float(y[m].mean()) if m.any() else None,
                    "recall": float(m[y == 1].mean())})
    # also precision at the recall-oriented operating point used by the user (top-k)
    for k in (100, 285, 500):
        idx = np.argsort(-p)[:k]
        pts.append({"top_k": k, "precision": float(y[idx].mean()),
                    "recall": float(y[idx].sum() / max(y.sum(), 1))})
    res["operating_points"] = pts
    # what distinguishes them: fit one model on everything for importances, and compare
    # the medians of the strongest features
    P = dict(BIN_PARAMS)
    m = lgb.LGBMClassifier(n_estimators=P.pop("n_estimators"), **P)
    m.fit(d[cols], y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(
        ascending=False)
    res["top_features"] = imp.head(15).round(1).to_dict()
    comp = []
    for f in imp.head(10).index:
        comp.append({"feature": f,
                     "median_yellow_red": float(np.nanmedian(d.loc[y == 1, f])),
                     "median_rest": float(np.nanmedian(d.loc[y == 0, f]))})
    res["feature_contrast"] = comp
    # how are Yellow_Red detectors classified by the 3-class function head?
    pr = pd.read_parquet(SW_PREDS / "function6_statewide.parquet")
    pr = pr[pr.win == "h6_sw"]
    pr["Detector"] = pr.Detector.astype(int)
    j = d.merge(pr[["DeviceId", "Detector", "p_advance", "p_presence", "p_count"]],
                on=["DeviceId", "Detector"], how="left")
    Pm = j[["p_advance", "p_presence", "p_count"]].to_numpy(float)
    lab3 = np.array(FUNCTIONS)[np.nan_to_num(Pm).argmax(1)]
    res["yr_seen_as"] = pd.Series(lab3[(j.func_std == "Yellow_Red").to_numpy()]
                                  ).value_counts().to_dict()
    res["yr_mean_topprob"] = float(np.nanmean(np.nan_to_num(Pm).max(1)[
        (j.func_std == "Yellow_Red").to_numpy()]))
    return res


def main() -> None:
    fr = load_frame()
    log(f"statewide function frame {fr.shape}; "
        f"{fr.DeviceId.nunique()} signals, classes {fr.func_std.value_counts().to_dict()}")
    out = {"task3_other": task3(fr), "task4_yellow_red": task4(fr)}
    json.dump(out, open(SW_PREDS / "other_yellowred_results.json", "w"), indent=1,
              default=str)
    log(json.dumps(out["task3_other"]["threshold_curve"], indent=1))
    log(json.dumps(out["task3_other"]["trained_other"], indent=1, default=str))
    log(json.dumps({k: v for k, v in out["task4_yellow_red"].items()
                    if k != "top_features"}, indent=1, default=str))


if __name__ == "__main__":
    main()
