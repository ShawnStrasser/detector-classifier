"""Stage 06 -- detector **function** v3: Advance / Presence / Count / Yellow_Red / Other,
trained on the Dec-2024 all-event-code data with the *rich* Feb-2025 labels joined on.

Key insight of this stage: `data/statewide_2025-02-25/all_configs.csv` and
`data/raw/detector-configs.csv` describe the SAME ODOT signals (same DeviceId GUIDs).  The
Dec-2024 file simply contains only the Advance / Presence / Count rows; the Feb-2025 file
also carries ~1,300 "Other"-type and ~285 Yellow_Red channels.  Joining them by
(DeviceId, channel) gives Other and Yellow_Red labels backed by **72 h of data with the
yellow / red-clearance events 8-11**, which stage 05 could not use (6 codes, 6 h).

    python src/function_v3.py --stage labels    # merged label table + folds + disagreements
    python src/function_v3.py --stage phase     # phase OOF over the 22-window mix
    python src/function_v3.py --stage frame     # function design matrix
    python src/function_v3.py --stage models    # 3 / 4 / 5-class, Yellow_Red, ablations
    python src/function_v3.py --stage ship      # models/beta_v1_candidate + review list

Nothing from earlier stages is overwritten: everything lands in `dc_work/function_v3/`.
The 43 TEST signals are removed before anything is read.
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

from common import (CACHE, DC_WORK, FEATURES, FOLDS_CSV, LABELS_DEV,  # noqa: E402
                    N_FOLDS, RAW, REPO)
# the function feature builders themselves ship with the model
import function as _fn  # noqa: E402
from function import (SIB_FEATS, add_shape_features,  # noqa: E402,F401
                      add_sibling_features)

WORK = DC_WORK / "function_v3"
WORK.mkdir(parents=True, exist_ok=True)

STATEWIDE_CFG = REPO / "data" / "statewide_2025-02-25" / "all_configs.csv"
TEST_CFG = REPO / "data" / "splits" / "test_config.csv"
DEC_CFG = RAW / "detector-configs.csv"

LABELS_V3 = WORK / "labels_v3.parquet"
FOLDS_V3 = WORK / "folds_v3.csv"
LABEL_DISAGREE = WORK / "label_disagreements.csv"

CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CLASSES4 = ["Advance", "Presence", "Count", "Other"]
CLASSES3 = ["Advance", "Presence", "Count"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ===================================================================== labels
def stage_labels(args) -> None:
    from make_label_map import load_map
    t0 = time.time()
    test = set(pd.read_csv(TEST_CFG).DeviceId.unique())
    have_events = {p.name.split("=", 1)[1] for p in (CACHE / "events").iterdir()
                   if p.is_dir()}

    sw = pd.read_csv(STATEWIDE_CFG).rename(columns={"Parameter": "Detector"})
    sw["Function"] = sw.Function.astype(str)
    sw["func_std"] = sw.Function.map(load_map())
    dec = pd.read_csv(DEC_CFG)
    for d in (sw, dec):
        d["Detector"] = d.Detector.astype(int)
        d["Phase"] = d.Phase.astype(int)
    # TEST never enters anything
    sw = sw[~sw.DeviceId.isin(test)].reset_index(drop=True)
    dec = dec[~dec.DeviceId.isin(test)].reset_index(drop=True)

    j = dec.merge(sw, on=["DeviceId", "Detector"], how="outer",
                  suffixes=("_dec", "_feb"), indicator=True)
    both = j[j._merge == "both"]
    cons = {
        "n_dec_rows_nontest": int(len(dec)), "n_feb_rows_nontest": int(len(sw)),
        "n_in_both": int(len(both)),
        "n_dec_only": int((j._merge == "left_only").sum()),
        "n_feb_only": int((j._merge == "right_only").sum()),
        "phase_agree": float((both.Phase_dec == both.Phase_feb).mean()),
        "function_agree": float((both.Function_dec == both.func_std).mean()),
        "n_phase_disagree": int((both.Phase_dec != both.Phase_feb).sum()),
        "n_phase_disagree_signals": int(both[both.Phase_dec != both.Phase_feb]
                                        .DeviceId.nunique()),
        "n_function_disagree": int((both.Function_dec != both.func_std).sum()),
        "function_confusion_dec_vs_feb":
            pd.crosstab(both.Function_dec, both.func_std).to_dict(),
    }
    log("Dec-2024 vs Feb-2025 label consistency: " +
        json.dumps({k: v for k, v in cons.items() if "confusion" not in k}, indent=1))

    dis = both[(both.Phase_dec != both.Phase_feb) |
               (both.Function_dec != both.func_std)].copy()
    dis["issue"] = np.where(dis.Phase_dec != both.Phase_feb.loc[dis.index],
                            np.where(dis.Function_dec != dis.func_std, "phase+function",
                                     "phase"), "function")
    dis[["DeviceId", "Detector", "Phase_dec", "Phase_feb", "Function_dec",
         "Function_feb", "func_std", "issue"]].to_csv(LABEL_DISAGREE, index=False)
    log(f"{len(dis)} Dec-vs-Feb label disagreements -> {LABEL_DISAGREE}")

    # ---- merged label table: Dec wins where it exists (it is contemporaneous with the
    # events we train on); Feb only ADDS channels the Dec file never listed.
    dec_lab = dec.assign(func5=dec.Function, src="dec")[
        ["DeviceId", "Detector", "Phase", "func5", "Function", "src"]]
    key = set(zip(dec.DeviceId, dec.Detector))
    add = sw[~pd.Series(list(zip(sw.DeviceId, sw.Detector)), index=sw.index).isin(key)]
    add = add.assign(func5=add.func_std, src="feb")[
        ["DeviceId", "Detector", "Phase", "func5", "Function", "src"]]
    lab = pd.concat([dec_lab, add], ignore_index=True)
    lab = lab[lab.DeviceId.isin(have_events)].reset_index(drop=True)
    lab = lab[lab.Detector.between(1, 64)].reset_index(drop=True)

    # ---- folds: keep the existing DEV folds exactly; signals that have Dec events and
    # Feb labels but no DEV fold get one (round robin over folds 1-5, seed 0).
    folds = pd.read_csv(FOLDS_CSV)
    known = set(folds.DeviceId)
    extra = sorted(set(lab.DeviceId) - known)
    rng = np.random.default_rng(0)
    newf = pd.DataFrame({"DeviceId": extra,
                         "fold": rng.integers(1, N_FOLDS, size=len(extra))})
    allf = pd.concat([folds, newf], ignore_index=True).sort_values("DeviceId")
    allf.to_csv(FOLDS_V3, index=False)
    lab = lab.merge(allf, on="DeviceId", how="left")
    lab.to_parquet(LABELS_V3, index=False)

    summ = {
        "consistency": cons,
        "n_labels": int(len(lab)), "n_signals": int(lab.DeviceId.nunique()),
        "class_counts": lab.func5.value_counts().to_dict(),
        "class_counts_by_source":
            lab.groupby(["src", "func5"]).size().unstack(fill_value=0).to_dict(),
        "n_signals_with_yellow_red": int(lab[lab.func5 == "Yellow_Red"].DeviceId.nunique()),
        "n_new_fold_signals": len(extra),
        "n_labels_dec_only_source": int((lab.src == "dec").sum()),
    }
    json.dump(summ, open(WORK / "labels_v3.json", "w"), indent=1, default=str)
    log(json.dumps({k: v for k, v in summ.items() if k != "consistency"}, indent=1,
                   default=str))
    log(f"labels done in {time.time()-t0:.0f}s -> {LABELS_V3}")


# ============================================================ feature loading
BASE_FILES = [FEATURES / "pair_features_windows.parquet",
              FEATURES / "pair_features_windows_B.parquet"]
V2_FILES = [FEATURES / "pair_features_v2_extra.parquet",
            FEATURES / "pair_features_v2_extra_B.parquet"]
SIM_FILES = [FEATURES / "det_similarity.parquet",
             FEATURES / "det_similarity_B.parquet"]
YR_FILES = [FEATURES / "func_yr_extra.parquet",
            FEATURES / "func_yr_extra_B.parquet"]
LAG_FILES = [FEATURES / "det_lag.parquet", FEATURES / "det_lag_B.parquet"]

PAIR_KEY = ["DeviceId", "Detector", "cand_phase", "win"]


def _cat(files, **kw) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f, **kw) for f in files], ignore_index=True)


def load_pairs_all(with_yr: bool = True) -> pd.DataFrame:
    """Stage-01 + v2 + v3 pair features over all 22 windows, DEV-ish signals only."""
    from features_partner import PDIFF_FEATS, add_partner_diffs
    df = _cat(BASE_FILES)
    v2 = _cat(V2_FILES)
    df = df.merge(v2, on=PAIR_KEY, how="left")
    del v2
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    if with_yr:
        yr = _cat(YR_FILES)
        df = df.merge(yr, on=PAIR_KEY, how="left")
        del yr
    folds = pd.read_csv(FOLDS_V3)
    df = df.merge(folds, on="DeviceId", how="inner")
    return df


# ====================================================================== phase
PHASE_OOF = WORK / "phase_oof_v3_bywindow.parquet"


def stage_phase(args) -> None:
    """Re-run the stage-04 phase pipeline (ranker -> joint decoder) over the full 22-window
    mix so that every window, including 5 and 10 min, has an out-of-fold predicted phase.

    The phase model is *unchanged*: same features, same folds, same labels
    (`labels_dev.parquet`).  The 6 extra signals have no Dec-2024 phase label, so they are
    never trained on -- they only receive predictions."""
    import decode_train as dc
    import lightgbm as lgb
    from train_lgbm_v2 import (KEY_EXCLUDE, RANK_PARAMS, oof_predict, summarise,
                               to_prob)
    t0 = time.time()
    df = load_pairs_all(with_yr=False)
    df = df.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    from train_lgbm_v2 import load_health
    df = df.merge(load_health(), on=["DeviceId", "Detector"], how="left")
    df["health_flag"] = df.health_flag.fillna("unknown")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    fc = [c for c in df.columns
          if c not in KEY_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    log(f"phase pairs={len(df):,} features={len(fc)} windows={df.win.nunique()}")

    cache = WORK / "phase_stage1_v3.parquet"
    if cache.exists():
        log(f"reusing first-stage cache {cache}")
        c = pd.read_parquet(cache)
        pr = df[["DeviceId", "Detector", "win", "cand_phase"]].merge(
            c, on=["DeviceId", "Detector", "win", "cand_phase"], how="left")
        p0 = pr.p0.to_numpy()
        r1 = json.load(open(WORK / "phase_v3_stage1.json"))
    else:
        s, _ = oof_predict(df, fc, RANK_PARAMS, train_mask=df.Phase.notna())
        p0 = to_prob(df, s)
        lab = df.Phase.notna().to_numpy()
        r1 = summarise(df[lab].reset_index(drop=True), p0[lab], "ranker_v3_22win")
        log(json.dumps(r1))
        pr = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
        pr["p0"] = p0
        pr.to_parquet(cache, index=False)
        json.dump(r1, open(WORK / "phase_v3_stage1.json", "w"), indent=1, default=str)
    ctx = df[["DeviceId", "Detector", "win", "cand_phase", "cand_green_share",
              "call43_per_cycle", "det_n_on", "log_win_hours"]]
    sim = _cat(SIM_FILES)
    X = dc.assemble(pr, pairs=ctx, sim=sim)
    del sim, pr, ctx
    X = X.merge(pd.read_csv(FOLDS_V3), on="DeviceId", how="inner")
    X = X.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    X["y"] = (X.cand_phase == X.Phase).astype(np.int8)
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    meta = json.load(open(REPO / "models" / "beta_v0" / "decode_lgbm_v2.json"))
    cols, mode = meta["features"], meta.get("mode", "binary")
    log(f"decoder ({meta['champion']}, {mode}) frame {X.shape}")
    s2 = dc.oof_second_stage(X, cols, mode=mode)
    d = pd.DataFrame({"g": X["win"].astype(str) + "|" + X.DeviceId + "|" +
                      X.Detector.astype(str), "s": np.clip(s2, 1e-9, None)})
    p2 = (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()
    lab2 = X.Phase.notna().to_numpy()
    r2 = summarise(X[lab2].reset_index(drop=True), p2[lab2], "decoded_v3_22win")
    log(json.dumps(r2))
    out = X[["DeviceId", "Detector", "win", "cand_phase"]].copy()
    out["prob"] = p2
    out["Detector"] = out.Detector.astype(np.int16)
    out["cand_phase"] = out.cand_phase.astype(np.int16)
    out.to_parquet(PHASE_OOF, index=False)
    json.dump({"ranker": r1, "decoded": r2}, open(WORK / "phase_v3_results.json", "w"),
              indent=1, default=str)
    log(f"phase done in {time.time()-t0:.0f}s -> {PHASE_OOF}")


# ====================================================================== frame
FRAME = WORK / "funcframe_v3.parquet"

YR_PREFIX = "yr_"
LAG_PREFIX = ("lagsib_", "lagany_", "lagbest_")


def add_lag_features(top: pd.DataFrame) -> pd.DataFrame:
    """`function.add_lag_features` on the cached `det_lag` tables."""
    return _fn.add_lag_features(top, _cat(LAG_FILES))


def stage_frame(args) -> None:
    t0 = time.time()
    from train_lgbm_v2 import load_health
    pairs = load_pairs_all(with_yr=True)
    probs = pd.read_parquet(PHASE_OOF)
    probs["Detector"] = probs.Detector.astype(pairs.Detector.dtype)
    probs["cand_phase"] = probs.cand_phase.astype(pairs.cand_phase.dtype)
    p = pairs.merge(probs, on=PAIR_KEY, how="inner")
    del pairs, probs
    i = p.groupby(["DeviceId", "Detector", "win"], sort=False)["prob"].idxmax()
    top = p.loc[i].copy()
    del p
    top = top.rename(columns={"cand_phase": "pred_phase", "prob": "top_prob"})
    top = add_shape_features(top)
    top = add_sibling_features(top)
    top = add_lag_features(top)
    top = top.reset_index(drop=True)
    lab = pd.read_parquet(LABELS_V3)
    lab["Detector"] = lab.Detector.astype(top.Detector.dtype)
    top = top.merge(lab[["DeviceId", "Detector", "func5", "Function", "src",
                         "Phase"]].rename(columns={"Phase": "label_phase"}),
                    on=["DeviceId", "Detector"], how="left")
    h = load_health()
    h["Detector"] = h.Detector.astype(top.Detector.dtype)
    top = top.merge(h, on=["DeviceId", "Detector"], how="left")
    top["health_flag"] = top.health_flag.fillna("unknown")
    for c in top.columns:
        if top[c].dtype == np.float64:
            top[c] = top[c].astype(np.float32)
    top.to_parquet(FRAME, index=False)
    log(f"frame {top.shape}; labelled rows {int(top.func5.notna().sum()):,}; "
        f"{top.win.nunique()} windows; {time.time()-t0:.0f}s -> {FRAME}")


# ===================================================================== models
NON_FEATURES = {"DeviceId", "Detector", "cand_phase", "win", "dev", "Phase", "Function",
                "fold", "y", "cyc", "partner_phase", "health_flag", "std_phase",
                "pred_phase", "prob", "func5", "src", "label_phase", "other_phase"}

FUNC_PARAMS = dict(objective="multiclass", learning_rate=0.05, num_leaves=31,
                   min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=1.0, n_estimators=1200, n_jobs=12,
                   verbose=-1)
BIN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=800, n_jobs=12,
                  verbose=-1, is_unbalance=True)

DUR_GROUPS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full72"]


def wingroup(w: pd.Series) -> pd.Series:
    return np.where(w == "full72", "full72", w.str.split("_").str[0])


def feat_cols(fr: pd.DataFrame) -> list[str]:
    return [c for c in fr.columns
            if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(fr[c])]


def load_frame() -> pd.DataFrame:
    fr = pd.read_parquet(FRAME)
    fr = fr[fr.func5.notna()].reset_index(drop=True)
    fr["wgroup"] = wingroup(fr.win)
    return fr


def oof_multiclass(fr, y, cols, classes, params=FUNC_PARAMS, sample_weight=None):
    import lightgbm as lgb
    n_class = len(classes)
    P = np.zeros((len(fr), n_class))
    ok = (fr.health_flag != "failed").to_numpy()
    folds = fr.fold.to_numpy()
    models, iters = [], []
    for k in range(N_FOLDS):
        te = folds == k
        inner = (k + 1) % N_FOLDS
        trm, vam = (~te) & ok & (folds != inner), (~te) & ok & (folds == inner)
        prm = dict(params)
        prm["num_class"] = n_class
        n = prm.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **prm)
        m.fit(fr.loc[trm, cols], y[trm],
              sample_weight=None if sample_weight is None else sample_weight[trm],
              eval_set=[(fr.loc[vam, cols], y[vam])], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        P[te] = m.predict_proba(fr.loc[te, cols])
        models.append(m)
        iters.append(m.best_iteration_ or n)
    return P, models, iters


def oof_binary(fr, y, cols, params=BIN_PARAMS):
    import lightgbm as lgb
    p = np.zeros(len(fr))
    ok = (fr.health_flag != "failed").to_numpy()
    folds = fr.fold.to_numpy()
    for k in range(N_FOLDS):
        te = folds == k
        prm = dict(params)
        m = lgb.LGBMClassifier(n_estimators=prm.pop("n_estimators"), **prm)
        m.fit(fr.loc[(~te) & ok, cols], y[(~te) & ok])
        p[te] = m.predict_proba(fr.loc[te, cols])[:, 1]
    return p


def prf(yt, yp, classes) -> dict:
    yt, yp = np.asarray(yt), np.asarray(yp)
    out = {}
    for c in classes:
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"n": int((yt == c).sum()), "precision": round(p, 4),
                  "recall": round(r, 4),
                  "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}
    out["macro_f1"] = round(float(np.mean([out[c]["f1"] for c in classes])), 4)
    return out


def to4(s):
    return np.where(np.asarray(s) == "Yellow_Red", "Other", np.asarray(s))


def score_block(fr, yt, yp, tag) -> dict:
    """Accuracy in the model's own space and in the common 4-class space, by duration."""
    full = (fr.win == "full72").to_numpy()
    yt, yp = np.asarray(yt), np.asarray(yp)
    d = {"tag": tag, "n": int(len(yt)),
         "acc_allwin": round(float((yt == yp).mean()), 4),
         "acc_full72": round(float((yt[full] == yp[full]).mean()), 4),
         "acc4_allwin": round(float((to4(yt) == to4(yp)).mean()), 4),
         "acc4_full72": round(float((to4(yt)[full] == to4(yp)[full]).mean()), 4),
         "acc_on_APC_full72": None}
    m3 = np.isin(yt, CLASSES3) & full
    d["acc_on_APC_full72"] = round(float((yt[m3] == yp[m3]).mean()), 4)
    m3a = np.isin(yt, CLASSES3)
    d["acc_on_APC_allwin"] = round(float((yt[m3a] == yp[m3a]).mean()), 4)
    d["per_class_full72"] = prf(yt[full], yp[full], sorted(set(yt) | set(yp)))
    d["by_duration"] = {g: round(float((yt[(fr.wgroup == g).to_numpy()] ==
                                        yp[(fr.wgroup == g).to_numpy()]).mean()), 4)
                        for g in DUR_GROUPS if (fr.wgroup == g).any()}
    d["by_duration_APC"] = {}
    for g in DUR_GROUPS:
        m = (fr.wgroup == g).to_numpy() & m3a
        if m.any():
            d["by_duration_APC"][g] = round(float((yt[m] == yp[m]).mean()), 4)
    return d


def _conf(yt, yp) -> dict:
    return pd.crosstab(pd.Series(yt, name="true"),
                       pd.Series(yp, name="pred")).to_dict()


def stage_models(args) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    fr = load_frame()
    cols = feat_cols(fr)
    full = (fr.win == "full72").to_numpy()
    log(f"function frame {fr.shape}, {len(cols)} features, "
        f"{fr.DeviceId.nunique()} signals, classes {fr[full].func5.value_counts().to_dict()}")
    res: dict = {"n_rows": int(len(fr)), "n_features": len(cols),
                 "n_signals": int(fr.DeviceId.nunique()),
                 "n_detectors_full72": int(full.sum()),
                 "class_counts_full72": fr[full].func5.value_counts().to_dict()}

    y5 = fr.func5.to_numpy()
    y4 = to4(y5)
    # coverage: a detector is scorable only if it actuated at least once in the window
    nlab = int(pd.read_parquet(LABELS_V3).shape[0])
    res["n_labels_total"] = nlab
    res["coverage_by_duration"] = {
        g: round(float((fr.wgroup == g).sum() /
                       max(nlab * (fr[fr.wgroup == g].win.nunique()), 1)), 4)
        for g in DUR_GROUPS if (fr.wgroup == g).any()}

    # ---------------- (A) 5-class
    i5 = {c: i for i, c in enumerate(CLASSES5)}
    P5, M5, it5 = oof_multiclass(fr, fr.func5.map(i5).to_numpy(), cols, CLASSES5)
    pred5 = np.array(CLASSES5)[P5.argmax(1)]
    res["m5_5class"] = score_block(fr, y5, pred5, "5class")
    res["m5_5class"]["confusion_full72"] = _conf(y5[full], pred5[full])
    res["m5_best_iters"] = it5

    # ---------------- (B) 4-class (Yellow_Red folded into Other)
    i4 = {c: i for i, c in enumerate(CLASSES4)}
    P4, M4, it4 = oof_multiclass(fr, pd.Series(y4).map(i4).to_numpy(), cols, CLASSES4)
    pred4 = np.array(CLASSES4)[P4.argmax(1)]
    res["m4_4class"] = score_block(fr, y4, pred4, "4class")
    res["m4_4class"]["confusion_full72"] = _conf(y4[full], pred4[full])

    # ---------------- (C) 3-class head + Other threshold
    m3rows = np.isin(y5, CLASSES3)
    fr3 = fr[m3rows].reset_index(drop=True)
    i3 = {c: i for i, c in enumerate(CLASSES3)}
    P3tr, M3, it3 = oof_multiclass(fr3, fr3.func5.map(i3).to_numpy(), cols, CLASSES3)
    # score the *whole* label set with fold-appropriate 3-class models
    P3 = np.zeros((len(fr), 3))
    for k in range(N_FOLDS):
        te = (fr.fold == k).to_numpy()
        P3[te] = M3[k].predict_proba(fr.loc[te, cols])
    top3 = P3.max(1)
    arg3 = np.array(CLASSES3)[P3.argmax(1)]
    curve = []
    for th in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        pr = np.where(top3 < th, "Other", arg3)
        b = score_block(fr, y4, pr, f"3class_th{th}")
        b["threshold"] = th
        b["other_precision"] = round(float((y4[pr == "Other"] == "Other").mean()), 4) \
            if (pr == "Other").any() else None
        b["other_recall"] = round(float((pr[y4 == "Other"] == "Other").mean()), 4)
        curve.append(b)
    res["m3_threshold_curve"] = curve
    best3 = max(curve, key=lambda r: r["per_class_full72"]["macro_f1"])
    res["m3_best_threshold"] = best3["threshold"]

    # ---------------- (D) Yellow_Red one-vs-rest
    yyr = (y5 == "Yellow_Red").astype(int)
    pyr = oof_binary(fr, yyr, cols)
    yr = {"base_rate_full72": round(float(yyr[full].mean()), 4),
          "n_pos_full72": int(yyr[full].sum()),
          "n_pos_signals": int(fr[(yyr == 1) & full].DeviceId.nunique()),
          "auc_full72": round(float(roc_auc_score(yyr[full], pyr[full])), 4),
          "ap_full72": round(float(average_precision_score(yyr[full], pyr[full])), 4),
          "auc_allwin": round(float(roc_auc_score(yyr, pyr)), 4),
          "ap_allwin": round(float(average_precision_score(yyr, pyr)), 4)}
    vc = full & np.isin(y5, ["Yellow_Red", "Count"])
    yr["auc_vs_count_full72"] = round(float(roc_auc_score(yyr[vc], pyr[vc])), 4)
    yr["ap_vs_count_full72"] = round(float(average_precision_score(yyr[vc], pyr[vc])), 4)
    pts = []
    for th in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        m = (pyr >= th) & full
        pts.append({"th": th, "n_pred": int(m.sum()),
                    "precision": round(float(yyr[m].mean()), 4) if m.any() else None,
                    "recall": round(float(m[full][yyr[full] == 1].mean()), 4)})
    yr["operating_points_full72"] = pts
    # the 5-class head's own Yellow_Red column
    jy = CLASSES5.index("Yellow_Red")
    yr["auc_from_5class_full72"] = round(
        float(roc_auc_score(yyr[full], P5[full, jy])), 4)
    yr["ap_from_5class_full72"] = round(
        float(average_precision_score(yyr[full], P5[full, jy])), 4)
    yr["vs_count_auc_from_5class"] = round(float(roc_auc_score(yyr[vc], P5[vc, jy])), 4)
    yr5 = []
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = (P5[:, jy] >= th) & full
        yr5.append({"th": th, "n_pred": int(m.sum()),
                    "precision": round(float(yyr[m].mean()), 4) if m.any() else None,
                    "recall": round(float(m[full][yyr[full] == 1].mean()), 4)})
    yr["operating_points_5class_full72"] = yr5
    yr["by_duration_recall_at_argmax"] = {
        g: round(float((pred5[(fr.wgroup == g).to_numpy() & (yyr == 1)] ==
                        "Yellow_Red").mean()), 4)
        for g in DUR_GROUPS if ((fr.wgroup == g).to_numpy() & (yyr == 1)).any()}
    res["yellow_red"] = yr

    # ---------------- (E) what the classes look like (medians of the top features)
    imp = pd.Series(np.mean([m.booster_.feature_importance("gain") for m in M5], axis=0),
                    index=cols).sort_values(ascending=False)
    imp.to_csv(WORK / "feature_importance_function_v3.csv", header=["gain"])
    res["top_features_5class"] = imp.head(30).round(1).to_dict()
    ff = fr[full]
    res["class_signatures"] = {
        f: ff.groupby("func5")[f].median().round(4).to_dict()
        for f in ["yr_f_on_yr", "yr_lift_yr", "yr_hit_yr", "yr_lag_med",
                  "call43_fwd_035", "queue_occ_pre_green", "det_dur_med",
                  "f_on_green", "occ_lift_green", "first_on_med", "det_on_per_hour",
                  "lagsib_best_lag", "lagsib_lead_frac", "lagsib_twin_coinc",
                  "f_occ_red", "occ_red_per_on_red", "late2_frac"]
        if f in ff.columns}

    # ---------------- (F) per raw label string
    o = ff[~ff.func5.isin(CLASSES3 + ["Yellow_Red"])].copy()
    o["pred5"] = pred5[full][~np.isin(y5[full], CLASSES3 + ["Yellow_Red"])]
    by = o.groupby("Function").agg(
        n=("Function", "size"),
        to_other=("pred5", lambda s: round(float((s == "Other").mean()), 3)))
    top_wrong = o[o.pred5 != "Other"].groupby("Function")["pred5"].agg(
        lambda s: s.value_counts().index[0] if len(s) else "")
    by["top_wrong_class"] = top_wrong
    res["other_by_raw_label_5class"] = by.sort_values(
        "n", ascending=False).reset_index().to_dict("records")

    # "advance presence" is behaviourally both -> how close are the top two of A/P?
    ap_mask = (ff.Function.str.lower() == "advance presence").to_numpy()
    if ap_mask.any():
        Q = P5[full][ap_mask]
        pa, pp = Q[:, 0], Q[:, 1]
        po = Q[:, 4]
        res["advance_presence"] = {
            "n": int(ap_mask.sum()),
            "argmax": pd.Series(np.array(CLASSES5)[Q.argmax(1)]).value_counts().to_dict(),
            "mean_p_other": round(float(po.mean()), 4),
            "share_close_AP_0.25": round(float((np.abs(pa - pp) < 0.25).mean()), 4),
            "share_close_AP_0.35": round(float((np.abs(pa - pp) < 0.35).mean()), 4)}
        for dlt in (0.2, 0.25, 0.3, 0.35, 0.4):
            rule = (P5[full].argmax(1) == 4) | (
                (np.abs(P5[full][:, 0] - P5[full][:, 1]) < dlt) &
                (P5[full][:, 0] + P5[full][:, 1] > 0.6))
            tgt = (y5[full] == "Other")
            res["advance_presence"][f"rule_delta{dlt}"] = {
                "recall_advpres": round(float(rule[ap_mask].mean()), 4),
                "other_recall": round(float(rule[tgt].mean()), 4),
                "other_precision": round(float(tgt[rule].mean()), 4) if rule.any() else None,
                "acc4": round(float((np.where(rule, "Other", to4(pred5[full])) ==
                                     y4[full]).mean()), 4)}

    # ---------------- (G) ablation on the 5-class model
    FAMILIES = {
        "v3_yellow_red": lambda c: c.startswith("yr_") or c.split("__")[0].startswith("yr_"),
        "v3_lag": lambda c: c.startswith(("lagsib_", "lagany_")),
        "sibling": lambda c: c == "sib_n" or "__sib" in c,
        "v2_pair": lambda c: any(k in c for k in
                                 ["pex_", "dtg_h", "first_on_", "cyc_hit_frac", "tog_h0",
                                  "dur_g2", "burst_g2", "release_lag", "queue_end_frac",
                                  "n_long", "call43_b", "call43_red", "cogreen_",
                                  "excl_secs_", "partner_lead_", "__pdiff"]),
    }
    abl = {}
    y5i = fr.func5.map(i5).to_numpy()
    for name, f in FAMILIES.items():
        sub = [c for c in cols if not f(c)]
        Pa, _, _ = oof_multiclass(fr, y5i, sub, CLASSES5)
        pa = np.array(CLASSES5)[Pa.argmax(1)]
        abl[f"minus_{name}"] = {
            "n_features": len(sub),
            "acc_full72": round(float((y5[full] == pa[full]).mean()), 4),
            "acc_allwin": round(float((y5 == pa).mean()), 4),
            "acc_APC_full72": round(float((y5[full & np.isin(y5, CLASSES3)] ==
                                           pa[full & np.isin(y5, CLASSES3)]).mean()), 4),
            "yr_ap_full72": round(float(average_precision_score(
                yyr[full], Pa[full, jy])), 4)}
        log(f"ablation {name}: {json.dumps(abl[f'minus_{name}'])}")
    # and the stage-04 feature set (no v3 families at all)
    sub = [c for c in cols if not FAMILIES["v3_yellow_red"](c)
           and not FAMILIES["v3_lag"](c)]
    Pa, _, _ = oof_multiclass(fr, y5i, sub, CLASSES5)
    pa = np.array(CLASSES5)[Pa.argmax(1)]
    abl["stage04_features_only"] = {
        "n_features": len(sub),
        "acc_full72": round(float((y5[full] == pa[full]).mean()), 4),
        "acc_allwin": round(float((y5 == pa).mean()), 4),
        "acc_APC_full72": round(float((y5[full & np.isin(y5, CLASSES3)] ==
                                       pa[full & np.isin(y5, CLASSES3)]).mean()), 4),
        "yr_ap_full72": round(float(average_precision_score(yyr[full], Pa[full, jy])), 4)}
    abl["full_v3"] = {"n_features": len(cols),
                      "acc_full72": res["m5_5class"]["acc_full72"],
                      "acc_allwin": res["m5_5class"]["acc_allwin"],
                      "acc_APC_full72": res["m5_5class"]["acc_on_APC_full72"],
                      "yr_ap_full72": yr["ap_from_5class_full72"]}
    res["ablation"] = abl

    # ---------------- outputs
    out = fr[["DeviceId", "Detector", "win", "fold", "func5", "Function", "src",
              "health_flag", "det_n_on", "top_prob", "pred_phase"]].copy()
    out[[f"p_{c.lower()}" for c in CLASSES5]] = P5
    out["pred5"] = pred5
    out["p_yr_ovr"] = pyr
    out.to_parquet(WORK / "function_oof_v3_bywindow.parquet", index=False)
    out[out.win == "full72"].drop(columns=["win"]).to_parquet(
        WORK / "function_oof_v3.parquet", index=False)
    json.dump(res, open(WORK / "function_v3_results.json", "w"), indent=1, default=str)
    log(json.dumps({k: res[k] for k in ("m5_5class", "m4_4class", "yellow_red")},
                   indent=1, default=str)[:6000])
    log(f"models done in {time.time()-t0:.0f}s")


# ======================================================================= ship
CAND_DIR = REPO / "models" / "beta_v1_candidate"


def stage_ship(args) -> None:
    """Fit the final DEV-wide 5-class function head, copy the (unchanged) phase models
    next to it, and write the manual-review list."""
    import shutil

    import lightgbm as lgb
    t0 = time.time()
    fr = load_frame()
    cols = feat_cols(fr)
    res = json.load(open(WORK / "function_v3_results.json"))
    ok = (fr.health_flag != "failed").to_numpy()
    y = fr.func5.map({c: i for i, c in enumerate(CLASSES5)}).to_numpy()
    prm = dict(FUNC_PARAMS)
    prm["num_class"] = len(CLASSES5)
    prm["n_estimators"] = int(np.mean(res["m5_best_iters"]))
    m = lgb.LGBMClassifier(**prm)
    m.fit(fr.loc[ok, cols], y[ok])
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    m.booster_.save_model(str(CAND_DIR / "function_lgbm_v3.txt"))
    meta = {
        "classes": CLASSES5, "features": cols,
        "n_estimators": prm["n_estimators"],
        "sibling_features": SIB_FEATS,
        "trained_on": {"windows": sorted(fr.win.unique().tolist()),
                       "n_rows": int(ok.sum()), "n_signals": int(fr.DeviceId.nunique()),
                       "labels": "detector-configs.csv (Dec-2024) + all_configs.csv "
                                 "(2025-02-25) for the channels the Dec file omits"},
        "yellow_red_threshold": 0.5,
        "other_threshold_note": "Other is an explicitly trained class; no probability "
                                "threshold is needed. p_other is the 5th column.",
        "advance_presence_rule": {"enabled": True, "delta": 0.2, "sum_min": 0.6,
                                  "note": "flag as Other when p_advance and p_presence "
                                          "are within delta and together exceed sum_min"},
        "metrics": {k: res[k] for k in ("m5_5class", "m4_4class", "yellow_red")
                    if k in res},
    }
    json.dump(meta, open(CAND_DIR / "function_lgbm_v3.json", "w"), indent=1, default=str)
    beta = REPO / "models" / "beta_v0"
    for f in ("phase_lgbm_v2.txt", "phase_lgbm_v2.json", "decode_lgbm_v2.txt",
              "decode_lgbm_v2.json", "decode_v2.json"):
        shutil.copy2(beta / f, CAND_DIR / f)
    log(f"wrote {CAND_DIR} ({len(cols)} features, {prm['n_estimators']} trees)")

    # ---------------- review list
    oof = pd.read_parquet(WORK / "function_oof_v3.parquet")
    pcols = [f"p_{c.lower()}" for c in CLASSES5]
    oof["conf"] = oof[pcols].max(1)
    bad = oof[(oof.pred5 != oof.func5) & (oof.conf >= 0.8)].copy()
    bad["reason"] = "confident function disagreement (72 h)"
    bad = bad[["DeviceId", "Detector", "func5", "Function", "src", "pred5", "conf",
               "det_n_on", "reason"]]
    dis = pd.read_csv(LABEL_DISAGREE)
    dis = dis.rename(columns={"Function_dec": "func5", "Function_feb": "Function"})
    dis["src"] = "dec-vs-feb"
    dis["pred5"] = dis.func_std
    dis["conf"] = np.nan
    dis["det_n_on"] = np.nan
    dis["reason"] = "label file disagreement Dec-2024 vs Feb-2025: " + dis.issue
    rev = pd.concat([bad, dis[bad.columns]], ignore_index=True)
    # confident Yellow_Red calls on channels the config files do not label Yellow_Red
    yrflag = oof[(oof.pred5 == "Yellow_Red") & (oof.func5 != "Yellow_Red") &
                 (oof.p_yellow_red >= 0.6)].copy()
    yrflag["conf"] = yrflag.p_yellow_red
    yrflag["reason"] = "looks like a Yellow_Red (downstream, non-calling) detector"
    rev = pd.concat([rev, yrflag[bad.columns]], ignore_index=True)
    rev = rev.drop_duplicates(subset=["DeviceId", "Detector", "reason"])
    rev.to_csv(WORK / "review_function.csv", index=False)
    log(f"review list {len(rev)} rows -> {WORK / 'review_function.csv'} "
        f"({rev.reason.str.split(':').str[0].value_counts().to_dict()})")
    log(f"ship done in {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["labels", "phase", "frame", "models", "ship"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
