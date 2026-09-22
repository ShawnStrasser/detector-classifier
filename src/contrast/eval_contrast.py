"""Stage 08 step 3 -- does the demand-contrast family earn its place?

Same 6 grouped folds, same OOF protocol as stages 04/06, restricted to the windows an
"extended" model would actually see (>= 24 h: `h24_a`, `h24_b`, `full72`).

    python src/contrast/eval_contrast.py --stage function
    python src/contrast/eval_contrast.py --stage phase

Every comparison is **paired**: the two models see identical rows and folds, so the
reported interval is a bootstrap over *signals* of the per-signal paired difference.
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
sys.path.insert(0, str(HERE.parent))
from common import DC_WORK, N_FOLDS  # noqa: E402

WORK = DC_WORK / "contrast"
WORK.mkdir(parents=True, exist_ok=True)
FEATS = WORK / "contrast_feats.parquet"
LONG = ["h24_a", "h24_b", "full72"]
KEY = ["DeviceId", "Detector", "cand_phase", "win"]
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CLASSES3 = ["Advance", "Presence", "Count"]
N_JOBS = 6

# contrast families to test
BIN_BASE = ["occred", "occgrn", "occdet", "queueocc", "durmed", "durq90"]
SIB_CONTRAST = ["cq_occred_d", "cq_occred_lr", "cq_durmed_lr", "cq_occdet_d",
                "cq_queueocc_lr", "cq_straddle_d", "cq_onrate_lr"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_contrast() -> pd.DataFrame:
    c = pd.read_parquet(FEATS)
    c["Detector"] = c.Detector.astype(np.int16)
    c["cand_phase"] = c.cand_phase.astype(np.int16)
    return c


def cq_cols(cols):
    return [c for c in cols if c.startswith("cq_") or c.startswith("sibq_")]


def cc_cols(cols):
    return [c for c in cols if c.startswith("cc_")]


def bq_cols(cols):
    return [c for c in cols if c.startswith("bq_")]


def exact_subset(cols):
    """the cq_ columns matching exactly the six quantities the binned variant covers"""
    return [c for c in cols if c.startswith("cq_")
            and (c.split("_")[1] in BIN_BASE or c.startswith("cq_ctx_"))]


def occ_only(cols, prefix, names):
    """colour-occupancy contrast columns of one flavour + that flavour's evidence context"""
    return [c for c in cols if c.startswith(prefix)
            and (c.split("_")[1] in names or c.startswith(prefix + "ctx_"))]


# the four ways of spelling "occupancy in green / occupancy in red", head to head
OCC_VARIANTS = {
    "occ_exact_interval": ("cq_", ["occred", "occgrn"]),        # ours: exact ON/OFF
    "occ_binned_15s": ("bq_", ["occred", "occgrn"]),            # 15 s quantised
    "occ_percycle_clipped": ("pq_", ["occred", "occgrn"]),      # atspm style, macro
    "occ_clipped_micro": ("pq_", ["occredmi", "occgrnmi"]),     # clipped, time-weighted
}


# ------------------------------------------------------------------ statistics
def boot_diff(dev: np.ndarray, ok_a: np.ndarray, ok_b: np.ndarray,
              n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap over signals of the paired accuracy difference (b - a)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"dev": dev, "a": ok_a.astype(float), "b": ok_b.astype(float)})
    g = d.groupby("dev").agg(sa=("a", "sum"), sb=("b", "sum"), n=("a", "size"))
    sa, sb, nn = g.sa.to_numpy(), g.sb.to_numpy(), g.n.to_numpy()
    m = len(sa)
    idx = rng.integers(0, m, size=(n, m))
    dif = (sb[idx].sum(1) - sa[idx].sum(1)) / nn[idx].sum(1)
    return {"diff": float(ok_b.mean() - ok_a.mean()),
            "lo90": float(np.percentile(dif, 5)),
            "hi90": float(np.percentile(dif, 95)),
            "p_gt0": float((dif > 0).mean())}


def fold_diffs(fold: np.ndarray, ok_a: np.ndarray, ok_b: np.ndarray) -> list:
    out = []
    for k in range(N_FOLDS):
        m = fold == k
        if m.any():
            out.append(round(float(ok_b[m].mean() - ok_a[m].mean()), 5))
        else:
            out.append(None)
    return out


def prf(yt, yp, classes) -> dict:
    yt, yp = np.asarray(yt), np.asarray(yp)
    out = {}
    for c in classes:
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"n": int((yt == c).sum()), "P": round(p, 4), "R": round(r, 4)}
    return out


# ================================================================== function
def stage_function(args) -> None:
    from sklearn.metrics import average_precision_score
    sys.path.insert(0, str(HERE.parent))
    from function_v3 import FRAME, feat_cols, oof_multiclass
    t0 = time.time()
    fr = pd.read_parquet(FRAME)
    fr = fr[fr.func5.notna() & fr.win.isin(LONG)].reset_index(drop=True)
    base = feat_cols(fr)
    c = load_contrast().rename(columns={"cand_phase": "pred_phase"})
    c["pred_phase"] = c.pred_phase.astype(fr.pred_phase.dtype)
    fr = fr.merge(c, on=["DeviceId", "Detector", "pred_phase", "win"], how="left")
    del c
    # sibling-relative contrast (same construction as function_v3.add_sibling_features)
    g = fr.groupby(["DeviceId", "win", "pred_phase"], sort=False)
    new = {}
    for f in SIB_CONTRAST:
        if f in fr.columns:
            new[f"sibq_{f}__sibdiff"] = fr[f] - g[f].transform("median")
            new[f"sibq_{f}__sibrank"] = g[f].rank(pct=True, method="average")
    fr = pd.concat([fr, pd.DataFrame(new, index=fr.index)], axis=1)
    allc = [c for c in fr.columns if c not in set(base)
            and (c.startswith(("cq_", "cc_", "bq_", "pq_", "sibq_")))]
    log(f"frame {fr.shape}: {len(base)} base + {len(allc)} contrast features, "
        f"{fr.DeviceId.nunique()} signals, windows {sorted(fr.win.unique())}")

    y5 = fr.func5.to_numpy()
    i5 = {c: i for i, c in enumerate(CLASSES5)}
    yi = fr.func5.map(i5).to_numpy()
    dev = fr.DeviceId.to_numpy()
    fold = fr.fold.to_numpy()
    full = (fr.win == "full72").to_numpy()
    h24 = fr.win.str.startswith("h24").to_numpy()
    apc = np.isin(y5, CLASSES3)
    yyr = (y5 == "Yellow_Red").astype(int)
    advpres = (fr.Function.astype(str).str.lower() == "advance presence").to_numpy()

    # negative control: same columns, permuted across detectors inside each window
    rng = np.random.default_rng(0)
    occdur = exact_subset(allc)
    sh = {}
    for c in occdur:
        v = fr[c].to_numpy().copy()
        for w in LONG:
            m = np.flatnonzero((fr.win == w).to_numpy())
            v[m] = v[rng.permutation(m)]
        sh[f"sh_{c}"] = v
    fr = pd.concat([fr, pd.DataFrame(sh, index=fr.index)], axis=1)
    shcols = list(sh)
    del sh

    variants = {
        "base": base,
        "base+cq": base + cq_cols(allc),
        "base+cc": base + cc_cols(allc),
        "base+cq_occdur": base + occdur,
        "base+cq_occdur_shuffled": base + shcols,
    }
    for vname, (pre, names) in OCC_VARIANTS.items():
        sub = occ_only(allc, pre, names)
        if sub:
            variants[f"base+{vname}"] = base + sub
    res, preds = {}, {}
    for name, cols in variants.items():
        t1 = time.time()
        P, M, it = oof_multiclass(fr, yi, cols, CLASSES5,
                                  params=dict(objective="multiclass", learning_rate=0.05,
                                              num_leaves=31, min_child_samples=40,
                                              feature_fraction=0.7, bagging_fraction=0.8,
                                              bagging_freq=1, lambda_l2=1.0,
                                              n_estimators=1200, n_jobs=N_JOBS, verbose=-1))
        pred = np.array(CLASSES5)[P.argmax(1)]
        preds[name] = (P, pred)
        ok = pred == y5
        r = {"n_features": len(cols),
             "acc_24h": round(float(ok[h24].mean()), 4),
             "acc_72h": round(float(ok[full].mean()), 4),
             "acc_all": round(float(ok.mean()), 4),
             "acc_APC_24h": round(float(ok[h24 & apc].mean()), 4),
             "acc_APC_72h": round(float(ok[full & apc].mean()), 4),
             "yr_ap_72h": round(float(average_precision_score(
                 yyr[full], P[full, CLASSES5.index("Yellow_Red")])), 4),
             "yr_ap_all": round(float(average_precision_score(
                 yyr, P[:, CLASSES5.index("Yellow_Red")])), 4),
             "advpres_to_other": round(float((pred[advpres] == "Other").mean()), 4),
             "per_class_72h": prf(y5[full], pred[full], CLASSES5),
             "best_iters": it, "secs": round(time.time() - t1, 1)}
        res[name] = r
        log(f"{name}: " + json.dumps({k: v for k, v in r.items()
                                      if k not in ("per_class_72h", "best_iters")}))
        if name == "base+cq":
            imp = pd.Series(np.mean([m.booster_.feature_importance("gain") for m in M],
                                    axis=0), index=cols).sort_values(ascending=False)
            imp.to_csv(WORK / "feature_importance_function_contrast.csv", header=["gain"])
            res["contrast_importance_share"] = round(
                float(imp[[c for c in cols if c in set(allc)]].sum() / imp.sum()), 4)
            res["top_contrast_features"] = imp[[c for c in cols if c in set(allc)]] \
                .head(15).round(1).to_dict()
            res["top_features_overall"] = imp.head(15).round(1).to_dict()

    okb = preds["base"][1] == y5
    for name in variants:
        if name == "base":
            continue
        okx = preds[name][1] == y5
        res[name]["vs_base"] = {
            "all": boot_diff(dev, okb, okx) | {"per_fold": fold_diffs(fold, okb, okx)},
            "72h": boot_diff(dev[full], okb[full], okx[full], seed=1)
                   | {"per_fold": fold_diffs(fold[full], okb[full], okx[full])},
            "24h": boot_diff(dev[h24], okb[h24], okx[h24], seed=2)
                   | {"per_fold": fold_diffs(fold[h24], okb[h24], okx[h24])},
            "APC_all": boot_diff(dev[apc], okb[apc], okx[apc], seed=3)
                       | {"per_fold": fold_diffs(fold[apc], okb[apc], okx[apc])},
        }
        log(f"{name} vs base: " + json.dumps(res[name]["vs_base"]["all"]))
    # occupancy flavours, head to head against the exact-interval one
    ref = "base+occ_exact_interval"
    if ref in preds:
        oe = preds[ref][1] == y5
        hh = {}
        for vname in OCC_VARIANTS:
            k = f"base+{vname}"
            if k == ref or k not in preds:
                continue
            ox = preds[k][1] == y5
            hh[vname] = boot_diff(dev, oe, ox, seed=7) | {
                "per_fold": fold_diffs(fold, oe, ox)}
            log(f"binned/percycle vs exact [{vname}]: " + json.dumps(hh[vname]))
        res["occupancy_flavour_vs_exact"] = hh

    out = fr[["DeviceId", "Detector", "win", "fold", "func5", "Function", "src",
              "health_flag", "det_n_on", "pred_phase"]].copy()
    out[[f"p_{c.lower()}" for c in CLASSES5]] = preds["base+cq"][0]
    out["pred5"] = preds["base+cq"][1]
    out.to_parquet(WORK / "function_oof_contrast_bywindow.parquet", index=False)
    json.dump(res, open(WORK / "function_contrast_results.json", "w"), indent=1,
              default=str)
    log(f"function done in {time.time()-t0:.0f}s")


# ===================================================================== phase
def stage_phase(args) -> None:
    import lightgbm as lgb
    sys.path.insert(0, str(HERE.parent))
    import decode_v2 as dc
    from common import CONCURRENT_PAIRS, DEFAULT_PHASE, LABELS_DEV, REPO
    from features_v2 import PDIFF_FEATS, add_partner_diffs
    from function_v3 import BASE_FILES, V2_FILES, SIM_FILES, FOLDS_V3, _cat
    from train_lgbm_v2 import KEY_EXCLUDE, load_health, oof_predict, to_prob
    t0 = time.time()
    flt = [("win", "in", LONG)]
    df = pd.concat([pd.read_parquet(f, filters=flt) for f in BASE_FILES],
                   ignore_index=True)
    v2 = pd.concat([pd.read_parquet(f, filters=flt) for f in V2_FILES], ignore_index=True)
    df = df.merge(v2, on=KEY, how="left")
    del v2
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    df = df.merge(pd.read_csv(FOLDS_V3), on="DeviceId", how="inner")
    df = df.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    df = df.merge(load_health(), on=["DeviceId", "Detector"], how="left")
    df["health_flag"] = df.health_flag.fillna("unknown")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    base = [c for c in df.columns
            if c not in KEY_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    df = df.merge(load_contrast(), on=KEY, how="left")
    allc = [c for c in df.columns if c.startswith(("cq_", "cc_", "bq_"))]
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    log(f"pairs {df.shape}: {len(base)} base + {len(allc)} contrast, "
        f"{df.DeviceId.nunique()} signals")

    RP = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1], learning_rate=0.05,
              num_leaves=63, min_child_samples=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=1200,
              n_jobs=N_JOBS, verbose=-1, label_gain=[0, 1])
    meta = json.load(open(REPO / "models" / "beta_v0" / "decode_lgbm_v2.json"))
    dcols, dmode = meta["features"], meta.get("mode", "binary")
    sim = _cat(SIM_FILES)
    sim = sim[sim.win.isin(LONG)]
    # negative control: the same columns, permuted across detectors within each window,
    # so the family keeps its marginal distribution but carries no information
    rng = np.random.default_rng(0)
    shuf = {}
    for c in cq_cols(allc):
        v = df[c].to_numpy().copy()
        for w in LONG:
            m = np.flatnonzero((df.win == w).to_numpy())
            v[m] = v[rng.permutation(m)]
        shuf[f"sh_{c}"] = v
    df = pd.concat([df, pd.DataFrame(shuf, index=df.index)], axis=1)
    shcols = list(shuf)
    del shuf

    res, tops = {}, {}
    for name, cols in (("base", base), ("base+cq", base + cq_cols(allc)),
                       ("base+cq_shuffled", base + shcols)):
        t1 = time.time()
        s, models = oof_predict(df, cols, RP, train_mask=df.Phase.notna())
        p0 = to_prob(df, s)
        pr = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
        pr["p0"] = p0
        ctx = df[["DeviceId", "Detector", "win", "cand_phase", "cand_green_share",
                  "call43_per_cycle", "det_n_on", "log_win_hours"]]
        X = dc.assemble(pr, pairs=ctx, sim=sim)
        X = X.merge(pd.read_csv(FOLDS_V3), on="DeviceId", how="inner")
        X = X.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
        X["y"] = (X.cand_phase == X.Phase).astype(np.int8)
        X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
        s2 = dc.oof_second_stage(X, dcols, mode=dmode)
        d = pd.DataFrame({"g": X["win"].astype(str) + "|" + X.DeviceId + "|" +
                          X.Detector.astype(str), "s": np.clip(s2, 1e-9, None)})
        p2 = (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()
        for tag, p in (("ranker", p0), ("decoded", p2)):
            src = df if tag == "ranker" else X
            t = src[["DeviceId", "Detector", "win", "cand_phase", "Phase", "fold"]].copy()
            t["p"] = p
            t = t[t.Phase.notna()]
            t = t.loc[t.groupby(["DeviceId", "Detector", "win"])["p"].idxmax()].copy()
            t["ok"] = (t.cand_phase == t.Phase).to_numpy()
            t["std_phase"] = t.Detector.map(DEFAULT_PHASE)
            t["is_std"] = (t.std_phase == t.Phase) & (t.Detector <= 40)
            tops[f"{name}|{tag}"] = t.reset_index(drop=True)
            f = t[t.win == "full72"]
            err = f[~f.ok]
            nconc = sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                        for a, b in zip(err.Phase, err.cand_phase))
            res[f"{name}|{tag}"] = {
                "acc_72h": round(float(f.ok.mean()), 4),
                "acc_24h": round(float(t[t.win.str.startswith("h24")].ok.mean()), 4),
                "acc_all": round(float(t.ok.mean()), 4),
                "acc_nonstd_72h": round(float(f[~f.is_std].ok.mean()), 4),
                "acc_nonstd_all": round(float(t[~t.is_std].ok.mean()), 4),
                "n_err_72h": int(len(err)), "n_err_concurrent_72h": int(nconc),
                "secs": round(time.time() - t1, 1)}
            log(f"{name}|{tag}: " + json.dumps(res[f'{name}|{tag}']))
        if name == "base+cq":
            # contract-format OOF predictions (72 h and 24 h slices) + by-window detail
            from common import PREDS
            PREDS.mkdir(parents=True, exist_ok=True)
            o = X[["DeviceId", "Detector", "win", "cand_phase"]].copy()
            o["prob"] = p2
            o.to_parquet(WORK / "phase_oof_contrast_bywindow.parquet", index=False)
            for w, sfx in (("full72", "72h"), ("h24_a", "24h")):
                q = o[o.win == w].drop(columns=["win"]).copy()
                q["prob"] = q.prob / q.groupby(["DeviceId", "Detector"])["prob"] \
                                      .transform("sum")
                q["Detector"] = q.Detector.astype(int)
                q["cand_phase"] = q.cand_phase.astype(int)
                q.to_parquet(PREDS / f"phase_oof_contrast_{sfx}.parquet", index=False)
            imp = pd.Series(np.mean([m.booster_.feature_importance("gain")
                                     for m in models], axis=0),
                            index=cols).sort_values(ascending=False)
            imp.to_csv(WORK / "feature_importance_phase_contrast.csv", header=["gain"])
            res["contrast_importance_share_phase"] = round(
                float(imp[[c for c in cols if c.startswith("cq_")]].sum() / imp.sum()), 4)
            res["top_contrast_features_phase"] = imp[
                [c for c in cols if c.startswith("cq_")]].head(10).round(1).to_dict()

    for tag in ("ranker", "decoded"):
      for cmpname in ("base+cq", "base+cq_shuffled"):
        a = tops[f"base|{tag}"].sort_values(["DeviceId", "Detector", "win"]).reset_index(drop=True)
        b = tops[f"{cmpname}|{tag}"].sort_values(["DeviceId", "Detector", "win"]).reset_index(drop=True)
        assert (a.DeviceId.to_numpy() == b.DeviceId.to_numpy()).all()
        full = (a.win == "full72").to_numpy()
        h24 = a.win.str.startswith("h24").to_numpy()
        ns = (~a.is_std).to_numpy()
        rk = f"{'cq' if cmpname == 'base+cq' else 'cq_shuffled'}_vs_base|{tag}"
        res[rk] = {
            "all": boot_diff(a.DeviceId.to_numpy(), a.ok.to_numpy(), b.ok.to_numpy())
                   | {"per_fold": fold_diffs(a.fold.to_numpy(), a.ok.to_numpy(), b.ok.to_numpy())},
            "72h": boot_diff(a.DeviceId.to_numpy()[full], a.ok.to_numpy()[full],
                             b.ok.to_numpy()[full], seed=1)
                   | {"per_fold": fold_diffs(a.fold.to_numpy()[full], a.ok.to_numpy()[full],
                                             b.ok.to_numpy()[full])},
            "24h": boot_diff(a.DeviceId.to_numpy()[h24], a.ok.to_numpy()[h24],
                             b.ok.to_numpy()[h24], seed=2)
                   | {"per_fold": fold_diffs(a.fold.to_numpy()[h24], a.ok.to_numpy()[h24],
                                             b.ok.to_numpy()[h24])},
            "nonstd": boot_diff(a.DeviceId.to_numpy()[ns], a.ok.to_numpy()[ns],
                                b.ok.to_numpy()[ns], seed=3)
                      | {"per_fold": fold_diffs(a.fold.to_numpy()[ns], a.ok.to_numpy()[ns],
                                                b.ok.to_numpy()[ns])}}
        log(f"{rk}: " + json.dumps(res[rk]["all"]))
        if cmpname == "base+cq":
            b.to_parquet(WORK / f"phase_top1_contrast_{tag}.parquet", index=False)
    json.dump(res, open(WORK / "phase_contrast_results.json", "w"), indent=1, default=str)
    log(f"phase done in {time.time()-t0:.0f}s")


# ============================================== phase, mixed durations (ship test)
# A separate "extended" model trained only on >= 24 h samples would be compared against
# the shipped model that is trained on the whole duration mix -- so the honest ship test
# is: keep one model over a duration mix, add the contrast columns (NaN below 24 h), and
# check that the long-window gain survives and the short windows do not regress.
MIXED = LONG + ["m30_a", "m30_c", "h1_a", "h1_b", "h3_a", "h6_a"]


def stage_phase_mixed(args) -> None:
    sys.path.insert(0, str(HERE.parent))
    import decode_v2 as dc
    from common import DEFAULT_PHASE, LABELS_DEV, REPO
    from features_v2 import PDIFF_FEATS, add_partner_diffs
    from function_v3 import BASE_FILES, FOLDS_V3, SIM_FILES, V2_FILES, _cat
    from train_lgbm_v2 import KEY_EXCLUDE, load_health, oof_predict, to_prob
    t0 = time.time()
    flt = [("win", "in", MIXED)]
    df = pd.concat([pd.read_parquet(f, filters=flt) for f in BASE_FILES], ignore_index=True)
    v2 = pd.concat([pd.read_parquet(f, filters=flt) for f in V2_FILES], ignore_index=True)
    df = df.merge(v2, on=KEY, how="left")
    del v2
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    df = df.merge(pd.read_csv(FOLDS_V3), on="DeviceId", how="inner")
    df = df.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    df = df.merge(load_health(), on=["DeviceId", "Detector"], how="left")
    df["health_flag"] = df.health_flag.fillna("unknown")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    base = [c for c in df.columns
            if c not in KEY_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    df = df.merge(load_contrast(), on=KEY, how="left")   # NaN on the short windows
    allc = [c for c in df.columns if c.startswith("cq_")]
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    log(f"mixed pairs {df.shape}: {len(base)} base + {len(allc)} contrast, "
        f"{df.win.nunique()} windows, {df.DeviceId.nunique()} signals")
    RP = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1], learning_rate=0.05,
              num_leaves=63, min_child_samples=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=1200,
              n_jobs=N_JOBS, verbose=-1, label_gain=[0, 1])
    meta = json.load(open(REPO / "models" / "beta_v0" / "decode_lgbm_v2.json"))
    dcols, dmode = meta["features"], meta.get("mode", "binary")
    sim = _cat(SIM_FILES)
    sim = sim[sim.win.isin(MIXED)]
    res, tops = {}, {}
    for name, cols in (("mixed_base", base), ("mixed_base+cq", base + allc)):
        t1 = time.time()
        s, _ = oof_predict(df, cols, RP, train_mask=df.Phase.notna())
        p0 = to_prob(df, s)
        pr = df[["DeviceId", "Detector", "win", "cand_phase"]].copy()
        pr["p0"] = p0
        ctx = df[["DeviceId", "Detector", "win", "cand_phase", "cand_green_share",
                  "call43_per_cycle", "det_n_on", "log_win_hours"]]
        X = dc.assemble(pr, pairs=ctx, sim=sim)
        X = X.merge(pd.read_csv(FOLDS_V3), on="DeviceId", how="inner")
        X = X.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
        X["y"] = (X.cand_phase == X.Phase).astype(np.int8)
        X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
        s2 = dc.oof_second_stage(X, dcols, mode=dmode)
        d = pd.DataFrame({"g": X["win"].astype(str) + "|" + X.DeviceId + "|" +
                          X.Detector.astype(str), "s": np.clip(s2, 1e-9, None)})
        p2 = (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()
        t = X[["DeviceId", "Detector", "win", "cand_phase", "Phase", "fold"]].copy()
        t["p"] = p2
        t = t[t.Phase.notna()]
        t = t.loc[t.groupby(["DeviceId", "Detector", "win"])["p"].idxmax()].copy()
        t["ok"] = (t.cand_phase == t.Phase).to_numpy()
        t["std_phase"] = t.Detector.map(DEFAULT_PHASE)
        t["is_std"] = (t.std_phase == t.Phase) & (t.Detector <= 40)
        t = t.reset_index(drop=True)
        tops[name] = t
        res[name] = {"acc_all": round(float(t.ok.mean()), 4),
                     "by_window": t.groupby("win").ok.mean().round(4).to_dict(),
                     "acc_long": round(float(t[t.win.isin(LONG)].ok.mean()), 4),
                     "acc_short": round(float(t[~t.win.isin(LONG)].ok.mean()), 4),
                     "acc_nonstd_all": round(float(t[~t.is_std].ok.mean()), 4),
                     "secs": round(time.time() - t1, 1)}
        log(f"{name}: " + json.dumps(res[name]))
    a, b = tops["mixed_base"], tops["mixed_base+cq"]
    a = a.sort_values(["DeviceId", "Detector", "win"]).reset_index(drop=True)
    b = b.sort_values(["DeviceId", "Detector", "win"]).reset_index(drop=True)
    lg = a.win.isin(LONG).to_numpy()
    for tag, m in (("all", np.ones(len(a), bool)), ("long", lg), ("short", ~lg)):
        res[f"cq_vs_base_mixed|{tag}"] = boot_diff(
            a.DeviceId.to_numpy()[m], a.ok.to_numpy()[m], b.ok.to_numpy()[m]) | {
            "per_fold": fold_diffs(a.fold.to_numpy()[m], a.ok.to_numpy()[m],
                                   b.ok.to_numpy()[m])}
        log(f"mixed cq vs base [{tag}]: " + json.dumps(res[f'cq_vs_base_mixed|{tag}']))
    json.dump(res, open(WORK / "phase_mixed_results.json", "w"), indent=1, default=str)
    log(f"phase_mixed done in {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["function", "phase", "phase_mixed"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
