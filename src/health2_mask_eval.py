"""Stage 09 (a): does MASKING the bad periods improve phase / function accuracy?

Two feature tables built by `src/health2_features.py` over the same 7 windows and the
same code path -- one with the bad periods removed, one without -- are scored with the
same 6-fold grouped OOF protocol.  Three comparisons:

  base         train on unmasked, score unmasked          (the shipped pipeline)
  mask@infer   train on unmasked, score MASKED features   (masking as data cleaning)
  mask@both    train on masked,   score masked            (what deployment would do)

Reported overall, on the detectors masking actually touches, and per fold, with a
paired bootstrap over signals.  A `--placebo` build (random intervals of the same total
duration per detector) is the control: it answers "does removing ANY 1 % of this
detector's occupancy move the number by the same amount?".

    python src/health2_mask_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DC_WORK, FOLDS_CSV, FUNCTIONS, LABELS_DEV, N_FOLDS  # noqa: E402
from features_v2 import PDIFF_FEATS, add_partner_diffs  # noqa: E402
import train_lgbm_v2 as t2  # noqa: E402
import function_v2 as fv2  # noqa: E402

OUT = DC_WORK / "health2"
KEYS = ["DeviceId", "Detector", "cand_phase", "win"]
GK = ["DeviceId", "Detector", "win"]
# stage-04 ranker, with the protocol's concurrency cap (other jobs are running)
RANK_PARAMS = dict(t2.RANK_PARAMS, n_jobs=6)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(tag: str) -> pd.DataFrame:
    if tag == "cache":      # the stage-01/04 tables, restricted to the stage-09 windows
        from health2_features import WINDOWS_H2
        wins = [w["win"] for w in WINDOWS_H2]
        df = pd.read_parquet(DC_WORK / "features" / "pair_features_windows.parquet")
        df = df[df.win.isin(wins)].reset_index(drop=True)
        v2 = pd.read_parquet(DC_WORK / "features" / "pair_features_v2_extra.parquet")
        v2 = v2[v2.win.isin(wins)].reset_index(drop=True)
    else:
        df = pd.read_parquet(OUT / f"pair_features_{tag}.parquet")
        v2 = pd.read_parquet(OUT / f"pair_features_v2_{tag}.parquet")
    df = df.merge(v2, on=KEYS, how="left")
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    df = df.merge(pd.read_csv(FOLDS_CSV), on="DeviceId", how="inner")
    df = df.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    df = df.merge(t2.load_health(), on=["DeviceId", "Detector"], how="left")
    df["health_flag"] = df.health_flag.fillna("unknown")
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    return df.sort_values(KEYS).reset_index(drop=True)


def align(a: pd.DataFrame, b: pd.DataFrame):
    """Keep only rows present in both tables, in the same order."""
    ka = a[KEYS].astype(str).agg("|".join, axis=1)
    kb = b[KEYS].astype(str).agg("|".join, axis=1)
    common = np.intersect1d(ka.to_numpy(), kb.to_numpy())
    a2 = a[ka.isin(common)].sort_values(KEYS).reset_index(drop=True)
    b2 = b[kb.isin(common)].sort_values(KEYS).reset_index(drop=True)
    assert (a2[KEYS].to_numpy() == b2[KEYS].to_numpy()).all()
    return a2, b2


def top1_frame(d: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    t = d[GK + ["cand_phase", "Phase", "fold", "det_n_on"]].copy()
    t["p"] = prob
    t = t.loc[t.groupby(GK, sort=False)["p"].idxmax()].copy()
    t["ok"] = (t.cand_phase == t.Phase).astype(float)
    return t


def oof_pair(dtrain: pd.DataFrame, dscore_list: dict, feat_cols: list) -> dict:
    """Fit 6 fold models on `dtrain` and score every table in `dscore_list`."""
    ok = dtrain.health_flag.ne("failed") & dtrain.Phase.notna()
    out = {k: np.zeros(len(v)) for k, v in dscore_list.items()}
    for k in range(N_FOLDS):
        te = (dtrain.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        tr = dtrain[(~te) & ok.to_numpy() & (dtrain.fold != inner).to_numpy()]
        va = dtrain[(~te) & ok.to_numpy() & (dtrain.fold == inner).to_numpy()]
        m = t2.fit_one(tr, va, feat_cols, RANK_PARAMS)
        for name, d in dscore_list.items():
            sel = (d.fold == k).to_numpy()
            out[name][sel] = m.predict(d.loc[sel, feat_cols])
        log(f"  fold {k} done (iter {m.best_iteration_})")
    return out


def acc_block(t: pd.DataFrame, sel: np.ndarray, tag: str) -> dict:
    s = t[sel]
    return {"tag": tag, "n": int(len(s)), "acc": float(s.ok.mean()),
            "acc_full72": float(s[s.win == "full72"].ok.mean()) if (s.win == "full72").any() else np.nan,
            "acc_m30": float(s[s.win.str.startswith("m30")].ok.mean()) if s.win.str.startswith("m30").any() else np.nan,
            "acc_h6": float(s[s.win.str.startswith("h6")].ok.mean()) if s.win.str.startswith("h6").any() else np.nan}


def paired(tb: pd.DataFrame, tm: pd.DataFrame, sel: np.ndarray, n_boot=300, seed=0):
    """Paired fold deltas + bootstrap over signals of mean(ok_masked - ok_base)."""
    b, m = tb[sel], tm[sel]
    folds = [(float(m[m.fold == k].ok.mean() - b[b.fold == k].ok.mean()))
             for k in range(N_FOLDS)]
    diff = (m.ok.to_numpy() - b.ok.to_numpy())
    devs = b.DeviceId.to_numpy()
    uniq = np.unique(devs)
    idx = {d: np.where(devs == d)[0] for d in uniq}
    rng = np.random.default_rng(seed)
    boot = [float(diff[np.concatenate([idx[p] for p in rng.choice(uniq, len(uniq), True)])].mean())
            for _ in range(n_boot)]
    return {"delta": float(diff.mean()), "lo": float(np.quantile(boot, 0.05)),
            "hi": float(np.quantile(boot, 0.95)),
            "folds_better": int(sum(f > 0 for f in folds)), "fold_deltas": folds}


# --------------------------------------------------------------------- function
def function_oof(pairs: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    probs = pairs[KEYS].copy()
    probs["prob"] = prob
    top = fv2.build_frame(pairs.drop(columns=["y"], errors="ignore"), probs)
    top = top[top.Function.isin(FUNCTIONS)].reset_index(drop=True)
    cols = [c for c in top.columns
            if c not in t2.KEY_EXCLUDE | {"pred_phase", "top_prob", "p", "ok"}
            and pd.api.types.is_numeric_dtype(top[c])]
    ycode = pd.Categorical(top.Function, categories=list(FUNCTIONS)).codes
    import lightgbm as lgb
    P = np.zeros((len(top), 3))
    for k in range(N_FOLDS):
        te = (top.fold == k).to_numpy()
        prm = dict(fv2.FUNC_PARAMS)
        prm["n_jobs"] = 6
        prm["n_estimators"] = 400        # no early stopping here; same for both arms
        m = lgb.LGBMClassifier(n_estimators=prm.pop("n_estimators"), **prm)
        m.fit(top.loc[~te, cols], ycode[~te])
        P[te] = m.predict_proba(top.loc[te, cols])
    top["fpred"] = np.array(FUNCTIONS)[P.argmax(1)]
    top["ok"] = (top.fpred == top.Function).astype(float)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked-tag", default="masked")
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--skip-function", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    db = load(a.base_tag)
    dm = load(a.masked_tag)
    log(f"base {len(db):,} rows / masked {len(dm):,} rows")
    db, dm = align(db, dm)
    log(f"aligned to {len(db):,} rows, {db.DeviceId.nunique()} signals")

    fc = sorted(set(t2.feature_cols(db)) & set(t2.feature_cols(dm)))
    log(f"{len(fc)} shared features")

    log("training on UNMASKED features ...")
    s_base = oof_pair(db, {"base": db, "mask": dm}, fc)
    log("training on MASKED features ...")
    s_mask = oof_pair(dm, {"mask": dm}, fc)

    tb = top1_frame(db, t2.to_prob(db, s_base["base"]))
    tmi = top1_frame(dm, t2.to_prob(dm, s_base["mask"]))
    tmb = top1_frame(dm, t2.to_prob(dm, s_mask["mask"]))
    for t in (tb, tmi, tmb):
        t.sort_values(GK, inplace=True)
        t.reset_index(drop=True, inplace=True)
    tb = tb[tb.Phase.notna()].reset_index(drop=True)
    tmi = tmi[tmi.Phase.notna()].reset_index(drop=True)
    tmb = tmb[tmb.Phase.notna()].reset_index(drop=True)
    assert len(tb) == len(tmi) == len(tmb)

    mk = pd.read_parquet(OUT / "masked_on_counts.parquet")[
        ["DeviceId", "Detector", "frac_occ", "frac_on"]]
    tb = tb.merge(mk, on=["DeviceId", "Detector"], how="left").fillna({"frac_occ": 0,
                                                                      "frac_on": 0})
    sels = {"all": np.ones(len(tb), bool),
            "affected >1% occ": (tb.frac_occ > 0.01).to_numpy(),
            "affected >5% occ": (tb.frac_occ > 0.05).to_numpy(),
            "affected >20% occ": (tb.frac_occ > 0.20).to_numpy(),
            "untouched": (tb.frac_occ <= 0).to_numpy()}

    res = {"phase": {}, "function": {}}
    for name, sel in sels.items():
        if sel.sum() == 0:
            continue
        res["phase"][name] = {
            "base": acc_block(tb, sel, "base"),
            "mask@infer": acc_block(tmi, sel, "mask@infer"),
            "mask@both": acc_block(tmb, sel, "mask@both"),
            "delta_infer": paired(tb, tmi, sel),
            "delta_both": paired(tb, tmb, sel)}
        d = res["phase"][name]
        log(f"[phase] {name:18s} n={d['base']['n']:6d} base={d['base']['acc']:.4f} "
            f"infer={d['mask@infer']['acc']:.4f} both={d['mask@both']['acc']:.4f} "
            f"| d_infer {d['delta_infer']['delta']*100:+.2f} pt "
            f"({d['delta_infer']['lo']*100:+.2f},{d['delta_infer']['hi']*100:+.2f}) "
            f"{d['delta_infer']['folds_better']}/6")

    if not a.skip_function:
        log("function model, base ...")
        fb = function_oof(db, s_base["base"])
        log("function model, masked ...")
        fm = function_oof(dm, s_mask["mask"])
        fb = fb.sort_values(GK).reset_index(drop=True)
        fm = fm.sort_values(GK).reset_index(drop=True)
        k1 = fb[GK].astype(str).agg("|".join, axis=1)
        k2 = fm[GK].astype(str).agg("|".join, axis=1)
        common = np.intersect1d(k1, k2)
        fb = fb[k1.isin(common)].sort_values(GK).reset_index(drop=True)
        fm = fm[k2.isin(common)].sort_values(GK).reset_index(drop=True)
        fb = fb.merge(mk, on=["DeviceId", "Detector"], how="left").fillna({"frac_occ": 0})
        fsel = {"all": np.ones(len(fb), bool),
                "affected >1% occ": (fb.frac_occ > 0.01).to_numpy(),
                "affected >5% occ": (fb.frac_occ > 0.05).to_numpy()}
        for name, sel in fsel.items():
            if sel.sum() == 0:
                continue
            res["function"][name] = {
                "base": acc_block(fb, sel, "base"),
                "mask@both": acc_block(fm, sel, "mask@both"),
                "delta": paired(fb, fm, sel)}
            d = res["function"][name]
            log(f"[func ] {name:18s} n={d['base']['n']:6d} base={d['base']['acc']:.4f} "
                f"masked={d['mask@both']['acc']:.4f} "
                f"| d {d['delta']['delta']*100:+.2f} pt "
                f"({d['delta']['lo']*100:+.2f},{d['delta']['hi']*100:+.2f}) "
                f"{d['delta']['folds_better']}/6")

    json.dump(res, open(OUT / f"mask_eval_{a.masked_tag}.json", "w"), indent=1,
              default=float)
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
