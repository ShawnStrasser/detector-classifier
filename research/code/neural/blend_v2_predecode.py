"""Stage 13: blend the GRU in BEFORE the joint decoder instead of after it.

The shipped pipeline is  ranker bag -> p0 -> joint decoder -> prob.  "After" blending
averages `prob` with the GRU's probability.  "Before" blending averages **p0** with the
GRU's probability and then runs the decoder on the mixture, so the second stage can use
the neural opinion when it decides which detector claims which phase.

To be a fair comparison the decoder is re-fitted out-of-fold for every mixing weight,
over the same rows, folds, features and parameters `src/official/fit_final_v1.py` used --
including the unlabelled active channels, which carry the decoder's neighbour evidence.
The GRU only has an opinion about channels that carry an official label, so an unlabelled
channel keeps its tree probability; the control w = 1.0 (no GRU at all) re-runs exactly
the shipped recipe and must reproduce the stored `prob` column.

    python src/official/blend_v2.py --stage before
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, N_FOLDS  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import blend_v2 as B  # noqa: E402

FEAT = DC_WORK / "features"
SFEAT = DC_WORK / "official" / "stg" / "features"
CTX = ["DeviceId", "Detector", "cand_phase", "win", "cand_green_share",
       "call43_per_cycle", "det_n_on", "log_win_hours"]
WEIGHTS = [1.0, 0.7, 0.5, 0.3]
GRP = ["DeviceId", "Detector", "win"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_context(keep: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The eight columns `decode_v2.assemble` needs, plus the similarity graph."""
    parts, sims = [], []
    for f in (FEAT / "pair_features_windows.parquet",
              FEAT / "pair_features_windows_B.parquet"):
        if f.exists():
            parts.append(pd.read_parquet(f, columns=CTX))
    for f in (FEAT / "det_similarity.parquet", FEAT / "det_similarity_B.parquet"):
        if f.exists():
            sims.append(pd.read_parquet(f))
    s = pd.read_parquet(SFEAT / "pair_features_stg.parquet", columns=CTX)
    s["DeviceId"] = s.DeviceId + "@stg"
    parts.append(s)
    ss = pd.read_parquet(SFEAT / "det_similarity_stg.parquet")
    ss["DeviceId"] = ss.DeviceId + "@stg"
    sims.append(ss)
    ctx = pd.concat(parts, ignore_index=True)
    sim = pd.concat(sims, ignore_index=True)
    for d in (ctx, sim):
        d["DeviceId"] = d.DeviceId.str.lower()
    ctx = ctx[ctx.DeviceId.isin(keep)].reset_index(drop=True)
    sim = sim[sim.DeviceId.isin(keep)].reset_index(drop=True)
    ctx["Detector"] = ctx.Detector.astype(int)
    ctx["cand_phase"] = ctx.cand_phase.astype(int)
    log(f"context {ctx.shape}, similarity {sim.shape}")
    return ctx, sim


def decode_oof(pr: pd.DataFrame, ctx: pd.DataFrame, sim: pd.DataFrame,
               meta: pd.DataFrame, n_jobs: int = 8) -> pd.DataFrame:
    """6-fold grouped OOF of the joint decoder on the first-stage probabilities `pr`."""
    X = dec.assemble(pr, pairs=ctx, sim=sim)
    X = X.merge(meta, on=B.KEY, how="left")
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
    s2 = np.zeros(len(X))
    lab = X.y.notna().to_numpy()
    for k in range(N_FOLDS):
        te = (X.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        base = (~te) & lab
        tr = X[base & (X.fold != inner).to_numpy()]
        va = X[base & (X.fold == inner).to_numpy()]
        P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0,
                 n_jobs=n_jobs)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols], tr.y.astype(int), eval_set=[(va[cols], va.y.astype(int))],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
        log(f"    decoder fold {k}: {m.best_iteration_} trees, "
            f"{int(te.sum()):,} rows scored")
    X["p2"] = T.norm_prob(X, s2)
    return X[B.KEY + ["p2"]]


def run() -> None:
    B.OUT.mkdir(parents=True, exist_ok=True)
    lg = pd.read_parquet(B.LG_OOF)
    lg["DeviceId"] = lg.DeviceId.str.lower()
    lg["Detector"] = lg.Detector.astype(int)
    lg["cand_phase"] = lg.cand_phase.astype(int)
    nn = B.gru_oof().rename(columns={"prob": "p_nn"})
    nn["DeviceId"] = nn.DeviceId.str.lower()
    nn["Detector"] = nn.Detector.astype(int)
    nn["cand_phase"] = nn.cand_phase.astype(int)
    lg = lg.merge(nn[B.KEY + ["p_nn"]], on=B.KEY, how="left")
    # renormalise the neural opinion over the candidates the tree frame carries
    tot = lg.groupby(GRP)["p_nn"].transform("sum")
    lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
    cov = float(lg.p_nn.notna().mean())
    log(f"GRU covers {cov:.1%} of the tree pipeline's rows")

    fold = B.folds()
    lg = lg.merge(fold, on="DeviceId", how="inner")
    lab = B.labels()
    lg["dev_plain"] = lg.DeviceId.str.replace("@stg", "", regex=False)
    lg = lg.merge(lab.rename(columns={"DeviceId": "dev_plain"}),
                  on=["dev_plain", "Detector"], how="left")
    lg["y"] = np.where(lg.Phase.notna(), (lg.cand_phase == lg.Phase).astype(float), np.nan)
    meta = lg[B.KEY + ["Phase", "fold", "y"]]
    ctx, sim = load_context(set(lg.DeviceId.unique()))

    # evaluate on EXACTLY the rows the "after" analysis uses, so the two are comparable
    keep = B.common_frame()[B.KEY].copy()
    log(f"evaluation rows {len(keep):,}")

    res = {"weights": {}, "gru_row_coverage": cov,
           "note": "w = weight on the LightGBM ranker, 1-w on the GRU, blended BEFORE "
                   "the joint decoder"}
    for w in WEIGHTS:
        t0 = time.time()
        pr = lg[B.KEY].copy()
        mix = np.where(lg.p_nn.notna(), w * lg.p0 + (1.0 - w) * lg.p_nn, lg.p0)
        pr["p0"] = mix
        pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
        log(f"  w={w}: decoding {len(pr):,} rows")
        out = decode_oof(pr, ctx, sim, meta)
        q = keep.merge(out, on=B.KEY, how="left").merge(
            lg[B.KEY + ["Phase", "fold"]], on=B.KEY, how="left")
        q["p2"] = q.p2.fillna(0.0)
        q["fam"] = q.win.map(B.fam_of)
        r = B.acc_by_fam(q, "p2")
        r["_fold0"] = B.acc_by_fam(q[q.fold == 0], "p2")
        res["weights"][str(w)] = r
        log(f"  w={w} done in {time.time()-t0:.0f}s: " + json.dumps(
            {f: round(v["acc"], 4) for f, v in r.items() if not f.startswith("_")}))
        json.dump(res, open(B.OUT / "blend_before.json", "w"), indent=1, default=str)

    # control: the stored decoded probability of the shipped pipeline on the same rows
    ctl = keep.merge(lg[B.KEY + ["prob", "Phase", "fold"]], on=B.KEY, how="left")
    ctl["fam"] = ctl.win.map(B.fam_of)
    res["shipped_prob_same_rows"] = B.acc_by_fam(ctl, "prob")
    json.dump(res, open(B.OUT / "blend_before.json", "w"), indent=1, default=str)
    log(f"wrote {B.OUT/'blend_before.json'}")


if __name__ == "__main__":
    run()
