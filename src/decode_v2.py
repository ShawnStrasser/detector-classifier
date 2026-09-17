"""Stage 04 step 3: joint per-signal decoding (second stage), phase-anonymous.

The pair ranker scores each (detector, candidate phase) independently.  A signal, however,
is a *joint* object: detectors on the same approach actuate together, channels are usually
wired in blocks, and every phase that registers vehicle calls normally owns at least one
detector.  This stage learns those constraints from the first-stage OOF probabilities.

Second-stage inputs per (detector, candidate c) -- all number-free:
  * first-stage probability, its logit, rank within the detector, gap to the detector's best
  * **similarity neighbours** (`cross_detector.py`): phi-weighted mean / max of the *other*
    detectors' probability for c, and the phi-weighted share of neighbours whose top-1 is c
  * **channel-adjacency neighbours** (|channel difference| <= 2): same three aggregates.
    Adjacency is a wiring convention, not a phase number; every result is reported with and
    without it because non-ODOT agencies may wire differently.
  * **signal-level occupancy of c**: how many other detectors claim c, the best other
    probability for c, whether c registers vehicle calls but is claimed by nobody
  * context: number of candidates, detector actuation count, window length

Trained as an LGBMRanker on the same 6 grouped folds as the first stage, so a signal's
second-stage score never sees a model that was fitted on that signal.

    python src/train_lgbm_v2.py --stage decode
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:                                   # only needed for training, not for inference
    import lightgbm as lgb
except ImportError:
    lgb = None
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, FOLDS_CSV, LABELS_DEV, MODELS, N_FOLDS, PREDS  # noqa: E402

SIM_FILE = FEATURES / "det_similarity.parquet"
BASE_FEAT = FEATURES / "pair_features_windows.parquet"
GROUP_KEYS = ["DeviceId", "Detector", "win"]
SHORT_WINS = ["m30_a", "m30_b", "m30_c", "m30_d", "h1_a", "h1_b", "h1_c"]

RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                   learning_rate=0.05, num_leaves=31, min_child_samples=60,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=1.0, n_estimators=800, n_jobs=12, verbose=-1,
                   label_gain=[0, 1])


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------- neighbour maths
def _agg_neighbours(pr: pd.DataFrame, nb: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """pr: DeviceId,win,Detector,cand_phase,p0,is_top1.  nb: DeviceId,win,Detector,other,w."""
    j = nb.merge(pr.rename(columns={"Detector": "other", "p0": "np0", "is_top1": "ntop1"}),
                 on=["DeviceId", "win", "other"], how="inner")
    j["wp"] = j.w * j.np0
    j["wt"] = j.w * j.ntop1
    g = j.groupby(["DeviceId", "win", "Detector", "cand_phase"], sort=False)
    out = g.agg(**{f"{prefix}_wsum": ("w", "sum"),
                   f"{prefix}_wp": ("wp", "sum"),
                   f"{prefix}_wt": ("wt", "sum"),
                   f"{prefix}_max": ("np0", "max"),
                   f"{prefix}_n": ("np0", "size")}).reset_index()
    ws = out[f"{prefix}_wsum"].replace(0, np.nan)
    out[f"{prefix}_mean"] = out[f"{prefix}_wp"] / ws
    out[f"{prefix}_top1"] = out[f"{prefix}_wt"] / ws
    return out.drop(columns=[f"{prefix}_wp", f"{prefix}_wt"])


def build_second_stage(pr: pd.DataFrame, ctx: pd.DataFrame,
                       sim: pd.DataFrame | None = None) -> pd.DataFrame:
    """pr = first-stage probabilities for ALL active detectors (labelled or not)."""
    pr = pr.copy()
    g = pr.groupby(["DeviceId", "win", "Detector"], sort=False)["p0"]
    pr["p0_max"] = g.transform("max")
    pr["is_top1"] = (pr.p0 >= pr.p0_max).astype(np.float32)
    pr["p0_gap"] = pr.p0 - pr.p0_max
    pr["p0_rank"] = g.rank(ascending=False, method="first")
    pr["n_cand"] = g.transform("size")
    pr["logit0"] = np.log(np.clip(pr.p0, 1e-6, 1 - 1e-6) /
                          (1 - np.clip(pr.p0, 1e-6, 1 - 1e-6)))

    slim = pr[["DeviceId", "win", "Detector", "cand_phase", "p0", "is_top1"]]

    # --- similarity neighbours
    sim = pd.read_parquet(SIM_FILE) if sim is None else sim.copy()
    sim["w"] = sim.phi.clip(lower=0) ** 2
    sim = sim[sim.w > 0]
    simf = _agg_neighbours(slim, sim[["DeviceId", "win", "Detector", "other", "w"]], "sim")

    # --- channel-adjacency neighbours (|dch| <= 2)
    dets = slim[["DeviceId", "win", "Detector"]].drop_duplicates()
    adj = dets.merge(dets.rename(columns={"Detector": "other"}), on=["DeviceId", "win"])
    d = (adj.Detector - adj.other).abs()
    adj = adj[(d >= 1) & (d <= 2)].copy()
    adj["w"] = np.where((adj.Detector - adj.other).abs() == 1, 1.0, 0.5)
    adjf = _agg_neighbours(slim, adj, "adj")

    # --- signal-level occupancy of the candidate
    sig = slim.groupby(["DeviceId", "win", "cand_phase"], sort=False).agg(
        sig_sum=("p0", "sum"), sig_max=("p0", "max"), sig_claims=("is_top1", "sum"),
        sig_ndet=("p0", "size")).reset_index()

    out = pr.merge(simf, on=["DeviceId", "win", "Detector", "cand_phase"], how="left")
    out = out.merge(adjf, on=["DeviceId", "win", "Detector", "cand_phase"], how="left")
    out = out.merge(sig, on=["DeviceId", "win", "cand_phase"], how="left")
    # leave-one-out versions so a detector never reads its own probability back
    out["sig_sum_o"] = (out.sig_sum - out.p0) / np.maximum(out.sig_ndet - 1, 1)
    out["sig_claims_o"] = out.sig_claims - out.is_top1
    out["sig_max_o"] = np.where(out.p0 >= out.sig_max, np.nan, out.sig_max)
    out = out.merge(ctx, on=["DeviceId", "win", "cand_phase"], how="left")
    # "a phase with vehicle calls should own at least one detector"
    out["calls_unclaimed"] = ((out.call43_per_cycle.fillna(0) > 0.25) &
                              (out.sig_claims_o < 0.5)).astype(np.float32)
    out["green_share_per_claim"] = out.cand_green_share / (out.sig_claims_o + 1.0)
    return out.drop(columns=["sig_sum", "sig_max", "sig_claims"])


SIM_COLS = ["sim_mean", "sim_max", "sim_top1", "sim_wsum", "sim_n"]
ADJ_COLS = ["adj_mean", "adj_max", "adj_top1", "adj_wsum", "adj_n"]
BASE_COLS = ["p0", "logit0", "p0_gap", "p0_rank", "n_cand",
             "sig_sum_o", "sig_claims_o", "sig_max_o", "sig_ndet",
             "cand_green_share", "call43_per_cycle", "calls_unclaimed",
             "green_share_per_claim", "det_n_on", "log_win_hours"]


def _groups(df):
    key = df["win"].astype(str) + "|" + df["DeviceId"] + "|" + df["Detector"].astype(str)
    _, idx, cnt = np.unique(key.to_numpy(), return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


BIN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_child_samples=60, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=800, n_jobs=12, verbose=-1)


def oof_second_stage(df: pd.DataFrame, cols: list[str], mode: str = "rank") -> np.ndarray:
    s = np.zeros(len(df))
    lab = df.Phase.notna()
    for k in range(N_FOLDS):
        te = (df.fold == k).to_numpy()
        inner = (k + 1) % N_FOLDS
        tr = df[(~te) & lab & (df.fold != inner)]
        va = df[(~te) & lab & (df.fold == inner)]
        cb = [lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)]
        if mode == "binary":
            P = dict(BIN_PARAMS)
            m = lgb.LGBMClassifier(n_estimators=P.pop("n_estimators"), **P)
            m.fit(tr[cols], tr.y, eval_set=[(va[cols], va.y)],
                  eval_metric="binary_logloss", callbacks=cb)
            s[te] = m.predict_proba(df.loc[te, cols])[:, 1]
        else:
            P = dict(RANK_PARAMS)
            m = lgb.LGBMRanker(n_estimators=P.pop("n_estimators"), **P)
            m.fit(tr[cols], tr.y, group=_groups(tr), eval_set=[(va[cols], va.y)],
                  eval_group=[_groups(va)], callbacks=cb)
            s[te] = m.predict(df.loc[te, cols])
    return s


def to_prob(df, s, T: float = 1.0):
    d = pd.DataFrame({"g": df["win"].astype(str) + "|" + df.DeviceId + "|" +
                      df.Detector.astype(str), "s": np.asarray(s) / T})
    d["s"] = np.exp(d.s - d.groupby("g")["s"].transform("max"))
    return (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()


T_GRID = np.exp(np.linspace(np.log(0.05), np.log(4.0), 41))


def fit_temperature(df: pd.DataFrame, s: np.ndarray, mask: np.ndarray) -> float:
    """Temperature that minimises the NLL of the correct candidate (ranker scores have no
    natural scale, so the raw softmax is far too flat / peaked for a usable confidence)."""
    sub = df[mask]
    ss = np.asarray(s)[mask]
    yy = sub.y.to_numpy().astype(bool)
    best, bt = np.inf, 1.0
    for T in T_GRID:
        p = to_prob(sub, ss, T)
        nll = -np.log(np.clip(p[yy], 1e-9, 1)).mean()
        if nll < best:
            best, bt = nll, float(T)
    return bt


def calibrated_prob(df: pd.DataFrame, s: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-fold-out temperature: fold k's probabilities use a T fitted on the other folds."""
    p = np.zeros(len(df))
    lab = df.Phase.notna().to_numpy()
    Ts = {}
    for k in sorted(df.fold.unique()):
        te = (df.fold == k).to_numpy()
        T = fit_temperature(df, s, (~te) & lab)
        Ts[int(k)] = T
        p[te] = to_prob(df[te], np.asarray(s)[te], T)
    return p, Ts


def assemble(pr: pd.DataFrame, pairs: pd.DataFrame | None = None,
             sim: pd.DataFrame | None = None) -> pd.DataFrame:
    """Second-stage design matrix from first-stage probabilities `pr` (DeviceId, Detector,
    win, cand_phase, p0).  `pairs` supplies the context columns; the cached stage-01 feature
    table is used when it is None."""
    need = ["DeviceId", "Detector", "win", "cand_phase", "cand_green_share",
            "call43_per_cycle", "det_n_on", "log_win_hours"]
    ctx = pairs[need] if pairs is not None else pd.read_parquet(BASE_FEAT, columns=need)
    ctxp = ctx.groupby(["DeviceId", "win", "cand_phase"], as_index=False).agg(
        cand_green_share=("cand_green_share", "first"),
        call43_per_cycle=("call43_per_cycle", "first"))
    det = ctx.groupby(["DeviceId", "win", "Detector"], as_index=False).agg(
        det_n_on=("det_n_on", "first"), log_win_hours=("log_win_hours", "first"))
    X = build_second_stage(pr, ctxp, sim)
    return X.merge(det, on=["DeviceId", "win", "Detector"], how="left")


def run_stage(stage: str) -> None:
    if stage == "decode":
        stage_decode()
    elif stage == "function":
        import function_v2
        function_v2.main_stage()
    elif stage == "final":
        stage_final()


def stage_decode() -> None:
    from train_lgbm_v2 import summarise
    t0 = time.time()
    pr = pd.read_parquet(PREDS / "phase_oof_v2_bywindow.parquet").rename(columns={"prob": "p0"})
    log(f"first-stage rows {len(pr):,}")
    X = assemble(pr)
    X = X.merge(pd.read_csv(FOLDS_CSV), on="DeviceId", how="inner")
    X = X.merge(pd.read_parquet(LABELS_DEV), on=["DeviceId", "Detector"], how="left")
    X["y"] = (X.cand_phase == X.Phase).astype(np.int8)
    X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    log(f"second-stage frame {X.shape}")

    lab = X.Phase.notna().to_numpy()
    res = {}
    variants = {"first_stage_only": None,
                "decode_sim_only": BASE_COLS + SIM_COLS,
                "decode_adj_only": BASE_COLS + ADJ_COLS,
                "decode_sim_adj": BASE_COLS + SIM_COLS + ADJ_COLS,
                "decode_sim_adj_binary": BASE_COLS + SIM_COLS + ADJ_COLS,
                "decode_sim_only_binary": BASE_COLS + SIM_COLS,
                "decode_adj_only_binary": BASE_COLS + ADJ_COLS,
                "decode_nostruct_binary": BASE_COLS}
    probs, temps = {}, {}
    for name, cols in variants.items():
        if cols is None:
            p = X.p0.to_numpy()
        elif name.endswith("_binary"):
            s = oof_second_stage(X, cols, mode="binary")
            d = pd.DataFrame({"g": X["win"].astype(str) + "|" + X.DeviceId + "|" +
                              X.Detector.astype(str), "s": np.clip(s, 1e-9, None)})
            p = (d.s / d.groupby("g")["s"].transform("sum")).to_numpy()
        else:
            p, Ts = calibrated_prob(X, oof_second_stage(X, cols))
            temps[name] = Ts
        probs[name] = p
        r = summarise(X[lab].reset_index(drop=True), p[lab], name)
        r["coverage_at_0.9"] = _coverage(X[lab], p[lab], 0.9)
        res[name] = r
        log(json.dumps(r))
    res["temperatures"] = temps
    json.dump(res, open(PREDS / "decode_v2_results.json", "w"), indent=1, default=str)

    # pick the champion on all-window accuracy among the decode variants
    champ = max([k for k in res if k not in ("first_stage_only", "temperatures")],
                key=lambda k: res[k]["acc_allwin"])
    log(f"champion decoder: {champ}")
    X["prob2"] = probs[champ]
    out = X.loc[X.win == "full72", ["DeviceId", "Detector", "cand_phase", "prob2"]].rename(
        columns={"prob2": "prob"})
    out["prob"] = out.prob / out.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
    out["Detector"] = out.Detector.astype(int)
    out["cand_phase"] = out.cand_phase.astype(int)
    out.to_parquet(PREDS / "phase_oof_v2_decoded.parquet", index=False)
    X[["DeviceId", "Detector", "win", "cand_phase", "prob2"]].rename(
        columns={"prob2": "prob"}).to_parquet(
        PREDS / "phase_oof_v2_decoded_bywindow.parquet", index=False)
    json.dump({"champion": champ, "cols": variants[champ],
               "mode": "binary" if champ.endswith("_binary") else "rank",
               "temperature": float(np.median(list(temps[champ].values())))
               if champ in temps else 1.0},
              open(MODELS / "decode_v2.json", "w"), indent=1)
    log(f"decode done in {time.time()-t0:.0f}s")


def _coverage(df: pd.DataFrame, p: np.ndarray, th: float) -> list:
    d = df[["DeviceId", "Detector", "win", "cand_phase", "Phase"]].copy()
    d["p"] = p
    t = d[d.win == "full72"]
    t = t.loc[t.groupby(["DeviceId", "Detector"])["p"].idxmax()]
    m = t.p >= th
    return [float(m.mean()), float((t[m].cand_phase == t[m].Phase).mean()) if m.any() else None]


def stage_final() -> None:
    raise SystemExit("stage_final lives in predict_build.py")
