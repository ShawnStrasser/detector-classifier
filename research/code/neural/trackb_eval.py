"""Track B (stage 15): score a Track B network on fold 0, on stage 13's own rows.

Every number here is held-out fold-0 accuracy on **exactly** the rows
`research/code/neural/blend_v2.py` used for the stage-13 table (§2 of
`research/notes/13_gru_blend.md`): the same 22 windows, the same scorable rule (the
labelled phase greens inside the window and the detector actuated), the same LightGBM
out-of-fold probabilities.  Three columns per candidate:

  net          the network alone;
  blend_after  0.5 x LightGBM's decoded probability + 0.5 x the network's;
  blend_before 0.5 x the LightGBM *ranker* + 0.5 x the network, then the joint decoder --
               the shipped arrangement.  Folds 1-5 keep the stage-13 GRU mixture, so the
               decoder is trained on identical rows for every candidate and only fold 0's
               scored rows change; with the stage-13 fold-0 GRU file as the candidate this
               reproduces `blend_before.json`'s `_fold0` entry exactly.

    python research/code/neural/trackb_eval.py --spec spec.json --out b1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402
import decode_train as dec  # noqa: E402
import train_official as T  # noqa: E402
import blend_v2 as B  # noqa: E402
import blend_v2_predecode as BP  # noqa: E402

TRACKB = DC_WORK / "trackB"
PREDS = TRACKB / "preds"
EVALDIR = TRACKB / "eval"
GRU_DIR = DC_WORK / "preds" / "gru2"
KEY, DET, GRP = B.KEY, B.DET, BP.GRP
FAMS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DeviceId"] = df.DeviceId.str.lower()
    df["Detector"] = df.Detector.astype(int)
    df["cand_phase"] = df.cand_phase.astype(int)
    return df


def read_pred(p: str) -> pd.DataFrame:
    """One prediction file; a bare name is looked up in trackB/preds then preds/gru2."""
    q = Path(p)
    for cand in (q, PREDS / p, PREDS / f"{p}_bywindow.parquet",
                 GRU_DIR / p, GRU_DIR / f"{p}_bywindow.parquet"):
        if cand.exists():
            return _norm(pd.read_parquet(cand))
    raise SystemExit(f"prediction file not found: {p}")


def ensemble(paths: list[str]) -> pd.DataFrame:
    """Average the per-candidate probabilities of several seeds, then renormalise."""
    parts = [read_pred(p)[KEY + ["prob", "n_act"]] for p in paths]
    d = parts[0].copy()
    for other in parts[1:]:
        d = d.merge(other[KEY + ["prob"]], on=KEY, how="inner", suffixes=("", "_o"))
        d["prob"] = d.prob + d.prob_o
        d = d.drop(columns=["prob_o"])
    d["prob"] = d.prob / len(parts)
    d["prob"] = d.prob / d.groupby(DET)["prob"].transform("sum")
    return d


# --------------------------------------------------------------- the fixed row set
class Frame:
    """Everything that does not depend on the candidate, built once."""

    def __init__(self) -> None:
        self.gru_all = B.gru_oof()                       # stage-13 GRU, all six folds
        self.base = B.common_frame(self.gru_all)         # p_lg, p0, Phase, fold, fam
        self.f0 = self.base[self.base.fold == 0].reset_index(drop=True)
        self.keep = self.f0[KEY].copy()
        log(f"fold-0 evaluation rows {len(self.keep):,}; "
            f"detector-windows {self.f0.groupby(DET).ngroups:,}")
        # ---- the before-decoder machinery (stage-13 recipe, folds 1-5 frozen) -------
        lg = _norm(pd.read_parquet(B.LG_OOF))
        fold = B.folds()
        lg = lg.merge(fold, on="DeviceId", how="inner")
        lab = B.labels()
        lg["dev_plain"] = lg.DeviceId.str.replace("@stg", "", regex=False)
        lg = lg.merge(lab.rename(columns={"DeviceId": "dev_plain"}),
                      on=["dev_plain", "Detector"], how="left")
        lg["y"] = np.where(lg.Phase.notna(),
                           (lg.cand_phase == lg.Phase).astype(float), np.nan)
        self.lg = lg
        self.meta = lg[KEY + ["Phase", "fold", "y"]]
        self.ctx, self.sim = BP.load_context(set(lg.DeviceId.unique()))
        self.nn_rest = _norm(self.gru_all)[KEY + ["prob"]].merge(
            fold, on="DeviceId", how="inner")
        self.nn_rest = self.nn_rest[self.nn_rest.fold != 0][KEY + ["prob"]]

    # ---------------------------------------------------------------- the two blends
    def net_and_after(self, cand: pd.DataFrame) -> tuple[dict, dict]:
        m = self.f0.drop(columns=["p_nn"]).merge(
            cand[KEY + ["prob"]].rename(columns={"prob": "p_nn"}), on=KEY, how="inner")
        m["p_nn"] = m.p_nn / m.groupby(DET)["p_nn"].transform("sum")
        net = B.acc_by_fam(m, "p_nn")
        m["p_b"] = 0.5 * m.p_lg + 0.5 * m.p_nn
        return net, B.acc_by_fam(m, "p_b")

    def before(self, cand: pd.DataFrame, w: float = 0.5) -> dict:
        """Mix into the ranker, decode, score fold 0.  One LightGBM fit."""
        nn = pd.concat([self.nn_rest, cand[KEY + ["prob"]]], ignore_index=True)
        nn = nn.rename(columns={"prob": "p_nn"})
        lg = self.lg.merge(nn, on=KEY, how="left")
        tot = lg.groupby(GRP)["p_nn"].transform("sum")
        lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
        pr = lg[KEY].copy()
        pr["p0"] = np.where(lg.p_nn.notna(), w * lg.p0 + (1.0 - w) * lg.p_nn, lg.p0)
        pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
        X = dec.assemble(pr, pairs=self.ctx, sim=self.sim).merge(self.meta, on=KEY,
                                                                 how="left")
        X = X.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(
            drop=True)
        cols = dec.BASE_COLS + dec.SIM_COLS + dec.ADJ_COLS
        te = (X.fold == 0).to_numpy()
        lab = X.y.notna().to_numpy()
        base = (~te) & lab
        tr = X[base & (X.fold != 1).to_numpy()]
        va = X[base & (X.fold == 1).to_numpy()]
        P = dict(T.BIN_PARAMS, bagging_seed=0, feature_fraction_seed=100, seed=0, n_jobs=8)
        n = P.pop("n_estimators")
        m = lgb.LGBMClassifier(n_estimators=n, **P)
        m.fit(tr[cols], tr.y.astype(int), eval_set=[(va[cols], va.y.astype(int))],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        s2 = np.zeros(len(X))
        s2[te] = m.predict_proba(X.loc[te, cols])[:, 1]
        X["p2"] = T.norm_prob(X, s2)
        q = self.keep.merge(X[KEY + ["p2"]], on=KEY, how="left").merge(
            self.lg[KEY + ["Phase"]], on=KEY, how="left")
        q["p2"] = q.p2.fillna(0.0)
        q["fam"] = q.win.map(B.fam_of)
        log(f"    decoder: {m.best_iteration_} trees, {int(te.sum()):,} fold-0 rows")
        return B.acc_by_fam(q, "p2")


def flat(r: dict) -> dict:
    return {f: round(r[f]["acc"], 5) for f in FAMS if f in r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True,
                    help="json {name: [pred file, ...]} -- several files = seed ensemble")
    ap.add_argument("--out", required=True, help="basename under trackB/eval/")
    ap.add_argument("--skip-before", action="store_true")
    a = ap.parse_args()
    EVALDIR.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(a.spec))
    fr = Frame()
    res: dict = {"n_rows_fold0": int(len(fr.keep)),
                 "n_detector_windows_fold0": int(fr.f0.groupby(DET).ngroups),
                 "lightgbm_fold0": flat(B.acc_by_fam(fr.f0, "p_lg")),
                 "candidates": {}}
    for name, paths in spec.items():
        t0 = time.time()
        paths = [paths] if isinstance(paths, str) else list(paths)
        cand = ensemble(paths) if len(paths) > 1 else read_pred(paths[0])
        net, after = fr.net_and_after(cand)
        entry = {"files": paths, "n_seeds": len(paths),
                 "net": flat(net), "blend_after": flat(after)}
        if not a.skip_before:
            entry["blend_before"] = flat(fr.before(cand))
        res["candidates"][name] = entry
        log(f"{name}: net {json.dumps(entry['net'])}")
        log(f"{name}: before {json.dumps(entry.get('blend_before', {}))} "
            f"({time.time()-t0:.0f}s)")
        json.dump(res, open(EVALDIR / f"{a.out}.json", "w"), indent=1, default=str)
    log(f"wrote {EVALDIR / f'{a.out}.json'}")


if __name__ == "__main__":
    main()
