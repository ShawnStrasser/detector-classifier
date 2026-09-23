"""Track B (stage 15): the six-fold held-out blend table, on stage 13's own rows.

Reruns `blend_v2_predecode.run()` at the shipped weight (0.5, before the joint decoder)
with a Track B network in place of the stage-13 GRU, over the same 701 signals, the same
22 windows and the same scorable rule, so the output sits directly under the table in
`research/notes/13_gru_blend.md` §2.

    python research/code/neural/trackb_eval_full.py --tags tb_final_f0_oof,... --out final6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402
import blend_v2 as B  # noqa: E402
import blend_v2_predecode as BP  # noqa: E402
import trackb_eval as TE  # noqa: E402

TRACKB = DC_WORK / "trackB"
EVALDIR = TRACKB / "eval"
KEY, DET, GRP = B.KEY, B.DET, BP.GRP
FAMS = TE.FAMS


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated prediction tags, one per fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight", type=float, default=0.5)
    a = ap.parse_args()
    EVALDIR.mkdir(parents=True, exist_ok=True)

    nn = pd.concat([TE.read_pred(t) for t in a.tags.split(",")], ignore_index=True)
    log(f"network out-of-fold: {len(nn):,} rows, {nn.DeviceId.nunique()} signals, "
        f"{nn.win.nunique()} windows")

    # the evaluation rows are stage 13's, so the two tables are on identical rows
    keep = B.common_frame()[KEY].copy()
    log(f"evaluation rows {len(keep):,}")

    lg = TE._norm(pd.read_parquet(B.LG_OOF))
    lg = lg.merge(B.folds(), on="DeviceId", how="inner")
    lab = B.labels()
    lg["dev_plain"] = lg.DeviceId.str.replace("@stg", "", regex=False)
    lg = lg.merge(lab.rename(columns={"DeviceId": "dev_plain"}),
                  on=["dev_plain", "Detector"], how="left")
    lg["y"] = np.where(lg.Phase.notna(), (lg.cand_phase == lg.Phase).astype(float), np.nan)
    lg = lg.merge(nn[KEY + ["prob"]].rename(columns={"prob": "p_nn"}), on=KEY, how="left")
    tot = lg.groupby(GRP)["p_nn"].transform("sum")
    lg["p_nn"] = np.where(tot > 0, lg.p_nn / tot.replace(0, np.nan), np.nan)
    cov = float(lg.p_nn.notna().mean())
    log(f"network covers {cov:.1%} of the tree pipeline's rows")
    meta = lg[KEY + ["Phase", "fold", "y"]]
    ctx, sim = BP.load_context(set(lg.DeviceId.unique()))

    res = {"tags": a.tags.split(","), "weight": a.weight, "row_coverage": cov,
           "n_rows": int(len(keep))}

    # --- columns that need no decoding ---------------------------------------
    flat = keep.merge(lg[KEY + ["prob", "p_nn", "Phase", "fold"]], on=KEY, how="left")
    flat["fam"] = flat.win.map(B.fam_of)
    for c in ("prob", "p_nn"):
        flat[c] = flat[c] / flat.groupby(DET)[c].transform("sum")
    res["lightgbm"] = TE.flat(B.acc_by_fam(flat, "prob"))
    res["net"] = TE.flat(B.acc_by_fam(flat.dropna(subset=["p_nn"]), "p_nn"))
    flat["p_b"] = np.where(flat.p_nn.notna(),
                           a.weight * flat.prob + (1 - a.weight) * flat.p_nn, flat.prob)
    res["blend_after"] = TE.flat(B.acc_by_fam(flat, "p_b"))

    # --- the shipped arrangement: blend into the ranker, then decode ----------
    pr = lg[KEY].copy()
    pr["p0"] = np.where(lg.p_nn.notna(),
                        a.weight * lg.p0 + (1 - a.weight) * lg.p_nn, lg.p0)
    pr["p0"] = pr.p0 / pr.groupby(GRP)["p0"].transform("sum")
    out = BP.decode_oof(pr, ctx, sim, meta)
    q = keep.merge(out, on=KEY, how="left").merge(lg[KEY + ["Phase", "fold"]], on=KEY,
                                                  how="left")
    q["p2"] = q.p2.fillna(0.0)
    q["fam"] = q.win.map(B.fam_of)
    res["blend_before"] = TE.flat(B.acc_by_fam(q, "p2"))
    res["blend_before_fold0"] = TE.flat(B.acc_by_fam(q[q.fold == 0], "p2"))
    res["per_fold_m30"] = {int(k): round(float(
        B.acc_by_fam(q[q.fold == k], "p2")["m30"]["acc"]), 5)
        for k in sorted(q.fold.dropna().unique())}
    json.dump(res, open(EVALDIR / f"{a.out}.json", "w"), indent=1, default=str)
    for k in ("lightgbm", "net", "blend_after", "blend_before"):
        log(f"{k:13s} " + json.dumps(res[k]))
    log(f"wrote {EVALDIR / f'{a.out}.json'}")


if __name__ == "__main__":
    main()
