"""Score a neural prediction file under the beta's minimum-evidence rule.

`src/evaluate.py` gives the protocol headline (unscorable excluded, i.e. >=1 actuation
and the labeled phase is a candidate).  The shipped LightGBM beta additionally refuses
detectors with fewer than 5 actuations, so this prints the same cumulative table so the
two stages are directly comparable.

    python src/neural/score_minact.py --phase preds.parquet [--bywindow bw.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, FOLDS_CSV, LABELS_DEV, connect  # noqa: E402
from evaluate import load_candidates  # noqa: E402

DUR = {"m30": 0.5, "h1": 1.0, "h3": 3.0, "h6": 6.0, "h24": 24.0, "full72": 72.0}


def top1(pred: pd.DataFrame) -> pd.DataFrame:
    i = pred.groupby(["DeviceId", "Detector"], sort=False)["prob"].idxmax()
    out = pred.loc[i, ["DeviceId", "Detector", "cand_phase", "prob"]]
    return out.rename(columns={"cand_phase": "pred", "prob": "top_prob"})


def base_table() -> pd.DataFrame:
    lab = pd.read_parquet(LABELS_DEV)
    folds = pd.read_csv(FOLDS_CSV)
    lab = lab.merge(folds, on="DeviceId", how="inner")
    cand = load_candidates()
    cand["is_cand"] = True
    lab = lab.merge(cand.rename(columns={"Phase": "cand_phase"}),
                    left_on=["DeviceId", "Phase"], right_on=["DeviceId", "cand_phase"],
                    how="left").drop(columns=["cand_phase"])
    lab["is_cand"] = lab.is_cand.fillna(False)
    return lab


def n_on_full() -> pd.DataFrame:
    con = connect(memory_limit="6GB", threads=6)
    df = con.sql(f"""SELECT DeviceId, Detector::INT AS Detector, n_on
                     FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}')""").df()
    con.close()
    return df


def cumulative(ev: pd.DataFrame, title: str) -> None:
    ev = ev[ev.is_cand].copy()
    print(f"\n{title}   (labeled phase is a candidate: n={len(ev)})")
    print("| min actuations | answered | share | accuracy |")
    print("|---|---|---|---|")
    for m in (1, 3, 5, 10, 20, 50, 100):
        s = ev[ev.n_act >= m]
        if not len(s):
            continue
        print(f"| >= {m:3d} | {len(s):5d} | {len(s)/len(ev):.3f} | "
              f"**{(s.pred == s.Phase).mean():.4f}** |")
    print("\n| band | n | accuracy |")
    print("|---|---|---|")
    edges = [(1, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 10**9)]
    for lo, hi in edges:
        s = ev[(ev.n_act >= lo) & (ev.n_act <= hi)]
        if len(s):
            lbl = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
            print(f"| {lbl} | {len(s)} | {(s.pred == s.Phase).mean():.4f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--bywindow", default="")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    lab = base_table()
    if args.folds:
        lab = lab[lab.fold.isin(args.folds)]

    p = top1(pd.read_parquet(args.phase))
    ev = lab.merge(p, on=["DeviceId", "Detector"], how="left").merge(
        n_on_full(), on=["DeviceId", "Detector"], how="left")
    ev["n_act"] = ev.n_on.fillna(0)
    ev["pred"] = ev.pred.fillna(-1).astype(int)
    cumulative(ev, "72 h window (actuations = whole 3-day log)")

    if args.bywindow:
        bw = pd.read_parquet(args.bywindow)
        for win in ["m30_a", "m30_b", "m30_c", "m30_d"]:
            pass
        bw["dur"] = [DUR["full72" if w == "full72" else w.split("_")[0]] for w in bw.win]
        for dur in sorted(bw.dur.unique()):
            sub = bw[bw.dur == dur]
            accs, ns, shares = [], [], []
            for win, g in sub.groupby("win"):
                t = top1(g)
                na = g.groupby(["DeviceId", "Detector"])["n_act"].first().reset_index()
                e = lab.merge(t, on=["DeviceId", "Detector"], how="left").merge(
                    na, on=["DeviceId", "Detector"], how="left")
                e["n_act"] = e.n_act.fillna(0)
                e["pred"] = e.pred.fillna(-1).astype(int)
                e = e[e.is_cand]
                s = e[e.n_act >= 5]
                accs.append((s.pred == s.Phase).mean()); ns.append(len(s))
                shares.append(len(s) / max(len(e), 1))
            print(f"{dur:6.1f} h  >=5 act: acc {np.mean(accs):.4f}  "
                  f"answered {np.mean(shares):.3f} ({int(np.mean(ns))} of {len(e)})")


if __name__ == "__main__":
    main()
