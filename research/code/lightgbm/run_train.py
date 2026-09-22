"""Task 3 driver - variants A / B / C, seed repeats, and the learning curve.

    python src/official/run_train.py --step variants --seeds 0,1,2
    python src/official/run_train.py --step curve
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
from common import DC_WORK, FOLDS_CSV  # noqa: E402
import train_official as T  # noqa: E402

OFFICIAL = DC_WORK / "official"
OUT = OFFICIAL / "train"
_EV = {}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    if "ev" in _EV:
        return _EV["ev"], _EV["sim"]
    folds = pd.read_csv(FOLDS_CSV)
    dev = set(folds.DeviceId)
    t0 = time.time()
    df, sim = T.load_dec(keep=dev)
    log(f"DEC frame {df.shape}, sim {sim.shape} in {time.time()-t0:.0f}s")
    df = T.attach_labels(df, T._official_phase())
    hand = T._hand_phase().rename(columns={"Phase": "Phase_hand"})
    hand["Detector"] = hand.Detector.astype(df.Detector.dtype)
    df = df.merge(hand, on=["DeviceId", "Detector"], how="left")
    df["y_hand"] = (df.cand_phase == df.Phase_hand).astype(np.int8)
    df = df.merge(folds, on="DeviceId", how="inner")
    df = T.add_scorable(df)
    hs = df.groupby(["DeviceId", "Detector", "win"], sort=False)["y_hand"].transform("max")
    df["scorable_hand"] = (df.Phase_hand.notna() & (df.det_n_on >= 1) & (hs > 0)).to_numpy()
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    log(f"eval frame {df.shape}; official-labelled rows "
        f"{int(df.Phase.notna().sum()):,}; hand-labelled {int(df.Phase_hand.notna().sum()):,}")
    _EV["ev"], _EV["sim"] = df, sim
    return df, sim


def build_stg(ids: list[str]):
    df, sim = T.load_stg(keep=set(ids))
    off = T._official_phase().copy()
    off["DeviceId"] = off.DeviceId + "@stg"
    df = T.attach_labels(df, off)
    df["Phase_hand"] = np.nan
    df["y_hand"] = 0
    df = T.add_scorable(df)
    df["scorable_hand"] = False
    return df, sim


def build_extra(split: str = "NEWTRAIN", keep_n: int | None = None):
    sp = pd.read_csv(OFFICIAL / "new_signal_split.csv")
    ids = sorted(sp[sp.split == split].DeviceId)
    if keep_n is not None:
        ids = ids[:keep_n]
    df, sim = T.load_stg(keep=set(ids))
    off = T._official_phase()
    off = off.copy(); off["DeviceId"] = off.DeviceId + "@stg"
    df = T.attach_labels(df, off)
    df = df[df.Phase.notna()].reset_index(drop=True)
    df["fold"] = -1
    df["Phase_hand"] = np.nan
    df["y_hand"] = 0
    log(f"extra {split} frame {df.shape}, {df.DeviceId.nunique()} signals")
    return df, sim


def align(ev: pd.DataFrame, extra: pd.DataFrame | None) -> list[str]:
    fc = T.feature_cols(ev)
    if extra is not None:
        fc = [c for c in fc if c in extra.columns]
    return fc


def score(ev, p0, p2, tag, labcol="Phase", scol="scorable", full=T.FULL_DEC) -> dict:
    r1 = T.metrics(T.top1(ev, p0, labcol, scol), full)
    r2 = T.metrics(T.top1(ev, p2, labcol, scol), full)
    return {"tag": tag, "ranker": r1, "decoded": r2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="variants",
                    choices=["variants", "curve", "devstg"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="A,B,C")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    ev, sim = build_eval()
    res = {}

    if a.step == "variants":
        want = a.variants.split(",")
        extra = sim_x = None
        if "C" in want:
            extra, sim_x = build_extra("NEWTRAIN")
        fc = align(ev, extra)
        log(f"{len(fc)} features")
        for v in want:
            for sd in seeds:
                t0 = time.time()
                if v == "A":
                    p0, p2 = T.run_oof(ev, None, fc, sd, sim, None,
                                       ycol="y_hand", labcol="Phase_hand")
                elif v == "B":
                    p0, p2 = T.run_oof(ev, None, fc, sd, sim, None)
                else:
                    p0, p2 = T.run_oof(ev, extra, fc, sd, sim, sim_x)
                r = score(ev, p0, p2, f"{v}_seed{sd}")
                # variant A is trained on hand labels but SCORED on official labels
                res[f"{v}_seed{sd}"] = r
                log(f"{v} seed {sd}: ranker {r['ranker']['primary']:.4f} "
                    f"decoded {r['decoded']['primary']:.4f} "
                    f"(full {r['decoded']['acc_full']:.4f} / m30 "
                    f"{r['decoded']['acc_m30']:.4f}) {time.time()-t0:.0f}s")
                json.dump(res, open(OUT / "variants.json", "w"), indent=1, default=str)
                if sd == seeds[0]:
                    o = ev.loc[ev.win == T.FULL_DEC,
                               ["DeviceId", "Detector", "cand_phase"]].copy()
                    o["prob"] = p2[(ev.win == T.FULL_DEC).to_numpy()]
                    o["prob"] = o.prob / o.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
                    o["Detector"] = o.Detector.astype(int)
                    o["cand_phase"] = o.cand_phase.astype(int)
                    o.to_parquet(DC_WORK / "preds" /
                                 f"phase_oof_official_{v}.parquet", index=False)
                    ob = ev[["DeviceId", "Detector", "win", "cand_phase"]].copy()
                    ob["prob"] = p2
                    ob.to_parquet(OUT / f"oof_bywindow_{v}.parquet", index=False)

    elif a.step == "devstg":
        # the SAME DEV signals, 21 months later: fold models are fitted on Dec-2024 only
        # and applied to the Sept-2026 rows of the signals they never saw.
        sigs = pd.read_csv(OFFICIAL / "stg" / "signals.csv")
        dev_stg = sorted(sigs[sigs.group == "DEV"].DeviceId)
        sdf, ssim = build_stg(dev_stg)
        folds = pd.read_csv(FOLDS_CSV)
        fm = dict(zip(folds.DeviceId, folds.fold))
        sdf["fold"] = sdf.DeviceId.str.replace("@stg", "", regex=False).map(fm)
        sdf = sdf[sdf.fold.notna()].reset_index(drop=True)
        comb = pd.concat([ev, sdf], ignore_index=True)
        sim = pd.concat([sim, ssim], ignore_index=True)
        del sdf
        comb = comb.sort_values(["win", "DeviceId", "Detector", "cand_phase"]) \
                   .reset_index(drop=True)
        fc = T.feature_cols(comb)
        keep = set(ev.DeviceId.unique())          # train on Dec-2024 rows only
        log(f"combined {comb.shape}; training pool {len(keep)} Dec-2024 signals")
        for sd in seeds[:1]:
            p0, p2 = T.run_oof(comb, None, fc, sd, sim, None, keep_train=keep)
            isd = (comb.src == "DEC").to_numpy()
            res["DEC_full72"] = score(comb[isd].reset_index(drop=True), p0[isd], p2[isd],
                                      "DEC", full=T.FULL_DEC)
            iss = ~isd
            res["STG_full66"] = score(comb[iss].reset_index(drop=True), p0[iss], p2[iss],
                                      "STG", full=T.FULL_STG)
            for k, v in res.items():
                log(f"{k}: ranker {v['ranker']['primary']:.4f} "
                    f"decoded {v['decoded']['primary']:.4f} "
                    f"(full {v['decoded']['acc_full']:.4f} n={v['decoded']['n_full']} / "
                    f"m30 {v['decoded']['acc_m30']:.4f})")
            json.dump(res, open(OUT / "devstg.json", "w"), indent=1, default=str)
            o = comb.loc[iss & (comb.win == T.FULL_STG).to_numpy(),
                         ["DeviceId", "Detector", "cand_phase"]].copy()
            o["prob"] = p2[(iss & (comb.win == T.FULL_STG).to_numpy())]
            o["DeviceId"] = o.DeviceId.str.replace("@stg", "", regex=False)
            o["prob"] = o.prob / o.groupby(["DeviceId", "Detector"])["prob"].transform("sum")
            o["Detector"] = o.Detector.astype(int)
            o["cand_phase"] = o.cand_phase.astype(int)
            o.to_parquet(DC_WORK / "preds" / "phase_oof_official_devstg.parquet", index=False)

    else:  # learning curve
        folds = pd.read_csv(FOLDS_CSV)
        rng = np.random.default_rng(0)
        devs = np.array(sorted(folds.DeviceId))
        order = devs[rng.permutation(len(devs))]
        extra_all, sim_x = build_extra("NEWTRAIN")
        nt = sorted(extra_all.DeviceId.unique())
        fc = align(ev, extra_all)
        points = [("100 DEV", 100, 0), ("200 DEV", 200, 0), ("375 DEV (all)", 375, 0),
                  ("375 DEV + 120 new", 375, 120), ("375 DEV + 225 new", 375, 225),
                  ("375 DEV + all new", 375, len(nt))]
        for name, nd, nx in points:
            keep = set(order[:nd])
            ex = None
            if nx:
                ex = extra_all[extra_all.DeviceId.isin(set(nt[:nx]))]
            for sd in seeds:
                t0 = time.time()
                p0, p2 = T.run_oof(ev, ex, fc, sd, sim, sim_x, keep_train=keep)
                r = score(ev, p0, p2, name)
                r["n_train_signals"] = nd + nx
                res[f"{name}|seed{sd}"] = r
                log(f"{name} seed {sd}: ranker {r['ranker']['primary']:.4f} "
                    f"decoded {r['decoded']['primary']:.4f} {time.time()-t0:.0f}s")
                json.dump(res, open(OUT / "curve.json", "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
