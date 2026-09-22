"""Stage 13: does blending the REFIT GRU with the shipped LightGBM pipeline help?

Everything is measured on held-out (out-of-fold) predictions of the same 701 training
signals, the same 22 windows and the same rows for all three models:

    LightGBM   `dc_work/official/final_v1/oof_bywindow.parquet`  (p0 = ranker bag,
               prob = after the joint decoder -- the shipped pipeline)
    GRU        `dc_work/preds/gru2/gru2_oof_f*_bywindow.parquet` (this stage's refit)
    blend      a weighted average of the two, either AFTER the decoder (average the
               final per-detector probabilities) or BEFORE it (average the ranker and
               GRU scores and then decode).

    python src/official/blend_v2.py --stage after
    python src/official/blend_v2.py --stage before
    python src/official/blend_v2.py --stage report
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
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import CONCURRENT_PAIRS, DC_WORK, DEFAULT_PHASE, N_FOLDS  # noqa: E402

OUT = DC_WORK / "official" / "blend_v2"
LG_OOF = DC_WORK / "official" / "final_v1" / "oof_bywindow.parquet"
GRU_DIR = DC_WORK / "preds" / "gru2"
KEY = ["DeviceId", "Detector", "win", "cand_phase"]
DET = ["DeviceId", "Detector", "win"]

FAMILIES = [("m5", 5), ("m10", 10), ("m30", 30), ("h1", 60), ("h3", 180),
            ("h6", 360), ("h24", 1440), ("full", 4200)]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fam_of(win: str) -> str:
    return "full" if win.startswith("full") else win.split("_")[0]


# ------------------------------------------------------------------ the rows
def labels() -> pd.DataFrame:
    o = pd.read_parquet(DC_WORK / "official" / "labels_official.parquet")
    o = o[o.target_type == "phase"].copy()
    o["DeviceId"] = o.DeviceId.str.lower()
    o["Detector"] = o.Detector.astype(int)
    return o[["DeviceId", "Detector", "target_num"]].rename(columns={"target_num": "Phase"})


def folds() -> pd.DataFrame:
    sys.path.insert(0, str(HERE.parents[0]))
    from neural.data2 import training_signals
    s = training_signals()
    s["DeviceId"] = np.where(s.period == "stg", s.DeviceId + "@stg", s.DeviceId)
    return s[["DeviceId", "fold"]]


def gru_oof(pattern: str = "gru2_oof_f*_bywindow.parquet") -> pd.DataFrame:
    files = sorted(f for f in GRU_DIR.glob(pattern) if "_s1" not in f.name
                   and "_s2" not in f.name)
    if not files:
        raise SystemExit(f"no GRU out-of-fold files in {GRU_DIR}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    log(f"GRU OOF: {len(files)} folds, {len(d):,} rows, "
        f"{d.DeviceId.nunique()} signals, {d.win.nunique()} windows")
    return d


def common_frame(gru: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (signal, detector, window, candidate) present in BOTH models."""
    nn = (gru if gru is not None else gru_oof()).rename(columns={"prob": "p_nn"})
    lg = pd.read_parquet(LG_OOF).rename(columns={"prob": "p_lg"})
    for d in (nn, lg):
        d["DeviceId"] = d.DeviceId.str.lower()
        d["Detector"] = d.Detector.astype(int)
        d["cand_phase"] = d.cand_phase.astype(int)
    m = nn[KEY + ["p_nn", "n_act"]].merge(lg[KEY + ["p_lg", "p0"]], on=KEY, how="inner")
    lab = labels()
    m["dev_plain"] = m.DeviceId.str.replace("@stg", "", regex=False)
    m = m.merge(lab.rename(columns={"DeviceId": "dev_plain"}),
                on=["dev_plain", "Detector"], how="inner")
    m = m.merge(folds(), on="DeviceId", how="inner")
    # scorable: the labelled phase greens inside the window and the detector actuated
    has = m.assign(hit=(m.cand_phase == m.Phase)).groupby(DET)["hit"].transform("max")
    na = m.groupby(DET)["n_act"].transform("max")
    m = m[(has > 0) & (na >= 1)].reset_index(drop=True)
    for c in ("p_lg", "p_nn", "p0"):
        m[c] = m[c] / m.groupby(DET)[c].transform("sum")
    m["fam"] = m.win.map(fam_of)
    log(f"common rows {len(m):,}; detectors/window {m.groupby(DET).ngroups:,}; "
        f"signals {m.DeviceId.nunique()}")
    return m


# ------------------------------------------------------------------- scoring
def top1(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.sort_values(DET + [col, "cand_phase"], ascending=[1, 1, 1, 0, 1])
    t = d.groupby(DET, as_index=False, sort=False).first()
    t["ok"] = (t.cand_phase == t.Phase).astype(np.int8)
    return t


def acc_by_fam(df: pd.DataFrame, col: str) -> dict:
    t = top1(df, col)
    out = {}
    for fam, _ in FAMILIES:
        g = t[t.win.map(fam_of) == fam]
        if not len(g):
            continue
        per = g.groupby("win").ok.mean()
        out[fam] = {"acc": float(per.mean()), "n": int(len(g) / max(len(per), 1)),
                    "per_window": {w: round(float(v), 5) for w, v in per.items()}}
    return out


def blend(df: pd.DataFrame, w: float, a: str = "p_lg", b: str = "p_nn") -> pd.DataFrame:
    df["p_b"] = w * df[a] + (1.0 - w) * df[b]
    return df


def best_weight(df: pd.DataFrame, fams: list[str], grid=None) -> tuple[float, dict]:
    grid = grid if grid is not None else np.round(np.arange(0.0, 1.001, 0.05), 2)
    sub = df[df.fam.isin(fams)].copy()
    res = {}
    for w in grid:
        blend(sub, float(w))
        t = top1(sub, "p_b")
        res[float(w)] = float(t.groupby("win").ok.mean().mean())
    bw = max(res, key=res.get)
    return bw, {str(k): round(v, 5) for k, v in res.items()}


# ------------------------------------------------------------ extra diagnostics
def diagnostics(df: pd.DataFrame, col: str, fam: str) -> dict:
    t = top1(df[df.fam == fam], col)
    std = t.Detector.map(DEFAULT_PHASE)
    ns = ~((std == t.Phase) & (t.Detector <= 40))
    err = t[t.ok == 0]
    conc = int(sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                   for a, b in zip(err.Phase, err.cand_phase)))
    out = {"acc": float(t.ok.mean()), "n": int(len(t)),
           "acc_nonstandard": float(t[ns].ok.mean()) if ns.any() else np.nan,
           "n_nonstandard": int(ns.sum()),
           "n_errors": int(len(err)), "n_errors_concurrent": conc}
    for th in (0.8, 0.9):
        k = t[col] >= th
        out[f"cov{th}"] = float(k.mean())
        out[f"acc_at_cov{th}"] = float(t[k].ok.mean()) if k.any() else np.nan
    return out


# ------------------------------------------------------------------ stage: after
def stage_after(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = common_frame()
    res = {"n_rows": int(len(m)), "n_detector_windows": int(m.groupby(DET).ngroups),
           "n_signals": int(m.DeviceId.nunique()),
           "n_detectors": int(m[["DeviceId", "Detector"]].drop_duplicates().shape[0])}
    res["lightgbm"] = acc_by_fam(m, "p_lg")
    res["gru"] = acc_by_fam(m, "p_nn")
    res["ranker_only"] = acc_by_fam(m, "p0")
    blend(m, 0.5)
    res["blend_50_50"] = acc_by_fam(m, "p_b")

    # ---- weight chosen on folds 1-5, checked on fold 0 ----------------------
    inner = m[m.fold != 0]
    short = [f for f, mins in FAMILIES if mins <= 60]
    bw, grid = best_weight(inner, short)
    res["weight_grid_folds_1_5_short"] = grid
    res["best_weight_folds_1_5_short"] = bw
    per_fam_w = {}
    for fam, _ in FAMILIES:
        w, g = best_weight(inner, [fam])
        per_fam_w[fam] = {"best_w": w, "grid": g}
    res["per_family_weight_folds_1_5"] = per_fam_w

    f0 = m[m.fold == 0].copy()
    res["fold0"] = {}
    for name, w in (("lightgbm", None), ("gru", None), ("blend_50_50", 0.5),
                    (f"blend_w{bw}", bw)):
        if w is None:
            col = "p_lg" if name == "lightgbm" else "p_nn"
            res["fold0"][name] = acc_by_fam(f0, col)
        else:
            blend(f0, w)
            res["fold0"][name] = acc_by_fam(f0, "p_b")

    # ---- the whole pool at the chosen weight --------------------------------
    blend(m, bw)
    res[f"blend_w{bw}"] = acc_by_fam(m, "p_b")

    # ---- gain by duration, and the cutoff ----------------------------------
    gains = {}
    for fam, mins in FAMILIES:
        if fam not in res["lightgbm"]:
            continue
        g05 = 100 * (res["blend_50_50"][fam]["acc"] - res["lightgbm"][fam]["acc"])
        gbw = 100 * (res[f"blend_w{bw}"][fam]["acc"] - res["lightgbm"][fam]["acc"])
        gains[fam] = {"minutes": mins, "gain_50_50_pt": round(g05, 3),
                      f"gain_w{bw}_pt": round(gbw, 3)}
    res["gain_by_duration"] = gains

    # ---- diagnostics at 30 minutes and at the full window -------------------
    res["diagnostics"] = {}
    for fam in ("m30", "full"):
        blend(m, bw)
        res["diagnostics"][fam] = {
            "lightgbm": diagnostics(m, "p_lg", fam),
            "gru": diagnostics(m, "p_nn", fam),
            "blend": diagnostics(m, "p_b", fam)}

    json.dump(res, open(OUT / "blend_after.json", "w"), indent=1, default=str)
    log(f"wrote {OUT/'blend_after.json'}")
    for fam, mins in FAMILIES:
        if fam not in res["lightgbm"]:
            continue
        log(f"{fam:5s} ({mins:5d} min)  lgbm {res['lightgbm'][fam]['acc']:.4f}  "
            f"gru {res['gru'][fam]['acc']:.4f}  50/50 {res['blend_50_50'][fam]['acc']:.4f}  "
            f"w={bw} {res[f'blend_w{bw}'][fam]['acc']:.4f}  "
            f"n={res['lightgbm'][fam]['n']}")


# ----------------------------------------------------------------- stage: seeds
def stage_seeds(a) -> None:
    """Run-to-run noise of the GRU and of the blend: fold 0, three independent fits."""
    OUT.mkdir(parents=True, exist_ok=True)
    base = common_frame()
    base = base[base.fold == 0].reset_index(drop=True)
    res = {"seeds": {}}
    accs, bl = {}, {}
    for s in ("", "_s1", "_s2"):
        f = GRU_DIR / f"gru2_oof_f0{s}_bywindow.parquet"
        if s == "":
            f = GRU_DIR / "gru2_oof_f0_bywindow.parquet"
        if not f.exists():
            continue
        nn = pd.read_parquet(f).rename(columns={"prob": "p_nn2"})
        nn["DeviceId"] = nn.DeviceId.str.lower()
        nn["Detector"] = nn.Detector.astype(int)
        nn["cand_phase"] = nn.cand_phase.astype(int)
        m = base.drop(columns=["p_nn"]).merge(nn[KEY + ["p_nn2"]], on=KEY, how="inner")
        m["p_nn2"] = m.p_nn2 / m.groupby(DET)["p_nn2"].transform("sum")
        name = "seed0" if s == "" else "seed" + s[-1]
        accs[name] = acc_by_fam(m, "p_nn2")
        m["p_b"] = 0.5 * m.p_lg + 0.5 * m.p_nn2
        bl[name] = acc_by_fam(m, "p_b")
    res["gru_by_seed"] = accs
    res["blend_by_seed"] = bl
    res["sd_pt"] = {}
    for fam, _ in FAMILIES:
        v = [a[fam]["acc"] for a in accs.values() if fam in a]
        b = [a[fam]["acc"] for a in bl.values() if fam in a]
        if len(v) > 1:
            res["sd_pt"][fam] = {"gru_sd_pt": round(100 * float(np.std(v, ddof=1)), 3),
                                 "blend_sd_pt": round(100 * float(np.std(b, ddof=1)), 3),
                                 "gru_range_pt": round(100 * (max(v) - min(v)), 3),
                                 "n_seeds": len(v)}
    json.dump(res, open(OUT / "blend_seeds.json", "w"), indent=1, default=str)
    log(f"wrote {OUT/'blend_seeds.json'}: " + json.dumps(res["sd_pt"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["after", "before", "seeds"])
    a = ap.parse_args()
    if a.stage == "before":
        import blend_v2_predecode as B
        B.run()
        return
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
