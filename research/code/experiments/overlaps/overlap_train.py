"""Task 4 - train / evaluate with OVERLAPS as extra candidates.

Two runs on identical rows and identical folds:
  PHASE_ONLY    candidates = phases only (today's model)
  WITH_OVERLAP  candidates = phases + overlaps; the only new input is the number-free
                flag `cand_is_overlap`

Output labels are NUMBERS on both sides: `assign_type` ('phase' | 'overlap') and
`assign_number` (the phase number, or the ODOT overlap number exactly as it appears in
`call_overlap` and as the Parameter of events 61-66).

    python src/official/overlap_train.py --root <dir> --source dec
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
from common import DC_WORK, FOLDS_CSV, N_FOLDS  # noqa: E402
import overlap_cache as OC  # noqa: E402
import train_official as T  # noqa: E402
from features_partner import PDIFF_FEATS, add_partner_diffs  # noqa: E402

OFFICIAL = DC_WORK / "official"
PAIR_KEY = ["DeviceId", "Detector", "cand_phase", "win"]
pd.set_option("display.width", 220)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(root: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    F = root / "features"
    df = pd.read_parquet(F / "pair_ovl.parquet")
    ex = pd.read_parquet(F / "pair_ovl_v2.parquet")
    df = df.merge(ex, on=PAIR_KEY, how="left")
    del ex
    df = add_partner_diffs(df, PDIFF_FEATS + ["on_lift_green", "occ_lift_green",
                                              "f_on_green", "excl_diff_min",
                                              "release_frac_long", "call43_fwd_lift"])
    sim = pd.read_parquet(F / "det_sim_ovl.parquet")
    df["cand_is_overlap"] = (df.cand_phase > OC.OVL_OFFSET).astype(np.int8)

    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    off["Detector"] = off.Detector.astype(df.Detector.dtype)
    off = off[off.target_type.isin(["phase", "overlap"])].copy()
    off["target_cand"] = np.where(off.target_type == "phase", off.target_num,
                                  OC.OVL_OFFSET + off.target_num)
    off["ovl_cand"] = np.where(off.call_overlap > 0,
                               OC.OVL_OFFSET + off.call_overlap, -1)
    df = df.merge(off[["DeviceId", "Detector", "target_type", "target_num", "target_cand",
                       "ovl_cand", "has_both_phase_and_overlap"]],
                  on=["DeviceId", "Detector"], how="left")
    df = df.rename(columns={"target_cand": "Phase"})
    df["y"] = (df.cand_phase == df.Phase).astype(np.int8)
    if source == "dec":
        folds = pd.read_csv(FOLDS_CSV)
    else:
        devs = np.array(sorted(df.DeviceId.unique()))
        rng = np.random.default_rng(0)
        folds = pd.DataFrame({"DeviceId": devs,
                              "fold": rng.integers(0, N_FOLDS, len(devs))})
    df = df.merge(folds, on="DeviceId", how="inner")
    df = T.add_scorable(df)
    df = df.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    log(f"{source}: frame {df.shape}, {df.DeviceId.nunique()} signals; "
        f"overlap candidate rows {int((df.cand_is_overlap == 1).sum()):,}; "
        f"labelled detectors {df[df.Phase.notna()].groupby(['DeviceId','Detector']).ngroups}")
    return df, sim


def evaluate(df: pd.DataFrame, prob: np.ndarray, tag: str, full: str) -> dict:
    d = df[["DeviceId", "Detector", "win", "cand_phase", "Phase", "target_type",
            "target_num", "ovl_cand", "has_both_phase_and_overlap", "scorable",
            "fold"]].copy()
    d["p"] = np.asarray(prob, float)
    d = d[d.scorable]
    d = d.sort_values(["DeviceId", "Detector", "win", "p", "cand_phase"],
                      ascending=[True, True, True, False, True])
    t = d.groupby(["DeviceId", "Detector", "win"], as_index=False, sort=False).first()
    t["ok"] = (t.cand_phase == t.Phase).astype(np.int8)
    t["ok_lenient"] = ((t.cand_phase == t.Phase) |
                       (t.has_both_phase_and_overlap & (t.cand_phase == t.ovl_cand))
                       ).astype(np.int8)
    f = t[t.win == full]
    ph = f[f.target_type == "phase"]
    ov = f[f.target_type == "overlap"]
    both = f[f.has_both_phase_and_overlap.fillna(False)]
    out = {"tag": tag,
           "n_all": int(len(f)), "acc_all": float(f.ok.mean()) if len(f) else np.nan,
           "n_phase": int(len(ph)), "acc_phase": float(ph.ok.mean()) if len(ph) else np.nan,
           "n_overlap_only": int(len(ov)),
           "acc_overlap_only": float(ov.ok.mean()) if len(ov) else np.nan,
           "n_both": int(len(both)),
           "acc_both_strict": float(both.ok.mean()) if len(both) else np.nan,
           "acc_both_lenient": float(both.ok_lenient.mean()) if len(both) else np.nan,
           "acc_allwin": float(t.ok.mean()),
           "n_pred_overlap": int((f.cand_phase > OC.OVL_OFFSET).sum()),
           "n_phase_target_pred_overlap":
               int(((ph.cand_phase > OC.OVL_OFFSET)).sum())}
    return out, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--source", default="dec", choices=["dec", "stg"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    full = "full72" if a.source == "dec" else "full66"
    df, sim = load(root, a.source)
    res = {}
    # --- run 1: phases only (today's model) --------------------------------
    d1 = df[df.cand_is_overlap == 0].reset_index(drop=True)
    d1 = T.add_scorable(d1.drop(columns=["scorable"]))
    fc1 = [c for c in T.feature_cols(d1) if c != "cand_is_overlap"]
    t0 = time.time()
    p0a, p2a = T.run_oof(d1, None, fc1, a.seed, sim, None)
    r, ta = evaluate(d1, p2a, "PHASE_ONLY", full)
    res["PHASE_ONLY"] = r
    log(f"PHASE_ONLY {json.dumps(r)} ({time.time()-t0:.0f}s)")
    # --- run 2: phases + overlaps ------------------------------------------
    fc2 = T.feature_cols(df) + ["cand_is_overlap"]
    fc2 = list(dict.fromkeys(fc2))
    t0 = time.time()
    p0b, p2b = T.run_oof(df, None, fc2, a.seed, sim, None)
    r2, tb = evaluate(df, p2b, "WITH_OVERLAP", full)
    res["WITH_OVERLAP"] = r2
    log(f"WITH_OVERLAP {json.dumps(r2)} ({time.time()-t0:.0f}s)")
    # paired comparison on the phase-target detectors present in both
    key = ["DeviceId", "Detector", "win"]
    m = ta[key + ["ok", "target_type"]].merge(tb[key + ["ok"]], on=key,
                                              suffixes=("_a", "_b"))
    mp = m[m.target_type == "phase"]
    res["paired_phase_targets"] = {
        "n": int(len(mp)), "acc_phase_only": float(mp.ok_a.mean()),
        "acc_with_overlap": float(mp.ok_b.mean()),
        "delta_pt": round(100 * float(mp.ok_b.mean() - mp.ok_a.mean()), 3),
        "broken_by_overlaps": int(((mp.ok_a == 1) & (mp.ok_b == 0)).sum()),
        "fixed_by_overlaps": int(((mp.ok_a == 0) & (mp.ok_b == 1)).sum())}
    log(json.dumps(res["paired_phase_targets"]))
    json.dump(res, open(root / f"overlap_results_{a.source}.json", "w"), indent=1,
              default=str)
    tb["assign_type"] = np.where(tb.cand_phase > OC.OVL_OFFSET, "overlap", "phase")
    tb["assign_number"] = np.where(tb.cand_phase > OC.OVL_OFFSET,
                                   tb.cand_phase - OC.OVL_OFFSET, tb.cand_phase)
    tb.to_parquet(root / f"overlap_top1_{a.source}.parquet", index=False)
    log(f"wrote {root/f'overlap_results_{a.source}.json'}")


if __name__ == "__main__":
    main()
