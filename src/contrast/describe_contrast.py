"""Stage 08 step 4 -- what does the demand contrast actually see?

Medians of the contrast features per function class at 72 h, plus the exact-vs-binned
occupancy agreement, so the write-up can say something in plain English.

    python src/contrast/describe_contrast.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DC_WORK  # noqa: E402

WORK = DC_WORK / "contrast"
SHOW = ["cq_occred_lo", "cq_occred_hi", "cq_occred_d", "cq_occred_lr", "cq_occred_sl",
        "cq_occdet_lo", "cq_occdet_hi", "cq_occdet_lr",
        "cq_durmed_lo", "cq_durmed_hi", "cq_durmed_lr",
        "cq_durq90_lo", "cq_durq90_hi", "cq_durq90_lr",
        "cq_queueocc_lo", "cq_queueocc_hi", "cq_queueocc_lr",
        "cq_straddle_lo", "cq_straddle_hi", "cq_straddle_d",
        "cq_fonred10_lo", "cq_fonred10_hi", "cq_fonred10_d",
        "cq_fraclong_lo", "cq_fraclong_hi", "cq_fraclong_d",
        "cq_call43_lo", "cq_call43_hi", "cq_call43_d",
        "cq_burst_lo", "cq_burst_hi", "cq_burst_d",
        "cq_ctx_demand_lr"]


def main() -> None:
    from function_v3 import FRAME
    fr = pd.read_parquet(FRAME, columns=["DeviceId", "Detector", "win", "func5",
                                         "Function", "pred_phase"])
    fr = fr[fr.func5.notna() & (fr.win == "full72")]
    c = pd.read_parquet(WORK / "contrast_feats.parquet")
    c = c[c.win == "full72"].rename(columns={"cand_phase": "pred_phase"})
    c["Detector"] = c.Detector.astype(fr.Detector.dtype)
    c["pred_phase"] = c.pred_phase.astype(fr.pred_phase.dtype)
    d = fr.merge(c, on=["DeviceId", "Detector", "pred_phase", "win"], how="left")
    tab = d.groupby("func5")[[s for s in SHOW if s in d.columns]].median().round(3).T
    pd.set_option("display.width", 200)
    print(tab.to_string())
    tab.to_csv(WORK / "class_signatures_contrast.csv")

    # exact vs 15 s binned occupancy: how much information is left after quantisation
    out = {}
    for f in ("occred", "occgrn", "occdet", "queueocc", "durmed", "durq90"):
        for suf in ("lo", "hi", "d", "lr"):
            a, b = f"cq_{f}_{suf}", f"bq_{f}_{suf}"
            if a in d.columns and b in d.columns:
                m = d[[a, b]].dropna()
                out[f"{f}_{suf}"] = {
                    "pearson": round(float(np.corrcoef(m[a], m[b])[0, 1]), 4),
                    "spearman": round(float(m[a].rank().corr(m[b].rank())), 4),
                    "median_exact": round(float(m[a].median()), 4),
                    "median_binned": round(float(m[b].median()), 4)}
    # class separation power of each, measured as one-way F-like ratio on A/P/C/YR
    sep = {}
    sub = d[d.func5.isin(["Advance", "Presence", "Count", "Yellow_Red"])]
    for f in ("occred_d", "occred_lr", "occdet_lr", "durmed_lr", "queueocc_lr"):
        for pre in ("cq_", "bq_"):
            col = pre + f
            if col not in sub.columns:
                continue
            v = sub[col].replace([np.inf, -np.inf], np.nan)
            g = sub.func5[v.notna()]
            v = v.dropna()
            gm = v.groupby(g).mean()
            n = v.groupby(g).size()
            between = float((n * (gm - v.mean()) ** 2).sum() / max(len(gm) - 1, 1))
            within = float(v.groupby(g).var().mul(n - 1).sum() / max(len(v) - len(gm), 1))
            sep[col] = round(between / within, 3) if within > 0 else None
    json.dump({"exact_vs_binned": out, "class_separation_F": sep},
              open(WORK / "binned_vs_exact.json", "w"), indent=1)
    print(json.dumps({"class_separation_F": sep}, indent=1))
    print(json.dumps({k: v["spearman"] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
