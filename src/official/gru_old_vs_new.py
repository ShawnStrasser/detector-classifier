"""Did refitting the GRU help, and by how much?  Old (stage 03) vs new (stage 13).

The two are compared on EXACTLY the rows both cover: the Dec-2024 DEV signals, the
windows the stage-03 inference used (30 min .. 72 h), the detectors the old model scored
(it only covered hand-labelled channels), and the official timing labels.

What changed between them: official labels instead of hand labels, 709 training signals
instead of 375, a cache whose 43/44 call stream is no longer truncated, phase-only loss,
and every fold trained to a documented plateau instead of an epoch cap.

    python src/official/gru_old_vs_new.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import DC_WORK  # noqa: E402
import blend_v2 as B  # noqa: E402

OLD = DC_WORK / "preds" / "neural_best_oof_bywindow.parquet"


def main() -> None:
    old = pd.read_parquet(OLD).rename(columns={"prob": "p_old"})
    new = B.gru_oof().rename(columns={"prob": "p_new"})
    for d in (old, new):
        d["DeviceId"] = d.DeviceId.str.lower()
        d["Detector"] = d.Detector.astype(int)
        d["cand_phase"] = d.cand_phase.astype(int)
    m = old[B.KEY + ["p_old", "n_act"]].merge(new[B.KEY + ["p_new"]], on=B.KEY, how="inner")
    lab = B.labels()
    m = m.merge(lab, on=["DeviceId", "Detector"], how="inner")
    has = m.assign(hit=(m.cand_phase == m.Phase)).groupby(B.DET)["hit"].transform("max")
    na = m.groupby(B.DET)["n_act"].transform("max")
    m = m[(has > 0) & (na >= 1)].reset_index(drop=True)
    for c in ("p_old", "p_new"):
        m[c] = m[c] / m.groupby(B.DET)[c].transform("sum")
    m["fam"] = m.win.map(B.fam_of)
    res = {"n_rows": int(len(m)),
           "n_detectors": int(m[["DeviceId", "Detector"]].drop_duplicates().shape[0]),
           "n_signals": int(m.DeviceId.nunique()),
           "windows": sorted(m.win.unique().tolist()),
           "note": "Dec-2024 DEV signals only, hand-labelled channels only (all the old "
                   "model covered), scored on OFFICIAL timing labels"}
    res["old_gru"] = B.acc_by_fam(m, "p_old")
    res["new_gru"] = B.acc_by_fam(m, "p_new")
    res["gain_pt"] = {f: round(100 * (res["new_gru"][f]["acc"] - res["old_gru"][f]["acc"]), 3)
                      for f in res["old_gru"]}
    # the old run's contract file is the 72 h window only; compare that separately
    oldf = DC_WORK / "preds" / "neural_best_oof_phase.parquet"
    if oldf.exists():
        o2 = pd.read_parquet(oldf).rename(columns={"prob": "p_old"})
        o2["DeviceId"] = o2.DeviceId.str.lower()
        o2["Detector"] = o2.Detector.astype(int)
        o2["cand_phase"] = o2.cand_phase.astype(int)
        o2["win"] = "full72"
        n2 = new[new.win == "full72"]
        m2 = o2[B.KEY + ["p_old"]].merge(n2[B.KEY + ["p_new", "n_act"]], on=B.KEY,
                                         how="inner").merge(
            lab, on=["DeviceId", "Detector"], how="inner")
        h2 = m2.assign(hit=(m2.cand_phase == m2.Phase)).groupby(B.DET)["hit"].transform("max")
        a2 = m2.groupby(B.DET)["n_act"].transform("max")
        m2 = m2[(h2 > 0) & (a2 >= 1)].reset_index(drop=True)
        for c in ("p_old", "p_new"):
            m2[c] = m2[c] / m2.groupby(B.DET)[c].transform("sum")
        m2["fam"] = "full"
        res["full72"] = {"old": B.acc_by_fam(m2, "p_old")["full"],
                         "new": B.acc_by_fam(m2, "p_new")["full"]}
        res["full72"]["gain_pt"] = round(100 * (res["full72"]["new"]["acc"] -
                                                res["full72"]["old"]["acc"]), 3)
        print(f"  full  old {res['full72']['old']['acc']:.4f}  "
              f"new {res['full72']['new']['acc']:.4f}  "
              f"{res['full72']['gain_pt']:+.2f} pt  (n={res['full72']['old']['n']})")
    out = B.OUT / "gru_old_vs_new.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(f"{res['n_detectors']:,} detectors / {res['n_signals']} signals, "
          f"{len(res['windows'])} windows")
    for f in res["old_gru"]:
        print(f"  {f:5s} old {res['old_gru'][f]['acc']:.4f}  new {res['new_gru'][f]['acc']:.4f}"
              f"  {res['gain_pt'][f]:+.2f} pt  (n={res['old_gru'][f]['n']})")
    print("wrote", out)


if __name__ == "__main__":
    main()
