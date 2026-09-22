"""Task 1c - detailed hand-vs-official analysis + config-drift estimate + delay/extend profile."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, REPO  # noqa: E402

pd.set_option("display.width", 220)
OFFICIAL = DC_WORK / "official"


def main() -> None:
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    off["Detector"] = off.Detector.astype(int)
    dec = pd.read_csv(REPO / "data" / "raw" / "detector-configs.csv")           # Dec-2024
    feb = pd.read_csv(REPO / "data" / "statewide_2025-02-25" / "all_configs.csv")  # Feb-2025
    feb = feb.rename(columns={"Parameter": "Detector"})
    dec["Detector"] = dec.Detector.astype(int); feb["Detector"] = feb.Detector.astype(int)
    dec = dec.drop_duplicates(["DeviceId", "Detector"]); feb = feb.drop_duplicates(["DeviceId", "Detector"])

    print("=== sizes")
    print(f"official rows {len(off):,} signals {off.DeviceId.nunique()}; "
          f"Dec-2024 hand {len(dec):,} / {dec.DeviceId.nunique()}; "
          f"Feb-2025 statewide {len(feb):,} / {feb.DeviceId.nunique()}")

    def agree(hand, tag):
        m = hand.merge(off, on=["DeviceId", "Detector"], how="inner")
        m["ok"] = (m.target_type == "phase") & (m.target_num == m.Phase)
        print(f"{tag}: common {len(m):,} signals {m.DeviceId.nunique()}, "
              f"agree {int(m.ok.sum()):,} = {m.ok.mean():.4f}")
        return m

    mdec = agree(dec, "official(Sep-2026) vs hand Dec-2024   ")
    mfeb = agree(feb, "official(Sep-2026) vs statewide Feb-25")

    # drift: on the signals present in BOTH hand files, do Dec-2024 and Feb-2025 agree
    dd = dec.merge(feb, on=["DeviceId", "Detector"], suffixes=("_dec", "_feb"))
    print(f"Dec-2024 vs Feb-2025 (2 months apart): common {len(dd):,}, "
          f"agree {(dd.Phase_dec == dd.Phase_feb).mean():.4f}")

    print("\n=== where the 5,469-row comparison loses rows")
    hd = dec.merge(off[["DeviceId", "Detector", "target_type"]], on=["DeviceId", "Detector"],
                   how="left", indicator=True)
    miss = hd[hd._merge == "left_only"]
    print(f"hand-labelled channels with NO official row: {len(miss)} on "
          f"{miss.DeviceId.nunique()} signals "
          f"({(~miss.DeviceId.isin(off.DeviceId)).sum()} of them on signals absent "
          "from the plan pull)")

    print("\n=== official label composition (all 35,818 programmed channels)")
    print(off.target_type.value_counts().to_string())
    print("phase+overlap both:", int(off.has_both_phase_and_overlap.sum()),
          "| overlap-only:", int((off.target_type == 'overlap').sum()),
          "| additional call phases:", int((off.n_add_phases > 0).sum()),
          "| switch_phase set:", int((off.switch_phase > 0).sum()),
          "| call_ped set:", int((off.call_ped > 0).sum()))

    print("\n=== real channels (>=1 actuation)")
    for tag, m in [("Dec-2024", off.real_dec2024), ("staging Sep-2026", off.real_staging),
                   ("either", off.real_dec2024 | off.real_staging)]:
        s = off[m]
        print(f"{tag:18s} n={len(s):6,}  signals={s.DeviceId.nunique():4d}  "
              f"overlap-only={int((s.target_type=='overlap').sum()):3d}  "
              f"add-phases={int((s.n_add_phases>0).sum()):4d}  "
              f"delay>0={int((s.delay>0).sum()):5d}  extend>0={int((s.extend>0).sum()):5d}")

    print("\n=== delay / extend on REAL channels (staging)")
    r = off[off.real_staging]
    print("delay :", r.delay.describe(percentiles=[.5, .9, .99]).round(2).to_dict())
    print("extend:", r.extend.describe(percentiles=[.5, .9, .99]).round(2).to_dict())
    print(pd.crosstab(pd.cut(r.delay, [-.1, .001, 3, 8, 100],
                             labels=["0", "<=3", "3-8", ">8"]),
                      pd.cut(r.extend, [-.1, .001, 1.5, 3, 100],
                             labels=["0", "<=1.5", "1.5-3", ">3"])).to_string())

    print("\n=== disagreement detail (hand Dec-2024 vs official)")
    d = mdec[~mdec.ok]
    print(f"n={len(d)}")
    print("official target_type of disagreeing rows:",
          d.target_type.value_counts().to_dict())
    d2 = d[d.target_type == "phase"]
    pair = (d2.Phase.astype(int).astype(str) + "->" + d2.target_num.astype(int).astype(str))
    print("top hand->official phase moves:")
    print(pair.value_counts().head(15).to_string())
    from common import CONCURRENT_PAIRS
    conc = sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
               for a, b in zip(d2.Phase, d2.target_num))
    print(f"concurrent-pair moves: {conc} / {len(d2)}")
    print("disagreement rate by whether the channel has delay/extend set:")
    mdec["delay_set"] = mdec.delay > 0
    mdec["extend_set"] = mdec.extend > 0
    print(mdec.groupby(["delay_set", "extend_set"]).ok.agg(["size", "mean"]).to_string())


if __name__ == "__main__":
    main()
