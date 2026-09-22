"""Task 1b - resolve `review/review_phase.csv` against the OFFICIAL programming.

For every review row, attach the official programming of that channel and a verdict:
    MODEL_RIGHT / HAND_RIGHT / BOTH_WRONG / NO_OFFICIAL / (summary rows: signal-level tally)

Writes `review/review_phase_resolved.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, REPO  # noqa: E402

OFFICIAL = DC_WORK / "official"
REVIEW = REPO / "review"


def _parse_list(s) -> list[int]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [int(t) for t in str(s).replace(";", ",").split(",") if t.strip().isdigit()
            and int(t) > 0]


def main() -> None:
    rv = pd.read_csv(REVIEW / "review_phase.csv")
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    off["Detector"] = off.Detector.astype(int)

    rows = rv[rv.Detector != "ALL"].copy()
    rows["Detector"] = rows.Detector.astype(int)
    m = rows.merge(off[["DeviceId", "Detector", "target_type", "target_num",
                        "additional_call_phases", "additional_call_overlaps",
                        "switch_phase", "delay", "extend", "call_ped", "description",
                        "n_on_dec2024", "n_on_staging"]],
                   on=["DeviceId", "Detector"], how="left")
    m = m.rename(columns={"target_num": "official_num", "target_type": "official_type"})
    m["official_phase"] = np.where(m.official_type == "phase", m.official_num, np.nan)
    m["official_overlap"] = np.where(m.official_type == "overlap", m.official_num, np.nan)

    add = m.additional_call_phases.map(_parse_list)
    lab = pd.to_numeric(m.label_phase, errors="coerce")
    mod = pd.to_numeric(m.model_phase, errors="coerce")

    verdict, note = [], []
    for i in range(len(m)):
        o = m.official_num.iloc[i]
        ot = m.official_type.iloc[i]
        L, P = lab.iloc[i], mod.iloc[i]
        a = add.iloc[i]
        if pd.isna(o) or ot == "none":
            verdict.append("NO_OFFICIAL"); note.append("channel not in the timing database")
            continue
        if ot == "overlap":
            v = "OFFICIAL_OVERLAP"
            n = f"official label is overlap {int(o)}; neither side proposed an overlap"
        elif (not pd.isna(P)) and P == o and (pd.isna(L) or L != o):
            v, n = "MODEL_RIGHT", ""
        elif (not pd.isna(L)) and L == o and (pd.isna(P) or P != o):
            v, n = "HAND_RIGHT", ""
        elif (not pd.isna(L)) and (not pd.isna(P)) and L == o and P == o:
            v, n = "BOTH_RIGHT", ""
        else:
            v = "BOTH_WRONG"
            n = ""
            if (not pd.isna(P)) and int(P) in a:
                v, n = "MODEL_RIGHT_ADDITIONAL", f"model phase is an ADDITIONAL call phase"
            elif (not pd.isna(L)) and int(L) in a:
                v, n = "HAND_RIGHT_ADDITIONAL", f"hand phase is an ADDITIONAL call phase"
            elif (not pd.isna(P)) and P == m.switch_phase.iloc[i] and m.switch_phase.iloc[i] > 0:
                v, n = "MODEL_IS_SWITCH_PHASE", "model phase equals the channel's switch phase"
        verdict.append(v); note.append(n)
    m["verdict"] = verdict
    m["verdict_note"] = note

    # ---- signal-level summary rows (Detector == 'ALL') ----------------------
    summ = rv[rv.Detector == "ALL"].copy()
    hand = pd.concat([pd.read_parquet(DC_WORK / "labels_dev.parquet"),
                      pd.read_parquet(DC_WORK / "labels_test.parquet")], ignore_index=True)
    hand["Detector"] = hand.Detector.astype(int)
    hm = hand.merge(off[["DeviceId", "Detector", "target_type", "target_num"]],
                    on=["DeviceId", "Detector"], how="inner")
    hm["ok"] = (hm.target_type == "phase") & (hm.target_num == hm.Phase)
    tally = hm.groupby("DeviceId").agg(n=("ok", "size"), n_hand_ok=("ok", "sum")).reset_index()
    summ = summ.merge(tally, on="DeviceId", how="left")
    summ["verdict"] = np.where(summ.n.isna(), "NO_OFFICIAL",
                               np.where(summ.n_hand_ok / summ.n < 0.6,
                                        "SIGNAL_RENUMBERED_MODEL_RIGHT",
                                        "SIGNAL_LABELS_MOSTLY_OK"))
    summ["verdict_note"] = ["official agrees with the hand label on "
                            f"{int(b) if not pd.isna(b) else 0}/{int(a) if not pd.isna(a) else 0}"
                            " labelled channels at this signal"
                            for a, b in zip(summ.n, summ.n_hand_ok)]
    out = pd.concat([summ, m], ignore_index=True).sort_values("priority")
    keep = [c for c in rv.columns] + ["official_type", "official_phase", "official_overlap",
                                      "official_num", "additional_call_phases",
                                      "additional_call_overlaps", "switch_phase",
                                      "delay", "extend", "call_ped", "description",
                                      "n_on_dec2024", "n_on_staging", "verdict",
                                      "verdict_note"]
    keep = [c for c in dict.fromkeys(keep) if c in out.columns]
    out[keep].to_csv(REVIEW / "review_phase_resolved.csv", index=False)
    print(f"wrote {REVIEW/'review_phase_resolved.csv'}  {len(out)} rows")
    print("\n=== verdict by category (detector rows)")
    t = pd.crosstab(m.category, m.verdict)
    print(t.to_string())
    print("\n=== overall")
    print(m.verdict.value_counts().to_string())
    print("\n=== summary rows")
    print(summ[["DeviceId", "n", "n_hand_ok", "verdict"]].to_string())


if __name__ == "__main__":
    main()
