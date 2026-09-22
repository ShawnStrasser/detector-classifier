"""Task 1 - OFFICIAL detector labels from the controllers' timing databases.

Builds `dc_work/official/labels_official.parquet`, one row per (DeviceId, Detector)
programmed channel, with

    target_type  'phase' | 'overlap'      (call_phase if > 0, else call_overlap)
    target_num   int                      (the phase / overlap number)
    target       'P<n>' / 'O<n>'          (a single string label, overlaps distinct)

and, as METADATA ONLY (never model inputs):
    additional_call_phases, additional_call_overlaps, switch_phase, call_ped,
    delay, extend, description, n_add_phases

plus realness flags per dataset:
    n_on_dec2024, n_on_staging, real_dec2024, real_staging

Also writes the hand-vs-official agreement tables and
`review/review_phase_resolved.csv` (Task 1, "who was right").

    python src/official/labels_official.py --step all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CACHE, DC_WORK, FOLDS_CSV, LABELS_DEV, LABELS_TEST, REPO, connect  # noqa: E402

OFFICIAL = DC_WORK / "official"
PLANS = REPO / "data" / "detector_plans.parquet"
STAGING = (DC_WORK / "data" / "staging" / "date=*" / "part_*.parquet").as_posix()
LABELS_OFFICIAL = OFFICIAL / "labels_official.parquet"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _parse_list(s) -> list[int]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    out = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) > 0:
            out.append(int(tok))
    return out


# --------------------------------------------------------------------- build
def build(con) -> pd.DataFrame:
    pl = con.sql(f"SELECT * FROM read_parquet('{PLANS.as_posix()}')").df()
    log(f"plans: {len(pl):,} rows, {pl.DeviceId.nunique()} signals")
    pl = pl.drop_duplicates(["DeviceId", "Detector"], keep="first")
    pl["Detector"] = pl.Detector.astype(int)

    cp = pl.call_phase.fillna(0).astype(int)
    co = pl.call_overlap.fillna(0).astype(int)
    pl["target_type"] = np.where(cp > 0, "phase", np.where(co > 0, "overlap", "none"))
    pl["target_num"] = np.where(cp > 0, cp, co).astype(int)
    pl["target"] = np.where(pl.target_type == "phase", "P", "O") + pl.target_num.astype(str)
    pl.loc[pl.target_type == "none", "target"] = ""
    pl["call_phase"] = cp
    pl["call_overlap"] = co
    pl["has_both_phase_and_overlap"] = (cp > 0) & (co > 0)
    pl["add_phases"] = pl.additional_call_phases.map(_parse_list)
    pl["add_overlaps"] = pl.additional_call_overlaps.map(_parse_list)
    pl["n_add_phases"] = pl.add_phases.map(len)
    pl["n_add_overlaps"] = pl.add_overlaps.map(len)
    pl["switch_phase"] = pl.switch_phase.fillna(0).astype(int)
    pl["call_ped"] = pl.call_ped.fillna(0).astype(int)
    pl["delay"] = pl.delay.fillna(0.0).astype(float)
    pl["extend"] = pl.extend.fillna(0.0).astype(float)

    # ---- realness: actuations in each dataset -------------------------------
    dec = con.sql(f"""SELECT DeviceId, Detector::INT AS Detector, n_on AS n_on_dec2024
                      FROM read_parquet('{(CACHE/'detector_meta.parquet').as_posix()}')""").df()
    stg = con.sql(f"""SELECT DeviceId, Parameter::INT AS Detector,
                             count(*)::BIGINT AS n_on_staging
                      FROM read_parquet('{STAGING}')
                      WHERE EventId = 82 AND Parameter BETWEEN 1 AND 64
                      GROUP BY 1,2""").df()
    out = pl.merge(dec, on=["DeviceId", "Detector"], how="left")
    out = out.merge(stg, on=["DeviceId", "Detector"], how="left")
    out["n_on_dec2024"] = out.n_on_dec2024.fillna(0).astype(np.int64)
    out["n_on_staging"] = out.n_on_staging.fillna(0).astype(np.int64)
    out["real_dec2024"] = out.n_on_dec2024 > 0
    out["real_staging"] = out.n_on_staging > 0

    cols = ["DeviceId", "DeviceName", "Detector", "target_type", "target_num", "target",
            "call_phase", "call_overlap", "call_ped", "has_both_phase_and_overlap",
            "additional_call_phases", "additional_call_overlaps", "n_add_phases",
            "n_add_overlaps", "switch_phase", "delay", "extend", "description",
            "n_on_dec2024", "n_on_staging", "real_dec2024", "real_staging"]
    out = out[cols].sort_values(["DeviceId", "Detector"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------- agreement
def agreement(off: pd.DataFrame) -> dict:
    hand = pd.concat([pd.read_parquet(LABELS_DEV), pd.read_parquet(LABELS_TEST)],
                     ignore_index=True)
    hand["Detector"] = hand.Detector.astype(int)
    folds = pd.read_csv(FOLDS_CSV)
    test_ids = set(pd.read_parquet(LABELS_TEST).DeviceId.unique())
    hand["split"] = np.where(hand.DeviceId.isin(test_ids), "TEST", "DEV")

    m = hand.merge(off, on=["DeviceId", "Detector"], how="inner")
    m["agree"] = (m.target_type == "phase") & (m.target_num == m.Phase)
    m["agree_incl_additional"] = m.agree | [
        (t == "phase") and (int(p) in _parse_list(a))
        for t, p, a in zip(m.target_type, m.Phase, m.additional_call_phases)]
    res = {"merged": m,
           "n_hand": len(hand), "n_official": len(off),
           "n_common": len(m),
           "n_agree": int(m.agree.sum()),
           "frac_agree": float(m.agree.mean()),
           "n_agree_incl_add": int(m.agree_incl_additional.sum()),
           "n_hand_only": int(len(hand) - len(m)),
           "n_official_overlap_only": int(((m.target_type == "overlap")).sum())}
    # by phase
    byph = m.groupby("Phase").agg(n=("agree", "size"), agree=("agree", "mean")).reset_index()
    res["by_phase"] = byph
    # by signal
    bysig = m.groupby("DeviceId").agg(n=("agree", "size"), n_dis=("agree", lambda s: int((~s).sum())),
                                      agree=("agree", "mean")).reset_index()
    bysig = bysig.merge(folds, on="DeviceId", how="left")
    bysig["split"] = np.where(bysig.DeviceId.isin(test_ids), "TEST", "DEV")
    res["by_signal"] = bysig.sort_values("agree")
    # wholesale-disagreement signals: >= 3 disagreements and >= 40 % of the signal
    ws = bysig[(bysig.n_dis >= 3) & (bysig.n_dis / bysig.n >= 0.4)]
    res["wholesale"] = ws
    # disagreement detail
    dis = m[~m.agree][["DeviceId", "Detector", "Phase", "Function", "target_type",
                       "target_num", "additional_call_phases", "switch_phase",
                       "delay", "extend", "n_on_dec2024", "n_on_staging", "split"]]
    res["disagreements"] = dis.sort_values(["DeviceId", "Detector"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all")
    a = ap.parse_args()
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    con = connect(threads=10)
    off = build(con)
    off.to_parquet(LABELS_OFFICIAL, index=False)
    log(f"wrote {LABELS_OFFICIAL}: {len(off):,} rows, {off.DeviceId.nunique()} signals")
    log("target_type: " + str(off.target_type.value_counts().to_dict()))
    log(f"real in Dec-2024: {int(off.real_dec2024.sum()):,}; real in staging: "
        f"{int(off.real_staging.sum()):,}; real in either: "
        f"{int((off.real_dec2024 | off.real_staging).sum()):,}")
    r = agreement(off)
    log(f"hand labels {r['n_hand']}, common with official {r['n_common']}, "
        f"agree {r['n_agree']} ({r['frac_agree']:.4f}), "
        f"incl. additional-call phases {r['n_agree_incl_add']}")
    r["disagreements"].to_csv(OFFICIAL / "hand_vs_official_disagreements.csv", index=False)
    r["by_signal"].to_csv(OFFICIAL / "hand_vs_official_by_signal.csv", index=False)
    r["by_phase"].to_csv(OFFICIAL / "hand_vs_official_by_phase.csv", index=False)
    print(r["by_phase"].to_string())
    print("wholesale-disagreement signals:")
    print(r["wholesale"].to_string())


if __name__ == "__main__":
    main()
