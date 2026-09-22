"""Stage 05: shared paths / windows / feature rules for the **6-code** variant.

The statewide 2025-02-25 pull contains only event codes 1, 7, 43, 44, 81, 82.  To get an
honest generalisation number on it we rebuild the whole v2 pipeline from those six codes
only, on the Dec-2024 DEV data with the same 6 folds, and compare.

What the six codes cost us relative to the full allowed set:
  * no 8/9/10/11  -> no yellow vs red-clearance split.  Green is [event 1, event 7), so the
    green bitmask timeline is unchanged; everything after green termination is "red".
  * no 131/150    -> no coordinated/free split.
  * no 83-88      -> no controller fault events for the health flag.
Everything else (ON intervals, cycles, calls, green mask, similarity) is unaffected.

All stage-05 artefacts live under `dc_work/statewide/`; nothing from earlier stages is
overwritten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import CACHE, DC_WORK  # noqa: E402
from train_lgbm_v2 import KEY_EXCLUDE  # noqa: E402

SW = DC_WORK / "statewide"
SW_EVENTS = SW / "events"                  # statewide 2025-02-25, 6 codes, by DeviceId
SW_FEAT = SW / "features"
SW_MODELS = SW / "models"
SW_PREDS = SW / "preds"
for _p in (SW_FEAT, SW_MODELS, SW_PREDS):
    _p.mkdir(parents=True, exist_ok=True)

DEV_EVENTS_GLOB = (CACHE / "events" / "**" / "*.parquet").as_posix()
SW_EVENTS_GLOB = (SW_EVENTS / "**" / "*.parquet").as_posix()

REPO_DATA = Path(r"S:\Data_Analysis\Python\detector-classifier\data")
STATEWIDE_CFG = REPO_DATA / "statewide_2025-02-25" / "all_configs.csv"
TEST_CFG = REPO_DATA / "splits" / "test_config.csv"

CODES6 = (1, 7, 43, 44, 81, 82)

# ---------------------------------------------------------------- feature rules
# columns that are NOT computable from the six codes (or become constant)
BANNED6 = {"f_on_yellow", "f_occ_yellow", "on_lift_green_coord", "on_lift_green_free"}


def feature_cols6(df: pd.DataFrame) -> list[str]:
    """Legal model inputs for the 6-code variant (phase-anonymous, codes 1/7/43/44/81/82)."""
    out = []
    for c in df.columns:
        if c in KEY_EXCLUDE or c.split("__")[0] in BANNED6:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


# ------------------------------------------------------------------- windows
def _w(name, start, secs):
    return {"win": name, "t0": pd.Timestamp(start), "secs": float(secs)}


# statewide day: 2025-02-25 09:00 .. 15:00 (6 h)
SW_T0 = pd.Timestamp("2025-02-25 09:00:00")
WINDOWS_SW = [_w("h6_sw", SW_T0, 6 * 3600)] + \
             [_w(f"m30_{i:02d}", SW_T0 + pd.Timedelta(minutes=30 * i), 1800)
              for i in range(12)] + \
             [_w(f"h1_{i:02d}", SW_T0 + pd.Timedelta(hours=i), 3600) for i in range(6)] + \
             [_w(f"h3_{i:02d}", SW_T0 + pd.Timedelta(hours=3 * i), 3 * 3600) for i in range(2)]


# ------------------------------------------------------------------- splits
def signal_groups() -> pd.DataFrame:
    """DeviceId -> group in {'dev', 'test', 'unseen'} for the statewide labelled signals."""
    cfg = pd.read_csv(STATEWIDE_CFG)
    folds = pd.read_csv(DC_WORK / "folds.csv")
    test = set(pd.read_csv(TEST_CFG).DeviceId.unique())
    dev = dict(zip(folds.DeviceId, folds.fold))
    devs = cfg.DeviceId.drop_duplicates().to_frame()
    devs["group"] = ["test" if d in test else ("dev" if d in dev else "unseen")
                     for d in devs.DeviceId]
    devs["fold"] = [dev.get(d, -1) for d in devs.DeviceId]
    return devs


def statewide_labels() -> pd.DataFrame:
    """Statewide labels with the standardised function class.  TEST signals are dropped
    here and never re-enter: the protocol forbids looking at them."""
    from make_label_map import load_map
    cfg = pd.read_csv(STATEWIDE_CFG)
    m = load_map()
    cfg["Function"] = cfg.Function.astype(str)
    cfg["func_std"] = cfg.Function.map(m)
    cfg = cfg.rename(columns={"Parameter": "Detector"})
    g = signal_groups()
    cfg = cfg.merge(g, on="DeviceId", how="left")
    cfg = cfg[cfg.group != "test"].reset_index(drop=True)
    cfg["Detector"] = cfg.Detector.astype(int)
    cfg["Phase"] = cfg.Phase.astype(int)
    return cfg[["DeviceId", "Detector", "Phase", "Function", "func_std", "group", "fold"]]
