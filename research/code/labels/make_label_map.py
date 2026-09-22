"""Stage 05 task 1: standardise the free-text `Function` strings of the statewide
label file into the model's output space.

    Advance / Presence / Count / Yellow_Red / Other

Writes `src/statewide/function_label_map.csv` (raw_function, std_function, n_rows,
n_signals, confidence, note) for the user to review.  Nothing downstream reads the raw
strings; everything joins through this file.

Conservative rules
------------------
* Only strings that unambiguously name one of the four trained functions are mapped to it.
* "advance presence" is **Other** by explicit user decision (2026-09-17): it is a combined
  detector, not a trained class.
* One/two-letter abbreviations (a, p, co, a*, co*) are mapped to Advance/Presence/Count:
  the same file uses the full words elsewhere and the abbreviation scheme is unambiguous
  (a=advance, p=presence, co=count).  Flagged `medium` confidence, 27 rows in total.
* Vehicle-class qualifiers ("advance cars", "advance trucks") keep the base class.
* Everything else -- bike, mid loop, special, broken, ped, right turn, phase, junk,
  free-text notes -- is Other.

    python src/statewide/make_label_map.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CFG = Path(r"S:\Data_Analysis\Python\detector-classifier\data\statewide_2025-02-25"
           r"\all_configs.csv")
OUT = Path(__file__).resolve().parent / "function_label_map.csv"

# raw string (as it appears, verbatim) -> (standard class, confidence, note)
MAP: dict[str, tuple[str, str, str]] = {
    # ---- the four trained classes, spelled out -----------------------------
    "Advance":   ("Advance",    "high",   "canonical label"),
    "Presence":  ("Presence",   "high",   "canonical label"),
    "Count":     ("Count",      "high",   "canonical label"),
    "Yellow_Red": ("Yellow_Red", "high",
                   "canonical label; statewide-only class, no Dec-2024 training data"),
    # ---- unambiguous abbreviations ----------------------------------------
    "a":  ("Advance",  "medium", "abbreviation of Advance (same file uses a/p/co)"),
    "a ": ("Advance",  "medium",
           "abbreviation of Advance with a trailing space - kept as a separate raw string "
           "so the user can see the whitespace problem in the source file"),
    "a*": ("Advance",  "medium", "abbreviation of Advance; '*' is a local annotation"),
    "p":  ("Presence", "medium", "abbreviation of Presence"),
    "co": ("Count",    "medium", "abbreviation of Count"),
    "co*": ("Count",   "medium", "abbreviation of Count; '*' is a local annotation"),
    # ---- descriptive but unambiguous --------------------------------------
    "stop bar": ("Presence", "high",
                 "stop-bar loop = presence detection at the stop line"),
    "stopbar":  ("Presence", "high", "spelling variant of stop bar"),
    "advance cars":   ("Advance", "high", "advance detector, car-only tuning"),
    "advance trucks": ("Advance", "high", "advance detector, truck-only tuning"),
    "advance truck":  ("Advance", "high", "advance detector, truck-only tuning"),
    # ---- explicitly Other --------------------------------------------------
    "advance presence": ("Other", "high",
                         "USER DECISION 2026-09-17: a combined advance+presence detector "
                         "is NOT one of the trained classes; must come out as Other"),
    "bike":  ("Other", "high", "bicycle detector - not a vehicle function class"),
    "bike zone presence": ("Other", "high", "bicycle detector"),
    "mid loop": ("Other", "high",
                 "mid-block / mid-approach loop; behaviour sits between advance and "
                 "presence and it is not a trained class"),
    "mid":       ("Other", "high", "short form of mid loop"),
    "cll-ll mid": ("Other", "high", "mid loop with a lane annotation"),
    "cl-ll mid":  ("Other", "high", "mid loop with a lane annotation"),
    "rl-crl mid": ("Other", "high", "mid loop with a lane annotation"),
    "rl mid":     ("Other", "high", "mid loop with a lane annotation"),
    "special": ("Other", "high", "agency-specific special function, unspecified"),
    "broken":  ("Other", "high",
                "detector marked broken by the agency - expect no / junk actuations"),
    "phase":   ("Other", "high", "not a detector function (column mis-use)"),
    "rt":          ("Other", "medium",
                    "right-turn detector; the wiring geometry is not one of the three "
                    "classes and the string does not say advance vs presence"),
    "right turn":  ("Other", "medium", "right-turn detector, class unspecified"),
    "left turn":   ("Other", "medium", "left-turn detector, class unspecified"),
    "ped":        ("Other", "high", "pedestrian detector - out of scope"),
    "ped camera": ("Other", "high", "pedestrian detection camera - out of scope"),
    "offramp":    ("Other", "high", "ramp detector, class unspecified"),
    "way far back there": ("Other", "high", "free-text note, not a function"),
    "lp#24": ("Other", "high", "loop identifier, not a function"),
    "lp#25": ("Other", "high", "loop identifier, not a function"),
    "add in phase 2 (8d/25)": ("Other", "high", "maintenance note, not a function"),
    "misc": ("Other", "high", "unspecified"),
    ",": ("Other", "high", "junk / data-entry error"),
    "?": ("Other", "high", "junk / data-entry error"),
}

STD4 = ("Advance", "Presence", "Count", "Other")   # model output space (task 3)


def load_map() -> dict[str, str]:
    """raw Function string -> standard class (Advance/Presence/Count/Yellow_Red/Other)."""
    if OUT.exists():
        m = pd.read_csv(OUT)
        return dict(zip(m.raw_function.astype(str), m.std_function))
    return {k: v[0] for k, v in MAP.items()}


def main() -> None:
    cfg = pd.read_csv(CFG)
    cfg["Function"] = cfg.Function.astype(str)
    g = cfg.groupby("Function").agg(n_rows=("Function", "size"),
                                    n_signals=("DeviceId", "nunique")).reset_index()
    missing = set(g.Function) - set(MAP)
    if missing:
        raise SystemExit(f"unmapped Function strings: {sorted(missing)}")
    g["std_function"] = g.Function.map(lambda s: MAP[s][0])
    g["confidence"] = g.Function.map(lambda s: MAP[s][1])
    g["note"] = g.Function.map(lambda s: MAP[s][2])
    g = g.rename(columns={"Function": "raw_function"})
    g = g.sort_values(["std_function", "n_rows"], ascending=[True, False])
    g = g[["raw_function", "std_function", "n_rows", "n_signals", "confidence", "note"]]
    g.to_csv(OUT, index=False)
    print(g.to_string(index=False))
    print()
    print(g.groupby("std_function").n_rows.sum().to_string())
    print(f"\nwrote {OUT}  ({len(g)} distinct raw strings, {g.n_rows.sum()} label rows)")


if __name__ == "__main__":
    main()
