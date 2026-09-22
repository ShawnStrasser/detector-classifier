"""Staging (Sept-2026) window definitions, mirroring the beta's variant-B 22-window mix.

Data span: 2026-09-18 16:15 (Fri) .. 2026-09-21 10:23 (Mon)  ->  ~66 h, and note that
TWO of the three days are a WEEKEND (Sat 19th, Sun 20th); the Dec-2024 training data is
three weekdays.  Window anchors below therefore deliberately spread over Fri PM peak,
Saturday, Sunday night and Monday AM peak.

Importing this module registers the sets in `features.WINDOW_SETS` so `features.py`,
`features_v2.py` and `cross_detector.py` can be driven with `--windows stgmixed`
without any edit to those files.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features  # noqa: E402
from features import _w  # noqa: E402

FULL = "full66"

WINDOWS_STG_MIXED = [
    # 30 min - Mon AM peak, Sat midday, Sat evening, Fri PM peak
    _w("m30_a", "2026-09-21 07:30:00", 1800),
    _w("m30_b", "2026-09-19 12:00:00", 1800),
    _w("m30_c", "2026-09-19 21:30:00", 1800),
    _w("m30_d", "2026-09-18 17:00:00", 1800),
    # 1 h - Sun PM, Sun deep night, Mon mid-morning
    _w("h1_a", "2026-09-20 17:00:00", 3600),
    _w("h1_b", "2026-09-20 02:00:00", 3600),
    _w("h1_c", "2026-09-21 09:00:00", 3600),
    # 3 h
    _w("h3_a", "2026-09-21 06:00:00", 3 * 3600),
    _w("h3_b", "2026-09-19 14:00:00", 3 * 3600),
    # 6 h
    _w("h6_a", "2026-09-20 06:00:00", 6 * 3600),
    _w("h6_b", "2026-09-19 12:00:00", 6 * 3600),
    # 24 h
    _w("h24_a", "2026-09-19 00:00:00", 24 * 3600),
    _w("h24_b", "2026-09-20 00:00:00", 24 * 3600),
    # full span
    _w(FULL, "2026-09-18 16:15:00", 66 * 3600),
]

WINDOWS_STG_SHORT_B = [
    _w("m5_a", "2026-09-21 07:45:00", 300),
    _w("m5_b", "2026-09-19 12:20:00", 300),
    _w("m5_c", "2026-09-19 22:10:00", 300),
    _w("m5_d", "2026-09-18 17:05:00", 300),
    _w("m10_a", "2026-09-21 08:05:00", 600),
    _w("m10_b", "2026-09-19 13:00:00", 600),
    _w("m10_c", "2026-09-20 02:30:00", 600),
    _w("m10_d", "2026-09-18 17:20:00", 600),
]

# extra 6 h anchors used only for the frozen-beta generalisation test (task 2)
WINDOWS_STG_BETA6H = [
    _w("h6_c", "2026-09-18 17:00:00", 6 * 3600),     # Fri PM peak -> evening
    _w("h6_d", "2026-09-21 04:00:00", 6 * 3600),     # Mon night -> AM peak
]

features.WINDOW_SETS["stgmixed"] = WINDOWS_STG_MIXED
features.WINDOW_SETS["stgshortb"] = WINDOWS_STG_SHORT_B
features.WINDOW_SETS["stgall"] = WINDOWS_STG_MIXED + WINDOWS_STG_SHORT_B
features.DURATION_OF["m5"] = 5.0 / 60.0
features.DURATION_OF["m10"] = 10.0 / 60.0
features.DURATION_OF[FULL] = 66.0

_orig_win_hours = features.win_hours


def win_hours(win: str) -> float:
    if win in features.DURATION_OF:
        return features.DURATION_OF[win]
    return features.DURATION_OF[win.split("_")[0]]


features.win_hours = win_hours

M30 = ["m30_a", "m30_b", "m30_c", "m30_d"]
H6 = ["h6_a", "h6_b", "h6_c", "h6_d"]
