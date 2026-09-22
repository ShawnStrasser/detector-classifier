"""Stage 06 step 0 -- build the pair/similarity features for the SHORT windows.

The stage-04 cache (`pair_features_windows.parquet`, `pair_features_v2_extra.parquet`,
`det_similarity.parquet`) covers 14 windows, the shortest being 30 min.  The beta
(`results/05_beta_package.md`, variant B) additionally trained on 4x5 min and 4x10 min
windows, and stage 06 has to report function accuracy at 5 min, so those eight windows are
built here into *separate* files (`*_B.parquet`); nothing from stage 04 is overwritten.

    python src/function_v3_prep.py            # all three files
    python src/function_v3_prep.py --only base
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features  # noqa: E402
from features import _w  # noqa: E402

# 4 x 5 min and 4 x 10 min, spread over the three days and over the time of day
WINDOWS_SHORT_B = [
    _w("m5_a", "2024-12-02 07:45:00", 300),
    _w("m5_b", "2024-12-03 12:20:00", 300),
    _w("m5_c", "2024-12-03 22:10:00", 300),
    _w("m5_d", "2024-12-04 17:05:00", 300),
    _w("m10_a", "2024-12-02 08:05:00", 600),
    _w("m10_b", "2024-12-03 13:00:00", 600),
    _w("m10_c", "2024-12-04 02:30:00", 600),
    _w("m10_d", "2024-12-04 17:20:00", 600),
]

# register so features.py / features_v2.py / cross_detector.py can all be driven with it
features.WINDOW_SETS["shortb"] = WINDOWS_SHORT_B
features.DURATION_OF["m5"] = 5.0 / 60.0
features.DURATION_OF["m10"] = 10.0 / 60.0


def run(mod_name: str, out: str, threads: int, chunk: int) -> None:
    import importlib
    mod = importlib.import_module(mod_name)
    argv = sys.argv
    sys.argv = [mod_name, "--windows", "shortb", "--out", out,
                "--threads", str(threads), "--chunk", str(chunk)]
    try:
        t0 = time.time()
        mod.main()
        print(f"[{mod_name}] {time.time()-t0:.0f}s", flush=True)
    finally:
        sys.argv = argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "base", "v2", "sim"])
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.only in ("all", "base"):
        run("features", "pair_features_windows_B.parquet", a.threads, a.chunk)
    if a.only in ("all", "v2"):
        run("features_v2", "pair_features_v2_extra_B.parquet", a.threads, a.chunk)
    if a.only in ("all", "sim"):
        run("cross_detector", "det_similarity_B.parquet", a.threads, a.chunk)


if __name__ == "__main__":
    main()
