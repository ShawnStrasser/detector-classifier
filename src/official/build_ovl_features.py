"""Build pair / v2 / similarity features over an OVERLAP cache (pseudo-phases 17..32).

`overlap_cache.patch_load_chunk` widens `features.load_chunk` to phases 1..32 at runtime,
so `features.py`, `features_v2.py` and `cross_detector.py` are used unmodified.

    python src/official/build_ovl_features.py --root <dir> --windows ovl6
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ap0 = argparse.ArgumentParser(add_help=False)
ap0.add_argument("--root", required=True)
_known, _ = ap0.parse_known_args()
ROOT = Path(_known.root)
os.environ["DC_WORK"] = str(ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import windows_stg  # noqa: E402,F401  (registers the staging window sets)
import overlap_cache as OC  # noqa: E402
from features import _w  # noqa: E402

# reduced, representative mixes (the overlap study is exploratory; ~24 labels statewide)
features.WINDOW_SETS["ovl6_dec"] = [
    _w("full72", "2024-12-02 00:00:00", 72 * 3600),
    _w("h6_a", "2024-12-03 06:00:00", 6 * 3600),
    _w("m30_a", "2024-12-02 07:30:00", 1800),
    _w("m30_b", "2024-12-03 12:00:00", 1800),
    _w("m30_c", "2024-12-03 21:30:00", 1800),
    _w("m30_d", "2024-12-04 16:45:00", 1800),
]
features.WINDOW_SETS["ovl6_stg"] = [
    _w("full66", "2026-09-18 16:15:00", 66 * 3600),
    _w("h6_a", "2026-09-20 06:00:00", 6 * 3600),
    _w("m30_a", "2026-09-21 07:30:00", 1800),
    _w("m30_b", "2026-09-19 12:00:00", 1800),
    _w("m30_c", "2026-09-19 21:30:00", 1800),
    _w("m30_d", "2026-09-18 17:00:00", 1800),
]


def run(mod_name: str, out: str, windows: str, threads: int, chunk: int) -> None:
    import importlib
    mod = importlib.import_module(mod_name)
    argv = sys.argv
    sys.argv = [mod_name, "--windows", windows, "--out", out,
                "--threads", str(threads), "--chunk", str(chunk)]
    try:
        t0 = time.time()
        mod.main()
        print(f"[{mod_name}] {time.time()-t0:.0f}s", flush=True)
    finally:
        sys.argv = argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--windows", default="ovl6_dec")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--only", default="all", choices=["all", "base", "v2", "sim"])
    a = ap.parse_args()
    OC.patch_load_chunk(ROOT / "cache")
    (ROOT / "features").mkdir(parents=True, exist_ok=True)
    if a.only in ("all", "base"):
        run("features", "pair_ovl.parquet", a.windows, a.threads, a.chunk)
    if a.only in ("all", "v2"):
        run("features_v2", "pair_ovl_v2.parquet", a.windows, a.threads, a.chunk)
    if a.only in ("all", "sim"):
        run("cross_detector", "det_sim_ovl.parquet", a.windows, a.threads, a.chunk)


if __name__ == "__main__":
    main()
