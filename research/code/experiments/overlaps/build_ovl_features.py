"""Build pair / v2 / similarity features over an OVERLAP cache (pseudo-phases 17..32).

`overlap_cache.patch_load_chunk` widens `build_features.load_chunk` to phases 1..32
at runtime, so the shipped feature builders are used unmodified.

    python src/official/build_ovl_features.py --root <dir> --windows ovl6
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ap0 = argparse.ArgumentParser(add_help=False)
ap0.add_argument("--root")
_known, _ = ap0.parse_known_args()
ROOT = Path(_known.root or ".")
os.environ["DC_WORK"] = str(ROOT)

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
import build_features as bf  # noqa: E402
import windows_stg  # noqa: E402,F401  (registers the staging window sets)
import overlap_cache as OC  # noqa: E402
from build_features import _w  # noqa: E402

# reduced, representative mixes (the overlap study is exploratory; ~24 labels statewide)
bf.WINDOW_SETS["ovl6_dec"] = [
    _w("full72", "2024-12-02 00:00:00", 72 * 3600),
    _w("h6_a", "2024-12-03 06:00:00", 6 * 3600),
    _w("m30_a", "2024-12-02 07:30:00", 1800),
    _w("m30_b", "2024-12-03 12:00:00", 1800),
    _w("m30_c", "2024-12-03 21:30:00", 1800),
    _w("m30_d", "2024-12-04 16:45:00", 1800),
]
bf.WINDOW_SETS["ovl6_stg"] = [
    _w("full66", "2026-09-18 16:15:00", 66 * 3600),
    _w("h6_a", "2026-09-20 06:00:00", 6 * 3600),
    _w("m30_a", "2026-09-21 07:30:00", 1800),
    _w("m30_b", "2026-09-19 12:00:00", 1800),
    _w("m30_c", "2026-09-19 21:30:00", 1800),
    _w("m30_d", "2026-09-18 17:00:00", 1800),
]


def run(what: str, out: str, windows: str, threads: int, chunk: int,
        limit: int = 0) -> None:
    t0 = time.time()
    bf.build(what, out, "det_lag_unused.parquet", windows, threads, chunk, limit)
    print(f"[{what}] {time.time()-t0:.0f}s", flush=True)


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
        run("base", "pair_ovl.parquet", a.windows, a.threads, a.chunk)
    if a.only in ("all", "v2"):
        run("extra", "pair_ovl_v2.parquet", a.windows, a.threads, a.chunk)
    if a.only in ("all", "sim"):
        run("sim", "det_sim_ovl.parquet", a.windows, a.threads, a.chunk)


if __name__ == "__main__":
    main()
