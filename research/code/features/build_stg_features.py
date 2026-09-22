"""Build pair / v2 / similarity features for the Sept-2026 STAGING data.

Runs the shipped feature builders through `build_features.build()` with
`DC_WORK` re-pointed at `dc_work/official/stg`, so nothing in the Dec-2024 work dir is
read or written.

    python research/code/features/build_stg_features.py --only base --windows stgmixed
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

STG = Path(os.environ.get("DC_WORK_STG",
                          str(Path.home() / "dc_work" / "official" / "stg")))
os.environ["DC_WORK"] = str(STG)

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
import build_features as bf  # noqa: E402
import windows_stg  # noqa: E402,F401  (registers the staging window sets)


def run(what: str, out: str, windows: str, threads: int, chunk: int,
        limit: int = 0) -> None:
    t0 = time.time()
    bf.build(what, out, "det_lag_unused.parquet", windows, threads, chunk, limit)
    print(f"[{what}] {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "base", "v2", "sim"])
    ap.add_argument("--windows", default="stgall")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    s = a.suffix
    if a.only in ("all", "base"):
        run("base", f"pair_features_stg{s}.parquet", a.windows, a.threads, a.chunk, a.limit)
    if a.only in ("all", "v2"):
        run("extra", f"pair_features_v2_stg{s}.parquet", a.windows, a.threads, a.chunk, a.limit)
    if a.only in ("all", "sim"):
        run("sim", f"det_similarity_stg{s}.parquet", a.windows, a.threads, a.chunk, a.limit)


if __name__ == "__main__":
    main()
