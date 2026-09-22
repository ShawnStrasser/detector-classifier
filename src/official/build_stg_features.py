"""Build pair / v2 / similarity features for the Sept-2026 STAGING data.

Runs the *existing* builders (`features.py`, `features_v2.py`, `cross_detector.py`) with
`DC_WORK` re-pointed at `dc_work/official/stg`, so nothing in the Dec-2024 work dir is
read or written.

    python src/official/build_stg_features.py --only base --windows stgmixed
    python src/official/build_stg_features.py --only all  --windows stgall
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import windows_stg  # noqa: E402,F401  (registers the staging window sets)


def run(mod_name: str, out: str, windows: str, threads: int, chunk: int, limit: int) -> None:
    import importlib
    mod = importlib.import_module(mod_name)
    argv = sys.argv
    sys.argv = [mod_name, "--windows", windows, "--out", out,
                "--threads", str(threads), "--chunk", str(chunk)]
    if limit:
        sys.argv += ["--limit", str(limit)]
    try:
        t0 = time.time()
        mod.main()
        print(f"[{mod_name}] {time.time()-t0:.0f}s", flush=True)
    finally:
        sys.argv = argv


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
        run("features", f"pair_features_stg{s}.parquet", a.windows, a.threads, a.chunk, a.limit)
    if a.only in ("all", "v2"):
        run("features_v2", f"pair_features_v2_stg{s}.parquet", a.windows, a.threads, a.chunk, a.limit)
    if a.only in ("all", "sim"):
        run("cross_detector", f"det_similarity_stg{s}.parquet", a.windows, a.threads, a.chunk, a.limit)


if __name__ == "__main__":
    main()
