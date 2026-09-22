"""Keep whichever of two runs of the same fold scored better on its own ES split."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK  # noqa: E402

RUNS = DC_WORK / "neural" / "runs"
MODELS = DC_WORK / "models" / "neural"


def main(base: str, ext: str):
    jb, je = RUNS / f"{base}.json", RUNS / f"{ext}.json"
    if not je.exists():
        print(f"{ext}: no run json, keeping {base}"); return
    b = json.load(open(jb))["best_es"]
    e = json.load(open(je))["best_es"]
    print(f"{base} es={b:.4f}   {ext} es={e:.4f}", end="  ")
    if e > b:
        shutil.copy(MODELS / f"{ext}.pt", MODELS / f"{base}.pt")
        shutil.copy(je, jb)
        print("-> extension kept")
    else:
        print("-> original kept")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
