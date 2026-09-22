"""Stage 03: collect run JSONs / curve CSVs into the comparison tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402

RUNS = DC_WORK / "neural" / "runs"


def main(tags: list[str]):
    rows = []
    for t in tags:
        p = RUNS / f"{t}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        rows.append(dict(arch=d["arch"], params=d["params"],
                         train_min=round(d["train_secs"] / 60, 1),
                         epochs=len(d["hist"]), best_epoch=d["best_epoch"],
                         es_acc=round(d["best_es"], 4),
                         fold0_30min=round(d["heldout"].get("m30", float("nan")), 4),
                         fold0_72h=round(d["heldout"].get("full72", float("nan")), 4)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    for t in tags:
        c = DC_WORK / "neural" / f"neural_{t}_fold0_curve.csv"
        if c.exists():
            print(f"--- {t} duration curve ---")
            print(pd.read_csv(c).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["tcn", "gru", "transformer", "cyc2d"])
