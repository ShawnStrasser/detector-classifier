"""Track B (stage 15): the greedy keep/drop gate, so the chain runs unattended.

One step at a time.  Each step compares a candidate's **fold-0 held-out blend-before-the-
decoder accuracy at 30 minutes** (the shipped arrangement, weight 0.5) with the current
champion's, and keeps the step only when it wins by at least `--margin` (0.3 pt, the
screening rule in AGENTS.md).  B1 is the exception: the TCN is adopted when it is *within*
the margin, because it trains and runs about twice as fast.

State lives in `%DC_WORK%/trackB/state/`:
    backbone.txt     tcn | gru
    best_args.txt    the training flags of the current champion
    best_seed0.txt   the prediction tag of the champion's seed-0 run
    trail.json       every decision, with the numbers behind it

    python research/code/neural/trackb_decide.py --step b1 --eval b1 \
        --base gru_f0_seed0 --cand tcn_f0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402

TRACKB = DC_WORK / "trackB"
STATE = TRACKB / "state"
EVALDIR = TRACKB / "eval"
METRIC = ("blend_before", "m30")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get(res: dict, name: str) -> float:
    return float(res["candidates"][name][METRIC[0]][METRIC[1]])


def write(p: Path, s: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="ascii")


def trail_append(entry: dict) -> None:
    p = STATE / "trail.json"
    t = json.loads(p.read_text()) if p.exists() else []
    t = [e for e in t if e["step"] != entry["step"]] + [entry]
    p.write_text(json.dumps(t, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--eval", required=True, help="basename under trackB/eval/")
    ap.add_argument("--base", required=True, help="champion's name in that json")
    ap.add_argument("--cand", nargs="+", required=True,
                    help="candidate names; the best of them is the challenger")
    ap.add_argument("--args", nargs="*", default=[],
                    help="training flags each candidate adds, in --cand order")
    ap.add_argument("--tags", nargs="*", default=[],
                    help="prediction tag of each candidate, in --cand order")
    ap.add_argument("--margin", type=float, default=0.003)
    ap.add_argument("--within", action="store_true",
                    help="B1 rule: adopt the candidate when it is within the margin")
    a = ap.parse_args()

    res = json.load(open(EVALDIR / f"{a.eval}.json"))
    base = get(res, a.base)
    scores = {c: get(res, c) for c in a.cand}
    best = max(scores, key=scores.get)
    i = a.cand.index(best)
    delta = scores[best] - base
    keep = (delta >= -a.margin) if a.within else (delta >= a.margin)

    cur_args = (STATE / "best_args.txt").read_text().strip() \
        if (STATE / "best_args.txt").exists() else ""
    new_args = cur_args
    new_tag = (STATE / "best_seed0.txt").read_text().strip() \
        if (STATE / "best_seed0.txt").exists() else a.base
    if keep:
        add = a.args[i] if i < len(a.args) else ""
        new_args = (cur_args + " " + add).strip()
        new_tag = a.tags[i] if i < len(a.tags) else best
        if a.step == "b1":
            write(STATE / "backbone.txt", "tcn" if "tcn" in add else "gru")
    elif a.step == "b1":
        write(STATE / "backbone.txt", "gru")
        new_args = "--arch gru"
        new_tag = a.base
    write(STATE / "best_args.txt", new_args)
    write(STATE / "best_seed0.txt", new_tag)

    entry = dict(step=a.step, base=a.base, base_m30=round(base, 5),
                 scores={k: round(v, 5) for k, v in scores.items()},
                 best=best, delta_pt=round(100 * delta, 3),
                 margin_pt=round(100 * a.margin, 2), rule="within" if a.within else "beat",
                 kept=bool(keep), best_args=new_args, best_seed0=new_tag)
    trail_append(entry)
    log(json.dumps(entry))


if __name__ == "__main__":
    main()
