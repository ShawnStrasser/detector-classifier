"""Task 2b - score the frozen beta's staging predictions against the OFFICIAL labels."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DC_WORK  # noqa: E402
import score_official as S  # noqa: E402
from beta_on_new import EVAL_WINS, OUTDIR, WINDOWS  # noqa: E402

OFFICIAL = DC_WORK / "official"
STG = OFFICIAL / "stg"
pd.set_option("display.width", 220)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="NEW")
    ap.add_argument("--wins", default=",".join(EVAL_WINS))
    ap.add_argument("--out", default="beta_new_results")
    a = ap.parse_args()
    off = S.load_official()
    sig = pd.read_csv(STG / "signals.csv")
    keep = set(sig[sig.group == a.group].DeviceId) if a.group != "ALL" else set(sig.DeviceId)
    off = off[off.DeviceId.isin(keep)]
    res, detail = {}, {}
    for win in a.wins.split(","):
        t0s, t1s = WINDOWS[win]
        cand = S.candidates_in_window(STG / "cache", t0s, t1s)
        cand = cand[cand.DeviceId.isin(keep)]
        for tb in ("notb", "tb"):
            f = OUTDIR / f"beta_{a.group}_{win}_{tb}.parquet"
            if not f.exists():
                continue
            pred = pd.read_parquet(f)
            m = S.prepare(pred, off, cand)
            detail[f"{win}|{tb}"] = m
            r = S.summary(m, f"{win}|{tb}")
            res[f"{win}|{tb}"] = r
            log(f"{win:8s} {tb:5s} acc={r['acc']:.4f} n={r['n_answered_scorable']} "
                f"nonstd={r['acc_nonstd']:.4f} ({r['n_nonstd']}) "
                f"cov_of_real={r['coverage_of_real']:.3f}")
    OUT = OFFICIAL / f"{a.out}.json"
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    log(f"wrote {OUT}")

    # ---- detailed tables on the full window, tiebreak off --------------------
    key = [k for k in detail if k.endswith("|notb") and k.startswith("full")]
    if key:
        m = detail[key[0]]
        print("\n=== coverage vs accuracy (full window, tie-breaker off)")
        print(S.table_coverage(m).round(4).to_string(index=False))
        print("\n=== additional call phases")
        print(S.table_additional(m).round(4).to_string(index=False))
        print("\n=== by DELAY")
        print(S.table_by(m, "delay_bin").round(4).to_string(index=False))
        print("\n=== by EXTEND")
        print(S.table_by(m, "extend_bin").round(4).to_string(index=False))
        print("\n=== error kinds")
        ans = m[(m.target_type == "phase") & m.answered & m.scorable]
        print(ans.pair_kind.value_counts().to_string())
        err = ans[~ans.correct]
        pair = [f"{min(int(t),int(p))}<->{max(int(t),int(p))}"
                for t, p in zip(err.target_num, err.phase_pred)]
        print("\n=== top confusions (official <-> predicted)")
        print(pd.Series(pair).value_counts().head(12).to_string())
        print("\n=== switch_phase channels")
        sw = ans[ans.switch_phase > 0]
        print(f"n={len(sw)} acc={sw.correct.mean():.4f}  "
              f"pred == switch_phase on {int((sw.phase_pred == sw.switch_phase).sum())} rows")
        print("\n=== overlap-target channels (beta cannot answer 'overlap')")
        ov = m[(m.target_type == "overlap") & (m.n_actuations > 0)]
        print(f"n={len(ov)}; answered {int(ov.answered.sum())}; "
              f"their predicted phases: {ov.phase_pred.dropna().astype(int).value_counts().to_dict()}")
        for k, v in detail.items():
            v.to_parquet(OFFICIAL / f"beta_detail_{a.group}_{k.replace('|','_')}.parquet",
                         index=False)


if __name__ == "__main__":
    main()
