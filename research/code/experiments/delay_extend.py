"""Task 5 - can `delay` and `extend` be INFERRED from the event log, and does that help
the confidence estimate?

Physics.  Hi-res 81/82 are logged AFTER the controller's delay/extend processing, so:

* `extend` stretches every ON by E seconds after the loop physically clears, therefore the
  SHORTEST observed ON of a channel cannot be much below E -- the distribution of ON
  durations is shifted right and the "chatter" band below ~0.3 s disappears;
* `delay` suppresses any presence shorter than D entirely and shortens what is left
  (logged duration = physical - D + E), so a delayed channel actuates less often than its
  siblings and never shows the short arrivals they show.

Estimators tried (label-free, so they work at inference time for any agency):
    est_extend = 1st percentile of ON duration (clipped at 0)
    est_delay  = how far the channel's ON-rate sits below the busiest sibling on the same
                 predicted phase, expressed in seconds of suppressed presence:
                 log-ratio of its short-ON share to the signal's median short-ON share.

The OFFICIAL delay / extend values are NEVER model inputs (they are not available at
inference time elsewhere); they are used only to score these estimators.

    python src/official/delay_extend.py --step estimate
    python src/official/delay_extend.py --step trust
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, connect  # noqa: E402

OFFICIAL = DC_WORK / "official"
STG = OFFICIAL / "stg"
pd.set_option("display.width", 220)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def estimate(cache: Path, out: Path) -> pd.DataFrame:
    con = connect(threads=8)
    con.execute(f"SET temp_directory='{(STG/'tmp').as_posix()}'")
    df = con.sql(f"""
        WITH iv AS (SELECT * FROM read_parquet('{(cache/'det_intervals.parquet').as_posix()}')),
        a AS (
          SELECT DeviceId, Detector::INT AS Detector, count(*) AS n_on,
                 min(dur) AS dur_min,
                 quantile_cont(dur, 0.01) AS dur_q01,
                 quantile_cont(dur, 0.05) AS dur_q05,
                 quantile_cont(dur, 0.25) AS dur_q25,
                 median(dur) AS dur_med,
                 avg(CASE WHEN dur < 0.3 THEN 1 ELSE 0 END) AS frac_lt03,
                 avg(CASE WHEN dur < 1.0 THEN 1 ELSE 0 END) AS frac_lt1,
                 avg(CASE WHEN dur < 2.0 THEN 1 ELSE 0 END) AS frac_lt2
          FROM iv WHERE dur > 0 GROUP BY 1,2)
        SELECT * FROM a WHERE n_on >= 20""").df()
    con.close()
    # signal-level context: the busiest / shortest-ON sibling defines "normal" here
    g = df.groupby("DeviceId")
    df["sig_med_frac_lt03"] = g.frac_lt03.transform("median")
    df["sig_med_q01"] = g.dur_q01.transform("median")
    df["sig_max_n_on"] = g.n_on.transform("max")
    df["est_extend"] = df.dur_q01.clip(lower=0)
    df["est_extend_rel"] = (df.dur_q01 - df.sig_med_q01).clip(lower=0)
    df["short_deficit"] = np.log1p(df.sig_med_frac_lt03) - np.log1p(df.frac_lt03)
    df["rate_deficit"] = -np.log((df.n_on / df.sig_max_n_on).clip(lower=1e-3))
    df["est_delay_score"] = df.short_deficit.clip(lower=0) * 3.0 + \
        0.5 * df.rate_deficit.clip(lower=0)
    df.to_parquet(out, index=False)
    log(f"wrote {out}: {len(df):,} channels")
    return df


def score(df: pd.DataFrame) -> dict:
    off = pd.read_parquet(OFFICIAL / "labels_official.parquet")
    off["Detector"] = off.Detector.astype(int)
    m = df.merge(off[["DeviceId", "Detector", "delay", "extend", "target_type"]],
                 on=["DeviceId", "Detector"], how="inner")
    res = {"n": len(m)}
    print(f"\n=== {len(m):,} channels with >=20 ONs and an official programming row")
    print("\n--- observed 1st-percentile ON duration vs programmed EXTEND")
    t = m.groupby(pd.cut(m.extend, [-.1, .001, 1.0, 2.0, 3.0, 5.0, 100],
                         labels=["0", "(0,1]", "(1,2]", "(2,3]", "(3,5]", ">5"]),
                  observed=True).agg(n=("dur_q01", "size"),
                                     med_q01=("dur_q01", "median"),
                                     med_min=("dur_min", "median"),
                                     med_frac_lt03=("frac_lt03", "median"))
    print(t.round(3).to_string())
    e = m[m.extend > 0]
    res["corr_extend_q01"] = float(np.corrcoef(m.extend, m.dur_q01)[0, 1])
    res["spearman_extend_q01"] = float(m[["extend", "dur_q01"]].corr("spearman").iloc[0, 1])
    res["mae_est_extend"] = float((m.est_extend - m.extend).abs().median())
    res["mae_est_extend_nonzero"] = float((e.est_extend - e.extend).abs().median()) if len(e) else np.nan
    print(f"\ncorr(extend, dur_q01) = {res['corr_extend_q01']:.3f}  "
          f"spearman {res['spearman_extend_q01']:.3f}; "
          f"median |est-extend - extend| = {res['mae_est_extend']:.2f} s "
          f"({res['mae_est_extend_nonzero']:.2f} s on the {len(e)} channels with extend > 0)")
    # detection of "extend is set at all"
    from sklearn.metrics import roc_auc_score
    y = (m.extend > 0).astype(int)
    res["auc_extend_set"] = float(roc_auc_score(y, m.dur_q01))
    print(f"AUC for 'extend is set' from dur_q01 alone: {res['auc_extend_set']:.3f} "
          f"(base rate {y.mean():.3f})")

    print("\n--- programmed DELAY vs the label-free deficit score")
    t2 = m.groupby(pd.cut(m.delay, [-.1, .001, 5.001, 100],
                          labels=["0", "(0,5]", ">5"]), observed=True).agg(
        n=("est_delay_score", "size"), med_score=("est_delay_score", "median"),
        med_frac_lt03=("frac_lt03", "median"), med_n_on=("n_on", "median"),
        med_q01=("dur_q01", "median"))
    print(t2.round(3).to_string())
    yd = (m.delay > 0).astype(int)
    res["auc_delay_set"] = float(roc_auc_score(yd, m.est_delay_score))
    res["auc_delay_set_fracshort"] = float(roc_auc_score(yd, -m.frac_lt03))
    print(f"AUC for 'delay is set': deficit score {res['auc_delay_set']:.3f}, "
          f"short-ON share alone {res['auc_delay_set_fracshort']:.3f} "
          f"(base rate {yd.mean():.3f})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(STG / "cache"))
    ap.add_argument("--out", default=str(OFFICIAL / "delay_extend_est_stg.parquet"))
    a = ap.parse_args()
    p = Path(a.out)
    df = estimate(Path(a.cache), p) if not p.exists() else pd.read_parquet(p)
    r = score(df)
    json.dump(r, open(OFFICIAL / "delay_extend_estimators.json", "w"), indent=1)


if __name__ == "__main__":
    main()
