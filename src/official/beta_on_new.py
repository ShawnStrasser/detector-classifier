"""Task 2 - ONE-SHOT honest test of the FROZEN beta (`models/beta_v0`, unchanged) on
signals that are in NEITHER DEV nor TEST, using the Sept-2026 staging data and the
OFFICIAL labels.

Nothing here is tuned; the models are loaded read-only from the repo folder.

    python src/official/beta_on_new.py --step predict --group NEW
    python src/official/beta_on_new.py --step predict --group DEV      # in-sample check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DC_WORK, connect  # noqa: E402
import predict as P  # noqa: E402
import windows_stg as W  # noqa: E402

STG = DC_WORK / "official" / "stg"
EVENTS = (STG / "cache" / "events" / "**" / "*.parquet").as_posix()
OUTDIR = DC_WORK / "official" / "preds_beta"

# window name -> (start, end) as timestamp strings
WINDOWS = {}
for _w in (W.WINDOWS_STG_MIXED + W.WINDOWS_STG_BETA6H):
    _t0 = _w["t0"]
    WINDOWS[_w["win"]] = (str(_t0), str(_t0 + pd.Timedelta(seconds=_w["secs"])))

EVAL_WINS = [W.FULL, "m30_a", "m30_b", "m30_c", "m30_d",
             "h6_a", "h6_b", "h6_c", "h6_d"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def signals(group: str) -> list[str]:
    s = pd.read_csv(STG / "signals.csv")
    return sorted(s[s.group == group].DeviceId) if group != "ALL" else sorted(s.DeviceId)


def _tb_variant(res: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the phase columns of `res` from tie-broken pair probabilities.

    The ODOT tie-breaker is pure post-processing on the (detector, candidate) probability
    table, and in the beta the function head reads the PRE-tiebreak probabilities, so both
    variants can be produced from a single pass of the model.
    """
    import tiebreak as TB
    ph2, sw = TB.apply_odot_tiebreak(ph, enabled=True, return_flags=True)
    p = ph2.sort_values(["DeviceId", "Detector", "prob"], ascending=[True, True, False])
    g = p.groupby(["DeviceId", "Detector"], sort=False)
    a = g.head(1).rename(columns={"cand_phase": "p1", "prob": "q1"})
    b = g.nth(1).rename(columns={"cand_phase": "p2", "prob": "q2"})
    out = res.copy()
    out = out.merge(a[["DeviceId", "Detector", "p1", "q1"]], on=["DeviceId", "Detector"],
                    how="left")
    out = out.merge(b[["DeviceId", "Detector", "p2", "q2"]], on=["DeviceId", "Detector"],
                    how="left")
    ansd = out.phase_pred.notna()
    out["phase_guess"] = out.p1.astype("Int64")
    out["phase_guess_prob"] = out.q1
    out["phase_pred"] = out.p1.where(ansd).astype("Int64")
    out["phase_prob"] = out.q1.where(ansd)
    out["phase_2nd"] = out.p2.where(ansd).astype("Int64")
    out["phase_2nd_prob"] = out.q2.where(ansd)
    out["phase_margin"] = (out.phase_prob.fillna(0) - out.phase_2nd_prob.fillna(0)).where(ansd)
    swk = set(sw.DeviceId.astype(str) + "|" + sw.Detector.astype(str)) if len(sw) else set()
    out["tiebreak_applied"] = (out.DeviceId.astype(str) + "|" +
                               out.Detector.astype(str)).isin(swk) & ansd
    return out.drop(columns=["p1", "q1", "p2", "q2"])


def run_window(devs: list[str], win: str, tiebreak: bool, chunk: int,
               threads: int, group: str) -> Path:
    """Runs the frozen beta once and writes BOTH the tie-breaker-off and -on outputs."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = "tb" if tiebreak else "notb"
    out = OUTDIR / f"beta_{group}_{win}_{tag}.parquet"
    out_tb = OUTDIR / f"beta_{group}_{win}_tb.parquet"
    if out.exists() and out_tb.exists():
        log(f"skip {out.name} (exists)")
        return out
    t0s, t1s = WINDOWS[win]
    parts = []
    t0 = time.time()
    parts_tb = []
    stash: dict = {}
    import tiebreak as TB
    _orig = TB.apply_odot_tiebreak

    def _capture(phase_probs_df, enabled=False, return_flags=False, **kw):
        stash["ph"] = phase_probs_df.copy()
        return _orig(phase_probs_df, enabled=enabled, return_flags=return_flags, **kw)

    P.apply_odot_tiebreak = _capture
    con = connect(threads=threads)
    con.execute(f"SET temp_directory='{(STG/'tmp').as_posix()}'")
    try:
        for i in range(0, len(devs), chunk):
            ch = devs[i:i + chunk]
            ids = ",".join("'" + d + "'" for d in ch)
            # the cache is hive-partitioned on DeviceId, so this reads only these signals
            ev = con.sql(f"""SELECT DeviceId, Timestamp, EventId::INT AS EventId,
                                    Parameter::INT AS Parameter
                             FROM read_parquet('{EVENTS}', hive_partitioning=true)
                             WHERE DeviceId IN ({ids})
                               AND Timestamp >= TIMESTAMP '{t0s}'
                               AND Timestamp <  TIMESTAMP '{t1s}'""").df()
            if not len(ev):
                continue
            stash.pop("ph", None)
            r = P.predict(ev, odot_tiebreak=False, threads=threads, memory="6GB")
            del ev
            if len(r):
                parts.append(r)
                if "ph" in stash and len(stash["ph"]):
                    parts_tb.append(_tb_variant(r, stash["ph"]))
                else:
                    parts_tb.append(r)
            if (i // chunk) % 10 == 0:
                log(f"  {win}: {i+len(ch)}/{len(devs)} signals, {time.time()-t0:.0f}s")
    finally:
        con.close()
        P.apply_odot_tiebreak = _orig
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    res.to_parquet(OUTDIR / f"beta_{group}_{win}_notb.parquet", index=False)
    rtb = pd.concat(parts_tb, ignore_index=True) if parts_tb else pd.DataFrame()
    rtb.to_parquet(out_tb, index=False)
    n_sw = int(rtb.tiebreak_applied.sum()) if len(rtb) else 0
    log(f"wrote {win}: {len(res)} rows, tie-breaker fired on {n_sw}, {time.time()-t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="NEW", choices=["NEW", "DEV", "ALL"])
    ap.add_argument("--wins", default=",".join(EVAL_WINS))
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tiebreak", default="both", choices=["both", "off", "on"])
    a = ap.parse_args()
    devs = signals(a.group)
    log(f"{a.group}: {len(devs)} signals")
    for win in a.wins.split(","):
        run_window(devs, win, False, a.chunk, a.threads, a.group)


if __name__ == "__main__":
    main()
