"""Turn the raw old-baseline outputs into the protocol's prediction files and score them.

Writes dc_work/preds/old_baseline_fold0_phase.parquet and ..._function.parquet from the
3-day run, and prints metrics for both the 3-day run and the Dec-3 09:00-15:00 slice.
"""
import sys

import numpy as np
import polars as pl

WORK = r"C:\Users\hwyr67g\dc_work"
LABELS = WORK + r"\data\raw\detector-configs.csv"
VALID = WORK + r"\data\splits\device_id_valid.csv"
FUNCS = ["Advance", "Count", "Presence"]          # function head class order (alphabetical dummies)
DEFAULT_PHASE = dict(zip(range(1, 41),
                         [1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 1, 3, 5, 6, 6, 6, 6, 6,
                          7, 8, 8, 8, 8, 8, 5, 7, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8]))
CONCURRENT = {frozenset(p) for p in [(2, 6), (4, 8), (1, 5), (3, 7), (1, 6), (2, 5), (3, 8), (4, 7)]}


def load(raw_path):
    raw = pl.read_parquet(raw_path)
    devs = pl.read_csv(VALID)["DeviceId"].to_list()
    lab = (pl.read_csv(LABELS).filter(pl.col("DeviceId").is_in(devs))
           .select(["DeviceId", "Detector", "Phase", "Function"])
           .unique(subset=["DeviceId", "Detector"], keep="first"))
    # labeled detectors with no prediction row at all (no 81/82 events) -> uniform
    missing = lab.join(raw, on=["DeviceId", "Detector"], how="anti").select(["DeviceId", "Detector"])
    if missing.height:
        missing = missing.with_columns(
            [pl.lit(0).alias("n_windows")] +
            [pl.lit(1 / 9).alias(f"p{p}") for p in range(1, 10)] +
            [pl.lit(1 / 3).alias(f"f_{f.lower()}") for f in FUNCS])
        raw = pl.concat([raw, missing.select(raw.columns).cast(raw.schema)])
    return raw, lab, len(devs)


def write_contract(raw, out_phase, out_func):
    long = (raw.select(["DeviceId", "Detector"] + [f"p{p}" for p in range(1, 10)])
            .unpivot(index=["DeviceId", "Detector"], variable_name="cand_phase", value_name="prob")
            .with_columns(pl.col("cand_phase").str.strip_prefix("p").cast(pl.Int32)))
    long = long.with_columns(
        (pl.col("prob") / pl.col("prob").sum().over(["DeviceId", "Detector"])).alias("prob")
    ).sort(["DeviceId", "Detector", "cand_phase"])
    long.write_parquet(out_phase)
    fn = raw.select(["DeviceId", "Detector",
                     pl.col("f_advance").alias("p_advance"),
                     pl.col("f_presence").alias("p_presence"),
                     pl.col("f_count").alias("p_count")]).sort(["DeviceId", "Detector"])
    fn.write_parquet(out_func)
    return long.height, fn.height


def metrics(raw, lab, tag, lines):
    pcols = [f"p{p}" for p in range(1, 10)]
    df = lab.join(raw, on=["DeviceId", "Detector"], how="left")
    P = df.select(pcols).to_numpy()
    nw = df["n_windows"].fill_null(0).to_numpy()
    y = df["Phase"].to_numpy()
    pred = P.argmax(1) + 1
    conf = P.max(1)
    ok = (pred == y) & (nw > 0)                 # no valid window -> counted wrong
    det = df["Detector"].to_numpy()
    fy = df["Function"].to_numpy()
    F = df.select(["f_advance", "f_count", "f_presence"]).to_numpy()
    fpred = np.array(FUNCS)[F.argmax(1)]
    fok = (fpred == fy) & (nw > 0)
    std_phase = np.array([DEFAULT_PHASE.get(int(d), -1) for d in det])
    nonstd = (std_phase != y) | (det > 40)

    lines.append(f"### {tag}")
    lines.append(f"- labeled detectors: {len(y)}; with >=1 valid window: {(nw > 0).sum()}; "
                 f"no valid window (counted wrong): {(nw == 0).sum()}")
    lines.append(f"- **phase acc (all labeled) = {ok.mean():.3f}**; "
                 f"excluding no-window detectors = {ok[nw > 0].mean():.3f}")
    lines.append(f"- standard-wired dets (n={(~nonstd).sum()}): {ok[~nonstd].mean():.3f} | "
                 f"non-standard (n={nonstd.sum()}): {ok[nonstd].mean():.3f} "
                 f"[standard-wiring lookup alone: all {(std_phase == y).mean():.3f}, "
                 f"non-std {(std_phase[nonstd] == y[nonstd]).mean():.3f}]")
    err = ~ok
    e = err & (nw > 0)
    conc = np.array([frozenset((int(a), int(b))) in CONCURRENT for a, b in zip(pred, y)])
    lines.append(f"- errors {err.sum()}: concurrent/opposing pair {int((e & conc).sum())}, "
                 f"other wrong phase {int((e & ~conc).sum())}, no window {int((nw == 0).sum())}")
    top_pairs = {}
    for a, b, bad in zip(y, pred, e):
        if bad:
            top_pairs[f"{a}->{b}"] = top_pairs.get(f"{a}->{b}", 0) + 1
    lines.append("- top confusions (true->pred): " +
                 ", ".join(f"{k} x{v}" for k, v in sorted(top_pairs.items(), key=lambda x: -x[1])[:8]))
    cov = []
    for t in (0.5, 0.7, 0.9):
        m = (conf >= t) & (nw > 0)
        cov.append(f"{t}: cov {m.mean():.3f} acc {ok[m].mean():.3f} (abstain=wrong {(ok & m).mean():.3f})")
    lines.append("- coverage/accuracy — " + "; ".join(cov))
    bins = [(0, 0), (1, 2), (3, 10), (11, 30), (31, 100)]
    lines.append("- acc by #windows — " + "; ".join(
        f"{a}-{b}: n={int(((nw >= a) & (nw <= b)).sum())} acc={ok[(nw >= a) & (nw <= b)].mean():.3f}"
        for a, b in bins if ((nw >= a) & (nw <= b)).sum()))
    # function
    f1s = []
    for c in FUNCS:
        tp = ((fpred == c) & (fy == c) & (nw > 0)).sum()
        fp = ((fpred == c) & (fy != c)).sum() + ((fpred == c) & (nw == 0)).sum()
        fn_ = ((fy == c) & ~((fpred == c) & (nw > 0))).sum()
        f1s.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn_))
    lines.append(f"- **function acc = {fok.mean():.3f}** (excl. no-window {fok[nw > 0].mean():.3f}), "
                 f"macro-F1 = {np.mean(f1s):.3f} (" +
                 ", ".join(f"{c} {f:.2f}" for c, f in zip(FUNCS, f1s)) + ")")
    cm = ["  function confusion (true\\pred " + " ".join(FUNCS) + "):"]
    for c in FUNCS:
        cm.append(f"  {c:<9}" + " ".join(f"{int(((fy == c) & (fpred == p) & (nw > 0)).sum()):>8}" for p in FUNCS) +
                  f"  nowin {int(((fy == c) & (nw == 0)).sum())}")
    lines += cm
    return lines


if __name__ == "__main__":
    out = []
    raw_full, lab, ndev = load(WORK + r"\cache\old_baseline_fold0_raw_full.parquet")
    n1, n2 = write_contract(raw_full, WORK + r"\preds\old_baseline_fold0_phase.parquet",
                            WORK + r"\preds\old_baseline_fold0_function.parquet")
    out.append(f"signals={ndev}, labeled detectors={lab.height}, phase rows={n1}, function rows={n2}")
    metrics(raw_full, lab, "Dec 2-4 2024, <=20 windows/detector/day", out)
    raw_6h, lab6, _ = load(WORK + r"\cache\old_baseline_fold0_raw_6h.parquet")
    metrics(raw_6h, lab6, "Dec 3 09:00-15:00 only (mirrors Feb-2025 statewide run)", out)
    txt = "\n".join(out)
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)
    open(WORK + r"\cache\old_baseline_metrics.txt", "w", encoding="utf-8").write(txt)
