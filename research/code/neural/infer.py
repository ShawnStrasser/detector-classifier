"""Stage 03: inference -> protocol-contract prediction files + duration curve.

    python src/neural/infer.py --ckpt tcn --folds 0 --tag neural_tcn_fold0
    python src/neural/infer.py --ckpt tcn --folds 0 --cpu-timing

A pair score is produced for one 30-minute window at a time; for longer analysis
periods the per-window probabilities of the same (detector, candidate) are averaged,
so a single model serves 30 min ... 72 h.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, PREDS  # noqa: E402
from neural.data import WinDataset, collate, devs_of, load_signal_table  # noqa: E402
from neural.models import IS_2D, PairNet  # noqa: E402
from neural.train import forward_batch  # noqa: E402

MODELDIR = DC_WORK / "models" / "neural"
T0 = pd.Timestamp("2024-12-02 00:00:00")
WIN_MS = 30 * 60 * 1000
MAXSUB = 48

# same anchors as src/features.py WINDOWS_MIXED, so stage 01 and 03 curves line up
ANCHORS = [
    ("m30_a", "2024-12-02 07:30:00", 1800), ("m30_b", "2024-12-03 12:00:00", 1800),
    ("m30_c", "2024-12-03 21:30:00", 1800), ("m30_d", "2024-12-04 16:45:00", 1800),
    ("h1_a", "2024-12-02 17:00:00", 3600), ("h1_b", "2024-12-03 02:00:00", 3600),
    ("h1_c", "2024-12-04 09:00:00", 3600),
    ("h3_a", "2024-12-02 06:00:00", 10800), ("h3_b", "2024-12-03 14:00:00", 10800),
    ("h6_a", "2024-12-03 06:00:00", 21600), ("h6_b", "2024-12-04 12:00:00", 21600),
    ("h24_a", "2024-12-02 00:00:00", 86400), ("h24_b", "2024-12-04 00:00:00", 86400),
    ("full72", "2024-12-02 00:00:00", 259200),
]
DURATION_OF = {"m30": 0.5, "h1": 1.0, "h3": 3.0, "h6": 6.0, "h24": 24.0, "full72": 72.0}


def sub_windows(t0: str, secs: int) -> list[int]:
    a = int((pd.Timestamp(t0) - T0).total_seconds() * 1000)
    n = max(1, secs * 1000 // WIN_MS)
    if n <= MAXSUB:
        return [a + i * WIN_MS for i in range(n)]
    return [a + int(round(x)) * WIN_MS for x in np.linspace(0, n - 1, MAXSUB)]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_model(ckpt: str, device: str = "cuda") -> tuple[PairNet, str]:
    p = Path(ckpt)
    if not p.exists():
        p = MODELDIR / f"{ckpt}.pt"
    d = torch.load(p, map_location=device)
    m = PairNet(d["arch"], **d.get("kw", {})).to(device).eval()
    m.load_state_dict(d["state"])
    return m, d["arch"]


@torch.no_grad()
def predict_windows(model, arch, table, devs, starts, device, workers=6, bs=4,
                    chunk=640, agg: str = "prob", T: int | None = None):
    """-> dict (dev, det) -> [cand, score, weight, n_act, sum_funcprob].

    agg: "prob"  = mean softmax over the 30 min sub-windows (old-baseline style)
         "logit" = mean log-prob (geometric mean; the listwise-correct pooling)
         "wprob" = mean softmax weighted by log1p(actuations) in the sub-window
    """
    cycles = arch in IS_2D
    ds = WinDataset(table, devs, cycles=cycles, T=T,
                    fixed=[(d, int(s), None) for d in devs for s in starts if d in table])
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=(device == "cuda"))
    acc: dict = {}
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logit, flog, pos = forward_batch(model, batch, device, cycles, chunk=chunk)
        prob = F.softmax(logit, dim=2) if agg != "logit" else F.log_softmax(logit, dim=2)
        best = logit.argmax(dim=2)
        B, D = best.shape
        bb = torch.arange(B, device=device)[:, None].expand(B, D).reshape(-1)
        dd = torch.arange(D, device=device)[None].expand(B, D).reshape(-1)
        fp = pos[bb, dd, best.reshape(-1)]
        fprob = torch.zeros(fp.numel(), 3, device=device)
        m = fp >= 0
        fprob[m] = F.softmax(flog[fp[m]], dim=1)
        prob = prob.cpu().numpy(); fprob = fprob.reshape(B, D, 3).cpu().numpy()
        nact = batch["nact"].numpy()
        for b, (dev_id, w0, dets, cand) in enumerate(batch["meta"]):
            k = len(cand)
            for j, d in enumerate(dets):
                e = acc.get((dev_id, int(d)))
                if e is None:
                    e = acc[(dev_id, int(d))] = [cand, np.zeros(k), 0.0, 0.0, np.zeros(3)]
                w = float(np.log1p(nact[b, j])) if agg == "wprob" else 1.0
                e[1] += w * prob[b, j, :k]; e[2] += w
                e[3] += float(nact[b, j]); e[4] += fprob[b, j]
    if agg == "logit":       # geometric mean -> back to a probability simplex
        for e in acc.values():
            v = e[1] / max(e[2], 1e-9)
            v = np.exp(v - v.max())
            e[1] = v / v.sum(); e[2] = 1.0
    elif agg == "wprob":
        for e in acc.values():
            if e[2] <= 1e-9:                    # no actuations anywhere: stay uniform
                e[1] = np.ones_like(e[1]); e[2] = float(len(e[1]))
    return acc


def to_frames(acc: dict, with_nact: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, frows = [], []
    for (dev, det), (cand, sp, n, na, sf) in acc.items():
        p = sp / max(n, 1)
        p = p / max(p.sum(), 1e-12)
        for c, pv in zip(cand, p):
            rows.append((dev, det, int(c), float(pv), float(na)))
        f = sf / max(n, 1)
        f = f / max(f.sum(), 1e-12)
        frows.append((dev, det, float(f[0]), float(f[1]), float(f[2]), float(na)))
    ph = pd.DataFrame(rows, columns=["DeviceId", "Detector", "cand_phase", "prob", "n_act"])
    fn = pd.DataFrame(frows, columns=["DeviceId", "Detector",
                                      "p_advance", "p_presence", "p_count", "n_act"])
    if not with_nact:                      # strict contract format
        ph = ph.drop(columns=["n_act"])
        fn = fn.drop(columns=["n_act"])
    return ph, fn


def acc_of(acc: dict, table: dict) -> tuple[float, float, int, int]:
    """(accuracy over all labeled, accuracy over SCORABLE, n_all, n_scorable).

    Scorable (user decision 2026-09-17) = the labeled phase is among the signal's
    candidates in this window AND the detector produced >=1 actuation in it.
    Everything else is "unscorable" and is excluded from the headline number.
    """
    n = c = ns = cs = 0
    for dev, rec in table.items():
        for j, d in enumerate(rec["dets"]):
            y = rec["y_phase"][j]
            n += 1
            e = acc.get((dev, int(d)))
            hit = e is not None and y >= 0 and int(np.argmax(e[1])) == int(y)
            c += hit
            if e is not None and y >= 0 and e[3] > 0:
                ns += 1; cs += hit
    return c / max(n, 1), cs / max(ns, 1), n, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=[0])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=640)
    ap.add_argument("--agg", default="prob", choices=["prob", "logit", "wprob"])
    ap.add_argument("--agg-scan", action="store_true",
                    help="compare the three window-pooling rules on this fold")
    ap.add_argument("--curve", action="store_true", help="accuracy-vs-duration curve")
    ap.add_argument("--anchors", default="",
                    help="comma-separated window prefixes to restrict --curve to "
                         "(e.g. m30,h1,h3,h6); default = all of features.WINDOWS_MIXED")
    ap.add_argument("--cpu-timing", action="store_true")
    ap.add_argument("--minutes-curve", action="store_true")
    args = ap.parse_args()
    tag = args.tag or f"neural_{Path(args.ckpt).stem}_fold{args.folds[0]}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, arch = load_model(args.ckpt, device)

    devs = devs_of(args.folds)
    table = load_signal_table(devs, labeled_only=False, drop_failed=False)
    devs = [d for d in devs if d in table]
    log(f"{arch}: {len(devs)} signals, {sum(len(v['dets']) for v in table.values())} detectors")

    if args.cpu_timing:
        cpu_timing(model, arch, table, devs)
        return
    if args.minutes_curve:
        minutes_curve(model, arch, table, devs, device, args, tag)
        return

    if args.agg_scan:
        st = sub_windows("2024-12-02 00:00:00", 259200)
        for a in ("prob", "logit", "wprob"):
            acc = predict_windows(model, arch, table, devs, st, device, args.workers,
                                  args.bs, args.chunk, agg=a)
            aa, ac, n, nc = acc_of(acc, table)
            log(f"agg={a:6s} 72h: all {aa:.4f} covered {ac:.4f}")
        return

    # ---- contract files at the 72 h setting ---------------------------------
    st = sub_windows("2024-12-02 00:00:00", 259200)
    t0 = time.time()
    acc = predict_windows(model, arch, table, devs, st, device, args.workers,
                          args.bs, args.chunk, agg=args.agg)
    ph, fn = to_frames(acc)
    PREDS.mkdir(parents=True, exist_ok=True)
    ph.to_parquet(PREDS / f"{tag}_phase.parquet", index=False)
    fn.to_parquet(PREDS / f"{tag}_function.parquet", index=False)
    a_all, a_cov, n, nc = acc_of(acc, table)
    log(f"72h: phase acc all {a_all:.4f} (n={n}) / covered {a_cov:.4f} (n={nc}) "
        f"in {time.time()-t0:.0f}s -> {tag}_phase.parquet")

    # ---- accuracy vs duration -----------------------------------------------
    if args.curve:
        rows, bw, bwf = [], [], []
        keep = [s.strip() for s in args.anchors.split(",") if s.strip()]
        anchors = [a for a in ANCHORS
                   if not keep or a[0].split("_")[0] in keep or a[0] in keep]
        for name, t, secs in anchors:
            st = sub_windows(t, secs)
            a = predict_windows(model, arch, table, devs, st, device, args.workers,
                                args.bs, args.chunk, agg=args.agg)
            aa, ac, n, nc = acc_of(a, table)
            dur = DURATION_OF["full72" if name == "full72" else name.split("_")[0]]
            rows.append(dict(win=name, hours=dur, n_sub=len(st), acc_all=aa,
                             acc_cov=ac, n=n, n_cov=nc))
            log(f"  {name:8s} {dur:5.1f}h  all {aa:.4f}  scorable {ac:.4f} (n={nc})")
            p, fq = to_frames(a, with_nact=True)
            p["win"] = name
            fq["win"] = name
            bw.append(p); bwf.append(fq)
        pd.DataFrame(rows).to_csv(DC_WORK / "neural" / f"{tag}_curve.csv", index=False)
        pd.concat(bw).to_parquet(PREDS / f"{tag}_bywindow.parquet", index=False)
        pd.concat(bwf).to_parquet(PREDS / f"{tag}_bywindowfn.parquet", index=False)


MINUTES = [1, 2, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1440, 2880, 4320]
CURVE_ANCHORS = ["2024-12-02 07:30:00", "2024-12-03 12:00:00",
                 "2024-12-03 21:30:00", "2024-12-04 02:00:00"]
DATA_MS = 72 * 3600 * 1000


def minutes_curve(model, arch, table, devs, device, args, tag: str):
    """Accuracy vs analysis-period length, 1 minute .. 72 hours, on this fold."""
    rows = []
    for mins in MINUTES:
        dur = mins * 60 * 1000
        T = min(mins * 60, WIN_MS // 1000)
        got = []
        for a in CURVE_ANCHORS:
            a0 = int((pd.Timestamp(a) - T0).total_seconds() * 1000)
            s = min(a0, DATA_MS - dur)
            if s < 0:
                continue
            starts = [x for x, _ in split_range(s, s + dur, max_chunks=MAXSUB)]
            if (s, len(starts)) in [(g[0], g[1]) for g in got]:
                continue
            acc = predict_windows(model, arch, table, devs, starts, device,
                                  args.workers, args.bs, args.chunk, agg="logit", T=T)
            aa, asc, n, ns = acc_of(acc, table)
            got.append((s, len(starts), aa, asc, ns))
        if not got:
            continue
        rows.append(dict(minutes=mins, n_anchors=len(got), n_sub=got[0][1],
                         acc_all=float(np.mean([g[2] for g in got])),
                         acc_scorable=float(np.mean([g[3] for g in got])),
                         n_scorable=int(np.mean([g[4] for g in got]))))
        log(f"  {mins:5d} min  ({len(got)} starts x {got[0][1]} sub-win)  "
            f"all {rows[-1]['acc_all']:.4f}  scorable {rows[-1]['acc_scorable']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(DC_WORK / "neural" / f"{tag}_minutes_curve.csv", index=False)
    return df


def cpu_timing(model, arch, table, devs, n_sig: int = 10):
    """Wall-clock CPU inference for one signal / one 30 min window."""
    from neural.raster import SignalStore
    torch.set_num_threads(4)
    model = model.cpu().eval()
    store = SignalStore()
    ts = []
    for dv in devs[:n_sig]:
        rec = table[dv]
        t0 = time.time()
        out = predict_signal(model, arch, store, dv, 12 * 3600 * 1000,
                             rec["dets"].tolist())
        ts.append(time.time() - t0)
    ts = np.array(ts[1:])
    log(f"CPU (4 threads) {arch}: {ts.mean()*1000:.0f} ms/signal/30min window "
        f"(median {np.median(ts)*1000:.0f} ms, n={len(ts)})")
    return ts


def split_range(t_start_ms: int, t_end_ms: int, chunk_ms: int = WIN_MS,
                max_chunks: int = 0) -> list[tuple[int, int]]:
    """Cut [start, end) into <=30 min pieces (the model's native window).

    A period shorter than 30 min stays a single short piece -- the models are fully
    convolutional / recurrent, so they accept any number of 1 s steps.
    """
    total = max(int(t_end_ms) - int(t_start_ms), 1000)
    if total <= chunk_ms:
        return [(int(t_start_ms), int(t_end_ms))]
    n = total // chunk_ms
    rem = total - n * chunk_ms
    out = [(int(t_start_ms) + i * chunk_ms, int(t_start_ms) + (i + 1) * chunk_ms)
           for i in range(n)]
    if rem >= 60_000:
        out.append((int(t_start_ms) + n * chunk_ms, int(t_end_ms)))
    if max_chunks and len(out) > max_chunks:
        idx = np.linspace(0, len(out) - 1, max_chunks).round().astype(int)
        out = [out[i] for i in idx]
    return out


@torch.no_grad()
def predict_range(model, arch, store, device_id: str, t_start_ms: int, t_end_ms: int,
                  dets: list[int], max_chunks: int = 0):
    """Score an ARBITRARY [start, end) period (1 minute .. 72 hours).

    Returns (cand_phases, phase_prob [n_det, n_cand], func_prob [n_det, 3],
             n_actuations [n_det]).  Pieces are pooled by the mean log-probability
    (geometric mean), which is the listwise-correct way to combine windows.
    """
    pieces = split_range(t_start_ms, t_end_ms, max_chunks=max_chunks)
    lp_sum = None
    f_sum = None
    nact = None
    for a, b in pieces:
        cand, lp, fp, na = _score_one(model, arch, store, device_id, a, b, dets)
        lp_sum = lp if lp_sum is None else lp_sum + lp
        f_sum = fp if f_sum is None else f_sum + fp
        nact = na if nact is None else nact + na
    p = np.exp(lp_sum - lp_sum.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    f = f_sum / max(len(pieces), 1)
    return cand, p, f / f.sum(axis=1, keepdims=True), nact


@torch.no_grad()
def _score_one(model, arch, store, device_id, a_ms, b_ms, dets):
    from neural.raster import BIN_MS, assemble_cycles_flat, assemble_flat, render
    cycles = arch in IS_2D
    T = max(int((int(b_ms) - int(a_ms)) // BIN_MS), 8)
    r = render(store, device_id, int(a_ms), dets, T=T, cycles=cycles)
    det = torch.from_numpy(r[0])[None]
    ph = torch.from_numpy(r[1])[None]
    sig = torch.from_numpy(r[2])[None]
    D, K = det.shape[1], ph.shape[1]
    ncand = torch.tensor([K])
    bi = torch.zeros(D * K, dtype=torch.long)
    di = torch.arange(D).repeat_interleave(K)
    ki = torch.arange(K).repeat(D)
    if cycles:
        x = assemble_cycles_flat(det, ph, sig, ncand, torch.from_numpy(r[4])[None],
                                 torch.from_numpy(r[5])[None], bi, di, ki)
    else:
        x = assemble_flat(det, ph, sig, ncand, bi, di, ki)
    pl, fl, _ = model(x)
    logit = pl.reshape(D, K)
    lp = F.log_softmax(logit, dim=1).numpy()
    top = logit.argmax(1)
    fprob = F.softmax(fl.reshape(D, K, 3)[torch.arange(D), top], dim=1).numpy()
    return store.cand(device_id), lp, fprob, r[3]


@torch.no_grad()
def predict_signal(model, arch, store, device_id: str, w0_ms: int, dets: list[int]):
    """CPU inference API: -> (cand_phases, prob [n_det, n_cand], func_prob [n_det,3])."""
    from neural.raster import assemble_cycles_flat, assemble_flat, render
    cycles = arch in IS_2D
    r = render(store, device_id, int(w0_ms), dets, cycles=cycles)
    det = torch.from_numpy(r[0])[None]
    ph = torch.from_numpy(r[1])[None]
    sig = torch.from_numpy(r[2])[None]
    D, K = det.shape[1], ph.shape[1]
    ncand = torch.tensor([K])
    bi = torch.zeros(D * K, dtype=torch.long)
    di = torch.arange(D).repeat_interleave(K)
    ki = torch.arange(K).repeat(D)
    if cycles:
        x = assemble_cycles_flat(det, ph, sig, ncand,
                                 torch.from_numpy(r[4])[None], torch.from_numpy(r[5])[None],
                                 bi, di, ki)
    else:
        x = assemble_flat(det, ph, sig, ncand, bi, di, ki)
    pl, fl, _ = model(x)
    logit = pl.reshape(D, K)
    prob = F.softmax(logit, dim=1)
    top = logit.argmax(1)
    fprob = F.softmax(fl.reshape(D, K, 3)[torch.arange(D), top], dim=1)
    return store.cand(device_id), prob.numpy(), fprob.numpy()


if __name__ == "__main__":
    main()
