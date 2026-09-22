"""Stage 03: train / evaluate the four pair-scoring architectures.

    python src/neural/train.py --arch tcn --train-folds 1 2 3 4 --es-fold 5 --test-fold 0

Listwise objective: a batch element is one (signal, 30 min window); every labeled
detector of that signal is scored against *all* of the signal's candidate phases and
the logits are softmaxed across candidates (cross-entropy against the labeled phase).
Function is a 3-way head read off the true-phase pair's embedding (top-1 at inference).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DC_WORK  # noqa: E402
from neural.data import WinDataset, collate, devs_of, load_signal_table  # noqa: E402
from neural.models import IS_2D, PairNet, n_params  # noqa: E402
from neural.raster import assemble_cycles_flat, assemble_flat  # noqa: E402

NEURAL = DC_WORK / "neural"
MODELDIR = DC_WORK / "models" / "neural"
DATA_MS = 72 * 3600 * 1000
WIN_MS = 30 * 60 * 1000


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ one batch
def pair_index(dmask: torch.Tensor, ncand: torch.Tensor):
    """Flat (b, d, k) indices of every valid pair in the padded batch."""
    B, D = dmask.shape
    K = int(ncand.max())
    kk = torch.arange(K, device=dmask.device)
    valid = dmask[:, :, None] & (kk[None, None, :] < ncand[:, None, None])
    bi, di, ki = valid.nonzero(as_tuple=True)
    return bi, di, ki, B, D, K


def forward_batch(model, batch, dev, cycles: bool, chunk: int = 0):
    det = batch["det"].to(dev, non_blocking=True)
    ph = batch["ph"].to(dev, non_blocking=True)
    sig = batch["sig"].to(dev, non_blocking=True)
    ncand = batch["ncand"].to(dev)
    dmask = batch["dmask"].to(dev)
    bi, di, ki, B, D, K = pair_index(dmask, ncand)
    if cycles:
        gi = batch["gidx"].to(dev); cl = batch["clen"].to(dev)
        x = assemble_cycles_flat(det, ph, sig, ncand, gi, cl, bi, di, ki)
    else:
        x = assemble_flat(det, ph, sig, ncand, bi, di, ki)
    if chunk and x.shape[0] > chunk:
        outs = [model(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)]
        pl = torch.cat([o[0] for o in outs]); fl = torch.cat([o[1] for o in outs])
    else:
        pl, fl = model(x)[:2]
    logit = torch.full((B, D, K), -1e4, device=dev, dtype=torch.float32)
    logit[bi, di, ki] = pl.float()
    pos = torch.full((B, D, K), -1, device=dev, dtype=torch.long)
    pos[bi, di, ki] = torch.arange(bi.numel(), device=dev)
    return logit, fl.float(), pos


def losses(logit, flog, pos, batch, dev, w_func: float = 0.3):
    yp = batch["y_phase"].to(dev)
    yf = batch["y_func"].to(dev)
    ok = yp >= 0
    lp = F.log_softmax(logit, dim=2)
    l_phase = -lp[ok][torch.arange(int(ok.sum()), device=dev), yp[ok]].mean()
    okf = ok & (yf >= 0)
    if okf.any():
        fp = pos[okf.nonzero(as_tuple=True) + (yp[okf],)]
        l_func = F.cross_entropy(flog[fp], yf[okf])
    else:
        l_func = torch.zeros((), device=dev)
    return l_phase + w_func * l_func, l_phase.detach(), l_func.detach()


# ------------------------------------------------------------------ evaluation
def fixed_windows(devs, table, starts_ms):
    return [(d, int(s), None) for d in devs for s in starts_ms if d in table]


ES_STARTS = [int(x) for x in (
    7.5 * 3600e3, 12 * 3600e3 + 24 * 3600e3, 21.5 * 3600e3 + 24 * 3600e3,
    16.75 * 3600e3 + 48 * 3600e3, 2 * 3600e3 + 24 * 3600e3, 9 * 3600e3 + 48 * 3600e3)]


@torch.no_grad()
def predict(model, table, devs, starts_ms, dev, cycles, workers=4, bs=4,
            max_det=0, chunk=0, agg="logit", T=None):
    """-> dict (device, detector) -> (cand, pooled prob, n_act, per-window hit list)."""
    model.eval()
    ds = WinDataset(table, devs, cycles=cycles, T=T,
                    fixed=fixed_windows(devs, table, starts_ms))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=True)
    acc: dict = {}
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit, _, _ = forward_batch(model, batch, dev, cycles, chunk=chunk)
        lp = F.log_softmax(logit, dim=2).cpu().numpy()
        nact = batch["nact"].numpy()
        for b, (device_id, w0, dets, cand) in enumerate(batch["meta"]):
            k = len(cand)
            for j, d in enumerate(dets):
                key = (device_id, int(d))
                e = acc.get(key)
                if e is None:
                    e = acc[key] = [cand, np.zeros(k), 0.0, 0.0, []]
                v = lp[b, j, :k]
                e[1] += np.exp(v) if agg == "prob" else v
                e[2] += 1.0
                e[3] += float(nact[b, j])
                e[4].append(int(np.argmax(v)))
    return {k: (v[0], v[1] / max(v[2], 1), v[3], v[4]) for k, v in acc.items()}


def top1_acc(pred: dict, table: dict) -> tuple[float, int, float]:
    """(pooled-window accuracy, n, mean single-window accuracy) on scorable detectors.

    Scorable = the true phase is among the signal's candidates.  Detectors with no
    actuations are kept here (they are part of the 30 min reality) but reported
    separately by evaluate.py.
    """
    n = c = 0
    sw_hit = sw_n = 0
    for dev, rec in table.items():
        for j, d in enumerate(rec["dets"]):
            y = rec["y_phase"][j]
            if y < 0:
                continue
            n += 1
            e = pred.get((dev, int(d)))
            if e is None:
                continue
            if int(np.argmax(e[1])) == int(y):
                c += 1
            sw_hit += sum(1 for h in e[3] if h == int(y))
            sw_n += len(e[3])
    return (c / max(n, 1), n, sw_hit / max(sw_n, 1))


@torch.no_grad()
def func_predict(model, table, devs, starts_ms, dev, cycles, workers=4, bs=4, chunk=0):
    """-> dict (device, detector) -> mean 3-way function prob at the top-1 pair."""
    model.eval()
    ds = WinDataset(table, devs, cycles=cycles,
                   fixed=fixed_windows(devs, table, starts_ms))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=True)
    acc: dict = {}
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit, flog, pos = forward_batch(model, batch, dev, cycles, chunk=chunk)
        best = logit.argmax(dim=2)
        B, D = best.shape
        bb = torch.arange(B, device=dev)[:, None].expand(B, D)
        dd = torch.arange(D, device=dev)[None].expand(B, D)
        fp = pos[bb.reshape(-1), dd.reshape(-1), best.reshape(-1)]
        prob = torch.zeros(fp.numel(), 3, device=dev)
        m = fp >= 0
        prob[m] = F.softmax(flog[fp[m]], dim=1)
        prob = prob.reshape(B, D, 3).cpu().numpy()
        for b, (device_id, w0, dets, cand) in enumerate(batch["meta"]):
            for j, d in enumerate(dets):
                key = (device_id, int(d))
                e = acc.setdefault(key, [np.zeros(3), 0.0])
                e[0] += prob[b, j]; e[1] += 1.0
    return {k: v[0] / max(v[1], 1) for k, v in acc.items()}


def parse_kw(s: str) -> dict:
    """"patch=8,d=128" -> {'patch': 8, 'd': 128}  (shell-quoting-proof)."""
    out: dict = {}
    for part in (s or "").replace(";", ",").split(","):
        part = part.strip().strip('"').strip("'")
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


# ------------------------------------------------------------------ train loop
def run(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cycles = args.arch in IS_2D

    tr_devs = devs_of(args.train_folds)
    if args.es_fold < 0:            # OOF mode: hold out a slice of the training signals
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(tr_devs))
        n_es = max(8, int(args.es_frac * len(tr_devs)))
        es_devs = [tr_devs[i] for i in perm[:n_es]]
        tr_devs = [tr_devs[i] for i in perm[n_es:]]
    else:
        es_devs = devs_of([args.es_fold])
    tr_tab = load_signal_table(tr_devs, drop_failed=True)
    es_tab = load_signal_table(es_devs, drop_failed=True)
    tr_devs = [d for d in tr_devs if d in tr_tab]
    es_devs = [d for d in es_devs if d in es_tab]
    log(f"arch={args.arch} train={len(tr_devs)} sig / es={len(es_devs)} sig")

    kw = parse_kw(args.model_kw)
    prev = None
    if args.init_from:
        p = Path(args.init_from)
        if not p.exists():
            p = MODELDIR / f"{args.init_from}.pt"
        prev = torch.load(p, map_location="cpu")
        kw = prev.get("kw", kw)
    model = PairNet(args.arch, **kw).to(dev)
    if prev is not None:
        model.load_state_dict(prev["state"])
        log(f"continuing from {args.init_from} (es {prev.get('es_score')})")
    npar = n_params(model)
    log(f"params: {npar:,}  kw={kw}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = (len(tr_devs) * args.nwin + args.bs - 1) // args.bs
    warm = args.warmup_epochs * steps
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.lr / 64)

    ds = WinDataset(tr_tab, tr_devs, nwin=args.nwin, seed=args.seed,
                    max_det=args.max_det, cycles=cycles, bs=args.bs)
    best, best_ep, hist = -1.0, -1, []
    MODELDIR.mkdir(parents=True, exist_ok=True)
    (NEURAL / "runs").mkdir(parents=True, exist_ok=True)
    ckpt = MODELDIR / f"{args.tag}.pt"
    t_train = 0.0
    gstep = 0
    stopped_early = False
    for ep in range(args.epochs):
        ds.set_epoch(ep)
        dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                        collate_fn=collate, pin_memory=True, drop_last=False)
        model.train()
        t0 = time.time(); tot = np.zeros(3); nb = 0
        for batch in dl:
            if gstep < warm:                       # linear warmup (matters for the ViT-ish transformer)
                for g in opt.param_groups:
                    g["lr"] = args.lr * (gstep + 1) / max(warm, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit, flog, pos = forward_batch(model, batch, dev, cycles,
                                                 chunk=args.chunk)
            loss, lp, lf = losses(logit, flog, pos, batch, dev, args.w_func)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            gstep += 1
            tot += [float(loss), float(lp), float(lf)]; nb += 1
        t_train += time.time() - t0
        if args.eval_every > 1 and (ep % args.eval_every) and ep != args.epochs - 1:
            log(f"ep {ep}: loss {tot[0]/nb:.4f} (no eval) {time.time()-t0:.0f}s")
            continue
        pr = predict(model, es_tab, es_devs, ES_STARTS, dev, cycles,
                     workers=args.workers, bs=args.bs, chunk=args.chunk)
        a_pool, n, a_single = top1_acc(pr, es_tab)
        score = 0.5 * (a_pool + a_single)          # balances 6x30 min pooled vs one 30 min window
        if gstep >= warm:
            sched.step(score)
        lr_now = opt.param_groups[0]["lr"]
        hist.append(dict(epoch=ep, loss=tot[0] / nb, l_phase=tot[1] / nb,
                         l_func=tot[2] / nb, es_pooled=a_pool, es_single=a_single,
                         es_score=score, lr=lr_now))
        log(f"ep {ep}: loss {tot[0]/nb:.4f} (ph {tot[1]/nb:.4f} fn {tot[2]/nb:.4f}) "
            f"es {score:.4f} (pool {a_pool:.4f} / 1win {a_single:.4f}) lr {lr_now:.2e} "
            f"{time.time()-t0:.0f}s")
        if score > best + 1e-5:
            best, best_ep = score, ep
            torch.save({"arch": args.arch, "kw": kw, "state": model.state_dict(),
                        "epoch": ep, "es_score": score}, ckpt)
        if ep - best_ep >= args.patience:
            log(f"early stop: {args.patience} evals without improvement")
            stopped_early = True
            break
    plateaued = stopped_early or (best_ep <= len(hist) - 1 - args.patience // 2)
    log(f"best es {best:.4f} @ep {best_ep} of {len(hist)} epochs; "
        f"plateaued={plateaued}; train time {t_train:.0f}s")

    # ---- held-out fold report ------------------------------------------------
    model.load_state_dict(torch.load(ckpt)["state"])
    res = {}
    if not args.skip_heldout:
        te_devs = devs_of([args.test_fold])
        te_tab = load_signal_table(te_devs, drop_failed=False)
        te_devs = [d for d in te_devs if d in te_tab]
        for name, starts in (("m30", ES_STARTS[:1]), ("full72", None)):
            st = starts if starts is not None else list(range(0, DATA_MS - WIN_MS,
                                                              int(1.5 * 3600e3)))
            pr = predict(model, te_tab, te_devs, st, dev, cycles,
                         workers=args.workers, bs=args.bs, chunk=args.chunk)
            a, n, _ = top1_acc(pr, te_tab)
            res[name] = a
            log(f"fold{args.test_fold} {name}: top1 {a:.4f} (n={n}, {len(st)} windows)")

    json.dump(dict(arch=args.arch, tag=args.tag, params=npar, kw=kw,
                   train_secs=t_train, best_es=best, best_epoch=best_ep,
                   epochs_run=len(hist), plateaued=bool(plateaued),
                   hist=hist, heldout=res, args=vars(args)),
              open(NEURAL / "runs" / f"{args.tag}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="tcn",
                    choices=["tcn", "gru", "transformer", "cyc2d", "convgru"])
    ap.add_argument("--model-kw", default="", help="backbone kwargs, e.g. patch=8,d=128")
    ap.add_argument("--init-from", default="", help="checkpoint tag to continue from")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--lr-patience", type=int, default=3)
    ap.add_argument("--warmup-epochs", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--train-folds", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--es-fold", type=int, default=5,
                    help="-1 = hold out --es-frac of the training signals instead")
    ap.add_argument("--es-frac", type=float, default=0.12)
    ap.add_argument("--skip-heldout", action="store_true")
    ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--nwin", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max-det", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-func", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tag is None:
        args.tag = args.arch
    run(args)


if __name__ == "__main__":
    main()
