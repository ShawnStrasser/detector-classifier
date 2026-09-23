"""Track B (stage 15): the stage-13 pair-scorer recipe with four knobs bolted on.

Everything else is `neural/train2.py` unchanged and imported from there -- the same
`data2` pipeline, the same official labels, the same 709 training signals, the same
signal-grouped folds, phase loss only, 5-30 min windows at a constant compute budget,
patience-based early stopping on an inner split of the TRAINING signals.  The knobs:

  B1  --arch tcn|gru      the backbone (dilated 1-D CNN instead of the BiGRU)
  B3  --dropout P         head dropout + dropout inside the recurrent / conv blocks
      --chdrop P          channel dropout: with probability P a training signal has its
                          three number-free context channels (other-phases-green,
                          other-phases-called, coordinated) zeroed for the whole window
      --jitter SECS       random +-SECS shift of the window start
  B4  --minutes 5,10,...  the cycle of training window lengths (5 min .. 6 h)
      --det-budget N      detectors x seconds per sample, so a step costs the same
                          whatever the window length (0 = off; 21600 = 12 dets at 30 min)

Artefacts go to `%DC_WORK%/trackB/` so nothing stage 13 wrote is ever touched.  Fully
resumable: a tag with a `.done.json` is skipped, an interrupted tag restarts from
`.last.pt`.

    python research/code/neural/trackb_train.py --tag tb_b1_tcn_f0 --arch tcn --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK  # noqa: E402
from neural import data2 as D2  # noqa: E402
from neural import train2 as T2  # noqa: E402
from neural.raster import assemble_flat  # noqa: E402
from neural.trackb_models import PairNetB, n_params  # noqa: E402

TRACKB = DC_WORK / "trackB"
MODELDIR = TRACKB / "models"
RUNDIR = TRACKB / "runs"
H = 3600 * 1000


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------- dataset with augmentation
class TrackBDataset(D2.PlanDataset):
    """`PlanDataset` plus window-start jitter and a length-aware detector budget."""

    def __init__(self, table, plan, stores=None, max_det: int = 12, seed: int = 0,
                 jitter: int = 0, det_budget: int = 0):
        super().__init__(table, plan, stores, max_det, seed)
        self.jitter = int(jitter)
        self.det_budget = int(det_budget)

    def __getitem__(self, i):
        from neural.raster import render
        key, w0, T = self.plan[i]
        rec = self.table[key]
        rng = np.random.default_rng((self.seed, i, w0))
        if self.jitter:
            lo, hi = D2.PERIODS[rec["period"]]["lo"], D2.PERIODS[rec["period"]]["hi"]
            off = int(rng.integers(-self.jitter, self.jitter + 1)) * 1000
            w0 = int(np.clip(w0 + off, lo, max(lo, hi - T * 1000)))
        md = self.max_det
        if self.det_budget:
            md = int(np.clip(self.det_budget // max(int(T), 1), 2, self.max_det))
        n = len(rec["dets"])
        if md and n > md:
            idx = rng.permutation(n)[:md]
            idx.sort()
        else:
            idx = np.arange(n)
        dets = rec["dets"][idx].tolist()
        st = self._st().get(rec["period"])
        det, ph, sig, nact = render(st, rec["dev"], int(w0), dets, T=int(T))
        return dict(det=det, ph=ph, sig=sig, nact=nact,
                    y_phase=rec["y_phase"][idx], ncand=len(rec["cand"]),
                    key=key, dev=rec["dev"], w0=int(w0),
                    dets=np.asarray(dets, np.int32), cand=rec["cand"])


# ---------------------------------------------------------------------- one batch
def forward_batch(model, batch, dev, chunk: int = 0, chdrop: float = 0.0):
    """`train2.forward_batch` with optional context-channel dropout (training only)."""
    det = batch["det"].to(dev, non_blocking=True)
    ph = batch["ph"].to(dev, non_blocking=True)
    sig = batch["sig"].to(dev, non_blocking=True)
    ncand = batch["ncand"].to(dev)
    dmask = batch["dmask"].to(dev)
    bi, di, ki, B, D, K = T2.pair_index(dmask, ncand)
    x = assemble_flat(det, ph, sig, ncand, bi, di, ki)
    if chdrop > 0.0:
        keep = (torch.rand(B, device=x.device) >= chdrop).to(x.dtype)
        x = torch.cat([x[:, :6], x[:, 6:9] * keep[bi].view(-1, 1, 1)], dim=1)
    if chunk and x.shape[0] > chunk:
        pl = torch.cat([model(x[i:i + chunk])[0] for i in range(0, x.shape[0], chunk)])
    else:
        pl = model(x)[0]
    logit = torch.full((B, D, K), -1e4, device=dev, dtype=torch.float32)
    logit[bi, di, ki] = pl.float()
    return logit


# ---------------------------------------------------------------------- train loop
def run(args) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    MODELDIR.mkdir(parents=True, exist_ok=True)
    RUNDIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    done = RUNDIR / f"{tag}.done.json"
    if done.exists() and not args.force:
        log(f"{tag}: already finished -- skipping")
        return
    minutes = tuple(int(x) for x in args.minutes.split(",")) if args.minutes \
        else D2.TRAIN_MINUTES

    sigs = D2.training_signals()
    tr_sigs = sigs if args.fold < 0 else sigs[sigs.fold != args.fold]
    table = D2.load_table(sigs)
    keys = [k for k in tr_sigs.key if k in table]
    rng = np.random.default_rng(args.seed + 7)
    perm = rng.permutation(len(keys))
    n_es = max(20, int(args.es_frac * len(keys)))
    es_keys = [keys[i] for i in perm[:n_es]]
    tr_keys = [keys[i] for i in perm[n_es:]]
    log(f"{tag}: arch={args.arch} dropout={args.dropout} chdrop={args.chdrop} "
        f"jitter={args.jitter} minutes={minutes} det_budget={args.det_budget}")
    log(f"{tag}: train {len(tr_keys)} signals, inner-val {len(es_keys)}, "
        f"{sum(len(table[k]['dets']) for k in tr_keys):,} labelled channels")

    model = PairNetB(args.arch, dropout=args.dropout).to(dev)
    log(f"params {n_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.lr / 64)

    ckpt_best = MODELDIR / f"{tag}.pt"
    ckpt_last = MODELDIR / f"{tag}.last.pt"
    best, best_ep, ep0, hist = -1.0, -1, 0, []
    if ckpt_last.exists() and not args.force:
        st = torch.load(ckpt_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, best_ep, ep0, hist = st["best"], st["best_ep"], st["epoch"] + 1, st["hist"]
        log(f"resumed {tag} at epoch {ep0} (best {best:.4f} @ {best_ep})")

    stores = D2.Stores()
    esplan = T2.es_plan(table, es_keys)
    t_train, stopped = sum(h.get("secs", 0.0) for h in hist), False
    for ep in range(ep0, args.epochs):
        plan = D2.sample_plan(table, tr_keys, args.nwin, args.seed, ep, minutes)
        batches = D2.batches_of(plan, D2.BATCH_MINUTES, args.max_bs, shuffle_seed=ep)
        ds = TrackBDataset(table, plan, max_det=args.max_det, seed=args.seed,
                           jitter=args.jitter, det_budget=args.det_budget)
        dl = DataLoader(ds, batch_sampler=batches, num_workers=args.workers,
                        collate_fn=D2.collate, pin_memory=(dev == "cuda"),
                        persistent_workers=False)
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for batch in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = forward_batch(model, batch, dev, chunk=args.chunk,
                                      chdrop=args.chdrop)
            loss = T2.phase_loss(logit, batch, dev)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        secs = time.time() - t0
        t_train += secs
        acc = T2.evaluate(model, table, esplan, dev, args.workers, args.chunk, stores)
        score = acc["score"]
        sched.step(score)
        lr_now = opt.param_groups[0]["lr"]
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), secs=secs, lr=lr_now,
                         **{f"acc_m{k}": v for k, v in acc.items() if isinstance(k, int)},
                         es_score=score))
        log(f"ep {ep}: loss {tot/max(nb,1):.4f} es {score:.4f} "
            + " ".join(f"m{k}={v:.4f}" for k, v in acc.items() if isinstance(k, int))
            + f" lr {lr_now:.2e} {secs:.0f}s ({nb} steps)")
        if score > best + 1e-5:
            best, best_ep = score, ep
            torch.save({"arch": args.arch, "kw": {}, "state": model.state_dict(),
                        "epoch": ep, "es_score": score, "dropout": args.dropout}, ckpt_best)
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "best_ep": best_ep, "hist": hist}, ckpt_last)
        if ep - best_ep >= args.patience:
            log(f"early stop: {args.patience} epochs without improvement")
            stopped = True
            break
    info = dict(tag=tag, fold=args.fold, arch=args.arch, params=n_params(model),
                train_secs=t_train, best_es=best, best_epoch=best_ep,
                epochs_run=len(hist), plateaued=bool(stopped),
                n_train_signals=len(tr_keys), n_val_signals=len(es_keys),
                train_minutes=list(minutes), nwin=args.nwin, hist=hist, args=vars(args))
    json.dump(info, open(RUNDIR / f"{tag}.json", "w"), indent=1, default=str)
    json.dump(info, open(done, "w"), indent=1, default=str)
    if not stopped:
        log(f"WARNING {tag} hit the epoch cap ({args.epochs}) while still improving")
    log(f"{tag}: best es {best:.4f} @ep {best_ep} of {len(hist)}; "
        f"plateaued={stopped}; {t_train/60:.0f} min train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default="tcn", choices=["tcn", "gru"])
    ap.add_argument("--fold", type=int, default=0, help="-1 = final model on everything")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--chdrop", type=float, default=0.0)
    ap.add_argument("--jitter", type=int, default=0, help="seconds")
    ap.add_argument("--minutes", default=None, help="e.g. 5,10,15,30,30,60,120,360")
    ap.add_argument("--det-budget", type=int, default=0, dest="det_budget")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--nwin", type=int, default=2)
    ap.add_argument("--max-bs", type=int, default=8)
    ap.add_argument("--max-det", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr-patience", type=int, default=2)
    ap.add_argument("--es-frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
