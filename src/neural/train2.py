"""Stage 13: refit the stage-03 GRU+attention pair scorer on the OFFICIAL labels.

Architecture unchanged (`neural.models.GRUAttn`: strided conv stem, 3-layer BiGRU,
attention pooling -- 853 k parameters).  Training recipe fixes, all reported:

  * phase loss only (the function head is not trained and is not exported);
  * detectors with no actuation inside the window are dropped from the loss -- their
    label is unknowable from that window, so they were pure noise in stage 03;
  * one model for 5 min .. 72 h: the window length is cycled over 5/10/15/20/30/60 min
    and the batch size is set from a constant compute budget, so short windows get big
    batches instead of idling the card;
  * no epoch cap is allowed to stop a run that is still improving: patience-based early
    stopping on an inner validation split of the TRAINING signals, LR halved on plateau;
  * fully resumable -- a fold that already has a `.done.json` is skipped, and an
    interrupted fold restarts from its last-epoch checkpoint.

    python src/neural/train2.py --fold 0
    python src/neural/train2.py --final           # all training signals, no held-out fold
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
from neural import data2 as D2  # noqa: E402
from neural.models import PairNet, n_params  # noqa: E402
from neural.raster import assemble_flat  # noqa: E402

MODELDIR = DC_WORK / "models" / "gru2"
RUNDIR = DC_WORK / "neural" / "runs2"
H = 3600 * 1000

# inner-validation windows: three lengths x four times of day, both periods
ES_SPEC = [(5, 0), (15, 1), (30, 2), (30, 3)]
ES_HOURS = {"dec": [7.5, 36.0, 45.5, 64.75], "stg": [31.5, 20.0, 29.5, 17.0]}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- one batch
def pair_index(dmask: torch.Tensor, ncand: torch.Tensor):
    B, D = dmask.shape
    K = int(ncand.max())
    kk = torch.arange(K, device=dmask.device)
    valid = dmask[:, :, None] & (kk[None, None, :] < ncand[:, None, None])
    bi, di, ki = valid.nonzero(as_tuple=True)
    return bi, di, ki, B, D, K


def forward_batch(model, batch, dev, chunk: int = 0):
    det = batch["det"].to(dev, non_blocking=True)
    ph = batch["ph"].to(dev, non_blocking=True)
    sig = batch["sig"].to(dev, non_blocking=True)
    ncand = batch["ncand"].to(dev)
    dmask = batch["dmask"].to(dev)
    bi, di, ki, B, D, K = pair_index(dmask, ncand)
    x = assemble_flat(det, ph, sig, ncand, bi, di, ki)
    if chunk and x.shape[0] > chunk:
        pl = torch.cat([model(x[i:i + chunk])[0] for i in range(0, x.shape[0], chunk)])
    else:
        pl = model(x)[0]
    logit = torch.full((B, D, K), -1e4, device=dev, dtype=torch.float32)
    logit[bi, di, ki] = pl.float()
    return logit


def phase_loss(logit, batch, dev):
    """Listwise cross-entropy over the signal's candidates, on detectors that actuated."""
    yp = batch["y_phase"].to(dev)
    na = batch["nact"].to(dev)
    ok = (yp >= 0) & (na > 0)
    if not bool(ok.any()):
        return None
    lp = F.log_softmax(logit, dim=2)
    return -lp[ok].gather(1, yp[ok][:, None]).mean()


# ------------------------------------------------------------------ evaluation
def es_plan(table: dict, keys: list[str]) -> list[tuple]:
    plan = []
    for k in keys:
        per = table[k]["period"]
        lo, hi = D2.PERIODS[per]["lo"], D2.PERIODS[per]["hi"]
        for mins, slot in ES_SPEC:
            w0 = int(ES_HOURS[per][slot] * H)
            w0 = max(lo, min(w0, hi - mins * 60 * 1000))
            plan.append((k, w0, mins * 60))
    return plan


@torch.no_grad()
def evaluate(model, table, plan, dev, workers=4, chunk=0, stores=None) -> dict:
    model.eval()
    ds = D2.FixedDataset(table, plan, stores)
    out = {}
    for bidx in D2.batches_of(plan, budget_minutes=240, max_bs=12):
        batch = D2.collate([ds[i] for i in bidx])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit = forward_batch(model, batch, dev, chunk=chunk)
        pred = logit.argmax(2).cpu().numpy()
        yp = batch["y_phase"].numpy()
        na = batch["nact"].numpy()
        T = plan[bidx[0]][2] // 60
        hit, n = out.get(T, (0, 0))
        m = (yp >= 0) & (na > 0)
        hit += int((pred[m] == yp[m]).sum()); n += int(m.sum())
        out[T] = (hit, n)
    acc = {T: h / max(n, 1) for T, (h, n) in sorted(out.items())}
    acc["score"] = float(np.mean(list(acc.values()))) if acc else 0.0
    acc["n"] = int(sum(n for _, n in out.values()))
    return acc


# ------------------------------------------------------------------ train loop
def run(args) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    MODELDIR.mkdir(parents=True, exist_ok=True)
    RUNDIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("gru2_final" if args.fold < 0 else f"gru2_f{args.fold}")
    done = RUNDIR / f"{tag}.done.json"
    if done.exists() and not args.force:
        log(f"{tag}: already finished -- skipping")
        return

    sigs = D2.training_signals()
    if args.fold >= 0:
        tr_sigs = sigs[sigs.fold != args.fold]
    else:
        tr_sigs = sigs
    table = D2.load_table(sigs)                     # every signal (held-out too, for infer)
    keys = [k for k in tr_sigs.key if k in table]
    rng = np.random.default_rng(args.seed + 7)
    perm = rng.permutation(len(keys))
    n_es = max(20, int(args.es_frac * len(keys)))
    es_keys = [keys[i] for i in perm[:n_es]]
    tr_keys = [keys[i] for i in perm[n_es:]]
    log(f"{tag}: train {len(tr_keys)} signals, inner-val {len(es_keys)}, "
        f"{sum(len(table[k]['dets']) for k in tr_keys):,} labelled channels")

    model = PairNet("gru").to(dev)
    log(f"params {n_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.lr / 64)

    ckpt_best = MODELDIR / f"{tag}.pt"
    ckpt_last = MODELDIR / f"{tag}.last.pt"
    best, best_ep, ep0, hist = -1.0, -1, 0, []
    if ckpt_last.exists() and not args.force:
        st = torch.load(ckpt_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, best_ep, ep0, hist = st["best"], st["best_ep"], st["epoch"] + 1, st["hist"]
        log(f"resumed {tag} at epoch {ep0} (best {best:.4f} @ {best_ep})")

    stores = D2.Stores()
    esplan = es_plan(table, es_keys)
    t_train, stopped = sum(h.get("secs", 0.0) for h in hist), False
    for ep in range(ep0, args.epochs):
        plan = D2.sample_plan(table, tr_keys, args.nwin, args.seed, ep)
        batches = D2.batches_of(plan, D2.BATCH_MINUTES, args.max_bs, shuffle_seed=ep)
        ds = D2.PlanDataset(table, plan, max_det=args.max_det, seed=args.seed)
        dl = DataLoader(ds, batch_sampler=batches, num_workers=args.workers,
                        collate_fn=D2.collate, pin_memory=(dev == "cuda"),
                        persistent_workers=False)
        model.train()
        t0 = time.time(); tot = 0.0; nb = 0
        for batch in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logit = forward_batch(model, batch, dev, chunk=args.chunk)
            loss = phase_loss(logit, batch, dev)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        secs = time.time() - t0
        t_train += secs
        acc = evaluate(model, table, esplan, dev, args.workers, args.chunk, stores)
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
            torch.save({"arch": "gru", "kw": {}, "state": model.state_dict(),
                        "epoch": ep, "es_score": score}, ckpt_best)
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "best_ep": best_ep, "hist": hist}, ckpt_last)
        if ep - best_ep >= args.patience:
            log(f"early stop: {args.patience} epochs without improvement")
            stopped = True
            break
    plateaued = bool(stopped)
    info = dict(tag=tag, fold=args.fold, arch="gru", params=n_params(model),
                train_secs=t_train, best_es=best, best_epoch=best_ep,
                epochs_run=len(hist), plateaued=plateaued,
                n_train_signals=len(tr_keys), n_val_signals=len(es_keys),
                train_minutes=list(D2.TRAIN_MINUTES), nwin=args.nwin,
                hist=hist, args=vars(args))
    json.dump(info, open(RUNDIR / f"{tag}.json", "w"), indent=1, default=str)
    json.dump(info, open(done, "w"), indent=1, default=str)
    if not plateaued:
        log(f"WARNING {tag} hit the epoch cap ({args.epochs}) while still improving")
    log(f"{tag}: best es {best:.4f} @ep {best_ep} of {len(hist)}; "
        f"plateaued={plateaued}; {t_train/60:.0f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="-1 = final model on everything")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=45)
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
    a = ap.parse_args()
    if a.final:
        a.fold = -1
    run(a)


if __name__ == "__main__":
    main()
