# 03_neural — small sequence models on the phase-anonymous (detector, candidate phase) pair
Comparison study, not a shipping candidate: LightGBM on engineered features is the champion.
Code `src/neural/`, weights `dc_work/models/neural/`, predictions `dc_work/preds/neural_*`.

## Input representation (this mattered more than the architecture)
One **1 s raster per (detector, candidate phase) pair**, 30 min = 1800 steps, 9 channels: detector
occupancy + ON-rate; p's green/yellow/red-clearance/call fractions; number-free context = count of
*other* phases green, share of other phases called, coordinated flag. No phase or channel numbers.
Blocks render once per signal-window; pairs assembled **on the GPU**. Loss = softmax over the
signal's candidates (listwise CE); function = 3-way head off the chosen pair. Trained on a **mix of
window lengths** (1-30 min) so one model serves 1 min .. 72 h, pooling by mean log-prob.

## Architectures (train folds 1-4, select on fold 5, fold 0 held out; < 1.4 M params, CPU-ready)
| arch | params | epochs to plateau | train min | CPU ms/signal/30 min | fold-5 30 min | fold-5 72 h | fold-0 72 h |
|---|---|---|---|---|---|---|---|
| **GRU + attention** | 853 k | 77 (+12) | 87 | 612 | **.9348** | .9796 | .9760 |
| TCN (dilated 1-D CNN) | 701 k | 66 (+17) | 49 | 461 | .9311 | .9796 | .9760 |
| conv + GRU hybrid | 955 k | 96 | 88 | 682 | .9362 | .9764 | .9760 |
| Transformer, patch 8 | 926 k | 84 | 42 | 308 | .8988 | .9699 | .9782 |
| Transformer, patch 4 | 1.32 M | 19 (stopped) | 57 | — | .8857 | .9678 | — |
| cycle-raster 2-D CNN | 619 k | 40 | 15 | 88 | .8581 | .9624 | .9673 |

Fold-5 columns are the common-conditions comparison (all six trained identically); fold 0 uses the
corrected cache. All but patch-4 hit a documented plateau (early stop, 10-12 evals without gain, LR
halved on plateau). **At 72 h the four 1-D models sit within 0.2 pt (n=484 — noise); separation is
at short windows** (transformer -3.5 pt, 2-D CNN -7.7 pt). Selected: GRU + attention.

## Best model, DEV 6-fold OOF, 72 h (unscorable excluded; `neural_best_oof_*.parquet`)
| labels | scorable | >=5 actuations | fold 0 | per-fold sd | LightGBM variant B |
|---|---|---|---|---|---|
| hand (`detector-configs`) | **.9763** (n=4,818) | .9767 | .9752 | .0028 | — |
| **official timing plans** | **.9781** (n=4,654) | .9785 | .9782 | .0078 | **.9733** |
| official, 30 min (4 anchors) | **.9646** | .9691 | .9553 | — | **.9540** |

**Caveat that decides the comparison:** the neural files only cover detectors carrying a *hand*
label, so official-label rows are 4,654 of LightGBM's 7,197 — the ~2,500 official-only channels are
absent. Not like-for-like; do not read as beating variant B.
**Accuracy vs sample length** (fold 0, GRU, 4 starts incl. off-peak, scorable): 1 min .883 · 2 min
.912 · 5 min .941 · 10 min .952 · 15 min .955 · 30 min .968 · 1 h .967 · 2 h .971 · 3 h .972 ·
6 h .977 · 12 h .977 · 72 h .981 — as with LightGBM, past ~15 min extra time buys coverage, not
accuracy.

## Why GRU ~= CNN > transformer, in plain English
The evidence is *local in time*: a detector pulse a fraction of a second before a phase call, a
queue discharging just after that phase goes green. A CNN slides a short template along the trace
and a GRU walks it — both are built to notice "these two things happened close together". A
transformer compares every slice with every other: powerful when meaning depends on far-apart
context (as in language), but here mostly wasted capacity — it must *learn* that only nearby pairs
matter, from ~300 signals, and it first chops the trace into patches, blurring the sub-second timing
that carries the signal. The 2-D cycle image did worst: re-cutting by the candidate's own cycles
discards the absolute timing that separates two phases green at the same moment. **LightGBM still
wins overall** because it gets ~200 quantities that already encode the right comparisons (chance-
corrected call linkage, exclusive-green lift, release fraction) computed exactly over the whole
sample, while the net must rediscover them from raw traces with ~5,000 labels; trees also score
every programmed channel, train in minutes and are auditable.

## Honest notes
* **Bug found late:** the 43/44 call stream in my cache was silently truncated at the end of Dec 3,
  so a third of every training window and all Dec-4 evaluation windows had *no phase calls* — the
  most valuable input; Dec-4-anchored windows scored ~14 pt low. After the fix plus an equal
  continuation for every architecture, TCN's fold-5 score went .8993 -> .9270 and 30-min accuracy
  rose ~1 pt. A cache assertion now guards it; the ranking was unchanged.
* Machine rebooted mid-OOF (folds 0-2 survived, fold 3 resumed from checkpoint). OOF folds got 40
  epochs; 0/2/5 hit that cap while flat-but-noisy (+0.0004 from a 2-epoch continuation on fold 0).
  They trained on the broken cache and were only *scored* on the fixed one, so a fresh fit would be
  slightly better. Patch-4 transformer stopped at 19 epochs: it tracked patch-8 exactly at 3-10x the
  cost while thrashing the 8 GB card. Fold 5 was the selection set so is mildly optimistic; fold 0
  never leaked, TEST never touched.
