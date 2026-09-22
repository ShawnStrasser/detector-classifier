# 07_tuning_trees — Optuna, bagging, feature pruning, XGBoost/CatBoost/sklearn (2026-09-21)
DEV 6-fold grouped OOF, variant-B 22-window mix, unscorable excluded (>=1 actuation). **Primary = (72 h + 30 min top-1 phase accuracy)/2.** Searches saw **folds 1-5 only**; fold 0 and TEST untouched. Code `src/tune/*`, `dc_work/tune/`.

## 0. The noise floor — measure this before believing anything
Champion pipeline re-run **6 times changing only the seeds**: primary **.9717 ± .00066** (.9711–.9727) · 72 h **.9799 ±
.00087** (.9793–.9811) · 30 min .9636 ± .00054 · non-standard **.9136 ± .0036** (.9102–.9186). The published
.9809 / .9203 sits at the **top** of that spread — a lucky seed, not a better model. Seed-bagging shrinks it (K=2 sd .00031, K=3 .00027, ~2.4x) for +0.03 pt of mean. **Gains are judged against that 0.066 pt sd as well as fold/bootstrap.**

## 1. Comparison ("decoded" = same champion joint decoder on top; full 22 windows, 6 folds)
| first stage | 72 h | 30 min | primary | non-std | train/fold | infer | model |
|---|---|---|---|---|---|---|---|
| **LightGBM champion** | .9809 | .9634 | **.97218** | .9203 | 40 s | 6.2 µs/row | 1.2 MB |
| LightGBM tuned (56 tr.) / 3-seed bag ² | .9813 / .9802 | .9632 / .9640 | .97226 / .97209 | .9203 / .9144 | 45 s / 3x | 6.2 / 10 µs | 1.2 / 3.9 MB |
| … tuned, 70 of 261 features | .9799 | .9629 | .97141 | .9136 | 22 s | 3.4 µs | 0.7 MB |
| **XGBoost** rank:pairwise (12 tr.) | .9793 | .9595 | **.96938** | .9085 | 145 s | 2.1 µs | 1.4 MB |
| blend LGBM+XGB (mean norm. prob) | **.9824** | .9630 | .97269 | **.9254** | 185 s | — | 2.6 MB |
| CatBoost QuerySoftMax (3 tr.) ¹ | — | — | .9571 | — | 590 s | — | — |
| sklearn HistGB / RandomForest / ExtraTrees ¹ | — | — | .9583 / .9457 / .9396 | — | 62 / 70 / 31 s | — | — |
| **function** champ / tuned+weights | .8203 / .8271 | .8053 | A/P/C **.8682** / .8634 | — | 80 s | — | 2.3 MB |

¹ search protocol only (folds 1-5, 9-window subset, **ranker score before decoding**, where LightGBM scores .9594):
too slow to refit over 22 windows x 6 folds. Each library got the **same 45-min wall-clock box**, not the same trial
count (LightGBM 56, XGBoost 12, CatBoost 3) — the honest caveat on the losers. ² mean of **two independent** 3-seed
bags (sd .00027 vs .00066): the bag buys the halved spread, not the +0.03 pt of mean.
**Verdicts** (paired bootstrap over signals, 90 % interval; folds better of 6): tuned ranker **+0.05 pt**
[-0.08,+0.18] 4/6 **ns** · tuned decoder **-0.00** [-0.15,+0.14] 3/6 **ns** · 70 features **-0.09 pt** decoded 1/6 ·
XGBoost **-0.29 pt** [-0.44,-0.15] 1/6 **really worse** · blend **+0.04 pt** [-0.05,+0.13] 3/6 **ns** · function tuned
trees **+0.04 pt** [-0.18,+0.25] **ns**, but **class weighting +0.66 pt** 5-class [+0.40,+0.92] 6/6 **real — at a cost
of 0.48 pt on Advance/Presence/Count**. **Bagging / pruning:** K = 1/3/5/10 ranker (search protocol)
.9579/.9595/.9594/.9591 (plateau at K=3, only +0.03 pt survives the decoder); function .8047/.8062/.8068/.8068;
coverage at p>=0.9 unchanged. Pruning 261 -> N: 200 .9586 · 100 .9590 · **70 .9589** · 50 .9572 · 25 .9550 · 10 .9382;
leave-one-family-out, the **phase-call events 43/44 cost 10.3 pt** and **every other family <= 0.35 pt** (~100 is the
safe floor).

## 2. What we learned (plain English)
1. **We were measuring noise**: a different random seed alone moves the headline ±0.2 pt at 72 h and ±0.9 pt on
   non-standard detectors — bigger than almost every "gain" chased here.
2. **Tuning bought nothing for phase**: ~60 Optuna trials over 11 knobs landed barely away from the hand-picked
   settings — the champion was already on the flat part of the surface.
3. **The three boosting libraries land close because they are the same idea** — shallow trees on binned features, fitted
   by gradient descent, rediscovering the same rules; sklearn's own histogram booster is within 0.1 pt, and the plain
   forests are 1.4–2 pt behind (no chaining of small corrections, badly calibrated probabilities).
4. **The joint decoder absorbs first-stage differences**, squeezing a +0.16 pt bagging gain and a -0.9 pt XGBoost
   deficit toward the middle: it mostly uses *which other detectors agree*, not the exact score.
5. **Most features are redundant; one is not** — three quarters can go for ~0.1 pt, but without the phase-call events
   the model collapses 10 pt. **Class weighting is a dial, not a win**: it moves accuracy from the three common
   detector functions to the rare ones — a product decision, not a modelling one.

## 3. Recommendation and honest negatives
**Adopt nothing for accuracy**: keep the champion ranker -> joint decoder and the unweighted 5-class function head, and
do **not** add XGBoost or CatBoost (blend +0.04 pt, far below the 0.3 pt bar, and it breaks the numpy-only inference
path). The one defensible change is **3-seed bagging of the pair ranker**, for the halved run-to-run variance rather
than the +0.03 pt. `models/beta_v2_candidate/` is exactly that (3 ranker models + params/features json) plus the
**byte-identical** stage-06 decoder and function head: **6.8 MB** total (was 3.7 MB), 3x tree evaluation on the ranker
only (< 0.2 ms per detector-candidate). `src/lgbm_numpy.py` gained `NumpyBoosterBag` / `predict_average`, reproducing
every model **and the 3-model average exactly** (max abs diff **0.0**), with a new case in
`tests/test_numpy_backend.py` (4/4 pass). If size beats 0.1 pt take the 100-feature ranker. Contract OOF (sum |p-1| =
2e-16): `dc_work/preds/{phase,function}_oof_v3_tuned{,_bywindow}.parquet`. **Negatives:** CatBoost (3 trials)
and XGBoost (12) got far fewer trials than LightGBM (56) in the same wall clock — a floor, not a ceiling; CatBoost
never refit over 22 windows (~100 min) so never entered a blend. TEST untouched.
