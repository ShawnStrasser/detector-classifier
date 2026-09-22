# 12_final_model — the shipped model, the blend check, and the single final scoring (2026-09-21)

Nothing was tuned after any number below was looked at. Code `src/official/{fit_final_v1,score_final,
curve_final,blend_check}.py`, `src/predict.py`, `src/plot_final_accuracy.py`; artefacts
`models/final_v1/`, `dc_work/official/final_v1/`, `dc_work/preds/final_test_DO_NOT_USE/`.

## 1. `models/final_v1` — what was frozen
Stage-10 variant C (DEV Dec-2024 + 334 NEWTRAIN Sept-2026 signals, **official timing labels**, the
beta's variant-B 22-window mix from 5 min to the full span) with **3-seed bagging of the pair ranker**
(stage 07), the stage-06 joint decoder re-fitted on this pool's out-of-fold first-stage probabilities,
and the stage-11 5-class function head copied in unchanged. 701 signals, 1,539,251 labelled
detector-window rows, 261 ranker features. Ranker seeds 797 / 413 / 601 trees, decoder 796 trees;
18.4 MB of text models. `src/lgbm_numpy.py` reproduces every model **and the 3-model average exactly**
(max abs diff 0.0). The bagged ranker is averaged as *per-detector normalised probabilities*, the same
averaging `predict.py` performs, so the OOF numbers describe the shipped pipeline.

| DEV 6-fold OOF, official labels, 7,197 detectors at 72 h | 72 h | 6 h | 30 min | non-std | cov@.9 / acc | fold 0 |
|---|---|---|---|---|---|---|
| ranker bag only | .9711 | .9629 | .9407 | .9262 | .915 / .9897 | .9636 |
| **+ joint decoder (ship)** | **.9748** | **.9723** | **.9585** | .9241 | **.957 / .9877** | .9776 |
| in-sample (same rows, the fitted models) | .9904 | .9880 | .9805 | .9690 | .996 | .9762 |
| stage-10 variant C, single seed (reference) | .9735 | — | .9577 | .9220 | .960 / .9877 | .9720 |

Ranker single-seed spread over the 3 seeds: primary .95514 / .95488 / .95471, **sd 0.02 pt**; stage 10's
three decoded single-seed runs gave sd 0.02 pt. In-sample − OOF = **1.6 pt**, normal for boosted trees.
The Sept-2026 half of the same pool scores .9858 at 66 h / .9669 at 30 min.

## 2. Blend with the neural model — not shipped now, but the strongest open lead
GRU OOF (`dc_work/preds/neural_best_oof_{phase,bywindow}.parquet`, stage 03) vs the final LightGBM OOF
on the rows common to both — the neural files only cover **hand**-labelled channels, so 4,654 of the
7,197 officially-labelled detectors (351 signals); **not like-for-like with §1 or §3**. Official
labels, probabilities renormalised per detector.

| window | LightGBM | GRU | 50/50 blend | gain |
|---|---|---|---|---|
| full 72 h | **.9820** | .9781 | .9839 | +0.19 pt |
| 30 min, mean of 4 anchors | .9654 | .9646 | **.9749** | **+0.94 pt** |
| 30 min, per anchor (a/b/c/d) | .9641/.9690/.9567/.9719 | .9689/.9679/.9546/.9669 | .9753/.9792/.9676/.9774 | +0.6 to +1.1 pt |

The weight is **not fitted** for the 30-min rows (a plain 0.5/0.5 average), and the gain is consistent
across all four anchors. Where the weight *was* searched (full window, folds 1-5) it landed at 0.35
LightGBM / 0.65 GRU, and on fold 0, which never saw the search, the blend gained +0.22 pt.
**Reading:** at long windows both models are near the label-noise ceiling and there is nothing to
gain; at 30 minutes their errors are genuinely different and averaging them removes a quarter of them.
**Not shipped in this release** — the pipeline was already frozen and exam-scored, and inference would
need torch — but it clears the 0.5 pt bar at short windows and is the best remaining idea.
GRU caveats carried from stage 03: its fold models were fitted on a cache whose 43/44 call stream was
truncated for one of the three days (fixed before scoring) and 3 of 6 folds stopped at an epoch cap,
so a fresh fit would be slightly *better*, not worse. `dc_work/official/final_v1/blend_check.json`.

## 3. Final scoring — once, after freezing
Every number is a full end-to-end `src/predict.py` run on raw events (no cached features). The scoring
run was made with the ODOT tie-breaker ON so that both columns could be read from one pass (it only
ever exchanges the top two candidates' probabilities, so the OFF answer is recovered exactly by
swapping them back); **the shipped `predict.py` no longer contains the tie-breaker at all** — the
"tb on" column below is what it *would* have given, and is why it was dropped. The frozen beta was run
on **identical rows** through a copy of its own entry point and models; both have since been removed
from the repo (restore `src/predict_beta_v0.py` and `models/beta_v0/` from git history to reproduce
that arm — its predictions are already saved under `dc_work/preds/final_test_DO_NOT_USE/`).
Headline = detectors with >=5 actuations whose official phase turns green in the window.

| set | window | **final** | beta | final non-std | beta non-std | final err / concurrent | beta err |
|---|---|---|---|---|---|---|---|
| **143 NEWTEST**, Sept-2026 (2,348 labels) | full 66 h | **.9862** | .9698 | **.9734** | .9268 | 32 / 22 | 70 |
| | 6 h (3 anchors) | **.9819** | .9678 | .9618 | .9281 | — | — |
| | 30 min (4 anchors) | **.9704** | .9567 | .9304 | .9022 | 60.5 / 41 | 89 |
| **43 TEST**, Sept-2026 (941) | full 66 h | **.9785** | .9742 | .9453 | .9219 | 20 / 6 | 24 |
| | 30 min | **.9709** | .9608 | .9224 | .8956 | — | — |
| **43 TEST**, Dec-2024 (834) | full 72 h | .9683 | **.9732** | .9040 | .8983 | 26 / 9 | 22 |
| | 30 min | **.9665** | .9594 | .8999 | .8800 | — | — |

Coverage: 99.0-99.4 % of labelled channels get an answer at the full window, 90.7-93.1 % at 30 min.
`phase_prob >= 0.9` keeps 96.0 % at **.9946** (NEWTEST full), 90.4 % at .9953 at 30 min. The
alternative refusal rule (`--min-actuations 1 --min-prob 0.9`) answers 88.2 % at **.9952** at 30 min
where the min-5 rule answers 93.1 % at .9704 — fewer answers, 2.5 pt better. The ODOT tie-breaker would
have added +0.05 pt at the full window and +0.1 to +0.5 pt at 30 min — not enough to keep the only
component that ever looked at a phase number, so it was **removed from the shipped model**; refusing on
confidence covers the same close calls better.
Function (5-class / A-P-C): NEWTEST .754 / .780 (n=321) · TEST-Sept .779 / .800 (729) ·
TEST-Dec .823 / .855 (711); beta .520 / .709, .657 / .781, .733 / .850. Yellow_Red precision
.88-1.00 at recall .71-.84 (n = 17-24 per set); Other recall .67-.69 vs the beta's .09-.29.

## 4. What is worth saying out loud
1. **The final model makes less than half the beta's phase errors on the biggest clean set** (32 vs 70)
   and gains most where it matters: non-standard wiring **+4.7 pt**.
2. **It lost once**: the 43 TEST signals at the full Dec-2024 window, .9683 vs .9732 — 26 errors vs 22,
   a four-detector difference on 821, inside either model's noise. Reported, not explained away.
3. The beta's published TEST figure (.9872 at 72 h) was against **hand** labels; against the official
   timing on the same signals it is .9732, which is the comparison above. Both models look worse on
   Dec-2024 TEST than on the Sept-2026 sets because those 43 signals carry the noisiest labels we have.
4. **Yellow_Red and Other are the whole function story.** The beta cannot name a Yellow_Red at all, and
   recalls Other at .09-.29; the trained classes are what moves 5-class accuracy +7 to +23 pt.

**Negatives / limits.** Function is measured on only 321 labelled channels on NEWTEST. The Sept-2026
window is two-thirds weekend. The decoder is a single seed (only the ranker is bagged). The in-sample
fold-0 figure (.9762) sits just below its OOF one (.9776) because the decoder was trained on OOF
first-stage probabilities and sees sharper ones in-sample — harmless, but worth knowing. Per-detector
test predictions live only in `dc_work/preds/final_test_DO_NOT_USE/` and were not inspected row by row.
