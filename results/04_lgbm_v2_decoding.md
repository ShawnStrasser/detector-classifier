# 04_lgbm_v2_decoding — error forensics, v2 features, joint per-signal decoding (2026-09-17)

DEV 6-fold grouped OOF, TEST untouched. "classifiable" = >=1 actuation (4873 of 5171). 30 min = mean of 4 windows.

| step | 72 h classif. | all labelled | 30 min | 1 h | non-std 72 h | errors | concurrent | cov@0.9 / acc |
|---|---|---|---|---|---|---|---|---|
| stage 01 ranker | 0.9637 | 0.9081 | 0.9316 | 0.9286 | 0.8816 | 177 | 80 | 0.922 / 0.9842 |
| + v2 pair features | 0.9653 | — | 0.9332 | 0.9294 | 0.8849 | 169 | 74 | 0.922 / 0.9842 |
| + decoding, similarity only | 0.9659 | — | 0.9384 | — | **0.8882** | 166 | 69 | 0.949 / 0.9814 |
| + decoding, adjacency only | 0.9682 | — | 0.9503 | — | 0.8816 | 155 | 61 | 0.952 / 0.9823 |
| **+ decoding, both (champion)** | **0.9696** | **0.9137** | 0.9520 | 0.9497 | 0.8816 | **148** | **58** | 0.955 / 0.9826 |
| + ODOT tie-breaker (opt-in) | 0.9700 | — | **0.9560** | **0.9546** | 0.8816 | 147 | 57 | — |
| function 3-class | **0.8951** | 0.8693 | | | macro-F1 0.864 | | stage 01: 0.8841 | |

Fold 0 (2025 hold-out) 0.9714 classifiable / 0.9406 all vs the BiLSTM's 0.9468 / 0.9188; decoding fixed 42 stage-01
errors and broke 13. Duration curve .9560/.9546/.9685/.9700/.9712/.9700 at 30 min/1 h/3 h/6 h/24 h/72 h; by
actuations <10 -> .890, 10-25 -> .930, 100-250 -> .962, >5 k -> .973 (decoding helps most where evidence is thin).
## 1. Forensics on stage 01's 177 classifiable errors
* **51 are unwinnable**: the labelled phase never turns green in 72 h (12 labelled phase 9) — config errors. Of the
  126 fixable: 67 concurrent pair (2<->6 = 36), 20 same-ring, 39 other; only **1** is low volume (wiring, not volume).
* Raw evidence almost always backs the model. At `ca10c12f…` channels 8, 9 (labelled 4) actuate in phase 2's green
  97.8 % of the time and in phase 4's green 0.1 %, and channels 22, 23 (labelled 8) sit on phase 6; phase 4 runs 1460
  cycles, so a phase-4 detector cannot be silent in its own green — the labels match the table, not the field.
* **Renumbering is rare and is pair swaps, not rotations: 6 signals** (`renumber_signals_v2.json`), cleanest is
  `d3bd786b…`, phases 4 and 7 swapped for every detector. **Label-noise ceiling**: if every confident (p>=0.8)
  disagreement is a label error, classifiable accuracy could reach **0.991** (0.986 at p>=0.9) — at 0.9696, ~2 pt
  is real model headroom and the rest is data. Of the 104 confident disagreements left after v2, 54 have a *label*
  equal to the standard-table phase and only 11 a *prediction* that does.
## 2. Features v2 — small gain, honest (`features_v2.py`)
Partner geometry (max-green-overlap candidate, found generically), exclusive-green lifts with evidence counts and
shrinkage, sub-second green-start bins, per-cycle first-ON latency, queue-release lag, red-restricted / finely
binned event-43 linkage, partner-difference (`__pdiff`) columns. **+0.18 pt at 72 h, +0.08 at 30 min.**
Leave-one-family-out from full v2 (72 h / 30 min, `ablations_v2.json`): partner_excl .9649/.9327, fine_timing
.9637/.9327, release .9637/.9324, calls_v2 .9649/**.9302**, partner_geom .9633/.9328, pdiff .9639/.9328; full v2
.9653/.9331, stage 01 alone .9635/.9323 — redundant with each other: each costs ~0.2 pt when removed, yet adding
any one alone to stage 01 changes nothing (several were slightly negative alone).
## 3. Joint decoding — the real win (`cross_detector.py`, `decode_v2.py`)
Second-stage LightGBM over the first-stage OOF probabilities with (a) **actuation-similarity neighbours** — phi
correlation of "channel active" in 2 s bins, so same-approach detectors that see the same vehicles cluster even
when two phases are always green together — (b) **channel-adjacency neighbours** (|dch| <= 2), plus signal-level
"who else claims this phase" and "this phase has vehicle calls but owns no detector". A **binary** second stage
beat a ranker one (0.9696 vs 0.9680) and is far better calibrated (coverage at p>=0.9: 0.955 vs 0.800); the control
with neither neighbour family scores 0.9647, so the gain is the structure, not the extra stage. **Similarity alone
is the only variant that improves non-standard detectors (0.8816 -> 0.8882), adjacency gives more overall but
nothing there** — non-ODOT agencies should use `decode_sim_only`.
## 4. ODOT tie-breaker (`src/tiebreak.py`, default OFF)
Fires only when the top-2 are a concurrent pair inside `margin`, the signal's confident predictions agree with the
standard table at least `standardness_min` of the time, and the channel's standard phase is the runner-up. Nested CV
over the short windows picks margin 0.5 / standardness 0.7 in all 6 folds: **150 switches, 143 helped, 6 hurt — and
all 6 harmed detectors are non-standard ones.** At 72 h it fires 1-4 times in 4873 detectors, so the benefit is all
on short windows (+0.4 pt at 30 min and 1 h, +0.04 at 72 h); safer setting margin 0.25 / standardness 0.85.
## 5. What we learned (plain English)
1. Most of what is left is **not model error** — it is detector configs copied from the standard wiring table and
   never checked in the field; 104 such detectors are listed for the user.
2. Two phases that are **always green together cannot be separated from one detector alone**, but they can be by
   *which other detectors it fires with*: cars on the same approach arrive in the same second. Biggest idea here.
3. Channel adjacency is a strong extra hint but an ODOT habit — it does nothing for the non-standard detectors we
   care most about, so it is measured and shipped separately. New pair features gave almost nothing.
4. Ranking models give good orderings but meaningless confidences; a plain binary model gives both. For function,
   describing a detector **relative to its siblings on the same predicted phase** was worth +1.1 pt.
## 6. Outputs and honest negatives
Code `src/{error_forensics,features_v2,cross_detector,train_lgbm_v2,decode_v2,function_v2,tiebreak,report_v2,predict}.py`;
models `dc_work/models/{phase_lgbm_v2,decode_lgbm_v2,function_lgbm_v2}.*`; preds `dc_work/preds/phase_oof_v2*`,
`function_oof_v2.parquet`, `{curves,ablations,decode_v2_results,tiebreak}_v2.json`, `errors_stage01_categorised.csv`,
**`review_list_v2.csv`** (467: 104 confident disagreements, 14 stubborn, 296 dead, 6 renumbered signals). `predict.py`
on 2 DEV signals x 30 min: **2.5 s wall (0.7 s after I/O)**, CPU only, <1 GB; 50/50 phase and 40/50 function correct
(in-sample for the DEV-wide final models, so indicative only). Negatives: v2 features were a near-miss for the
effort; the champion decoder leaves non-standard accuracy unchanged at 0.8816; temperature scaling of the function
head did nothing (ECE .0133 -> .0137); the "Other" threshold (0.6) is unvalidated as DEV has no Other labels (the
statewide 6-code CSV was left for later); all-5171 accuracy is still capped at 0.9137 by 298 dead channels.
