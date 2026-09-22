# 10_official_labels — official timing labels, a clean test on 478 never-seen signals, overlaps, delay/extend (2026-09-21)

**Label source.** PHASE truth everywhere below is now the controllers' timing database
(`data/detector_plans.parquet`: `call_phase`, else `call_overlap`) → `dc_work/official/labels_official.parquet`
(35,818 programmed channels / 945 signals; 17,289 have >=1 actuation in Dec-2024 or Sept-2026).
Hand `detector-configs.csv` phase labels are used **only** for the agreement report and as the historical row (A).
Function labels are untouched here. Code `src/official/*`, artefacts `dc_work/official/`.

## 1. Splits (signals / programmed channels / channels that actuate)
| split | signals | channels | real Dec-2024 | real Sep-2026 | old hand labels |
|---|---|---|---|---|---|
| DEV (train + 6-fold CV, Dec-2024) | 375 | 17,225 | 7,265 | 8,126 | 5,171 |
| NEWTRAIN (may train, Sep-2026) | 335 | 10,090 | — | 5,620 | 0 |
| **TEST (locked, untouched)** | **43** | 1,963 | 834 | 941 | 590 |
| **NEWTEST (locked, untouched)** | **143** | 4,287 | — | 2,351 | 0 |

Official labels give DEV **7,265** usable phase labels vs 5,171 by hand (+41 %). TEST was excluded from the
staging cache entirely; NEWTEST was split once by signal, seed 0, 70/30 (`dc_work/official/newtest_signals.csv`).

## 2. Hand vs official — and who was right (task 1)
5,469 channels are in both files: **97.24 % agree, 151 differ**. Agreement by phase: 2/6 ≈ .988, 1/4/5/8 ≈ .966-.972,
3/7 ≈ .938, **phase 9 = 0/12**. 9 DEV signals disagree wholesale (>=3 and >=40 % of the signal).
Config drift is *not* the explanation: hand Dec-2024 vs hand Feb-2025 agree **99.76 %**, and official-vs-Feb-2025
(97.19 %) ≈ official-vs-Dec-2024 (97.24 %), so the 2.8 % is stable label noise, not 21 months of changes.
`review/review_phase_resolved.csv` (155 rows) resolves the old review list — the model wins nearly 3:2 overall:

| category | MODEL right | HAND right | both wrong | no official row | official = overlap |
|---|---|---|---|---|---|
| A whole-signal swap (21) | **16** | 0 | 3 | 0 | 2 |
| B confident disagreement (56) | **28** | 25 | 0 | 3 | 0 |
| C label phase never green (36) | **15** | 3 | 10 | 7 | 1 |
| D stubborn (36) | 2 | 13 | 2 | 5 | 0 (14 both right) |
| **total (155)** | **61** | 41 | 15 | 15 | 3 |

## 3. Frozen beta (`models/beta_v0`, unchanged) on 478 signals in NEITHER DEV nor TEST (task 2)
Sept-2026 staging data, scored against official labels. One shot, nothing tuned afterwards.
Headline = detectors that got an answer (>=5 actuations) whose official phase turns green in the window.

| window | answered | accuracy (tb off / on) | non-standard | coverage of live channels |
|---|---|---|---|---|
| full 66 h | 7,849 | **.9753 / .9755** | .9423 (n=2,461) | 98.6 % |
| 6 h (mean of 4 anchors) | 7,688 | .9726 / .9733 | .9369 | 97.4 % |
| 30 min (mean of 4 anchors) | 6,909 | **.9612 / .9620** | .9095 | 90.2 % |

Errors at 66 h: 194 (93 concurrent pairs, 2↔6 the biggest at 46). Confidence is well calibrated:
p>=0.9 keeps 90 % at **.9960**, p>=0.8 keeps 94 % at .9936. Held-out slice: NEWTEST alone .9698 (2,317 detectors),
NEWTRAIN .9776. 199 detectors have *additional* call phases — .8945 against the primary phase but **.9950** if any
programmed call phase counts, i.e. the model picks one of the phases the channel really calls.
**Same DEV signals, 21 months later** (fold models fitted on Dec-2024 only, applied to Sept-2026):
.9747 at the long window / .9517 at 30 min vs .9733 / .9540 on their own Dec-2024 data — no time decay.

## 4. Retrain on official labels and more signals (task 3)
DEV 6-fold OOF, identical evaluation rows (7,197 detectors at the long window), all scored on OFFICIAL labels.
Seed repeats (3) give sd 0.02-0.06 pt, so differences above ~0.15 pt are real.

| variant | primary | 72 h | 30 min | non-std | fold 0 | errors | concurrent | cov@.9 / acc |
|---|---|---|---|---|---|---|---|---|
| A DEV, **hand** labels (historical) | .9610 | .9694 | .9526 | .9107 | .9748 | 220 | 98 | .947 / .9828 |
| B DEV, **official** labels | .9637 | .9733 | .9540 | .9193 | .9748 | 192 | 81 | .949 / .9877 |
| **C DEV + 334 NEWTRAIN, official** | **.9656** | .9735 | **.9577** | .9220 | .9720 | 191 | 81 | **.960** / .9877 |

Official labels: **+0.27 pt** primary, +0.39 pt at 72 h, 13 % fewer errors and 17 % fewer concurrent-pair errors.
More signals (+334, a different season, mostly a weekend): **+0.19 pt** primary, **+0.37 pt at 30 min**, nothing at
72 h — extra signals buy short-sample robustness, not a higher ceiling.
**Learning curve** (primary, seed 0, same eval rows; training pool size → score): 100 DEV .9597 · 200 DEV .9610 ·
375 DEV .9637 · +120 new .9651 · +225 new .9647 · +334 new (709 signals) **.9655**. Still rising but flattening —
tripling the pool from 100 to 375 bought +0.40 pt, doubling again from 375 to 709 bought +0.18 pt. More signals
still help; the return is roughly logarithmic and the next +0.2 pt would need another ~700 signals.

## 5. Overlaps as candidates (task 4) — **do not ship**
Overlaps are numbers, and they line up: every programmed `call_overlap` number runs as an event-61 Parameter at
59/62 Sept-2026 signals (15/17 in Dec-2024); a ±1 shift fits much worse, so there is **no offset**.
Overlap candidates were built by re-encoding 61/63/64/65 as begin-green/yellow/red-clear/end-red on pseudo-phases,
with one extra number-free flag `cand_is_overlap`; output would be `assign_type` + `assign_number` ("overlap 4").

| data | phase targets, phases only | + overlap candidates | overlap-only targets | predicted as overlap |
|---|---|---|---|---|
| Dec-2024, 207 signals | .9651 (n=4,275) | .9677 | **4 of 15** | 5 of 4,290 |
| Sep-2026, 420 signals | .9788 (n=8,977) | .9785 | **3 of 18** | 4 of 8,995 |

Paired over all windows: +0.06 pt (Dec) and **−0.16 pt** (Sept, 377 broken vs 295 fixed) — no gain, a small loss.
Why: **73 % of overlaps have a green that is >=0.95 Jaccard-identical to a phase's green** (median 0.99), so the
distinction is unlearnable from timing alone; and overlaps have no call events (43/44), the single strongest
feature family. With **24 overlap-only labels statewide** the class can be neither trained nor validated.
Recommendation: keep the phase-only output; flag the ~0.1 % of channels programmed to an overlap from the plan.

## 6. Delay and extend (tasks 2 + 5) — they do *not* explain the hard cases
Accuracy on the new signals by programmed value (66 h): delay none .9747 (n=7,325) / <=5 s **.9900** (399) /
>5 s .9600 (125); extend none .9740 (6,376) / <=2 s **.9850** (1,329) / >2 s .9444 (144). Small settings are
*better* than none; only the thin large tails are 2-3 pt worse. **Extend is invisible in the log**: only 2 % of the
1,867 channels with extend >= 1.5 s show an ON-duration floor near their extend (5 of 417 signals do), so 81/82 are
evidently not stretched here. Delay leaves a weak trace (AUC 0.71 from a label-free "short-ON deficit" score; delayed
channels have median ON 3.5 s vs 1.3 s), but it is confounded with presence-type detectors. A trust model over
`phase_prob` + evidence reaches AUC .946 with or without inferred delay/extend — and **the model's own
`phase_prob` alone is better (AUC .958 / logloss .069)**. Even the *true* programmed values add nothing (AUC .946).
So: no delay/extend feature is recommended, inferred or official.

## 7. What we learned (plain English)
1. **The model was right more often than the hand labels.** On the review list it wins 61-41, and on whole-signal
   renumbering 16-0. Roughly 40 % of what looked like model error was the config file.
2. **Official labels are simply better data**: +41 % more labelled channels and +0.27 pt accuracy, for free.
3. **It generalises.** .9753 on 478 signals nobody had touched, within 0.2 pt of its own DEV held-out number, and
   unchanged on the same signals 21 months later. The 43 TEST + 143 NEWTEST signals stay locked away.
4. **More signals help only short samples.** At 30 minutes +0.37 pt; at 66 hours nothing. Evidence, not variety,
   is what the long window lacks.
5. **Overlaps are not a learnable class from timing.** Three quarters of them are a copy of a phase's green.
6. **Delay/extend were a red herring.** They are barely visible in the log, and knowing them exactly does not
   improve either accuracy or confidence. Hard cases are concurrent pairs (2↔6), not delayed detectors.
7. Still open: 193 channels where the model is >=0.80 confident against the plan are listed in
   `review/official_vs_model_disagreements.csv` with an evidence sentence — 21 of them carry a phase number in the
   technician's own channel description, and 4 of those back the model (one reads "RadB Ph7 Count" while the plan
   calls phase 6). Non-standard-wired detectors remain the weak spot (.92 vs .99).

## 8. Artefacts
`dc_work/official/labels_official.parquet` · `new_signal_split.csv` / `newtest_signals.csv` ·
`beta_new_results.json` + per-window detail · `train/variants*.json`, `devstg.json`, `curve.json` ·
`ovl_{dec,stg}/overlap_results_*.json` + `overlap_green_similarity.csv` · `delay_extend_estimators.json` ·
`trust_results{,_m30}.json` · `review/review_phase_resolved.csv`, `review/official_vs_model_disagreements.csv` ·
models `models/beta_v2_official_candidate/` (+ `settings.json`, numpy-backend verified) ·
OOF contract files `dc_work/preds/phase_oof_official_{A,B,C,devstg,candidate}.parquet`.
**Negatives:** a first pass leaked three label-derived columns into the features and was thrown away (the guard is
now in `train_official.feature_cols`); the learning curve uses one seed; the Sept-2026 window is two-thirds weekend.
