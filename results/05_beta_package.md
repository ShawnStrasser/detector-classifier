# 05_beta_package — shippable BETA, honest duration curve, one-time TEST scoring (2026-09-17)

Deliverable: `models/beta_v0/` + `src/predict.py` + `tests/` + `docs/BETA_REPORT.md` +
`docs/img/accuracy_vs_minutes.{png,json}`. Headline definition is the new one: **unscorable
detectors excluded** (zero actuations in the window, or the labelled phase never green in it).

## 1. What changed in the model
Stage 04's training mix had no window shorter than 30 min. Adding **4×5 min and 4×10 min** windows
(variant B) beat variant A on the fold-0 hold-out **at every duration**: 1 min .794→.842,
5 min .918→.933, 15 min .952→.960, 30 min .961→.963, 72 h .9793→.9814. Variant B was therefore
retrained on all 375 DEV signals and is what ships. Everything else (features, decoder,
tie-breaker, function head, thresholds) is unchanged from stage 04.

## 2. DEV 6-fold OOF, new definition (4,822 scorable of 5,171 at 72 h; 349 unscorable = 298 dead + 51 label-never-green)

| | 72 h | 30 min | old "all labelled" 72 h | non-std 72 h |
|---|---|---|---|---|
| ranker v2 | .9749 | .9432 | .9091 | .9153 |
| **+ decoder (ship)** | **.9818** ±.0056 | **.9624** ±.0051 | .9155 | .9203 |
| + ODOT tie-breaker | .9826 | .9661 | .9163 | .9169 |
| function | .8970 ±.0209 | .8799 | .8453 | — |

Fold 0: .9793 phase / .9305 function vs the 2025 BiLSTM's .9525 / .9059 (same detectors, same
definition). Stage 04 (variant A, same definition) was .9799 / .9085 non-std — variant B gains
+1.2 pt on non-standard detectors, which is where the headroom is.

## 3. Honest duration curve (retrained on folds 1-5, scored on fold 0)
`report_v2.py`'s curve reused OOF models trained with fold 0 inside; this one does not. 90 windows,
14 durations × up to 8 start times. Phase (tie-breaker OFF/ON) / function / scorable share:
1 min .842/.864/.758/55 % · 5 min .934/.946/.854/75 % · 15 min .960/.966/.891/82 % ·
30 min .963/.969/.900/85 % · 1 h .967/.972/.909/87 % · 6 h .976/.979/.921/92 % ·
72 h .981/.981/.924/96 %. By actuations: <5 → .881, 10–25 → .955, >100 → .978. **Evidence, not
clock time, is the driver**; the tie-breaker is worth +2 pt at 1 min and nothing past 6 h.

## 4. Missing event codes (measured end to end, 38 signals, 30 min & 3 h)
Dropping **43/44 costs 18–22 pt** of phase accuracy and 17 pt of function — by far the most
important optional codes. Dropping 8/9/10/11 but keeping **7** costs only 0.4 pt (7 is now a
fall-back for 8, and 9 for 10 — new in `predict.py`). Keeping **only event 1** costs 3 pt phase and
17 pt function. 131/150/83-88/173 change nothing measurable; **150 and 173 are read but unused**.

## 5. TEST, once, frozen (user-authorised exception)
43 signals, 590 labels, 548 scorable at 72 h. Decoder .9872 (72 h) / .9684 (30 min), tie-breaker ON
.9872 / .9758, function .9013 / .8879, non-standard .9189 (n=37), p≥0.9 → 96.2 % coverage at
99.2 % accuracy. **TEST > DEV OOF**, so nothing is tuned to DEV. Predictions live only in
`dc_work/preds/beta_test_DO_NOT_USE/`; no per-detector TEST analysis was done. TEST stays reserved
for the final study scoring.

## 6. Engineering
`predict.py` rewritten: importable `predict(events_df_or_path, start, end, odot_tiebreak)`, accepts
lowercase column names and DataFrames, repo-relative model dir, never raises on thin input (1 min,
no 43/44, no event 1, empty selection — 10 smoke tests), `chunk_signals=N` to bound memory.
Verified bit-identical to the cached-feature pipeline on 4 DEV signals × 72 h (max prob diff 8e-4).
`evaluate.py` now reports the unscorable-excluded headline with the old figure as a footnote.
Footprint: 3.7 MB of models; 0.7 s / 218 MB for 1 signal × 30 min, 6.2 s / 1.17 GB for 20 signals ×
24 h, 533 MB with `chunk_signals=5`.

## 7. Negatives / open
Function is still ~.90 and the "Other" threshold 0.6 is unvalidated (no Other labels in DEV).
The tie-breaker's 30-min gain (+0.4 pt OOF) is smaller than the fold-0 curve suggests (+0.6–2 pt).
Standard-wired detectors are at .990 and non-standard at .920 — the remaining gap is the real work.
