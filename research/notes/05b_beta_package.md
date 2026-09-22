# 05_beta_package — shippable BETA, minimum-evidence rule, honest duration curve, TEST once (2026-09-17)

Deliverable: `models/beta_v0/` + `src/predict.py` + `tests/` + `docs/BETA_REPORT.md` +
`docs/img/accuracy_vs_minutes.{png,json}`.

## 1. What changed in the model
Stage 04's training mix had no window shorter than 30 min. Adding **4x5 min and 4x10 min** windows
(variant B) beat variant A on the fold-0 hold-out at every duration (1 min .794->.842, 5 min
.918->.933, 15 min .952->.960, 72 h .9793->.9814), so variant B was retrained on all 375 DEV signals
and ships. Features, decoder, tie-breaker, function head and thresholds are unchanged from stage 04.

## 2. Minimum-evidence rule (new, user-requested)
`predict.py` now refuses to answer detectors with fewer than **`min_actuations = 5`** ON events:
`status = "not enough data: N actuations in sample (need >= 5)"`, `phase_pred`/`function_pred` blank,
`review_flag` true, raw opinion preserved in `phase_guess`/`function_guess`. 5-19 actuations are
answered but flagged `ok - low evidence (N actuations)`. DEV OOF accuracy by actuation count
(all window lengths pooled, phase / function / n):
1-2 .887/.808/6305 · 3-4 .918/.847/4806 · 5-9 .942/.860/8964 · 10-14 .956/.860/6197 ·
15-19 .961/.869/4503 · 20-29 .963/.878/6675 · 30-49 .967/.877/8546 · 50-99 .969/.887/10422 ·
100+ .979/.897/40679. Cumulative accuracy of what we answer: >=1 .961 (100 % answered),
**>=5 .969 (89 %)**, >=10 .972 (79 %), >=20 .974 (68 %); the >=100 plateau is .979. 5 is the smallest
minimum within 1 pt of the plateau. The user chose 5 over 10 to answer more detectors; it is a
parameter, so a stricter service can set 10-20.

## 3. DEV 6-fold OOF under the new rule (72 h: 4,818 answered of 5,171; 30 min: 4,112)

| | 72 h | 30 min | answered @30 min |
|---|---|---|---|
| ranker v2 | .9749 | .9505 | 80.3 % |
| **+ decoder (ship)** | **.9817** +/-.0056 | **.9668** +/-.0052 | 80.3 % |
| + ODOT tie-breaker | .9826 | .9697 | 80.3 % |
| function | .8971 +/-.0208 | .8846 +/-.0253 | — |
| non-standard, decoder | .9203 (n=590) | .8874 (n=497) | — |

Unanswered at 30 min: 657 no actuations + 361 below the minimum; 40 answered detectors have a label
whose phase never turns green and are left out of accuracy. Fold 0: .9793 / .9305 (phase/function,
72 h) vs the 2025 BiLSTM's .9525 / .9059. Footnote under the old ">=1 actuation" rule: .9818 / .9624.

## 4. Duration curve (fold-0 hold-out, nested windows)
26 log-spaced sample lengths (1 min .. 72 h), each measured from the **same 10 anchor start times**
(nested windows, anchor moved earlier when it would overrun), averaged over anchors; 246 unique
windows, 991 k feature rows. Phase / phase+tie-breaker / function / % answered:
1 min .900/.927/.784/16 % · 5 min .960/.968/.842/51 % · 16 min .962/.969/.880/70 % ·
30 min .966/.972/.905/77 % · 56 min .972/.976/.910/82 % · 6 h .975/.978/.924/92 % ·
72 h .981/.981/.924/97 %. The 1-min point is *higher* than the 2-min point (.900 vs .840) because
at 1 min only the busiest 16 % of detectors clear 5 actuations; at 2 min a wave of marginal ones
crosses the line. Past ~15 min the accuracy curve is flat — extra time buys **coverage**, not
accuracy. Nested windows + a 3-point moving average + PCHIP fit removed the jaggedness of the
earlier sparse, independently-sampled grid; raw averages are drawn as faint dots and stored in the
json alongside the smoothed series.

## 5. Missing event codes (measured end to end, 38 signals, 30 min & 3 h)
Dropping **43/44 costs 18-22 pt** of phase accuracy and 17 pt of function. Dropping 8/9/10/11 but
keeping **7** costs only 0.4 pt (7 is now a fall-back for 8, and 9 for 10 — new in `predict.py`).
Keeping **only event 1** costs 3 pt phase and 17 pt function. 131/150/83-88/173 change nothing
measurable; **150 and 173 are read but unused**.

## 6. TEST, once, frozen (user-authorised exception)
43 signals, 590 labels. Decoder .9872 (72 h) / .9684 (30 min), tie-breaker ON .9872 / .9758,
function .9013 / .8879, non-standard .9189 (n=37), p>=0.9 -> 96.2 % kept at 99.2 % accuracy.
**Computed under the original ">=1 actuation" rule and NOT re-scored after the rule change**, so the
figures are marginally pessimistic. TEST > DEV OOF, so nothing is tuned to DEV. Predictions only in
`dc_work/preds/beta_test_DO_NOT_USE/`; no per-detector TEST analysis. TEST stays reserved for the
final study scoring.

## 7. Engineering
`predict.py` rewritten: importable `predict(events_df_or_path, start, end, odot_tiebreak,
min_actuations)`, accepts lowercase column names and DataFrames, repo-relative model dir, never
raises on thin input (13 smoke tests), `chunk_signals=N` to bound memory. Verified bit-identical to
the cached-feature pipeline on 4 DEV signals x 72 h (max prob diff 8e-4). `evaluate.py` reports the
unscorable-excluded headline with the old figure as a footnote. Footprint: 3.7 MB of models;
0.7 s / 218 MB for 1 signal x 30 min, 6.2 s / 1.17 GB for 20 signals x 24 h, 533 MB with
`chunk_signals=5`.

## 8. Negatives / open
Function is still ~.90 and the "Other" threshold 0.6 is unvalidated (no Other labels in DEV).
Standard-wired detectors are at .990 and non-standard at .920 — that gap is the remaining work.
The 5-19 actuation band is answered at .94-.96, noticeably below the .979 plateau.
