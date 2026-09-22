# 01_foundation_lgbm — harness, event cache, pair features, LightGBM (2026-09-17)

Built `dc_work\folds.csv` (fold 0 = the 38 `device_id_valid` signals; folds 1–5 = the 337 train signals, seed 0), `labels_{dev,test}.parquet`,
a deduplicated event cache (`cache\events`, 225.4 M rows, 421 signals, 660 MB, hive-partitioned by DeviceId) + derived tables (ON intervals,
phase colour cycles, a **per-signal green bitmask timeline**, coordination, meta), `src\evaluate.py`, windowed pair features, LightGBM models.

## Headline (DEV, 6-fold grouped OOF, 72 h window)

| | all labelled (unclassifiable = wrong) | classifiable (>=1 actuation) |
|---|---|---|
| **Phase top-1** | **0.9081** (n=5171) | **0.9637** (n=4873, coverage 0.942) |
| Fold 0 (2025 hold-out) | 0.9287 | 0.9591 |
| per-fold mean ± sd | 0.9097 ± 0.0107 | — |
| Non-standard detectors | 0.8084 (n=663) | 0.8816 (n=608) |
| **Function 3-class** | **0.8588**, macro-F1 0.855 | 0.8841 |

Same harness: ODOT standard-wiring lookup **0.8329** all / 0.8752 on labels only (the protocol's 92.6 % is the ch≤40 subset — verified) and
only **0.113** on non-standard detectors; 2025 BiLSTM re-scored on fold 0 **0.9188** all / 0.9468 classifiable, so LGBM is **+1.0 / +1.2 pt**
on the old model's own hold-out. At p≥0.9: 86.8 % coverage, **98.5 %** accuracy. Schema and usage: `src\README.md`.

## Accuracy vs data duration (mean over sampled windows, incl. off-peak)

| training set | 30 min | 1 h | 3 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|---|
| **mixed durations** | 0.9316 | 0.9286 | 0.9549 | 0.9610 | 0.9632 | 0.9637 |
| 72 h only | 0.8691 | 0.8754 | 0.9305 | 0.9423 | 0.9557 | 0.9592 |

Mixing window lengths wins **everywhere**: +6.3 pt at 30 min, still +0.5 pt at 72 h. By actuations in the window:
<10 → 0.871, 10–25 → 0.921, 25–50 → 0.944, 100–250 → 0.953, >5 k → 0.967; worst window = 02:00 (0.890) — volume,
not duration.

## Ablations (champion = LGBMRanker, lambdarank, group = detector × window)

| variant | 72 h | all windows | short (30 m/1 h) | n feat |
|---|---|---|---|---|
| full | 0.9635 | 0.9470 | 0.9321 | 207 |
| **no event 43/44 features** | **0.8576** | 0.7914 | **0.7424** | 176 |
| no rank/z/argmax companions | 0.9641 | 0.9465 | 0.9313 | 95 |

Binary classifier + per-detector softmax vs ranker: 0.9629 vs **0.9637** (72 h); the ranker also won the inner tuning
split. Top gains: `call43_fwd_lift__z`, `call43_rev_frac__rank`, **`excl_diff_min`**, `release_frac_long`, `on_lift_green`.

## Errors and label suspects

298 of 5171 are unclassifiable (no events; 100 have a label that is not even a candidate phase). Of the other 177 errors: 80 concurrent-pair
(2↔6 = 36, 4↔7 = 13, 4↔8 = 10, 1↔6 = 9) and 97 other — the classic 2↔6 confusion is now a minority of a much smaller error set. 88 errors are
confident (p≥0.8) and only 10 agree with the standard wiring, so ~78 look like label errors, several signals contributing 3–4 each
(whole-signal renumbering). Review lists: `preds\suspect_labels_lgbm.csv` (top 40), `preds\review_list_lgbm.csv` (confident-wrong + p<0.5).

## What worked / did not / surprises

* **Event 43 (phase call registered) is the backbone** — removing 43/44 costs 10.6 pt at 72 h, 19 pt at 30 min. Despite recall suppression on
  coordinated phases, chance-corrected forward/reverse call linkage is the top feature family. The sequence model must see 43/44.
* The **green bitmask timeline** makes every "p green while q is not" question trivial; `excl_diff_min` (worst-case
  exclusive-green lift advantage over all other phases) is the best non-call feature — exactly what splits 2 from 6.
* **Rank/z companions bought nothing** (−0.06 pt), unexpectedly: chance-corrected lifts are already comparable
  across candidates, so the 95-feature model is as good at half the cost. Dedup (2.0 % duplicate rows) ≈ +0.5 pt;
  the ON-interval pairing was already immune to duplicates.
* Function is the weak head (0.884 classifiable; Advance↔Presence confusion 179 + 247) and is the next target.
  Cost: cache 2 min, features 11 min (14 windows × 421 signals), training 5 min, ablations 3 min.
