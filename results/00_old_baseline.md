# 00_old_baseline — honest held-out number for the 2025 dual-head BiLSTM

Re-ran `baseline/model5.pth` (nothing in `baseline/` touched; code copied to `src/old_baseline/`) on the **38
fold-0 signals** (`device_id_valid.csv` = the old model's manual-early-stopping validation set: never trained on,
so mildly optimistic but genuinely held out), Dec 2-4 2024 local data. Pipeline reproduced from
`baseline/inference.ipynb`: events 1,7,43,44,81,82, Parameter<=64, 20 channels, non-overlapping 1000-event windows
per detector, keep iff sum(DetectorState)>=10 and max(Delta)<=100 s, prediction = mean softmax over windows.
Cap 20 windows/detector/day (<=60 over 3 days; median 60). Function head order **verified** from
`baseline/dataprep.py` (`get_dummies(Function)` over {Advance,Count,Presence} -> alphabetical); phase = index+1.
Of 505 labeled detectors, 16 have no 81/82 events at all and 3 more no valid window -> **19 counted wrong**.

| Metric (per detector) | Dec 2-4 (<=60 win) | Dec 3 09-15 only |
|---|---|---|
| Phase acc, all 505 labeled (no window = wrong) | **0.917** | 0.842 |
| Phase acc, detectors with >=1 valid window | **0.953** (n=486) | 0.951 (n=447) |
| Phase acc, standard-wired dets (>=1 window) | 0.959 (n=438) | — |
| Phase acc, non-standard dets (>=1 window) | 0.896 (n=48) | — |
| Standard-wiring lookup alone (all 505) | 0.901 | 0.901 |
| Function acc, all labeled / with window | 0.875 / 0.909 | 0.812 / 0.917 |
| Function macro-F1 | 0.861 | 0.805 |
| Coverage / acc @0.5 / @0.7 / @0.9 | .923/.979, .889/.989, .796/.993 | .850/.972, .826/.978, .743/.989 |

- **Bar to beat: 0.917 phase / 0.875 function on all 505 labeled fold-0 detectors** (0.953 / 0.909 on covered ones).
  The lookup table alone gets 0.901 on the same detectors, so the old model's *net* gain is ~1.6 pts overall — but
  it gets 0.896 on the 50 non-standard detectors, where the lookup is 0.000 by construction. That is the value.
- Errors (3 day): 42 = 19 no-window + 12 concurrent/opposing pair + 11 other. Top: 2->6 x6, 6->2 x3, 2->8 x2,
  8->1 x2. Confirms the review: cross-barrier errors rare, 2<->6 dominates.
- Accuracy vs windows: 11-30 windows 0.824 (n=34) vs 31-60 windows 0.962 (n=448). Low-volume detectors are the
  hard ones; the 6 h slice loses 39 more detectors to "no valid window" (58 total), which alone costs 7.5 pts.
- Function confusion is symmetric Advance<->Presence (18 pairs) plus Count leakage; nothing structural.
- Files: `dc_work/preds/old_baseline_fold0_{phase,function}.parquet` (protocol contract; 808 detectors = 792
  predicted + 16 label-only, no-window ones uniform 1/9 and scored wrong; 303 predicted channels have no label).

## Caveats
- The notebook's sort key is (DeviceId, Detector, Timestamp, EventId) only, so **simultaneous phase events with the
  same EventId are ordered arbitrarily** — the original is itself nondeterministic there. My faster (equivalent)
  version precomputes the phase-state stream once per signal-day; `src/old_baseline/verify_transform.py` compares it
  to a verbatim copy of the notebook code: identical detector sets and window counts, 0.5% of feature cells differ
  (transient rows at simultaneous begin-greens), mean |dp| = 0.010, **0/44 argmax changes** — stable, not bit-exact.
- The 6 h numbers are NOT comparable to the Feb-2025 statewide 95.5%: that run's signals overlapped training.
  Minor: `device_id_valid.csv` has an unnamed index column; DuckDB needs explicit `names=`.
