# Research review and next-wave plan (2026-09-17)

Review of the Dec 2024 – Feb 2025 work, done before restarting research.
**Primary goal:** detector → phase assignment. **Secondary:** detector function (Advance / Presence / Count).

## 1. What was done

- **Data:** 3 days (2024-12-02..04) of hi-res logs, 421 signals, 339M events, all 132 event codes
  (`data/raw/`). Labels: 5,761 detectors / 418 signals, one phase + one function each.
  Device-level split: 337 train / 38 valid / 43 Kaggle-test, no overlap (`data/splits/`).
- **Features:** only events 1, 7 (green on/off), 43, 44 (phase call on/off), 81, 82 (detector off/on).
  One row per event, 20 channels (PhaseGreen1-9, PhaseCall1-9, DetectorState, Delta seconds),
  windows of 1000 events, random phase-channel permutation as augmentation.
- **Models (PyTorch):** dual-head BiLSTM (the keeper, `baseline/model5.pth`), BiLSTM+attention,
  phase-guided attention, transformer, dual-stream transformer (crashed). None of the later
  variants was ever properly measured; the Feb 2025 statewide inference run used the plain BiLSTM.
- **Statewide inference (2025-02-25, 6 h, 958 signals):** non-overlapping windows, mean of softmax per
  detector, abstain below 0.7.

## 2. How well it worked (recomputed from saved predictions, clean labels only)

| Metric (per detector, statewide run) | Phase | Function |
|---|---|---|
| All labeled detectors, no abstention (mean-prob) | **95.5%** | 92.0% |
| With 0.7 abstention, abstain = wrong | 89.7% | 83.5% |
| Confident (>=0.7) only, ~91% coverage | ~98% | ~96% |

- Confidence is well calibrated (0.9+ bin = 98.5% correct).
- **Phase errors are almost all concurrent/opposing pairs:** 2<->6 dominant, then 4<->8, 4<->7, 1<->6, 5<->2, 8<->1.
  Cross-barrier errors are ~0. The model knows *when* a detector is served but not *which* of the
  simultaneously-green phases owns it.
- Detectors with <=2 windows: 85% vs 95% otherwise (low-volume detectors are the hard ones).
- The statewide labeled signals very likely overlap the training signals (same agency and DeviceIds; overlap
  not yet quantified), so these numbers are probably optimistic;
  there is **no clean held-out number for the final model**. The 43 test signals were never scored.

## 3. Problems found

- Val loss used one-hot float targets (train used class indices) and only the first val batch; early
  stopping was done by hand on that number. Loss curves are not trustworthy.
- Label join by positional ROW_NUMBER (fragile). BiLSTM reads only the last timestep. Transformer layers
  were accidentally weight-tied. Loss weights were CE/9 + CE/3, not the documented 0.5/0.5.
- `validate/val_config.csv` devices are all inside the train config (contaminated; archived).
- Labels: Phase 9/10/11 rows (unlearnable, ~19), 21% of statewide function labels are free text
  ("advance presence", "bike", "Yellow_Red", "mid loop", ...). ~5.5% of configured detectors have no events;
  about half of active detector channels have no config row.

## 4. Event-code check (vs. Indiana enumerations PDF in this folder)

**The Dec 2024 training pull already contains every code worth having. No re-pull needed for research.**

| Tier | Codes | Why | In `data/raw`? |
|---|---|---|---|
| 1 | 81/82; 1,7,8,9,10,11; 43/44; 45, 89/90 | Detector waveform; full phase color state; call registered is a near-direct detector->phase signal; ped calls explain service without vehicle calls | yes, all (414-421 devices) |
| 2 | 4, 5, 6, 13 (gap/max/force-off); 61-66 overlaps; 32/33 FYA; 131, 150 | Last actuation before gap-out identifies the *extending* detector's phase - directly attacks 2<->6 confusion; overlaps/FYA explain right-turn and permissive-left detectors | yes (4: 412 dev, 13: 377, overlaps: 237, FYA: 76) |
| 3 (masks only) | 83-88 faults, 173 flash, 102/105 preempt, 182/184 power | Exclude garbage intervals/channels | yes |

Caveats: 43 is suppressed for phases on recall (coordinated 2/6 emit few) - exactly the phases we confuse,
so gap-out/extension evidence (4/13) and green-utilisation matter most there. 81/82 are logged after
delay/extend processing. Vendors differ in 43/44 and 4-vs-13 behaviour.

Only gap: `data/statewide_2025-02-25/All_ODOT_Signals.csv` has just the old 6 codes and 6 hours. It is fine as a
baseline comparison, but applying a richer model statewide later will need a new pull with the Tier 1+2 codes
(ideally 24 h+). Not needed yet.

## 5. Next wave

1. **Evaluation harness first.** Key-based joins, device-level split, per-detector metrics with
   coverage/accuracy curve, confusion by ring/barrier pair, scored on valid (38) and untouched test (43) signals.
   Re-score `baseline/model5.pth` on it to get the true number to beat.
2. **Candidate-phase ranking formulation.** For each (detector, candidate phase) pair compute features and score
   with a shared model; softmax over candidates. Permutation invariance for free.
3. **Auditable feature baseline (gradient boosting):** per candidate phase - P(43 within 0.5 s of 82 | phase not green),
   actuation-time histogram relative to begin green / yellow / red, occupancy by color state, share of gap-outs (4/13)
   preceded by this detector's last actuation, green extension correlation, overlap/FYA equivalents.
4. **Joint per-signal decoding:** detectors at one signal constrain each other (channel-number adjacency, each
   phase usually has a presence detector, ring/barrier structure). Set model or assignment post-pass.
5. **Sequence model on the richer event set** only after 1-4 give a trustworthy bar; ensemble with 3.
6. **Function:** red/green occupancy ratio, discharge burst after green, gap-out relationship (Advance extends,
   Count does not), call behaviour. Clean the label vocabulary first; decide what to do with "advance presence",
   "bike", "Yellow_Red".
