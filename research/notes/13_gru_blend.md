# 13_gru_blend — refitting the GRU and blending it with the trees (`models/final_v2`, 2026-09-22)

Stage 12 left one lead open: averaging the LightGBM answer with a neural one was worth +0.9 pt at 30
minutes, but that GRU had been trained on hand labels, on a cache whose phase-call stream was truncated
for a day, with half its folds stopped by an epoch cap. This stage refits it properly, measures the
blend and ships it. Longer than the usual 60 lines because it also carries what would have gone into the
final report. Code `src/neural/{ncache2,data2,train2,infer2,export_gru,bench_gru,gru_speedup}.py`,
`src/{gru_onnx,gru_input,gru_blend}.py`, `src/official/{blend_v2,blend_v2_predecode,ship_final_v2,
score_final_v2,gru_old_vs_new}.py`; artefacts `dc_work/official/blend_v2/`, `dc_work/models/gru2/`.

## 1. The refit
Same architecture as stage 03 (conv stem, 3-layer BiGRU, attention pooling, 853 k weights). What changed:
**official timing labels** for every labelled live channel (12,755, not the 4,818 hand-labelled ones);
**709 training signals** over two seasons instead of 375; the **repaired cache** (the 43/44 assertion
passes for all three Dec-2024 days and for Sept-2026); **phase loss only**; detectors with no actuation
dropped from the loss; window lengths 5–30 min at a constant compute budget per step. Six signal-grouped
folds using the same fold map the shipped LightGBM was fitted with, so every row lines up.
On the 4,599 detectors the old model also covered, refitting is worth **+0.28 pt at 30 min** (.9646 →
.9674) and +0.11 pt at the full window. Its real value is coverage and honesty, not raw accuracy: 2.8×
more detectors scored, and the blend can be measured on the trees' own labels, signals and folds.

## 2. Held-out blend — 701 training signals, 12,751 detectors, 22 windows, official labels
Identical rows for all three; LightGBM = the shipped `final_v1` pipeline's out-of-fold answer.

| sample | LightGBM | GRU alone | blend 50/50 (shipped) | gain | blend seed noise (sd) |
|---|---|---|---|---|---|
| 5 min | .9287 | .9273 | **.9519** | **+2.32 pt** | 0.12 pt |
| 10 min | .9423 | .9475 | **.9614** | **+1.91 pt** | 0.12 pt |
| 30 min | .9623 | .9640 | **.9747** | **+1.24 pt** | 0.07 pt |
| 1 h | .9621 | .9658 | **.9747** | **+1.26 pt** | 0.04 pt |
| 3 h | .9762 | .9709 | .9797 | +0.35 pt | 0.02 pt |
| 6 h | .9770 | .9706 | .9801 | +0.31 pt | 0.07 pt |
| 24 h | .9797 | .9671 | .9817 | +0.20 pt | 0.09 pt |
| full span | .9808 | .9668 | .9828 | +0.20 pt | 0.15 pt |

**Weight** searched 0.00–1.00 on folds 1–5 over the short windows: it landed on **0.50**, and fold 0,
which never saw the search, gains +1.12 pt at 30 min there. **Where**: blending *before* the joint
decoder (average the ranker's probabilities, then decode) beats blending *after* it at every short length
(+0.10 to +0.29 pt) and is what ships; the control at weight 1.0 reproduces the stored `final_v1`
probabilities exactly. **Cut-off**: the gain is ≥ 1.2 pt out to 1 hour and ≤ 0.35 pt past 3 hours while
the cost grows with the sample, so the network is switched off above **120 minutes**. At 30 min the blend
also fixes the hard cases: non-standard wiring .914 → .940, concurrent-pair errors 885 → 550 (−38 %),
and `p ≥ 0.9` keeps 88 % of answers at **.9926** (trees alone 89 % at .9904).

## 3. One runtime — onnxruntime — and what it costs
The network is exported to **ONNX** (opset 17, batch-norm folded into the conv stem, pair and time axes
dynamic) and run by **onnxruntime on the CPU**: `src/gru_onnx.py` is the only runtime, with no PyTorch
path and no numpy fallback. It matches the PyTorch model it came from to **5.7e-07** on probabilities
with identical argmax, and reproduces the stored reference outputs to **8.9e-07** — the blended
pipeline's own phase probabilities come out bit-identical. Controlled bench, 4 threads, ~100–140
detector/phase pairs per signal (`dc_work/official/blend_v2/runtime_bench_onnx_t4.json`;
`runtime_bench_t4.json` holds the numpy implementation that was written and measured first, then deleted):

| | **onnxruntime 1.30 (shipped)** | numpy (measured, then deleted) | PyTorch CPU |
|---|---|---|---|
| 1 signal × 30 min | **0.38 s** | 1.20 s | 0.38 s |
| 1 signal × 2 h | **1.54 s** | 6.30 s | 1.39 s |
| 20 signals × 30 min | **0.47 s/signal** | 1.14 s/signal | 0.41 s/signal |
| peak extra RAM | 2–1,093 MB | 48–151 MB | 0–673 MB |

End to end through `predict.py` (20 signals, 8 threads): 30 min **0.11 → 0.59 s/signal**, 2 h 0.12 →
2.00 s/signal, 6 h 0.16 → 0.16 s/signal (the network is off above the cut-off). The GRU is **81 % of the
wall clock** at 30 minutes, 94 % at 2 hours, 0 % above it. Extrapolated to a statewide ~900 signals: a
30-minute sample takes ~1.7 min with final_v1 and **~9 min** with final_v2 (it would have been ~19 min on
numpy); a 24-hour sample is unchanged at a few minutes because the network never runs. ONNX was chosen
over numpy because it is **2.4–4.1× faster for identical numbers**. numpy would also have met the
"< 5 s per signal per 30-min window" rule with no new dependency, but the saving on a statewide sweep is
real; the cost is one hard requirement (`onnxruntime>=1.17`) and a larger transient memory footprint when
many pairs are batched. More CPU threads help neither runtime — the matrix products are too small to
split further. Research note: the locked-signal scoring in §4 was finished with the forward pass on the
**GPU** (`src/neural/gru_speedup.py`, torch, research only, never imported by `predict.py`), verified
equal to the shipped session — max probability difference **5.1e-07**, argmax identical.

## 4. The locked signals, opened a second time
The 43 TEST and 143 NEWTEST signals were already used once, to score `final_v1`. **This is a second look
at the same exam signals.** Everything — weights, blend weight, location, cut-off, runtime — was frozen
first and nothing changed after. Both models ran end to end on identical rows; headline rule as before
(≥ 5 actuations, official phase green in the window). Short windows use four start times.

| set | sample | final_v1 | final_v2 | errors | non-standard |
|---|---|---|---|---|---|
| **143 NEWTEST**, Sept-2026 | 5 min | .9373 | **.9510** | 96 → 75 | .853 → .887 |
| | 15 min | .9652 | **.9768** | 69 → 46 | .921 → .946 |
| | 30 min | .9728 | **.9815** | 58 → 39 | .935 → .957 |
| | 1 h | .9758 | **.9838** | 53 → 36 | .945 → .962 |
| | 6 h | .9823 | .9825 | 41 → 40 | .964 → .964 |
| | full 66 h | .9862 | .9862 | 32 → 32 | .973 → .973 |
| **43 TEST**, Sept-2026 (853) | 30 min | .9683 | .9695 | 27 → 26 | .917 → .911 |
| | full 66 h | .9785 | .9785 | 20 → 20 | .945 → .945 |
| **43 TEST**, Dec-2024 (719) | 30 min | .9621 | **.9667** | 27 → 24 | .889 → .894 |
| | full 72 h | .9683 | .9683 | 26 → 26 | .904 → .904 |

At 30 minutes on the big set the new model makes **a third fewer phase errors**, concurrent-pair errors
fall 39 → 26, and `p ≥ 0.9` keeps 92 % of answers at **.9963**. Function accuracy is identical to the
last decimal on all three sets (.754 / .779 / .823 five-class): by design the function head still reads
the trees-only phase. Above the cut-off the two models run the same code; the ≤ 0.02 pt wobble at 6 h is
the pipeline's own run-to-run noise (running `final_v1` twice on one window moves probabilities by up to
6e-4). Per-detector predictions live only in `dc_work/preds/final_test_DO_NOT_USE/v2/`. Three of the four
60-minute windows and everything after them were computed with the GPU forward pass once CPU cost became
the bottleneck (§3); the earlier windows used the numpy implementation. Both agree with the shipped
onnxruntime session to under 1e-06 on probabilities with identical argmax, so the table is the shipped
model's.

## 5. Plain English, and honest notes
Two models read the same log in completely different ways — one through 261 hand-built timing
measurements, one through the raw second-by-second trace — so with little data they make *different*
mistakes. Averaging removes about a third of the errors on a half-hour sample, and most of what goes is
the hardest kind: detectors wired against convention, and phases that are green together. With hours of
data both already sit on the label noise, so the network is simply switched off. Refitting the GRU
mattered less than it looked (+0.3 pt); what it bought was a model fitted on the same labels, signals and
folds as the trees, so the blend could be measured on 12,751 detectors instead of 4,654.

* **The exam signals were opened twice.** The decision was made on out-of-fold data, but these 186
  signals are no longer virgin — read §4 as a confirmation, not an independent estimate.
* Folds 1 and 3 hit the 45-epoch cap with their best inner-validation score on the last epoch. Fold 1 was
  continued and stopped on patience seven epochs later **without improving**, which proves it had been at
  a plateau; its predictions are unchanged. Fold 3 was left at the cap on instruction. Both sat at or one
  step above the minimum learning rate, oscillating inside a ±0.3 pt band, so this is a flat plateau
  rather than real improvement — but it makes the GRU column a slight under-statement.
* Noise floor: three independent fits of fold 0 (full recipe, different seeds) move the GRU by
  0.10–0.45 pt (sd) and the blend by only 0.02–0.15 pt. Averaging with the trees damps the seed noise,
  and the +1.24 pt gain at 30 minutes is about 18× the blend's own sd there.
* The GRU scores only programmed channels out of fold (97.0 % of the tree pipeline's rows); unlabelled
  active channels keep their tree probability in the "before the decoder" experiment.
* GPU cost ~15 h for 6 folds + a final model + 3 noise-floor fits + held-out inference — above the
  10–12 h budget, partly because a crash forced a restart.
* The accuracy-vs-data chart (`docs/img/final_accuracy_vs_minutes.png/json`) was regenerated for
  final_v2: the 14 sample lengths at or below the cut-off were re-run end to end, the longer ones reuse
  the final_v1 runs because the network never runs there and the code path is identical. Smoothed curve,
  final_v1 → final_v2: 1 min .871 → .865, 5 min .934 → **.956**, 15 min .957 → **.975**, 30 min
  .965 → **.979**, 1 h .971 → **.980**, 6 h .980 → .981, 1 day .985 → .985, full .986 → .986. The one
  place it is not better is a **one-minute** sample, where only 13 % of detectors get an answer at all
  (227 of 2,348) and the difference is well inside the noise of so few rows.
