# How the detector classifier was built

For the curious — what was tried, what worked, and what the limits are. How to run the model is in
[README.md](README.md); every number below traces to a stage note in `research/notes/`.

## 1. Problem and result

A traffic-signal cabinet logs an event every time a detector turns on or off and every time a phase
changes colour. Which detector is wired to which phase, and what kind of detector it is, lives only
in a configuration file that is often wrong. The task was to read it back out of the log. The
shipped model gets the phase right **98.6 %** of the time from a few days of data and **98.2 %**
from thirty minutes, on 186 signals locked away from the whole study; detector function is right
75–82 % over five classes. The 2025 model it replaces scored about 95 % and refused to answer many
detectors.

## 2. Model types tried

Phase accuracy on signals the model never trained on. The first block is the architecture bake-off
(all on the same held-out signals, hand labels); the second block is the final pipeline measured on
the official timing labels, so it is not directly comparable with the first.

| Architecture bake-off (same held-out signals) | 30 min | days | Verdict |
|---|---|---|---|
| 2025 dual-head BiLSTM (the model to beat) | — | .953 | Replaced. Declined to answer many detectors |
| **LightGBM detector-vs-candidate-phase ranker** | .941 | .971 | **Shipped.** Trains in minutes |
| **+ joint per-signal decoder** | .959 | .975 | **Shipped.** The single biggest modelling idea |
| GRU on the raw 1-second trace | .935 | .980 | Level with the trees, needs a deep-learning runtime |
| TCN (1-D convolutions) | .931 | .980 | Ties with the GRU and trains about twice as fast |
| Conv + GRU hybrid | .936 | .976 | No gain from combining |
| Transformer, three sizes | .899 | .970 | 3–4 pt worse on short samples; too little data for it |
| 2-D "image of cycles" CNN | .858 | .962 | No |
| XGBoost / CatBoost / random forest | — | −0.3 to −2 pt | Same family as LightGBM, no better |
| 3-seed bagging of the ranker | +0.03 pt | +0.03 pt | Shipped anyway: halves run-to-run wobble |

| Final pipeline (official labels, 12,751 held-out detectors) | 30 min | days | Verdict |
|---|---|---|---|
| LightGBM ranker + decoder | .962 | .981 | The trees alone |
| GRU, refit on the official labels | .964 | .967 | Refit bought +0.28 pt over the first GRU |
| **LightGBM + GRU averaged 50/50** | **.975** | **.983** | **Shipped.** +1.24 pt at 30 min, +0.20 pt with days |

The lesson is that **features beat architecture**. Every model type landed within a point or two of
every other once it had the right inputs, and no architecture came close to the value of simply
giving the trees the phase-call events. The trees win on cost, not on skill: the useful evidence is
a handful of timing facts ("a phase call registers 0.1 s after this detector turns on", "it sits
occupied through this phase's red") that can be computed directly and handed over, while a network
has to rediscover them from raw traces with only a few thousand labelled detectors. That is also
why the transformers lose — they have the least built-in structure and the most to learn, and this
dataset is small. The blend ships because the two halves read the same log in completely different
ways and therefore make *different* mistakes on short samples: averaging removes about a third of
the phase errors at 30 minutes, and most of what goes is the hardest kind (non-standard wiring
.914 → .940, concurrent-pair errors down 38 %). With hours of data both sit on the label-noise
floor, so the network is switched off above two hours.

## 3. Feature engineering and inputs tried

| Idea | Effect | Note |
|---|---|---|
| **Score "detector vs each candidate phase"** rather than "which of 8 phases" | the foundation | Works for any number of phases, any numbering, any agency |
| **Phase-call events 43 / 44** | **+10 to +20 pt** | By far the strongest family. Remove them and accuracy collapses |
| **Exclusive-green features** (p green while its usual partner is not) | large | The only per-detector evidence that separates a concurrent pair |
| **Sibling / joint-decoding features** (which channels fire in the same seconds) | +0.4 pt (days), **+1.8 pt (30 min)** | Removes 1 in 5 concurrent-pair errors |
| **Training on a mix of window lengths** (5 min … 3 days) | **+6 pt at 30 min** | One model serves every sample length |
| Six event codes (1, 7, 43, 44, 81, 82) are enough | −0.4 pt | A cheap statewide pull is viable |
| **Official controller timing as the phase truth** | +0.3 pt, 41 % more labels | Where it disagreed with the hand file, the model was right 61 to 41 |
| More training signals (375 → 709) | +0.4 pt at 30 min, 0 with days | Variety buys short-sample robustness, not a higher ceiling |
| Demand-invariant rates everywhere, never raw counts | required | Lets one model read 30 minutes and three days alike |
| Peak vs off-peak demand contrast | +0.23 pt → noise | A shuffled-noise control reproduced +0.16 pt of it |
| Binned (15 s) occupancy instead of exact intervals | −0.3 pt | Quantising destroys the 0.2–2 s detail |
| Detector-health masking, and a learned trust score | 0 | The model's own probability is already the best "is this right" ranking |
| Delay / extend settings | 0 | Extend is invisible in the log; knowing the true values adds nothing |
| Overlaps as a predictable class | not learnable | 73 % have a green almost identical to a phase's, and no call events |
| Optuna tuning (~60 trials, 11 knobs) | +0.05 pt = noise | |
| Pruning 261 features to 70 | −0.09 pt | A third of the size, if size ever matters |

## 4. What limits accuracy now

* **Label noise is the ceiling.** The hand-maintained config and the controllers' own timing agree
  on only 97.2 % of shared channels, stably over 21 months. About 40 % of what looked like model
  error was the label. A realistic ceiling is around 99 %.
* **Concurrent pairs.** Two thirds of the remaining phase errors are 2↔6 and its cousins — phases
  that are green together almost all the time, and can only be separated by the company a detector
  keeps.
* **Non-standard wiring** is still the harder group (.96 vs .99), though the gap narrowed a lot.
* **Function labels** are one hand-maintained file with no second opinion, and nearly all the gap
  is the fuzzy "Other" bucket: bike loops, mid-block loops, channels that behaviourally are both
  advance and presence.
* **Time of day matters only through actuation count.** More data mainly buys more *answered*
  detectors, not better ones: past about fifteen minutes the accuracy lines are nearly flat while
  coverage keeps climbing. Roughly 100 actuations on a detector is worth about 97 % accuracy, so a
  daytime window is worth more than a night-time one of the same length.

## 5. Lessons

* **Features beat architecture.** The phase-call events alone were worth more than every modelling
  trick combined.
* **Check the noise floor before believing a gain.** Re-fitting the same model with a different
  seed moves accuracy by ±0.07 pt, and a column of pure noise once "gained" +0.16 pt. Half the
  promising ideas died against shuffled controls and seed re-fits.
* **The labels were the ceiling, not the model.** Measuring against the controller's own timing
  instead of the hand file was worth more than any architecture change, and it came free.
* **Short samples are where models differ.** With days of data everything converges on the noise
  floor; every real decision in this project was made on 30-minute windows.
* **Two different readings of the same data beat one good one.** The blend gains precisely because
  the trees and the network fail on different detectors.
