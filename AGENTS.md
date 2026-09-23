# Rules for AI agents working on this repository

Read this before touching anything. [README.md](README.md) is how to run the model,
[REPORT.md](REPORT.md) is what was learned; neither is a substitute for the rules below.

## Layout: what is production and what is not

* `model/` — **production**. `predict.py` plus the modules it imports, `weights/`, a bundled
  sample, `requirements.txt` and `check.py`. Nothing here may import from `research/`, contain an
  absolute path, need a GPU, or need torch / lightgbm / scipy / sklearn. Changing anything here
  means re-running `python model/check.py` and re-freezing the references if the numbers move.
* `research/notes/` — one short note per stage, 00 … 13, unchanged. **Every number in the two
  top-level documents must trace to one of these.** They refer to the old `src/…` paths; the map
  to the new ones is in `research/code/README.md`.
* `research/charts/`, `research/code/` — kept for posterity, not needed to run anything.
  `research/code/rpath.py` puts every research folder and `model/` on `sys.path`; feature and
  decoder *definitions* are imported from `model/`, so there is one copy of each.

## Data locations

* Repo: `S:\Data_Analysis\Python\detector-classifier` — a slow network share, so code, notes and
  the shipped weights only. Git needs `-c safe.directory=*`. **Do not commit or push** — the human
  does that. The Indiana event-code enumerations PDF sits in `research/notes/`, git-ignored because
  it is a third-party document.
* Work directory `%DC_WORK%` (default `~/dc_work`), on a fast local disk: **everything heavy**.
  `data/raw/` (the event pulls and the config exports), `data/splits/`, `cache/` (derived event,
  interval, cycle and green-state tables), `features/`, `preds/`, `models/`, `tmp/`. DuckDB's
  `temp_directory` belongs there too.
* `data/`, `archive/`, `review/` and `research/code/download/` are git-ignored and stay that way.
  The download scripts, the review spreadsheets and every label or event file are agency-specific
  and are **never** committed. Before finishing, grep the tracked tree for server names, table
  names, user names and credentials.

## Binding rules

* **Phase-anonymous.** The model never sees a phase number or a detector channel number. It scores
  a (detector, candidate phase) pair from behaviour only; candidates are every phase with a Begin
  Green in the sample; the prediction is a normalisation over those candidates. Any channel-to-phase
  wiring table is for evaluation or post-processing only, never a model input. `model/check.py`
  enforces this by renumbering the phases in a log.
* **Allowed event codes:** 81, 82 (detector), 1, 7, 8, 9, 10, 11 (colour state), 43, 44 (phase
  call), 131 / 150 (coordination), 83–88 (detector faults), 173 (flash). **Not allowed:**
  gap-out / max-out / force-off (4, 5, 6, 13), overlaps (61–66), FYA (32, 33), pedestrian events
  (21–23, 45, 89, 90), preemption. `Parameter > 64` on 81/82 are dummy detectors: drop them.
  De-duplicate first — about 2 % of detector rows are exact duplicates.
* **Locked hold-outs.** 43 TEST signals (`%DC_WORK%/data/splits/test_config.csv`) and 143 NEWTEST
  signals (`%DC_WORK%/official/newtest_signals.csv`). **Never train on them, never tune on them,
  never look at a metric on them** without being asked. Model selection uses the six signal-grouped
  folds in `%DC_WORK%/folds.csv`; fold 0 is the 2025 model's own hold-out, so old and new stay
  comparable there.
* **Label sources.** Phase truth = the official controller timing (`call_phase`, else
  `call_overlap`). Hand phase labels are retired: use them only to report agreement. Function truth
  = the newest hand-maintained config export. One label per detector: one phase or one overlap.
* **Unscorable** = zero actuations in the window, or the labelled phase never turns green in the
  window. Excluded from headline accuracy, counted and reported separately.
* **Measurement discipline.** No gain is real until it beats the noise floor: re-fit with at least
  two more seeds, and run a shuffled-label or noise-column control for anything that looks like a
  new feature family. Seed spread is about ±0.07 pt for the trees and 0.1–0.45 pt for the network.
  Report every headline metric with its coverage, and report failures honestly.
* **Operating rules.** Data pulls are one-time downloads for training: no polling, no new pulls, no
  extra login prompts unless asked. Other agents may share the machine — cap DuckDB at
  `memory_limit='10GB'`, `threads<=12`, and never load a whole day of events into pandas. Charts are
  drawn from saved sample data, never re-scored on the fly. Score with the fastest verified runtime.
  Each stage ends with a note of at most 60 lines in `research/notes/`.
* **No attribution lines** in commits or pull requests.

## Checking the production model

```bash
python model/check.py
```
Reproduces the stored answers on the bundled sample, proves phase-number invariance, and proves the
package runs with torch, lightgbm, scipy and sklearn blocked from import. All three must pass before
anything ships.

## Future work — a four-week greedy search

State of play is in `research/STATUS.md` — read it first. Two tracks. **Function first** (its accuracy is the weak half and most of its gap is labels and
per-lane structure, not model capacity), then the **neural track** for short-sample phase accuracy.
Budget: one RTX A1000 (8 GB), about **2–3 full fold-runs a day**; one fold is ~50 min for a TCN and
~90 min for the GRU. **Screening rule:** try the cheapest version first (CPU, one fold); promote to
all six folds only if it beats the noise floor by **>= 0.3 pt** (phase, 30 min) or **>= 1 pt**
(function). Anything that does not is dropped, not tuned.

### Track A — function (CPU, days not weeks)

The user's review of the 79 function misses on the 143 locked signals (`review/`; his corrections
are the truth from now on) showed **37 of 79 labels were wrong**; with corrected labels function
accuracy is 86.8 %, not 75.5 %, before any retraining. Corrected labels: `research/labels/`
(built by A1). His domain definitions (use them as FEATURES, never as override rules):

* **Count**: small zone at the stop bar; off during red, pulses during green. Usually set to *pulse*
  (every ON lasts exactly one 0.1 s tick); sometimes *normal* (ON for the whole vehicle passage).
* **Yellow_Red**: same location as Count, but with a ~5 mph speed filter, so it misses vehicles
  starting from a stop — expect it to miss the first car(s) of green. Often ONE detector spanning
  all lanes to save inputs.
* **Presence**: ~20 ft zone at the stop bar; occupied through red; turns off late in green once the
  queue has cleared.
* **Advance**: a count zone far upstream; random arrivals regardless of colour when free, platoons
  during green when coordinated. Queue spill-back over it happens only well into red. Set to pulse
  or normal.
* **Other** (not classified, but must be recognised): ETA zones (on while a vehicle is 3–5 s out),
  extension zones spanning both lanes between the advance loops and the stop bar, bike loops,
  departure zones ("makes sense these look like count zones"), long advance-presence zones.
* **Lane structure**: two lane-by-lane advance loops + one closer loop spanning both lanes
  correlate tightly at low volume and *diverge at peak* (two side-by-side vehicles = 2 actuations on
  the pair, 1 on the spanning loop). Long zones plateau in the day; count zones do not. Two count
  zones on a phase ⇒ two lanes ⇒ at most two presence and two Yellow_Red zones. An advance
  actuation is followed a few seconds later by the presence actuation in the same lane. Coordination
  is only detectable over ~a day (coord by day, free at night).

Ordered steps (each ends with a note in `research/notes/`):

A1. **Corrected labels + scoring fix.** Fold the user's corrections into the function label table
    (rows marked `?` are dropped from training and scoring; Bike/Departure map to Other). Fix the
    phase scorer to accept the timing's `switch_phase` and `additional_call_phases` as correct
    (e.g. a detector that calls 5 and switches to 4). Re-score final_v2. Re-run the confident
    function-disagreement list on ALL labelled signals (not only the locked ones) for the next
    review round — the label file is the ceiling.
A2. **Expert-shaped features** (phase-anonymous, per detector and per pair of detectors on the
    same predicted phase): pulse signature (share of ONs = one tick; duration bimodality); first-
    actuation lag after begin-green vs a co-located detector (Yellow_Red misses the first car);
    zero-lag co-location vs several-second lead (advance → presence); 15-min-count correlation
    off-peak vs peak and its divergence (spanning-lane loops); daytime count plateau vs linear
    growth; arrival randomness (Poisson index) split by coord/free where the sample is >= 1 day;
    spill-back events late in red. One CPU retrain of the function model; ablate.
A3. **Per-lane / per-phase joint decoding for function.** Group a phase's detectors into lanes
    from timing correlation + zero-lag co-location (no lane labels needed); then assign roles as a
    constrained problem: <= 1 Presence, <= 1 Count, <= 1 Advance **per lane** (a phase has 1–3
    lanes, usually 1–2 — two presence zones on a phase is NOT a violation, it means two lanes),
    Yellow_Red may span lanes, everything left over is Other. 44 signals carry RL/CL/LL lane text
    in the channel description: use them to validate the lane grouping. Output `n_lanes` per phase as a by-product. Evaluate
    with and without; report how often the inferred lane count is plausible. If the user later
    supplies lane-count labels, use them to validate, never as an input.
A4. **"Other" as rejection, not a class.** Train Advance/Presence/Count/Yellow_Red only; call
    Other when no class is confident OR the detector's features are far from the training
    distribution (e.g. isolation-forest / kNN distance on the feature vector). Score on held-out
    Other *subtypes* the model was never shown (leave one subtype out) — that is the property a
    trained Other class cannot have. Ship whichever wins on the corrected labels.

### Track B — phase on short samples (GPU)

B1. DONE — **TCN is the research backbone** (ties the GRU in the blend, 11× faster to train). It
    is for exploration only, NOT a shipping decision: at the end of the search the winning set-up
    is re-fitted with a GRU backbone as well and the two are compared before anything ships.
B2. **Seed-ensemble the network** — fold models exist; free.
B3. **Dropout + augmentation** (time shifts, channel dropout). The net has neither. ~2 folds.
B4. **Mixed sample lengths in network training** (5 min … 6 h). ~3 folds.
B5. **Sibling-detector context inside the network** (the joint-decoder idea). Biggest expected
    gain, most work; budget a week.
B6. **Add the TCN to the blend** alongside/instead of the GRU.
B7. **Mamba / state-space backbone.** Time-boxed; Windows CUDA build risk.
B8. **Stacking** LightGBM ⇄ network. One careful test; overfitting risk.
B9. **LightGBM neighbour-trace summary features** (hand-built version of B5).
B10. Wider / deeper network, last.

**Do not retry:** Optuna on the trees, XGBoost / CatBoost / forests, peak-vs-off-peak contrast
features for phase, detector-health masking, overlaps as a class, delay / extend settings, a
transformer at this data size. All measured, all failed; the notes say why.

**Nothing ships during the search.** `model/` stays at final_v2 until every track is finished.

**Stop rule (end of search only).** Decide everything on held-out folds. When a candidate wins there, run **one**
confirmation on the locked signals and change nothing afterwards — and say in the note that the
exam signals have been opened again, so it is a sanity check, not an independent estimate.
