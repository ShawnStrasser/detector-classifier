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

Budget: one RTX A1000 (8 GB), about **2–3 full fold-runs a day**; one fold is ~50 min for a TCN and
~90 min for the GRU. **Screening rule:** train **one fold first** (~1.5 h), and promote a candidate
to all six folds only if it beats the noise floor by **>= 0.3 pt at 30 minutes**. Anything that does
not is dropped, not tuned.

In order, best expected value per hour first:

1. **Swap the GRU backbone for the TCN.** They tie on accuracy and the TCN trains and runs about
   twice as fast. Do this first: it buys the compute for everything below. ~1 fold to confirm.
2. **Seed-ensemble the network.** Free — the fold models already exist. Averaging damps the
   0.1–0.45 pt seed noise the same way bagging did for the trees.
3. **Dropout and augmentation** (random time shifts, channel dropout). The net currently has
   neither, and it is the cheapest regularisation left. ~2 folds.
4. **Mixed sample lengths in network training** (5 min … 6 h). Worth +6 pt for the trees; the net is
   still trained on 5–30 min only. ~3 folds.
5. **Give the network sibling-detector context** — the joint-decoder idea, inside the network.
   Biggest expected gain and the most work: the input has to carry the other channels of the signal.
   Budget a week.
6. **Add the TCN to the blend alongside the GRU.** Three different readings, cheap once (1) is done.
7. **Mamba / state-space backbone.** Uncertain payoff and a real risk of a Windows CUDA build
   fight. Time-box it.
8. **Stack the LightGBM scores or features into the net, or the net's into the trees.** One careful
   test only — the overfitting risk here is high and out-of-fold hygiene is easy to get wrong.
9. **LightGBM: neighbour-trace summary features.** Hand-built versions of what (5) would learn.
10. **Wider / deeper network.** Last, and only if the data has grown.

**Do not retry:** Optuna on the trees, XGBoost / CatBoost / forests, peak-vs-off-peak contrast
features, detector-health masking, overlaps as a class, delay / extend settings, or a transformer at
this data size. All were measured and all failed; the notes say why.

**Stop rule.** Decide everything on the held-out folds. When a candidate wins there, run **one**
confirmation on the locked signals and change nothing afterwards — and state plainly in the note
that those exam signals have now been opened a third time, so the confirmation is a sanity check,
not an independent estimate.
