# Research protocol (wave 2) — every agent reads this first

## Goal
From hi-res controller event logs, infer for each vehicle detector channel (events 81/82, Parameter = channel)
(1, primary) which phase it is assigned to, (2, secondary) its function: Advance / Presence / Count.

## Hard rules
- **Phase-anonymous.** The model never sees a phase *number* or a detector channel *number*. It scores a
  (detector, candidate phase) pair from the behaviour of that phase (green/yellow/red state, calls) relative to the
  detector, optionally with number-free context about the *other* phases (e.g. "any other phase green", "this phase's
  share of cycle time"). Candidates = every phase that has a Begin Green (event 1) in that signal's log. Prediction =
  softmax/normalisation of pair scores over the signal's candidates. Must generalise beyond ODOT conventions.
- The ODOT standard wiring (`docs/standard_detector_mapping.md`) may ONLY be used in an optional post-processing
  tie-breaker, never as a model input.
- **Allowed event codes:** 81, 82 (detector), 1, 7, 8, 9, 10, 11 (phase colour state), 43, 44 (phase call),
  131/150 (coordination status: coordinated vs free), 83-88 (detector faults, for masking/flagging), 173 (flash, masking).
  **Not allowed:** gap-out/max-out/force-off (4,5,6,13), overlaps (61-66), FYA (32/33), ped events (21-23,45,89,90),
  preemption.
- Parameter > 64 on 81/82 are dummy detectors: drop.
- Prefer better modelling over heavy data manipulation. Detector-failure handling is the one data-cleaning priority.
- Final model must run inference on CPU on a modest machine. Training box: i7-14700 (28 threads), 32 GB RAM,
  RTX A1000 8 GB. **Other agents may be running concurrently: cap DuckDB at `memory_limit='10GB'`, threads<=12,
  keep pandas frames modest, never load a whole day of events into pandas.**

## Paths
- Python: `C:\Users\hwyr67g\venvs\detector-classifier\Scripts\python.exe` (torch+CUDA, lightgbm, duckdb, polars,
  sklearn, atspm). `pip install` extra packages there if needed.
- Fast local work dir: `C:\Users\hwyr67g\dc_work\` — `data\raw\Train_Dec_{2,3,4}_2024.parquet`
  (DeviceId, Timestamp, EventId, Parameter; 339M rows, 421 signals, 3 days), `data\raw\detector-configs.csv`
  (DeviceId, Phase, Function, Detector — 5,761 labeled detectors, 418 signals), `data\splits\`. Put ALL caches,
  feature tables, predictions, model files under `dc_work\` (subfolders `cache\`, `features\`, `preds\`, `models\`).
  DuckDB temp_directory = `C:\Users\hwyr67g\dc_work\tmp`.
- Repo (slow network share, code + small docs only): `S:\Data_Analysis\Python\detector-classifier\` —
  code in `src\`, short result notes in `results\<stage>.md`. git needs `git -c safe.directory=*`. Do not commit.
  Do not touch `archive\`, `baseline\`, `data\`.

## Splits
- TEST = the 43 signals in `data\splits\test_config.csv`. **Never train, tune, or look at metrics on them** until the
  orchestrator asks for the single final scoring.
- DEV = the other 375 labeled signals. `dc_work\folds.csv` (DeviceId, fold): fold 0 = exactly the 38 signals in
  `device_id_valid.csv` (the old 2025 model's hold-out, so old vs new is comparable there); folds 1-5 = the 337
  `device_id_train.csv` signals split randomly (seed 0). All model selection uses 6-fold out-of-fold (OOF)
  predictions, grouped by signal.

## Prediction file contract (so one harness scores everything)
- Phase: parquet, long format: `DeviceId, Detector, cand_phase (int), prob (float, sums to 1 per detector)`.
- Function: parquet: `DeviceId, Detector, p_advance, p_presence, p_count`.
- `src/evaluate.py` scores such files against labels and reports, per detector:
  phase top-1 accuracy (overall; fold 0 only; per fold mean±sd); accuracy on **non-standard detectors** (label differs
  from the standard wiring table, or channel > 40) vs standard ones; errors grouped by concurrent pair
  (2/6, 4/8, 1/5, 3/7, 1/6, 2/5, 3/8, 4/7) vs other; coverage-vs-accuracy curve (threshold on top prob);
  count of detectors whose true phase is not among candidates or that have no events (counted as wrong, listed separately);
  function accuracy, macro-F1, confusion. Reference line: standard-wiring lookup alone = 92.6% phase accuracy.

## Function label "Other" (user decision 2026-09-17)
No detector is ever excluded from scoring. Function output space is Advance / Presence / Count / **Other**.
Any label outside the three (e.g. "advance presence", "bike", "mid loop", "special") is scored as true class
Other; a prediction counts as Other when max(p_advance,p_presence,p_count) < threshold (threshold tuned on DEV OOF;
an explicitly trained Other class may be tested where Other labels exist). The Dec-2024 DEV labels contain only the
three classes, so Other can only be evaluated on the statewide 2025-02-25 labels
(`data\statewide_2025-02-25\all_configs.csv`). Function prediction files may add an optional `p_other` column.
Stubborn / low-confidence / confident-but-disagreeing detectors are collected into a manual-review list for the user.

## Data-duration requirement (user, 2026-09-17)
In practice the model must work on as little as ~30 minutes of data, and use more when available. Every model stage
reports an accuracy-vs-data-amount curve: evaluate with features/inputs built from windows of 30 min, 1 h, 3 h, 6 h,
24 h, 72 h (windows sampled at several times of day incl. off-peak; report mean over windows; also accuracy vs number of
detector actuations in the window). Features must therefore be rates/shares/normalised (not raw counts that scale with
duration), and models should be trained on a MIX of window lengths so one model serves all durations.

## Data hygiene + detector health (stage 02 findings, adopt everywhere)
- Feed models RAW events, but `SELECT DISTINCT` first (2% of 81/82 rows are exact duplicates). atspm aggregates are not used.
- `src\health.py` + `dc_work\atspm\detector_health.parquet` give a label-free health flag per channel
  (healthy / suspect / failed). Only *absence of actuations* predicts errors. 297 DEV labeled detectors have zero events
  (unclassifiable by any model). Drop `failed` from training; at inference output "cannot classify: no actuations".
- Report every headline metric two ways: **all labeled detectors** (unclassifiable = wrong) and
  **classifiable detectors** (>=1 actuation in the window), plus coverage.

## Unscorable detectors (user decision 2026-09-17, supersedes "counted as wrong")
Headline accuracy EXCLUDES detectors that are impossible given the sample: (a) zero actuations in the window, or
(b) labeled phase never turns green in the window (not a candidate). Report their counts separately as "unscorable",
and put (b) on the manual-review list (possible mislabels). Still report the old "all labeled" figure as a footnote.
Neural training: no model is discarded for lack of training time — train to a documented plateau.

## Label sources (user decision 2026-09-21)
- **Phase truth = official controller timing** (`data/detector_plans.parquet`: `call_phase`, else `call_overlap`). Hand phase
  labels are retired: never train or score phase on them; use them only to report agreement. The timing itself can
  occasionally be wrong (detector programmed to the wrong phase) — rare; confident model-vs-official disagreements are
  listed for a field check, not treated as model errors to be fixed.
- **Function truth = hand-maintained config** (the newest version supersedes older files). No official function labels exist.
- Output is a single label per detector: one phase OR one overlap. Additional call phases/overlaps, delay and extend are
  metadata for diagnosis only, never model inputs.
- Data downloads are one-time pulls for training. No polling, no new downloads unless the user asks.

## Reporting
Each agent ends by writing `results\<stage>.md` (<= 60 lines: what was tried, table of numbers, what worked, what
did not, surprises) and returning a summary under 500 words. Report failures honestly; never tune on TEST.
