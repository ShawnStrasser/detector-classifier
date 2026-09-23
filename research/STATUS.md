# Research status — the single source of truth for whoever works next

Rules and the ordered plan live in `AGENTS.md`. This file holds the STATE: what is done, what is
running, what is next, what the user still has to decide. **Every agent updates this file at the end
of every step** (and before handing back), so the next session can pick up without the chat history.
Keep it short; put detail in `research/notes/`.

_Last updated: 2026-09-23 07:05 by the Track B agent, after B1 + B2._

## Shipped

`model/` = **final_v2** (commit `e28c40f`, restructured in `656d22b`): LightGBM ranker → joint
decoder, plus a GRU blended in (ONNX runtime) for samples under 2 h; 5-class function head.
Locked-signal exam (143 signals): phase **.9815** at 30 min, **.9862** with days; function **.868**
on the user's corrected labels (.755 on the raw config). Thresholds and usage: `README.md`.

## Done (see research/notes/00 … 14)

| # | what | outcome |
|---|---|---|
| 00–12 | first wave: LightGBM pipeline, official phase labels, more data, final_v1 | phase .986 (days) |
| 13 | GRU refit + blend, ONNX | +1.2 pt phase at 30 min → final_v2 |
| 14 (Track A) | function: user-corrected labels; expert features; lane decoding; Other-as-rejection | labels +11 pt; features +0.16; lanes failed (see caveat below); rejection = flag only |

**Track A caveats to carry forward.** (1) The lane experiment judged "two presence zones on one
phase" as a label violation — wrong: a phase has 1–3 lanes, so the rule is <= 1 Presence / Count /
Advance **per lane**, up to 3 lanes; Yellow_Red may span lanes. Redo the lane check with that rule
before concluding anything. (2) Same-lane pairing from timing alone reached only AUC .74; 44
signals carry RL/CL/LL lane text in the channel description (`data/detector_plans.parquet`) — a
ready-made lane label set to validate against. (3) The round-1 corrections were all on locked
(test) signals, so they fixed *scoring* but no *training* label — round 2 is the one that changes
what the model learns.

## Label override table — the truth for function

`research/labels/function_labels_v2.parquet` (+ README): the user's corrections beat the config
file; rows marked `?` are excluded from training AND scoring; Bike / Bike Loop / Departure → Other.
**Every future function stage must read this table, not the config file.** Merge new review rounds
into it the same way (keep a `source` column: config / review_round1 / review_round2 …).

## Running now

**Nothing. The GPU is idle** (verified `nvidia-smi` 0 MiB at 07:05). No detached chain is armed.

**Track B, B1 and B2 are finished** — numbers, verdicts and caveats in `research/notes/15_neural_trackB.md`.

| step | verdict | fold-0 blend at 30 min | rule |
|---|---|---|---|
| B1 TCN backbone | **kept** | .9737 vs GRU .9745 (−0.08 pt), 11× faster to train, 2.6× faster on CPU | within 0.3 pt ⇒ adopt |
| B2 three-seed ensemble | **dropped** | +0.10 pt over the mean of three seeds | needs ≥ 0.3 pt |

Two corrections to the previous hand-over. (1) The GRU's fold-0 inner-validation score is **.9528**, not
.9185; the TCN's .9501 is inside the GRU's own seed spread, so inner validation was never evidence for
the TCN — the held-out table is. (2) The first night's chain died at 18:20 on a launcher bug (a path
embedded in a `python -c` string is not converted from `/c/...` by Git Bash), costing ~11.5 GPU-hours;
also `research/code/rpath.py` had left `research/code/` behind `model/` on `sys.path`, so **every**
research script failed on `from common import DC_WORK`. Both are fixed.

**Carry forward:** the TCN blend's seed sd at 30 min is **~0.2 pt** (0.19 pt on the quantity stage 13
measured as 0.069 pt for the GRU), so judge B3/B4 against ~0.2 pt and fit two seeds before believing a gain.

**What `model/` would need to ship the TCN** (not done, nothing in `model/` was edited): `gru_onnx.py`
is already architecture-agnostic — it feeds `[N,9,T]` to an ONNX session and reshapes to `[D,K]`, so the
TCN needs **no code change**, only `model/weights/gru.onnx` replaced by the TCN export (identical input
/output names `x`/`score` and dynamic axes). `gru_input.py` and the `blend.json` settings (0.5, "before",
120 min) are unchanged; the 120-min cut-off is worth revisiting separately now the net is 2.6× cheaper.
Docstrings in `gru_onnx.py`/`gru_blend.py` say "3-layer bidirectional GRU" and would need rewording, the
weights file could keep or change its name (`WEIGHTS_FILE` in `gru_blend.py`), `model/check.py`
references must be re-frozen, and the shipped net must come from a `--final` fit on all training
signals, not a fold model.

## Next (in order)

1. Finish Track B: **B3** (dropout + augmentation) → **B4** (mixed sample lengths), fold 0 each, on the
   TCN; then all six folds for whatever was kept, so the blend numbers sit on the stage-13 rows.
   Everything is in place: `research/code/neural/trackb_{train,infer,eval,eval_full,decide,export}.py`,
   `dc_work/run_trackb.sh` (fix its `spec()` helper the way `run_trackb_b2.sh` does before reusing it),
   state in `dc_work/trackB/state/` (`best_args.txt` = `--arch tcn`, `best_seed0.txt` = `tb_tcn_f0_oof`),
   and `trackb_eval.py` reproduces the stage-13 fold-0 blend exactly, so any candidate is one command.
   A TCN fold is ~12–20 min of GPU plus ~4 min of inference, so B3 + B4 + six folds is about 3 h.
2. When the user returns round-2 function corrections (`review/function_disagreements_round2.xlsx`,
   586 rows, 193 signals): merge into the override table, retrain the function head on corrected
   labels **with** the expert features (`dc_work\trackA\models\function_v5_expert\`), re-score.
3. Lane structure, done right (per-lane rule, 1–3 lanes; validate on the RL/CL/LL signals).
4. Ship a `final_v3` only via the stop rule in `AGENTS.md` (held-out folds decide; one locked
   confirmation; say the exam signals were opened again).

## Decisions pending from the user

* Return the round-2 label review (`review/function_disagreements_round2.xlsx`).
* Whether lane-count labels can be pulled (or whether the RL/CL/LL text is enough).
* Whether the +0.16 pt expert features ship (recommendation: yes, with the next function retrain).

## Parked ideas (user's, not yet tried)

* **Preload the training rasters into GPU memory.** The TCN is small (700 k params, ~2 GB used of
  8 GB) and GPU utilisation drops to 0 % between steps while the CPU builds batches — training is
  data-bound, not compute-bound. Building each fold's rasters once and keeping them resident on the
  GPU could cut epoch time substantially. Try when GPU throughput becomes the bottleneck.

## Never forget

Locked signals (43 TEST + 143 NEWTEST, paths in `AGENTS.md`) are never trained/tuned on. One-time
downloads only. No polling, no login prompts, quiet waiting, sample-based charts, fastest verified
runtime for scoring. Nothing ODOT-specific committed. No Co-Authored-By lines.
