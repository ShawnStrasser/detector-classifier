# Research status — the single source of truth for whoever works next

Rules and the ordered plan live in `AGENTS.md`. This file holds the STATE: what is done, what is
running, what is next, what the user still has to decide. **Every agent updates this file at the end
of every step** (and before handing back), so the next session can pick up without the chat history.
Keep it short; put detail in `research/notes/`.

_Last updated: 2026-09-23 05:50 by the orchestrator, at hand-off to a new chat._

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

**Track B (GPU), B1 = TCN backbone, fold 0.** Trained and inferred (23 epochs, plateaued, best
inner-val .950 vs GRU fold-0 .9185 at the same metric — promising), state in
`C:\Users\hwyr67g\dc_work\trackB\` (runs/, models/, preds/, state/), logs
`C:\Users\hwyr67g\dc_work\logs\trackB_*.log`, launcher chain `trackB_chain.log`. The agent's
comparison script crashed on a missing `state/spec_b1.json` (see `trackB_eval.log`) — B1's
verdict (TCN vs GRU at 5/15/30/60 min on the held-out fold) is therefore **not yet written**;
compute it from `dc_work\trackB\preds\` vs the stage-13 GRU fold-0 predictions, then continue
B2 → B3 → B4 per `AGENTS.md`. Note file for the track: `research/notes/15_neural_trackB.md`
(create it if the agent did not).

## Next (in order)

1. Finish Track B (B1 verdict → B2 seed ensemble → B3 dropout/augmentation → B4 mixed lengths;
   6-fold run only for kept steps; blend numbers on the stage-13 rows).
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

## Never forget

Locked signals (43 TEST + 143 NEWTEST, paths in `AGENTS.md`) are never trained/tuned on. One-time
downloads only. No polling, no login prompts, quiet waiting, sample-based charts, fastest verified
runtime for scoring. Nothing ODOT-specific committed. No Co-Authored-By lines.
