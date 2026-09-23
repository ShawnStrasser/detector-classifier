# 14_function_trackA — Track A, detector function (2026-09-22)
Scripts + artefacts in `%DC_WORK%/trackA` (`a1_*` … `a4_*`); six signal-grouped folds, corrected
labels, 261,614 rows. **Nothing was copied into `model/`.**

## A1 — corrected labels + the phase scoring fix
`research/labels/function_labels_v2.parquet` (8,633 rows / 466 signals) = the config export mapped
to five classes with the engineer's review of the 79 locked-143 misses folded in: **37 changed, 38
confirmed, 4 dropped (`?`)**, Bike / Bike Loop / Departure → Other; biggest moves Presence→Count 13,
Presence→Advance 8, Other→Advance 6. Every correction is on a NEWTEST signal, so **no training label
moved** — the exam's truth did (provenance: `research/labels/README.md`). Re-scoring the frozen
`final_v2` predictions, no model change and no new inference:

| 143 locked signals | n | 5-class config → corrected | A/P/C config → corrected |
|---|---|---|---|
| 30 min (4 anchors) | 293 | .7560 → **.8424** | .7946 → **.8987** |
| 6 h (2 anchors) | 315 | .7555 → **.8460** | .7770 → **.8844** |
| full 66 h | 317 | .7539 → **.8675** | .7797 → **.9156** |

Full-window P/R: Adv .887/.932 · Pres .855/.959 · Count .792/.838 · YR 1.00/.739 · Other .929/.754.
**11.4 pt of the "function is only .75" story was label error.** *Scoring fix:* a prediction equal
to the timing's `switch_phase` or an `additional_call_phases` entry (`data/detector_plans.parquet`,
unioned over plans) now counts correct — 117 of 2,317 scored locked detectors have such a target,
and the effect is small and honest: full .98619 → .98662 (32 → 31 errors), 6 h .98248 → .98314,
30 min .98148 → .98207. The one full-window recovery is the engineer's own case, 13011 det 10 calls
5 and switches to 4. *Round 2:* `review/function_disagreements_round2.xlsx`, 586 confident (p ≥ .80)
disagreements on 193 signals (509 train/dev out-of-fold, 77 locked TEST-43), round-1 rows removed,
sorted by each signal's most confident row — Adv→Pres 97, Other→Adv 98, Count→Pres 59, Pres→Count 55.

## A2 — expert-shaped features: real, far too small to ship
54 phase-anonymous columns from the engineer's definitions: pulse signature and duration bimodality;
red occupancy in thirds (spill-back reaches an advance loop late in red); share of green actuations
in the first 5 s (a Yellow_Red misses the first car); occupancy saturation, busiest 15-min bins vs
quietest; dispersion index split by coord/free; per-pair binned-count correlation off-peak vs peak.

| 6-fold OOF, 3 seeds | all win | 30 min | 6 h | full | A/P/C |
|---|---|---|---|---|---|
| production features (388) | .7908 | .7958 | .8168 | .8144 | .8311 |
| **+ 54 expert features (442)** | **.7924** | **.7975** | **.8181** | **.8164** | **.8326** |
| + 54 pure-noise columns (control) | .7892 | .7939 | .8157 | .8133 | .8300 |

Seed sd 0.05 pt; paired over signals **+0.164 pt** all-windows [+0.054, +0.273] p .993 — clears the
noise floor, misses the **1 pt** bar by a mile; leave-one-family-out (fold 0) moves nothing beyond
fold noise (.8214–.8249 vs .8223). Why: `px_dur_q75`, `px_pulse_frac`, `px_bimodality` rank 3/4/5 of
442, so the trees *prefer* the new spelling of duration shape, but it was already in `det_dur_*`;
none of the genuinely new ideas (saturation, dispersion, spill-back, pair correlation) make the top
40. Put the other way, **the 54 expert features alone score .8183 at the full window** vs .8428 for
all 388. Candidate: `dc_work/trackA/models/function_v5_expert/`; `px_pulse_frac` is an agency wiring
habit, not physics.

## A3 — per-lane decoding: the lanes are not recoverable well enough
Lanes = average linkage on "do these two see the same vehicles" (correlogram excess × binned-count
correlation × count balance); roles = exact Hungarian assignment, ≤1 Advance/Presence/Count per
lane, Yellow_Red free, rest Other. Folds 1–5: argmax .7894 / A-P-C .8293; the constrained decode
**never wins** — at its best 5-class (.7912) the constraint barely binds and A/P/C is already .8260,
and where lane counts look physical it costs 1–5 pt, only ever trading A/P/C for Other recall.
**`n_lanes` per phase** (3,084 phases, 3.21 detectors each): **1 lane 29.5 %, 2 lanes 31.6 %, 3+
38.9 %** — never too small (≥ the labels' lower bound, max #Count/#Presence, on **98.5 %**) but
exact on 34 %: it over-splits (2.72 vs 1.47). Spot-checking 10 phases plus the 44 signals whose text
names the lane (`RL`/`CL`/`LL`; 175 phases, 993 pairs) says why — the best cue separates same-lane
from different-lane pairs at only **AUC .736** (correlogram excess; count correlation .681). At
03009 phase 2 ("Loop 1 - RL Advance" / "Rad A - RL Count") every channel got its own lane, and an
oracle using the "Rad A/B/C/D" unit as the lane still lost 10 pt — **the labels themselves break
≤1-per-lane on 25 % of those groups**. Keep it as a *review flag*: the 26 % of detectors on a phase
where some class is over-predicted score **.643 vs .869**.

## A4 — "Other" as rejection: loses on today's labels, wins on unseen subtypes
R = a 4-class head (A/P/C/YR) with every Other row removed from training; Other is called when the
head is unconfident or an isolation forest puts the vector far from training. Thresholds come from
the inner validation fold's *real-class* rows, so no Other label fits or calibrates anything.

| 6-fold OOF, all windows | 5-class | A/P/C | Other P | Other R |
|---|---|---|---|---|
| **T — trained Other (keep)** | **.7924** | **.8326** | .674 | .619 |
| R — rejection, best rule | .7255 | .8131 | .381 | .324 |
| R with rejection switched off | — | **.8648** | — | .000 |
| Other if either says so | .7679 | .7926 | .502 | .691 |

Rejection costs **6.7 pt**, and the isolation forest hurts monotonically: Other is not
out-of-distribution, it sits in the middle of the feature space, so 84 % of what only R calls Other
is a normal class. Leave-one-subtype-out reverses it (fold 0, subtype deleted from training) —
recall on the held-out subtype, T-trained / T-unseen / R: advance presence (n 1,442) .600 / **.087**
/ **.614**; mid loop (933) .567 / **.220** / **.682**; bike (799) .817 / **.234** / **.846**;
mid (306) .801 / **.533** / **.647**. A trained Other cannot name a kind of Other it has never met;
the rejection head can, matching T's *fully trained* recall on three of four. **Keep T; carry R's
confidence as a review flag.** Aside: a head trained without Other is **3.2 pt better on the three
real classes** (.8648 vs .8326).

## If any of this shipped, `model/` would need
Only A2 is a candidate: `model/features_expert.py` (the three DuckDB blocks of `a2_features.py`,
run against the temp tables `predict.py` already builds — `onev`, `cyc4_all`, `coordiv`), three
merges in `_function_frame()`, the new booster in `score_function()`, a weights folder and card,
then `model/check.py` re-run and references re-frozen. A3 would also need a pure-numpy assignment
(scipy is banned in `model/`) and is not worth shipping.
