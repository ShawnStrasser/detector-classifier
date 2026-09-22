# 05_statewide_other — 6-code model, a different day, Other and Yellow_Red (2026-09-17)

Statewide pull (52.2 M rows, 958 signals, **2025-02-25 09:00–15:00**, only codes 1,7,43,44,81,82) → `dc_work\statewide\events\` (47.7 M rows after DISTINCT + dummy drop, **52 s** from the DuckDB copy). Labels `all_configs.csv` (7,847 rows, 444 signals). **The 43 TEST signals were dropped before anything was scored or inspected.** Code `src\statewide\`, artefacts `dc_work\statewide\`.

## 1. Label standardisation — `src\statewide\function_label_map.csv` (for review)
39 raw strings → Advance 2,593 / Presence 2,302 / Count 1,367 / Other 1,300 / Yellow_Red 285, each with `n_rows`, `n_signals`, `confidence` (high/medium) and a `note`. Conservative: only strings naming a class map to it. `a`/`a*`/`p`/`co`/`co*` → Advance/Presence/Count (medium, 27 rows — the same file uses the full words, so the scheme is unambiguous); `advance cars|trucks`→Advance, `stop bar|stopbar`→Presence. **`advance presence` (407) → Other** per user decision. bike, mid loop, mid, special, broken, ped*, rt, right/left turn, phase, junk → Other. **Label problems:** the 6 `a` labels are really `"a "` (trailing space); `,` `?` `lp#24` `way far back there` `add in phase 2 (8d/25)` are free text in a function column.

## 2. Does the full event set buy anything? — 6-code v2 pipeline, DEV, same 6 folds
Cycles rebuilt from 1 (green start) + 7 (green termination) only, so green = [1,7) and the green bitmask timeline is *identical*; lost are the yellow/red-clearance split, the coordinated/free flag and the 83–88 faults (`f_on_yellow`, `on_lift_green_{coord,free}` dropped). 249 features vs 281; verified against the stage-01 cache (green_share, on_lift_green, call43 corr = 1.0000).

| DEV OOF — classifiable / all labelled | 72 h | 6 h | 30 min |
|---|---|---|---|
| 6-code ranker (first stage) | .9631 / .9076 | .9625 / .8818 | .9330 / .8144 |
| **6-code + joint decoding** | **.9704 / .9145** | **.9697 / .8884** | **.9511 / .8302** |
| full-code v2 (stage 04, same metric) | .9696 / .9137 | .9691 / .8878 | .9520 / .8310 |
| function 3-class, 6-code (full-code .8951) | .8958 | .8936 | .8748 |

**Events 8–11, 131/150 and 83–88 are worth ~0.0 pt** (+0.08 / +0.06 / −0.09 pt). Phase identity lives in 43/44 and in the green mask, both of which survive six codes — so a 6-code pull is enough and statewide inference is cheap. Best finding of this stage.

## 3. Generalisation to a different day (6 h window, 2025-02-25)

| group | signals | labels | all labelled | classifiable | coverage | non-standard | errors (concurrent) |
|---|---|---|---|---|---|---|---|
| **(a) in neither DEV nor TEST** | 26 | 606 | **.9373** | **.9676** (587) | .969 | .9268 (164) | 19 (12) |
| (b) DEV signals, fold-appropriate model | 375 | 6,534 | .8711 | .9631 (5,910) | .905 | .8924 (1,106) | 218 (98) |

30-min sub-windows: (a) .9493, (b) .9487 classifiable; spread over the 12 sub-windows .9363–.9602 and .9445–.9521 — **no time-of-day effect inside the 6 h**. 1 h .9597/.9568, 3 h .9625/.9615. Coverage-vs-accuracy (a): p≥0.6 → 95 % cov at .9740, p≥0.9 → 91 % at .9801; (b) p≥0.9 → 84 % at .9827. Errors are still concurrent pairs (2↔6 = 6 of 19 in (a); 2↔6, 1↔6, 4↔7 in (b)). Function 3-class: (a) .8977 classifiable / .8753 all, (b) .8902.
**The clean number — unseen signals *and* unseen date, 6 h of data: .9676, no measurable loss against the .9697 DEV OOF at 6 h.** Caveats: (a) is only 26 signals / 606 detectors; the all-labelled column falls to .87 for (b) purely because a 6 h midday window leaves more channels silent (coverage .905 vs .942 at 72 h); **13 labelled signals (245 labels) emit no detector events at all**; 4 labels sit on channels >64 (dummies); the 19 Phase 9/10/11 labels are right 5 % of the time — those phases barely ever turn green.

## 4. "Other": threshold vs an explicitly trained class (statewide, TEST excluded, n = 6,497)

| approach | 4-class acc | macro-F1 | Other recall | Other precision | acc on A/P/C |
|---|---|---|---|---|---|
| 3-class head + threshold 0.75 (best macro-F1) | .7337 | .7089 | .457 | .457 | .8028 |
| 3-class head + threshold 0.6 (current default) | .7337 | .6784 | .234 | .483 | .8584 |
| **trained Other class, grouped 5-fold by signal** | **.8076** | **.7926** | **.626** | **.714** | .8530 |

True-Other detectors at 0.75 — share sent to Other / share *confidently* (p≥0.8) mislabelled: `bike` .73/.17, `mid` .72/.19, `mid loop` .58/.34, **`advance presence` .44/.47 (→Presence)**, `special` .29/.61, **`Yellow_Red` .10/.86 (→Count)**. **Recommendation: train Other explicitly** (+7.4 pt 4-class, +8.4 pt macro-F1, precision .71 vs .46, costing 0.5 pt on the three real classes); if the 3-class head must be kept, raise the threshold from 0.6 to **0.75**. Even then Other recall is only 0.63 — a third still get a confident wrong class, and `advance presence` is the worst case because behaviourally it *is* both.

## 5. Yellow_Red feasibility (285 labels / 268 scored, 55 signals, statewide only)
One-vs-rest, grouped CV by signal: **AUC .973, AP .762** at a 4.1 % base rate; p≥0.9 → precision .87 / recall .49; top-100 → precision .93. Against **Count alone** (the hard case; all 55 signals have both) AUC drops to **.902**. In plain words a Yellow_Red channel **never registers a phase call** (43-within-0.35 s: 0.00 vs 0.37), has **no queue on it before green** (0.2 s vs 2.9 s), actuates **almost only in its own green** (occupancy lift 4.4 vs 0.94), fires **late in green** (.18 vs .03) and carries **more volume than its siblings** (+12.7 ONs/h) — a downstream, non-calling counter. The 3-class head calls 255/268 of them **Count** at mean confidence .90.
**Verdict: not shippable yet** — real signal, but it is a sub-type of Count and at usable recall (~0.6) precision is only ~0.78, from 55 signals in one 6-hour window. To make it viable: (i) a **multi-day all-codes pull including 8/9/10/11 for the ~60 Yellow_Red signals** — the class is *defined* by actuations during yellow and red clearance, which six codes literally cannot see; (ii) more signals. Their channels cluster at 42–46, a giveaway but a wiring convention, not a legal input.

## 6. vs the 2025 BiLSTM, subset (a) only (`baseline\inference_results.parquet`)
It emits a prediction for **497 of 606** detectors. Where both predict: BiLSTM .9779 vs LGBM6 .9819, 99.2 % agreement (3 fixed, 1 broken). Over all 606: **.802 vs .937** — the gap is almost entirely the 109 detectors it declines. Function 3-class **.753 vs .875**, and it has no Other class, so all 245 true-Other detectors there are wrong by construction.

Outputs: `src\statewide\{make_label_map,common6,features6,train6,apply_statewide,other_yellowred,baseline_cmp}.py` + `function_label_map.csv`; `dc_work\statewide\{events,features,models,preds}\` — `dev6_results.json`, `statewide_phase_results.json`, `other_yellowred_results.json`, `yr_extra.json`, `baseline_cmp.json`, `phase6_statewide.parquet`, `function6_statewide.parquet`, `phase6_eval_h6.parquet`, per-fold + all-DEV 6-code models. Cost: convert 52 s, DEV features 16 min, statewide features 10 min, training 3 min, scoring 20 s.
