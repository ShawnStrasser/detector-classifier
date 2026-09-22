# 06_function_v3 — Yellow_Red and a trained Other on 72 h of all-code data (2026-09-17)

`all_configs.csv` (2025-02-25) and `detector-configs.csv` (Dec-2024) describe the **same ODOT signals** (same DeviceId GUIDs); the Dec file simply lists only the Advance/Presence/Count rows. Joining by (DeviceId, channel) puts the Other and Yellow_Red labels onto **72 h of data containing events 8–11**, which stage 05 (6 codes, 6 h) could not see. TEST removed before anything was read. Code `src/{function_v3,features_v3,function_v3_prep,predict_v1}.py`; artefacts `dc_work/function_v3/`.

## 1. The label join — the two files agree, and the merge is what makes this stage possible
Of the 5,136 non-TEST channels present in both, **Phase agrees 99.73 %** (14 rows / 8 signals) and **Function 98.77 %** (63 rows / 11 signals) → 72 disagreement rows / 15 signals, all on the review list; drift is systematic (Advance→Presence 26, Presence→Count 13, Advance→"advance presence" 10) and two signals were re-catalogued wholesale (`82653e6b…` 22 rows, `ca10c12f…` 12). **Dec wins where both exist** (contemporaneous with the events); Feb only *adds* channels. Merged DEV set = **6,636 labels / 378 signals**: Advance 2,286 · Presence 2,002 · Count 1,108 · **Other 1,027** · **Yellow_Red 213 (48 signals)**, of which 1,504 rows are Feb-only (all Other, all Yellow_Red, 264 A/P/C the Dec file omitted); 6 signals with Dec events but no DEV fold got new folds (seed 0). Windows: **22** (4×5 min, 4×10 min, 4×30 min, 3×1 h, 2×3 h, 2×6 h, 2×24 h, 72 h) — the beta's variant-B mix. Phase re-run over all 22: ranker .9643, **+decoder .9707** at 72 h (stage 04 .9696), non-standard .8931 (was .8816).

## 2. Three ways to spell "Other" — DEV 6-fold grouped OOF, 72 h, 6,188 scorable of 6,636
| model | 4-class acc | macro-F1 | Other P / R | acc on A/P/C | Yellow_Red |
|---|---|---|---|---|---|
| 3-class head + threshold 0.70 (its best 4-class acc) | .7668 | .728 | .394 / .406 | .8445 | impossible |
| 4-class, Other trained | .8198 | .801 | .736 / .608 | .8668 | impossible |
| **5-class, + Yellow_Red (ship)** | **.8234** | .793 (5-cls) | .713 / .607 | **.8688** | **P .914 / R .653** |

A trained Other beats the best threshold by **+5.7 pt** 4-class accuracy and **+2.2 pt** on the three real classes; the threshold rule can only buy Other recall by wrecking A/P/C (at th 0.90, Other recall .72 costs 20 pt; stage 05's pick of 0.75 gives .765 / .827). Adding Yellow_Red on top is **free** (+0.4 pt 4-class, +0.2 pt A/P/C). Headline 5-class **.8227 ±.0256** per fold, .7672 counting the 448 dead channels wrong. p≥0.7 → 80 % coverage at .894; p≥0.9 → 54 % at .931.

## 3. Yellow_Red — now shippable
| | stage 05 (6 codes, 6 h) | **stage 06 (all codes, 72 h)** |
|---|---|---|
| one-vs-rest AUC / AP | .973 / .762 | **.9931 / .894** |
| AUC / AP **vs Count only** (the hard case) | .902 / — | **.9688 / .902** |
| 5-class head at p_YR ≥ 0.2 / 0.3 | — | P .841 / R .755 · P .873 / R .699 |

**Verdict: ship it.** Both heads clear the suggested bar (precision ≥ .85 at recall ≥ .7): the one-vs-rest head gives **P .910 at R .725** (p≥0.8), the 5-class column **P .873 at R .699** (p≥0.3); at plain argmax P .914 / R .653. Every miss is the same one — **61 of the 68 lost Yellow_Reds are called Count** — and only 12 of the 140 predicted Yellow_Reds are not one. n = 196 detectors on 44 signals, so read precision as ±0.05. Recall by sample length: 5 min .52 · 30 min .63 · 6 h .69 · 72 h .65. Cost to Advance/Presence/Count: **none measurable** (.8688 vs .8668 for the 4-class model).

## 4. Accuracy vs sample duration (one model trained on the 22-window mix; unscorable excluded)
| | 5 min | 10 min | 30 min | 1 h | 3 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|---|---|---|---|
| 5-class accuracy | .763 | .779 | .804 | .800 | .824 | .828 | .827 | .823 |
| accuracy on A/P/C only | .821 | .833 | .852 | .846 | .864 | .866 | .868 | .869 |
| scorable share of labels | .750 | .735 | .846 | .809 | .887 | .903 | .927 | .933 |

The 1 h dip is one window at 02:00–03:00. Flat past 3 h — evidence, not clock time, drives it, as in the beta.

## 5. Ablation (5-class, leave-one-family-out) — the lag family is the win
| feature set | n feat | 72 h | all windows | A/P/C 72 h | YR AP |
|---|---|---|---|---|---|
| **full v3** | 388 | **.8227** | **.8003** | **.8688** | **.843** |
| − v3 lag (pairwise cross-correlogram) | 360 | .8140 | .7819 | .8642 | .768 |
| − sibling-relative (stage 04) | 318 | .8156 | .7921 | .8656 | .827 |
| − v3 yellow/red-clearance | 347 | .8203 | .8004 | .8670 | .837 |
| stage-04 features only (no v3) | 319 | .8137 | .7796 | .8619 | .759 |

`lagsib_best_lag` (signed peak of the same-phase actuation cross-correlogram) is the **highest-gain feature of the whole model**; dropping the v2 pair family costs only 0.3 pt. The yellow/red-clearance family adds little accuracy but carries Yellow_Red when lag is gone; together the two v3 families move YR AP .759 → .843.

## 6. What distinguishes each class (medians at 72 h)
| | calls the phase | queue before green | median ON | occupied in red | fires in own green | last 2 s of green | lag to siblings |
|---|---|---|---|---|---|---|---|
| **Advance** | .56 | 1.6 s | 1.0 s | .63 | .48 | .03 | **leads by +2.8 s** (71 % of siblings) |
| **Presence** | .43 | **7.0 s** | **2.4 s** | **.90** | .43 | .03 | ±0 |
| **Count** | .02 | 0.2 s | 0.2 s | .12 | .87 | .12 | follows, −0.8 s |
| **Yellow_Red** | **.00** | 0.2 s | 0.2 s | **.02** | **.92** | **.20** | −0.3 s, **twin-coincidence .25** |

Advance = upstream, calls the phase, fires ~3 s before the stop-bar loop. Presence = at the stop bar, sits occupied through the whole red, longest ONs. Count = never calls, no queue, short free-flow ONs, follows its siblings. **Yellow_Red = a Count that is downstream of the stop bar**: it keeps firing in the last seconds of green and into yellow / red clearance (`yr_hit_yr` .154 vs .110 for Count, first ON 2.4 s after begin-yellow vs 6.0 s for Presence) and it sees the *same vehicles* as the stop-bar loop a beat later (twin-coincidence .25 vs .03 for Count). Other sits in the middle of everything (calls .50, queue 2.3 s, ON 1.2 s) — which is exactly why it is hard.

## 7. Where the "Other" strings go (share sent to Other, 5-class head, 72 h)
`bike` .81 (else Presence) · `mid` .86 · `mid loop` .54 (else Advance) · **`advance presence` .51** (else Advance) · `special` .24 (else Presence) · `bike zone presence`/`ped` 1.00 · `broken` unscorable (15 of 16 channels dead; the one alive has 14 k actuations — on the review list). **`advance presence` is inherently split**: 57 % have |p_advance − p_presence| < 0.25 and mean p_other .40, because the detector behaviourally *is* both. Shipped rule (`function_lgbm_v3.json`): report **Other when p_other is top OR when |p_advance − p_presence| < 0.2 and p_advance + p_presence > 0.6** — lifts `advance presence`→Other .51→.59 and overall Other recall .607→.654 at no cost in 4-class accuracy (.8234→.8235); δ = 0.3 reaches .63 / .67 for −0.2 pt.

## 8. Outputs, cost, honest negatives
`models/beta_v1_candidate/`: `function_lgbm_v3.{txt,json}` (5 classes, 388 features, 131 trees, 2.3 MB) + the **unchanged** phase/decoder models copied from `beta_v0`. `src/predict_v1.py` is a drop-in patching only `build_chunk_tables` (event 11 → red-clearance split), `build_features` (yr_ + lag tables) and `score_function`; **`predict.py` is untouched and its smoke test still passes**, and `tests/test_predict_v1_smoke.py` (5 tests incl. "phase identical to beta") passes. 0.66 s / 229 MB for 1 signal × 30 min — same as the beta. Review list `dc_work/function_v3/review_function.csv`, **463 rows**: 387 confident (p≥0.8) function disagreements, 72 Dec-vs-Feb label conflicts, 4 unlabelled channels that look like Yellow_Reds. **Negatives:** (a) 5-class accuracy .82 is far below the 3-class .894 — Other is genuinely fuzzy; 39 % of true Others still get a real class. (b) Feb-only labels score .643 vs .871 for Dec labels: three months out of date *and* the hard classes. (c) Yellow_Red rests on 44 signals. (d) The v3 features barely help the pure Advance-vs-Presence-vs-Count problem (3-class 72 h .894 vs stage 04's .895); they help Yellow_Red and Other. (e) Nothing here touched TEST.
