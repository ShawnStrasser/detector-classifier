# 11_function_v4 — function on the current config table, two periods 21 months apart (2026-09-21)
Labels = `dc_work/data/labels/detector_config_current.parquet` (pulled today) via `function_label_map_v2.csv`, which **supersedes** the Dec-2024 + Feb-2025 hand files. Phase truth = official timing; every phase-relative / sibling / lag feature is built on the phase model's **predicted** phase (OOF), never the label. Code `src/function_v4{,_prep}.py`, artefacts `dc_work/function_v4/`. The 43 TEST and 143 NEWTEST signals are dropped in `load_config_labels()` and re-asserted absent in all 7 frames. Nothing earlier overwritten.

## 1. Frame — a detector in both periods gives two samples, folds grouped by SIGNAL
8,659 channels / 466 signals → minus 43 TEST + 16 NEWTEST signals = **7,514 labels / 407 signals**; 7,418 (402 signals) have events in >=1 period. DEV folds kept; 29 new signals drawn into folds 1-5, seed 0, so both periods of a signal share a fold.
| | Advance | Presence | Count | Yellow_Red | Other | signals |
|---|---|---|---|---|---|---|
| usable labels (DEV / NEWTRAIN) | 2,153/106 | 2,018/169 | 1,179/143 | 233/60 | 1,190/167 | 373/29 |
| samples with Dec-2024 data (6,868) | 2,168 | 2,045 | 1,194 | 237 | 1,224 | 372 |
| samples with Sept-2026 data (7,172) | 2,154 | 2,112 | 1,299 | 288 | 1,319 | 384 |

6,622 detectors are in **both** periods, 246 Dec-only, 550 Sept-only. Over the 22-window variant-B mix that is **261,614 scorable rows** (122,607 Dec / 139,007 Sept) x 389 features. Scorable share of labels: long window .919 (Dec) / .941 (Sept), 5 min .738 / .832. Phase inputs (this stage's own 6-fold OOF, official labels, Dec-fitted fold models applied to both periods): top-1 **.9738** at 72 h, **.9762** at 66 h — stage 10 reproduced.

## 2. Three variants, identical rows (118,459 / 372 Dec signals), 6-fold grouped OOF
| variant | 5-class 72 h | 6 h | 30 min | 5 min | A/P/C 72 h |
|---|---|---|---|---|---|
| **A** v3 as shipped (old labels), OOF re-scored on the new table | .8168 | .8207 | .7971 | .7550 | **.8663** |
| **B** retrained on new labels, Dec-2024 only | .8206 | .8246 | .8019 | .7559 | .8605 |
| **C** retrained on new labels, Dec-2024 + Sept-2026 | **.8224** | **.8268** | **.8032** | **.7573** | .8607 |

Noise floor (C, 3 seeds): sd **0.12 pt** long window, 0.08 pt all windows. Paired bootstrap over signals (90 %): A→B **+0.38 pt** [-0.12,+0.88] *ns*; B→C on Dec **+0.32** [-0.11,+0.75] *ns*; B→C on **Sept +0.73** [+0.21,+1.24] **real**. C on its own Sept rows: 66 h .8001 · 6 h .8061 · 30 min .7848 · 5 min .7453, A/P/C .8444. Coverage vs accuracy at the long window (Dec | Sept): p>=.7 → 78 % at .891 | .871 · p>=.8 → 68 % at .912 | .894 · p>=.9 → 47 % at .932 | .919.
| C, precision / recall, long window | Advance | Presence | Count | Yellow_Red | Other |
|---|---|---|---|---|---|
| Dec-2024 | .843/.888 | .831/.857 | .840/.814 | .927/.680 | .722/.665 |
| Sept-2026 | .834/.835 | .799/.858 | .796/.838 | .852/.658 | .732/.638 |

## 3. Does function survive 21 months? Yes — the season costs more than the years
DEV signals, long window: fold models fitted on **Dec-2024 alone** score **.7979** on the same signals' Sept-2026 data vs **.8217** on their own Dec data. Training on both recovers 0.8 pt (.8055), and a model that *has* both still scores Dec .8251 vs Sept .8080 **on the same 5,964 detectors with the same labels** — so ~1.7 pt of the gap is the Sept sample (66 h, two thirds weekend), not decay. The model returns the **same class for the same detector across the 21 months 89.7 %** of the time.

## 4. Yellow_Red with 337 labels (was 213) — ship it, and it still costs nothing
293 usable labels / 282 detectors / **59 signals**. Long window, both periods: AUC **.9886**, AP **.848**, **precision .883 at recall .709** (p_YR >= .39); vs Count only AUC .958 / AP .873; plain argmax P .89 / R .67. Per period Dec AP .876 (P **.924** at R .704), Sept AP .831 (P **.861** at R .706) — both clear the .85 / .70 bar. Recall by length .57 (5 min) · .67 (30 min) · **.75 (6 h)** · .67 (long). **Cost to A/P/C: none** — folding Yellow_Red into Other and refitting gives .8511 vs the 5-class head's .8522 (paired **-0.11 pt** [-0.33,+0.12]).

## 5. The v3 "advance presence" rule — re-validated, and now switched OFF
With the richer Other class the rule (Other when |p_adv-p_pres| < .2 and p_adv+p_pres > .6) now **costs** 0.16 pt of 5-class and **0.84 pt of A/P/C** (.8522 → .8438), buying Other recall .613 → .650 for precision .666 → .608. The trained Other already sends 63 % of `advance presence`, 84 % of `bike` and 57 % of `mid loop` to Other unaided. Shipped `"enabled": false`, parameters and the measured trade-off kept in the json.

## 6. Label noise — the config file moved toward the model, a little
`review/function_vs_config_disagreements.csv`: **509** confident (p>=.8) disagreements on 174 signals; the top 300 carry DeviceName, raw + mapped config function, model class, probability, official phase, channel description and an evidence sentence. Of stage 06's **387** confident disagreements, 381 are still in the table and the new table now **agrees with the model in 47 (12.3 %)** — 28 ex-Advance, 16 ex-Presence — 330 unchanged, 4 moved to a third class, 6 dropped. So about one in eight was a genuine label error since corrected; the rest are real model errors or edits never made.

## 7. Ship, and what the final `predict.py` needs (not edited here)
`models/final_candidate/function/function_lgbm_v4.{txt,json}` — 5 classes, 389 features, 146 trees, 2.6 MB, fitted on all 260,524 non-failed rows of both periods; `src/lgbm_numpy.NumpyBooster` reproduces it **exactly** (max abs diff **0.0**). `predict.py` needs exactly the three `predict_v1.py` patches — (1) four-colour cycles keeping event 11 (end red clearance) apart, (2) `features_v3.SQL_YR` + `SQL_LAG` merged onto the pair frame, (3) the 5-class `score_function` building shape / sibling / lag features **on its own predicted phase** — plus `DEFAULT_MODEL_DIR = models/final_candidate`, reading `function_lgbm_v4.json`, leaving `advance_presence_rule` disabled, and exposing `p_yellow_red` with the .39 operating point.

**Negatives.** (a) Every Dec-2024 row is labelled from a table pulled 21 months later, yet Dec scores 2 pt *higher* than Sept — drift is not the binding constraint, but it is unmeasured. (b) Other is still the fuzzy class (P .72 / R .66) and is the whole 4-pt gap between 5-class .82 and A/P/C .86. (c) Yellow_Red rests on 59 signals; read precision as +-.04. (d) Retraining on the new labels is worth only +0.4 pt and *loses* 0.6 pt on A/P/C — a prior shift toward the bigger Other class, not a better model. (e) Health flags exist only for Dec-2024; Sept rows are `unknown`, nothing dropped. (f) No ablation re-run; the v3 families are assumed unchanged (`lagsib_best_lag` is still the top-gain feature). (g) TEST and NEWTEST were never read.
