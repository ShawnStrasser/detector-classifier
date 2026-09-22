# `src/`

Everything heavy lives under `%DC_WORK%` (default `~/dc_work`); this folder is code only.
Run training with `C:\Users\hwyr67g\venvs\detector-classifier\Scripts\python.exe`.

## Shipping the FINAL model (`models/final_v2/`) — see `results/13_gru_blend.md`

`src/predict.py` is the only file an integrator needs. It loads the models from the
repo-relative folder `models/final_v2/` (no absolute paths anywhere in the inference path),
needs only `requirements-inference.txt`, and works as a CLI **and** as a function:

```python
from predict import predict
out = predict("events.parquet", start=None, end=None, min_actuations=5, min_prob=0.0)
```

It carries the three patches that used to live in `src/predict_v1.py` (now removed): the
four-colour cycle table that keeps event 11 apart, the `features_v3` yellow/red-clearance and
detector-lag tables, and the 5-class function head built on the model's own predicted phase.
The phase ranker is a **3-model seed bag** (`phase_lgbm_v4_s{0,1,2}.txt`), averaged as
per-detector probabilities. **No wiring table is used anywhere**: the ODOT tie-breaker was
measured and then dropped, so `tiebreak.py` is research code only and `predict.py` does not
import it.

**New in final_v2 (stage 13).** On samples up to the frozen cut-off (`blend.json`,
120 minutes) the ranker's per-detector probabilities are averaged 50/50 with a small
GRU that reads the raw 1-second trace, and the joint decoder then runs on the mixture.
The network runs with **onnxruntime on the CPU** — `src/gru_onnx.py` is the only runtime,
there is no PyTorch path and no optional import. Above the cut-off it is not run at all.
The function head still reads the trees-only phase, so function output is byte-identical
to final_v1.

| file | what it does |
|---|---|
| `predict.py` | **the shipped entry point** (`models/final_v2`) |
| `lgbm_numpy.py` | `NumpyBooster` / `NumpyBoosterBag` — run the LightGBM text models with numpy only |
| `gru_onnx.py` | the GRU pair scorer: one onnxruntime session, dynamic pair and time axes |
| `gru_input.py` | raw events → interval streams → the 1 s, 9-channel raster (training definitions) |
| `gru_blend.py` | `config()` / `phase_probs()` / `mix()` — the blend as `predict.py` uses it |
| `neural/{ncache2,data2,train2,infer2,export_gru,bench_gru,gru_speedup}.py` | stage-13 refit: cache, dataset, training, held-out predictions, ONNX export, runtime benchmark, research-only GPU forward pass |
| `official/fit_final_v1.py` | fits the LightGBM half (`--stage oof \| models \| card`) |
| `official/ship_final_v2.py` | assembles `models/final_v2` and its model card |
| `official/blend_v2.py` (+ `blend_v2_predecode.py`) | the stage-13 blend analysis (`--stage after \| before \| seeds`) |
| `official/score_final.py` / `score_final_v2.py` | the two scorings of the locked test signals |
| `official/curve_final.py` + `plot_final_accuracy.py` | the accuracy-vs-sample-length chart |
| `official/blend_check.py` | the stage-12 (old GRU) blend check |

Tests: `tests/test_predict_smoke.py`, `tests/test_numpy_backend.py`,
`tests/test_number_invariance.py`, `tests/test_gru_runtime.py` — run each in its own process.

The beta's own entry point (`predict_beta_v0.py`, `models/beta_v0`) was used once for the
before/after comparison in `results/12_final_model.md` and has since been removed; restore it
from git history to re-run that arm of `official/score_final.py`.

## Stage 01 foundation

| file | what it does |
|---|---|
| `common.py` | paths, allowed event codes, ODOT standard wiring, capped DuckDB connection |
| `build_cache.py` | `--step labels\|events\|derived\|check` → folds, label tables, event cache, derived tables |
| `evaluate.py` | scoring harness (protocol contract), reference + sanity predictors |
| `features.py` | windowed phase-anonymous pair features |
| `train_lgbm.py` | `--stage main\|ablations` → OOF predictions, models, ablations; `predict()` inference API |

```
python src/build_cache.py --step all
python src/features.py --windows mixed --out pair_features_windows.parquet
python src/train_lgbm.py --stage main
python src/evaluate.py --phase C:/Users/hwyr67g/dc_work/preds/phase_oof.parquet \
                       --function C:/Users/hwyr67g/dc_work/preds/function_oof.parquet --folds 0
```

## Stage 04 (v2) — see `results/04_lgbm_v2_decoding.md`

Stage-01 files and outputs are untouched; everything here writes `*_v2` artefacts.

| file | what it does |
|---|---|
| `error_forensics.py` | categorises stage-01 errors, tests the whole-signal label-permutation hypothesis |
| `features_v2.py` | *extra* pair features (partner geometry, exclusive-green with evidence, fine ON timing, queue release, conditioned 43-linkage) + `add_partner_diffs()` |
| `cross_detector.py` | phi similarity between detector channels (2 s activity bins) → `features/det_similarity.parquet` |
| `train_lgbm_v2.py` | `--stage ranker\|ablation\|decode\|function\|final` |
| `decode_v2.py` | second-stage joint per-signal decoder (similarity + channel-adjacency neighbours, signal-level claims) |
| `function_v2.py` | function model v2 (sibling-relative features, calibration, Other threshold) |
| `tiebreak.py` | `apply_odot_tiebreak(phase_probs_df, enabled=False, ...)` — post-processing only, default OFF |
| `report_v2.py` | duration / actuation curves, tie-breaker nested CV, `preds/review_list_v2.csv` |
| **`predict.py`** | **single CLI entry point: raw events → phase + function + health + review CSV** |

```
python src/features_v2.py --windows mixed --out pair_features_v2_extra.parquet
python src/cross_detector.py --windows mixed
python src/train_lgbm_v2.py --stage ranker      # then decode, function, final
python src/report_v2.py
python src/predict.py --events events.parquet --out preds.csv \
    --device-ids <guid>,<guid> --start "2024-12-03 12:00:00" --end "2024-12-03 12:30:00" \
    [--odot-tiebreak]
```

`partner_phase` in `pair_features_v2_extra.parquet` is a **join key only**, never a model input
(`train_lgbm_v2.KEY_EXCLUDE`); the decoder's neighbour features use channel *adjacency*, never a
channel→phase table. The ODOT wiring table is read in exactly one place: `tiebreak.py`.

---

## Splits (`dc_work\`)

* `folds.csv` — `DeviceId, fold`; 375 DEV signals. **fold 0 = the 38 `device_id_valid.csv` signals**
  (the 2025 model's hold-out); folds 1–5 = the 337 `device_id_train.csv` signals, `numpy` seed 0.
* `labels_dev.parquet` / `labels_test.parquet` — `DeviceId, Detector, Phase, Function`
  (5,171 / 590 rows). Never read the test one except for the authorised final scoring.

## Event cache — `dc_work\cache\events\`

Hive-partitioned parquet, **one directory per signal**, time-sorted inside each file:

```
cache/events/DeviceId=<guid>/d2_<uuid>.parquet   # 2024-12-02
                            d3_<uuid>.parquet
                            d4_<uuid>.parquet
```

Columns `DeviceId VARCHAR (from the partition), Timestamp TIMESTAMP, EventId UTINYINT, Parameter UTINYINT`.
225,412,568 rows, 421 signals, ~660 MB zstd (from 339 M raw rows).

Filters applied: `SELECT DISTINCT` (2.0 % of the raw rows are exact duplicates — stage-02 finding);
only the protocol's allowed codes `1, 7, 8, 9, 10, 11, 43, 44, 81, 82, 83–88, 131, 150, 173`;
rows with `EventId IN (81,82) AND Parameter > 64` (dummy detectors) dropped. Nothing else is removed.

Load one signal:

```python
con.sql("SELECT * FROM read_parquet('…/cache/events/DeviceId=<guid>/*.parquet') ORDER BY Timestamp")
# or many, with partition pruning:
con.sql("SELECT * FROM read_parquet('…/cache/events/**/*.parquet', hive_partitioning=true) "
        "WHERE DeviceId IN (…)")
```

### Event-code semantics (Indiana enumerations, verified against the data)

| code | meaning | Parameter |
|---|---|---|
| 1 / 8 / 10 / 11 | begin green / begin yellow / begin red clearance / end red clearance | phase |
| 7 / 9 | green termination / end yellow (≈ simultaneous with 8 / 10) | phase |
| 43 / 44 | phase call registered / dropped | phase |
| 81 / 82 | detector **off** / detector **on** | detector channel |
| 83–88 | detector restored (83) / fault codes (84–88) | detector channel |
| 131 | coordination pattern change | pattern (0 or 254 = free, 255 = flash) |
| 150 | coordinated phase yield point | coordinated phase |
| 173 | unit flash status | — |

## Derived cache tables (single parquet files, sorted by `DeviceId`)

| table | rows | columns |
|---|---|---|
| `det_intervals.parquet` | 71,315,610 | `DeviceId, Detector UTINYINT, t_on, t_off, dur DOUBLE` — each 82 paired with the next 81 on that channel |
| `phase_cycles.parquet` | 4,439,597 | `DeviceId, Phase, cyc, green_start, yellow_start, red_start, redclr_end, next_green, green_secs` — one row per begin-green |
| `green_state.parquet` | 5,850,948 | `DeviceId, t_start, t_end, mask` — per-signal timeline; **bit `Phase-1` of `mask` is set while that phase is green**. Makes every "p green and q not green" question an O(rows) numpy operation |
| `coord_state.parquet` | 5,600 | `DeviceId, t_start, t_end, pattern, is_coord` |
| `signal_meta.parquet` | 421 | `DeviceId, t0, t1, n_events, span_secs, n_cand, cand_phases UTINYINT[], coord_phases, coord_frac, n_det_channels` — **`cand_phases` is the candidate list (every phase with an event 1)** |
| `detector_meta.parquet` | 8,745 | `DeviceId, Detector, n_on, occ_secs, dur_{mean,med,q10,q90,q99,max}, frac_dur_*, n_hours_active, first_on, last_on, on_per_hour, occ_frac, n_fault_events, n_restore_events, unhealthy` |

`unhealthy` = any 84–88 fault event, or `dur_max > 900 s` (stuck on), or >50 % of ONs shorter than
0.15 s (chatter), or fewer than 20 ONs / 6 active hours (dead). It is a *feature*, not a filter —
nothing is dropped on it.

Stage 02's richer flag lives in `dc_work\atspm\detector_health.parquet`
(`health_flag` ∈ healthy / suspect / failed). `train_lgbm.py` **excludes `failed` channels from
training** but still scores them; `evaluate.py` reports every headline metric for all labelled
detectors *and* for classifiable ones (≥ 1 actuation in the window) with the coverage.

## Pair features — `dc_work\features\pair_features_windows.parquet`

One row per `(DeviceId, Detector, cand_phase, win)`; **all 421 signals**, every *active* detector
channel (labelled or not) × every candidate phase. `pair_features.parquet` is the `win == 'full72'`
slice, i.e. the protocol's one-row-per-(detector, candidate) table.

`win` identifies a time window (`features.WINDOWS_MIXED`): 4 × 30 min, 3 × 1 h, 2 × 3 h, 2 × 6 h,
2 × 24 h and the full 72 h, at different times of day including off-peak. Every feature is a
rate / share / lift, so it is duration-invariant; the amount of evidence is exposed through
`win_secs`, `det_n_on`, `det_on_per_hour`, `n_cycles`, `log_det_n_on`, `log_win_hours`.

**Phase-anonymity:** `DeviceId, Detector, cand_phase, win` are keys, never inputs.
`features.FEATURE_COLS(df)` returns the legal model inputs. No phase number, channel number or
label-derived quantity appears anywhere in them.

Feature families (see `features.py` for the exact SQL):

* colour state at each detector ON: `f_on_{green,yellow,red}`, `f_occ_*`, `dur_mean_{green,red}`
* time-since-begin-green / begin-red / to-next-green histograms `dtg_b*`, `dtr_b*`, `tog_b*`
* queue: `queue_occ_pre_green`, `f_on_red_last10`; discharge burst `burst_rate_g4/g8`
* **release**: `release_frac`, `release_frac_long`, `straddle_frac` — long occupancies that end just
  after this phase's begin green (the queue this phase released)
* call linkage: `call43_fwd_035/1`, `call43_fwd_lift` (chance-corrected), `call43_rev_*`, same for 44
* green-mask conditionals: `on_lift_green`, `occ_lift_green`, `solo_lift`, and the *exclusive-green*
  family `excl_lift_min/mean/max`, `excl_lift_other_*`, `excl_diff_min/mean`, `excl_partner_diff`
  (behaviour when p is green and a usually-concurrent partner q is not)
* green extension: `ext_corr_late`, `ext_green_gain`, `late2_frac`, `late_green_rate`
* coordination split: `on_lift_green_coord`, `on_lift_green_free`, `looks_recall`
* phase context: `cand_green_share`, `n_cycles`, `green_mean/sd/med`, `cycle_mean`, `call4*_per_cycle`
* detector level (function model): `det_n_on`, `det_dur_*`, `det_occ_frac`, `det_frac_short/long`
* rank features: for ~28 key features, `__rank` (pct rank), `__z`, `__mgap` (value − group max) and
  `__argmax` within the detector's candidate set

## Predictions — `dc_work\preds\`

Protocol format. Phase: `DeviceId, Detector, cand_phase, prob` (sums to 1 per detector).
Function: `DeviceId, Detector, p_advance, p_presence, p_count` (optional `p_other`).
`*_bywindow.parquet` adds a `win` column and is **not** contract format — it is for the
accuracy-vs-duration curve only.
