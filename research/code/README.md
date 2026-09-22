# Research code — index

Kept for posterity. None of it is needed to run the model: that is `../../model/`.
It reads and writes a local work directory (`%DC_WORK%`, default `~/dc_work`) that is
not in this repository, so most of it cannot be re-run as is.

Every script starts by importing `rpath`, which puts every folder here **and**
`../../model/` on `sys.path`; the feature and decoder *definitions* live with the model
and are imported from there, so there is exactly one copy of each.

The stage notes in `../notes/` refer to the old `src/…` paths. `src/predict.py` and the
modules it imported are now `model/`; `src/official/*`, `src/tune/*`, `src/statewide/*`,
`src/contrast/*`, `src/neural/*`, `src/old_baseline/*` are the folders below.

| file | what it did |
|---|---|
| **`common.py`** | work-dir paths, event codes, fold count, the ODOT wiring table (evaluation only), capped DuckDB connection |
| **`rpath.py`** | the import-path helper every script uses |
| **`features/`** | |
| `build_cache.py` | raw events → folds, label tables, event cache, derived tables (`--step labels\|events\|derived\|check`) |
| `build_features.py` | the training feature tables: window sets, cache loader, CLI over the shipped builders (`--what base\|extra\|sim\|yrlag`) |
| `build_detector_health.py` | the per-detector health table from raw events |
| `health_sql.py` | the DuckDB form of the health rules, bad-period masking, the cached-table reader |
| `build_stg_cache.py` / `build_stg_features.py` / `build_stg_flat.py` | the same, for the Sept-2026 pull |
| `windows_stg.py` | the Sept-2026 window anchors (registers them with `build_features`) |
| **`labels/`** | |
| `labels_official.py` | official controller timing → the phase truth table |
| `make_label_map.py` | free-text function strings → the five classes |
| `make_split.py` / `split_counts.py` | the locked TEST / NEWTEST splits and their counts |
| `analyse_labels.py` | hand config vs controller timing: how far they agree, and how stably |
| `disagreements.py` / `make_disagreements.py` | confident model-vs-label disagreements, with an evidence sentence each |
| `resolve_review.py` | folds returned review files back into the label tables |
| `trust.py` | per-signal label-trust summary |
| **`lightgbm/`** | |
| `train_lgbm.py` | stage 01: the first pair ranker, ablations, the scoring API |
| `train_lgbm_v2.py` | stage 04: ranker with the partner features, ablations, decoder, function head |
| `decode_train.py` | training and calibration of the joint per-signal decoder |
| `function_v2.py` / `function_v3.py` / `function_v4.py` | the three generations of the function head; v4 is what ships |
| `train_official.py` / `run_train.py` / `score_variants.py` | stage 10: refitting on the official timing labels, variant comparison |
| `fit_final.py` / `fit_final_v1.py` | the candidate and the final LightGBM fit (`--stage oof\|models\|card`) |
| `ship_final_v2.py` | assembles the shipped weights folder and its model card |
| **`neural/`** | |
| `raster.py` / `ncache.py` / `ncache2.py` | the 1 s, 9-channel raster and its cache (two generations) |
| `data.py` / `data2.py` | datasets and samplers |
| `models.py` | GRU, TCN, conv-GRU, transformer, 2-D cycle CNN |
| `train.py` / `train2.py` | training (stage 03, then the stage-13 refit) |
| `infer.py` / `infer2.py` / `score_neural.py` / `score_minact.py` | out-of-fold and held-out scoring |
| `arch_select.py` / `pick_best.py` / `summarize.py` / `concat_oof.py` | the architecture comparison and its bookkeeping |
| `export_gru.py` / `bench_gru.py` / `gru_speedup.py` | ONNX export, the runtime benchmark, the research-only GPU forward pass |
| `blend_v2.py` / `blend_v2_predecode.py` / `blend_check.py` / `gru_old_vs_new.py` | the blend: weight, where to apply it, cut-off, and the refit comparison |
| **`evaluation/`** | |
| `evaluate.py` | the scoring harness: top-1, non-standard wiring, concurrent pairs, coverage curves, function metrics |
| `error_forensics.py` | what the remaining errors actually are |
| `score_final.py` / `score_final_v2.py` / `score_official.py` | the scorings of the locked signals |
| `curve_final.py` / `plot_final_accuracy.py` | the accuracy-vs-sample-length curve and its chart |
| `analyze_detector_health.py` | how much of the population is suspect or failed, and how much it costs |
| **`experiments/`** (things that did not ship) | |
| `old_baseline/` | re-running and scoring the 2025 BiLSTM on the same hold-out |
| `tune/` | Optuna, seed bagging, XGBoost/CatBoost/forests, feature pruning |
| `statewide/` | the six-event-code statewide pull and the Other / Yellow_Red classes |
| `contrast/` | peak vs off-peak demand-contrast features |
| `health2/` | detector-health masking and a learned trust score |
| `overlaps/` | overlaps as a predictable output class |
| `delay_extend.py` | whether programmed delay / extend settings help |
| **`download/`** | agency-specific pull scripts — git-ignored, never committed |
