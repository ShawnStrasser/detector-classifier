# detector-classifier
Predicting traffic signal detector phase assignment (primary) and detector function (secondary) from
hi-res event logs, to aid in configuring detectors for ATSPMs. Work in progress.

**Beta model (2026-09-17): [docs/BETA_REPORT.md](docs/BETA_REPORT.md)** — how to run it, required EventIds, accuracy
(phase 98% / function 90% on held-out signals with 72 h of data; 96% / 88% with 30 minutes), model in `models/beta_v0`,
entry point `src/predict.py`. Research is ongoing (`results/`, `docs/research_protocol.md`).

Background review of the 2024-25 work: [docs/research_review_2026-09.md](docs/research_review_2026-09.md)

![accuracy vs minutes](docs/img/accuracy_vs_minutes.png)

## Layout

| Path | Contents |
|---|---|
| `docs/` | Indiana hi-res enumerations PDF; review of the 2024-25 work and next-wave plan |
| `data/raw/` | `Train_Dec_{2,3,4}_2024.parquet` - 3 days, 421 signals, all event codes (DeviceId, Timestamp, EventId, Parameter); `detector-configs.csv` labels (DeviceId, Phase, Function, Detector) |
| `data/splits/` | Device-level split: `device_id_train.csv` (337), `device_id_valid.csv` (38), `test_config.csv` / `Test_Devices.parquet` (43 held-out signals), `train_config.csv` |
| `data/statewide_2025-02-25/` | 6 h statewide pull (events 1,7,43,44,81,82 only), `all_configs.csv` labels (detector column is named `Parameter`; free-text functions), DuckDB with the Feb 2025 inference results |
| `baseline/` | The 2025 model to beat: dual-head BiLSTM (`models.py`, `model5.pth`), its data prep, inference notebook and predictions. Paths inside these files predate the reorganisation. |
| `archive/` | Everything else from the first wave (notebooks, other checkpoints, derived feature files, Power BI). Not used going forward. |

`data/` and `archive/` are not versioned. Datasets are also on the
[Kaggle competition page](https://www.kaggle.com/c/traffic-signal-detector-classifier).

## Baseline result (Feb 2025, per detector, clean labels)

Phase 95.5% / function 92.0% with no abstention; ~98% / ~96% on the ~91% of detectors with confidence >= 0.7.
Errors are dominated by concurrent phase pairs (2<->6, 4<->8). Not a clean held-out number - see the review.
