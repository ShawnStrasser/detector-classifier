# detector-classifier
Predicting traffic signal detector phase assignment (primary) and detector function (secondary) from
hi-res event logs, to aid in configuring detectors for ATSPMs.

**Final model (2026-09-21): [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)** — what it does, how to run it,
required EventIds, accuracy, what worked and what did not. On 186 signals locked away from the whole
study and scored once: **phase 98.6 %** with 2¾ days of data, 98.2 % with 6 hours, 97.0 % with 30 minutes;
function 75–85 %. Model in `models/final_v1`, entry point `src/predict.py`, CPU only, numpy is enough.

![accuracy vs minutes](docs/img/final_accuracy_vs_minutes.png)

This replaces the September beta ([docs/BETA_REPORT.md](docs/BETA_REPORT.md); its files are in the git history). Stage-by-stage research notes are in
`results/`; the rules everything was run under are in [docs/research_protocol.md](docs/research_protocol.md).
Background review of the 2024-25 work: [docs/research_review_2026-09.md](docs/research_review_2026-09.md)

## Layout

| Path | Contents |
|---|---|
| `src/predict.py` | the only file an integrator needs: raw events → one row per detector |
| `models/final_v1/` | the shipped models (3-seed phase ranker, joint decoder, 5-class function head) + `model_card.json` |
| `docs/` | final report, beta report, research protocol, Indiana hi-res enumerations PDF, charts |
| `results/` | one short note per research stage (00 … 11) |
| `tests/` | smoke tests, phase-number-invariance test, numpy-backend equality test |
| `data/raw/` | `Train_Dec_{2,3,4}_2024.parquet` - 3 days, 421 signals (DeviceId, Timestamp, EventId, Parameter); `detector-configs.csv` hand labels |
| `data/splits/` | `device_id_train.csv` (337), `device_id_valid.csv` (38), `test_config.csv` (43 locked signals) |
| `data/statewide_2025-02-25/` | 6 h statewide pull (events 1,7,43,44,81,82 only) and its labels |
| `baseline/` | the 2025 dual-head BiLSTM this replaces (`models.py`, `model5.pth`) |
| `archive/` | first-wave notebooks and checkpoints. Not used going forward. |

`data/` and `archive/` are not versioned. Datasets are also on the
[Kaggle competition page](https://www.kaggle.com/c/traffic-signal-detector-classifier).
