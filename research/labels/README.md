# Function labels, v2 — provenance

`function_labels_v2.parquet` is the detector-**function** truth from 2026-09-22 onward.
It replaces the raw config mapping for every purpose: training, ablation and scoring.

## Where it comes from

1. **Base** — the hand-maintained config export
   `%DC_WORK%/data/labels/detector_config_current.parquet` (8,659 channels / 466 signals,
   pulled 2026-09-21), free-text `Function` strings mapped to the five model classes by
   `%DC_WORK%/data/labels/function_label_map_v2.csv`. Channels outside 1–64 dropped →
   **8,633 rows**.
2. **Corrections** — the traffic engineer's review of the 79 function misses on the 143
   locked signals, `review/function_misclassified_locked143_REVIEWED.xlsx`. His rules,
   applied by `dc_work/trackA/a1_labels.py`:

   | `correct_function` cell | what happens |
   |---|---|
   | blank | keep the config label (10 rows) |
   | `?` | **drop** the detector from function training *and* scoring (4 rows) |
   | `Bike`, `Bike Loop`, `Departure` | → `Other` (7 rows; not classified classes yet) |
   | a class name | that class is the truth |

   Result: **37 labels changed, 38 confirmed, 4 dropped**. The changes are
   Presence→Count 13, Presence→Advance 8, Other→Advance 6, Advance→Other 3,
   Count→Presence 2, Presence→Other 2, Other→Presence 2, Other→Count 2, Count→Other 1.
   Every correction is on a NEWTEST (locked) signal, so **no training label moved**;
   what moved is the exam's truth.

## Columns

`DeviceId, DeviceName, Detector, cfg_phase, config_function` (raw string),
`func5_config` (the old mapped label), **`func5`** (the truth — NaN when dropped),
`label_source` ∈ {config, review_confirmed, review_confirmed_config, review_corrected,
review_dropped}, `reviewed`, `drop_from_use`, `review_raw`, `review_comment`,
`description` (the technician's channel text), `locked` ∈ {"", TEST, NEWTEST}.

**Use `func5`, and skip rows where it is null.**

## Effect (locked 143 signals, frozen `final_v2` predictions, full 66 h window)

| truth | 5-class | A/P/C |
|---|---|---|
| config label | .7539 | .7797 |
| corrected label | **.8675** | **.9156** |

## Next round

`review/function_disagreements_round2.xlsx` — 586 confident (p ≥ 0.80) model-vs-label
disagreements on 193 signals, all labelled signals this time, rows already reviewed in
round 1 removed. Sorted so the signals with the most confident disagreement come first.
Filling in `correct_function` / `comment` there and re-running `a1_labels.py` with the
second workbook extends this table.
