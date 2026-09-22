# 02 — atspm vs raw events, and detector health
## 1. What atspm (v2.4.0, source read) does to raw events
- **Load = `SELECT DISTINCT`** + range sanity on EventId/Parameter — the *only* cleaning (no Parameter>64 filter).
  On 60 DEV signals / 3 d: **2.0 % of 81/82 rows are exact duplicates**; dropping them cuts apparent unmatched
  OFF (81→81) by 99 % (193,979→1,914) and unmatched ON (82→82) by 28 % (738k→534k).
- **atspm never pairs 81/82.** `timeline` pairs 1/7, 8/9 (>10 s ⇒ invalid), 10/11, 43/44, 173/174 flash and
  **84-88→83 fault intervals**, each with `IsValid`; occupancy/ON-duration is ours to compute either way.
- `has_data` = device online per bin (device-level, not per detector); `actuations` = COUNT(82) per
  (bin, DeviceId, Detector): config-free, phase-anonymous.
- **`detector_health` is infeasible on 3 days.** Not SQL: `traffic_anomaly.decompose`+`.anomaly` on binned
  actuations with `rolling_window_days=7, drop_days=7, min_rolling_window_samples=96*5,
  min_time_of_day_samples=7` **plus a weekly seasonal term** ⇒ needs ≥ ~2 weeks. Hence the table below.
- **Circular (config-requiring):** `arrival_on_green` (Function='Advance'+Phase), `split_failures` ('Presence'),
  `yellow_red` ('Yellow_Red') consume the detector→phase map we are inferring. They *could* run once per
  candidate phase with a synthetic config (≈9× cost), but AOG's "% actuations on green" is the same pair feature
  stage 01 builds from raw far more cheaply, and 15-min bins destroy the sub-second ON/OFF timing that carries
  the phase signal. Allowed **and** config-free: only `actuations`, `has_data`, `timeline`.
## 2. Health table — `dc_work\atspm\detector_health.parquet`
All 424 signals × 9,057 channels (labeled + unlabeled, label-free), 35 cols: n_on/n_off, unmatched ON/OFF rates,
frac_time_on, longest_on_s, median_on_s, max_day_gap_s (06-20 h), chatter_rate (<0.1 s ONs), max_on_per_min,
per-day counts + day_ratio, fault counts 83-88, health_flag/health_reason. Pure DuckDB (6 GB/6 threads), **~2.5 min for all
421 signals × 3 d** — no subsetting needed. Code `src/build_detector_health.py`, rules `src/health.py`,
analysis `src/analyze_detector_health.py`.
## 3. Numbers (DEV only; TEST never touched)
| population | healthy | suspect | failed |
|---|---|---|---|
| 5,171 labeled DEV detectors | 3,812 (73.7 %) | 1,026 (19.8 %) | 329 (6.4 %) + 4 ch>64 |
| old-baseline fold-0 top-1 accuracy | **95.4 %** (n=392) | **91.4 %** (n=93) | **25.0 %** (n=20) |

- `failed` = 4 % of fold-0 detectors but **36.6 %** of its errors; failed+suspect = 22 % of detectors, 56 % of
  errors. Refusing to classify `failed` moves fold 0 from 91.9 % → **94.6 % at 96 % coverage**.
- **Configured detectors with no events at all: 330 = 297 DEV + 33 TEST** (308 on channel ≤64; the user's "318"
  sits between those filters). Of the 297 DEV, 69 are in 8 wholly dead signals, 228 are genuine per-channel
  failures. Old baseline scores **6.3 %** on them (mean top prob 0.11).
- Only *absence of actuations* predicts error. Per-criterion fold-0 accuracy: no events 6 %, <20 act/day 21 %,
  <100 act/day 43 %, faults 83-88 87 %, daytime gap ≥6 h 92 % — **but** unmatched-ON >25 %, longest ON ≥1 h,
  day_ratio <0.25 and stuck-ON all → 100 %. Surprise: stuck/chattering/unpaired detectors are rare (26 stuck in
  5,171) and **do not hurt accuracy**.
## 4. Recommendation
- **(a) Feed raw, deduplicated events; do not route the model through atspm.** Adopt three atspm *ideas*, re-
  implemented in `src/health.py` (~30 lines SQL, no dependency, no circularity): `SELECT DISTINCT` dedup
  (mandatory before ON/OFF pairing), `has_data`-style outage masking, 84-88→83 fault + 173/174 flash intervals.
- **(b)** `src/health.py`: `load_health()`, `flag_detectors()` (rules identical to the SQL that built the parquet,
  verified row-for-row), `is_classifiable()`, `mask_intervals_sql()` → (DeviceId, Detector, mask_start, mask_end,
  reason) for fault/flash/stuck-ON periods. Stages should **drop `failed` from training**, keep it at inference,
  and use `health_flag` / `on_per_day` / `median_on_s` as features.
- **(c)** Never give a `failed` detector a phase — `status_for_user()` emits *"cannot classify: detector produced
  no actuations in the analysis window"*; `suspect` → predict but mark *"low confidence: data quality issue
  (<reason>)"* and add to the manual-review list.
- Caveat: only `no_events` (n=16, fold 0) is strongly evidenced; the other 3 `failed` rules rest on n=4 (+36 detectors DEV-wide). Re-check on 6-fold OOF.
