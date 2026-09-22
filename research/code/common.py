"""Shared paths, constants and helpers for the detector-classifier project.

All heavy artefacts live under DC_WORK (fast local disk); only code / small docs
live in the repo.  See docs/research_protocol.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

# Heavy work dir (training only).  Inference never touches it -- see src/predict.py.
DC_WORK = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[2]          # the repository root

RAW = DC_WORK / "data" / "raw"
SPLITS = DC_WORK / "data" / "splits"
CACHE = DC_WORK / "cache"
FEATURES = DC_WORK / "features"
PREDS = DC_WORK / "preds"
MODELS = DC_WORK / "models"
LOGS = DC_WORK / "logs"
TMP = DC_WORK / "tmp"

FOLDS_CSV = DC_WORK / "folds.csv"
LABELS_DEV = DC_WORK / "labels_dev.parquet"
LABELS_TEST = DC_WORK / "labels_test.parquet"

RAW_DAYS = {2: RAW / "Train_Dec_2_2024.parquet",
            3: RAW / "Train_Dec_3_2024.parquet",
            4: RAW / "Train_Dec_4_2024.parquet"}

# ---------------------------------------------------------------- event codes
# Indiana hi-res enumerations (docs/Indiana ... Enumerations.pdf).
# Only these may be used (research_protocol.md "Allowed event codes").
EV_BEGIN_GREEN = 1
EV_GREEN_TERM = 7
EV_BEGIN_YELLOW = 8
EV_END_YELLOW = 9
EV_BEGIN_RED_CLEAR = 10
EV_END_RED_CLEAR = 11
EV_CALL_ON = 43          # phase call registered
EV_CALL_OFF = 44         # phase call dropped
EV_DET_OFF = 81          # detector off  (Parameter = detector channel)
EV_DET_ON = 82           # detector on
EV_DET_FAULT = (83, 84, 85, 86, 87, 88)   # restored / other / watchdog / open / short / excessive-change
EV_COORD_PATTERN = 131   # Parameter = pattern (0 or 254 => free, 255 => flash)
EV_COORD_YIELD = 150     # Parameter = coordinated phase
EV_FLASH = 173

ALLOWED_EVENTS = (1, 7, 8, 9, 10, 11, 43, 44, 81, 82,
                  83, 84, 85, 86, 87, 88, 131, 150, 173)

MAX_DETECTOR_CHANNEL = 64   # Parameter > 64 on 81/82 are dummy detectors -> dropped

# ODOT standard wiring.  POST-PROCESSING / EVALUATION ONLY -- never a model input.
DEFAULT_PHASE = dict(zip(range(1, 41),
                         [1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 1, 3, 5, 6, 6, 6, 6, 6,
                          7, 8, 8, 8, 8, 8, 5, 7, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8]))

# ring/barrier concurrent or opposing pairs used for error grouping
CONCURRENT_PAIRS = {frozenset(p) for p in
                    [(2, 6), (4, 8), (1, 5), (3, 7), (1, 6), (2, 5), (3, 8), (4, 7)]}

FUNCTIONS = ("Advance", "Presence", "Count")
FUNCTIONS4 = ("Advance", "Presence", "Count", "Other")

N_FOLDS = 6


def connect(memory_limit: str = "10GB", threads: int = 10,
            read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the resource caps mandated by the protocol."""
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def events_glob() -> str:
    """Glob for the cached event store (hive partitioned by DeviceId)."""
    return (CACHE / "events" / "**" / "*.parquet").as_posix()


def device_events_glob(device_id: str) -> str:
    return (CACHE / "events" / f"DeviceId={device_id}" / "*.parquet").as_posix()


def cache_table(name: str) -> str:
    """Path glob for a non-partitioned cache table."""
    return (CACHE / f"{name}.parquet").as_posix()


def is_concurrent_pair(a: int, b: int) -> bool:
    return frozenset((a, b)) in CONCURRENT_PAIRS
