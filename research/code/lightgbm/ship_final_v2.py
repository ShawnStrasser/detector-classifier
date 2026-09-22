"""Assemble the shipped weights folder (`model/weights/`) and write its model card.

final_v2 = final_v1 (the LightGBM files, copied byte for byte) + the refit GRU pair
scorer (`gru.onnx`, run by onnxruntime on the CPU) + `blend.json`, the three numbers
frozen on out-of-fold data before the locked test signals were opened a second time.

    python src/official/ship_final_v2.py --stage copy    # byte-identical tree files
    python src/official/ship_final_v2.py --stage card    # model_card.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402  -- research/code/** and model/ on sys.path
from common import DC_WORK, REPO  # noqa: E402

V1 = REPO / "models" / "final_v1"
V2 = REPO / "model" / "weights"          # the shipped folder
BLEND = DC_WORK / "official" / "blend_v2"
RUNS = DC_WORK / "neural" / "runs2"
TREE_FILES = ["phase_lgbm_v4.json", "phase_lgbm_v4_s0.txt", "phase_lgbm_v4_s1.txt",
              "phase_lgbm_v4_s2.txt", "decode_lgbm_v4.json", "decode_lgbm_v4.txt",
              "function_lgbm_v4.json", "function_lgbm_v4.txt"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_copy(a) -> None:
    V2.mkdir(parents=True, exist_ok=True)
    for f in TREE_FILES:
        shutil.copy2(V1 / f, V2 / f)
    bad = [f for f in TREE_FILES if sha(V1 / f) != sha(V2 / f)]
    log(f"copied {len(TREE_FILES)} tree files; byte-identical: {not bad} {bad}")


def _load(p: Path):
    return json.load(open(p)) if p.exists() else None


def stage_card(a) -> None:
    card = json.load(open(V1 / "model_card.json"))
    after = _load(BLEND / "blend_after.json")
    before = _load(BLEND / "blend_before.json")
    seeds = _load(BLEND / "blend_seeds.json")
    bench = _load(BLEND / "runtime_bench_onnx_t4.json")
    cfg = json.load(open(V2 / "blend.json"))
    runs = {p.stem: _load(p) for p in sorted(RUNS.glob("gru2_*.json"))
            if not p.name.endswith(".done.json")}

    fam_min = {"m5": 5, "m10": 10, "m30": 30, "h1": 60, "h3": 180, "h6": 360,
               "h24": 1440, "full": "full span"}
    w = str(cfg["weight_lightgbm"])
    held = {}
    if after and before:
        for fam, mins in fam_min.items():
            if fam not in after["lightgbm"]:
                continue
            held[fam] = {
                "minutes": mins,
                "lightgbm_final_v1": round(after["lightgbm"][fam]["acc"], 5),
                "gru_alone": round(after["gru"][fam]["acc"], 5),
                "blend_before_decoder_shipped": round(
                    before["weights"][w][fam]["acc"], 5),
                "blend_after_decoder": round(after["blend_50_50"][fam]["acc"], 5),
                "gain_pt_shipped": round(100 * (before["weights"][w][fam]["acc"] -
                                                after["lightgbm"][fam]["acc"]), 3),
                "n_detectors": after["lightgbm"][fam]["n"]}

    card["name"] = "final_v2"
    card["date"] = time.strftime("%Y-%m-%d")
    card["supersedes"] = ("models/final_v1 (2026-09-21) -- the LightGBM files in this "
                          "folder are byte-for-byte copies of it")
    card["what_changed"] = (
        "A small recurrent network that reads the raw second-by-second detector and "
        "phase trace is averaged into the phase model on short samples. Nothing else "
        "changed: the three tree files, the joint decoder and the 5-class function head "
        "are the same files as final_v1, and the function model still reads the "
        "trees-only phase, so function output is unchanged.")
    card["pipeline"] = [
        "LightGBM pair ranker, 3 seeds averaged",
        f"GRU + attention pair scorer on the 1 s raster, averaged in at weight "
        f"{1 - cfg['weight_lightgbm']} on samples up to {cfg['cutoff_minutes']} minutes "
        f"(onnxruntime on the CPU, src/gru_onnx.py)",
        "LightGBM joint per-signal decoder, run on the mixture",
        "LightGBM 5-class function head on the trees-only predicted phase",
        "detector-health status + minimum-evidence rule"]
    card["blend"] = {
        "weight_lightgbm": cfg["weight_lightgbm"],
        "weight_gru": round(1 - cfg["weight_lightgbm"], 3),
        "where": cfg["where"],
        "cutoff_minutes": cfg["cutoff_minutes"],
        "chunk_minutes": cfg["chunk_minutes"],
        "runtime": "onnxruntime on the CPU (src/gru_onnx.py) -- the only runtime; no "
                   "torch, no numpy GRU, no optional import and no fallback",
        "how_frozen": ("the weight was searched on folds 1-5 of the 701 training "
                       "signals and checked on fold 0, which never saw the search; the "
                       "location and the cut-off were chosen on the same out-of-fold "
                       "predictions. All three were fixed before the locked test signals "
                       "were opened."),
        "held_out_phase_accuracy_by_sample_length": held,
        "seed_noise": (seeds or {}).get("sd_pt"),
    }
    card["neural_model"] = {
        "file": "gru.onnx",
        "format": "ONNX opset 17, pair and time axes dynamic",
        "architecture": ("conv stem (9 channels -> 64, kernel 7, stride 4, batch-norm "
                         "folded into the convolution at export) -> 3-layer "
                         "bidirectional GRU, 128 hidden -> learned-query attention "
                         "pooling concatenated with mean and max -> 128-wide projection "
                         "-> one score per (detector, candidate phase) pair, softmaxed "
                         "across the signal's candidates"),
        "inputs": ("one 1 s raster per pair, 9 channels: detector occupancy and ON rate; "
                   "the candidate phase's green / yellow / red-clearance / call "
                   "fractions; and number-free context -- how many OTHER phases are "
                   "green, the share of other phases called, and whether the signal is "
                   "coordinated. No phase number and no channel number is ever an input."),
        "parameters": (runs.get("gru2_final") or {}).get("params"),
        "label_source": "official controller timing (call_phase)",
        "training": {
            "signals": (runs.get("gru2_final") or {}).get("n_train_signals"),
            "window_lengths_minutes": (runs.get("gru2_final") or {}).get("train_minutes"),
            "epochs_to_plateau": {k: v.get("best_epoch") for k, v in runs.items() if v},
            "plateaued": {k: v.get("plateaued") for k, v in runs.items() if v},
            "note": ("phase loss only -- the network's function head was not trained and "
                     "is not exported; detectors with no actuation inside the window are "
                     "dropped from the loss")},
        "onnx_vs_pytorch": (bench or {}).get("onnx_vs_torch"),
        "cpu_speed": (bench or {}).get("speed"),
    }
    card["inference"]["requirements"] = (
        "numpy, pandas, duckdb, pyarrow, onnxruntime (lightgbm optional -- "
        "src/lgbm_numpy.py evaluates the trees with numpy only; onnxruntime is required "
        "because it is the single runtime of the neural half)")
    sizes = {p.name: p.stat().st_size for p in sorted(V2.glob("*.txt"))}
    sizes["gru.onnx"] = (V2 / "gru.onnx").stat().st_size
    card["model_files_bytes"] = sizes
    card["total_model_bytes"] = int(sum(sizes.values()))
    card["file_sha256"] = {p.name: sha(p) for p in sorted(V2.glob("*"))
                           if p.name != "model_card.json"}
    json.dump(card, open(V2 / "model_card.json", "w"), indent=1, default=str)
    log(f"wrote {V2/'model_card.json'} ({card['total_model_bytes']/1e6:.1f} MB of models)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["copy", "card"])
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
