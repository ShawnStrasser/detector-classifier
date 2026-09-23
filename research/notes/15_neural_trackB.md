# 15_neural_trackB — a cheaper backbone for the neural half (B1, B2)

Track B of `AGENTS.md`, screened on **fold 0**, on exactly the rows of the stage-13 table (`13_gru_blend.md` §2):
the 22 out-of-fold windows, official timing labels, the scorable rule — 210,038 rows, 34,552 detector-windows.
Two columns each time, the **net alone** and the shipped **blend before the joint decoder** (weight 0.5 on the
trees). Folds 1–5 always keep the stage-13 GRU mixture, so the decoder trains on identical rows and only fold 0's
scored rows change; fed the stage-13 fold-0 GRU file the harness reproduces `blend_before.json`'s `_fold0` entry to
the last decimal. Code `research/code/neural/trackb_*.py`, artefacts `dc_work/trackB/`, logs `dc_work/logs/trackB_*.log`.
**The out-of-fold window set has no 15-minute family** (5 / 10 / 30 / 60 min …), so the tables read 5 / 10 / 30 / 60;
the 15-minute figures in `13_gru_blend.md` §4 are locked-signal scores and were not touched.

## B1 — TCN backbone instead of the BiGRU: **kept**

Same `data2`/`train2` pipeline, official labels, 709 training signals, phase loss only, 5–30 min windows.
`neural.models.TCN` (dilated 1-D CNN, 701 k weights) for `GRUAttn` (853 k).

| fold-0 held out | 5 min | 10 min | 30 min | 1 h |
|---|---|---|---|---|
| LightGBM alone | .9232 | .9363 | .9633 | .9606 |
| GRU net / TCN net | .9301 / .9203 | .9490 / .9423 | .9619 / .9573 | .9677 / .9534 |
| **GRU blend (shipped)** | **.9522** | **.9625** | **.9745** | **.9741** |
| **TCN blend** | **.9492** | **.9622** | **.9737** | **.9710** |
| TCN − GRU, blend | −0.30 pt | −0.04 pt | **−0.08 pt** | −0.31 pt |

**Cost.** Fold 0 trained in **11.6 min** against the GRU's **127 min** (23 epochs to plateau against 45) — about
**11×**. ONNX on the CPU, 4 threads, 120 pairs × 30 min, both benched in the same contended minute: **0.18 s** per
signal per window against the shipped GRU's **0.47 s** (~2.6×); 2.79 MB against 3.39 MB.

**Verdict.** At 30 min the blend is 0.08 pt behind — one standard deviation of the blend's own seed spread. The B1
rule ("within 0.3 pt or better ⇒ adopt, it is far cheaper") is met at 5, 10 and 30 min and missed by 0.01 pt at 1 h.
**TCN becomes the backbone.** Caveats: the *network alone* is clearly weaker (−0.46 pt at 30 min, −1.43 pt at 1 h)
and survives only because the trees carry the blend; past 3 h the TCN blend is ahead (+0.12 to +0.32 pt), which
production never sees because the network is off above 120 min; and B2 shows fold-0 seed 0 was the *worst* of three
TCN seeds, so −0.08 pt understates the backbone.
**ONNX** `dc_work/trackB/models/tb_tcn_f0.onnx`: opset 17, pair and time axes dynamic, function head dropped — the
`export_gru.py` contract. Against the PyTorch module it came from, on real rasters rendered by the production
`model/gru_input.py`: max absolute probability difference **1.7e-06**, argmax identical. Inside the 1e-4 bar.

## B2 — three-seed ensemble of the TCN on fold 0: **dropped**

Seeds 0/1/2, probabilities averaged per candidate then renormalised, i.e. before the blend.

| fold 0, blend before the decoder | 5 min | 10 min | 30 min | 1 h |
|---|---|---|---|---|
| seed 0 / 1 / 2 | .9492 / .9514 / .9512 | .9622 / .9643 / .9652 | .9737 / .9739 / .9775 | .9710 / .9741 / .9751 |
| mean of the three seeds | .9506 | .9639 | .9750 | .9734 |
| **three-seed ensemble** | **.9541** | **.9651** | **.9760** | **.9739** |
| ensemble − seed mean | +0.35 pt | +0.12 pt | **+0.10 pt** | +0.05 pt |
| single-seed sd | 0.12 pt | 0.16 pt | 0.21 pt | 0.21 pt |

**Verdict.** Ensembling buys **+0.10 pt at 30 min** over the average seed — half a standard deviation, well under
the 0.3 pt bar. **Dropped.** Against seed 0 alone it looks like +0.23 pt, but seed 0 is the worst of the three, so
that is seed luck, not ensembling. The net alone gains more (+0.43 pt at 30 min over the seed mean): averaging with
LightGBM already damps most of the seed noise an ensemble would remove. Carry forward that the **TCN blend is ~3×
noisier across seeds than the GRU blend**: like for like at 30 min (blend *after* the decoder, the quantity stage 13
measured) 0.19 pt sd against 0.069 pt, and 0.21 pt before the decoder; the net alone is 0.33 pt against 0.165 pt — so
judge later steps against ~0.2 pt. Taking the three-seed TCN mean (.9750) against the GRU's seed-0 blend (.9745), the
two backbones are indistinguishable at 30 minutes — the honest reading of B1.

## GPU hours, two bugs, one correction

B1 0.19 h train + 0.07 h infer; B2 0.62 h train + 0.14 h infer — **Track B so far 1.0 h**. About **11.5 h of the
first night were lost** to the first bug, not to computation. (1) The chain died at its first gate: `run_trackb.sh`
built a json by embedding a path inside a `python -c` source string, and Git Bash rewrites path-shaped *arguments*
for a Windows interpreter but never text inside a quoted source string, so Python was asked to open `/c/Users/...`.
Paths in a `-c` body must be passed as arguments (`sys.argv`), as `run_trackb_b2.sh` does. (2) `research/code/rpath.py`
skipped directories already on `sys.path`, leaving `research/code/` *behind* `model/`, so every research script died
on `from common import DC_WORK`; it now re-inserts instead of skipping. (3) `STATUS.md` recorded the GRU's fold-0
inner-validation score as .9185; it is **.9528** (stage-13 seeds .9528 / .9477 / .9532), so the TCN's .9501 sits
inside that spread and inner validation never was evidence for the TCN — the held-out table is.
