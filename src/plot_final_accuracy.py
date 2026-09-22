"""Draw docs/img/final_accuracy_vs_minutes.png from final_accuracy_vs_minutes.json.

Same style as the beta chart (`plot_accuracy_vs_minutes.py`): smooth monotone lines, no
dots, plain descriptive titles, log x axis, second panel = share of detectors with enough
actuations to get a prediction.  `docs/img/accuracy_vs_minutes.png` (the beta) is not touched.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "img")
d = json.load(open(os.path.join(IMG, "final_accuracy_vs_minutes.json")))
raw = d["by_duration_raw"]
x = np.log10([r["minutes"] for r in raw])
w = np.array([r["n_scored"] for r in raw])
grid = np.linspace(x.min(), x.max(), 400)


def smooth(key, weights):
    """More data never hurts: monotone (isotonic) fit, then a gaussian blur in log-minutes."""
    y = np.array([r[key] for r in raw])
    iso = IsotonicRegression(increasing=True).fit(x, y, sample_weight=weights)
    step = iso.predict(grid)
    pad = 80
    ext = np.concatenate([np.full(pad, step[0]), step, np.full(pad, step[-1])])
    k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / 22.0) ** 2)
    return np.convolve(ext, k / k.sum(), mode="same")[pad:-pad]


phase = smooth("acc_phase", w)
func = smooth("acc_function", np.array([r["n_func"] for r in raw]))
ans = smooth("answered_share", np.ones_like(w))

BLUE, GREEN, PURPLE, GREY = "#2474d8", "#14ad78", "#4a36a8", "#555555"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 7.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.22})
fig.patch.set_facecolor("#fcfcfa")
for a in (a1, a2):
    a.set_facecolor("#fcfcfa"); a.grid(color="#dddddd", lw=0.9)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(colors=GREY, labelsize=11, length=0)

a1.plot(grid, phase * 100, color=BLUE, lw=3, solid_capstyle="round")
a1.plot(grid, func * 100, color=GREEN, lw=3, solid_capstyle="round")
lo = int(min(func.min(), phase.min()) * 100 // 5 * 5)          # never clip a line
a1.set_ylim(lo, 100); a1.set_yticks(range(lo, 101, 5))
a1.set_yticklabels([f"{v}%" for v in range(lo, 101, 5)])
a1.set_title("Accuracy of predicted phase assignment and detector function",
             loc="left", fontsize=14, fontweight="bold")
a1.text(grid[-1] + 0.05, phase[-1] * 100, f"Phase {phase[-1]*100:.0f}%", color=BLUE,
        fontsize=12, fontweight="bold", va="center")
a1.text(grid[-1] + 0.05, func[-1] * 100, f"Function {func[-1]*100:.0f}%", color=GREEN,
        fontsize=12, fontweight="bold", va="center")

a2.plot(grid, ans * 100, color=PURPLE, lw=3, solid_capstyle="round")
a2.set_ylim(0, 100); a2.set_yticks([0, 25, 50, 75, 100])
a2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
a2.set_title("Share of detectors with enough actuations to get a prediction",
             loc="left", fontsize=14, fontweight="bold")
a2.text(grid[-1] + 0.05, ans[-1] * 100, f"{ans[-1]*100:.0f}%", color=PURPLE,
        fontsize=12, fontweight="bold", va="center")
a2.text(0.99, 0.08, "Detectors with fewer than 5 actuations in the sample\n"
                    "are reported as “not enough data” instead of guessed.",
        transform=a2.transAxes, ha="right", va="bottom", fontsize=10, color=GREY)

ticks = [1, 5, 30, 60, 180, 360, 720, 1440, 4320]
ticks = [t for t in ticks if np.log10(t) <= grid[-1] + 0.02]
a2.set_xticks(np.log10(ticks))
a2.set_xticklabels(["1 min", "5 min", "30 min", "1 h", "3 h", "6 h", "12 h", "1 day",
                    "3 days"][:len(ticks)])
a2.set_xlim(grid[0] - 0.05, grid[-1] + 0.78)
a2.set_xlabel("how much data is in the sample", fontsize=12, color=GREY)
fig.suptitle("Detector phase & function prediction accuracy\n"
             "vs. amount of hi-res data in the sample",
             x=0.06, y=0.985, ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.06, 0.853,
         f"Final model, tested on {d.get('n_signals', 143)} ODOT signals "
         f"({d['n_labelled_detectors']:,} detectors) it had never seen: they were locked away\n"
         "and used for nothing until this one test. Sample length from 1 minute to 2.75 days, "
         f"averaged over\n{len(d['anchors'])} start times across day, night and weekend.",
         fontsize=10.5, color=GREY)
fig.subplots_adjust(top=0.785, left=0.09, right=0.955, bottom=0.09)
fig.savefig(os.path.join(IMG, "final_accuracy_vs_minutes.png"), dpi=150,
            facecolor=fig.get_facecolor())

d["by_duration_smoothed"] = [dict(minutes=round(float(10 ** g), 3),
                                  acc_phase=round(float(p), 4),
                                  acc_function=round(float(f), 4),
                                  answered_share=round(float(s), 4))
                             for g, p, f, s in zip(grid, phase, func, ans)]
d["smoothing"] = ("Monotone (isotonic, weighted by detectors scored) fit in log-minutes, "
                  "then gaussian blur. Raw averages are in by_duration_raw.")
json.dump(d, open(os.path.join(IMG, "final_accuracy_vs_minutes.json"), "w"), indent=1)
for m in (1, 5, 15, 30, 60, 360, 1440, 3960):
    i = int(np.argmin(np.abs(grid - np.log10(m))))
    print(m, round(phase[i], 3), round(func[i], 3), round(ans[i], 3))
