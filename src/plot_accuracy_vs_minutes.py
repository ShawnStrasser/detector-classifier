"""Redraw docs/img/accuracy_vs_minutes.png from the numbers in accuracy_vs_minutes.json (smooth lines only)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "img")
d = json.load(open(os.path.join(IMG, "accuracy_vs_minutes.json")))
raw = d["by_duration_raw"]
x = np.log10([r["minutes"] for r in raw])
w = np.array([r["n_scored"] for r in raw])
grid = np.linspace(x.min(), x.max(), 400)


def smooth(key, weights):
    """More data never hurts: monotone (isotonic) fit, then a gaussian blur in log-minutes to round the steps."""
    y = np.array([r[key] for r in raw])
    iso = IsotonicRegression(increasing=True).fit(x, y, sample_weight=weights)
    step = iso.predict(grid)
    pad = 80
    ext = np.concatenate([np.full(pad, step[0]), step, np.full(pad, step[-1])])
    k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / 22.0) ** 2)
    return np.convolve(ext, k / k.sum(), mode="same")[pad:-pad]


phase, func, ans = smooth("acc_phase", w), smooth("acc_function", w), smooth("answered_share", np.ones_like(w))

BLUE, GREEN, PURPLE, GREY = "#2474d8", "#14ad78", "#4a36a8", "#555555"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.22})
fig.patch.set_facecolor("#fcfcfa")
for a in (a1, a2):
    a.set_facecolor("#fcfcfa"); a.grid(color="#dddddd", lw=0.9)
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(colors=GREY, labelsize=11, length=0)

a1.plot(grid, phase * 100, color=BLUE, lw=3, solid_capstyle="round")
a1.plot(grid, func * 100, color=GREEN, lw=3, solid_capstyle="round")
a1.set_ylim(70, 100); a1.set_yticks(range(70, 101, 5)); a1.set_yticklabels([f"{v}%" for v in range(70, 101, 5)])
a1.set_title("How accurate are the answers?", loc="left", fontsize=14, fontweight="bold")
a1.text(grid[-1] + 0.05, phase[-1] * 100, f"Phase {phase[-1]*100:.0f}%", color=BLUE, fontsize=12, fontweight="bold", va="center")
a1.text(grid[-1] + 0.05, func[-1] * 100, f"Function {func[-1]*100:.0f}%", color=GREEN, fontsize=12, fontweight="bold", va="center")

a2.plot(grid, ans * 100, color=PURPLE, lw=3, solid_capstyle="round")
a2.set_ylim(0, 100); a2.set_yticks([0, 25, 50, 75, 100]); a2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
a2.set_title("What % of detectors get an answer?", loc="left", fontsize=14, fontweight="bold")
a2.text(grid[-1] + 0.05, ans[-1] * 100, f"{ans[-1]*100:.0f}%", color=PURPLE, fontsize=12, fontweight="bold", va="center")
a2.text(0.99, 0.08, "Detectors with fewer than 5 actuations in the sample\nare reported as “not enough data” instead of guessed.",
        transform=a2.transAxes, ha="right", va="bottom", fontsize=10, color=GREY)

ticks = [1, 5, 30, 60, 180, 360, 720, 1440, 4320]
a2.set_xticks(np.log10(ticks)); a2.set_xticklabels(["1 min", "5 min", "30 min", "1 h", "3 h", "6 h", "12 h", "1 day", "3 days"])
a2.set_xlim(grid[0] - 0.05, grid[-1] + 0.62)
a2.set_xlabel("how much data is in the sample", fontsize=12, color=GREY)
fig.suptitle("How much data does the detector classifier need?", x=0.06, y=0.985, ha="left", fontsize=16, fontweight="bold")
fig.text(0.06, 0.935, "Tested on 38 signals the model never saw during training, averaged over 10 start times across day and night.",
         fontsize=10.5, color=GREY)
fig.subplots_adjust(top=0.86, left=0.09, right=0.97, bottom=0.09)
fig.savefig(os.path.join(IMG, "accuracy_vs_minutes.png"), dpi=150, facecolor=fig.get_facecolor())

d["by_duration_smoothed"] = [dict(minutes=round(float(10 ** g), 3), acc_phase=round(float(p), 4), acc_function=round(float(f), 4),
                                  answered_share=round(float(s), 4)) for g, p, f, s in zip(grid, phase, func, ans)]
d["smoothing"] = "Monotone (isotonic, weighted by detectors scored) fit in log-minutes, then gaussian blur. Raw averages are in by_duration_raw."
json.dump(d, open(os.path.join(IMG, "accuracy_vs_minutes.json"), "w"), indent=1)
for m in (1, 5, 15, 30, 60, 360, 4320):
    i = int(np.argmin(np.abs(grid - np.log10(m)))); print(m, round(phase[i], 3), round(func[i], 3), round(ans[i], 3))
