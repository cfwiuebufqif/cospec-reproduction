"""
Figure 3: the crossover signature, with uncertainty, and the normalised
robustness check overlaid.

Reads stats_ci.json (written by stats_ci.py), so every bar, whisker and
marker is a number the analysis script computed.

(a) the parameter whose stated range the prompt manipulates, with the
    normalised range-share marked on the same row: the gap between the
    orange bar and the green marker is the part of the raw range-dependence
    that was only rescaling.
(b) the parameter whose range is held fixed (the control).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

C_RANGE, C_VIGN, C_NORM = "#D55E00", "#0072B2", "#009E73"   # validated
INK, MUTED, FAINT = "#1a1a1a", "#6b6b6b", "#d8d8d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK, "figure.dpi": 200,
    "axes.linewidth": 0.7,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(_HERE, "stats_ci.json")))

MODELS = ["qwen2.5:7b", "deepseek-chat", "gpt-5.6"]
CONDS = [("A_vary_omega", "A", "omega_n", "zeta"),
         ("B_vary_zeta", "B", "zeta", "omega_n")]

rows, labels = [], []
for m in MODELS:
    for ckey, ctag, varied, ctrl in CONDS:
        rows.append((m, ckey, varied, ctrl))
        labels.append(f"{m}  ({ctag})")

y = np.arange(len(rows))[::-1]
h = 0.33

fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True)


def draw(a, which, title, note, overlay_norm):
    for i, (m, ck, varied, ctrl) in enumerate(rows):
        p = varied if which == "varied" else ctrl
        d = S[m][ck][p]
        r, v = 100 * d["eta2_range"], 100 * d["eta2_vignette"]
        rl, rh = [100 * z for z in d["eta2_range_ci"]]
        vl, vh = [100 * z for z in d["eta2_vignette_ci"]]
        a.barh(y[i] + h / 2, r, height=h, color=C_RANGE, zorder=3)
        a.barh(y[i] - h / 2, v, height=h, color=C_VIGN, zorder=3)
        a.plot([rl, rh], [y[i] + h / 2] * 2, color="white", lw=0.9, zorder=4)
        a.plot([vl, vh], [y[i] - h / 2] * 2, color="white", lw=0.9, zorder=4)
        if overlay_norm:
            nrm = 100 * d["eta2_range_normalised"]
            a.scatter([nrm], [y[i] + h / 2], s=17, color=C_NORM, zorder=6,
                      linewidths=0.6, edgecolors="white")
    a.set_xlim(0, 104)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=11)
    a.text(0, 1.015, note, transform=a.transAxes, fontsize=6.1, color=MUTED,
           va="bottom")
    a.set_xlabel("share of variance (%)")
    a.xaxis.grid(True, color=FAINT, lw=0.5, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(axis="y", length=0)


draw(ax[0], "varied", "(a) manipulated parameter",
     "its stated range is what the prompt moves", True)
draw(ax[1], "control", "(b) control parameter",
     "its range is held fixed", False)

ax[0].set_yticks(y)
ax[0].set_yticklabels(labels, fontsize=6.7)

handles = [Patch(facecolor=C_RANGE, label="stated RANGE (raw)"),
           Patch(facecolor=C_VIGN, label="VIGNETTE (patient)"),
           Line2D([], [], marker="o", color="none", markerfacecolor=C_NORM,
                  markeredgecolor="white", markersize=4.6,
                  label="RANGE after normalising")]
fig.legend(handles=handles, frameon=False, loc="lower center",
           bbox_to_anchor=(0.55, -0.045), ncol=3, handlelength=1.0,
           columnspacing=1.3, fontsize=6.4)

plt.tight_layout(pad=0.4, w_pad=1.0, rect=[0, 0.085, 1, 1])
plt.savefig("fig4_crossover.pdf", bbox_inches="tight",
            metadata={"Creator": "", "Producer": "", "CreationDate": None,
                      "Title": "", "Subject": "", "Keywords": ""})
plt.savefig("fig4_crossover.png", bbox_inches="tight", dpi=230, metadata={"Software": ""})
print("wrote fig4_crossover.{pdf,png}")
