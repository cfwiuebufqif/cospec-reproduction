"""
Figure: the direct cost elicitation test (reviewer Major point #2).

The model is asked directly for the LQR cost weights (q1, q2, r), not for
plant parameters. Every free-form output is archived and pushed through the
certificate. Values come from the two cost_*.json files after scoring, and
the variance shares from cost_analysis.json, so the figure cannot drift.

(a) true cost J/J* of the admitted weights for the two models
(b) the scale-invariant weight shape q1/q2 (the mapped I4 route fixes this at
    10 identically), points split by the fast/slow prognosis
(c) share of each shape dimension's variance that lies across vignettes
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cost_probe as cp

C_BAD, C_GOOD, C_WARN = "#D55E00", "#0072B2", "#E69F00"
C_FAST, C_SLOW = "#0072B2", "#E69F00"
INK, MUTED, FAINT = "#1a1a1a", "#6b6b6b", "#d8d8d6"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 7.6, "axes.titlesize": 7.6,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    "axes.linewidth": 0.8,
})

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("deepseek-chat", "deepseek-chat"), ("qwen2.5_7b", "qwen2.5:7b")]
data = {}
for fn, name in MODELS:
    rows = cp.score(json.load(open(os.path.join(HERE, f"cost_{fn}.json"))))
    data[name] = [r for r in rows if r["admitted"]]
ANA = json.load(open(os.path.join(HERE, "cost_analysis.json")))

fig, ax = plt.subplots(1, 3, figsize=(5.6, 2.4),
                       gridspec_kw={"width_ratios": [1.0, 1.15, 1.05]})

# ------------------------------------------------- (a) true cost J/J*
a = ax[0]
names = [m for _, m in MODELS]
for i, (_, m) in enumerate(MODELS):
    cr = np.array([r["cost_ratio"] for r in data[m]])
    rng = np.random.default_rng(0)
    a.scatter(np.full(len(cr), i) + rng.uniform(-0.13, 0.13, len(cr)),
              cr, s=5, color=C_GOOD, alpha=0.45, zorder=3,
              edgecolors="none")
    a.plot([i - 0.18, i + 0.18], [np.median(cr)] * 2, color=INK, lw=1.4,
           zorder=4)
a.axhline(1.0, color=MUTED, lw=0.8, ls=":")
a.text(1.42, 1.0, "optimal", fontsize=6.3, color=MUTED, va="center",
       ha="left")
a.set_xticks(range(2))
a.set_xticklabels(["deepseek\n(frontier)", "qwen2.5\n(7B local)"],
                  fontsize=6.5)
a.set_ylabel("true cost $J(\\hat K)/J(K^\\star)$", fontsize=7.4)
a.set_ylim(0.9, 1.8)
a.set_title("(a) performance of elicited weights", loc="left",
            fontweight="bold", pad=4)
a.text(0.98, 0.04, "100% parse, 100% admitted, zero rejected",
       transform=a.transAxes, fontsize=6.2, color=MUTED, va="bottom",
       ha="right")
a.yaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)

# ------------------------------------------------- (b) q1/q2 shape
a = ax[1]
for i, (_, m) in enumerate(MODELS):
    rng = np.random.default_rng(1)
    for j, prog in enumerate(("fast", "slow")):
        sub = [r for r in data[m] if r["prognosis"] == prog
               and np.isfinite(r["q1_q2"])]
        x = i + (j - 0.5) * 0.34
        a.scatter(np.full(len(sub), x) + rng.uniform(-0.07, 0.07, len(sub)),
                  [r["q1_q2"] for r in sub], s=5,
                  color=C_FAST if prog == "fast" else C_SLOW, alpha=0.55,
                  zorder=3, edgecolors="none")
a.axhline(10.0, color=C_BAD, lw=1.0, ls="--", zorder=2)
a.text(1.46, 10.0, "mapped $I_4$ fixes $q_1/q_2=10$", fontsize=6.3,
       color=C_BAD, va="center", ha="left")
a.set_xticks(range(2))
a.set_xticklabels(["deepseek", "qwen2.5"], fontsize=6.8)
a.set_ylabel("weight shape $q_1/q_2$", fontsize=7.4)
a.set_ylim(0, 12)
a.set_title("(b) elicited shape, not a fixed scale", loc="left",
            fontweight="bold", pad=8)
for k, (prog, col) in enumerate([("fast", C_FAST), ("slow", C_SLOW)]):
    a.plot([], [], "o", ms=3.5, color=col, label=prog)
a.legend(frameon=False, loc="upper right", fontsize=6.4,
         handlelength=1.0, labelspacing=0.2)
a.yaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)

# ------------------------------------------------- (c) eta^2 across vignettes
a = ax[2]
dims = [("q1/q2", "q12_eta_vignette"), ("q1/r", None)]
labels = ["$q_1/q_2$", "$q_1/r$"]
w = 0.34
for i, (_, m) in enumerate(MODELS):
    s = ANA[m]
    # q1/r share is not stored in the summary; recompute directly via eta2
    q1r = cp.eta2(data[m], "vignette", "q1_r")
    q12 = s["q12_eta_vignette"]
    xs = np.arange(2) + (i - 0.5) * w
    a.bar(xs, [100 * q12, 100 * q1r], width=w,
          color=C_GOOD if m.startswith("deepseek") else C_WARN, zorder=3)
a.set_xticks(range(2))
a.set_xticklabels(labels, fontsize=7.2)
a.set_ylabel("variance across vignettes (%)", fontsize=7.4)
a.set_ylim(0, 80)
a.set_title("(c) where patient info appears", loc="left", fontweight="bold",
            pad=4)
a.text(0.83, 70.5, "deepseek:\neffort balance", fontsize=5.9,
       color=C_GOOD, ha="center", va="bottom")
a.text(0.17, 42.5, "qwen:\nspeed/smoothness", fontsize=5.9,
       color="#a8740a", ha="center", va="bottom")
a.yaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)

plt.tight_layout(pad=0.5, w_pad=2.4)
plt.savefig("fig7_directcost.pdf", bbox_inches="tight",
            metadata={"Creator": "", "Producer": "", "CreationDate": None,
                      "Title": "", "Subject": "", "Keywords": ""})
plt.savefig("fig7_directcost.png", bbox_inches="tight", dpi=230,
            metadata={"Software": ""})
print("wrote fig7_directcost.{pdf,png}")
for _, m in MODELS:
    cr = np.array([r["cost_ratio"] for r in data[m]])
    print(f"  {m:14s} J/J* median {np.median(cr):.2f} max {np.max(cr):.2f}")
