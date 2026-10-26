"""
Figure 4: the case study, run on the 540 logged LLM outputs.

Every value is read from case_study_results.json (written by case_study.py),
so the figure cannot drift from the experiment.

(a) the same numbers, two routings: how often the TRUE closed loop is unstable
(b) I3 instability broken out by the numeric range the PROMPT stated
(c) what the pipeline's own Monte Carlo gate does with the I3 path
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_BAD, C_GOOD, C_WARN = "#D55E00", "#0072B2", "#E69F00"   # validated
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

_HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(_HERE, "case_study_results.json")))
N = len(R)

fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.75),
                       gridspec_kw={"width_ratios": [0.85, 1.0, 1.15]})


def pct(sub, key="i3_unstable"):
    return 100.0 * sum(r[key] for r in sub) / max(len(sub), 1)


def label_bars(a, xs, vals, fmt="{:.1f}%", dy=1.6, fs=7.6, color=INK):
    for x, v in zip(xs, vals):
        a.text(x, v + dy, fmt.format(v), ha="center", va="bottom",
               fontsize=fs, color=color, fontweight="bold")


# ---------------------------------------------- (a) two routings
a = ax[0]
i3, i4 = pct(R, "i3_unstable"), pct(R, "i4_unstable")
a.bar([0], [i3], width=0.56, color=C_BAD, zorder=3)
a.bar([1], [i4], width=0.56, color=C_GOOD, zorder=3)
label_bars(a, [0, 1], [i3, i4])
a.set_xticks([0, 1])
a.set_xticklabels(["$I_3$\nplant", "$I_4$\ncost"], fontsize=6.4)
a.set_ylim(0, max(i3 * 1.45, 6))
a.set_ylabel("true loop unstable (%)", fontsize=7.4)
a.set_title("(a) two routings", loc="left", fontweight="bold",
            pad=12)
a.text(0, 1.01, f"all N={N} logged LLM outputs", transform=a.transAxes,
       fontsize=6.7, color=MUTED, va="bottom")
a.yaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)
a.annotate("certificate admits\n100% of these",
           xy=(1, i4), xytext=(1, max(i3 * 1.45, 6) * 0.42),
           ha="center", fontsize=6.8, color=C_GOOD,
           arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=0.8))

# ------------------------------- (b) instability by the stated range
a = ax[1]
order = ["narrow_low", "mid", "wide_high"]
nice = {"narrow_low": "narrow", "mid": "mid", "wide_high": "wide"}
vals = [pct([r for r in R if r["range_condition"] == k]) for k in order]
a.bar(range(3), vals, width=0.52, color=C_BAD, zorder=3)
label_bars(a, range(3), vals, fs=6.6)
a.set_xticks(range(3))
a.set_xticklabels([nice[k] for k in order], fontsize=6.6,
                  rotation=28, ha="right", rotation_mode="anchor")
a.set_xlabel("range stated in the prompt", fontsize=7.4)
a.set_ylim(0, max(vals) * 1.42)
a.set_ylabel("$I_3$ unstable (%)")
a.set_title("(b) set by the stated range", loc="left", fontweight="bold",
            pad=12)
a.text(0, 1.01, "same patients, same models", transform=a.transAxes,
       fontsize=6.7, color=MUTED, va="bottom")
a.yaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)

# --------------------------------- (c) what the Monte Carlo gate does
a = ax[2]
acc_ok = 100 * sum(r["i3_gate_pass"] and not r["i3_unstable"] for r in R) / N
acc_bad = 100 * sum(r["i3_gate_pass"] and r["i3_unstable"] for r in R) / N
rej_bad = 100 * sum(not r["i3_gate_pass"] and r["i3_unstable"] for r in R) / N
rej_ok = 100 * sum(not r["i3_gate_pass"] and not r["i3_unstable"] for r in R) / N

a.barh([1], [acc_ok], height=0.5, color=FAINT, zorder=3,
       label="stable (correctly let through)")
a.barh([1], [acc_bad], left=[acc_ok], height=0.5, color=C_BAD, zorder=3,
       label="UNSTABLE (wrongly let through)")
a.barh([0], [rej_bad], height=0.5, color=C_WARN, zorder=3,
       label="unstable (correctly caught)")
a.barh([0], [rej_ok], left=[rej_bad], height=0.5, color="#efefed", zorder=3,
       label="stable (needlessly rejected)")

a.text(acc_ok + acc_bad + 2.5, 1.0, f"{acc_bad:.1f}% of all runs\nwere unstable",
       ha="left", va="center", fontsize=6.9, color=C_BAD, fontweight="bold")
a.set_yticks([1, 0])
a.set_yticklabels(["gate\nACCEPTS", "gate\nrejects"], fontsize=7.2)
a.set_xlim(0, 118)
a.set_xlabel("share of the 540 runs (%)", fontsize=7.4)
a.set_title("(c) the pipeline's guard", loc="left", fontweight="bold",
            pad=12)
a.text(0, 1.01, r"$\pm25\%$, 50 draws, accept at $\geq$80% stable",
       transform=a.transAxes, fontsize=6.7, color=MUTED, va="bottom")
a.xaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.42),
         ncol=2, handlelength=1.0, columnspacing=1.0, labelspacing=0.22,
         fontsize=6.6)

plt.tight_layout(pad=0.5, w_pad=2.8, rect=[0, 0.11, 1, 1])
plt.savefig("fig5_case.pdf", bbox_inches="tight",
            metadata={"Creator": "", "Producer": "", "CreationDate": None,
                      "Title": "", "Subject": "", "Keywords": ""})
plt.savefig("fig5_case.png", bbox_inches="tight", dpi=230, metadata={"Software": ""})
print("wrote fig5_case.{pdf,png}")
print(f"  I3 {i3:.1f}%  I4 {i4:.1f}%   by-range {vals}")
print(f"  gate: accept-ok {acc_ok:.1f}  accept-BAD {acc_bad:.1f}  "
      f"reject-bad {rej_bad:.1f}  reject-ok {rej_ok:.1f}")
