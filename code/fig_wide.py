"""
Figure: the interface asymmetry over 29 plants, by physical family.

Reads wide_study.json (written by interface_study_wide.py) and draws what
Table 3 says, which a table says badly: at I3 the fragility curve is a
property of the plant and spreads over the whole range, while at I4 it is
flat at zero for every plant at every level, because Fact 1 is
plant-independent.

Writes fig7_wide.{pdf,png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

C = {"original": "#1a1a1a", "mass-spring-damper": "#D55E00",
     "series RLC": "#0072B2", "two-tank thermal": "#009E73"}
INK, MUTED, FAINT = "#1a1a1a", "#6b6b6b", "#cfcfcf"

plt.rcParams.update({
    "font.size": 7.4, "axes.labelsize": 7.4, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
})

EPS = ["0.05", "0.1", "0.25", "0.5", "1.0", "2.0"]
XS = [5, 10, 25, 50, 100, 200]

res = json.load(open(os.path.join(_HERE, "wide_study.json")))

fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.15),
                       gridspec_kw={"width_ratios": [1.35, 1]})

# ---------------------------------------------------- (a) I3, every plant
a = ax[0]
for name, v in res.items():
    fam = v["family"]
    a.plot(XS, [v["i3"][e] for e in EPS], color=C[fam], alpha=0.30,
           lw=0.7, zorder=2)
for fam in ["original", "mass-spring-damper", "series RLC",
            "two-tank thermal"]:
    sel = [v for v in res.values() if v["family"] == fam]
    if not sel:
        continue
    m = [float(np.mean([v["i3"][e] for v in sel])) for e in EPS]
    a.plot(XS, m, color=C[fam], lw=1.9, marker="o", ms=3.0, zorder=4,
           label=f"{fam} ({len(sel)})")
a.set_xscale("log")
a.set_xticks(XS)
a.set_xticklabels([str(x) for x in XS])
a.set_ylim(-3, 103)
a.set_xlabel("relative error in $(A,B)$  (%)")
a.set_ylabel("true loop unstable (%)")
a.set_title("(a) $I_3$: plant-model error, 29 plants", loc="left",
            fontweight="bold", pad=4)
a.legend(frameon=False, loc="upper left", handlelength=1.1,
         borderpad=0.1, labelspacing=0.2, borderaxespad=0.15)
a.text(0.97, 0.03, "thin lines: individual plants", fontsize=6.0,
       color=MUTED, transform=a.transAxes, ha="right", va="bottom")

# ---------------------------------------------------- (b) I4, every plant
b = ax[1]
ETA = ["0.5", "2.0", "8.0"]
XE = [0.5, 2.0, 8.0]
for name, v in res.items():
    b.plot(XE, [v["i4"][e]["unstable_pct"] for e in ETA],
           color=C[v["family"]], alpha=0.30, lw=0.7, marker="o", ms=2.2)
b.set_xscale("log")
b.set_xticks(XE)
b.set_xticklabels([str(e) for e in XE])
b.set_ylim(-3, 103)
b.set_xlabel(r"log-error $\eta$ in $(Q,R)$")
b.set_title("(b) $I_4$: cost-weight error", loc="left",
            fontweight="bold", pad=4)
n_draw = sum(v["i4"][e]["n"] for v in res.values() for e in ETA)
b.annotate(f"0% for all 29 plants\nat every level\n({n_draw:,} draws,\n"
           f"0 bound violations)",
           xy=(0.5, 0.52), xycoords="axes fraction", ha="center",
           va="center", fontsize=6.4, color=MUTED)

plt.tight_layout(pad=0.4, w_pad=1.6)
plt.savefig(os.path.join(_HERE, "fig7_wide.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(_HERE, "fig7_wide.png"), bbox_inches="tight",
            dpi=220)
print("wrote fig7_wide.{pdf,png}")
print(f"  I4 total draws {n_draw}, all-zero unstable: "
      f"{all(v['i4'][e]['unstable_pct'] == 0 for v in res.values() for e in ETA)}")
