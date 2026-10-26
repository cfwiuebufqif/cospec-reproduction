"""
Figure: the identified-plant robustness test (reviewer Major point #1).

When the plant is not known but recovered online, the gain is synthesized on
an estimated plant and closed on the true one. Values come from
identified_robustness.json, so the figure cannot drift.

(a) recursive least-squares convergence: relative model error (left log axis)
    over 3 seeds and the worst-cost true-loop spectral radius (right axis)
(b) worst-cost true-loop spectral radius against relative plant mismatch,
    three directions; the crossing of rho=1 is the stability threshold
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_BAD, C_GOOD, C_WARN = "#D55E00", "#0072B2", "#E69F00"
C_DARK, INK, MUTED, FAINT = "#404040", "#1a1a1a", "#6b6b6b", "#d8d8d6"

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
J = json.load(open(os.path.join(HERE, "identified_robustness.json")))

fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.4),
                       gridspec_kw={"width_ratios": [1.05, 1.0]})

# ------------------------------------------------- (a) RLS convergence
a = ax[0]
ar = a.twinx()
ar.spines["top"].set_visible(False)
for s, path in enumerate(J["rls"]):
    t = [p["t"] for p in path]
    err = [100 * p["err"] for p in path]
    a.plot(t, err, color=C_GOOD, lw=1.0 if s else 1.5,
           alpha=0.55 if s else 0.95, zorder=3,
           label="identification error" if not s else None)
# seed-0 worst-cost spectral radius on the right axis
p0 = J["rls"][0]
ar.plot([p["t"] for p in p0], [p["rho_worst"] for p in p0],
        color=C_WARN, lw=1.3, marker="o", ms=2.6, zorder=4,
        label="worst-cost spectral radius")
ar.axhline(1.0, color=C_BAD, lw=0.9, ls="--", zorder=2)
a.axhline(18.5, color=C_BAD, lw=0.7, ls=":", zorder=1)
a.text(402, 20.0, "mismatch threshold 18.5%", fontsize=6.0,
       color=C_BAD, ha="right", va="bottom")

a.set_yscale("log")
a.set_ylim(0.1, 120)
a.set_xlabel("samples used for identification", fontsize=7.4)
a.set_ylabel("relative model error (%)", fontsize=7.4, color=C_GOOD)
a.set_xlim(0, 400)
ar.set_ylim(0.5, 1.06)
ar.set_ylabel("true-loop spectral radius", fontsize=7.4, color=C_WARN)
ar.spines["right"].set_visible(True)
ar.spines["right"].set_color(MUTED)
ar.yaxis.set_minor_locator(plt.NullLocator())
a.set_title("(a) RLS converges before the margin is reached", loc="left",
            fontweight="bold", pad=4)
a.text(0.98, 0.06, "full state, persistent excitation, low noise",
       transform=a.transAxes, fontsize=6.2, color=MUTED, ha="right",
       va="bottom")
a.yaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)

# ------------------------------------------------- (b) mismatch sweep
a = ax[1]
styles = [("omega", "natural frequency $\\omega_n$", C_BAD),
          ("zeta", "damping $\\zeta$", C_GOOD),
          ("both", "both", C_DARK)]
for key, lab, col in styles:
    rows = J["mismatch"][key]["rows"]
    d = [100 * r["delta"] for r in rows]
    rw = [r["rho_worst"] for r in rows]
    rw = np.clip(rw, 0, 1.6)
    a.plot(d, rw, color=col, lw=1.3, label=lab, zorder=3)
a.axhline(1.0, color=INK, lw=0.9, ls="--", zorder=2)
a.text(59, 1.015, "stability limit", fontsize=6.0, color=INK,
       ha="right", va="bottom")
a.annotate("19%", xy=(19, 1.0), xytext=(25, 1.28),
           fontsize=6.6, color=C_BAD, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.8))
a.text(55, 0.78, "damping robust\nup to 60%", fontsize=6.0,
       color=C_GOOD, ha="right", va="center")
a.set_xlim(0, 60)
a.set_ylim(0.5, 1.55)
a.set_xlabel("relative plant mismatch (%)", fontsize=7.4)
a.set_ylabel("worst-cost spectral radius", fontsize=7.4)
a.set_title("(b) margin is set by plant mismatch", loc="left",
            fontweight="bold", pad=4)
a.legend(frameon=False, loc="upper left", fontsize=6.5,
         handlelength=1.3, labelspacing=0.25)
a.yaxis.grid(True, color=FAINT, lw=0.6, zorder=0)
a.set_axisbelow(True)

plt.tight_layout(pad=0.5, w_pad=2.6)
plt.savefig("fig8_robust.pdf", bbox_inches="tight",
            metadata={"Creator": "", "Producer": "", "CreationDate": None,
                      "Title": "", "Subject": "", "Keywords": ""})
plt.savefig("fig8_robust.png", bbox_inches="tight", dpi=230,
            metadata={"Software": ""})
print("wrote fig8_robust.{pdf,png}")
for k in ("omega", "zeta", "both"):
    m = J["mismatch"][k]
    print(f"  {k:5s} nominal {m['threshold_nominal']:.0%}  "
          f"worst {m['threshold']:.0%}")
