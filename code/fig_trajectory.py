"""
Supporting figure --- the phenomenon behind Figure 1.

Figure 1 argues the mechanism. This one shows it happening, on one real
logged answer, computed from the released data:

  (a) the cause.  One vignette, one model, three stated ranges. The answer
      lands at nearly the same RELATIVE position inside whichever interval
      the prompt happened to state -- which is what anchoring looks like
      geometrically.

  (b) the consequence.  One of those logged answers, routed two ways
      against the SAME true plant. As a plant model it produces a gain that
      makes the true closed loop diverge. As cost weights it cannot, by
      Fact 1. Same numbers, same plant, different interface.

Writes fig8_trajectory.{pdf,png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_discrete_are

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = [os.path.join(_HERE, os.pardir, "data"), _HERE, os.getcwd()]

INK, MUTED, FAINT = "#1a1a1a", "#6b6b6b", "#cfcfcf"
BAD, GOOD = "#D55E00", "#0072B2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 7.8,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    "axes.linewidth": 0.7, "lines.linewidth": 1.4,
})

DT = 0.5
Q_TASK = np.diag([10.0, 1.0])
R_TASK = np.array([[0.5]])
WN_TRUE, ZETA_TRUE = 0.8, 0.5
OMEGA_REF, ZETA_REF = 1.05, 1.025


def _resolve(fn):
    for d in _DATA:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def _zoh(A, B, dt):
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def second_order(wn, zeta, dt=DT):
    A = np.array([[0.0, 1.0], [-wn**2, -2.0 * zeta * wn]])
    B = np.array([[0.0], [wn**2]])
    return _zoh(A, B, dt)


def lqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


A, B = second_order(WN_TRUE, ZETA_TRUE)

# ---------------------------------------------------------------- the data
rows = [r for r in json.load(open(_resolve("anchoring_v3_qwen2.5_7b.json")))
        if r["parse_ok"] and r["condition"] == "A_vary_omega"]

ORDER = ["narrow_low", "mid", "wide_high"]
NICE = {"narrow_low": "[0.1, 0.5]", "mid": "[0.1, 2.0]",
        "wide_high": "[0.5, 5.0]"}

# one vignette, so the patient is held fixed across the three ranges
VIG = "v6"
by_range = {}
for k in ORDER:
    vals = [r["omega_n"] for r in rows
            if r["range_condition"] == k and r["vignette"] == VIG]
    lo, hi = next(r["wn_range"] for r in rows if r["range_condition"] == k)
    by_range[k] = (lo, hi, float(np.median(vals)), len(vals))

fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.05),
                       gridspec_kw={"width_ratios": [1.0, 1.12]})

# =================================================== (a) the cause
a = ax[0]
for i, k in enumerate(ORDER):
    lo, hi, med, n = by_range[k]
    y = 3 - i
    a.plot([lo, hi], [y, y], color=FAINT, lw=5.0, solid_capstyle="butt",
           zorder=2)
    a.plot([lo, hi], [y, y], color=MUTED, lw=0.6, zorder=3)
    for x in (lo, hi):
        a.plot([x, x], [y - 0.17, y + 0.17], color=MUTED, lw=0.7, zorder=3)
    a.plot([med], [y], marker="o", ms=5.0, color=BAD, zorder=5)
    a.text(med, y + 0.24, f"{med:.2f}", ha="center", va="bottom",
           fontsize=7.2, color=BAD, fontweight="bold")
    frac = (med - lo) / (hi - lo)
    a.text(med, y - 0.42, f"{frac:.0%} in", ha="center", va="center",
           fontsize=6.3, color=MUTED)
    a.text(-0.45, y, NICE[k], ha="right", va="center", fontsize=6.6,
           color=INK, family="monospace")

a.axvline(WN_TRUE, color=GOOD, lw=0.9, ls=(0, (3, 2)), zorder=1)
a.text(WN_TRUE + 0.12, 3.62, "the plant the loop actually has",
       fontsize=6.2, color=GOOD, ha="left", va="top")
a.set_xlim(-1.95, 5.65)
a.set_ylim(0.45, 3.85)
a.set_yticks([])
a.set_xticks([0, 1, 2, 3, 4, 5])
a.set_xlabel(r"the model's $\widehat{\omega}_n$")
a.set_title("(a) same patient, three stated ranges", loc="left",
            fontweight="bold", pad=4)
a.spines["left"].set_visible(False)

# =================================================== (b) the consequence
b = ax[1]
LO, HI, MED, _ = by_range["narrow_low"]
z_med = float(np.median([r["zeta"] for r in rows
                         if r["range_condition"] == "narrow_low"
                         and r["vignette"] == VIG]))
wn_hat, z_hat = MED, z_med

Ah, Bh = second_order(wn_hat, z_hat)
K3 = lqr(Ah, Bh, Q_TASK, R_TASK)                       # I3 routing
Qh = Q_TASK * (wn_hat / OMEGA_REF)
Rh = R_TASK * (z_hat / ZETA_REF)
K4 = lqr(A, B, Qh, Rh)                                  # I4 routing
Kstar = lqr(A, B, Q_TASK, R_TASK)

N = 26
x0 = np.array([1.0, 0.0])


def roll(K):
    Acl = A - B @ K
    xs = np.zeros((N, 2))
    xs[0] = x0
    for t in range(1, N):
        xs[t] = Acl @ xs[t - 1]
    return xs


t = np.arange(N) * DT
r3, r4 = rho(A - B @ K3), rho(A - B @ K4)
for K, col, lab, z in [
        (K3, BAD,   f"$I_3$: as a plant model  ($\\rho={r3:.2f}>1$)", 4),
        (K4, GOOD,  f"$I_4$: as cost weights  ($\\rho={r4:.2f}<1$)", 4),
        (Kstar, MUTED, "true optimum", 2)]:
    xs = roll(K)
    b.plot(t, xs[:, 0], color=col, lw=1.6 if col != MUTED else 0.9,
           ls="-" if col != MUTED else (0, (2, 2)), label=lab, zorder=z)

b.axhline(0, color=FAINT, lw=0.6, zorder=1)
b.set_yscale("symlog", linthresh=1.0)
b.set_xlabel("time (s)")
b.set_ylabel("patient state $x_1$")
b.set_title("(b) the same answer, routed two ways", loc="left",
            fontweight="bold", pad=4)
b.legend(frameon=False, loc="upper left", handlelength=1.4,
         borderpad=0.1, labelspacing=0.22, borderaxespad=0.2)

plt.tight_layout(pad=0.4, w_pad=1.5)
plt.savefig(os.path.join(_HERE, "fig8_trajectory.pdf"), bbox_inches="tight",
            metadata={"Creator": "", "Producer": "", "CreationDate": None,
                      "Title": "", "Subject": "", "Keywords": ""})
plt.savefig(os.path.join(_HERE, "fig8_trajectory.png"), bbox_inches="tight",
            dpi=220, metadata={"Software": ""})
print("wrote fig8_trajectory.{pdf,png}")
for k in ORDER:
    lo, hi, med, n = by_range[k]
    print(f"  {k:11s} range [{lo},{hi}]  median answer {med:.2f}  "
          f"position {100*(med-lo)/(hi-lo):.0f}%  (n={n})")
print(f"  I3 rho = {r3:.4f}   I4 rho = {r4:.4f}   "
      f"(true plant wn={WN_TRUE}, zeta={ZETA_TRUE})")
