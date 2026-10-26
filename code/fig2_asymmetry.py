"""
Figure 2: the interface asymmetry, merged.

(a) plant-model error (I3) destabilises the true loop: 29 plants, thin lines
    for individual plants, thick lines for the physical-family mean.
(b) cost-weight error (I4) never destabilises: 0% for every plant at every
    level (17,400 draws, zero violations of the beta/alpha bound).
(c) stability is not performance: the cost-ratio error stays admissible but
    does degrade the cost, and the beta/alpha condition-number bound sits
    above every observed value.

(a, b) read wide_study.json (interface_study_wide.py); (c) is computed here
on the same 4-state plant used in the rest of the paper.
Writes fig2_asymmetry.{pdf,png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.linalg import (expm, solve_discrete_are, solve_discrete_lyapunov)

_HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260921)

# Okabe-Ito, CVD-validated
C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
CFAM = {"original": "#1a1a1a", "mass-spring-damper": "#D55E00",
        "series RLC": "#0072B2", "two-tank thermal": "#009E73"}
INK, MUTED, FAINT = "#1a1a1a", "#6b6b6b", "#cfcfcf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 6.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
})

EPS = ["0.05", "0.1", "0.25", "0.5", "1.0", "2.0"]
XS = [5, 10, 25, 50, 100, 200]
ETA = ["0.5", "2.0", "8.0"]
XE = [0.5, 2.0, 8.0]

res = json.load(open(os.path.join(_HERE, "wide_study.json")))

fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.2),
                       gridspec_kw={"width_ratios": [1.25, 0.95, 1.15]})

# ------------------------------------------------- (a) I3, 29 plants
a = ax[0]
for v in res.values():
    a.plot(XS, [v["i3"][e] for e in EPS], color=CFAM[v["family"]],
           alpha=0.28, lw=0.7, zorder=2)
for fam in ["original", "mass-spring-damper", "series RLC",
            "two-tank thermal"]:
    sel = [v for v in res.values() if v["family"] == fam]
    if not sel:
        continue
    m = [float(np.mean([v["i3"][e] for v in sel])) for e in EPS]
    a.plot(XS, m, color=CFAM[fam], lw=1.9, marker="o", ms=3.0, zorder=4,
           label=f"{fam} ({len(sel)})")
a.set_xscale("log"); a.set_xticks(XS); a.set_xticklabels([str(x) for x in XS])
a.xaxis.set_minor_formatter(NullFormatter())
a.set_ylim(-3, 103)
a.set_xlabel("relative error in $(A,B)$  (%)")
a.set_ylabel("true loop unstable (%)")
a.set_title("(a) plant-model error, $I_3$", loc="left", fontweight="bold",
            pad=4)
a.legend(frameon=False, loc="upper left", handlelength=1.1,
         borderpad=0.1, labelspacing=0.18, borderaxespad=0.15)
a.text(0.97, 0.03, "thin lines: individual plants", fontsize=6.0,
       color=MUTED, transform=a.transAxes, ha="right", va="bottom")

# ------------------------------------------------- (b) I4, all zero
b = ax[1]
for v in res.values():
    b.plot(XE, [v["i4"][e]["unstable_pct"] for e in ETA],
           color=CFAM[v["family"]], alpha=0.28, lw=0.7, marker="o", ms=2.2)
b.set_xscale("log"); b.set_xticks(XE); b.set_xticklabels([str(e) for e in XE])
b.xaxis.set_minor_formatter(NullFormatter())
b.set_ylim(-3, 103)
b.set_xlabel(r"log-error $\eta$ in $(Q,R)$")
b.set_title("(b) cost-weight error, $I_4$", loc="left", fontweight="bold",
            pad=4)
n_draw = sum(v["i4"][e]["n"] for v in res.values() for e in ETA)
b.annotate(f"0% for all 29 plants\nat every level\n({n_draw:,} draws)",
           xy=(0.5, 0.52), xycoords="axes fraction", ha="center",
           va="center", fontsize=6.3, color=MUTED)

# ------------------------------------------------- (c) beta/alpha bound
def _zoh(AA, BB, dt):
    n, m = AA.shape[0], BB.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = AA; M[:n, n:] = BB
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def psych_4state():
    AA = np.array([[-.020, .005, .005, -.008], [.005, -.030, .005, -.006],
                   [.005, .005, -.040, -.004], [-.004, -.004, -.004, -.015]])
    BB = np.array([[-.30, -.15], [-.20, -.30], [-.15, -.35], [.25, .12]])
    Ad, Bd = _zoh(AA, BB, .5)
    return Ad, Bd, np.diag([15., 12., 6., 4.]), np.diag([.5, .5])


def lqr(AA, BB, QQ, RR):
    P = solve_discrete_are(AA, BB, QQ, RR)
    return np.linalg.solve(RR + BB.T @ P @ BB, BB.T @ P @ AA)


def _rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def _cost(AA, BB, QQ, RR, K):
    Acl = AA - BB @ K
    if _rho(Acl) >= 1 - 1e-9:
        return np.inf
    return float(np.trace(solve_discrete_lyapunov(
        Acl.T, QQ + K.T @ RR @ K)))


ETAC = [.25, .5, 1., 2., 4., 8.]
NDRAW = 200
A, B, Q, R = psych_4state()
K = lqr(A, B, Q, R); Js = _cost(A, B, Q, R, K)
med, mx, bnd = [], [], []
for eta in ETAC:
    rs, bs = [], []
    for _ in range(NDRAW):
        fq = np.exp(eta * rng.normal(size=Q.shape[0]))
        fr = np.exp(eta * rng.normal(size=R.shape[0]))
        Kh = lqr(A, B, np.diag(np.diag(Q) * fq), np.diag(np.diag(R) * fr))
        if _rho(A - B @ Kh) >= 1:
            continue
        rs.append(_cost(A, B, Q, R, Kh) / Js)
        bs.append(max(fq.max(), fr.max()) / min(fq.min(), fr.min()))
    med.append(np.median(rs)); mx.append(np.max(rs)); bnd.append(np.median(bs))

c = ax[2]
c.plot(ETAC, bnd, color=MUTED, ls=":", label=r"bound $\beta/\alpha$")
c.plot(ETAC, mx, color=C[1], ls="--", marker="s", ms=3, label="worst observed")
c.plot(ETAC, med, color=C[0], marker="o", ms=3, label="median observed")
c.set_xscale("log"); c.set_yscale("log")
c.set_xticks(ETAC); c.set_xticklabels([str(e) for e in ETAC])
c.xaxis.set_minor_formatter(NullFormatter())
c.set_xlabel(r"log-error $\eta$ in $(Q,R)$")
c.set_ylabel(r"suboptimality $J(\hat K)/J(K^\star)$")
c.set_title("(c) stability $\\neq$ performance", loc="left",
            fontweight="bold", pad=4)
c.legend(frameon=False, loc="upper left", handlelength=1.3,
         borderpad=0.1, labelspacing=0.18, borderaxespad=0.2)

plt.tight_layout(pad=0.4, w_pad=1.2)
plt.savefig(os.path.join(_HERE, "fig2_asymmetry.pdf"), bbox_inches="tight",
            metadata={"Creator": "", "Producer": "", "CreationDate": None,
                      "Title": "", "Subject": "", "Keywords": ""})
plt.savefig(os.path.join(_HERE, "fig2_asymmetry.png"), bbox_inches="tight",
            dpi=220, metadata={"Software": ""})
print("wrote fig2_asymmetry.{pdf,png}")
for e, m, x, b in zip(ETAC, med, mx, bnd):
    print(f"  eta={e:>4}  median={m:8.2f}  max={x:12.2f}  "
          f"median bound={b:12.2f}")
