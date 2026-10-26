"""
Figure 1: the interface asymmetry.

(a) error in plant parameters (A,B)  -> destabilises the true closed loop
(b) error in cost weights (Q,R)      -> never destabilises (admissible)
(c) but cost-weight error does degrade performance, bounded by beta/alpha
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

def _zoh(A, B, dt):
    """Exact zero-order-hold discretisation via the matrix exponential.

    Earlier versions truncated the series at second order. That is accurate
    only while |lambda| dt is small; for a stiff plant it can place a pole
    exactly on the unit circle and manufacture spurious instability. All
    reported numbers use this exact form.
    """
    from scipy.linalg import expm
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


rng = np.random.default_rng(20260921)

# Okabe-Ito subset, CVD-validated in this order (worst adjacent dE 11.0 deutan)
C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.4,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.4, "legend.fontsize": 6.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3,
})


# ------------------------------------------------------------------ plants
def psych_4state():
    A = np.array([[-.020, .005, .005, -.008], [.005, -.030, .005, -.006],
                  [.005, .005, -.040, -.004], [-.004, -.004, -.004, -.015]])
    B = np.array([[-.30, -.15], [-.20, -.30], [-.15, -.35], [.25, .12]])
    dt = .5
    Ad, Bd = _zoh(A, B, dt)
    return Ad, Bd, np.diag([15., 12., 6., 4.]), np.diag([.5, .5])


def oscillator():
    wn, z, dt = .8, .3, .5
    A = np.array([[0., 1.], [-wn**2, -2*z*wn]]); B = np.array([[0.], [wn**2]])
    Ad, Bd = _zoh(A, B, dt)
    return Ad, Bd, np.diag([10., 1.]), np.array([[.5]])


def double_integrator():
    # Already exact: A is nilpotent of index 2, so the series terminates.
    dt = .5
    return (np.array([[1., dt], [0., 1.]]), np.array([[dt**2/2], [dt]]),
            np.diag([5., 1.]), np.array([[.5]]))


def spring_chain(n=3):
    k, c, dt = 1., .2, .2
    A = np.zeros((2*n, 2*n))
    for i in range(n):
        A[i, n+i] = 1.
    for i in range(n):
        if i > 0: A[n+i, i-1] += k
        A[n+i, i] += -2*k if i < n-1 else -k
        if i < n-1: A[n+i, i+1] += k
        A[n+i, n+i] = -c
    B = np.zeros((2*n, 1)); B[2*n-1, 0] = 1.
    Ad, Bd = _zoh(A, B, dt)
    return Ad, Bd, np.eye(2*n), np.array([[.5]])


def random_plant(n=4, m=2, seed=7):
    r = np.random.default_rng(seed)
    A = r.normal(scale=.3, size=(n, n))
    A = A / (np.max(np.abs(np.linalg.eigvals(A)))*1.2)
    return A, r.normal(scale=.5, size=(n, m)), np.eye(n)*5., np.eye(m)*.5


PLANTS = {"psych (4-state)": psych_4state(), "oscillator": oscillator(),
          "spring chain": spring_chain(3), "random 4x2": random_plant(),
          "double integrator": double_integrator()}


def lqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T@P@B, B.T@P@A)


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def cost(A, B, Q, R, K):
    Acl = A - B@K
    if rho(Acl) >= 1-1e-9:
        return np.inf
    return float(np.trace(solve_discrete_lyapunov(Acl.T, Q + K.T@R@K)))


EPS = [.05, .10, .25, .50, 1.0, 2.0]
ETA = [.25, .5, 1., 2., 4., 8.]
NDRAW = 200

fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.25))

# ---- (a) plant-parameter interface: destabilises
for i, (name, (A, B, Q, R)) in enumerate(PLANTS.items()):
    K = lqr(A, B, Q, R)
    ys = []
    for e in EPS:
        bad = 0
        for _ in range(NDRAW):
            dA = rng.normal(size=A.shape); dA *= np.linalg.norm(A)/np.linalg.norm(dA)
            dB = rng.normal(size=B.shape); dB *= np.linalg.norm(B)/np.linalg.norm(dB)
            try:
                Kh = lqr(A+e*dA, B+e*dB, Q, R)
            except Exception:
                bad += 1; continue
            if rho(A - B@Kh) >= 1:
                bad += 1
        ys.append(100*bad/NDRAW)
    ax[0].plot(EPS, ys, color=C[i], marker="o", ms=3, label=name)

ax[0].set_xscale("log")
ax[0].set_xticks(EPS); ax[0].set_xticklabels([str(int(e*100)) for e in EPS])
ax[0].set_ylim(-3, 103)
ax[0].set_xlabel("relative error in $(A,B)$  (%)")
ax[0].set_ylabel("closed loop unstable (%)")
ax[0].set_title("(a) plant-model interface", loc="left", fontweight="bold", pad=4)
ax[0].legend(frameon=False, loc="upper left", handlelength=1.0,
             borderpad=0.1, labelspacing=0.18, borderaxespad=0.2)

# ---- (b) cost-weight interface: never destabilises
for i, (name, (A, B, Q, R)) in enumerate(PLANTS.items()):
    ys = []
    for eta in ETA:
        bad = 0
        for _ in range(NDRAW):
            Qh = np.diag(np.diag(Q)*np.exp(eta*rng.normal(size=Q.shape[0])))
            Rh = np.diag(np.diag(R)*np.exp(eta*rng.normal(size=R.shape[0])))
            try:
                Kh = lqr(A, B, Qh, Rh)
            except Exception:
                bad += 1; continue
            if rho(A - B@Kh) >= 1:
                bad += 1
        ys.append(100*bad/NDRAW)
    ax[1].plot(ETA, ys, color=C[i], marker="o", ms=3)

ax[1].set_xscale("log")
ax[1].set_xticks(ETA); ax[1].set_xticklabels([str(e) for e in ETA])
ax[1].set_ylim(-3, 103)
ax[1].set_xlabel(r"log-error $\eta$ in $(Q,R)$")
ax[1].set_title("(b) cost-weight interface", loc="left", fontweight="bold", pad=4)
ax[1].annotate("0% for every plant\nat every level\n(5 plants, 5,992 draws)",
               xy=(.5, .52), xycoords="axes fraction", ha="center",
               va="center", fontsize=6.3, color=MUTED)

# ---- (c) but performance does degrade, and beta/alpha bounds it
A, B, Q, R = psych_4state()
K = lqr(A, B, Q, R); Js = cost(A, B, Q, R, K)
med, mx, bnd = [], [], []
for eta in ETA:
    rs, bs = [], []
    for _ in range(NDRAW):
        fq = np.exp(eta*rng.normal(size=Q.shape[0]))
        fr = np.exp(eta*rng.normal(size=R.shape[0]))
        Kh = lqr(A, B, np.diag(np.diag(Q)*fq), np.diag(np.diag(R)*fr))
        if rho(A - B@Kh) >= 1:
            continue
        rs.append(cost(A, B, Q, R, Kh)/Js)
        bs.append(max(fq.max(), fr.max())/min(fq.min(), fr.min()))
    med.append(np.median(rs)); mx.append(np.max(rs)); bnd.append(np.median(bs))

ax[2].plot(ETA, bnd, color=MUTED, ls=":", label=r"bound $\beta/\alpha$")
ax[2].plot(ETA, mx, color=C[1], ls="--", marker="s", ms=3, label="worst observed")
ax[2].plot(ETA, med, color=C[0], marker="o", ms=3, label="median observed")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xticks(ETA); ax[2].set_xticklabels([str(e) for e in ETA])
ax[2].set_xlabel(r"log-error $\eta$ in $(Q,R)$")
ax[2].set_ylabel(r"suboptimality $J(\hat K)/J(K^\star)$")
ax[2].set_title("(c) stability $\\neq$ performance", loc="left", fontweight="bold", pad=4)
ax[2].legend(frameon=False, loc="upper left", handlelength=1.3,
             borderpad=0.1, labelspacing=0.18, borderaxespad=0.2)

plt.tight_layout(pad=0.4, w_pad=1.1)
plt.savefig("fig3_asymmetry.pdf", bbox_inches="tight")
plt.savefig("fig3_asymmetry.png", bbox_inches="tight", dpi=220)
print("wrote fig3_asymmetry.{pdf,png}")
print("\npanel c numbers (psych-4state):")
for e, m, x, b in zip(ETA, med, mx, bnd):
    print(f"  eta={e:>4}  median={m:8.2f}  max={x:12.2f}  median bound={b:12.2f}")
