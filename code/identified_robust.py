"""
Identified-plant robustness experiment (response to reviewer Major point #1).

The main-text stability statement (Fact 1) uses the gain K = LQR(A, B, Qh, Rh)
with the TRUE plant (A, B). In a deployed pipeline the plant is not known: it is
recovered by an online estimator (e.g. recursive least squares), so the gain is
in fact K = LQR(Ah, Bh, Qh, Rh), closed on the true plant (A, B). Fact 1 then
does not apply directly.

This experiment quantifies what happens on the identified-plant route. Two
complementary views:

  (A) RLS convergence.  We excite the true discrete plant, recover (Ah, Bh) by
      least squares from the first t samples, and close the certainty-equivalence
      gain on the true plant. Estimation error decays with t; we report the
      closed-loop spectral radius at each t, both for the nominal cost and for
      the WORST of a grid of certified admissible costs.

  (B) Mismatch sweep.  We parametrise the identified plant directly by a
      relative error delta in (omega_n, zeta) and find the largest delta for
      which the closed loop stays stable across all certified costs.

The finding that delimits CoSpec's scope: once the estimator is accurate enough
(delta below a threshold), certified cost weights -- including any admissible
LLM-specified cost -- introduce no additional instability; the stability
boundary is set entirely by plant mismatch, which is the classical
certainty-equivalence / robust-LQR problem (cf. Mania et al. 2019), not by the
LLM. CoSpec keeps the LLM out of the plant channel, so its anchoring error is
never added to that boundary.

Run:
    python identified_robust.py
"""
import json

import numpy as np

import case_study as cs

rng = np.random.default_rng(20261016)

WN_TRUE, ZETA_TRUE = 0.8, 0.5
A, B = cs.second_order(WN_TRUE, ZETA_TRUE)
n, m = A.shape[0], B.shape[1]

# Grid of certified admissible costs: independent positive scales on Q and R.
# Every entry is symmetric PSD/pd and detectable, so Fact 1 applies with the
# plant used for synthesis. It spans a beta/alpha range of 0.2..50.
SQ = [0.2, 1.0, 5.0, 50.0]
SR = [0.2, 1.0, 5.0, 50.0]
COST_GRID = [(cs.Q_TASK * sq, cs.R_TASK * sr, sq, sr)
             for sq in SQ for sr in SR]


def worst_rho_closed(Ah, Bh, true_A, true_B):
    """Worst true-loop spectral radius over the certified cost grid."""
    nom = None
    worst = 0.0
    any_ok = False
    for Qh, Rh, sq, sr in COST_GRID:
        try:
            K = cs.lqr(Ah, Bh, Qh, Rh)
        except Exception:
            return None, float("inf")
        r = cs.rho(true_A - true_B @ K)
        worst = max(worst, r)
        any_ok = True
        if sq == 1.0 and sr == 1.0:
            nom = r
    if not any_ok:
        return None, float("inf")
    return nom, worst


# ---------------------------------------------------------------- (A) RLS
def ls_fit(X, U, Xp):
    """Batch least squares of [A;B] from z=[x;u] regressors. Returns Ah, Bh."""
    Z = np.hstack([X, U])                      # T x (n+m)
    ZtZ = Z.T @ Z
    try:
        Theta = np.linalg.solve(ZtZ, Z.T @ Xp)  # (n+m) x n
    except np.linalg.LinAlgError:
        Theta = np.linalg.pinv(Z) @ Xp
    Ah = Theta[:n, :].T
    Bh = Theta[n:, :].T
    return Ah, Bh


def rel_err(Ah, Bh):
    num = np.linalg.norm(Ah - A, "fro") + np.linalg.norm(Bh - B, "fro")
    den = np.linalg.norm(A, "fro") + np.linalg.norm(B, "fro")
    return num / den


def run_rls(seeds=3, T=400, noise=0.01, checkpoints=None):
    if checkpoints is None:
        checkpoints = list(range(10, 60, 10)) + list(range(80, T + 1, 40))
    paths = []
    for s in range(seeds):
        r = np.random.default_rng(1000 + s)
        X = np.zeros((T, n))
        U = np.zeros((T, m))
        Xp = np.zeros((T, n))
        x = np.array([1.0, 0.0])
        for t in range(T):
            # persistently exciting, bounded input
            u = np.array([1.2 * np.sin(0.5 * t)]) + 0.8 * r.standard_normal(m)
            xn = A @ x + B @ u + noise * r.standard_normal(n)
            X[t] = x
            U[t] = u
            Xp[t] = xn
            x = xn
        pts = []
        for t in checkpoints:
            Ah, Bh = ls_fit(X[:t], U[:t], Xp[:t])
            err = rel_err(Ah, Bh)
            nom, worst = worst_rho_closed(Ah, Bh, A, B)
            pts.append(dict(t=t, err=float(err),
                            rho_nominal=nom,
                            rho_worst=float(worst),
                            stable_worst=bool(worst < 1.0)))
        paths.append(pts)
    return paths


# --------------------------------------------------------- (B) mismatch
def mismatch_plant(wn, zeta):
    return cs.second_order(wn, zeta)


def run_mismatch(deltas=None):
    if deltas is None:
        # fine grid across the natural-frequency threshold (~18%), coarse
        # elsewhere, so the reported crossing is resolved to 0.5%
        deltas = np.concatenate([np.arange(0.0, 0.15, 0.025),
                                 np.arange(0.15, 0.251, 0.005),
                                 np.arange(0.275, 0.601, 0.025)])
    out = {}
    for name in ("omega", "zeta", "both"):
        rows = []
        for d in deltas:
            rho_nom, rho_worst = [], []
            for sgn in (1.0, -1.0):
                wn = (WN_TRUE * (1 + sgn * d) if name in ("omega", "both")
                      else WN_TRUE)
                ze = (ZETA_TRUE * (1 + sgn * d) if name in ("zeta", "both")
                      else ZETA_TRUE)
                Ah, Bh = mismatch_plant(wn, ze)
                nom, worst = worst_rho_closed(Ah, Bh, A, B)
                rho_nom.append(nom)
                rho_worst.append(worst)
            wn_r = max(rho_worst)
            nn_r = max(x for x in rho_nom if x is not None)
            rows.append(dict(delta=float(d),
                             rho_nominal=float(nn_r),
                             rho_worst=float(wn_r),
                             stable_nominal=bool(nn_r < 1.0),
                             stable_worst=bool(wn_r < 1.0)))
        stable = [r["delta"] for r in rows if r["stable_worst"]]
        stable_n = [r["delta"] for r in rows if r["stable_nominal"]]
        out[name] = dict(
            rows=rows,
            threshold=float(max(stable)) if stable else 0.0,
            threshold_nominal=float(max(stable_n)) if stable_n else 0.0)
    return out


def print_rls(paths):
    print("=" * 78)
    print("(A) RLS IDENTIFICATION -> certainty-equivalence closed on TRUE plant")
    print("=" * 78)
    print(" samples | rel. err | rho nominal cost | rho worst certified cost | "
          "stable(worst)")
    # average across seeds
    for pts in paths[:1]:
        pass
    for i, pts in enumerate(paths):
        if i == 0:
            for p in pts:
                nom = f"{p['rho_nominal']:.3f}" if p['rho_nominal'] is not None \
                    else "  -- "
                print(f"  {p['t']:4d}   | {p['err']:7.1%} | {nom:>16} | "
                      f"{p['rho_worst']:23.3f} | {p['stable_worst']}")
    print("  (rows show seed 0; all seeds aggregated in JSON)")


def print_mismatch(out):
    print("\n" + "=" * 78)
    print("(B) PLANT MISMATCH -> stability threshold over certified costs")
    print("=" * 78)
    for name in ("omega", "zeta", "both"):
        print(f"  [{name:5s}] stable for rel. error <= "
              f"{out[name]['threshold_nominal']:5.1%} with nominal cost, "
              f"<= {out[name]['threshold']:5.1%} with worst certified cost")
    print("  threshold = largest relative plant error keeping the true loop "
          "stable; the gap between the two columns is the robustness margin "
          "lost by aggressive (high beta/alpha) costs.")


if __name__ == "__main__":
    paths = run_rls()
    print_rls(paths)
    mismatch = run_mismatch()
    print_mismatch(mismatch)

    json.dump(dict(rls=paths,
                   mismatch=mismatch,
                   true_plant=dict(omega_n=WN_TRUE, zeta=ZETA_TRUE),
                   cost_grid=dict(sq=SQ, sr=SR)),
              open("identified_robustness.json", "w"), indent=1)
    print("\nwrote identified_robustness.json")
