"""
End-to-end case study: route the SAME real LLM outputs through I3 and I4.

This answers the objection that Section 5 only *illustrates* the theory. We
take the 540 logged (omega_n, zeta) pairs that three real models actually
produced (data/anchoring_v3_*.json) and push each one through the two
routings the case-study pipeline conflates:

  I3  (what the pipeline does): build a plant model (Ahat,Bhat) from the
      LLM numbers, synthesise K = LQR(Ahat,Bhat,Q,R), apply to the TRUE
      plant. Guard it with the pipeline's own Monte Carlo gate.
  I4  (what we recommend): map the same numbers to cost weights
      (Qhat,Rhat), synthesise K = LQR(A,B,Qhat,Rhat), apply to the TRUE
      plant. Guard it with the O(n^3) admissibility certificate.

and measure, for each: how often the true closed loop is unstable, what the
true cost is, and -- for I3 -- how often the Monte Carlo gate ACCEPTS a
parameter set that in fact destabilises the true plant (its false-accept
rate, which is the number the certificate is meant to replace).

Honest scoping, stated in the paper as well:
  * There is no ground-truth (omega_n, zeta) for a synthetic vignette, so
    the "true" plant is a fixed nominal choice, and we report a sensitivity
    sweep over that choice rather than a single number.
  * The I4 mapping below is *our* explicit monotone choice, not the
    pipeline's proprietary qScale/rScale constants, which we do not have.
    What matters for the comparison is that it is the same LLM numbers
    entering at a different point, not the particular constants.

Outputs case_study_results.json and prints the tables used in Section 5.
"""
import json
import os

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = [os.path.join(_HERE, os.pardir, "data"), _HERE, os.getcwd()]

FILES = {
    "qwen2.5:7b": "anchoring_v3_qwen2.5_7b.json",
    "deepseek-chat": "anchoring_v3_deepseek-chat.json",
    "gpt-5.6": "anchoring_v3_gpt5.6.json",
}

DT = 0.5
Q_TASK = np.diag([10.0, 1.0])
R_TASK = np.array([[0.5]])

# reference points for the I4 mapping (midpoint of the 'mid' range)
OMEGA_REF, ZETA_REF = 1.05, 1.025


def _resolve(fn):
    for d in _DATA:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


# ------------------------------------------------------------------ plant
def second_order(wn, zeta, dt=DT):
    """Continuous 2nd-order psychological model, discretised."""
    A = np.array([[0.0, 1.0], [-wn**2, -2.0 * zeta * wn]])
    B = np.array([[0.0], [wn**2]])
    return _zoh(A, B, dt)


def lqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def cost(A, B, Q, R, K):
    Acl = A - B @ K
    if rho(Acl) >= 1 - 1e-9:
        return np.inf
    return float(np.trace(solve_discrete_lyapunov(Acl.T, Q + K.T @ R @ K)))


# ------------------------------------------------------- admissibility
def admissible(A, Qh, Rh, tol=1e-9):
    """O(n^3) certificate of Fact 2."""
    if not np.allclose(Qh, Qh.T, atol=1e-10):
        return False, "Q not symmetric"
    if not np.allclose(Rh, Rh.T, atol=1e-10):
        return False, "R not symmetric"
    if np.min(np.linalg.eigvalsh(Qh)) < -tol:
        return False, "Q not psd"
    if np.min(np.linalg.eigvalsh(Rh)) <= tol:
        return False, "R not pd"
    # detectability of (A, Qh^{1/2}) on modes with |lambda| >= 1, via PBH
    w, V = np.linalg.eigh(Qh)
    C = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    n = A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1 - 1e-12:
            M = np.vstack([A - lam * np.eye(n), C])
            if np.linalg.matrix_rank(M, tol=1e-8) < n:
                return False, f"mode {abs(lam):.4f} undetectable"
    return True, "ok"


def alpha_beta(Qh, Rh, Q, R):
    """Extreme generalised eigenvalues of the pencils (Qh,Q) and (Rh,R)."""
    def ext(M, N):
        L = np.linalg.cholesky(N)
        Li = np.linalg.inv(L)
        S = Li @ M @ Li.T
        e = np.linalg.eigvalsh((S + S.T) / 2)
        return float(e[0]), float(e[-1])
    a1, b1 = ext(Qh, Q)
    a2, b2 = ext(Rh, R)
    return min(a1, a2), max(b1, b2)


# ------------------------------------------------- the pipeline's MC gate
def mc_gate(Ah, Bh, Q, R, n_draws=50, eps=0.25, need=0.80):
    """The case-study pipeline's own guard: perturb the IDENTIFIED model by
    +-25%, resynthesise, and require >= 80% of draws to stay stable --
    judged, as the pipeline must, against the identified model, since the
    true plant is unknown to it."""
    ok = 0
    for _ in range(n_draws):
        dA = rng.uniform(-eps, eps, size=Ah.shape)
        dB = rng.uniform(-eps, eps, size=Bh.shape)
        Ap, Bp = Ah * (1 + dA), Bh * (1 + dB)
        try:
            Kp = lqr(Ap, Bp, Q, R)
        except Exception:
            continue
        if rho(Ah - Bh @ Kp) < 1:
            ok += 1
    return ok / n_draws >= need, ok / n_draws


# --------------------------------------------------------------- experiment
def run(wn_true=0.8, zeta_true=0.5, verbose=True):
    A, B = second_order(wn_true, zeta_true)
    K_star = lqr(A, B, Q_TASK, R_TASK)
    J_star = cost(A, B, Q_TASK, R_TASK, K_star)

    rows = []
    for model, fn in FILES.items():
        for r in json.load(open(_resolve(fn))):
            if not r["parse_ok"]:
                continue
            wn, z = r["omega_n"], r["zeta"]
            if wn is None or z is None or wn <= 0 or z <= 0:
                continue

            rec = dict(model=model, vignette=r["vignette"],
                       condition=r["condition"],
                       range_condition=r["range_condition"],
                       omega_n=wn, zeta=z)

            # ---- I3: the numbers become the plant model
            Ah, Bh = second_order(wn, z)
            try:
                K3 = lqr(Ah, Bh, Q_TASK, R_TASK)
                rec["i3_rho"] = rho(A - B @ K3)
                rec["i3_unstable"] = rec["i3_rho"] >= 1.0
                rec["i3_cost_ratio"] = cost(A, B, Q_TASK, R_TASK, K3) / J_star
            except Exception:
                rec["i3_rho"] = float("nan")
                rec["i3_unstable"] = True
                rec["i3_cost_ratio"] = float("inf")
            passed, frac = mc_gate(Ah, Bh, Q_TASK, R_TASK)
            rec["i3_gate_pass"] = bool(passed)
            rec["i3_gate_frac"] = frac

            # ---- I4: the same numbers become cost weights
            Qh = Q_TASK * (wn / OMEGA_REF)
            Rh = R_TASK * (z / ZETA_REF)
            ok, why = admissible(A, Qh, Rh)
            rec["i4_cert_pass"] = bool(ok)
            rec["i4_cert_reason"] = why
            K4 = lqr(A, B, Qh, Rh)
            rec["i4_rho"] = rho(A - B @ K4)
            rec["i4_unstable"] = rec["i4_rho"] >= 1.0
            rec["i4_cost_ratio"] = cost(A, B, Q_TASK, R_TASK, K4) / J_star
            al, be = alpha_beta(Qh, Rh, Q_TASK, R_TASK)
            rec["alpha"], rec["beta"] = al, be
            rec["bound"] = be / al
            rec["bound_holds"] = rec["i4_cost_ratio"] <= rec["bound"] + 1e-6
            rows.append(rec)

    if verbose:
        report(rows, wn_true, zeta_true)
    return rows


def report(rows, wn_true, zeta_true):
    n = len(rows)
    i3u = sum(r["i3_unstable"] for r in rows)
    i4u = sum(r["i4_unstable"] for r in rows)
    cert = sum(r["i4_cert_pass"] for r in rows)
    gate = sum(r["i3_gate_pass"] for r in rows)
    false_acc = sum(r["i3_gate_pass"] and r["i3_unstable"] for r in rows)
    missed = sum((not r["i3_gate_pass"]) and (not r["i3_unstable"])
                 for r in rows)

    print(f"\n{'='*78}")
    print(f"CASE STUDY  true plant: omega_n={wn_true}, zeta={zeta_true}   "
          f"N={n} real LLM outputs")
    print('='*78)
    print(f"  I3 (numbers -> plant model)   unstable: {i3u:4d}/{n}  "
          f"= {100*i3u/n:5.1f}%")
    print(f"  I4 (numbers -> cost weights)  unstable: {i4u:4d}/{n}  "
          f"= {100*i4u/n:5.1f}%")
    print(f"  admissibility certificate passes: {cert}/{n} "
          f"= {100*cert/n:.1f}%")
    print(f"\n  Monte Carlo gate on the I3 path (+-25%, 50 draws, >=80%):")
    print(f"    accepted            : {gate}/{n} = {100*gate/n:5.1f}%")
    print(f"    FALSE ACCEPTS       : {false_acc}/{n} = {100*false_acc/n:5.1f}%"
          f"   (gate passed, true loop unstable)")
    if gate:
        print(f"    false-accept rate among accepted: "
              f"{100*false_acc/gate:5.1f}%")
    print(f"    rejected a safe one : {missed}/{n} = {100*missed/n:5.1f}%")

    # split by stated range -- the striking bit
    print(f"\n  I3 instability by the range the PROMPT stated "
          f"(same patients, same models):")
    for rc in ["narrow_low", "mid", "wide_high"]:
        sub = [r for r in rows if r["range_condition"] == rc]
        if sub:
            u = sum(r["i3_unstable"] for r in sub)
            print(f"    {rc:11s}  {100*u/len(sub):5.1f}%   (n={len(sub)})")

    print(f"\n  I3 instability by model:")
    for m in FILES:
        sub = [r for r in rows if r["model"] == m]
        if sub:
            u = sum(r["i3_unstable"] for r in sub)
            print(f"    {m:14s} {100*u/len(sub):5.1f}%   (n={len(sub)})")

    fin4 = [r["i4_cost_ratio"] for r in rows if np.isfinite(r["i4_cost_ratio"])]
    fin3 = [r["i3_cost_ratio"] for r in rows if np.isfinite(r["i3_cost_ratio"])]
    print(f"\n  true-cost ratio J/J*  (finite cases only)")
    print(f"    I3  median {np.median(fin3):8.2f}   p95 {np.percentile(fin3,95):10.2f}"
          f"   max {np.max(fin3):10.2f}   (n={len(fin3)})")
    print(f"    I4  median {np.median(fin4):8.2f}   p95 {np.percentile(fin4,95):10.2f}"
          f"   max {np.max(fin4):10.2f}   (n={len(fin4)})")

    viol = sum(not r["bound_holds"] for r in rows)
    ratios = [r["bound"] / r["i4_cost_ratio"] for r in rows
              if np.isfinite(r["i4_cost_ratio"]) and r["i4_cost_ratio"] > 0]
    print(f"\n  beta/alpha bound: violations {viol}/{n};  "
          f"slack (bound / actual) median {np.median(ratios):.2f}x, "
          f"p95 {np.percentile(ratios,95):.2f}x, max {np.max(ratios):.2f}x")


if __name__ == "__main__":
    main_rows = run(0.8, 0.5)
    json.dump(main_rows, open("case_study_results.json", "w"), indent=1)

    print(f"\n\n{'#'*78}\nSENSITIVITY to the assumed true plant\n{'#'*78}")
    for wn_t, z_t in [(0.5, 0.5), (0.8, 0.5), (1.2, 0.5),
                      (0.8, 0.3), (0.8, 0.8)]:
        rs = run(wn_t, z_t, verbose=False)
        n = len(rs)
        i3 = 100 * sum(r["i3_unstable"] for r in rs) / n
        i4 = 100 * sum(r["i4_unstable"] for r in rs) / n
        fa = 100 * sum(r["i3_gate_pass"] and r["i3_unstable"] for r in rs) / n
        print(f"  omega_n*={wn_t:<4} zeta*={z_t:<4}   "
              f"I3 unstable {i3:5.1f}%   I4 unstable {i4:5.1f}%   "
              f"gate false-accept {fa:5.1f}%")
    print("\nwrote case_study_results.json")
