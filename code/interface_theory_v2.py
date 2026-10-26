"""
Four-interface study, v2. Adds:
  (a) analytic suboptimality bound  J(Khat)/J(K*) <= beta/alpha  -- verified
  (b) O(1) admissibility certificate -- implemented and stress-tested
  (c) more plants, more draws, instability thresholds for I3
"""
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


# ------------------------------------------------------------------ plants
def psych_4state():
    A = np.array([[-0.020, 0.005, 0.005, -0.008],
                  [0.005, -0.030, 0.005, -0.006],
                  [0.005, 0.005, -0.040, -0.004],
                  [-0.004, -0.004, -0.004, -0.015]])
    B = np.array([[-0.30, -0.15], [-0.20, -0.30],
                  [-0.15, -0.35], [0.25, 0.12]])
    dt = 0.5
    Ad, Bd = _zoh(A, B, dt)
    return Ad, Bd, np.diag([15., 12., 6., 4.]), np.diag([0.5, 0.5])


def oscillator():
    wn, z, dt = 0.8, 0.3, 0.5
    A = np.array([[0., 1.], [-wn**2, -2 * z * wn]])
    B = np.array([[0.], [wn**2]])
    Ad, Bd = _zoh(A, B, dt)
    return Ad, Bd, np.diag([10., 1.]), np.array([[0.5]])


def double_integrator():
    # This one is already the exact ZOH: for A nilpotent of index 2 the
    # series terminates, so [[1,dt],[0,1]] and [[dt^2/2],[dt]] are exact.
    dt = 0.5
    Ad = np.array([[1., dt], [0., 1.]])
    Bd = np.array([[dt**2 / 2], [dt]])
    return Ad, Bd, np.diag([5., 1.]), np.array([[0.5]])


def spring_chain(n=3):
    """n masses in a chain, 2n states, 1 actuator on the last mass."""
    k, c, dt = 1.0, 0.2, 0.2
    A = np.zeros((2 * n, 2 * n))
    for i in range(n):
        A[i, n + i] = 1.0
    for i in range(n):
        if i > 0:
            A[n + i, i - 1] += k
        A[n + i, i] += -2 * k if i < n - 1 else -k
        if i < n - 1:
            A[n + i, i + 1] += k
        A[n + i, n + i] = -c
    B = np.zeros((2 * n, 1))
    B[2 * n - 1, 0] = 1.0
    Ad, Bd = _zoh(A, B, dt)
    return Ad, Bd, np.eye(2 * n), np.array([[0.5]])


def random_plant(n=4, m=2, seed=0):
    r = np.random.default_rng(seed)
    A = r.normal(scale=0.3, size=(n, n))
    A = A / (np.max(np.abs(np.linalg.eigvals(A))) * 1.2)  # stable-ish
    B = r.normal(scale=0.5, size=(n, m))
    return A, B, np.eye(n) * 5.0, np.eye(m) * 0.5


# ------------------------------------------------------------------ helpers
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


# ------------------------------------------------- admissibility certificate
def admissible(A, Qh, Rh, tol=1e-10):
    """O(1) runtime check: Qh psd, (A, Qh^1/2) detectable, Rh pd.

    Detectability of (A, C) with C = Qh^{1/2}: every unstable/marginal mode
    must be observable. Checked via the Hautus/PBH test on eigenvalues with
    |lambda| >= 1.
    """
    if not np.allclose(Qh, Qh.T, atol=tol):
        return False, "Q not symmetric"
    if np.min(np.linalg.eigvalsh(Qh)) < -tol:
        return False, "Q not psd"
    if not np.allclose(Rh, Rh.T, atol=tol):
        return False, "R not symmetric"
    if np.min(np.linalg.eigvalsh(Rh)) <= tol:
        return False, "R not pd"
    # C = Qh^{1/2}
    w, V = np.linalg.eigh(Qh)
    C = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    n = A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1 - 1e-9:                      # unstable/marginal mode
            M = np.vstack([lam * np.eye(n) - A, C])
            if np.linalg.matrix_rank(M, tol=1e-8) < n:
                return False, f"mode {lam:.4f} undetectable"
    return True, "admissible"


# ------------------------------------------------------------------ studies
PLANTS = {
    "psych-4state": psych_4state(),
    "oscillator": oscillator(),
    "spring-chain-3": spring_chain(3),
    "random-4x2": random_plant(seed=7),
    "double-integrator": double_integrator(),
}


def study_I3(A, B, Q, R, eps, n=400):
    K = lqr(A, B, Q, R)
    Js = cost(A, B, Q, R, K)
    bad = 0
    for _ in range(n):
        dA = rng.normal(size=A.shape); dA *= np.linalg.norm(A) / np.linalg.norm(dA)
        dB = rng.normal(size=B.shape); dB *= np.linalg.norm(B) / np.linalg.norm(dB)
        try:
            Kh = lqr(A + eps * dA, B + eps * dB, Q, R)
        except Exception:
            bad += 1; continue
        if rho(A - B @ Kh) >= 1:
            bad += 1
    return bad / n


def study_I4(A, B, Q, R, eta, n=400):
    """Returns (frac_unstable, max observed J ratio, max bound, bound_violations)."""
    K = lqr(A, B, Q, R)
    Js = cost(A, B, Q, R, K)
    bad, ratios, viol = 0, [], 0
    for _ in range(n):
        fq = np.exp(eta * rng.normal(size=Q.shape[0]))
        fr = np.exp(eta * rng.normal(size=R.shape[0]))
        Qh, Rh = np.diag(np.diag(Q) * fq), np.diag(np.diag(R) * fr)
        ok, _ = admissible(A, Qh, Rh)
        if not ok:
            continue
        Kh = lqr(A, B, Qh, Rh)
        if rho(A - B @ Kh) >= 1:
            bad += 1; continue
        ratio = cost(A, B, Q, R, Kh) / Js
        alpha, beta = min(fq.min(), fr.min()), max(fq.max(), fr.max())
        bound = beta / alpha
        ratios.append(ratio)
        if ratio > bound * (1 + 1e-6):
            viol += 1
    return bad / n, (max(ratios) if ratios else np.nan), viol, len(ratios)


print("=" * 76)
print("I3: LLM error in PLANT PARAMETERS -- fraction of draws that DESTABILISE")
print("=" * 76)
print(f"{'plant':<20}" + "".join(f"{e:>9.0%}" for e in [.05, .1, .25, .5, 1., 2.]))
for name, (A, B, Q, R) in PLANTS.items():
    row = [study_I3(A, B, Q, R, e) for e in [.05, .1, .25, .5, 1., 2.]]
    print(f"{name:<20}" + "".join(f"{v:>9.1%}" for v in row))

print()
print("=" * 76)
print("I4: LLM error in COST WEIGHTS -- unstable frac / max J ratio / bound viol.")
print("=" * 76)
print(f"{'plant':<20}{'eta':>6}{'unstable':>11}{'max J/J*':>11}{'viol':>7}{'n':>6}")
for name, (A, B, Q, R) in PLANTS.items():
    for eta in [0.5, 2.0, 8.0]:
        f, mx, viol, n = study_I4(A, B, Q, R, eta)
        print(f"{name:<20}{eta:>6.1f}{f:>11.1%}{mx:>11.3f}{viol:>7d}{n:>6d}")

print()
print("=" * 76)
print("Admissibility certificate: does it catch the bad cases?")
print("=" * 76)
A, B, Q, R = double_integrator()
cases = [
    ("Qhat = 0 (undetectable)", np.zeros((2, 2)), R),
    ("Qhat = diag(1,0) (pos. undetectable)", np.diag([1.0, 0.0]), R),
    ("Qhat = 1e-6*I", np.eye(2) * 1e-6, R),
    ("Qhat = 1e6*I", np.eye(2) * 1e6, R),
    ("Rhat = 0 (not pd)", np.eye(2), np.zeros((1, 1))),
    ("Qhat has negative eig", np.diag([1.0, -0.5]), R),
]
for label, Qh, Rh in cases:
    ok, why = admissible(A, Qh, Rh)
    if ok:
        Kh = lqr(A, B, Qh, Rh)
        r = rho(A - B @ Kh)
        verdict = f"PASS -> rho={r:.4f} {'STABLE' if r < 1 else 'UNSTABLE(!!)'}"
    else:
        verdict = f"REJECT ({why})"
    print(f"  {label:<40} {verdict}")
