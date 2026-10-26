"""
Does the I4 result depend on the mapping we chose?

Section 5 routes each logged answer to the cost weights through a mapping
that is ours, not the pipeline's: Qhat = Q * (omega_n / OMEGA_REF) and
Rhat = R * (zeta / ZETA_REF). The obvious objection is that the I4 numbers
are an artefact of those two constants.

Two halves of that objection, and they have different answers.

  STABILITY cannot depend on the mapping. Any strictly positive scaling of
  (Q,R) is admissible, so Fact 1 forbids destabilisation for every mapping
  in this family, not just ours. We verify this numerically rather than
  assert it, but a non-zero result here would mean the code is wrong, not
  that the theorem is.

  PERFORMANCE can. The realised cost ratio J(Khat)/J(K*) and the slack in
  the beta/alpha envelope are properties of the particular weights, so they
  move with the mapping. That is the part worth measuring, and it is what
  this script reports.

Sweeps both reference constants over a wide grid and also two alternative
functional forms (square-root compression and a squared expansion), on all
540 logged answers each time.

Writes mapping_sweep.json.
"""
import json
import math
import os

import numpy as np
from scipy.linalg import expm, solve_discrete_are, solve_discrete_lyapunov

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = [os.path.join(_HERE, os.pardir, "data"), _HERE, os.getcwd()]

DT = 0.5
Q_TASK = np.diag([10.0, 1.0])
R_TASK = np.array([[0.5]])
WN_TRUE, ZETA_TRUE = 0.8, 0.5
FILES = {"qwen2.5:7b": "anchoring_v3_qwen2.5_7b.json",
         "deepseek-chat": "anchoring_v3_deepseek-chat.json",
         "gpt-5.6": "anchoring_v3_gpt5.6.json"}


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


def cost(A, B, Q, R, K):
    Acl = A - B @ K
    if rho(Acl) >= 1 - 1e-9:
        return float("inf")
    return float(np.trace(solve_discrete_lyapunov(Acl.T, Q + K.T @ R @ K)))


def alpha_beta(Qh, Rh, Q, R):
    def ext(X, Y):
        w = np.linalg.eigvals(np.linalg.solve(Y, X)).real
        return float(np.min(w)), float(np.max(w))
    a1, b1 = ext(Qh, Q)
    a2, b2 = ext(Rh, R)
    return min(a1, a2), max(b1, b2)


A, B = second_order(WN_TRUE, ZETA_TRUE)
K_STAR = lqr(A, B, Q_TASK, R_TASK)
J_STAR = cost(A, B, Q_TASK, R_TASK, K_STAR)

answers = []
for model, fn in FILES.items():
    for r in json.load(open(_resolve(fn))):
        if r["parse_ok"] and r["omega_n"] and r["zeta"] \
                and r["omega_n"] > 0 and r["zeta"] > 0:
            answers.append((r["omega_n"], r["zeta"]))


def evaluate(fq, fr, label):
    """Route every logged answer through a given mapping and score it."""
    unstable, ratios, slacks, viol = 0, [], [], 0
    for wn, z in answers:
        Qh = Q_TASK * fq(wn)
        Rh = R_TASK * fr(z)
        if np.min(np.diag(Qh)) <= 0 or np.min(np.diag(Rh)) <= 0:
            continue
        K = lqr(A, B, Qh, Rh)
        if rho(A - B @ K) >= 1.0:
            unstable += 1
            continue
        ratio = cost(A, B, Q_TASK, R_TASK, K) / J_STAR
        al, be = alpha_beta(Qh, Rh, Q_TASK, R_TASK)
        bound = be / al
        ratios.append(ratio)
        slacks.append(bound / ratio)
        if ratio > bound + 1e-6:
            viol += 1
    return dict(label=label, n=len(ratios),
                unstable_pct=100 * unstable / max(len(answers), 1),
                med_ratio=float(np.median(ratios)),
                p95_ratio=float(np.percentile(ratios, 95)),
                max_ratio=float(np.max(ratios)),
                med_slack=float(np.median(slacks)),
                violations=viol)


if __name__ == "__main__":
    rows = []

    # --- the reference constants, swept over a wide grid
    for om in [0.5, 0.75, 1.05, 1.5, 2.0]:
        for ze in [0.5, 1.025, 2.0]:
            rows.append(evaluate(lambda w, o=om: w / o,
                                 lambda z, e=ze: z / e,
                                 f"linear  OMEGA_REF={om}, ZETA_REF={ze}"))

    # --- alternative functional forms at the nominal constants
    rows.append(evaluate(lambda w: math.sqrt(w / 1.05),
                         lambda z: math.sqrt(z / 1.025),
                         "sqrt compression"))
    rows.append(evaluate(lambda w: (w / 1.05) ** 2,
                         lambda z: (z / 1.025) ** 2,
                         "squared expansion"))

    json.dump(rows, open(os.path.join(_HERE, "mapping_sweep.json"), "w"),
              indent=1)

    print(f"{len(answers)} logged answers, routed through "
          f"{len(rows)} mappings\n")
    print(f"{'mapping':40s} {'unstable':>9s} {'med J/J*':>9s} "
          f"{'p95':>8s} {'med slack':>10s} {'viol':>5s}")
    for r in rows:
        print(f"{r['label']:40s} {r['unstable_pct']:8.1f}% "
              f"{r['med_ratio']:9.2f} {r['p95_ratio']:8.2f} "
              f"{r['med_slack']:9.2f}x {r['violations']:5d}")

    u = {r["unstable_pct"] for r in rows}
    v = {r["violations"] for r in rows}
    print(f"\nstability: unstable rate takes the value(s) {u} across every "
          f"mapping")
    print(f"bound:     violations take the value(s) {v}")
    print(f"performance: median J/J* ranges "
          f"{min(r['med_ratio'] for r in rows):.2f}--"
          f"{max(r['med_ratio'] for r in rows):.2f}, so it does depend on "
          f"the mapping, as expected")
