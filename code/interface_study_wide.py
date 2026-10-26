"""
The interface asymmetry across 29 plants instead of 5.

Proposition 2 says plant-model error can destabilize and Fact 1 says
cost-weight error cannot. The original numerics used five hand-chosen
plants, which invites the obvious objection that five is a small and
possibly convenient sample. This script repeats both sweeps over the five
original plants plus the 24 physical systems of physics_probe.py -- 8
mass-spring-dampers, 8 series RLC circuits, 8 two-tank thermal plants,
spanning omega_n from 0.25 to 31.6 and zeta from 0.05 to 5.05, all with
parameters fixed by physics rather than chosen by us.

No language model is involved here: this is the control-theoretic claim,
measured on a wider and independently-specified plant set.

Writes wide_study.json and prints the tables used in Section 3.
"""
import json
import math
import os

import numpy as np
from scipy.linalg import expm, solve_discrete_are, solve_discrete_lyapunov

rng = np.random.default_rng(20260921)
_HERE = os.path.dirname(os.path.abspath(__file__))

EPS = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
ETA = [0.5, 2.0, 8.0]
NDRAW = 200


# --------------------------------------------------------------- the plants
def disc(A, B, dt):
    """Exact zero-order-hold discretisation.

    An earlier version truncated the series at second order, which is fine
    when |lambda| dt is small but silently wrong when it is not: for the
    stiff thermal plant t3 (poles at -0.1 and -10) at dt = 0.2 the truncation
    maps the fast pole to exactly 1.0 instead of e^-2 = 0.135, manufacturing
    a marginally stable plant and, with it, spurious 'instability' at the
    cost interface. Fact 1 was never in danger; the discretisation was.
    """
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def second_order(wn, zeta, dt):
    A = np.array([[0.0, 1.0], [-wn**2, -2 * zeta * wn]])
    B = np.array([[0.0], [wn**2]])
    return disc(A, B, dt)


def original_plants():
    P = {}
    A = np.array([[-.020, .005, .005, -.008], [.005, -.030, .005, -.006],
                  [.005, .005, -.040, -.004], [-.004, -.004, -.004, -.015]])
    B = np.array([[-.30, -.15], [-.20, -.30], [-.15, -.35], [.25, .12]])
    Ad, Bd = disc(A, B, 0.5)
    P["psych (4-state)"] = (Ad, Bd, np.diag([15., 12., 6., 4.]),
                            np.diag([.5, .5]), "original")
    Ad, Bd = second_order(0.8, 0.3, 0.5)
    P["oscillator"] = (Ad, Bd, np.diag([10., 1.]), np.array([[.5]]), "original")
    P["double integrator"] = (np.array([[1., .5], [0., 1.]]),
                              np.array([[.125], [.5]]),
                              np.diag([5., 1.]), np.array([[.5]]), "original")
    n, k, c, dt = 3, 1., .2, .2
    A = np.zeros((2 * n, 2 * n))
    for i in range(n):
        A[i, n + i] = 1.
    for i in range(n):
        if i > 0:
            A[n + i, i - 1] += k
        A[n + i, i] += -2 * k if i < n - 1 else -k
        if i < n - 1:
            A[n + i, i + 1] += k
        A[n + i, n + i] = -c
    B = np.zeros((2 * n, 1)); B[2 * n - 1, 0] = 1.
    Ad, Bd = disc(A, B, dt)
    P["spring chain (n=3)"] = (Ad, Bd, np.eye(2 * n), np.array([[.5]]),
                               "original")
    r = np.random.default_rng(7)
    A = r.normal(scale=.3, size=(4, 4))
    A = A / (np.max(np.abs(np.linalg.eigvals(A))) * 1.2)
    P["random 4x2"] = (A, r.normal(scale=.5, size=(4, 2)), np.eye(4) * 5.,
                       np.eye(2) * .5, "original")
    return P


def physics_plants():
    """The 24 systems of physics_probe.py, as plants."""
    spec = []
    for nm, m, k, c in [("m1", 4., 1., .4), ("m2", 1., 1., .1),
                        ("m3", 2., 8., 1.6), ("m4", 1., 16., 4.),
                        ("m5", .5, 32., 2.), ("m6", 2., 200., 80.),
                        ("m7", .25, 25., 1.5), ("m8", .1, 40., .8)]:
        spec.append((nm, "mass-spring-damper", math.sqrt(k / m),
                     c / (2 * math.sqrt(k * m))))
    for nm, L, C, R in [("r1", 1., 1., .2), ("r2", .1, .4, .5),
                        ("r3", .01, 1., .06), ("r4", 4., 1., .4),
                        ("r5", .25, .01, 2.), ("r6", 1., .25, 4.),
                        ("r7", .04, .25, .2), ("r8", 2., .5, .4)]:
        spec.append((nm, "series RLC", 1 / math.sqrt(L * C),
                     (R / 2) * math.sqrt(C / L)))
    for nm, t1, t2 in [("t1", 1., 1.), ("t2", .5, 2.), ("t3", .1, 10.),
                       ("t4", .2, .5), ("t5", .05, .2), ("t6", 2., 8.),
                       ("t7", .02, .05), ("t8", 4., 4.)]:
        spec.append((nm, "two-tank thermal", 1 / math.sqrt(t1 * t2),
                     (t1 + t2) / (2 * math.sqrt(t1 * t2))))
    P = {}
    for nm, fam, wn, z in spec:
        dt = min(0.2 / wn, 0.5)          # sample well inside the bandwidth
        Ad, Bd = second_order(wn, z, dt)
        P[nm] = (Ad, Bd, np.diag([10., 1.]), np.array([[.5]]), fam)
    return P


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


def study(PL):
    out = {}
    for name, (A, B, Q, R, fam) in PL.items():
        try:
            K = lqr(A, B, Q, R)
        except Exception:
            print(f"  skip {name}: DARE failed"); continue
        rec = dict(family=fam, i3={}, i4={})
        for e in EPS:                                   # I3
            bad = 0
            for _ in range(NDRAW):
                dA = rng.normal(size=A.shape)
                dA *= np.linalg.norm(A) / np.linalg.norm(dA)
                dB = rng.normal(size=B.shape)
                dB *= np.linalg.norm(B) / np.linalg.norm(dB)
                try:
                    Kh = lqr(A + e * dA, B + e * dB, Q, R)
                except Exception:
                    bad += 1; continue
                if rho(A - B @ Kh) >= 1:
                    bad += 1
            rec["i3"][str(e)] = 100 * bad / NDRAW
        Js = cost(A, B, Q, R, K)
        for eta in ETA:                                 # I4
            bad, worst, viol, n = 0, 0.0, 0, 0
            for _ in range(NDRAW):
                fq = np.exp(eta * rng.normal(size=Q.shape[0]))
                fr = np.exp(eta * rng.normal(size=R.shape[0]))
                try:
                    Kh = lqr(A, B, np.diag(np.diag(Q) * fq),
                             np.diag(np.diag(R) * fr))
                except Exception:
                    bad += 1; continue
                n += 1
                if rho(A - B @ Kh) >= 1:
                    bad += 1; continue
                ratio = cost(A, B, Q, R, Kh) / Js
                bound = max(fq.max(), fr.max()) / min(fq.min(), fr.min())
                worst = max(worst, ratio)
                if ratio > bound + 1e-6:
                    viol += 1
            rec["i4"][str(eta)] = dict(unstable_pct=100 * bad / max(n, 1),
                                       worst_ratio=worst, violations=viol,
                                       n=n)
        out[name] = rec
    return out


if __name__ == "__main__":
    PL = {**original_plants(), **physics_plants()}
    print(f"{len(PL)} plants "
          f"({sum(v[4]=='original' for v in PL.values())} original + "
          f"{sum(v[4]!='original' for v in PL.values())} physical)\n")
    res = study(PL)

    print("=" * 78)
    print("I3  plant-model error: % of draws that destabilise the true loop")
    print("=" * 78)
    print(f"{'plant':22s}" + "".join(f"{int(e*100):>7d}%" for e in EPS))
    for fam in ["original", "mass-spring-damper", "series RLC",
                "two-tank thermal"]:
        sel = {k: v for k, v in res.items() if v["family"] == fam} if \
            fam != "original" else \
            {k: v for k, v in res.items() if v["family"] == "original"}
        if not sel:
            continue
        print(f"-- {fam}")
        for k, v in sel.items():
            print(f"   {k:19s}" + "".join(f"{v['i3'][str(e)]:7.1f}%"
                                          for e in EPS))
        mean = [np.mean([v["i3"][str(e)] for v in sel.values()]) for e in EPS]
        print(f"   {'mean':19s}" + "".join(f"{m:7.1f}%" for m in mean))

    allv = [[v["i3"][str(e)] for v in res.values()] for e in EPS]
    print(f"\n{'ALL 29 plants':22s}" +
          "".join(f"{np.mean(c):7.1f}%" for c in allv))
    print(f"{'  min / max':22s}" +
          "".join(f"{np.min(c):3.0f}/{np.max(c):<3.0f}" for c in allv))

    print("\n" + "=" * 78)
    print("I4  cost-weight error, same 29 plants")
    print("=" * 78)
    tot_unstable = tot_viol = tot_n = 0
    for eta in ETA:
        u = [res[k]["i4"][str(eta)]["unstable_pct"] for k in res]
        w = [res[k]["i4"][str(eta)]["worst_ratio"] for k in res]
        v = sum(res[k]["i4"][str(eta)]["violations"] for k in res)
        n = sum(res[k]["i4"][str(eta)]["n"] for k in res)
        tot_unstable += sum(u); tot_viol += v; tot_n += n
        print(f"  eta={eta:<4} unstable {np.mean(u):5.2f}%   "
              f"worst J/J* over plants {np.max(w):12.4g}   "
              f"bound violations {v}/{n}")
    print(f"\n  across every draw: {tot_n} samples, "
          f"{'0' if tot_unstable == 0 else tot_unstable} unstable, "
          f"{tot_viol} violations of the beta/alpha bound")
    json.dump(res, open("wide_study.json", "w"), indent=1)
    print("\nwrote wide_study.json")
