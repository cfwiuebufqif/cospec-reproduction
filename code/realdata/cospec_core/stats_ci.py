"""
Confidence intervals and a permutation test for the variance decomposition.

The review of an earlier draft correctly objected that the eta^2 numbers in
the anchoring study were reported as bare descriptives. This script adds:

  1. Percentile bootstrap 95% CIs on every eta^2 (resampling calls within
     each (condition) block, B = 10000).
  2. A permutation test: shuffle the RANGE labels within each vignette and
     recompute eta^2_range, giving the null distribution of "how much
     variance a meaningless grouping would explain". Reported as p and as
     the null's 95th percentile, so the reader can see the floor.
  3. A vignette-level (cluster) bootstrap, resampling vignettes
     with replacement, which is the honest -- and much wider -- interval
     for generalising to new patients.
  4. The normalised-position robustness check: instead of the raw value,
     each answer is re-expressed as its position within the stated range,
     (x - lo) / (hi - lo). If a model were individuating the patient and
     merely rescaling to whatever interval it was handed, raw values would
     move with the range but normalised positions would not. This is the
     analysis that separates genuine anchoring from the trivial fact that
     a shifted interval shifts raw numbers.

Writes stats_ci.json and prints the tables used in Section 4.
"""
import json
import os
import sys

import numpy as np

rng = np.random.default_rng(20260921)
B_BOOT = 10000
B_PERM = 10000

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = [os.path.join(_HERE, os.pardir, "data"), _HERE, os.getcwd()]

FILES = {
    "qwen2.5:7b": "anchoring_v3_qwen2.5_7b.json",
    "deepseek-chat": "anchoring_v3_deepseek-chat.json",
    "gpt-5.6": "anchoring_v3_gpt5.6.json",
}

# The 30-vignette replication (v4). Same design, same analysis, 5x the
# vignettes; only the two models it was collected on. Selected with
#     python stats_ci.py --v4
FILES_V4 = {
    "qwen2.5:7b": "anchoring_v4_qwen2.5_7b.json",
    "deepseek-chat": "anchoring_v4_deepseek-chat.json",
}
CONDS = [("A_vary_omega", "omega_n", "zeta"),
         ("B_vary_zeta", "zeta", "omega_n")]


def _resolve(fn):
    for d in _DATA:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def eta2(vals, groups):
    """Share of variance in `vals` explained by the grouping `groups`."""
    vals = np.asarray(vals, float)
    groups = np.asarray(groups)
    if vals.size < 2:
        return np.nan
    grand = vals.mean()
    ss_tot = ((vals - grand) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_g = 0.0
    for g in np.unique(groups):
        sub = vals[groups == g]
        if sub.size:
            ss_g += sub.size * (sub.mean() - grand) ** 2
    return ss_g / ss_tot


def norm_pos(r, param):
    """Answer re-expressed as its position within the stated range."""
    lo, hi = (r["wn_range"] if param == "omega_n" else r["zeta_range"])
    if hi <= lo:
        return np.nan
    return (r[param] - lo) / (hi - lo)


def boot_ci(vals, groups, B=B_BOOT):
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    n = len(vals)
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = eta2(vals[idx], groups[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def cluster_boot_ci(rows, param, key, B=2000):
    """Resample vignettes with replacement."""
    vigs = sorted({r["vignette"] for r in rows})
    out = []
    for b in range(B):
        pick = rng.choice(vigs, len(vigs), replace=True)
        vals, groups = [], []
        for v in pick:
            for r in rows:
                if r["vignette"] == v and r[param] is not None:
                    vals.append(r[param])
                    groups.append(r[key])
        if len(set(groups)) > 1:
            out.append(eta2(vals, groups))
    if not out:
        return (np.nan, np.nan)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def perm_test(rows, param, B=B_PERM):
    """Null: range labels are exchangeable within a vignette."""
    vals = np.array([r[param] for r in rows], float)
    groups = np.array([r["range_condition"] for r in rows])
    vigs = np.array([r["vignette"] for r in rows])
    obs = eta2(vals, groups)
    null = np.empty(B)
    for b in range(B):
        perm = groups.copy()
        for v in np.unique(vigs):
            m = vigs == v
            perm[m] = rng.permutation(groups[m])
        null[b] = eta2(vals, perm)
    p = (1.0 + np.sum(null >= obs)) / (B + 1.0)
    return obs, float(p), float(np.percentile(null, 95))


def main():
    global FILES
    v4 = "--v4" in sys.argv
    if v4:
        FILES = FILES_V4
    results = {}
    print("=" * 92)
    print("VARIANCE DECOMPOSITION WITH UNCERTAINTY  "
          "(eta^2, 95% bootstrap CI, B=10000)")
    print("=" * 92)

    for model, fn in FILES.items():
        allrows = [r for r in json.load(open(_resolve(fn))) if r["parse_ok"]]
        results[model] = {}
        print(f"\n### {model}")
        for cond, varied, fixed in CONDS:
            rows = [r for r in allrows if r["condition"] == cond
                    and r[varied] is not None and r[fixed] is not None]
            print(f"\n  {cond}   (range of '{varied}' manipulated)")
            for param, role in [(varied, "VARIED "), (fixed, "control")]:
                vals = [r[param] for r in rows]
                e_r = eta2(vals, [r["range_condition"] for r in rows])
                e_v = eta2(vals, [r["vignette"] for r in rows])
                lo_r, hi_r = boot_ci(vals, [r["range_condition"] for r in rows])
                lo_v, hi_v = boot_ci(vals, [r["vignette"] for r in rows])
                cl_r = cluster_boot_ci(rows, param, "range_condition")
                obs, p, null95 = perm_test(rows, param)

                # normalised-position version
                npos = [norm_pos(r, param) for r in rows]
                e_rn = eta2(npos, [r["range_condition"] for r in rows])
                e_vn = eta2(npos, [r["vignette"] for r in rows])

                print(f"    {param:8s}[{role}] "
                      f"RANGE {e_r:6.1%} [{lo_r:.1%},{hi_r:.1%}]"
                      f"  (vignette-cluster CI [{cl_r[0]:.1%},{cl_r[1]:.1%}])"
                      f"   VIGNETTE {e_v:6.1%} [{lo_v:.1%},{hi_v:.1%}]")
                print(f"    {'':18s}perm-test p={p:.4f}  "
                      f"(null 95th pct = {null95:.1%})"
                      f"   | NORMALISED: range {e_rn:6.1%}  "
                      f"vignette {e_vn:6.1%}")

                results[model].setdefault(cond, {})[param] = dict(
                    role=role.strip(), eta2_range=e_r, eta2_range_ci=[lo_r, hi_r],
                    eta2_range_cluster_ci=list(cl_r),
                    eta2_vignette=e_v, eta2_vignette_ci=[lo_v, hi_v],
                    perm_p=p, perm_null_p95=null95,
                    eta2_range_normalised=e_rn,
                    eta2_vignette_normalised=e_vn)

    print("\n" + "=" * 92)
    print("READING THE NORMALISED COLUMN")
    print("=" * 92)
    print("""
  A model that understood the range as a unit -- individuating the patient
  and then rescaling into whatever interval it was handed -- would show
  high raw range-variance (trivially) but LOW normalised range-variance,
  because its position within the interval would be set by the patient.
  Normalised range-variance that stays high means the answer is pinned to
  a location in the stated interval rather than to the patient: that is
  anchoring, not rescaling.""")

    json.dump(results, open("stats_ci_v4.json" if v4 else "stats_ci.json", "w"), indent=1)
    print("\nwrote " + ("stats_ci_v4.json" if v4 else "stats_ci.json"))


if __name__ == "__main__":
    main()
