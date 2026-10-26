"""Cross-model agreement on v3 (clean-vignette) anchoring data.

For each model, take the CONTROL parameter's mean value per vignette,
averaged over the 3 range conditions within condition A (control=zeta)
and condition B (control=omega_n) -- i.e. the values least contaminated
by the deliberate range manipulation -- and compare across models.
"""
import json
import numpy as np
from scipy.stats import pearsonr

FILES = {
    "qwen2.5:7b": "anchoring_v3_qwen2.5_7b.json",
    "deepseek-chat": "anchoring_v3_deepseek-chat.json",
    "gpt5.6": "anchoring_v3_gpt5.6.json",
}

# Resolve data files whether this is run from code/, from the package root,
# or from a flat directory holding everything.
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [os.path.join(_HERE, os.pardir, "data"), _HERE,
                   os.path.join(_HERE, "data"), os.getcwd()]


def _resolve(fn):
    for d in _CANDIDATE_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"{fn} not found; looked in: {', '.join(_CANDIDATE_DIRS)}")


data = {m: json.load(open(_resolve(f))) for m, f in FILES.items()}
VIGS = [f"v{i}" for i in range(1, 7)]

def control_means(rows, cond, param):
    out = {}
    for v in VIGS:
        xs = [r[param] for r in rows if r["condition"] == cond
              and r["vignette"] == v and r["parse_ok"]]
        out[v] = np.mean(xs)
    return out

print("Per-vignette CONTROL-parameter means (avg over range conditions folded in already via full set)\n")
means = {}
for m, rows in data.items():
    means[m] = {
        "zeta_ctrl_in_A": control_means(rows, "A_vary_omega", "zeta"),
        "omega_ctrl_in_B": control_means(rows, "B_vary_zeta", "omega_n"),
    }
    print(m, means[m])

print("\n== Cross-model Pearson r on control-omega_n (condition B), by vignette ==")
models = list(FILES)
for i in range(len(models)):
    for j in range(i+1, len(models)):
        a = [means[models[i]]["omega_ctrl_in_B"][v] for v in VIGS]
        b = [means[models[j]]["omega_ctrl_in_B"][v] for v in VIGS]
        r, p = pearsonr(a, b)
        print(f"  {models[i]:14s} vs {models[j]:14s}  r={r:+.3f}  p={p:.3f}  "
              f"ratio(max/min of means)={max(a)/min(a):.2f}/{max(b)/min(b):.2f}")

print("\n== Cross-model Pearson r on control-zeta (condition A), by vignette ==")
for i in range(len(models)):
    for j in range(i+1, len(models)):
        a = [means[models[i]]["zeta_ctrl_in_A"][v] for v in VIGS]
        b = [means[models[j]]["zeta_ctrl_in_A"][v] for v in VIGS]
        r, p = pearsonr(a, b)
        print(f"  {models[i]:14s} vs {models[j]:14s}  r={r:+.3f}  p={p:.3f}")

print("\n== Fast vs slow prognosis, control-omega_n means (cond B), by model ==")
prog = {"v1":"fast","v2":"fast","v3":"fast","v4":"slow","v5":"slow","v6":"slow"}
for m in models:
    d = means[m]["omega_ctrl_in_B"]
    fast = np.mean([d[v] for v in VIGS if prog[v]=="fast"])
    slow = np.mean([d[v] for v in VIGS if prog[v]=="slow"])
    print(f"  {m:14s}  fast={fast:.3f}  slow={slow:.3f}  sep={fast-slow:+.3f}  ratio={fast/slow:.2f}x")
