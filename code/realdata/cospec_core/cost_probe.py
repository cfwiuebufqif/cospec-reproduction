"""
Direct cost elicitation experiment (response to reviewer Major point #2).

The I4 pipeline in the paper asks the model for plant parameters (omega_n,
zeta) and then maps them to cost weights. That mapping is a monotone positive
scaling, so it (a) always yields admissible weights and (b) fixes the ratio
q1/q2 = 10 identically -- the "0% instability" on that route is then a
consequence of Fact 1, not an empirical finding.

Here we ask the model DIRECTLY for the LQR cost weights (q1, q2, r) of
    J = E[sum_t x1^2 q1 + x2^2 q2 + u^2 r],
for the SAME six frozen vignettes. Every free-form output is archived raw and
pushed through the O(n^3) admissibility certificate. We measure:

  (1) parse rate and certificate rejection rate, with rejection reasons;
  (2) for admitted outputs, the beta/alpha interval and the true cost J/J*;
  (3) the effect of failing closed to the safe default on rejection;
  (4) whether outputs carry patient-specific information beyond a single
      arbitrary positive scale. Uniform scaling leaves the LQR gain unchanged,
      so genuine personalisation must appear in the weight SHAPE: q1/q2 and
      the state/effort balance. The mapped route fixes q1/q2 = 10 by
      construction and cannot express that dimension.

Run:
    python cost_probe.py collect ollama qwen2.5:7b
    $env:DEEPSEEK_API_KEY = (Get-Content $env:TEMP\\cospec_ds_key.txt)
    $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
    python cost_probe.py collect openai deepseek-chat
Analyse:
    python cost_probe.py analyze
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

import case_study as cs

TEMPERATURE = 0.3
N_REPEATS = 20
VIGNETTE_KEYS = ["v1", "v2", "v3", "v4", "v5", "v6"]

import anchoring_probe_v4 as ap

VIGNETTES = {k: ap.VIGNETTES[k] for k in VIGNETTE_KEYS}

PROMPT = """You are designing a linear-quadratic regulator (LQR) for a \
patient's digital psychological intervention.

The state is x = [symptom deviation from baseline; rate of change of \
symptoms], and the control u is the intensity of intervention delivered. \
The running cost is
    J = E[sum_t ( q1 * x1^2 + q2 * x2^2 + r * u^2 )].
- q1 penalizes symptom deviation (larger q1 = prioritize returning symptoms \
to baseline quickly).
- q2 penalizes the rate of change and oscillation (larger q2 = prefer a \
smooth, non-oscillatory recovery).
- r penalizes intervention effort (larger r = avoid aggressive or burdensome \
treatment).
q1, q2, r are cost weights; q1 and q2 are non-negative, r must be strictly \
positive.

Patient: {text}

Choose q1, q2, r for THIS patient, reflecting the trade-off between recovery \
speed, smoothness, and treatment burden for their situation. Respond with \
ONLY a JSON object: {{"q1": <number>, "q2": <number>, "r": <number>, \
"reasoning": "<one sentence>"}}
"""


# ---------------------------------------------------------------- backends
def _post(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_model(prompt, backend, model):
    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        out = _post(f"{host}/api/generate",
                    {"model": model, "prompt": prompt, "stream": False,
                     "options": {"temperature": TEMPERATURE}},
                    {"Content-Type": "application/json"})
        return out.get("response", "")
    if backend == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.environ.get("DEEPSEEK_API_KEY",
                             os.environ.get("OPENAI_API_KEY"))
        out = _post(f"{base}/chat/completions",
                    {"model": model,
                     "messages": [{"role": "user", "content": prompt}],
                     "temperature": TEMPERATURE},
                    {"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        return out["choices"][0]["message"]["content"]
    raise ValueError(f"unknown backend {backend}")


def extract(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- collect
def collect(backend, model):
    rows = []
    jobs = [(vk, v, t) for vk in VIGNETTE_KEYS for v in [VIGNETTES[vk]]
            for t in range(N_REPEATS)]
    print(f"backend={backend} model={model} temp={TEMPERATURE} "
          f"calls={len(jobs)}")
    for i, (vk, v, t) in enumerate(jobs):
        prompt = PROMPT.format(text=v["text"])
        try:
            raw = call_model(prompt, backend, model)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            raw = ""
        d = extract(raw)
        rec = dict(model=model, vignette=vk, prognosis=v["prognosis"],
                   trial=t, raw_response=raw, parse_ok=d is not None)
        if d is not None:
            for k in ("q1", "q2", "r"):
                rec[k] = d.get(k)
        else:
            rec["q1"] = rec["q2"] = rec["r"] = None
        rows.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(jobs)}")
        if backend != "ollama":
            time.sleep(0.2)
    fn = f"cost_{model.replace('/', '_').replace(':', '_')}.json"
    json.dump(rows, open(fn, "w"), indent=2)
    print(f"wrote {fn} ({sum(r['parse_ok'] for r in rows)}/{len(rows)} parsed)")
    return rows


# ---------------------------------------------------------------- scoring
def score(rows, wn_true=0.8, zeta_true=0.5):
    """Attach the certificate decision, true cost and beta/alpha to each row."""
    A, B = cs.second_order(wn_true, zeta_true)
    K_star = cs.lqr(A, B, cs.Q_TASK, cs.R_TASK)
    J_star = cs.cost(A, B, cs.Q_TASK, cs.R_TASK, K_star)

    for r in rows:
        r["admitted"] = False
        r["reject_reason"] = None
        if not r["parse_ok"]:
            r["reject_reason"] = "unparseable"
            continue
        q1, q2, rr = r["q1"], r["q2"], r["r"]
        if not all(isinstance(x, (int, float)) and np.isfinite(x)
                   for x in (q1, q2, rr)):
            r["reject_reason"] = "missing/non-numeric"
            continue
        Qh = np.diag([float(q1), float(q2)])
        Rh = np.array([[float(rr)]])
        if np.min(np.linalg.eigvalsh(Qh)) < -1e-9:
            r["reject_reason"] = "Q not psd"
            continue
        if np.min(np.linalg.eigvalsh(Rh)) <= 1e-9:
            r["reject_reason"] = "R not pd"
            continue
        ok, why = cs.admissible(A, Qh, Rh)
        if not ok:
            r["reject_reason"] = why
            continue
        r["admitted"] = True
        K = cs.lqr(A, B, Qh, Rh)
        r["rho"] = cs.rho(A - B @ K)
        r["unstable"] = r["rho"] >= 1.0
        r["cost_ratio"] = cs.cost(A, B, cs.Q_TASK, cs.R_TASK, K) / J_star
        al, be = cs.alpha_beta(Qh, Rh, cs.Q_TASK, cs.R_TASK)
        r["alpha"], r["beta"], r["bound"] = al, be, be / al
        r["bound_holds"] = r["cost_ratio"] <= r["bound"] + 1e-6
        # weight shape (scale-invariant): the only degrees of freedom that
        # actually change the LQR gain
        r["q1_q2"] = float(q1 / q2) if q2 > 0 else np.inf
        r["q1_r"] = float(q1 / rr)
        r["q2_r"] = float(q2 / rr) if rr > 0 else np.inf
    return rows


# ---------------------------------------------------------------- analyze
def eta2(rows, key, field):
    vals = [r[field] for r in rows if r.get(field) is not None
            and np.isfinite(r[field])]
    if len(vals) < 2:
        return float("nan")
    g = np.mean(vals)
    tot = sum((v - g) ** 2 for v in vals)
    if tot == 0:
        return 0.0
    ss = 0.0
    for grp in set(r[key] for r in rows):
        sub = [r[field] for r in rows if r[key] == grp
               and r.get(field) is not None and np.isfinite(r[field])]
        if sub:
            ss += len(sub) * (np.mean(sub) - g) ** 2
    return ss / tot


def _perm_diff(vals_a, vals_b, n=10000, seed=0):
    """Permutation test for difference of means; returns observed diff, p."""
    rng = np.random.default_rng(seed)
    a, b = np.array(vals_a, float), np.array(vals_b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n):
        perm = rng.permutation(pool)
        count += abs(perm[:na].mean() - perm[na:].mean()) >= abs(obs) - 1e-12
    return obs, (count + 1) / (n + 1)


def analyze():
    fns = sorted(f for f in os.listdir(".")
                 if re.match(r"cost_(?!analysis).*\.json$", f))
    summary = {}
    for fn in fns:
        rows = score(json.load(open(fn)))
        n = len(rows)
        parsed = sum(r["parse_ok"] for r in rows)
        admitted = sum(r["admitted"] for r in rows)
        adm = [r for r in rows if r["admitted"]]
        reasons = {}
        for r in rows:
            if not r["admitted"]:
                reasons[r["reject_reason"]] = reasons.get(r["reject_reason"], 0) + 1

        print("\n" + "=" * 78)
        print(f"DIRECT COST ELICITATION   model={rows[0]['model']}   N={n}")
        print("=" * 78)
        print(f"  parsed           : {parsed}/{n} = {100*parsed/n:.1f}%")
        print(f"  certificate ADMIT: {admitted}/{n} = {100*admitted/n:.1f}%")
        print(f"  rejected         : {n-admitted}/{n} = {100*(n-admitted)/n:.1f}%")
        for why, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"      {why:18s}: {c} ({100*c/n:.1f}%)")

        if adm:
            cr = np.array([r["cost_ratio"] for r in adm], float)
            bd = np.array([r["bound"] for r in adm], float)
            q12 = np.array([r["q1_q2"] for r in adm
                            if np.isfinite(r["q1_q2"])], float)
            print(f"\n  true cost J/J*  (n={len(cr)}): median {np.median(cr):.2f}"
                  f"  p95 {np.percentile(cr,95):.2f}  max {np.max(cr):.2f}")
            print(f"  beta/alpha      : median {np.median(bd):.1f}  "
                  f"p95 {np.percentile(bd,95):.1f}  max {np.max(bd):.1f}")
            print(f"  bound violations: {sum(not r['bound_holds'] for r in adm)}"
                  f"  (zero by Prop 3 / consistency)")
            print(f"  unstable on true loop: {sum(r['unstable'] for r in adm)}"
                  f"  (zero by Fact 1 / consistency, not empirical)")
            print(f"\n  weight SHAPE q1/q2 (mapped I4 fixes this at 10):")
            print(f"    median {np.median(q12):.2f}  p5 {np.percentile(q12,5):.2f}"
                  f"  p95 {np.percentile(q12,95):.2f}")
            fast = [r["q1_q2"] for r in adm if r["prognosis"] == "fast"
                    and np.isfinite(r["q1_q2"])]
            slow = [r["q1_q2"] for r in adm if r["prognosis"] == "slow"
                    and np.isfinite(r["q1_q2"])]
            if fast and slow:
                diff, p = _perm_diff(fast, slow, seed=1)
                print(f"    fast median {np.median(fast):.2f}  "
                      f"slow median {np.median(slow):.2f}  "
                      f"diff {diff:+.2f}  perm p={p:.4f}")
            print(f"    variance share across vignettes (eta^2): "
                  f"{eta2(adm,'vignette','q1_q2'):.1%}")
            print(f"    variance share across vignettes (q1/r): "
                  f"{eta2(adm,'vignette','q1_r'):.1%}")

        # fail-closed behaviour: rejected instances -> safe default
        rejected = [r for r in rows if not r["admitted"]]
        print(f"\n  fail-closed: {len(rejected)} rejected instances revert to "
              f"the safe default (J/J* = 1.00); personalisation is withheld "
              f"on {100*len(rejected)/n:.1f}% of calls.")

        summary[rows[0]["model"]] = dict(
            n=n, parsed=parsed, admitted=admitted,
            admit_rate=admitted / n, reject_reasons=reasons,
            cost_median=float(np.median(cr)) if adm else None,
            cost_p95=float(np.percentile(cr, 95)) if adm else None,
            cost_max=float(np.max(cr)) if adm else None,
            bound_median=float(np.median(bd)) if adm else None,
            bound_p95=float(np.percentile(bd, 95)) if adm else None,
            q12_fast=float(np.median(fast)) if adm and fast else None,
            q12_slow=float(np.median(slow)) if adm and slow else None,
            q12_eta_vignette=float(eta2(adm, "vignette", "q1_q2")) if adm
            else None,
            rejected=len(rejected))

    json.dump(summary, open("cost_analysis.json", "w"), indent=2)
    print("\n\nwrote cost_analysis.json")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "analyze":
        analyze()
    else:
        print("usage: python cost_probe.py collect <backend> <model> | analyze")
