"""
Anchoring probe v3.

Changes from v2 (both matter for reviewers):
  1. VIGNETTES REWRITTEN. v2's vignettes leaked the answer -- they contained
     phrases like "came on fast ... already feel better" and "always taken
     months to lift", i.e. they stated the recovery speed in words and the
     model only had to map a word onto a number. v3 vignettes contain ONLY
     clinical facts that a clinician would use to form a prognosis
     (episode count, duration, onset, severity, comorbidity, support,
     prior trials) and never name the speed, trajectory, or prognosis.
  2. Six vignettes instead of four, spanning the prognostic spectrum.

Design (unchanged, this is the point):
  Condition A: omega_n's range varied, zeta's held fixed
  Condition B: zeta's range varied, omega_n's held fixed
If the anchoring follows the MANIPULATION rather than the PARAMETER, the
effect is a property of prompt structure, not of the quantity.

--------------------------------------------------------------------------
HOW TO RUN

  # local
  export ANCHOR_BACKEND=ollama
  export ANCHOR_MODEL=qwen2.5:7b
  python anchoring_probe_v3.py

  # DeepSeek
  export ANCHOR_BACKEND=openai
  export OPENAI_BASE_URL=https://api.deepseek.com
  export OPENAI_API_KEY=...
  export ANCHOR_MODEL=deepseek-chat
  python anchoring_probe_v3.py

  # OpenAI
  export ANCHOR_BACKEND=openai
  export OPENAI_BASE_URL=https://api.openai.com/v1
  export OPENAI_API_KEY=...
  export ANCHOR_MODEL=gpt-4o
  python anchoring_probe_v3.py

  # Anthropic
  export ANCHOR_BACKEND=anthropic
  export ANTHROPIC_API_KEY=...
  export ANCHOR_MODEL=claude-sonnet-4-5
  python anchoring_probe_v3.py

  # re-analyse a saved file without re-running
  python anchoring_probe_v3.py anchoring_v3_<model>.json

Each run = 2 conditions x 3 ranges x 6 vignettes x 5 repeats = 180 calls.
Roughly 5-8 minutes on an API, a few cents. Paste the printed table back.
--------------------------------------------------------------------------
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

BACKEND = os.environ.get("ANCHOR_BACKEND", "ollama")
MODEL = os.environ.get("ANCHOR_MODEL", "qwen2.5:7b")
TEMPERATURE = float(os.environ.get("ANCHOR_TEMP", "0.3"))
N_REPEATS = int(os.environ.get("ANCHOR_REPEATS", "5"))

# --------------------------------------------------------------- vignettes
# Clinical facts only. No word describes speed, trajectory, or prognosis.
# `prognosis` is our held-out label, never shown to the model.
VIGNETTES = {
    "v1": dict(prognosis="fast", text=(
        "29-year-old. First lifetime depressive episode, onset four weeks ago "
        "following a job relocation. PHQ-9 = 10. No psychiatric history, no "
        "comorbid conditions, no substance use. Employed full-time throughout. "
        "Lives with a partner; two siblings in the same city. Attended the "
        "first appointment unaccompanied and on time. No prior treatment.")),
    "v2": dict(prognosis="fast", text=(
        "24-year-old graduate student. Single episode, onset six weeks ago in "
        "the week following a failed qualifying exam. PHQ-9 = 9. No prior "
        "episodes. Trains with a running club four mornings a week. Shares an "
        "apartment with two friends from the program. No medications.")),
    "v3": dict(prognosis="fast", text=(
        "36-year-old. Second episode; the first was eleven years ago and "
        "remitted within three months on an SSRI, discontinued without "
        "incident. Current onset five weeks ago after a bereavement. "
        "PHQ-9 = 13. No comorbidity. Married, two children, extended family "
        "nearby. Working reduced hours by arrangement with the employer.")),
    "v4": dict(prognosis="slow", text=(
        "54-year-old. Third episode across fifteen years. Current episode "
        "began at least nine months ago; the patient cannot identify a "
        "precipitant. PHQ-9 = 19. Comorbid generalized anxiety. Lives alone; "
        "contact with one adult child several times a year. Two prior "
        "medication trials, both discontinued by the patient within six "
        "weeks. Currently on medical leave.")),
    "v5": dict(prognosis="slow", text=(
        "61-year-old, widowed four years. Persistent low mood documented "
        "over two decades with a superimposed current episode of at least a "
        "year. PHQ-9 = 21. Comorbid insomnia and chronic lower back pain. "
        "Lives alone, no regular contact outside a weekly grocery delivery. "
        "Three prior trials: two discontinued for side effects, one for lack "
        "of response after adequate duration.")),
    "v6": dict(prognosis="slow", text=(
        "47-year-old. Fourth episode; episodes have occurred roughly every "
        "three years since the mid-twenties, each lasting eight months or "
        "longer. Current onset insidious, no identified precipitant, "
        "approximately seven months. PHQ-9 = 18. Comorbid alcohol use, "
        "currently drinking most evenings. Divorced, limited contact with "
        "former colleagues. One prior trial, partial response, self-"
        "discontinued.")),
}

RANGES = {
    "narrow_low": (0.1, 0.5),    # mid 0.30
    "mid":        (0.1, 2.0),    # mid 1.05
    "wide_high":  (0.5, 5.0),    # mid 2.75
}
FIXED_WN = (0.1, 2.0)
FIXED_ZETA = (0.05, 2.0)

PROMPT = """You are estimating individualized parameters of a second-order \
psychological dynamics model for a patient, based on their clinical \
description.

Patient: {text}

Estimate two parameters:
- natural frequency (omega_n): governs how fast this patient's symptoms would \
naturally recover, within range [{wn_lo}, {wn_hi}]
- damping ratio (zeta): governs how oscillatory vs smooth the recovery \
trajectory would be, within range [{z_lo}, {z_hi}]

Respond with ONLY a JSON object: {{"omega_n": <number>, "zeta": <number>, \
"reasoning": "<one sentence>"}}
"""


# ----------------------------------------------------------------- backends
def _post(url, payload, headers, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_model(prompt):
    if BACKEND == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        out = _post(f"{host}/api/generate",
                    {"model": MODEL, "prompt": prompt, "stream": False,
                     "options": {"temperature": TEMPERATURE}},
                    {"Content-Type": "application/json"})
        return out.get("response", "")
    if BACKEND == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        out = _post(f"{base}/chat/completions",
                    {"model": MODEL,
                     "messages": [{"role": "user", "content": prompt}],
                     "temperature": TEMPERATURE},
                    {"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return out["choices"][0]["message"]["content"]
    if BACKEND == "anthropic":
        out = _post("https://api.anthropic.com/v1/messages",
                    {"model": MODEL, "max_tokens": 300,
                     "temperature": TEMPERATURE,
                     "messages": [{"role": "user", "content": prompt}]},
                    {"Content-Type": "application/json",
                     "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01"})
        return out["content"][0]["text"]
    raise ValueError(f"unknown backend {BACKEND}")


def extract(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------- run
def run():
    jobs = []
    for cond in ["A_vary_omega", "B_vary_zeta"]:
        for rname, (lo, hi) in RANGES.items():
            for vname, v in VIGNETTES.items():
                for t in range(N_REPEATS):
                    wn, z = ((lo, hi), FIXED_ZETA) if cond == "A_vary_omega" \
                        else (FIXED_WN, (lo, hi))
                    jobs.append((cond, rname, lo, hi, vname, v, wn, z, t))

    print(f"backend={BACKEND}  model={MODEL}  temp={TEMPERATURE}  "
          f"calls={len(jobs)}\n")
    rows = []
    for i, (cond, rname, lo, hi, vname, v, wn, z, t) in enumerate(jobs):
        p = PROMPT.format(text=v["text"], wn_lo=wn[0], wn_hi=wn[1],
                          z_lo=z[0], z_hi=z[1])
        try:
            raw = call_model(p)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            raw = ""
        d = extract(raw)
        rows.append(dict(
            condition=cond, range_condition=rname, range_lo=lo, range_hi=hi,
            range_mid=(lo + hi) / 2, vignette=vname,
            prognosis=v["prognosis"], trial=t,
            omega_n=d.get("omega_n") if d else None,
            zeta=d.get("zeta") if d else None,
            reasoning=d.get("reasoning") if d else None,
            raw_response=raw, parse_ok=d is not None,
            wn_range=list(wn), zeta_range=list(z), model=MODEL))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(jobs)}")
        if BACKEND != "ollama":
            time.sleep(0.25)

    fn = f"anchoring_v3_{MODEL.replace('/', '_').replace(':', '_')}.json"
    json.dump(rows, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}")
    return rows


# ----------------------------------------------------------------- analysis
def var_by(rows, param, key):
    vals = [r[param] for r in rows if r[param] is not None]
    if len(vals) < 2:
        return float("nan")
    g = np.mean(vals)
    tot = sum((v - g) ** 2 for v in vals)
    if tot == 0:
        return 0.0
    ss = 0.0
    for grp in set(r[key] for r in rows):
        sub = [r[param] for r in rows if r[key] == grp and r[param] is not None]
        if sub:
            ss += len(sub) * (np.mean(sub) - g) ** 2
    return ss / tot


def analyze(rows):
    ok = [r for r in rows if r["parse_ok"]]
    model = rows[0].get("model", "?")
    print(f"\nmodel={model}   parsed {len(ok)}/{len(rows)}")

    for cond, varied, fixed in [("A_vary_omega", "omega_n", "zeta"),
                                ("B_vary_zeta", "zeta", "omega_n")]:
        sub = [r for r in ok if r["condition"] == cond]
        if not sub:
            continue
        print(f"\n{'='*76}\n{cond}:  '{varied}' range VARIED, "
              f"'{fixed}' range FIXED\n{'='*76}")
        for param in [varied, fixed]:
            tag = "VARIED " if param == varied else "control"
            print(f"\n  {param:8s} [{tag}]   var by RANGE = "
                  f"{var_by(sub, param, 'range_condition'):6.1%}   |   "
                  f"var by VIGNETTE = {var_by(sub, param, 'vignette'):6.1%}")
            print(f"    {'':10s}" + "".join(f"{r:>13s}" for r in RANGES))
            for vname, v in VIGNETTES.items():
                cells = []
                for rname in RANGES:
                    x = [r[param] for r in sub if r["vignette"] == vname
                         and r["range_condition"] == rname
                         and r[param] is not None]
                    cells.append(f"{np.mean(x):.2f}±{np.std(x):.2f}"
                                 if x else "   --   ")
                print(f"    {vname+'('+v['prognosis'][0]+')':10s}"
                      + "".join(f"{c:>13s}" for c in cells))
            for rname in RANGES:
                f = [r[param] for r in sub if r["prognosis"] == "fast"
                     and r["range_condition"] == rname and r[param] is not None]
                s = [r[param] for r in sub if r["prognosis"] == "slow"
                     and r["range_condition"] == rname and r[param] is not None]
                if f and s:
                    print(f"      sep(fast-slow) @ {rname:11s} = "
                          f"{np.mean(f)-np.mean(s):+.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        analyze(json.load(open(sys.argv[1])))
    else:
        analyze(run())
