"""
PhysAnchor: a benchmark for whether a stated range overrides the physics.

The clinical study cannot measure accuracy, because a synthetic vignette has
no true (omega_n, zeta). Here it does. Each description states the physical
quantities of a second-order system in words, and the parameters follow from
them by definition:

    mass-spring-damper   omega_n = sqrt(k/m)          zeta = c / (2 sqrt(km))
    series RLC           omega_n = 1/sqrt(LC)         zeta = (R/2) sqrt(C/L)
    two-tank thermal     omega_n = 1/sqrt(t1 t2)      zeta = (t1+t2)/(2 sqrt(t1 t2))

Nothing here is a label we invented: the ground truth is computed from the
text. That is what makes this a benchmark rather than a vignette study --
there is a right answer, the design was frozen before any model was called,
and the systems, prompts, ground truth and per-call outputs are released so
the score is reproducible and comparable across models.

One claim we do NOT make: this is not a community-accepted evaluation set.
It is a benchmark we built and released, used here for the first time. Call
it that and nothing more.

THE DECISIVE MANIPULATION.  We place the stated range in three ways
relative to the true value:

    contains   the true value lies inside the range
    below      the whole range lies below the true value
    above      the whole range lies above the true value

If a model reports the physics, the `below` and `above` conditions should
produce answers outside the stated range -- the range is, after all, wrong.
If the range wins, answers will sit inside it and the physics will be
discarded. The clinical experiment can only show that answers move with the
range; this one shows what they move away from.

Design frozen before any model was called: 24 systems (8 per family)
spanning two decades of omega_n and zeta from clearly under- to clearly
over-damped; 3 range placements; 2 conditions (which parameter's range is
manipulated); 3 repeats. 432 calls per model.

    export ANCHOR_BACKEND=openai ANCHOR_MODEL=deepseek-chat
    export OPENAI_BASE_URL=https://api.deepseek.com OPENAI_API_KEY=...
    python physics_probe.py
    python physics_probe.py physics_<model>.json     # re-analyse
"""
import json
import math
import os
import re
import sys
import time
import urllib.request

import numpy as np

BACKEND = os.environ.get("ANCHOR_BACKEND", "ollama")
MODEL = os.environ.get("ANCHOR_MODEL", "qwen2.5:7b")
TEMPERATURE = float(os.environ.get("ANCHOR_TEMP", "0.3"))
N_REPEATS = int(os.environ.get("ANCHOR_REPEATS", "3"))

SYSTEMS = []


def msd(name, m, k, c):
    SYSTEMS.append(dict(
        name=name, family="mass-spring-damper",
        text=(f"A mass of {m} kg is attached to a spring of stiffness "
              f"{k} N/m and a viscous damper with damping coefficient "
              f"{c} N*s/m. The mass slides on a frictionless rail and the "
              f"input is a force applied to the mass."),
        omega_n=math.sqrt(k / m), zeta=c / (2 * math.sqrt(k * m))))


def rlc(name, L, C, R):
    SYSTEMS.append(dict(
        name=name, family="series RLC",
        text=(f"A series RLC circuit has inductance {L} H, capacitance "
              f"{C} F and resistance {R} ohm. The input is the source "
              f"voltage and the output is the voltage across the "
              f"capacitor."),
        omega_n=1 / math.sqrt(L * C),
        zeta=(R / 2) * math.sqrt(C / L)))


def thermal(name, t1, t2):
    SYSTEMS.append(dict(
        name=name, family="two-tank thermal",
        text=(f"Two well-stirred thermal masses are connected in series. "
              f"The first has a time constant of {t1} s and the second a "
              f"time constant of {t2} s. The input is heat applied to the "
              f"first mass and the output is the temperature of the "
              f"second."),
        omega_n=1 / math.sqrt(t1 * t2),
        zeta=(t1 + t2) / (2 * math.sqrt(t1 * t2))))


# 8 mass-spring-damper: omega_n 0.5 .. 20, zeta 0.05 .. 1.5
msd("m1", 4.0, 1.0, 0.4)        # wn 0.50  z 0.100
msd("m2", 1.0, 1.0, 0.1)        # wn 1.00  z 0.050
msd("m3", 2.0, 8.0, 1.6)        # wn 2.00  z 0.200
msd("m4", 1.0, 16.0, 4.0)       # wn 4.00  z 0.500
msd("m5", 0.5, 32.0, 2.0)       # wn 8.00  z 0.250
msd("m6", 2.0, 200.0, 80.0)     # wn 10.0  z 1.265
msd("m7", 0.25, 25.0, 1.5)      # wn 10.0  z 0.300
msd("m8", 0.1, 40.0, 0.8)       # wn 20.0  z 0.200

# 8 series RLC
rlc("r1", 1.0, 1.0, 0.2)        # wn 1.00  z 0.100
rlc("r2", 0.1, 0.4, 0.5)        # wn 5.00  z 0.500
rlc("r3", 0.01, 1.0, 0.06)      # wn 10.0  z 0.300
rlc("r4", 4.0, 1.0, 0.4)        # wn 0.50  z 0.100
rlc("r5", 0.25, 0.01, 2.0)      # wn 20.0  z 0.200
rlc("r6", 1.0, 0.25, 4.0)       # wn 2.00  z 1.000
rlc("r7", 0.04, 0.25, 0.2)      # wn 10.0  z 0.250
rlc("r8", 2.0, 0.5, 0.4)        # wn 1.00  z 0.100

# 8 two-tank thermal (zeta >= 1 by construction: real distinct poles)
thermal("t1", 1.0, 1.0)         # wn 1.00  z 1.000
thermal("t2", 0.5, 2.0)         # wn 1.00  z 1.250
thermal("t3", 0.1, 10.0)        # wn 1.00  z 5.050
thermal("t4", 0.2, 0.5)         # wn 3.16  z 1.106
thermal("t5", 0.05, 0.2)        # wn 10.0  z 1.250
thermal("t6", 2.0, 8.0)         # wn 0.25  z 1.250
thermal("t7", 0.02, 0.05)       # wn 31.6  z 1.106
thermal("t8", 4.0, 4.0)         # wn 0.25  z 1.000

PLACEMENTS = ["contains", "below", "above"]


def make_range(true_val, placement):
    """A range of fixed relative width, placed relative to the truth."""
    if placement == "contains":
        lo, hi = true_val / 2.0, true_val * 2.0
    elif placement == "below":
        lo, hi = true_val / 20.0, true_val / 5.0
    else:                                   # above
        lo, hi = true_val * 5.0, true_val * 20.0
    return round(lo, 4), round(hi, 4)


PROMPT = """You are estimating the parameters of a second-order dynamic \
system from its description.

System: {text}

Estimate two parameters:
- natural frequency (omega_n), within range [{wn_lo}, {wn_hi}]
- damping ratio (zeta), within range [{z_lo}, {z_hi}]

Respond with ONLY a JSON object: {{"omega_n": <number>, "zeta": <number>, \
"reasoning": "<one sentence>"}}
"""


def _post(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers)
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


def run():
    jobs = []
    for cond, manip in [("A_place_omega", "omega_n"),
                        ("B_place_zeta", "zeta")]:
        for pl in PLACEMENTS:
            for sy in SYSTEMS:
                for t in range(N_REPEATS):
                    if manip == "omega_n":
                        wn_r = make_range(sy["omega_n"], pl)
                        z_r = make_range(sy["zeta"], "contains")
                    else:
                        wn_r = make_range(sy["omega_n"], "contains")
                        z_r = make_range(sy["zeta"], pl)
                    jobs.append((cond, manip, pl, sy, wn_r, z_r, t))

    print(f"backend={BACKEND}  model={MODEL}  calls={len(jobs)}\n")
    rows = []
    for i, (cond, manip, pl, sy, wn_r, z_r, t) in enumerate(jobs):
        p = PROMPT.format(text=sy["text"], wn_lo=wn_r[0], wn_hi=wn_r[1],
                          z_lo=z_r[0], z_hi=z_r[1])
        try:
            raw = call_model(p)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            raw = ""
        d = extract(raw)
        rows.append(dict(
            condition=cond, manipulated=manip, placement=pl,
            system=sy["name"], family=sy["family"],
            true_omega_n=sy["omega_n"], true_zeta=sy["zeta"],
            wn_range=list(wn_r), zeta_range=list(z_r), trial=t,
            omega_n=d.get("omega_n") if d else None,
            zeta=d.get("zeta") if d else None,
            reasoning=d.get("reasoning") if d else None,
            raw_response=raw, parse_ok=d is not None, model=MODEL))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jobs)}")
        if BACKEND != "ollama":
            time.sleep(0.2)

    fn = f"physics_{MODEL.replace('/', '_').replace(':', '_')}.json"
    json.dump(rows, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}  ({sum(r['parse_ok'] for r in rows)}/{len(rows)} "
          f"parsed)")
    return rows


def analyze(rows):
    ok = [r for r in rows if r["parse_ok"]]
    print(f"\nmodel={rows[0].get('model','?')}   parsed {len(ok)}/{len(rows)}")
    print("=" * 76)
    print("Does the stated range beat the physics?")
    print("=" * 76)
    for cond, manip in [("A_place_omega", "omega_n"),
                        ("B_place_zeta", "zeta")]:
        sub = [r for r in ok if r["condition"] == cond]
        if not sub:
            continue
        rng_key = "wn_range" if manip == "omega_n" else "zeta_range"
        true_key = "true_omega_n" if manip == "omega_n" else "true_zeta"
        print(f"\n{cond}   (the range on '{manip}' is the one placed)")
        print(f"  {'placement':10s} {'in-range':>9s} {'med |rel err|':>14s} "
              f"{'med ans/true':>13s}   n")
        for pl in PLACEMENTS:
            s = [r for r in sub if r["placement"] == pl and r[manip]]
            if not s:
                continue
            inside = np.mean([r[rng_key][0] <= r[manip] <= r[rng_key][1]
                              for r in s])
            rel = np.median([abs(r[manip] - r[true_key]) / r[true_key]
                             for r in s])
            ratio = np.median([r[manip] / r[true_key] for r in s])
            print(f"  {pl:10s} {100*inside:8.1f}% {rel:13.2f} "
                  f"{ratio:13.2f}   {len(s)}")
        print("    reading: under 'below'/'above' the range EXCLUDES the true")
        print("    value, so a model reporting the physics must answer")
        print("    outside the range. High in-range % there = the range wins.")
    print("\n" + "=" * 76)
    print("Accuracy when the range is honest ('contains')")
    print("=" * 76)
    for manip, cond in [("omega_n", "A_place_omega"), ("zeta", "B_place_zeta")]:
        s = [r for r in ok if r["condition"] == cond
             and r["placement"] == "contains" and r[manip]]
        if not s:
            continue
        tk = "true_omega_n" if manip == "omega_n" else "true_zeta"
        rel = np.array([abs(r[manip] - r[tk]) / r[tk] for r in s])
        print(f"  {manip:8s} median |rel err| {np.median(rel):.2f}   "
              f"within 10% {100*np.mean(rel<0.10):5.1f}%   "
              f"within 25% {100*np.mean(rel<0.25):5.1f}%   n={len(s)}")
    for fam in sorted({r["family"] for r in ok}):
        s = [r for r in ok if r["family"] == fam
             and r["condition"] == "A_place_omega"
             and r["placement"] == "contains" and r["omega_n"]]
        if s:
            rel = np.median([abs(r["omega_n"] - r["true_omega_n"])
                             / r["true_omega_n"] for r in s])
            print(f"    {fam:22s} median |rel err| on omega_n {rel:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        analyze(json.load(open(sys.argv[1])))
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        print(f"{len(SYSTEMS)} systems")
        for s in SYSTEMS:
            print(f"  {s['name']:4s} {s['family']:22s} "
                  f"omega_n={s['omega_n']:7.3f}  zeta={s['zeta']:6.3f}")
    else:
        analyze(run())
