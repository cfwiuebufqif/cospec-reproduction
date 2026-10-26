"""
PhysAnchor: add one more model to the benchmark, replaying the exact
conditions the first two models saw.

WHY THIS FILE EXISTS, AND WHAT IT GUARANTEES
--------------------------------------------
The qwen2.5:7b and deepseek-chat runs in ../data/ were collected by a
script that is not `physics_probe.py` in this directory (different field
names, and a different set of 24 systems). To add a third model to the same
table, the third run has to see the same systems and the same stated
ranges, or the three columns are not comparable.

This runner reads `physanchor_frozen.json`, which contains:

  * the 24 systems, with their physical parameters RECOVERED from the
    released data and VERIFIED -- each recovered (m,k,c) / (L,C,R) /
    (tau1,tau2) reproduces the ground truth recorded in the data through
    the closed-form expression, to within 0.2%. All 24 verify. Nothing here
    was invented.

  * the 144 (system, condition, placement) cells with the EXACT stated
    ranges taken from the released data. These are not regenerated: the
    released runs are deterministic in the ranges (144/144 cells single
    valued) and both models saw identical ranges in every cell, so the
    numbers here are copied, not recomputed.

PROMPT WORDING
--------------
Reconstructed from the template in `physics_probe.py` rather than recovered:
the released data stores the models' answers, not the text they were shown.
It matches the quantities and units the earlier models quote back in their
own reasoning. The original collection script no longer exists.

USAGE
-----
    export ANCHOR_BACKEND=openai ANCHOR_MODEL=gpt-5.6
    export OPENAI_API_KEY=...
    # for a non-OpenAI endpoint also set OPENAI_BASE_URL
    python physanchor_run.py

432 calls. Writes physics_<model>.json in the schema physics_analyze.py
reads, so scoring the result is:

    python physics_analyze.py physics_<model>.json
"""
import json
import os
import re
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))

BACKEND = os.environ.get("ANCHOR_BACKEND", "openai")
MODEL = os.environ.get("ANCHOR_MODEL", "gpt-5.6")
TEMPERATURE = float(os.environ.get("ANCHOR_TEMP", "0.3"))

# Reconstructed; see the header. Identical to physics_probe.py's template.
PROMPT = """You are estimating the parameters of a second-order dynamic \
system from its description.

System: {text}

Estimate two parameters:
- natural frequency (omega_n), within range [{wn_lo}, {wn_hi}]
- damping ratio (zeta), within range [{z_lo}, {z_hi}]

Respond with ONLY a JSON object: {{"omega_n": <number>, "zeta": <number>, \
"reasoning": "<one sentence>"}}
"""


def load_frozen():
    p = os.path.join(_HERE, "physanchor_frozen.json")
    with open(p) as f:
        return json.load(f)


def call_model(prompt):
    if BACKEND == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        url, payload, hdr = (
            f"{host}/api/generate",
            {"model": MODEL, "prompt": prompt, "stream": False,
             "options": {"temperature": TEMPERATURE}},
            {"Content-Type": "application/json"})
    else:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url, payload, hdr = (
            f"{base}/chat/completions",
            {"model": MODEL,
             "messages": [{"role": "user", "content": prompt}],
             "temperature": TEMPERATURE},
            {"Content-Type": "application/json",
             "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=hdr)
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode())
    if BACKEND == "ollama":
        return body.get("response", "")
    return body["choices"][0]["message"]["content"]


def extract(raw):
    """Pull the JSON object out of the reply; tolerate fences and prose."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    try:
        wn = float(d["omega_n"])
        z = float(d["zeta"])
    except (KeyError, TypeError, ValueError):
        return None
    return dict(omega_n=wn, zeta=z, reasoning=d.get("reasoning"))


def main():
    fz = load_frozen()
    systems, cells, trials = fz["systems"], fz["cells"], fz["trials"]
    total = len(cells) * trials
    print(f"backend={BACKEND}  model={MODEL}  temp={TEMPERATURE}  "
          f"calls={total}")
    print(f"{len(systems)} systems, {len(cells)} cells, {trials} trials each\n")

    rows, t0 = [], time.time()
    for ci, (sysname, cond, placement, wn_r, z_r, manip) in enumerate(cells):
        s = systems[sysname]
        prompt = PROMPT.format(text=s["text"], wn_lo=wn_r[0], wn_hi=wn_r[1],
                               z_lo=z_r[0], z_hi=z_r[1])
        for t in range(trials):
            try:
                raw = call_model(prompt)
            except Exception as e:
                print(f"  [cell {ci} trial {t}] ERROR: {e}")
                raw = ""
            d = extract(raw)
            rows.append(dict(
                condition=cond, placement=placement, manipulated=manip,
                domain=s["domain"], system=sysname, trial=t,
                true_wn=s["true_wn"], true_zeta=s["true_zeta"],
                wn_range=list(wn_r), zeta_range=list(z_r),
                omega_n=d.get("omega_n") if d else None,
                zeta=d.get("zeta") if d else None,
                reasoning=d.get("reasoning") if d else None,
                raw_response=raw, parse_ok=d is not None, model=MODEL))
            if BACKEND != "ollama":
                time.sleep(0.2)
        if (ci + 1) % 20 == 0:
            done = (ci + 1) * trials
            el = time.time() - t0
            print(f"  {done}/{total}   {el:.0f}s elapsed, "
                  f"~{el/done*(total-done):.0f}s left")

    fn = os.path.join(
        _HERE, f"physics_{MODEL.replace('/', '_').replace(':', '_')}.json")
    with open(fn, "w") as f:
        json.dump(rows, f, indent=2)
    ok = sum(r["parse_ok"] for r in rows)
    print(f"\nwrote {fn}   ({ok}/{len(rows)} parsed)")
    if ok < len(rows):
        print("  NOTE: unparsed calls are kept in the file with "
              "parse_ok=false; nothing is dropped or retried.")
    print(f"\nscore it with:\n    python physics_analyze.py {fn}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        fz = load_frozen()
        print(f"{len(fz['systems'])} systems, {len(fz['cells'])} cells, "
              f"{fz['trials']} trials => "
              f"{len(fz['cells'])*fz['trials']} calls")
        for n, s in sorted(fz["systems"].items()):
            print(f"  {n:10s} {s['domain']:10s} "
                  f"omega_n={s['true_wn']:10.4f}  zeta={s['true_zeta']:7.4f}"
                  f"   params={s['params']}")
        sys.exit(0)
    main()
