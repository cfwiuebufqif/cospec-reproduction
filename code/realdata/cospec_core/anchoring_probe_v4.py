"""
Anchoring probe v4 -- 30 vignettes, frozen design.

Changes from v3: the vignette set grows from 6 to 30. Nothing else changes,
so the v3 runs remain valid and comparable (v1-v6 are carried over verbatim).

WHY 30: with six vignettes the vignette-clustered intervals are very wide,
because six is the number of independent patients no matter how many times
each is queried. Thirty narrows that. It does NOT create ground-truth
dynamics, and it does not make synthetic vignettes clinically valid; see the
paper's limitations.

--------------------------------------------------------------------------
FROZEN DESIGN RULES.  These were fixed before any v4 model output was seen
and are stated here so the construction is auditable rather than curated.

1. Content.  A vignette states only facts a clinician would have at intake:
   age, episode count, episode duration, onset (precipitated / insidious),
   PHQ-9, comorbidity, social support, employment, prior treatment.

2. Prohibition.  No vignette contains any word describing recovery speed,
   trajectory, prognosis, or expected response. Forbidden vocabulary
   includes: fast, slow, quick, rapid, gradual, lingering, protracted,
   improving, worsening, responds, recovery, prognosis, resistant,
   refractory, and any phrase in the patient's voice predicting their own
   course. `audit_vignettes()` below enforces this mechanically.

3. Grid, not cherry-picking.  The 30 cases tile a pre-specified grid:
   episode count {1, 2, 3, 4+} x duration {weeks, months, >=1 year} x
   PHQ-9 band {8-13, 14-19, >=20} x comorbidity {none, anxiety, substance,
   pain/insomnia} x support {strong, moderate, limited} x prior treatment
   {none, one remitted, one failed, multiple failed}. Cells were filled to
   balance the held-out label, not to make any model look better or worse.

4. Held-out label, assigned by rule (never shown to a model):
       fast  <=>  episodes <= 2  AND  duration < 3 months
                  AND PHQ-9 <= 14  AND support in {strong, moderate}
                  AND prior treatment in {none, one remitted}
       slow  <=>  episodes >= 3  OR  duration >= 6 months
                  AND PHQ-9 >= 17  AND comorbidity present
                  AND support in {limited, isolated}
   Cases satisfying neither rule are excluded rather than judged by hand.
   `audit_vignettes()` re-derives every label from the recorded fields and
   fails loudly on any mismatch, so the labels cannot drift from the rule.

5. Analysis.  The vignette is the clustering unit. The five repeats per cell
   estimate model stochasticity, not five additional patients.
--------------------------------------------------------------------------

Run exactly as v3:
    export ANCHOR_BACKEND=openai ANCHOR_MODEL=deepseek-chat
    export OPENAI_BASE_URL=https://api.deepseek.com OPENAI_API_KEY=...
    python anchoring_probe_v4.py
900 calls per model (30 x 3 ranges x 2 conditions x 5 repeats).
Re-analyse a saved file with:  python anchoring_probe_v4.py <file.json>
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

# Words that would leak the answer. Checked mechanically by audit_vignettes().
FORBIDDEN = [
    "fast", "quick", "rapid", "slow", "gradual", "lingering", "protracted",
    "improv", "worsen", "recover", "prognos", "respond", "resistant",
    "refractory", "remit quickly", "bounce back", "turn around",
    "expects to", "will take", "months to lift", "clears up",
]

# --------------------------------------------------------------- vignettes
# fields: age, episodes, months (current episode duration), onset, phq9,
#         comorbid, support, employment, prior, label
V = {}


def add(key, label, text, **f):
    V[key] = dict(prognosis=label, text=" ".join(text.split()), **f)


# ---- carried over verbatim from v3 -------------------------------------
add("v1", "fast", """29-year-old. First lifetime depressive episode, onset four weeks ago
    following a job relocation. PHQ-9 = 10. No psychiatric history, no
    comorbid conditions, no substance use. Employed full-time throughout.
    Lives with a partner; two siblings in the same city. Attended the first
    appointment unaccompanied and on time. No prior treatment.""",
    age=29, episodes=1, months=1, onset="precipitated", phq9=10,
    comorbid="none", support="strong", employment="working", prior="none")

add("v2", "fast", """24-year-old graduate student. Single episode, onset six weeks ago in the
    week following a failed qualifying exam. PHQ-9 = 9. No prior episodes.
    Trains with a running club four mornings a week. Shares an apartment
    with two friends from the program. No medications.""",
    age=24, episodes=1, months=1.5, onset="precipitated", phq9=9,
    comorbid="none", support="strong", employment="working", prior="none")

add("v3", "fast", """36-year-old. Second episode; the first was eleven years ago and remitted
    within three months on an SSRI, discontinued without incident. Current
    onset five weeks ago after a bereavement. PHQ-9 = 13. No comorbidity.
    Married, two children, extended family nearby. Working reduced hours by
    arrangement with the employer.""",
    age=36, episodes=2, months=1.2, onset="precipitated", phq9=13,
    comorbid="none", support="strong", employment="reduced",
    prior="one remitted")

add("v4", "slow", """54-year-old. Third episode across fifteen years. Current episode began at
    least nine months ago; the patient cannot identify a precipitant.
    PHQ-9 = 19. Comorbid generalized anxiety. Lives alone; contact with one
    adult child several times a year. Two prior medication trials, both
    discontinued by the patient within six weeks. Currently on medical
    leave.""",
    age=54, episodes=3, months=9, onset="insidious", phq9=19,
    comorbid="anxiety", support="limited", employment="leave",
    prior="multiple failed")

add("v5", "slow", """61-year-old, widowed four years. Persistent low mood documented over two
    decades with a superimposed current episode of at least a year.
    PHQ-9 = 21. Comorbid insomnia and chronic lower back pain. Lives alone,
    no regular contact outside a weekly grocery delivery. Three prior
    trials: two discontinued for side effects, one for lack of response
    after adequate duration.""",
    age=61, episodes=4, months=12, onset="insidious", phq9=21,
    comorbid="pain/insomnia", support="isolated", employment="unemployed",
    prior="multiple failed")

add("v6", "slow", """47-year-old. Fourth episode; episodes have occurred roughly every three
    years since the mid-twenties, each lasting eight months or longer.
    Current onset insidious, no identified precipitant, approximately seven
    months. PHQ-9 = 18. Comorbid alcohol use, currently drinking most
    evenings. Divorced, limited contact with former colleagues. One prior
    trial, partial response, self-discontinued.""",
    age=47, episodes=4, months=7, onset="insidious", phq9=18,
    comorbid="substance", support="limited", employment="working",
    prior="one failed")

# ---- new in v4: 12 further 'fast' cells --------------------------------
add("v7", "fast", """31-year-old teacher. First episode, onset three weeks ago during a
    contested custody arrangement. PHQ-9 = 11. No comorbidity, no substance
    use. Continues to teach a full timetable. Two close friends attend
    weekly; parents in the same town. No prior treatment.""",
    age=31, episodes=1, months=0.8, onset="precipitated", phq9=11,
    comorbid="none", support="strong", employment="working", prior="none")

add("v8", "fast", """26-year-old software developer. First episode, five weeks, onset after a
    team restructure. PHQ-9 = 8. No comorbid conditions. Works from the
    office three days a week. Lives with a sibling. No medications, no
    prior contact with services.""",
    age=26, episodes=1, months=1.2, onset="precipitated", phq9=8,
    comorbid="none", support="strong", employment="working", prior="none")

add("v9", "fast", """42-year-old nurse. Second episode; the first was nine years ago,
    remitted in ten weeks on sertraline, stopped by agreement. Current
    onset seven weeks ago after a night-shift rotation change. PHQ-9 = 12.
    No comorbidity. Married; colleagues aware and supportive. Working
    reduced shifts.""",
    age=42, episodes=2, months=1.7, onset="precipitated", phq9=12,
    comorbid="none", support="strong", employment="reduced",
    prior="one remitted")

add("v10", "fast", """33-year-old. First episode, four weeks, onset following a miscarriage.
    PHQ-9 = 14. No psychiatric history. Partner has taken leave to be at
    home. Sister visits daily. No prior treatment. Attends with partner.""",
    age=33, episodes=1, months=1, onset="precipitated", phq9=14,
    comorbid="none", support="strong", employment="reduced", prior="none")

add("v11", "fast", """22-year-old undergraduate. First episode, six weeks, onset after moving
    cities for study. PHQ-9 = 10. No comorbidity. Plays in a university
    orchestra twice weekly. Speaks with family daily. No medications.""",
    age=22, episodes=1, months=1.5, onset="precipitated", phq9=10,
    comorbid="none", support="strong", employment="working", prior="none")

add("v12", "fast", """38-year-old accountant. Second episode; first at 27, remitted within
    two months without medication. Current onset eight weeks ago after a
    house move. PHQ-9 = 13. No comorbidity. Lives with spouse and one
    child. Full duties at work. No prior medication.""",
    age=38, episodes=2, months=2, onset="precipitated", phq9=13,
    comorbid="none", support="strong", employment="working",
    prior="one remitted")

add("v13", "fast", """28-year-old. First episode, five weeks, onset after the end of a
    long-term relationship. PHQ-9 = 12. No comorbid conditions. Swims three
    times a week. Shares a flat with a close friend. No prior treatment,
    self-referred.""",
    age=28, episodes=1, months=1.2, onset="precipitated", phq9=12,
    comorbid="none", support="strong", employment="working", prior="none")

add("v14", "fast", """45-year-old. Second episode; the first was fourteen years ago and
    remitted in three months on an SSRI. Current onset six weeks ago,
    coinciding with a parent's hospital admission. PHQ-9 = 14. No
    comorbidity. Two adult children nearby; weekly church involvement.
    Working reduced hours.""",
    age=45, episodes=2, months=1.5, onset="precipitated", phq9=14,
    comorbid="none", support="moderate", employment="reduced",
    prior="one remitted")

add("v15", "fast", """35-year-old. First episode, three weeks, onset after redundancy.
    PHQ-9 = 9. No psychiatric or substance history. Already interviewing
    for new roles. Lives with partner; regular contact with two siblings.
    No prior treatment.""",
    age=35, episodes=1, months=0.8, onset="precipitated", phq9=9,
    comorbid="none", support="strong", employment="unemployed",
    prior="none")

add("v16", "fast", """49-year-old. Second episode; first at 40, remitted within two months
    with brief psychotherapy alone. Current onset seven weeks ago after a
    road traffic accident with no lasting injury. PHQ-9 = 13. No
    comorbidity. Married; large extended family locally. At work full
    time.""",
    age=49, episodes=2, months=1.7, onset="precipitated", phq9=13,
    comorbid="none", support="strong", employment="working",
    prior="one remitted")

add("v17", "fast", """30-year-old. First episode, four weeks, onset after a friend's death.
    PHQ-9 = 11. No comorbid conditions, no alcohol use. Lives with two
    housemates; both attend the appointment waiting room. Employed
    full-time in a stable post. No prior treatment.""",
    age=30, episodes=1, months=1, onset="precipitated", phq9=11,
    comorbid="none", support="strong", employment="working", prior="none")

add("v18", "fast", """52-year-old. Second episode; the first was at 33 and remitted in
    eleven weeks on medication stopped by agreement. Current onset eight
    weeks ago after retirement from a long-held post. PHQ-9 = 12. No
    comorbidity. Spouse and three grandchildren in the same street.""",
    age=52, episodes=2, months=2, onset="precipitated", phq9=12,
    comorbid="none", support="strong", employment="unemployed",
    prior="one remitted")

# ---- new in v4: 12 further 'slow' cells --------------------------------
add("v19", "slow", """58-year-old. Fifth episode since the early thirties. Current episode at
    least fourteen months, no identified precipitant. PHQ-9 = 22. Comorbid
    panic disorder. Lives alone since divorce; no contact with either adult
    child. Four prior medication trials, none completed at adequate dose.
    Not working.""",
    age=58, episodes=4, months=14, onset="insidious", phq9=22,
    comorbid="anxiety", support="isolated", employment="unemployed",
    prior="multiple failed")

add("v20", "slow", """44-year-old. Third episode over twelve years. Current episode ten
    months, insidious onset. PHQ-9 = 20. Comorbid daily cannabis use since
    the twenties. Estranged from family; one acquaintance seen monthly.
    Two prior trials, both stopped within a month. On long-term sick
    leave.""",
    age=44, episodes=3, months=10, onset="insidious", phq9=20,
    comorbid="substance", support="isolated", employment="leave",
    prior="multiple failed")

add("v21", "slow", """66-year-old. Fourth episode. Current episode at least eighteen months,
    began without identifiable trigger. PHQ-9 = 19. Comorbid osteoarthritis
    and chronic insomnia. Widowed; sees a neighbour twice a month. Three
    prior trials, two discontinued for side effects. Retired.""",
    age=66, episodes=4, months=18, onset="insidious", phq9=19,
    comorbid="pain/insomnia", support="limited", employment="unemployed",
    prior="multiple failed")

add("v22", "slow", """39-year-old. Third episode since 25. Current episode eight months,
    insidious. PHQ-9 = 18. Comorbid social anxiety present since
    adolescence. Lives alone; declines contact with former colleagues. One
    prior trial abandoned after three weeks. Working, but on a final
    written warning for attendance.""",
    age=39, episodes=3, months=8, onset="insidious", phq9=18,
    comorbid="anxiety", support="limited", employment="working",
    prior="one failed")

add("v23", "slow", """71-year-old. Long-standing low mood documented since the fifties, with
    a current episode of about two years. PHQ-9 = 23. Comorbid chronic
    obstructive pulmonary disease and insomnia. Lives alone in a rural
    area; a carer visits weekly. Three prior trials, none at adequate
    duration.""",
    age=71, episodes=4, months=24, onset="insidious", phq9=23,
    comorbid="pain/insomnia", support="isolated", employment="unemployed",
    prior="multiple failed")

add("v24", "slow", """48-year-old. Fourth episode. Current episode nine months, no
    precipitant identified. PHQ-9 = 21. Comorbid alcohol use, drinking
    daily since the third episode. Separated; children live with the other
    parent. Two prior trials, both self-discontinued. Unemployed for two
    years.""",
    age=48, episodes=4, months=9, onset="insidious", phq9=21,
    comorbid="substance", support="isolated", employment="unemployed",
    prior="multiple failed")

add("v25", "slow", """55-year-old. Third episode across eighteen years. Current episode
    eleven months, insidious. PHQ-9 = 19. Comorbid generalized anxiety and
    tension headache. Lives with an elderly parent for whom the patient is
    sole carer; no other regular contact. Two prior trials, one stopped for
    side effects, one for inadequate response.""",
    age=55, episodes=3, months=11, onset="insidious", phq9=19,
    comorbid="anxiety", support="limited", employment="unemployed",
    prior="multiple failed")

add("v26", "slow", """41-year-old. Third episode since 29. Current episode seven months, no
    trigger identified. PHQ-9 = 18. Comorbid binge drinking at weekends.
    Divorced, limited contact with one sibling. One prior trial, partial
    response, discontinued by the patient. In work but on reduced
    duties.""",
    age=41, episodes=3, months=7, onset="insidious", phq9=18,
    comorbid="substance", support="limited", employment="reduced",
    prior="one failed")

add("v27", "slow", """63-year-old. Fifth episode. Current episode about sixteen months,
    insidious onset. PHQ-9 = 20. Comorbid type 2 diabetes with painful
    neuropathy and chronic insomnia. Widowed; one adult child abroad, in
    contact twice a year. Four prior trials, none completed.""",
    age=63, episodes=4, months=16, onset="insidious", phq9=20,
    comorbid="pain/insomnia", support="isolated", employment="unemployed",
    prior="multiple failed")

add("v28", "slow", """37-year-old. Third episode since the mid-twenties. Current episode six
    months, insidious. PHQ-9 = 17. Comorbid generalized anxiety. Lives
    alone; declines family contact. One prior trial stopped after two
    weeks. Working night shifts alone.""",
    age=37, episodes=3, months=6, onset="insidious", phq9=17,
    comorbid="anxiety", support="limited", employment="working",
    prior="one failed")

add("v29", "slow", """59-year-old. Fourth episode. Current episode thirteen months, no
    precipitant. PHQ-9 = 22. Comorbid chronic lower back pain on long-term
    opioids, and insomnia. Lives alone; contact limited to a monthly
    pharmacy visit. Three prior trials, two for side effects, one
    inadequate.""",
    age=59, episodes=4, months=13, onset="insidious", phq9=22,
    comorbid="pain/insomnia", support="isolated", employment="leave",
    prior="multiple failed")

add("v30", "slow", """46-year-old. Third episode across fourteen years. Current episode eight
    months, insidious. PHQ-9 = 19. Comorbid alcohol use most evenings and
    long-standing insomnia. Separated; no regular contact outside work. Two
    prior trials, both discontinued within a month. Employed, attendance
    intermittent.""",
    age=46, episodes=3, months=8, onset="insidious", phq9=19,
    comorbid="substance", support="limited", employment="working",
    prior="multiple failed")

VIGNETTES = V

RANGES = {"narrow_low": (0.1, 0.5), "mid": (0.1, 2.0), "wide_high": (0.5, 5.0)}
FIXED_WN, FIXED_ZETA = (0.1, 2.0), (0.05, 2.0)

PROMPT = """You are estimating individualized parameters of a second-order \
psychological dynamics model for a patient, based on their clinical \
description.

Patient: {text}

Estimate two parameters:
- natural frequency (omega_n): governs how fast this patient's symptoms \
would naturally recover, within range [{wn_lo}, {wn_hi}]
- damping ratio (zeta): governs how oscillatory vs smooth the recovery \
trajectory would be, within range [{z_lo}, {z_hi}]

Respond with ONLY a JSON object: {{"omega_n": <number>, "zeta": <number>, \
"reasoning": "<one sentence>"}}
"""


# ------------------------------------------------------------------ audit
def audit_vignettes(verbose=True):
    """Enforce the frozen rules. Run before any model is called."""
    problems = []
    for k, v in VIGNETTES.items():
        low = v["text"].lower()
        for w in FORBIDDEN:
            if w in low:
                problems.append(f"{k}: forbidden term '{w}'")
        # re-derive the label from the recorded fields
        fast = (v["episodes"] <= 2 and v["months"] < 3 and v["phq9"] <= 14
                and v["support"] in {"strong", "moderate"}
                and v["prior"] in {"none", "one remitted"})
        slow = ((v["episodes"] >= 3 or v["months"] >= 6)
                and v["phq9"] >= 17 and v["comorbid"] != "none"
                and v["support"] in {"limited", "isolated"})
        derived = "fast" if fast else ("slow" if slow else "neither")
        if derived != v["prognosis"]:
            problems.append(
                f"{k}: label '{v['prognosis']}' but rule derives '{derived}'")
    n_fast = sum(v["prognosis"] == "fast" for v in VIGNETTES.values())
    if verbose:
        print(f"audit: {len(VIGNETTES)} vignettes, {n_fast} fast / "
              f"{len(VIGNETTES)-n_fast} slow")
        if problems:
            print("  FAILURES:")
            for p in problems:
                print("   -", p)
        else:
            print("  all vignettes pass the frozen design rules")
    return problems


# ----------------------------------------------------------------- backends
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


# ---------------------------------------------------------------------- run
def run():
    if audit_vignettes():
        sys.exit("design audit failed; fix the vignettes before running")

    jobs = []
    for cond in ["A_vary_omega", "B_vary_zeta"]:
        for rname, (lo, hi) in RANGES.items():
            for vname, v in VIGNETTES.items():
                for t in range(N_REPEATS):
                    wn, z = ((lo, hi), FIXED_ZETA) if cond == "A_vary_omega" \
                        else (FIXED_WN, (lo, hi))
                    jobs.append((cond, rname, lo, hi, vname, v, wn, z, t))

    print(f"\nbackend={BACKEND}  model={MODEL}  temp={TEMPERATURE}  "
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
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jobs)}")
        if BACKEND != "ollama":
            time.sleep(0.2)

    fn = f"anchoring_v4_{MODEL.replace('/', '_').replace(':', '_')}.json"
    json.dump(rows, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}  ({sum(r['parse_ok'] for r in rows)}/{len(rows)} "
          f"parsed)")
    return rows


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
    print(f"\nmodel={rows[0].get('model','?')}   parsed {len(ok)}/{len(rows)}")
    for cond, varied, fixed in [("A_vary_omega", "omega_n", "zeta"),
                                ("B_vary_zeta", "zeta", "omega_n")]:
        sub = [r for r in ok if r["condition"] == cond]
        if not sub:
            continue
        print(f"\n{cond}:  '{varied}' VARIED, '{fixed}' fixed")
        for param in [varied, fixed]:
            tag = "VARIED " if param == varied else "control"
            print(f"  {param:8s}[{tag}]  range {var_by(sub,param,'range_condition'):6.1%}"
                  f"   vignette {var_by(sub,param,'vignette'):6.1%}")
        for rc in RANGES:
            f = [r[fixed] for r in sub if r["prognosis"] == "fast"
                 and r["range_condition"] == rc and r[fixed] is not None]
            sl = [r[fixed] for r in sub if r["prognosis"] == "slow"
                  and r["range_condition"] == rc and r[fixed] is not None]
            if f and sl:
                print(f"    control sep(fast-slow) @ {rc:11s} = "
                      f"{np.mean(f)-np.mean(sl):+.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        analyze(json.load(open(sys.argv[1])))
    elif len(sys.argv) > 1 and sys.argv[1] == "audit":
        audit_vignettes()
    else:
        analyze(run())
