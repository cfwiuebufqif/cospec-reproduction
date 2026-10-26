"""
PhysAnchor analysis: does the stated range beat the physics?

Reads the per-call JSON produced by the collection run and scores it. The
ground truth is recorded per record (true_wn, true_zeta) and was computed
from the system description by the closed-form expressions, not labelled by
us, so "correct" here is a fact about the system rather than a judgement.

THE DECISIVE READING. Each system is asked under three placements of the
stated range:

    enclose   the range contains the true value
    below     the whole range lies BELOW the true value
    above     the whole range lies ABOVE the true value

Under `below` and `above` the range is wrong by construction. A model that
reports the physics must therefore answer OUTSIDE the stated range. A high
in-range fraction under those two placements is the range beating the
physics -- which is the claim the clinical vignettes cannot test, because a
synthetic patient has no true omega_n.

    python physics_analyze.py ../data/physics_qwen2.5_7b.json
    python physics_analyze.py ../data/*.json        # several at once
"""
import json
import sys

import numpy as np

PLACEMENTS = ["enclose", "below", "above"]
# condition 'A' places the range on wn, 'B' places it on zeta
COND = {"A": ("wn", "omega_n", "wn_range", "true_wn"),
        "B": ("zeta", "zeta", "zeta_range", "true_zeta")}


def _rows(path):
    d = json.load(open(path))
    return [r for r in d if r["parse_ok"]], d


def analyze(path):
    ok, all_rows = _rows(path)
    model = all_rows[0].get("model", "?")
    print("\n" + "#" * 78)
    print(f"# {model}    parsed {len(ok)}/{len(all_rows)}")
    print("#" * 78)

    summary = {}
    for cond, (manip_tag, ans_key, rng_key, true_key) in COND.items():
        sub = [r for r in ok if r["condition"] == cond
               and r.get(ans_key) is not None and r[true_key]]
        if not sub:
            continue
        print(f"\ncondition {cond}: the stated range is placed on "
              f"{manip_tag}; the answer scored is '{ans_key}'")
        print(f"  {'placement':9s} {'in stated range':>16s} "
              f"{'med |rel err|':>14s} {'med ans/true':>13s} {'n':>5s}")
        for pl in PLACEMENTS:
            s = [r for r in sub if r["placement"] == pl]
            if not s:
                continue
            lo_hi = [r[rng_key] for r in s]
            ans = np.array([r[ans_key] for r in s], dtype=float)
            true = np.array([r[true_key] for r in s], dtype=float)
            inside = np.mean([lo <= a <= hi
                              for (lo, hi), a in zip(lo_hi, ans)])
            rel = np.abs(ans - true) / true
            print(f"  {pl:9s} {100*inside:15.1f}% {np.median(rel):14.2f} "
                  f"{np.median(ans/true):13.2f} {len(s):5d}")
            summary[(cond, pl)] = dict(in_range=100 * float(inside),
                                       med_rel=float(np.median(rel)),
                                       med_ratio=float(np.median(ans / true)),
                                       n=len(s))
        print("    under 'below'/'above' the range EXCLUDES the true value:")
        print("    a model reporting the physics must answer outside it, so a")
        print("    high in-range % there is the range beating the physics.")

    # accuracy when the range is honest
    print(f"\n  accuracy under 'enclose' (the range is not misleading)")
    for cond, (manip_tag, ans_key, rng_key, true_key) in COND.items():
        s = [r for r in ok if r["condition"] == cond
             and r["placement"] == "enclose"
             and r.get(ans_key) is not None and r[true_key]]
        if not s:
            continue
        ans = np.array([r[ans_key] for r in s], dtype=float)
        true = np.array([r[true_key] for r in s], dtype=float)
        rel = np.abs(ans - true) / true
        print(f"    {ans_key:8s} median |rel err| {np.median(rel):6.2f}   "
              f"within 10% {100*np.mean(rel < .10):5.1f}%   "
              f"within 25% {100*np.mean(rel < .25):5.1f}%   n={len(s)}")

    # by physical domain, on the decisive placements
    print(f"\n  in-range % under the two WRONG placements, by domain")
    print(f"    {'domain':12s} {'cond A (wn)':>12s} {'cond B (zeta)':>14s}")
    for dom in sorted({r["domain"] for r in ok}):
        cells = []
        for cond, (_, ans_key, rng_key, true_key) in COND.items():
            s = [r for r in ok if r["domain"] == dom
                 and r["condition"] == cond
                 and r["placement"] in ("below", "above")
                 and r.get(ans_key) is not None]
            if not s:
                cells.append(float("nan")); continue
            cells.append(100 * np.mean(
                [r[rng_key][0] <= r[ans_key] <= r[rng_key][1] for r in s]))
        print(f"    {dom:12s} {cells[0]:11.1f}% {cells[1]:13.1f}%")

    # headline: pooled over both wrong placements and both conditions
    s = [r for r in ok if r["placement"] in ("below", "above")]
    hits = []
    for r in s:
        _, ans_key, rng_key, _ = COND[r["condition"]]
        if r.get(ans_key) is None:
            continue
        lo, hi = r[rng_key]
        hits.append(lo <= r[ans_key] <= hi)
    print(f"\n  HEADLINE  answers falling inside a range that is wrong by "
          f"construction: {100*np.mean(hits):.1f}%  ({sum(hits)}/{len(hits)})")
    return summary


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
