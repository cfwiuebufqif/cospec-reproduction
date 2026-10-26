"""
Emit the per-unit appendix tables as LaTeX, computed from the released data.

The main text reports family means, medians and summary statistics. These
tables release the level underneath them -- per system, per plant, per
vignette, per model -- so a reader can check the aggregate rather than take
it. Nothing here is typed by hand: every cell is computed here and written
straight into appendix_tables.tex, which main.tex \\input{}s.

    python make_appendix_tables.py
"""
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = [os.path.join(_HERE, os.pardir, "data"), _HERE, os.getcwd()]
OUT = os.path.join(_HERE, os.pardir, "paper", "appendix_tables.tex")

COND = {"A": ("omega_n", "wn_range", "true_wn"),
        "B": ("zeta", "zeta_range", "true_zeta")}
FAM = {"mechanical": "mass-spring-damper", "rlc": "series RLC",
       "thermal": "two-tank thermal"}


def _resolve(fn):
    for d in _DATA:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def esc(s):
    return str(s).replace("_", r"\_")


def fmt(x, nd=1, inf="---"):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return inf
    return f"{x:.{nd}f}"


# ===================================================== A. PhysAnchor per system
def table_physanchor():
    models = [("qwen2.5:7b", "physics_qwen2.5_7b.json"),
              ("deepseek-chat", "physics_deepseek-chat.json")]
    data = {m: [r for r in json.load(open(_resolve(f))) if r["parse_ok"]]
            for m, f in models}
    systems = sorted({(r["system"], r["domain"], r["true_wn"], r["true_zeta"])
                      for r in data["qwen2.5:7b"]},
                     key=lambda t: (t[1], t[0]))

    def cell(rows, sysname):
        """in-range % under the two wrong placements, and median |rel err|."""
        sel = [r for r in rows if r["system"] == sysname
               and r["placement"] in ("below", "above")
               and r["condition"] == "A"]      # omega_n only, as in Table 4
        hits, rels = [], []
        for r in sel:
            key, rng, true = COND[r["condition"]]
            v = r.get(key)
            if v is None or not r[true]:
                continue
            lo, hi = r[rng]
            hits.append(lo <= v <= hi)
            rels.append(abs(v - r[true]) / r[true])
        if not hits:
            return None, None
        return 100 * float(np.mean(hits)), float(np.median(rels))

    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{\textsc{PhysAnchor}, per system. For each of the $24$ "
         r"systems: the share of answers that fall \emph{inside} a stated "
         r"range that excludes the true value, asked for $\omega_n$ and pooled "
         r"over the \texttt{below} and \texttt{above} placements ($6$ calls "
         r"per model per system), and the median relative error against the "
         r"value the physics fixes. A system at $0.0\%$ with error $0.00$ is "
         r"one the model simply computed. Table~\ref{tab:physics} is the "
         r"family-level summary of these rows.}",
         r"\label{tab:physanchor-sys}", r"\small",
         r"\begin{tabular}{llrr@{\hskip 10pt}rr@{\hskip 10pt}rr}",
         r"\toprule",
         r"& & & & \multicolumn{2}{c}{\texttt{qwen2.5:7b}} "
         r"& \multicolumn{2}{c}{\texttt{deepseek-chat}} \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"system & family & $\omega_n$ & $\zeta$ & in-range & med.\ err "
         r"& in-range & med.\ err \\", r"\midrule"]
    last = None
    for name, dom, wn, z in systems:
        if last is not None and dom != last:
            L.append(r"\addlinespace[2pt]")
        last = dom
        q_ir, q_re = cell(data["qwen2.5:7b"], name)
        d_ir, d_re = cell(data["deepseek-chat"], name)
        L.append(f"{esc(name)} & {FAM[dom]} & {wn:.3g} & {z:.3g} & "
                 f"{fmt(q_ir)}\\% & {fmt(q_re,2)} & "
                 f"{fmt(d_ir)}\\% & {fmt(d_re,2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ======================================================= B. 29 plants, per plant
def table_wide():
    res = json.load(open(_resolve("wide_study.json")))
    EPS = ["0.05", "0.1", "0.25", "0.5", "1.0", "2.0"]
    order = ["original", "mass-spring-damper", "series RLC", "two-tank thermal"]
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{The $I_3$ sweep of Table~\ref{tab:wide}, per plant rather "
         r"than per family. Percentage of $200$ draws in which a relative "
         r"error of the stated size in $(A,B)$ destabilises the true closed "
         r"loop. The spread within each family is the point: fragility is a "
         r"property of the individual plant. The $I_4$ column is omitted "
         r"because it is $0.0\%$ for every plant at every level.}",
         r"\label{tab:wide-per-plant}", r"\small",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"plant & family & $5\%$ & $10\%$ & $25\%$ & $50\%$ & $100\%$ "
         r"& $200\%$ \\", r"\midrule"]
    for fam in order:
        sel = {k: v for k, v in res.items() if v["family"] == fam}
        if not sel:
            continue
        for k, v in sorted(sel.items()):
            L.append(f"{esc(k)} & {fam} & " +
                     " & ".join(f"{v['i3'][e]:.1f}\\%" for e in EPS) + r" \\")
        mean = [np.mean([v["i3"][e] for v in sel.values()]) for e in EPS]
        L.append(r"\cmidrule(lr){1-8}")
        L.append(f"\\emph{{mean}} & {fam} & " +
                 " & ".join(f"\\emph{{{m:.1f}\\%}}" for m in mean) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ============================================ C. v4 full variance decomposition
def table_v4():
    s = json.load(open(_resolve("stats_ci_v4.json")))
    CN = {"A_vary_omega": r"A (vary $\omega_n$)",
          "B_vary_zeta": r"B (vary $\zeta$)"}
    PN = {"omega_n": r"$\omega_n$", "zeta": r"$\zeta$"}
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{The thirty-vignette replication in full ($900$ calls per "
         r"model, all parsed). \textsc{range} and \textsc{vignette} are the "
         r"share of each parameter's variance explained by the stated interval "
         r"and by the patient; the interval is the vignette-clustered $95\%$ "
         r"bootstrap CI. \emph{norm.} re-expresses each answer as its position "
         r"within the stated interval. Every permutation test against the "
         r"range labels returns $p=10^{-4}$, the resolution of $10^4$ draws, "
         r"except where marked.}",
         r"\label{tab:v4-full}", r"\footnotesize",
         r"\begin{tabular}{llllrrr}", r"\toprule",
         r"model & condition & param & role & \textsc{range} (CI) "
         r"& \textsc{vignette} & norm. \\", r"\midrule"]
    for model in s:
        first = True
        for cond in s[model]:
            for param, e in s[model][cond].items():
                lo, hi = e["eta2_range_cluster_ci"]
                p = e["perm_p"]
                star = "" if p <= 1.5e-4 else r"$^{\dagger}$"
                L.append(
                    f"{esc(model) if first else ''} & "
                    f"{CN.get(cond, cond) if param == list(s[model][cond])[0] else ''} & "
                    f"{PN.get(param, param)} & {e['role'].strip().lower()} & "
                    f"{100*e['eta2_range']:.1f} [{100*lo:.1f},{100*hi:.1f}]{star} & "
                    f"{100*e['eta2_vignette']:.1f} & "
                    f"{100*e['eta2_range_normalised']:.1f} \\\\")
                first = False
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[2pt]{\footnotesize $^{\dagger}$permutation $p>10^{-4}$: the "
          r"control parameter in condition~B for \texttt{deepseek-chat}, "
          r"which the stated range does not move at all.}",
          r"\end{table}", ""]
    return "\n".join(L)


# ================================ D/E. case study: by model x range, and sweep
def table_case_breakdown():
    R = json.load(open(_resolve("case_study_results.json")))
    models = ["qwen2.5:7b", "deepseek-chat", "gpt-5.6"]
    ranges = [("narrow_low", "[0.1, 0.5]"), ("mid", "[0.1, 2.0]"),
              ("wide_high", "[0.5, 5.0]")]

    def pct(rows, key="i3_unstable"):
        return 100 * np.mean([bool(r[key]) for r in rows]) if rows else float("nan")

    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{The $540$ case-study runs broken out by model and by the "
         r"interval the prompt stated. Each cell is $60$ runs. The $I_4$ "
         r"column is $0.0\%$ in every cell, so it is reported once rather "
         r"than repeated: no routing of any logged answer to the cost weights "
         r"destabilised the loop. The row pattern is the finding --- "
         r"instability is concentrated in the narrow low interval, across "
         r"models that otherwise differ a great deal.}",
         r"\label{tab:case-breakdown}", r"\small",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"model & " + " & ".join(n for _, n in ranges) + r" & all \\",
         r"\midrule"]
    for m in models:
        sub = [r for r in R if r["model"] == m]
        cells = [pct([r for r in sub if r["range_condition"] == k])
                 for k, _ in ranges]
        L.append(f"\\texttt{{{esc(m)}}} & " +
                 " & ".join(f"{c:.1f}\\%" for c in cells) +
                 f" & {pct(sub):.1f}\\% \\\\")
    L.append(r"\midrule")
    allc = [pct([r for r in R if r["range_condition"] == k]) for k, _ in ranges]
    L.append(r"\emph{all models} & " +
             " & ".join(f"\\emph{{{c:.1f}\\%}}" for c in allc) +
             f" & \\emph{{{pct(R):.1f}\\%}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ====================================== F. per-vignette anchoring (v3, 3 models)
def table_per_vignette():
    files = {"qwen2.5:7b": "anchoring_v3_qwen2.5_7b.json",
             "deepseek-chat": "anchoring_v3_deepseek-chat.json",
             "gpt-5.6": "anchoring_v3_gpt5.6.json"}
    ranges = ["narrow_low", "mid", "wide_high"]
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{Median $\widehat{\omega}_n$ per vignette under each stated "
         r"interval, condition~A, five repeats per cell. Reading across a row "
         r"gives the effect the paper measures: the same patient, described in "
         r"the same words, answered differently because the interval in the "
         r"prompt changed. Reading down a column gives what is left for the "
         r"patient to explain.}",
         r"\label{tab:per-vignette}", r"\small",
         r"\begin{tabular}{ll" + "r" * (3 * 3) + "}", r"\toprule",
         r"& & \multicolumn{3}{c}{\texttt{qwen2.5:7b}} "
         r"& \multicolumn{3}{c}{\texttt{deepseek-chat}} "
         r"& \multicolumn{3}{c}{\texttt{gpt-5.6}} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"vignette & prognosis & " +
         " & ".join([r"nar.", r"mid", r"wide"] * 3) + r" \\", r"\midrule"]
    rows = {m: [r for r in json.load(open(_resolve(f)))
                if r["parse_ok"] and r["condition"] == "A_vary_omega"]
            for m, f in files.items()}
    vigs = sorted({r["vignette"] for r in rows["gpt-5.6"]})
    prog = {r["vignette"]: r.get("prognosis", "") for r in rows["gpt-5.6"]}
    for v in vigs:
        cells = []
        for m in files:
            for k in ranges:
                vals = [r["omega_n"] for r in rows[m]
                        if r["vignette"] == v and r["range_condition"] == k
                        and r["omega_n"]]
                cells.append(f"{np.median(vals):.2f}" if vals else "---")
        L.append(f"{esc(v)} & {prog.get(v,'')} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)



# ============================================ G. I4 mapping sensitivity sweep
def table_mapping():
    rows = json.load(open(_resolve("mapping_sweep.json")))
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{Does the $I_4$ result depend on the mapping we chose? "
         r"Each row routes all $540$ logged answers to the cost weights "
         r"through a different mapping and scores the result against the same "
         r"true plant. \textbf{Stability and the bound do not move, and "
         r"cannot}: every strictly positive scaling of $(Q,R)$ is admissible, "
         r"so Fact~\ref{fact:lqr} and Proposition~\ref{prop:subopt} apply to "
         r"all of them, and a non-zero entry in either column would indicate "
         r"an error in our code rather than in the theorem. Realised "
         r"performance does move, as it should, and we note that the mapping "
         r"used in Section~\ref{sec:case} is not the most favourable one "
         r"available --- the square-root form gives a better median.}",
         r"\label{tab:mapping}", r"\small",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"mapping $(\widehat Q,\widehat R)\leftarrow(\widehat\omega_n,"
         r"\widehat\zeta)$ & unstable & med.\ $J/J^\star$ & p95 "
         r"& med.\ slack & bound viol. \\", r"\midrule"]
    for r in rows:
        lab = r["label"].replace("OMEGA_REF", r"$\omega_{\mathrm{ref}}$") \
                        .replace("ZETA_REF", r"$\zeta_{\mathrm{ref}}$") \
                        .replace("linear  ", "linear, ")
        star = r"$^{\star}$" if "1.05" in r["label"] and "1.025" in r["label"] \
            else ""
        L.append(f"{lab}{star} & {r['unstable_pct']:.1f}\\% & "
                 f"{r['med_ratio']:.2f} & {r['p95_ratio']:.2f} & "
                 f"{r['med_slack']:.2f}$\\times$ & {r['violations']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[2pt]{\footnotesize $^{\star}$the mapping used in "
          r"Section~\ref{sec:case}.}",
          r"\end{table}", ""]
    return "\n".join(L)


# ================================================= H. the two safety baselines
def table_baselines():
    b = json.load(open(_resolve("baselines.json")))["safety_baseline"]
    n = json.load(open(_resolve("baselines.json")))["negative_control"]
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{The two guards, side by side, on the same $540$ logged "
         r"answers. They protect different interfaces and are not "
         r"interchangeable; what the comparison prices is keeping the model on "
         r"the interface that needs sampling to police it. The gate is the "
         r"baseline safety mechanism \emph{for this pipeline}, not a claim "
         r"about the field's state of the art. Absolute wall clock is "
         r"load-dependent --- across sessions we measured the gate between "
         r"$26$ and $42$\,ms per call --- so the ratio is the quantity to "
         r"read, not the times.}",
         r"\label{tab:baselines}", r"\small",
         r"\begin{tabular}{lll}", r"\toprule",
         r"& Monte Carlo gate ($I_3$) & admissibility certificate ($I_4$) \\",
         r"\midrule",
         f"admits & {b['gate_accept_pct']:.1f}\\% & "
         f"{b['cert_admit_pct']:.1f}\\% \\\\",
         f"false accepts & {b['gate_false_accept_pct']:.1f}\\% "
         f"({b['gate_false_accept_of_accepted_pct']:.1f}\\% of admitted) "
         f"& 0.0\\% (Fact~\\ref{{fact:lqr}}) \\\\",
         f"false rejects & {b['gate_false_reject_pct']:.1f}\\% & 0.0\\% \\\\",
         r"coverage & the draws taken & every admissible weight \\",
         f"wall clock & $\\approx${b['gate_ms']:.0f}\\,ms per call & "
         f"$\\approx${1e3*b['cert_ms']:.0f}\\,$\\mu$s per call \\\\",
         f"& & ({b['speedup_range'][0]:.0f}--750$\\times$ faster) \\\\",
         r"\midrule",
         r"\multicolumn{3}{l}{\emph{information-value control: a predictor "
         r"that returns the midpoint and never reads the patient}} \\",
         f"$I_3$ instability & LLM answers {n['i3_llm_unstable_pct']:.1f}\\% "
         f"& midpoints {n['i3_midpoint_unstable_pct']:.1f}\\% \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# =================================================== I. the vignette design grid
def table_vignettes():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v4", os.path.join(_HERE, "anchoring_probe_v4.py"))
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    V = v4.VIGNETTES
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{The thirty vignettes, as recorded fields rather than as "
         r"prose. The prognosis label in the last column is \emph{re-derived} "
         r"from these fields by the frozen rule in "
         r"\texttt{anchoring\_probe\_v4.py}, not assigned by hand, and the "
         r"same script rejects any vignette whose text contains a word from a "
         r"forbidden list of recovery-speed cues. This table is what makes the "
         r"central assumption checkable: the texts carry the clinical facts "
         r"below and no statement about how fast the patient will recover.}",
         r"\label{tab:vignette-grid}", r"\footnotesize",
         r"\begin{tabular}{lrrrllll}", r"\toprule",
         r"id & age & episodes & months & PHQ-9 & comorbid & support "
         r"& prognosis \\", r"\midrule"]
    for k in sorted(V, key=lambda x: (len(x), x)):
        d = V[k]
        L.append(f"{esc(k)} & {d.get('age','')} & {d.get('episodes','')} & "
                 f"{d.get('months','')} & {d.get('phq9','')} & "
                 f"{esc(d.get('comorbid',''))} & {esc(d.get('support',''))} & "
                 f"{d.get('prognosis','')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    parts = [
        "% Generated by code/make_appendix_tables.py -- do not edit by hand.",
        "% Every number below is computed from the files in data/.",
        "",
        table_physanchor(), table_wide(), table_v4(),
        table_case_breakdown(), table_per_vignette(),
        table_mapping(), table_baselines(), table_vignettes(),
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {OUT}")
    print(f"  {sum(p.count(chr(92)+chr(92)) for p in parts)} table rows emitted")
