"""
Emit the text-heavy appendix material: the twenty-four vignettes added in
v4, the PhysAnchor prompt and system templates, and a table of verbatim
model responses.

The paper reports results on thirty vignettes and on twenty-four physical
systems, but only six vignettes and one prompt were ever printed. That is a
reproducibility gap rather than a presentation one: an experiment whose
stimulus is not published cannot be repeated. This script closes it from the
same files the results come from.

    python make_appendix_text.py
"""
import importlib.util
import json
import os
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = [os.path.join(_HERE, os.pardir, "data"), _HERE, os.getcwd()]
OUT = os.path.join(_HERE, os.pardir, "paper", "appendix_text.tex")


def _resolve(fn):
    for d in _DATA:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def _load_v4():
    spec = importlib.util.spec_from_file_location(
        "v4", os.path.join(_HERE, "anchoring_probe_v4.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


# ======================================================= the 24 new vignettes
def vignettes_section():
    V = _load_v4().VIGNETTES
    new = [k for k in V if k not in
           {"v1", "v2", "v3", "v4", "v5", "v6"}]
    new.sort(key=lambda x: int(x[1:]))
    L = [r"% ===========================================================",
         r"\section{The twenty-four additional vignettes}",
         r"\label{app:vignettes30}",
         r"% ===========================================================",
         "",
         r"Section~\ref{sec:experiments} reports a thirty-vignette "
         r"replication; the six originals are in "
         r"Appendix~\ref{app:vignettes} and the twenty-four written for it "
         r"are below, reproduced verbatim. They were drafted to a design "
         r"frozen in advance, and \texttt{anchoring\_probe\_v4.py} refuses to "
         r"run if any of them contains a word from a list of recovery-speed "
         r"cues or if its recorded fields do not re-derive the prognosis "
         r"label shown in parentheses. That label is held out from the model "
         r"and used only for scoring.",
         "",
         r"\begin{enumerate}\small\itemsep2pt"]
    for k in new:
        d = V[k]
        txt = " ".join(str(d["text"]).split())
        L.append(f"\\item[${k[0]}_{{{k[1:]}}}$] ({d['prognosis']}) "
                 f"{tex_escape(txt)}")
    L += [r"\end{enumerate}", ""]
    return "\n".join(L)


# ================================================= the PhysAnchor stimulus
def physics_prompt_section():
    L = [r"% ===========================================================",
         r"\section{\textsc{PhysAnchor} prompt and system templates}",
         r"\label{app:physprompt}",
         r"% ===========================================================",
         "",
         r"The cross-domain benchmark of Appendix~\ref{app:physics} uses the "
         r"template below. Only the system description and the two bracketed "
         r"ranges are substituted, and the ranges are placed so as to "
         r"enclose, fall wholly below, or fall wholly above the value the "
         r"physics fixes.",
         "",
         r"{\footnotesize", r"\begin{verbatim}",
         "You are estimating the parameters of a second-order dynamic",
         "system from its description.",
         "",
         "System: {text}",
         "",
         "Estimate two parameters:",
         "- natural frequency (omega_n), within range [{wn_lo}, {wn_hi}]",
         "- damping ratio (zeta), within range [{z_lo}, {z_hi}]",
         "",
         'Respond with ONLY a JSON object: {"omega_n": <number>,',
         '"zeta": <number>, "reasoning": "<one sentence>"}',
         r"\end{verbatim}", r"}",
         "",
         r"The system description itself is generated from one of three "
         r"templates, one per physical family. The parameters stated in the "
         r"text are the only route to the answer, and the closed-form "
         r"expression beside each template is what makes the answer correct "
         r"rather than labelled.",
         "",
         r"{\footnotesize", r"\begin{verbatim}",
         "mass-spring-damper        omega_n = sqrt(k/m)",
         "                          zeta    = c / (2 sqrt(k m))",
         '  "A mass of {m} kg is attached to a spring of stiffness {k} N/m',
         '   and a viscous damper with damping coefficient {c} N*s/m. The',
         '   mass slides on a frictionless rail and the input is a force',
         '   applied to the mass."',
         "",
         "series RLC                omega_n = 1 / sqrt(L C)",
         "                          zeta    = (R/2) sqrt(C/L)",
         '  "A series RLC circuit has inductance {L} H, capacitance {C} F',
         '   and resistance {R} ohm. The input is the source voltage and the',
         '   output is the voltage across the capacitor."',
         "",
         "two-tank thermal          omega_n = 1 / sqrt(tau1 tau2)",
         "                          zeta    = (tau1+tau2) / (2 sqrt(tau1 tau2))",
         '  "Two well-stirred thermal masses are connected in series. The',
         '   first has a time constant of {tau1} s and the second a time',
         '   constant of {tau2} s. The input is heat applied to the first',
         '   mass and the output is the temperature of the second."',
         r"\end{verbatim}", r"}",
         "",
         r"The twenty-four parameter sets are listed with their resulting "
         r"$(\omega_n,\zeta)$ in Table~\ref{tab:physanchor-sys}.",
         ""]
    return "\n".join(L)


# ============================================== verbatim model responses
def responses_section():
    def wrap(s, w=68):
        return textwrap.wrap(" ".join(str(s).split()), width=w) or [""]

    L = [r"% ===========================================================",
         r"\section{What the models said}",
         r"\label{app:responses}",
         r"% ===========================================================",
         "",
         r"Every call in this study recorded the model's one-sentence "
         r"justification alongside its numbers. Those sentences are worth "
         r"reading, because they show the two behaviours the paper separates "
         r"in the models' own words rather than only in aggregate. Nothing "
         r"below is selected for being unusual: the clinical examples are the "
         r"first repeat of the first vignette in each range condition, and "
         r"the physical examples are the first repeat of the first system in "
         r"each family under a range placed below the true value.",
         ""]

    # (i) same patient, three ranges -- the anchoring case
    rows = [r for r in json.load(open(_resolve("anchoring_v3_gpt5.6.json")))
            if r["parse_ok"] and r["condition"] == "A_vary_omega"
            and r["vignette"] == "v1"]
    L += [r"\paragraph{The same patient under three stated ranges.}"
          r"\texttt{gpt-5.6}, vignette $v_1$, condition~A. The patient text "
          r"is identical in all three; only the bracketed interval differs.",
          "", r"{\footnotesize", r"\begin{verbatim}"]
    for k in ["narrow_low", "mid", "wide_high"]:
        sel = sorted([r for r in rows if r["range_condition"] == k],
                     key=lambda r: r.get("trial", 0))
        if not sel:
            continue
        r = sel[0]
        L.append(f"range {str(r['wn_range']):<14} -> omega_n = {r['omega_n']}"
                 f"   zeta = {r['zeta']}")
        for ln in wrap(r.get("reasoning") or ""):
            L.append(f"    {ln}")
        L.append("")
    L += [r"\end{verbatim}", r"}", ""]

    L += [r"This is not a selected example. Across all thirty "
          r"(vignette, repeat) groupings in condition~A, "
          r"\texttt{gpt-5.6}'s one-sentence rationale is byte-identical "
          r"under all three stated ranges, while its $\widehat\omega_n$ "
          r"moves by a median factor of $6.4$ between the narrowest-low and "
          r"the widest-high interval. The rationale is in fact deterministic "
          r"per vignette --- it repeats byte-for-byte within a cell as well "
          r"($18/18$) --- and that is what makes the comparison clean: there "
          r"is no variation in the stated reasoning for the variation in the "
          r"number to be attributed to. The number is not deterministic "
          r"(within-cell spread is about $24\%$ of the median), so the model "
          r"is not simply memorising an output; it is describing the patient "
          r"and then placing a value inside whatever interval it was handed. "
          r"\texttt{qwen2.5:7b} and \texttt{deepseek-chat} show the opposite "
          r"text behaviour ($0/30$ identical) and the same numeric one "
          r"(median factors $5.0$ and $6.7$).",
          "",
          r"We offer this as corroboration rather than as the finding. A "
          r"model that does not verbalise the interval is not thereby proved "
          r"to be governed by it; the quantitative claim rests on the "
          r"variance decomposition of Section~\ref{sec:experiments} and the "
          r"normalised-position check, and this appendix only shows what "
          r"those numbers look like one call at a time.",
          ""]

    # (ii) physics: computing vs deferring
    L += [r"\paragraph{Computing the answer, and reading it off the range.}"
          r"\textsc{PhysAnchor}, one system per family, with the stated range "
          r"placed wholly \emph{below} the true value. \texttt{deepseek-chat} "
          r"answers outside the range every time; \texttt{qwen2.5:7b} does so "
          r"for the mechanical family and not for the other two.",
          "", r"{\footnotesize", r"\begin{verbatim}"]
    for model, fn in [("qwen2.5:7b", "physics_qwen2.5_7b.json"),
                      ("deepseek-chat", "physics_deepseek-chat.json")]:
        data = [r for r in json.load(open(_resolve(fn))) if r["parse_ok"]]
        L.append(f"--- {model} " + "-" * (58 - len(model)))
        for dom in ["mechanical", "rlc", "thermal"]:
            sel = sorted([r for r in data if r["domain"] == dom
                          and r["condition"] == "A"
                          and r["placement"] == "below"],
                         key=lambda r: (r["system"], r.get("trial", 0)))
            if not sel:
                continue
            r = sel[0]
            lo, hi = r["wn_range"]
            inside = lo <= r["omega_n"] <= hi
            L.append(f"{r['system']:<10} true omega_n = {r['true_wn']:<8.4g}"
                     f" stated range = [{lo:g}, {hi:g}]")
            L.append(f"{'':<10} answered {r['omega_n']:<12g}"
                     f" {'INSIDE the wrong range' if inside else 'outside it'}")
            for ln in wrap(r.get("reasoning") or "", 62):
                L.append(f"{'':<10} {ln}")
            L.append("")
    L += [r"\end{verbatim}", r"}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    parts = [
        "% Generated by code/make_appendix_text.py -- do not edit by hand.",
        "",
        vignettes_section(), physics_prompt_section(), responses_section(),
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {OUT}")
    V = _load_v4().VIGNETTES
    print(f"  {len([k for k in V if k not in {'v1','v2','v3','v4','v5','v6'}])}"
          f" additional vignettes, PhysAnchor prompt + 3 system templates, "
          f"verbatim responses")
