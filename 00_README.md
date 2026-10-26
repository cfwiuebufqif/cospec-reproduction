# CoSpec — Reproduction Materials

Reproduction materials for *CoSpec: Specify the Cost, Not the System for LLMs in
Linear-Quadratic Control Loops*. The LaTeX sources of the manuscript are shipped
separately and are not duplicated here.

## 1. Environment

- Python 3.10+.
- Dependencies: numpy, scipy, matplotlib. All analysis, simulation and plotting run
  offline and cost nothing.
- Model access is needed only to re-collect LLM outputs: the three commercial models
  go through their own chat APIs, the two local models through a local Ollama server
  (`ollama pull qwen2.5:0.5b`, `ollama pull qwen2.5:7b`, default endpoint
  `http://localhost:11434`).
- Continuous-time models are discretised by exact zero-order hold (scipy `expm`),
  never by a truncated series.
- Every deterministic script ships its own output under `code/results/`, so the
  numbers can be checked against the manuscript directly.

## 2. data/ — raw model outputs (8 files, irreplaceable)

| File | Model | Calls | Used in |
|---|---|---|---|
| anchoring_v3_qwen2.5_7b.json | qwen2.5:7b (local Ollama) | 180 | anchoring study, Section 4, Table 3, Table 14 |
| anchoring_v3_deepseek-chat.json | deepseek-chat | 180 | Section 4, Table 3, Table 14 |
| anchoring_v3_gpt5.6.json | gpt-5.6 | 180 | Section 4 (field `gpt5.6-selfrun` = official API self-collection) |
| anchoring_v4_qwen2.5_7b.json | qwen2.5:7b | 900 | thirty-vignette replication, Table 3, Table 12 |
| anchoring_v4_deepseek-chat.json | deepseek-chat | 900 | Table 3, Table 12 |
| physics_qwen2.5_7b.json | qwen2.5:7b | 432 | PhysAnchor (24 systems × 3 placements), Section 4.6, Figure 6, Table 9 |
| physics_deepseek-chat.json | deepseek-chat | 432 | Section 4.6, Figure 6, Table 9 |
| anchoring_crossmodel_table.csv | — | summary | variance decomposition summary (v2/v3) |

These are the per-call raw outputs, unfiltered and unedited. They are the only part of
the project that cannot be regenerated.

The thirty-vignette replication covers qwen2.5:7b and deepseek-chat; gpt-5.6 has the
180-call six-vignette data only.

## 3. code/ — 31 scripts, 2 frozen design JSON files, 7 intermediate JSON files, results/

**A. Numerical checks of the theory (Section 3)**
- interface_theory_v2.py: error sweeps over the four interfaces, verification of the
  Proposition 3 bound, stress test of the admissibility certificate → Table 6 (I3
  destabilisation rates), Table 7 (cost-side error sweep), fixed seed. The conceptual
  four-interface comparison is Table 1 in the text.
- interface_study_wide.py: the same sweeps over 29 physical systems (2–5 minutes) →
  Table 8, Table 11; writes wide_study.json.

**B. Anchoring study (Section 4)**
- stats_ci.py: eta-squared variance decomposition, 10000 bootstrap CIs, 10000
  permutation tests, normalised-position check → Figure 5, Table 14; writes
  stats_ci.json. With `--v4` it analyses the thirty-vignette replication → Table 3,
  Table 12.
- anchoring_probe_v3.py: prints the per-model cross tables for the v3 data.
  Usage: `python anchoring_probe_v3.py ../data/anchoring_v3_<model>.json`.
- anchoring_probe_v4.py: analyses the v4 data; `audit` mechanically checks the frozen
  thirty-vignette design (banned words); with no argument it re-collects the data
  (requires model access).
- cross_model_v3.py: cross-model correlation (Pearson r and ratio differences of the
  values each model gives for the same vignette).

**C. Case study (Section 5)**
- case_study.py: routes the 540 real LLM outputs through the same pipeline on both
  interfaces, I3 and I4, and measures destabilisation rate, realised cost and Monte
  Carlo gate false accepts → Section 5, Figure 7, Table 13; writes
  case_study_results.json.
- baselines.py: two baselines — a safety-mechanism baseline (Monte Carlo gate; false
  accept/reject, coverage, runtime; Table 16) and an information-value negative
  control (interval-midpoint predictor, which never reads the patient).
- mapping_sweep.py: I4 mapping sweep (17 mappings × 540 answers) → Table 15, showing
  that the conclusion does not depend on the mapping we chose.

**D. PhysAnchor (Section 4.6)**
- physics_analyze.py: scores physics_*.json.
  Usage: `python physics_analyze.py ../data/physics_*.json` → Figure 6, Table 9.
- physanchor_run.py: replays PhysAnchor from physanchor_frozen.json (24 systems,
  144 intervals); use it to add a new model.
- physics_probe.py: early collection script. Its schema and system set differ from the
  released data and it is kept for history only; use physics_analyze.py for analysis.

**E. Direct elicitation and identification robustness (Section 6)**
- cost_probe.py: asks the model for the cost weights (q1, q2, r) directly; collection,
  scoring and analysis in one script, 120 calls per model. Raw archives
  cost_deepseek-chat.json and cost_qwen2.5_7b.json, summary cost_analysis.json →
  Section 6.1, Figure 9.
- temp_sweep.py: temperature × model-scale sweep over two local open-weight models,
  qwen2.5:0.5b and qwen2.5:7b, at temperatures 0 / 0.3 / 0.7 / 1.0, 2 vignettes × 12
  calls per temperature. Select the model with `SWEEP_MODEL=qwen2.5:0.5b` or
  `qwen2.5:7b` (requires a local Ollama server). Result: the cost route destabilises
  nothing at either scale or any of the four temperatures; at temperature 0.7 the 0.5B
  model produced one call (4.2%) that failed the certificate and fell back to a safe
  gain. Temperature and scale move the distribution of what the gate admits; they do
  not break safety → Section 6.2.
- identified_robust.py: robustness when an estimator supplies the plant, including RLS
  convergence and the mismatch threshold; writes identified_robustness.json →
  Section 6.3, Figure 10.

**F. External validation on published benchmarks, and nonlinearity (Section 6.4, Proposition 9)**
- cartpole_nl.py: the correct nonlinear cart-pole dynamics (explicit inertia I,
  fourth-order Runge–Kutta integration, numerical linearisation at the upright
  equilibrium so that the model used for synthesis and the model used for simulation
  agree). Literature parameters M=0.5, m=0.5, b=0.1, l=0.3, I=0.006 (Control
  Tutorials). Shared module for nl_experiment and scan_all_params.
- lit_benchmark.py: mild linear relative-error sweeps (5%–60%, the bulk of the
  measured distribution) on two literature systems (cart-pole, DC motor; parameters
  from Control Tutorials). Both routes stay stable across this range; it is the mild
  baseline. Destabilisation appears only under order-of-magnitude error or
  nonlinearity.
- lit_cross.py: extreme linear error sweeps (0.05×–10×) on the same two systems. At
  cart-pole pendulum mass ×0.05 the plant route gives rho = 1.999 and destabilises
  while the cost route stays at 0.949; the DC motor is open-loop stable and does not
  fall over, but the plant margin degrades monotonically with error (0.984 → 0.990)
  while the cost margin is constant (0.985).
- nl_experiment.py: comparison on the nonlinear plant in four regimes — no saturation,
  saturation at 15 N, saturation at 8 N, and saturation plus dead zone plus Coulomb
  friction. Shows that saturation occasionally clips an over-aggressive gain and lets a
  design survive by accident, while a neighbouring error kills it.
- scan_all_params.py: nonlinear sweep over all parameters (cart mass M, pendulum mass
  m, pendulum length l, inertia I) × direction × saturation limit (none / 15 N / 8 N) →
  Table 4. The cost route stays upright under every condition; the plant route falls
  over under order-of-magnitude error, and mostly still falls over at 8 N saturation
  (M×0.1, M×5, m×0.02, l×0.05, l×10, and others).
- sg2.py: exact destabilisation crossing against the discrete Lyapunov small-gain
  guaranteed margin, swept in both error directions (correcting an earlier version that
  swept the positive direction only). Numerical instance of Proposition 9. For the most
  aggressive weights (sq=50, sr=0.2) the exact crossing is 19.1% against a small-gain
  guarantee of 1.5%; for nominal weights (sq=1, sr=1), 60.0% against 6.6%.

**G. Figure generation**
- fig2_asymmetry.py → Figure 3 (fig2_asymmetry.pdf; needs wide_study.json; panel c
  computes the psychological beta/alpha bound itself).
- fig_crossover.py → Figure 5 (fig4_crossover.pdf; needs stats_ci.json).
- fig_case_study.py → Figure 7 (fig5_case.pdf; needs case_study_results.json).
- fig_trajectory.py → Figure 8 (fig8_trajectory.pdf; one answer routed two ways).
- fig_directcost.py → Figure 9 (fig7_directcost.pdf).
- fig_robust.py → Figure 10 (fig8_robust.pdf).
- fig_asymmetry.py, fig_wide.py: earlier figure scripts (5-plant and 29-plant panels
  kept separate). Their content has been merged into fig2_asymmetry.py; they are not
  referenced any more and are kept for history only.
- make_appendix_tables.py → the eight per-unit appendix tables, Table 10–17. Tables
  6–9 come from interface_theory_v2.py, interface_study_wide.py and physics_analyze.py
  and are written into `appendix_tables.tex` directly; this script does not regenerate
  them.
- make_appendix_text.py → the thirty vignettes, the prompt templates and the verbatim
  replies.

**H. Frozen design JSON (inputs, not outputs)**
- physanchor_frozen.json: the frozen design for the PhysAnchor replay.
- recovered_systems.json: the 24 systems recovered from the data, with ground truth.

**I. Intermediate result JSON (script outputs, shipped so plots can be made without recomputing)**
- stats_ci.json, case_study_results.json, wide_study.json, cost_analysis.json,
  cost_deepseek-chat.json, cost_qwen2.5_7b.json, identified_robustness.json. Delete
  them and rerun the corresponding script to regenerate.

**J. results/ — outputs of the deterministic scripts**
- sg2.txt, lit_benchmark.txt, lit_cross.txt, nl_experiment.txt, scan_all_params.txt:
  the complete stdout of the five deterministic simulation scripts (no model required,
  seconds to minutes), matching Proposition 9, Section 6.4 and Table 4. temp_sweep.py
  needs two models pulled into a local Ollama, so no output is shipped; the script can
  be rerun.

## 4. Recommended order (no model access required)

1. `python interface_theory_v2.py` (Table 6, Table 7)
2. `python interface_study_wide.py` (Table 8, Table 11)
3. `python stats_ci.py` (Figure 5, Table 14)
4. `python stats_ci.py --v4` (Table 3, Table 12)
5. `python case_study.py` (Section 5, Figure 7, Table 13)
6. `python baselines.py` (Table 16)
7. `python mapping_sweep.py` (Table 15)
8. `python cost_probe.py` (Section 6.1, Figure 9; scoring existing data is free,
   re-collection needs model access)
9. `python identified_robust.py` (Section 6.3, Figure 10)
10. `python sg2.py` (exact crossing and small-gain margin for Proposition 9)
11. `python lit_benchmark.py`, `python lit_cross.py` (mild and extreme linear sweeps on
    the literature systems)
12. `python nl_experiment.py`, `python scan_all_params.py` (nonlinear cart-pole with
    saturation, Table 4)
13. `python physics_analyze.py ../data/physics_*.json` (Figure 6, Table 9)
14. `python make_appendix_tables.py` (Table 10–17)
15. `python make_appendix_text.py`
16. Plotting: fig2_asymmetry.py, fig_crossover.py, fig_case_study.py, fig_trajectory.py,
    fig_directcost.py, fig_robust.py. Run the matching analysis script first, or use
    the intermediate JSON files shipped here.
17. With a local Ollama: `$env:SWEEP_MODEL="qwen2.5:0.5b"; python temp_sweep.py`, then
    again with `qwen2.5:7b`.

## 5. Figure index (ten figures in the manuscript)

| Figure | Generated by | Content |
|---|---|---|
| Figure 1 | fig1_overview.tex (TikZ, manuscript sources) | Overview of the two routes and the mechanism (I3 fails, I4 passes) |
| Figure 2 | fig_positioning.pdf (designed by hand, final PDF in the source archive) | Positioning in the interface-safety design space |
| Figure 3 | fig2_asymmetry.py → fig2_asymmetry.pdf | Interface asymmetry over 29 systems |
| Figure 4 | fig_framework.pdf (designed by hand, final PDF in the source archive) | The CoSpec runtime |
| Figure 5 | fig_crossover.py → fig4_crossover.pdf | Crossover signature across three models |
| Figure 6 | fig6_physanchor.tex (TikZ, manuscript sources) | PhysAnchor by family (24 physical systems) |
| Figure 7 | fig_case_study.py → fig5_case.pdf | Case study on 540 model outputs |
| Figure 8 | fig_trajectory.py → fig8_trajectory.pdf | One answer routed two ways |
| Figure 9 | fig_directcost.py → fig7_directcost.pdf | Direct cost elicitation (including temperature × scale) |
| Figure 10 | fig_robust.py → fig8_robust.pdf | Identified-plant robustness and small-gain margin |

The .png files under `code/` are bitmap previews of the corresponding PDFs, provided
only for quick inspection; the manuscript uses the PDFs throughout.

## 6. Notes

- Fact 1 is the classical LQR stability theorem (Anderson & Moore); its use as a design
  principle comes from Zanon & Gros (safe RL-MPC) and is not a theorem of this paper.
  The contribution here is to connect an untrusted LLM to the cost interface, which
  that theorem can protect, rather than to the plant interface, which it cannot.
- The beta/alpha performance bound is not tight in the scalar case; Proposition 8
  formalises this. The common scalar envelope overestimates systematically at order
  Theta(sqrt(t)), and tightening it requires plant or direction structure. It is not an
  empirical oversight.
- The certificate has the same discriminating power as synthesising K and then checking
  rho(A-BK) < 1; the two agree on all 540 real weight sets. What the certificate adds is
  rejection before synthesis, a reason and a performance envelope rather than a single
  number, and a detectability structure that carries over to the terminal pair of
  nonlinear MPC.
- Literature system parameters are taken from the public Control Tutorials. The
  nonlinear dynamics use explicit inertia I, RK4 integration and numerical
  linearisation, so that the linear synthesis model and the nonlinear simulation model
  agree.
- The temperature × model-scale sweep covers two local open-weight models only. The
  three commercial models appear at a single temperature in the text; the temperature
  extension to frontier models is not covered, and this is stated in the Limitations.
- Two earlier scripts are deliberately not shipped: small_gain_check.py (swept the
  positive direction only, wrong margins) and nonlinear_check.py (nonlinear equations
  written from memory, inconsistent with the linearisation). They are superseded by
  sg2.py and cartpole_nl.py.
- All closed-loop results are simulation. No claim of clinical validity is made.

## 7. Replication on real patient text (Section 7, CounselChat)

Replaces the synthetic vignettes of Section 5 with public real patient text. The
control architecture, the certificate, the assumed plant and the Monte Carlo gate are
unchanged.

- **Data**: `data/counselchat/20220401_counsel_chat.csv`, CounselChat (MIT; scraped
  from counselchat.com on 2022-04-01, public and anonymised; Hugging Face
  `nbertagnolli/counsel-chat`). Deduplicated by questionID into 940 unique first-person
  questions across 31 topics. Only questionText is used as input (falling back to
  questionTitle when empty); all therapist-side columns (therapistInfo, answerText,
  upvotes, views and the rest) are stripped.
- **Code**: `code/realdata/`
  - `run_realdata_experiment.py`: main entry point, modes mapped / directcost / both,
    with `--dry-run` to load the data without calling a model.
  - `dataset_loaders.py`: CounselChat deduplication and column stripping (2775 → 940).
  - `llm_backends.py`: three backends (local vLLM OpenAI-compatible endpoint,
    OpenAI-compatible API, mock) with an on-disk cache for resuming; credentials come
    from the environment variables OPENAI_BASE_URL and OPENAI_API_KEY, nothing is
    hardcoded.
  - `cospec_core/`: the same certificate, control and statistics modules as Section 5
    (case_study, anchoring_probe_v4, cost_probe, baselines, stats_ci), reused verbatim
    rather than reimplemented.
  - `requirements.txt`: dependency list.
- **Results (shipped, no recomputation needed)**: `code/realdata/results/`, where `7b_*`
  and `14b_*` are Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct over the full 940 posts:
  summary, per-row records and run logs, corresponding to Table 5 and to Table 18 in the
  appendix.
- **Backends differ, keep them apart**: Section 7 uses transformers with full-precision
  bf16 Qwen2.5-7B/14B-Instruct, while the 540 outputs of Section 5 come from the local
  quantised Ollama build qwen2.5:7b (Q4). The weight precision differs, so the two
  regimes are reported separately and the numbers are not mixed.
- **Reproducing**:
  1. Without a model: `python run_realdata_experiment.py --dataset counselchat
     --dataset-path data/counselchat/20220401_counsel_chat.csv --backend mock --mode
     both --dry-run` (checks the 2775 → 940 load and the therapist-side stripping).
  2. Full run (needs a local vLLM server or an OpenAI-compatible endpoint): set
     OPENAI_BASE_URL and OPENAI_API_KEY, then `--backend openai --model <model name>
     --mode both --repeats 1 --temperature 0.3`.
- **Scope**: the corpus provides patient language and model replies only — no dynamics,
  no cost labels, no treatment outcomes. The dynamics remain an assumed second-order
  plant, so this is a counterfactual closed-loop test against that plant, not a clinical
  trial, and no clinical validity is claimed. CounselChat is public, MIT licensed and
  already anonymised; no user was contacted and no new subject data was collected.
