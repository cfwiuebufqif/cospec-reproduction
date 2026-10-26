# CoSpec real-data adapter

Adapter layer for running the CoSpec experiments on real datasets: it replaces the
synthetic vignettes of the original study with real help-seeking text and reuses,
unchanged, the control law, the admissibility certificate, the suboptimality bound, the
Monte Carlo gate and the statistics functions from the reproduction package. Both
interfaces are exercised — I3 (plant model) and I4 (cost weights) — reproducing the case
study and the direct cost-elicitation experiment.

Target machine is Ubuntu with one RTX 4090; the code itself is cross-platform. All paths
are built with `os.path`, with no hardcoded separators.

---

## 0. Placeholder data and licensing (read first)

- Every number produced by `--backend mock` or `--dataset selftest` is synthetic
  placeholder data. Result files carry `provenance = selftest-mock-placeholder` and the
  raw replies embed a `[MOCK-PLACEHOLDER]` marker. **These numbers must not appear in
  the paper and must not be cited as any empirical finding.**
- Results used in the paper must come from `--backend vllm` (local open-weight model) or
  `--backend openai` (OpenAI-compatible remote endpoint) together with a real dataset;
  `provenance` is then `real-data-vllm:*` or `real-data-openai:*`.
- The self-test artifacts shipped under `results/` (`mapped_*_<timestamp>.json`,
  `directcost_*_<timestamp>.json`, `selftest_console.txt`, `dryrun_*.txt`) exist only to
  check the schema and the load counts, and can be deleted before a real run. The
  `dryrun_*.txt` files record local load statistics only and contain no LLM output.
- `counselbench_eval` is **CC BY-NC-ND 4.0** (non-commercial, no derivatives): internal
  research use only, and any redistribution requires a legal review first. Only the
  question text is used as input; all answer and rating columns are stripped (see 3.1
  and 3.3).
- This layer supplies real unstructured text input only. It does not infer or annotate
  any real treatment outcome, state matrix, cost-weight label or optimal trajectory
  (scope in section 8).

---

## 1. Layout

```
realdata/
├── README.md                     # this file
├── requirements.txt              # numpy / scipy (required); vLLM optional, GPU only
├── run_realdata_experiment.py    # entry point: collect, score both routes, summarise,
│                                 # sensitivity sweep, write results; supports --dry-run
├── dataset_loaders.py            # unified schema {id, text, meta}, loaders, registry
├── llm_backends.py               # vllm / openai / mock backends (stdlib urllib)
├── cospec_core/                  # verbatim copies of the original modules (do not edit)
│   ├── __init__.py
│   ├── case_study.py             # second-order model, LQR, spectral radius, cost,
│   │                             # certificate, bounds, MC gate
│   ├── cost_probe.py             # direct cost prompt, extract(), score()
│   ├── anchoring_probe_v4.py     # plant-parameter prompt, extract()
│   ├── stats_ci.py               # eta^2, bootstrap CI, cluster bootstrap, permutation test
│   └── baselines.py              # original baseline module, kept importable
├── data/
│   ├── selftest_data.jsonl       # 6 synthetic placeholder texts (not clinical text)
│   ├── counselchat/              # 20220401_counsel_chat.csv
│   ├── counselbench/             # optional, licence-restricted: counselbench_eval.csv
│   └── mentalchat16k/            # optional long narratives: Interview_Data_6K.csv
└── results/                      # run artifacts (per-row JSON + summary JSON + console log)
```

---

## 2. Environment

Python 3.10+ (verified on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Required: `numpy`, `scipy`.
- `vllm` is **not imported by this adapter**; install it separately on a GPU machine to
  serve a local model: `pip install vllm`. Neither offline self-tests nor
  `--backend openai` against a remote endpoint need it.
- Collecting LLM outputs does not require `requests`; HTTP calls use the standard library
  `urllib`.

---

## 3. Datasets (the code reads local paths only; nothing is downloaded automatically)

Unified record schema:

```json
{"id": "str", "text": "first-person help-seeking text", "meta": {"dataset": "...", "topic": "...", ...}}
```

### 3.1 Datasets and stripping rules

| `--dataset` | Role | Source / licence | text field | Rules |
|---|---|---|---|---|
| `counselchat` | **recommended** | HF `nbertagnolli/counsel-chat`; GitHub `nbertagnolli/counsel-chat`; **MIT** | `questionText`, falling back to `questionTitle`; deduplicated by `questionID` | Anonymous first-person English help-seeking posts scraped from counselchat.com on 2022-04-01. `answerText`, `therapistInfo`, `therapistURL`, `upvotes` and `views` are **all stripped**: they enter neither the LLM context nor `meta`. Measured: 2,775 records, 940 unique questions, 31 topics. |
| `counselbench_eval` | alternative retest set | HF `izi-ano/CounselBench-Eval`; **CC BY-NC-ND 4.0** | `questionText`, falling back to `questionTitle`; deduplicated by `questionID` | 100 clinically screened real CounselChat questions (20 topics × 5 upvoted questions). `response`, `responder`, `survey_id` and all sixteen rating/reason/annotation columns are **stripped** (`questionID` is used for deduplication only, `topic` only enters provenance metadata) so that no structured scale leaks in. Licence-restricted: internal research only, legal review required before redistribution. Small; not intended as the primary set. |
| `mentalchat16k` | alternative long narratives (with reservations) | HF `ShenLab/MentalChat16K`; **MIT** | `input` | **Only `Interview_Data_6K.csv` is accepted**: 6,338 items derived from 378 behavioural-health coach–caregiver interview transcripts, rewritten and summarised page by page with Mistral-7B-Instruct-v0.2. The speaker is a **caregiver, not the patient**, so it is second-hand report and must be reported separately from first-person patient text. `Synthetic_Data_10K.csv` is GPT-3.5 Turbo generated and the **loader refuses to read it**; it does not count as real input. |
| `jsonl` | generic fallback | user-supplied JSONL, one record per line | fields set by `--text-field` / `--id-field` (default `text` / `id`); everything else goes into `meta` | licence is the user's responsibility |
| `selftest` | offline placeholder | `data/selftest_data.jsonl` shipped here | `text` | synthetic placeholder, must not be used in the paper |

### 3.2 Downloading (run on the machine; then point `--dataset-path` at the local file)

```bash
mkdir -p data/counselchat data/counselbench data/mentalchat16k

# recommended: CounselChat (MIT, public direct download, not gated)
huggingface-cli download nbertagnolli/counsel-chat --repo-type dataset \
  --local-dir data/counselchat
# single-file fallback:
#   https://huggingface.co/datasets/nbertagnolli/counsel-chat/resolve/main/20220401_counsel_chat.csv

# alternative retest set: CounselBench-Eval (CC BY-NC-ND 4.0; confirm internal compliance first)
huggingface-cli download izi-ano/CounselBench-Eval --repo-type dataset \
  --local-dir data/counselbench
# single-file fallback:
#   https://huggingface.co/datasets/izi-ano/CounselBench-Eval/resolve/main/counselbench_eval.csv

# alternative long narratives: MentalChat16K, Interview subset only
# (do not download or load the Synthetic subset)
curl -L -o data/mentalchat16k/Interview_Data_6K.csv \
  https://huggingface.co/datasets/ShenLab/MentalChat16K/resolve/main/Interview_Data_6K.csv
```

If `huggingface.co` is not reachable, substitute the public mirror `hf-mirror.com` for
the host in the URLs above. The adapter only reads local files and contains no
downloading logic.

### 3.3 Verified fields and open items

**Verified (headers checked against the local CSV files)**

- CounselChat, ten columns: `questionID, questionTitle, questionText, questionLink,
  topic, therapistInfo, therapistURL, answerText, upvotes, views`. The `split` column
  listed in the older HF card template is absent from the actual header; the ten
  measured columns are authoritative.
- CounselBench-Eval, twenty columns: `questionID, questionTitle, questionText, response,
  topic, responder, survey_id, overall_score, overall_reason, empathy_score,
  specificity_score, medical_advice_score, medical_copy, medical_reason,
  factual_consistency_score, factual_copy, factual_reason, toxicity_score,
  toxicity_copy, toxicity_reason`.
- MentalChat16K, three columns: `instruction, input, output` (Interview header
  verified; the Synthetic file has been excluded).

**Open items (nothing inferred or filled in)**

1. The three older snapshots in the CounselChat GitHub `data/` directory
   (`20200325_counsel_chat.csv`, `counsel_chat_250-tokens_full.json`,
   `counselchat-data.csv`) have not been compared column by column against the HF
   20220401 version; the HF main table is authoritative.
2. The availability and column names of the unofficial mirror
   `mpingale/mental-health-chat-dataset` are **unverified** and it is not a default
   source; only the official `nbertagnolli/counsel-chat` repository is recognised.
3. Column-level verification of the `Synthetic_Data_10K.csv` header on disk is
   outstanding; the file is rejected by the loader either way.
4. The internal legal review of the CC BY-NC-ND 4.0 terms for CounselBench-Eval /
   CounselBench-Adv is still open.
5. Which dataset to use, how many records, whether to restrict `topic`, and whether to
   fold in the alternative retest set are left to the user; this layer does not
   pre-commit to a selection.

### 3.4 Local load check on the real CounselChat file

`20220401_counsel_chat.csv` (4,129,977 bytes) loaded with the `counselchat` loader in
dry-run mode, without calling any model:

- **2,775** raw records; **940** unique questions after deduplication by `questionID`;
  **31** topics.
- 139 rows have an empty `questionText` and fall back to `questionTitle`; all 940
  deduplicated records carry non-empty text.
- Merged text length (characters): min 19 / median 241.5 / mean 299.9 / max 2,690. A
  separate count over `questionText` alone gives median 215 / mean 252 / max 2,702; the
  two differ because the merged text includes the title fallback and whitespace
  folding. Cite the `questionText` figures in the paper.
- Stripped columns (`answerText`, `therapistInfo`, `therapistURL`, `upvotes`, `views`)
  leave zero residue in `meta`; the loader self-check passes.
- The full console log is in `results/dryrun_counselchat.txt`. No LLM backend was
  constructed and no model output was produced.
- The 4 MB CSV is a local copy used for this check and need not be uploaded to the
  machine; re-download it from the official link in 3.2 when deploying.

---

## 4. Running

### 4.1 Option A: vLLM with a local open-weight model (RTX 4090)

```bash
# 1) install vLLM and start the OpenAI-compatible server (model name is an example)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# 2) in another terminal, set the environment and run (CounselChat, mapped + directcost)
export COSPEC_VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# optional: export COSPEC_VLLM_URL=http://localhost:8000/v1/chat/completions (default)

python run_realdata_experiment.py \
  --dataset counselchat \
  --dataset-path data/counselchat/20220401_counsel_chat.csv \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --mode both \
  --n 200
```

Comparable 7B-class instruct models (suggestions only): Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, Yi-1.5-9B-Chat.

### 4.2 Option B: OpenAI-compatible remote API (OpenAI / DeepSeek / other compatible endpoints)

```bash
export OPENAI_BASE_URL=https://api.deepseek.com/v1   # or https://api.openai.com/v1
export OPENAI_API_KEY=sk-xxxx                        # DEEPSEEK_API_KEY also accepted

python run_realdata_experiment.py \
  --dataset counselchat \
  --dataset-path data/counselchat/20220401_counsel_chat.csv \
  --backend openai \
  --model deepseek-chat \
  --mode both \
  --n 200
```

If `OPENAI_BASE_URL` or `OPENAI_API_KEY` is missing, the backend raises an explicit
`RuntimeError` **before** issuing any request; it never fails silently.

### 4.3 Alternative datasets

```bash
# CounselBench-Eval retest set (confirm the CC BY-NC-ND 4.0 position first; questions only)
python run_realdata_experiment.py \
  --dataset counselbench_eval \
  --dataset-path data/counselbench/counselbench_eval.csv \
  --backend vllm --model Qwen/Qwen2.5-7B-Instruct --mode both

# MentalChat16K Interview subset (second-hand report; report it separately.
# Do not pass Synthetic_Data_10K.csv — the loader rejects it)
python run_realdata_experiment.py \
  --dataset mentalchat16k \
  --dataset-path data/mentalchat16k/Interview_Data_6K.csv \
  --backend vllm --model Qwen/Qwen2.5-7B-Instruct --mode both
```

### 4.4 The two modes

| `--mode` | Mirrors | Collects | Scores |
|---|---|---|---|
| `mapped` | `case_study.py`: the same `(omega_n, zeta)` down the I3 plant route and the I4 cost route | `anchoring_probe_v4.PROMPT` asking for `omega_n / zeta` over the ranges `[0.1,2.0]` and `[0.05,2.0]` (the original `mid` range) | the I3/I4 metrics of `case_study.run`, row by row |
| `directcost` | `cost_probe.py`: asks for `q1, q2, r` directly | `cost_probe.PROMPT` | `cost_probe.score`, which calls the `cs.*` functions internally |
| `both` | both passes above | — | — |

### 4.5 Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | required | `counselchat` (recommended) / `counselbench_eval` / `mentalchat16k` / `jsonl` / `selftest` |
| `--dataset-path` | registry default | local CSV / JSONL path; required for `jsonl` |
| `--backend` | required unless `--dry-run` | `vllm` / `openai` / `mock` |
| `--model` | environment or default | model name; optional for `mock` |
| `--n` | all | take only the first n records (debugging) |
| `--mode` | `both` | `mapped` / `directcost` / `both` |
| `--out` | `results` | output directory (relative to the script or absolute) |
| `--dry-run` | off | load the dataset and print record count, topic distribution, text-length statistics and the first sample; **no backend, no LLM calls, no result files** |
| `--selftest` | off | mark the run as an offline self-test; artifacts carry placeholder provenance |
| `--wn-true` / `--zeta-true` | `0.8` / `0.5` | nominal true plant (no ground truth exists; used with the sensitivity sweep) |
| `--temperature` | `0.3` | aligned with `cost_probe.TEMPERATURE` and `anchoring_probe_v4.TEMPERATURE` |
| `--repeats` | `1` | collections per record (the original cost probe used 20) |
| `--seed` | `20260921` | seed for the paired bootstrap and the paired permutation test; the global RNGs inside `cs.mc_gate` and `stats_ci` keep the fixed seeds of the original modules and are not overridden |
| `--text-field` / `--id-field` | `text` / `id` | used by `jsonl` only |
| `--sleep` | `0.2` | seconds between remote calls (aligned with the original 0.2 s) |

---

## 5. Offline mock self-test and load check (no GPU, no network, no weights)

```bash
pip install -r requirements.txt

# A) load the real dataset and check fields and counts (no LLM calls)
python run_realdata_experiment.py --dataset counselchat --dry-run

# B) end-to-end run against the shipped placeholder data
python run_realdata_experiment.py --dataset selftest --backend mock --mode both --selftest
```

Output of the mock self-test (placeholder; must not be used in the paper). Python 3.13,
numpy 2.5.4, scipy 1.18.1; 6 records from `selftest_data.jsonl`; nominal true plant
`omega_n=0.8, zeta=0.5`; temperature 0.3; repeats 1. Full console log in
`results/selftest_console.txt`.

**mapped mode (N=6)**

- Parsed 6/6; I3 destabilisation 0/6; I4 destabilisation 0/6; certificate passed 6/6.
- Monte Carlo gate (±25%, 50 draws, ≥80%): accepted 6/6, false accepts 0/6, safe samples
  wrongly rejected 0/6.
- Cost ratio J/J* over finite samples: I3 median 1.22 / p95 1.30 / max 1.31; I4 median
  1.03 / p95 1.09 / max 1.09.
- Beta/alpha bound violations 0/6; margin median 1.90×.
- Destabilisation-rate difference I3 − I4 = 0.000, 95% CI [0.000, 0.000], permutation
  p = 1.0000.
- Five-point nominal-plant sensitivity sweep (0.5/0.5, 0.8/0.5, 1.2/0.5, 0.8/0.3,
  0.8/0.8): I3/I4 destabilisation and gate false-accept rates are 0.0% at all five
  points on this placeholder sample, which only shows that the pipeline runs through.

**directcost mode (N=6)**

- Parsed 6/6; certificate admitted 6/6, rejected 0.
- J/J*: median 1.01 / p95 1.02 / max 1.03; beta/alpha median 2.4 / p95 3.3 / max 3.4.
- Bound violations 0; destabilisation among admitted samples 0; weight shape q1/q2
  median 16.17 / p5 9.63 / p95 30.42; fail-closed 0.

Mock values are derived from a seed fixed by the md5 of the prompt text, so they
reproduce across machines. The all-zero rates say nothing about any real model. The
branching paths (parse failure, rejection, gate false accept, I3 destabilisation,
sensitivity sweep) are implemented and only produce non-zero counts with a real backend
on real data.

---

## 6. Artifacts

Each real run writes into `results/` with a `YYYYMMDD_HHMMSS` timestamp:

| File | Content |
|---|---|
| `mapped_rows_<stamp>.json` | per-record rows: raw reply, parsed fields, the full I3/I4 metric set, `provenance` |
| `mapped_summary_<stamp>.json` | summary: destabilisation rates, certificate pass rate, gate false-accept rate, cost ratio median/p95/max, bound violations, rate-difference CI and permutation p, five-point sensitivity sweep, topic variance decomposition |
| `directcost_rows_<stamp>.json` | per-record rows: raw reply, q1/q2/r, admit/reject reason, rho, cost ratio, alpha/beta/bound, q1/q2, q1/r, q2/r, `provenance` |
| `directcost_summary_<stamp>.json` | summary: parse rate, admit rate, reject-reason distribution, cost ratio and beta/alpha distributions, bound violations, destabilisation count, weight-shape distribution, topic variance decomposition, fail-closed count |
| `selftest_console.txt` | full console output of the offline mock self-test |
| `dryrun_<dataset>.txt` | `--dry-run` load check (load statistics only, no LLM output) |

The `meta` field of each summary records the timestamp, provenance, backend, model,
dataset, temperature, repeats, seed, nominal plant, record count, the list of core
modules used, and the placeholder declaration. Non-finite values (NaN, infinity) are
written as `null` so that the JSON parses with any standard reader.

---

## 7. Reuse of the original modules (imported, not reimplemented)

### 7.1 `import case_study as cs` (verbatim copy of `cospec_core/case_study.py`)

| Symbol | Use |
|---|---|
| `cs.second_order(wn, zeta, dt=cs.DT)` | exact zero-order-hold discretisation of the continuous second-order psychological model |
| `cs.lqr(A, B, Q, R)` | gain from the discrete ARE |
| `cs.rho(M)` | closed-loop spectral radius |
| `cs.cost(A, B, Q, R, K)` | realised closed-loop cost (infinity when unstable) |
| `cs.admissible(A, Qh, Rh, tol=1e-9)` | O(n^3) admissibility certificate, returns `(bool, reason)` |
| `cs.alpha_beta(Qh, Rh, Q, R)` | suboptimality bound, returns `(alpha, beta)` |
| `cs.mc_gate(Ah, Bh, Q, R, n_draws=50, eps=0.25, need=0.80)` | Monte Carlo gate baseline, returns `(pass, frac)` |
| `cs.DT / cs.Q_TASK / cs.R_TASK / cs.OMEGA_REF / cs.ZETA_REF` | task parameters and mapping reference points, used throughout |

### 7.2 `import cost_probe as cp` / `import anchoring_probe_v4 as ap` (verbatim copies)

| Symbol | Use |
|---|---|
| `cp.PROMPT` | direct cost-elicitation prompt (`{text}` placeholder) |
| `cp.extract(text)` | `\{.*\}` regex followed by `json.loads` |
| `cp.score(rows, wn_true, zeta_true)` | direct cost scoring (admit/reject, rho, cost, alpha/beta/bound, weight shape), calling `cs.*` internally |
| `ap.PROMPT` | plant-parameter prompt (`{text}`, `{wn_lo,wn_hi}`, `{z_lo,z_hi}` placeholders) |
| `ap.extract(text)` | the same regex parse |

### 7.3 `import stats_ci` (verbatim copy of `cospec_core/stats_ci.py`)

| Symbol | Use |
|---|---|
| `stats_ci.eta2(vals, groups)` | between-group share of variance |
| `stats_ci.boot_ci(vals, groups, B=...)` | percentile bootstrap 95% CI (used by both summaries) |
| `stats_ci.cluster_boot_ci(rows, param, key, B=2000)` | patient-level cluster bootstrap CI (topic decomposition in `mapped`) |
| `stats_ci.perm_test` | **not called directly**: its permutation structure assumes several range labels inside one vignette, which a single real patient text does not have. `_rate_diff_permutation` here reimplements the paired test with the same algorithm (B=10000, `p=(1+#null>=|obs|)/(B+1)`); the comment in the function records the derivation. `_rate_diff_bootstrap` likewise follows `stats_ci.boot_ci` (B=10000, 2.5/97.5 percentiles). |

### 7.4 `import baselines`

The original baseline module is shipped so that the package stays complete and
importable. This adapter does not copy its safety-mechanism or negative-control
implementations and does not call its functions again.

---

## 8. Statistics and scope

1. **No ground-truth dynamics.** Real text has no corresponding true
   `(omega_n, zeta)`, exactly as in the original `case_study.py`: a nominal true plant is
   used, with a five-point sensitivity sweep at the end of each summary
   (`(0.5,0.5), (0.8,0.5), (1.2,0.5), (0.8,0.3), (0.8,0.8)` — the same point set as the
   `__main__` block of the original script).
2. **What public corpora cannot supply.** Real treatment outcomes (pre/post measures,
   remission, relapse), the state matrices `A/B`, LQR cost-weight labels `q1,q2,r`,
   optimal trajectories or control sequences, and longitudinal state records must not be
   reverse-engineered from public conversation text and used as controller ground truth.
3. **`mapped` summary** (same definitions as `case_study.report`): I3/I4 destabilisation
   counts and rates, certificate pass rate, gate accept rate, gate false accepts (gate
   passes and the true loop is unstable), safe samples wrongly rejected by the gate, and
   cost ratio J/J* median/p95/max over finite samples, plus beta/alpha violations and
   the margin distribution.
4. **`directcost` summary** (same definitions as `cost_probe.analyze`): parse rate,
   certificate admit rate, reject-reason distribution, cost ratio and beta/alpha
   distributions over admitted samples, bound violations, destabilisation count among
   admitted samples, weight-shape q1/q2 distribution, fail-closed count. The original
   fast/slow prognosis split depends on frozen labels of the synthetic vignettes, which
   real datasets do not carry, so it is not reproduced here.
5. **Rate-difference test**: paired bootstrap CI and paired permutation test, following
   the algorithms in `stats_ci` (see 7.3).
6. **Dataset-specific limits**
   - CounselChat consists of written forum posts, not session transcripts; it differs
     from clinical conversation in discourse density, emotional intensity and question
     length, and the paper's Limitations must address external validity.
   - The MentalChat16K Interview subset is caregiver report rewritten by a model and must
     be reported separately from first-person patient text; the Synthetic subset does not
     count as real input.
   - For CounselBench-Eval only the question text is used; `response` and every
     rating/annotation column are stripped. The NC-ND licence boundary is in section 0
     and 3.1.
7. All closed-loop results are simulation and constitute no claim of clinical validity.
   This adapter covers the psychological-intervention LQR experiment only; it contains no
   driving simulation.

---

## 9. Checklist before a real run

- [ ] `python -m py_compile` passes on everything (commands below).
- [ ] `--dataset counselchat --dry-run` gives the expected counts and topic distribution,
      with no residue from the stripped columns.
- [ ] The mock + selftest run completes end to end with
      `provenance=selftest-mock-placeholder`.
- [ ] After downloading a real dataset, run `--n 20` first to check fields, parse rate
      and reject reasons.
- [ ] If using `counselbench_eval`: confirm the CC BY-NC-ND 4.0 position has been
      cleared.
- [ ] Confirm that `provenance` is `real-data-*` before any result enters the paper.
- [ ] Delete the placeholder self-test artifacts and dry-run logs from `results/` before
      storing real results.
- [ ] Confirm that no synthetic subset such as `Synthetic_Data_10K.csv` was placed in a
      real-input directory.

### Compile check

```bash
python -m py_compile dataset_loaders.py llm_backends.py run_realdata_experiment.py \
  cospec_core/case_study.py cospec_core/cost_probe.py cospec_core/anchoring_probe_v4.py \
  cospec_core/stats_ci.py cospec_core/baselines.py
```
