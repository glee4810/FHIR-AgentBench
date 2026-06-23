# Findings — what actually moves an LLM agent's accuracy on FHIR clinical QA

*Capstone synthesis of this fork's experiments. Substrate throughout:
[FHIR-AgentBench](https://arxiv.org/abs/2509.19319) (Lee et al., ML4H 2025) — MIMIC-IV-on-FHIR demo,
single-patient retrieval QA — loaded into a self-hosted [Medplum](https://www.medplum.com) R4 server.
Headline numbers are the **trustworthy re-grade** (deterministic numeric + a 3-Claude-judge panel,
cross-checked by an independent codex/GPT panel and validated against non-LLM ground truth on numerics);
see [TRUSTWORTHY_REGRADE.md](TRUSTWORTHY_REGRADE.md) for why the benchmark's default gpt-5-mini judge could
not be trusted and how a boolean-grading bug in our own first fix was caught and corrected.*

---

## The question

When you put an LLM agent in front of a real FHIR server, **what actually makes it more accurate** — and how
do you measure it *honestly*, without shipping a confound as a win? We swept the levers an engineer would
reach for, each as a **paired** comparison (same questions, both arms) with exact statistics.

## Bottom line — the bottleneck is context/data-plumbing, not tool design, payload, thinking time, or compute

Every "win" in this project decomposed to one thing: **whether a bounded slice of FHIR reaches the model's
context.** A single MIMIC patient can carry 10k+ resources; serialized into a 32k window it overflows, and
the agent is scored wrong. Tool catalogs, payload coaching, and more thinking time did nothing. And the
code interpreter — the lever the early result seemed to point at — turns out to help only by **avoiding the
overflow** (it hands the payload to a sandbox instead of the prompt), not by reasoning better.

| Lever tested | Effect on accuracy | Evidence |
|---|---|---|
| **Purpose-built / typed tool catalog** (vs one generic `fhir-request`) | **NULL** | Opus structure-lift +0.08, 95% CI **[-0.12, +0.28]**, p=0.69; GPT-5.5 curve flat; the early +11pp was a context-overflow artifact. |
| **Payload shaping** (`_elements`/`_summary` coaching) | **cost-only** | Δ0.00 (p=1.0); changes token spend, not accuracy. |
| **Reasoning effort** (medium → high) | **NULL** | **0/30** answer flips on a fixed retrieved context; ~1.6× cost for identical answers. |
| **Code interpreter** (`+execute_python_code`) | **no benefit where the no-code agent can answer; helps only by avoiding overflow** | **Matched budget (both answer, n=140): −3.6pp, 95% CI −7.7…+0.6, p=0.18 → not significant.** Pooled +39.9pp is **entirely** the 262/409 (64%) questions where the no-code agent overflows the 32k cap. |

The honest headline: **for FHIR-agent accuracy, the first-order lever is getting a bounded, projected slice
of FHIR into context.** A code path helps because sandboxing the payload is *one way* to dodge the overflow;
payload projection (`_elements`/views) plausibly targets the same lever. Compute is not the bottleneck — on
questions both agents can fit, the interpreter adds nothing.

---

## The code interpreter, in full — an overflow-avoidance effect, not a reasoning win

Standard-vs-standard, using the benchmark's **own** paired reference agents unchanged: `multi_turn_resource`
(retrieve → reason in natural language) vs `multi_turn_code_resource` (retrieve → `execute_python_code` →
answer). GPT-5.5 both arms, full 409-question held-out test split. Trustworthy grading.

| Stratum | n | resource | code | Δ (95% CI) | McNemar | what it means |
|---|---|---|---|---|---|---|
| **Matched budget** (both arms answered) | 140 | 71.4% | 67.9% | **−3.6pp** (−7.7…+0.6) | **p=0.18 → not significant** | when both can fit the data, the interpreter gives no significant benefit (slight negative point estimate) |
| **Resource-real** (predefined by resource success) | 147 | 70.8% | 64.6% | **−6.1pp** (−10.8…−1.4) | p=0.022 → significant | on answerable questions code is *worse* — but ~⅓ of that is code's higher **error** rate, not reasoning |
| **Large records** (resource overflows 32k) | 262 | 0% | 65.6% | — | by construction | code answers a class the no-code agent **structurally cannot fit** — the one real, large effect, and it is **architectural** |
| Pooled (mixes the strata) | 409 | 25.4% | 65.3% | +39.9pp (+34.6…+45.1) | p≈0 | a valid *deployed-architecture* comparison, **not** a reasoning gain |

- The no-code arm **overflows on 262/409 = 64%** of questions. That single fact is the entire pooled gap.
- In the controlled **matched-budget** stratum, code **fixed 2, broke 7**: a slight, non-significant deficit.
  The interpreter does not make GPT-5.5 reason better about FHIR.
- On the predefined **resource-real** stratum code is significantly worse (−6.1pp), but decomposing its 11
  "broke": **4 are the code arm erroring/overflowing** where simple retrieval succeeded (a reliability gap),
  **7 are buggy generated code**. So the defensible claim is: *code is no better, and somewhat less reliable,
  on questions the no-code agent can already handle.*

### Why the earlier "+11pp" and "−8.6pp" were both wrong

This number moved three times under adversarial review; each move corrected a real defect:

1. **+11pp "code is the lever."** Naive pooled accuracy with gpt-5-mini, plus a harness bug (overflow/error
   answers fed to the judge instead of auto-scored 0). → context-overflow artifact.
2. **−8.6pp "code HURTS at matched budget."** After fixing the pre-filter, canonical gpt-5-mini said code
   *significantly hurt* (p=0.02). A **judge artifact**: gpt-5-mini is **61% accurate** against non-LLM numeric
   ground truth and wrongly rejects exact answers (43 false negatives, median error 0.0; 0 false positives) —
   a one-directional precision bias that targets the code arm's terse numerics.
3. **−3.6pp, not significant (trustworthy).** Deterministic numeric grading + a 3-Claude-judge panel (after
   fixing a boolean-grading bug in our own first fix — 115 Yes/No golds had been mis-graded as numeric). The
   matched-budget effect is a non-significant slight negative. **Independent confirmation:** a codex/GPT panel
   agrees with the Claude panel on **97.1%** of non-numeric labels and reproduces the *same* stratified result;
   both panels score **98-99%** vs numeric ground truth (gpt-5-mini: 61%). Full audit:
   [TRUSTWORTHY_REGRADE.md](TRUSTWORTHY_REGRADE.md).

The methodological lesson is sharp: **an unreliable LLM judge doesn't just add noise — it adds *directional*
bias** (here a confident phantom "code hurts"). We only caught it because numeric answers can be graded
against ground truth, exposing the judge.

## Why the tool catalog is a null (the first time the overflow confound bit)

The project *started* from a tempting result: a 5-tool catalog beat one generic tool by +11pp (≈39%→50%).
Under controls — a context-cap factorial, paired McNemar/bootstrap, a coached-generic control — it
**decomposed to nothing** on both Opus and GPT-5.5. The only robust effect was the **context budget**:
reference-resolution (`_include`) tools overflow the 32k cap (one Opus arm overflowed 20/25 questions at 32k
→ 0.16 accuracy; raising the cap fixed it, p=0.0005). The "tool benefit" was the catalog *dodging the
overflow* — **the same mechanism that later masqueraded as a code-interpreter compute win.** Replicates the
parent paper's ablation (o4-mini: generic 0.25 ≥ specialized 0.22). Full decomposition in `REPORT.md`.

## A second, transferable finding — LLM-as-judge reliability in clinical QA

Because we graded numerics against known ground truth, we could measure the judges themselves:

- The benchmark's **default judge (gpt-5-mini) is 61% accurate** on numeric clinical-QA grading, with a
  one-directional **precision-punishing** bias (rejects exact answers; never rewards a gross miss).
- A **3-vote panel recovers it** — Claude 98.2%, codex/GPT 99.1% — and the two families agree with each other
  97.1%.
- Consequence for anyone running FHIR-AgentBench (or similar) with an LLM judge: **audit the judge against
  ground truth on a deterministic slice, and use a multi-vote panel.** A single small judge can invert a real
  effect's sign.

## Honest caveats (read before citing)

- **The trustworthy conclusion is a null (matched budget) plus an overflow architecture effect — not a
  reasoning win, and not "code is broadly worse."** If you cite one line: *a code interpreter gives GPT-5.5 no
  significant FHIR-QA accuracy benefit at matched context budget (−3.6pp, p=0.18); its value is avoiding
  context overflow on large patients, which payload projection plausibly also achieves.*
- **The strata are post-hoc** (defined by which arm overflowed/answered), not by pre-tokenized record size;
  the matched-budget stratum conditions on success, so it is a conservative read. **Questions cluster by
  patient** (~90 patients), which McNemar ignores — a clustered bootstrap leaves the matched-budget null and
  the overflow effect intact, but the resource-real −6.1pp significance is **fragile**: leave-one-patient-out
  lifts its p to 0.065. The load-bearing result is the matched-budget null; resource-real is directional
  corroboration, not a hard claim.
- **The mechanism is not fully identified.** The code arm bundles interpreter + code-tailored prompt +
  pointer-based payload routing; we show the win is overflow-driven and the reasoning effect null, but we have
  not isolated compute from prompt/routing (needs a same-payload, no-execution control).
- **Single seed, single model, single substrate.** GPT-5.5 ≠ o4-mini, Medplum ≠ GCP, retrieval capped at 10k
  resources/type. Contributions: (1) the correct decomposition (the code "win" is a context-budget artifact,
  not compute — the same confound faked the tool-catalog win); (2) the judge-reliability finding +
  deterministic/panel grading that caught a directionally biased judge and a boolean-grading bug; (3) the
  reusable standard-vs-standard harness on Medplum.
- **Reproducibility is split.** The **trustworthy re-grade is fully reproducible** from committed artifacts
  (`runs/full409/{det_labels,panel_votes*,human_review}.json` + `build_labels.py` + `final_grade.py`). The
  **agent run** needs the Medplum substrate + a funded OpenAI key. The **Opus tool-ablation numbers are NOT**
  reproducible (torn-down EC2; `REPORT.md` §1).

## Map

- `TRUSTWORTHY_REGRADE.md` — the grading audit: the harness bug, the unreliable gpt-5-mini judge, the boolean
  bug in our own fix, the deterministic+panel grading, the codex/GPT cross-check, the judge leaderboard.
- `REPORT.md` — the tool-ablation null in full (cap-factorial, paired stats, prior art).
- `CODE_EXPERIMENT.md` — the code-interpreter result + mechanism.
- `runs/full409/human_review.{json,csv}` — **every one of the 409 questions**, both arms' final answers, all
  four judges' labels (gpt-5-mini / deterministic / Claude panel / codex panel) and the final label, for human
  audit.
- `runs/full409/_trustworthy_summary.json`, `_judge_leaderboard.json`, `_magnitude_analysis.json` — the final
  numbers. Reproduce: `python build_labels.py && python final_grade.py`.
