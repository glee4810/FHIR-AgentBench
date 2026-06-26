# Code-interpreter experiment — an overflow-avoidance effect, not a reasoning win (GPT-5.5)

After three nulls (typed tools, payload shaping, reasoning effort), this tests the one lever the early
evidence seemed to point at: a **code interpreter**. Run as **standard-vs-standard** — the benchmark's *own*
reference agents, unchanged. The honest result, after a harness-bug fix, a judge-reliability fix, and a
boolean-grading fix in our own pipeline, is that the interpreter's apparent accuracy win is **context-overflow
avoidance (data plumbing), not a compute or reasoning gain.** Where both agents can fit the data, it is not a
win — if anything a slight, non-significant loss.

## Setup

- **Arms (FHIR-AgentBench's own agents):** `multi_turn_resource` (retrieve → reason in natural language, no
  code) vs `multi_turn_code_resource` (retrieve → `execute_python_code` → answer). The code arm differs in
  three ways at once — the interpreter, a code-tailored prompt, **and** data plumbing: the resource arm
  serializes retrieved FHIR JSON into the prompt; the code arm passes a *pointer* to the payload into a Python
  sandbox (`retrieved_resources` global) and never serializes it into context. That plumbing difference is the
  result.
- **Model:** GPT-5.5 (`gpt-5.5-2026-04-23`), both arms. **Data:** full 409-question held-out test split,
  MIMIC-IV-on-FHIR demo loaded into a self-hosted **Medplum**.
- **Grading:** **deterministic numeric + a 3-Claude-judge panel**, cross-checked by an independent codex/GPT
  judge panel and validated against non-LLM ground truth on numerics. The benchmark's default gpt-5-mini judge
  is **not** trustworthy here (61% accurate vs numeric ground truth — see
  [TRUSTWORTHY_REGRADE.md](TRUSTWORTHY_REGRADE.md)). **Stats:** paired McNemar with 95% CIs.

## Result — stratified (the honest primary result)

| Stratum | n | resource | code | Δ (95% CI) | McNemar | reading |
|---|---|---|---|---|---|---|
| **Matched budget** (both arms answered) | 140 | 71.4% | 67.9% | **−3.6pp** (−7.7…+0.6) | **p=0.18 → not significant** | no reasoning benefit when both can fit the data |
| **Resource-real** (predefined by resource success) | 147 | 70.8% | 64.6% | **−6.1pp** (−10.8…−1.4) | p=0.022 → significant | code worse on answerable Qs (≈⅓ is code's higher error rate) |
| **Large records** (resource overflows 32k) | 262 | 0% | 65.6% | — | by construction | code answers what the no-code agent can't fit — architectural |
| Pooled (mix) | 409 | 25.4% | 65.3% | +39.9pp (+34.6…+45.1) | p≈0 | the overflow stratum; not a reasoning gain |

- **Matched budget is the controlled comparison, and it is not a win** (code fixed 2, broke 7; −3.6pp, p=0.18).
- **The pooled gap is overflow, not compute.** The resource arm hits `Input tokens exceeded` on **262/409
  (64%)**; the code arm produced a real answer on **240/262** of those. Strip the overflow and the advantage
  evaporates.
- **Resource-real (predefined) is significantly negative**, but decomposing the 11 questions code broke:
  **4 are code erroring/overflowing** where retrieval sufficed (reliability), **7 are buggy generated code**.
- **Cost:** ~$34 total agent run (resource $11.63 + code $22.75); judge panels separate.

## Where the apparent lift comes from (mechanism)

It is **not** "Python does temporal-ordering/aggregation the model can't." At matched budget — where the
no-code agent also has the data — the code arm shows no significant edge (−3.6pp, p=0.18). The lift is
concentrated where it is **mechanically forced**: the 262 large-record questions where the resource arm cannot
fit the patient into a 32k window and is scored 0, while the code arm processes the same payload
out-of-context in a sandbox. **The effect is about where the data lives, not computational reasoning.** Payload
projection (`_elements`/`_summary`/views) — keeping the serialized payload under the cap — is a different lever
that targets the *same* overflow. The later A0′ control tested one blunt recency-capped projection and recovered
only a third of the code arm's overflow accuracy; a query-aware projection remains untested.

## The three-round correction (why this matters)

| pass | grading | matched-budget verdict | defect |
|---|---|---|---|
| 1 | gpt-5-mini, no error pre-filter | "+11pp, code is the lever" | harness bug + pooled mixes overflow |
| 2 | gpt-5-mini, canonical pre-filter | "−8.6pp, code HURTS" (p=0.02) | judge artifact (gpt-5-mini 61% accurate, precision-punishing) |
| 3a | deterministic-numeric + Claude panel | "+1.4pp, null" | our own bug: 115 boolean Yes/No golds mis-graded as numeric |
| 3b | **deterministic(clean) + Claude panel, fixed** | **−3.6pp, not significant** | the trustworthy answer |

Independent codex/GPT panel reproduces 3b (matched −3.6pp p=0.18, pooled +39.1%). Full audit, judge leaderboard
(gpt-5-mini 61% vs panels 98-99% on numeric ground truth), and the magnitude analysis are in
[TRUSTWORTHY_REGRADE.md](TRUSTWORTHY_REGRADE.md).

## Honest caveats

- **The finding is a matched-budget null + an architecture effect**, not "code helps" and not "code is broadly
  worse." Its only positive value is dodging context overflow; its cost is slightly lower reliability on
  answerable questions.
- **"Matched budget" conditions on both arms answering** (post-treatment), so it is conservative; the
  resource-real stratum (predefined) is significantly negative but ≈⅓ reliability-driven. **Questions cluster
  by patient** (~90), which McNemar ignores.
- **Mechanism not isolated:** interpreter vs prompt vs payload-routing are confounded. **Single seed, single
  model, single substrate.** GPT-5.5 ≠ o4-mini, Medplum ≠ GCP, retrieval capped at 10k resources/type.

## Reproduce

```bash
# substrate: docker compose up in medplum-eval-bundle/ + bash scripts/load_mimic.sh (incl. chartevents)
export MEDPLUM_BASE_URL=http://localhost:8103 OPENAI_API_KEY=...
bash run_409.sh                              # both arms, full 409-question test split (resumable)
python build_labels.py && python final_grade.py   # deterministic + Claude-panel trustworthy grading (boolean-fixed)
python judge_leaderboard.py                  # judge accuracy vs non-LLM ground truth (numeric subset)
bash codex_panel.sh && python codex_judge_compare.py   # independent GPT cross-check (judge-family independence)
```

Per-question outputs: `runs/full409/multi_turn_{resource,code_resource}.json` (large, gitignored; regenerate).
**Human-auditable**: `runs/full409/human_review.{json,csv}` — all 409 questions, both arms' final answers, all
four judges' labels, final label. Durable answer backup: `medplum-eval/full409_answers.json`.

## How it fits the project

The arc: original eval = **null** → three nulls (tools / payload / reasoning effort) → a tempting
code-interpreter "+11pp" → which, under a harness-bug fix, a judge-reliability fix, a boolean-grading fix, and
stratification, **decomposes to a matched-budget null plus an overflow-avoidance architecture effect.** Unified
conclusion: **for FHIR agents, accuracy is gated by context/data-plumbing — getting a bounded, projected slice
of FHIR to the model — not by tool design, payload coaching, thinking time, or a code interpreter's compute.**
