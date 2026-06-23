# Code-interpreter experiment — the one lever that works (GPT-5.5)

After three nulls (typed tools, payload shaping, reasoning effort — none improved accuracy), this tests
the one lever the evidence pointed at: a **code interpreter**. It is run as **standard-vs-standard** —
the benchmark's *own* reference agents, unchanged, so there are no hand-picked "our tools" to argue about.

## Setup

- **Arms (FHIR-AgentBench's own agents):** `multi_turn_resource` (retrieve → reason in natural language,
  no code) vs `multi_turn_code_resource` (retrieve → `execute_python_code` → answer). Only the code tool
  differs.
- **Model:** GPT-5.5 (`gpt-5.5-2026-04-23`), both arms. **Judge:** gpt-5-mini (the benchmark's judge).
- **Data:** 200 held-out **test**-split questions (seed 20260621), MIMIC-IV-on-FHIR demo loaded into a
  self-hosted **Medplum** (the benchmark's GCP client is swapped for a Medplum backend via
  `medplum_fhir_client.py`; FHIR R4 is FHIR R4, agent logic unchanged).
- **Stats:** paired McNemar (same questions both arms).

## Result — significant (full 409-question test set)

| arm | accuracy |
|---|---|
| resource (no code) | **27.6%** (113/409) |
| code (+`execute_python_code`) | **38.6%** (158/409) |
| **delta** | **+11.0 pp** |

**McNemar:** code **fixed 76**, **broke 31** → **exact p = 1.6e-5 (highly significant).** Net **+45** questions.
**Conservative:** the code arm only *answered* 390/409 (19 timed out / errored on heavy patients and were
scored wrong), so the true lift is if anything larger.
**Cost:** **$34.38** total agent (resource $11.63 + code $22.75); the gpt-5-mini judge re-run cost <$1.

> **n=200 preview (earlier, consistent):** on the first 200 test questions the lift was **+6.5pp**
> (21.0%→27.5%, McNemar p=0.0106, fixed 18/broke 5, cost $5.36). The full 409-set lift is *larger* — the
> additional 209 questions skew more compute-heavy, where code helps most. **Same direction, stronger, more
> significant.** (The 409 was re-judged 2026-06-23 after the first scoring pass erred on OpenAI quota; the
> agent answers were intact — see `medplum-eval/FULL409_STATUS.md`.)

The code interpreter genuinely lifts GPT-5.5's accuracy on realistic FHIR QA — confirming the parent
paper's finding (which used o4-mini) on a *new model* and a *self-hosted substrate*, with paired stats.

## Where the lift comes from (mechanism)

Of the **76 questions code fixed** (full 409 set), **≈89% (68/76) are compute-shaped** — temporal-ordering
("first/last/since…" → sort events by date) and aggregation ("count occasions", "average…", "total…") —
exactly what a no-code agent can't do reliably by reading FHIR JSON in-context, and what Python does
perfectly. (Keyword heuristic, consistent with the hand-verified **17/18 compute-heavy** on the n=200
subset.) The **31 it broke** are *also* mostly compute-heavy (temporal/aggregate where the model wrote buggy
code) — **code is the lever, not a guarantee**, but the net is strongly positive (76 vs 31).

## Honest caveats

- **Absolute accuracy is moderate** (27.6% / 38.6%) vs the paper's 0.50 ceiling. Main driver: the **resource
  agent overflows** on heavy patients (a patient can have 10k+ observations — they don't fit in context,
  so it can't answer; the code agent sidesteps this by processing them in code). Also: GPT-5.5 ≠ o4-mini,
  Medplum ≠ GCP, and the adapter caps retrieval at 10k resources/type (Medplum's offset limit). **The
  *delta* is the finding, not the absolute level** — and the delta is significant.
- **Single seed, single model, full 409-question test split.** The lift replicates a known direction; it is
  not a novel discovery that "code helps" — the contribution is the clean GPT-5.5 confirmation + the
  failure-mode map + the reusable standard-vs-standard harness on Medplum.

## Reproduce

```bash
# substrate: docker compose up in medplum-eval-bundle/ + bash scripts/load_mimic.sh (incl. chartevents)
export MEDPLUM_BASE_URL=http://localhost:8103 OPENAI_API_KEY=...
bash run_409.sh           # both arms, full 409-question test split (resumable; resumes by answered rows)
# scoring runs at the end of run_409.sh; if the judge erred (quota), finish it with:
python rejudge_409.py     # re-judges only the failed labels from medplum-eval/full409_answers.json
bash progress409.sh       # progress bar + prints the final result when done
```

Raw per-question outputs: `runs/full409/{multi_turn_resource,multi_turn_code_resource}.json` (gitignored;
regenerate with the above). Durable answer backup: `medplum-eval/full409_answers.json`. Summaries:
`medplum-eval/full409_summary.json` (409, final) and `medplum-eval/code_experiment_summary.json` (n=200 preview).

## How it fits the project

The arc: original eval = **null** → three nulls (tools / payload / reasoning effort) → **the lever is a
code interpreter** → **confirmed highly significant on the full 409-question test set** (+11.0pp, p=1.6e-5;
n=200 preview was +6.5pp, p=0.01). For accuracy on FHIR agents, the bottleneck is *compute over the data*,
not tool design, payload size, or thinking time. The cheap, defensible move for any FHIR agent: give it a
sandboxed Python/code path, not a bigger tool catalog.
