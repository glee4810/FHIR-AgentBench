# No-token retrieval eval

This is the cheaper first pass before burning model tokens or calibrating an LLM judge.

## Thesis

For FHIR-AgentBench, the benchmark already gives `true_fhir_ids`. That means we can score the retrieval layer directly:

- **recall:** did the agent retrieve the gold FHIR resource IDs?
- **precision:** how much non-gold resource material did it pull?
- **payload pressure:** how many prompt tokens / trace chars / tool calls did that strategy spend?
- **failure shape:** quota, harness, and context failures stay visible instead of becoming "wrong answer."

Final answer correctness can stay secondary until retrieval shows a real difference.

## Prototype

Run from the repo root:

```bash
python retrieval_quality.py medplum-eval/results --out-dir medplum-eval/retrieval-metrics
```

The committed `medplum-eval/results/*.json` files already contain:

- `true_fhir_ids`
- `agent_fhir_resources`
- `usage`
- saved traces

So this pass makes no model calls and needs no Medplum server.

Generated files:

- `retrieval_summary.csv` — arm-level recall/precision/token footprint.
- `retrieval_by_type.csv` — recall split by FHIR resource type.
- `retrieval_per_question.csv` — one row per question per arm.
- `retrieval_vs_control.csv` — every non-control arm compared against `control` for the same question.
- `retrieval_report.md` — compact human report with wins, regressions, and bloat cases.

## What the existing GPT-5.5 artifacts say

The catalog arms retrieve more gold resources, but the price is huge payload waste:

| arm | recall | precision | retrieved IDs | false positives |
|---|---:|---:|---:|---:|
| control | 0.281 | 0.076 | 1048 | 968 |
| validated5 | 0.512 | 0.040 | 3612 | 3466 |
| cat2 | 0.526 | 0.043 | 3453 | 3303 |
| cat4 | 0.526 | 0.045 | 3364 | 3214 |
| arm_ref | 0.400 | 0.035 | 3244 | 3130 |

Compared against `control` question-by-question:

- `clean_recall_win`: 2
- `recall_bought_with_bloat`: 8
- `pure_bloat`: 59
- `retrieval_regression`: 28
- `same_or_cheaper`: 53

This is the point of the no-token pass: it exposes that "more tools" mostly meant "retrieve a lot more FHIR material," not "retrieve the right material cleanly."

## What this does not prove yet

The current artifacts were produced by an LLM agent, so `prompt_tokens` and trace size are observational, not a pure retrieval-only benchmark. The next version should add a retrieval-only runner that executes planned FHIR calls, stores exact payload bytes, and never calls an LLM. This prototype is the scoreboard shape.

## Stronger next step

Compare strategies, not tool count:

1. generic `fhir_request`
2. generic + payload-aware instructions
3. typed catalog
4. payload-aware planner that chooses `_elements`, `_count`, pagination, and `_include` only when needed

Primary metric should be gold-resource recall per 1k prompt tokens, with precision and overflow rate next to it.

## Spend gate

Do not run another answer-quality matrix until a retrieval-only strategy beats `validated5` on at least two of these three metrics:

1. same or better gold-resource recall,
2. at least 2x better precision,
3. at least 2x better recall per 1k prompt tokens.

If the retrieval layer cannot clear that bar without model generation, a larger LLM run is just paying to rediscover payload bloat.
