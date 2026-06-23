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

## Result — significant

| arm | accuracy |
|---|---|
| resource (no code) | **21.0%** (42/200) |
| code (+`execute_python_code`) | **27.5%** (55/200) |
| **delta** | **+6.5 pp** |

**McNemar:** code **fixed 18**, **broke 5** → **exact p = 0.0106 (significant, p < 0.05).**
**Cost:** resource $0.55 + code $4.82 = **$5.36 total** for n=200 (the code arm is ~9× pricier — it
writes/runs code over multiple turns).

The code interpreter genuinely lifts GPT-5.5's accuracy on realistic FHIR QA — confirming the parent
paper's finding (which used o4-mini) on a *new model* and a *self-hosted substrate*, with paired stats.

## Where the lift comes from (mechanism)

The 18 questions code **fixed** break down as:

- **10 temporal** ("first/last/since…" → requires sorting events by date)
- **7 aggregate/count** ("total output", "sum of urine output", "count occasions…")
- **1 existence/abstention**

→ **17 of 18 are compute-heavy** questions — exactly what a no-code agent can't do reliably by reading
FHIR JSON in-context, and what Python does perfectly. The 5 it **broke** were mostly aggregations too
(the model wrote buggy filtering/grouping code) — code is the lever, not a guarantee.

## Honest caveats

- **Absolute accuracy is low** (21% / 27.5%) vs the paper's 0.50 ceiling. Main driver: the **resource
  agent overflows** on heavy patients (a patient can have 10k+ observations — they don't fit in context,
  so it can't answer; the code agent sidesteps this by processing them in code). Also: GPT-5.5 ≠ o4-mini,
  Medplum ≠ GCP, and the adapter caps retrieval at 10k resources/type (Medplum's offset limit). **The
  *delta* is the finding, not the absolute level** — and the delta is significant.
- **Single seed, single model, n=200.** The lift replicates a known direction; it is not a novel
  discovery that "code helps" — the contribution is the clean GPT-5.5 confirmation + the failure-mode map
  + the reusable standard-vs-standard harness on Medplum.

## Reproduce

```bash
# substrate: docker compose up in medplum-eval-bundle/ + bash scripts/load_mimic.sh (incl. chartevents)
export MEDPLUM_BASE_URL=http://localhost:8103 OPENAI_API_KEY=...
bash run_full.sh          # both arms, n=200 (resumable; skips a completed arm)
python score_full.py      # resumable judge -> runs/full/_summary.json (paired McNemar + cost)
```

Raw per-question outputs: `runs/full/{multi_turn_resource,multi_turn_code_resource}.json` (gitignored;
regenerate with the above). Summary: `medplum-eval/code_experiment_summary.json`.

## How it fits the project

The arc: original eval = **null** → three nulls (tools / payload / reasoning effort) → **the lever is a
code interpreter** → **confirmed significant here** (+6.5pp, p=0.01). For accuracy on FHIR agents, the
bottleneck is *compute over the data*, not tool design, payload size, or thinking time. The cheap,
defensible move for any FHIR agent: give it a sandboxed Python/code path, not a bigger tool catalog.
