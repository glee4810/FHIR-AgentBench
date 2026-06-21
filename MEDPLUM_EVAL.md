# Evaluating MCP tool surfaces for agents — experimental

> ⚗️ **Experimental / exploratory work.** This is a rig for asking a simple but slippery question:
> **when you expose tools to an LLM agent through an [MCP](https://modelcontextprotocol.io) server,
> do those tools actually make the agent better — and how would you *honestly* measure that?**
>
> The worked example here is FHIR / [Medplum](https://www.medplum.com) (a fork of
> [FHIR-AgentBench](https://github.com/glee4810/FHIR-AgentBench)), but the harness and the lessons are
> meant to generalize to evaluating *any* MCP tool surface. Treat the numbers as preliminary.

## What we're actually trying to do

"Add an MCP server with N purpose-built tools and the agent gets smarter" is the kind of claim that's
**easy to assert and easy to fool yourself about.** Vendor eval blog posts routinely ship a +X%
headline that doesn't survive scrutiny. This project is an attempt to build the eval *correctly* — to
measure whether an MCP tool surface helps, while controlling for the things that usually masquerade as
a tool win:

- **Context budget.** More/richer tools return bigger payloads; bigger payloads overflow the model's
  context window. A "tool win" is often just "this arm overflowed less." → we vary the context cap as
  an explicit factor (a **2×2 cap-factorial**) to separate *reasoning gain* from *cap-dodging*.
- **Prompt vs. structure.** Is the typed tool better, or did we just *tell* the agent something we
  never told the generic tool? → a **coached-generic control** (generic tool + the same coaching in its
  description only).
- **Noise.** An 8-point gap across 25–30 questions is inside the noise floor. → **paired McNemar +
  bootstrap** on per-question deltas, not eyeballed averages.
- **Hidden failures.** Overflows / rate-limits / no-answers ≠ wrong answers. → an **answerable-set
  accuracy** + a by-cause failure taxonomy.

The reusable artifact is this methodology, not any one number.

## The harness (generalizes to any MCP eval)

| File | Role |
|---|---|
| `agent/mcp_agent.py` | Agent that retrieves through an **MCP server** — the only variable across arms is the advertised tool surface. |
| `agent/ai_agent.py` | Variant that routes completions through a server-side LLM proxy (here, Medplum's in-FHIR `$ai` op) — test the lift on the platform's *own* agentic surface, not just an external client. |
| `treatment_mcp_server.py` | **Catalog-driven** FastMCP server: one server, `TOOL_SUBSET`-selectable arms (a baseline "generic" tool byte-for-byte matching the platform's shipped one, plus typed tools you toggle in). |
| `run_matrix.py` | Parameterized ablation runner: per-cell tool subset, a **nested dose-response staircase** (1→2→4→… tools), the cap-factorial, and a **hard $-budget ledger** that stops cleanly instead of surprising you. |
| `score_taxonomy.py` | raw + **answerable-set** accuracy, by-cause failure taxonomy, **paired McNemar + bootstrap**. |
| `eval_budget.py` | Token-cost ledger with a hard cap. |
| `medplum-eval-bundle/` | Reproducible substrate: `docker-compose` (Medplum + Postgres + Redis, MCP enabled) + an open-access MIMIC-IV-on-FHIR-demo loader. |

To point it at a *different* MCP server/domain, swap the tool server + the dataset; the runner, budget
ledger, cap-factorial, and scorer are domain-agnostic.

## The case study & what we found (preliminary)

**Question:** does a catalog of purpose-built FHIR tools beat Medplum's single generic `fhir_request`
tool on single-patient retrieval QA?

**Finding (honest, and a null):** across Claude Opus 4.8 and GPT-5.5, the tool catalog showed **no
statistically significant accuracy advantage** over the single generic tool. An early, promising
+11pp (generic ≈39% → 5-tool catalog ≈50%) **did not survive** the controls above — it folded in a
context-budget/overflow artifact. The only robust, significant effect was the **context cap**:
reference-resolution (`_include`) tools are token-hungry and overflow the default budget (e.g. one arm
overflowed 20/25 medication questions at the stock 32k cap; the cap effect was p=0.0005). The
takeaway: **MCP tool ergonomics on frontier models are dominated by token economics, not tool count** —
and, at least here, a well-designed single tool is plenty. Full write-up + tables: [REPORT.md](REPORT.md).

## Reproduce

```bash
# 1. Stand up the substrate + load the open-access MIMIC-IV-on-FHIR demo (no PhysioNet gate)
cd medplum-eval-bundle && docker compose up -d && bash scripts/load_mimic.sh

# 2. Run the ablation (gpt-5-mini is the judge; project cost before spending)
export OPENAI_API_KEY=...                          # and/or ANTHROPIC_API_KEY for the opus arms
EVAL_GPT_MODEL=gpt-5.5 python run_matrix.py --pilot 3
EVAL_GPT_MODEL=gpt-5.5 python run_matrix.py --n 30 --cap 50 --out-dir runs/curve
python score_taxonomy.py runs/curve
```

## Caveats — this is experimental

- **Preliminary & underpowered** (n=25–30/cell): an ≤8pp effect can't be resolved. "No effect
  *detectable at this power*," not "definitively none."
- **Incomplete in places**: the 8-tool GPT endpoint never ran (API quota exhausted mid-experiment), so
  the strong "too many tools *hurt*" hypothesis is untested; the 1→6 curve is flat.
- **Narrow scope**: one benchmark, single-patient retrieval only (MIMIC-IV-on-FHIR demo, 100 patients).
  Says nothing about multi-patient cohort/aggregate queries.
- Raw GPT-5.5 per-question answers are under `medplum-eval/results/` (with a note on what's real vs the
  errored 8-tool cell). Don't cite the zeroed `_scores.csv` aggregates — the per-question JSON is the
  source of truth. See [REPORT.md](REPORT.md) §8 for the full limitations list.
