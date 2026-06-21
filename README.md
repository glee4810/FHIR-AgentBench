# Evaluating MCP tool surfaces for agents — an experimental FHIR / Medplum eval

> ⚗️ **Experimental / exploratory work.** This is a fork of
> [glee4810/FHIR-AgentBench](https://github.com/glee4810/FHIR-AgentBench) carrying a rig for a simple but
> slippery question: **when you expose tools to an LLM agent through an
> [MCP](https://modelcontextprotocol.io) server, do those tools actually make the agent better — and how
> would you *honestly* measure that?** The worked example is FHIR / [Medplum](https://www.medplum.com),
> but the harness and the lessons generalize to evaluating *any* MCP tool surface. Treat the numbers as
> preliminary; see [Reproducibility status](#where-this-was-actually-run--reproducibility-status).

## TL;DR

- **Question:** does a catalog of purpose-built FHIR tools beat Medplum's single generic `fhir-request`
  tool on single-patient retrieval QA, across **Claude Opus 4.8** and **GPT-5.5**?
- **Result — an underpowered null.** No *detectable* tool advantage (MDE at this n is ~34–46pp), and the
  early, promising **+11pp** (generic ≈39% → 5-tool catalog ≈50%) turned out to be a **context-budget
  confound**, not a tool win. Read this as "no big tool win, and the headline was an artifact" — *not*
  "tools provably don't help."
- **The one robust effect** is the model's **context budget**: `_include`/reference-resolution tools are
  token-hungry and overflow the default cap (one Opus arm overflowed 20/25 questions at 32k). FHIR agent
  ergonomics are dominated by **token economics, not tool count**.
- **The contribution is the methodology** — a paired-stats + cap-factorial + budget-ledger harness that
  catches a confound a naive eval (and most vendor eval blog posts) would ship as a win.
- ⚠️ **Most numbers were produced on now-torn-down EC2 boxes and are not recomputable from committed
  data.** Be precise about what survives — see [below](#where-this-was-actually-run--reproducibility-status).
- Full write-up, tables, paired stats, and limitations: **[REPORT.md](REPORT.md)**.

## What we're actually trying to do

"Add an MCP server with N purpose-built tools and the agent gets smarter" is the kind of claim that's
**easy to assert and easy to fool yourself about.** This project is an attempt to build the eval
*correctly* — to measure whether an MCP tool surface helps while controlling for the things that usually
masquerade as a tool win:

- **Context budget.** More/richer tools return bigger payloads; bigger payloads overflow the model's
  context window. A "tool win" is often just "this arm overflowed less." → we vary the context cap as an
  explicit factor (a **2×2 cap-factorial**) to separate *reasoning gain* from *cap-dodging*.
- **Prompt vs. structure.** Is the typed tool better, or did we just *tell* the agent something we never
  told the generic tool? → a **coached-generic control** (generic tool + the same coaching in its
  description only).
- **Noise.** The original +11pp headline (and the residual ~8pp tool-structure lift after controls) sits
  across only 25–30 questions/cell — inside the noise floor (our MDE at this n is **~34–46pp**, REPORT §9.2).
  → **paired McNemar + bootstrap** on per-question deltas, not eyeballed averages.
- **Hidden failures.** Overflows / rate-limits / no-answers ≠ wrong answers. → an **answerable-set
  accuracy** + a by-cause failure taxonomy.

The reusable artifact is this methodology, not any one number.

## The harness (generalizes to any MCP eval)

| File | Role |
|---|---|
| `agent/mcp_agent.py` | Agent that retrieves through an **MCP server** — the only variable across arms is the advertised tool surface. |
| `agent/ai_agent.py` | Variant that routes completions through a server-side LLM proxy (here, Medplum's in-FHIR `$ai` op) — test the lift on the platform's *own* agentic surface, not just an external client. |
| `treatment_mcp_server.py` | **Catalog-driven** FastMCP server: one server, `TOOL_SUBSET`-selectable arms. The baseline "generic" arm is a **local FastMCP re-implementation** whose *description string* is copied byte-for-byte from Medplum's shipped `fhir-request` tool, proxying the same FHIR REST surface — it is **not** the platform's production MCP tool path. (The smoke test confirms Medplum advertises a tool playing the same role — named `fhir-request` with a hyphen; our local control registers `fhir_request` with an underscore — so it's a description-matched reimplementation, not the identical tool.) Typed tools toggle in on top. |
| `run_matrix.py` | Parameterized ablation runner: per-cell tool subset, a **nested dose-response staircase** (1→2→4→… tools), the cap-factorial, and a **hard $-budget ledger** that stops cleanly instead of surprising you. |
| `score_taxonomy.py` | raw + **answerable-set** accuracy, by-cause failure taxonomy, **paired McNemar + bootstrap**. |
| `robustness_analysis.py` | Judge-free post-hoc pass (no re-run, no spend): deterministic re-score, minimum-detectable-effect (MDE) power sim, Holm-Bonferroni. |
| `eval_budget.py` | Token-cost ledger with a hard cap. |
| `medplum-eval-bundle/` | Reproducible substrate: `docker compose` (Medplum + Postgres + Redis, MCP enabled) + an open-access MIMIC-IV-on-FHIR-demo loader. |

To point it at a *different* MCP server/domain, swap the tool server + the dataset; the runner, budget
ledger, cap-factorial, and scorer are domain-agnostic.

## The case study & what we found (preliminary)

**Finding (honest, and a null):** across Claude Opus 4.8 and GPT-5.5, the tool catalog showed **no
statistically significant accuracy advantage** over the single generic tool. The early **+11pp** (generic
≈39% → 5-tool catalog ≈50%) **did not survive** the controls above — it folded in a
context-budget/overflow artifact, has no paired statistics, and does not replicate on either model. The
only robust, significant effect was the **context cap**: reference-resolution (`_include`) tools overflow
the default budget (e.g. one Opus arm overflowed 20/25 medication questions at the stock 32k cap;
p=0.0005, the one comparison that survives Holm-Bonferroni). Takeaway: **MCP tool ergonomics on frontier
models are dominated by token economics, not tool count** — and, at least here, a well-designed single
tool is plenty. Full write-up + tables + paired stats: **[REPORT.md](REPORT.md)**.

> Externally corroborated: the FHIR-AgentBench paper itself reports ~3M-token full records and that
> "naive loading consistently failed," and the [WSO2 FHIR MCP server](https://github.com/wso2/fhir-mcp-server)
> recently shipped FHIRPath response-*filtering* (payload reduction) — both point at token economics, not
> tool count, as the bottleneck.

## Setup (Docker) — verified boot path

The whole substrate is a `docker compose` bundle in [`medplum-eval-bundle/`](medplum-eval-bundle/):
self-hosted **Medplum** (server + Postgres + Redis, MCP enabled) loaded with the open-access
**MIMIC-IV-on-FHIR demo** (100 real de-identified ICU patients, ODbL — no PhysioNet credentialing). The
boot/auth/FHIR/MCP path below was **smoke-tested on macOS + Docker Desktop 28.4.0 on 2026-06-21**
([`medplum-eval-bundle/SMOKE_TEST.md`](medplum-eval-bundle/SMOKE_TEST.md)).

**Prereqs:** Docker + Compose v2, Python 3.10+, **`wget`** (the MIMIC loader needs it — `brew install wget`
on macOS), and an `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`. The judge is OpenAI `gpt-5-mini`, so an
OpenAI key is needed even for the opus arms. **Keys are read from environment variables only** (litellm) —
the MCP ablation path does *not* read `config.yml` (that's for the upstream GCP agents). `requirements.txt`
includes the MCP SDK (`mcp`); `pip install -r requirements.txt` covers the harness.

```bash
# 0. harness deps
pip install -r requirements.txt

# Steps 1–2 are SMOKE-VERIFIED on this laptop Docker path. Steps 3–4 are NOT (see "Where this was
# actually run") — the load + ablation only ever ran on EC2; run them here at your own (budget) risk.

# 1. [verified] Stand up Medplum (postgres:16 + redis:7 + medplum/medplum-server:latest, MCP enabled).
#    First boot runs DB migrations — verified healthy in ~75s on a laptop.
cd medplum-eval-bundle && docker compose up -d
for i in $(seq 1 40); do curl -s http://localhost:8103/healthcheck | grep -q '"ok":true' && break; sleep 5; done
#    -> {"ok":true,"version":"5.1.21-...","postgres":true,"redis":true}
#    If it never goes healthy (~2 min), check: docker compose logs -f medplum-server

# 2. [verified] bare-PKCE admin token + confirm the MCP server advertises the generic baseline tool
python3 scripts/get_token.py | head -c 16   # 695-char JWT for admin@example.com / medplum_admin
#    GET 401 unauth / POST tools/list 200 -> {search, fetch, fhir-request}  (fhir-request = the baseline arm)

# 3. [⚠️ NOT smoke-verified on this path] Load the MIMIC-IV-on-FHIR demo (8 gold resource types; needs
#    wget; ~1h EC2-measured, slower on a laptop — use W=4 there; idempotent — PUTs UUID ids so the
#    benchmark's true_fhir_ids match). It hard-errors (not silent) if the download yields 0 files.
bash scripts/load_mimic.sh
#    sanity: curl -s "http://localhost:8103/fhir/R4/Patient?_summary=count" -H "Authorization: Bearer $(python3 scripts/get_token.py)"  # ~100
cd ..

# 4. [⚠️ NOT smoke-verified on this path] Project cost first (a real run is a decision, not a surprise),
#    then run the full ablation matrix + score. Run from the REPO ROOT (steps 1–3 cd'd into the bundle).
export ANTHROPIC_API_KEY=...   # opus arms      export OPENAI_API_KEY=...   # gpt arms + the judge
EVAL_GPT_MODEL=gpt-5.5-2026-04-23 python run_matrix.py --pilot 3            # prints projected $, exits
EVAL_GPT_MODEL=gpt-5.5-2026-04-23 python run_matrix.py --n 25 --n-med 40 --cap 100 --out-dir runs/tier1
python score_taxonomy.py runs/tier1            # raw + answerable-set acc, by-cause taxonomy, paired McNemar
python robustness_analysis.py medplum-eval/results   # judge-free re-score + MDE + Holm (no re-run, no spend)
```

**cwd note:** `get_token.py` / `load_mimic.sh` live in `medplum-eval-bundle/scripts/` (steps 1–3 run from
inside the bundle); `run_matrix.py` / `score_taxonomy.py` / `robustness_analysis.py` run from the **repo
root** (step 4). `run_matrix.py` skips any cell whose output JSON already exists in `--out-dir`, so use a
fresh `--out-dir` to re-run from scratch.

Knobs (all env-overridable, defaults shown): `EVAL_RUN=gpt` (or `opus` — selects the GPT-5.5 staircase
[RUN-2] vs the Opus medication-slice + cap-factorial [RUN-1]; run each into a **separate** `--out-dir` and
score separately), `EVAL_OPUS_MODEL=claude-opus-4-8`, `EVAL_GPT_MODEL=gpt-5`, `EVAL_JUDGE_MODEL=gpt-5-mini`,
`EVAL_STOCK_CAP=32000` / `EVAL_RAISED_CAP=100000` (the cap-factorial), `EVAL_WORKERS=6`. `run_matrix.py`
spawns `treatment_mcp_server.py` per cell with the right `TOOL_SUBSET` (preset
`control`/`cat2`/`cat4`/`validated5`/`arm_ref`/`arm_full8`, or a comma-list) and points the agent at it via
`MEDPLUM_MCP_URL` (the local treatment server at `127.0.0.1:8765/mcp`) — **not** Medplum's own
`8103/mcp/stream` surface, which is only the smoke-test target. Don't override `MEDPLUM_MCP_URL` by hand or
you'll silently measure the wrong server. `--cap` is a **dollar budget** enforced by a shared ledger: it
stops submitting new questions the moment the cap trips, so overspend is bounded to the in-flight batch
(≤ `EVAL_WORKERS` questions) — hard *between and within* cells, not just between them. Two footguns:
(1) the judge defaults to `gpt-5-mini` — if that model id 404s for your account, `score_taxonomy` now
**fails closed** (raises if a cell's judging totally fails, and excludes judge-errored questions from the
denominator rather than scoring them 0), so set `EVAL_JUDGE_MODEL` if needed; (2) the cost projector has
no built-in rate for `gpt-5.5`, so it
falls back to `gpt-5`'s rate — set `EVAL_RATES` for a precise projection.

## Where this was actually run — reproducibility status

**The eval results in [REPORT.md](REPORT.md) were produced on AWS EC2, not on this laptop Docker path, and
most of them are not recomputable from committed artifacts.** The full pipeline (the ~1h MIMIC load + the
multi-hour Opus and GPT-5.5 ablation runs) ran on ephemeral EC2 boxes (`t3.xlarge`, us-east-2) that have
since been torn down. Be precise about what survives:

**Reproducible from this repo:**
- The Docker **boot path** (steps 1–2): containers up, Medplum healthy, bare-PKCE token, FHIR read/write
  round-trip, MCP advertises the generic `fhir-request` tool — smoke-verified on macOS
  ([`medplum-eval-bundle/SMOKE_TEST.md`](medplum-eval-bundle/SMOKE_TEST.md)).
- The **GPT-5.5 deterministic re-score curve + overflow taxonomy** — recomputed live from the committed
  per-question JSON by `robustness_analysis.py` (REPORT §9.1).

**NOT reproducible (reconstructed from lost runs):**
- **All LLM-judge accuracies.** The committed per-question JSON carries agent *answers* but **no
  correctness labels** — the judge scoring run was lost with its box.
- **All paired p-values, and the entire Opus run** (including the cap-factorial and the headline "only
  robust effect," cap-on-`arm_ref` p_holm=0.005). There is **no committed Opus data at all**.

So treat the bundle as a faithful, boot-smoke-verified **recipe** of the EC2 environment — not an
independently reproduced result. The load-bearing, verifiable claims are the **null** and the **flat
deterministic curve**; the cap finding and the absolute accuracies are credible-but-unverifiable.
Re-running steps 3–4 on Docker and committing the raw per-question + judge labels is the open
reproducibility item. (`robustness_analysis.py` prints a provenance banner that separates computed-live
from reconstructed numbers.)

## Caveats — this is experimental

- **Badly underpowered** (n=25–30/cell): the minimum detectable effect at this n is **~34–46pp** (REPORT
  §9.2), so a commercially-decisive 5–10pp lift is structurally invisible. The honest null is "no tool
  effect larger than ~the MDE," not "definitively none."
- **Uncalibrated judge**: gpt-5-mini LLM-as-judge, no human gold set, κ unmeasured. Mitigated — not
  replaced — by a judge-free deterministic re-score that reproduces the same flat curve (REPORT §9.1).
- **Family-wise correction matters**: after Holm-Bonferroni over all 10 comparisons, **only the
  context-cap overflow effect survives** (p_holm=0.005); every tool-count/tool-design comparison is null
  (REPORT §9.3). No pre-registration; single seed/cell.
- **Incomplete in places**: the 8-tool GPT endpoint never ran (API quota exhausted mid-experiment), so the
  strong "too many tools *hurt*" hypothesis is untested; the 1→6 curve is flat.
- **Narrow scope**: one benchmark, single-patient retrieval only (MIMIC-IV-on-FHIR demo, 100 patients).
  Says nothing about multi-patient cohort/aggregate queries.
- **Control is a faithful re-implementation**, not Medplum's production MCP tool path (description copied
  byte-for-byte; see the harness table).

Full limitations + reproducibility status: [REPORT.md](REPORT.md) §1 and §8–§9.

## Repo layout (this fork's additions)

```
treatment_mcp_server.py      # catalog-driven FastMCP server (the arms)
run_matrix.py                # ablation runner: staircase + cap-factorial + $-budget ledger
score_taxonomy.py            # answerable-set accuracy + by-cause taxonomy + paired McNemar/bootstrap
robustness_analysis.py       # judge-free re-score + MDE power sim + Holm-Bonferroni
eval_budget.py               # token-cost ledger with a hard cap
agent/mcp_agent.py           # agent that retrieves via an MCP server
agent/ai_agent.py            # agent that routes via Medplum's in-FHIR $ai op
medplum-eval-bundle/         # docker compose substrate + MIMIC loader + smoke test
  ├── docker-compose.yml
  ├── scripts/{get_token,bulk_load,load_mimic}
  ├── README.md              # Docker runbook
  └── SMOKE_TEST.md          # verified boot path
medplum-eval/                # design docs, results data, robustness output
  ├── results/               # committed GPT-5.5 per-question answers (no correctness labels)
  └── ROBUSTNESS_ANALYSIS.txt
REPORT.md                    # full honest synthesis (numbers, stats, limitations)
```

The original benchmark's code (dataset construction, the upstream agent implementations, `evaluation_metrics.py`,
`fhir_client.py`, etc.) is unchanged from upstream and documented in their materials.

## Attribution & citation

This repository is a **fork of [glee4810/FHIR-AgentBench](https://github.com/glee4810/FHIR-AgentBench)**
(licensed **CC BY 4.0** — see [`LICENSE`](LICENSE)). The benchmark, dataset, and upstream agent
implementations are the work of the original authors — a joint research effort between **Verily Life
Sciences, KAIST, and MIT**. Everything in [Repo layout](#repo-layout-this-forks-additions) above is this
fork's addition on top of their work.

If you use FHIR-AgentBench, cite the paper:

> Gyubok Lee, Elea Bach, Eric Yang, Tom Pollard, Alistair Johnson, Edward Choi, Yugang Jia, Jong Ha Lee.
> **"FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering."**
> ML4H 2025. arXiv:[2509.19319](https://arxiv.org/abs/2509.19319).

```bibtex
@inproceedings{lee2025fhiragentbench,
  title     = {FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering},
  author    = {Lee, Gyubok and Bach, Elea and Yang, Eric and Pollard, Tom and
               Johnson, Alistair and Choi, Edward and Jia, Yugang and Lee, Jong Ha},
  booktitle = {Proceedings of Machine Learning for Health (ML4H)},
  year      = {2025},
  eprint    = {2509.19319},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2509.19319}
}
```
