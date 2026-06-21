# FHIR Tool-Ablation Eval — Final Honest Synthesis

**Does a clinical agent with a catalog of purpose-built FHIR tools beat Medplum's single generic `fhir_request` tool, on FHIR-AgentBench?**

*Substrate: FHIR-AgentBench (MIMIC-IV-on-FHIR demo, single-patient retrieval QA). Models: Claude Opus 4.8, GPT-5.5. Judge: gpt-5-mini LLM-as-judge (0/1 correctness). Paired stats: McNemar exact + paired bootstrap.*

---

## 1. Bottom line

This is a **null result, and a clean one.** Across two frontier models, a catalog of purpose-built FHIR tools showed **no statistically significant accuracy advantage** over Medplum's single generic `fhir_request` tool on single-patient retrieval. The original headline ("5 tools beat 1 by +11pp") was **confounded** — it folded in a context-budget/overflow artifact and had no paired statistics or controls, and it does not replicate on either model under rigorous testing. The **only robust, significant effect anywhere** is the context cap: `_include`/reference-resolution tools are token-hungry and overflow the default 32k budget, and fixing that — not adding tools — is what moved accuracy.

---

## 2. The arc: how a +11pp win became a null

**RUN-0 (original, confounded):** Generic Medplum tool ≈39% → 5-tool catalog ≈50%, a **+11pp** apparent win. This is the result that motivated the whole pitch. But it was measured **without any of the controls that would let you trust it**:

- **No cap-factorial.** The 32k vs 100k context cap was never varied. The 5-tool arm and the 1-tool arm were not matched on context budget, so any difference in how often each arm *overflowed* the cap got silently scored as a "tool quality" difference.
- **No paired statistics.** Arms weren't compared question-by-question (McNemar / paired bootstrap), so an 11pp gap across ~25–30 questions — which is **well inside the noise floor at that sample size** — was read as signal.
- **No coached-generic control.** Nobody checked whether simply *telling* the generic tool about `_include`/`_elements` (description-only coaching) closes the gap. So "the typed tools are better" was never separated from "the typed tools happen to do something the generic tool was never told to do."
- **No answerable-set accounting.** Overflowed/errored questions weren't separated from genuinely-wrong answers.

**RUN-1 (Opus, rigorous):** Re-ran the same comparison with a cap-factorial, paired McNemar/bootstrap, a coached-generic control, and answerable-set accounting. The +11pp **decomposed into nothing**: the structure-lift attributable to tool *design* was +8pp with a 95% CI of [-0.12, +0.28] and p=0.69 — indistinguishable from zero. Coaching the generic did **literally nothing** (Δ0.00). The lift that *was* real came from the **context cap** (p=0.039 / p=0.0005), not the tools.

**RUN-2 (GPT-5.5, nested staircase):** Re-ran across 1→2→4→5→6 tools on a second, stronger model. The curve is **flat and noisy (0.70–0.83)**, every paired step is null, and the 1-tool generic (0.80) is **never beaten** by the 5-tool catalog (0.767). The effect is model-specific *and* artifact-driven — it survives on neither model.

> An independent crude deterministic re-score of the GPT-5.5 raw answers reproduced these numbers almost exactly (control 0.80, cat4 0.80, validated5 0.79, arm_ref 0.83), so the directional conclusion is **not** an LLM-judge artifact. (Note: the `_scores.csv` / `_paired.json` aggregates in the gptcurve dir are zeroed-out — an *unscored serialization artifact*. Do **not** cite them; the per-question raw JSON files are the source of truth and they back the RESULTS_DATA numbers.)

---

## 3. All the numbers

### RUN-1 — Opus (claude-opus-4-8), rigorous, n=25 paired/cell

**MEDICATION slice** (answers spanning MedicationRequest → Medication):

| arm | tools | accuracy |
|---|---|---|
| control | 1 generic | 0.64 |
| control_include | 1 generic + `_include` coaching | 0.64 |
| arm_ref | 6 (incl. `resolve_references`) | 0.72 |

**Paired comparisons:**

| comparison | Δ | 95% CI | p | verdict |
|---|---|---|---|---|
| coaching-lift (control → control_include) | **0.00** | — | 1.0 | **NULL** — coaching did nothing |
| structure-lift (control_include → arm_ref) | +0.08 | [-0.12, +0.28] | 0.69 | **NULL** |
| total-lift (control → arm_ref) | +0.08 | [-0.08, +0.24] | 0.625 | **NULL** |
| **cap effect on control (32k → 100k)** | **+0.28** | — | **0.039** | **SIGNIFICANT** |
| **cap effect on arm_ref (32k → 100k)** | **+0.56** | — | **0.0005** | **HIGHLY SIGNIFICANT** |

> arm_ref @ 32k cap **overflowed on 20/25 questions** (scoring just 0.16) — its `_include` bundles blow the stock context budget.

**REPRESENTATIVE slice (n=25):** control 0.72 → arm_full8 (8 tools) 0.76, Δ +0.04, p=1.0 — **NULL**.

### RUN-2 — GPT-5.5 (gpt-5.5-2026-04-23), nested dose-response staircase, rep slice n=30

| tools | arm | raw acc | answerable acc | note |
|---|---|---|---|---|
| 1 (generic) | control | **0.80** | 0.80 | generic alone already near the top |
| 2 | cat2 | 0.70 | 0.875 | 6/30 overflowed |
| 4 | cat4 | 0.767 | 0.767 | |
| 5 | validated5 | 0.767 | 0.793 | 1 overflow |
| 6 (+`resolve_references`) | arm_ref | **0.833** | 0.833 | |
| 8 | arm_full8 | **INCOMPLETE** | — | OpenAI quota exhausted; 30/30 errored, agent never ran |

**Paired steps (ALL NULL):**

| step | Δ | p | verdict |
|---|---|---|---|
| 1 → 2 | -0.10 | 0.375 | NULL |
| 2 → 4 | +0.067 | 0.625 | NULL |
| 4 → 5 | 0.00 | 1.0 | NULL |
| 5 → 6 | +0.067 | 0.625 | NULL |

Curve is flat/noisy across 0.70–0.83 with no significant climb.

---

## 4. The diminishing-returns question ("do too many tools hurt?")

**The honest answer: the 1→6 curve is flat; the strong form is untestable.**

- **What the curve shows (1→6 tools):** flat and noisy, 0.70–0.83, every paired step null (p ≥ 0.375). Adding tools buys no monotonic benefit.
- **The 5-vs-1 reality:** control (1 tool) = **0.80** ≥ validated5 (5 tools) = **0.767**. The 5-tool arm did **not** beat the 1-tool arm — it was, if anything, marginally worse. This directly inverts the RUN-0 premise.
- **The honest caveat — do NOT claim "too many tools hurt":** the 8-tool endpoint is **incomplete**. All 30 arm_full8 records are `RateLimitError: exceeded your current quota` — **0 LLM calls, the 8-tool agent never executed a single question.** There is no 8-tool datapoint. The flat 1→6 trend *refutes a monotonic-benefit story* and is *weakly consistent* with "more tools don't help," but it is **not** evidence for an overload cliff. Claiming "too many tools hurt" would be inventing the one datapoint we don't have. The disciplined statement is: **"flat through 6 tools; the 8-tool endpoint failed to run."**

---

## 5. The one real finding: context budget / `_include` overflow

The only robust, significant lever in the entire experiment is the **context cap**, isolated by the Opus cap-factorial:

- **control, 32k → 100k:** 0.36 → 0.64, Δ **+0.28**, **p=0.039** (significant).
- **arm_ref, 32k → 100k:** 0.16 → 0.72, Δ **+0.56**, **p=0.0005** (highly significant).
- At the 32k cap, **arm_ref overflowed 20/25 questions** — its `resolve_references`/`_include` bundles are payload-heavy and blow the stock budget.
- On GPT-5.5, cat2 overflowed 6/30 and validated5 overflowed 1/30 — same mechanism, smaller blast radius.

**The mechanism:** what looked like "purpose-built tools make the agent smarter" was substantially "the larger-context arm dodged a context-overflow cliff the harness imposed." **FHIR tool ergonomics on frontier models are dominated by token economics, not tool count.** `_include`/reference-resolution is a concrete context-budget trap: if you give an agent FHIR tools, design for payload size, not tool-surface area.

---

## 6. Illustrative traces (GPT-5.5, control vs arm_ref)

**Case A — both correct, generic is *more* detailed.**
*Q: "What was the admission type during the first hospital admission for patient 10021118?" (true: URGENT)*
- **control (1 tool):** "urgent — Encounter ID 24490144, `Encounter.priority`, code UR — urgent."
- **arm_ref (6 tools):** "urgent (`UR`)."
- Both correct; the generic tool actually returned *richer* structured detail.

**Case B — both correct, typed tools surface different breadth.**
*Q: "Has any lab test been performed on patient 10016150 in 2142?" (true: Yes)*
- **control:** "Yes — WBC 9.4 K/uL, 2142-05-11, final."
- **arm_ref:** "Yes — microbiology susceptibilities on 2142-05-21 (AMPICILLIN Resistant, TOBRAMYCIN Resistant, …)."
- Both answer "Yes" correctly. arm_ref's typed `search_observations` surfaces more varied result categories, but the *correctness* is identical — the difference is elaboration, not accuracy.

**Case C — identical answer (the modal pattern).**
*Q: "Has patient 10037975 been given a creatinine, urine test since 15 months ago?" (true: Yes)*
- **control & arm_ref:** both return "Yes — Creatinine, Urine, 2185-01-19T11:22, 52 mg/dL, final." Functionally identical.

**Takeaway from the traces:** typed tools don't unlock new *correctness* — they surface additional or differently-categorized data. For strong models on well-structured FHIR, the generic tool is already near-optimal, which is exactly why the curve is flat.

---

## 7. What it means + recommendation

**Do NOT pitch Medplum on "your generic tool is worse, buy our tool catalog."** The data says the opposite. For strong models the single generic tool is plenty, and GPT-5.5 with the bare generic (0.80) *beat* Opus with the bare generic (0.72). If anything, this result **validates Medplum's minimalist single-tool design.** Pitching a confounded +11pp you've already disproven is a fast way to get eaten alive by a maintainer who reads your raw data.

**Instead, ship the methodology and the overflow finding as the contribution.** The accuracy result is null, but the *experiment* is the asset. You built a paired-McNemar + bootstrap + cap-factorial harness that **catches a confound a naive eval (RUN-0, and frankly most vendor eval blog posts) would have shipped as a win.** The reframed, defensible thesis:

> *"Across two frontier models on single-patient FHIR retrieval, a catalog of purpose-built FHIR tools showed no statistically significant accuracy advantage over Medplum's single generic `fhir_request` tool. The only robust effect was the model's context budget — `_include`/reference-resolution tools actively hurt because they overflow the default cap. FHIR agent ergonomics are governed by token economics, not tool count."*

That is true, specific, useful to a Medplum maintainer, and a strong hiring signal — "I disproved my own promising result and traced it to a context-budget confound" beats "I found tools help."

**Keep the broader thesis cleanly separate.** This eval tests the **narrowest slice**: single-patient retrieval. The founder's actual thesis — agent **aggregation/cohort** over FHIR (the second-badge / aggregate-ABAC cell) — is **explicitly out of scope of FHIR-AgentBench** (it excludes multi-patient questions). So a flat single-patient curve says **nothing** about the cohort/aggregate-ABAC angle. Don't let this null get over-generalized into "the agent-layer thesis is dead." It isn't tested here.

---

## 8. Limitations (state these loudly and first)

- **Underpowered.** n=25–30/cell. An 8pp effect cannot be resolved at this sample size (MDE would need ~n≥150). The honest claim is "no effect *detectable at this power*; the effect, if any, is ≤8pp and dwarfed by the cap effect" — **not** "tools definitively don't help."
- **8-tool endpoint incomplete.** All 30 arm_full8 records are quota errors; the 8-tool agent never ran. The strong form of the diminishing-returns hypothesis is untestable. Disclose this before anyone finds the RateLimitErrors.
- **Single benchmark, single-patient.** FHIR-AgentBench only, MIMIC-IV-on-FHIR demo (100 patients), single-patient retrieval QA. No cohort/aggregate coverage.
- **Both API accounts hit quota mid-experiment.** Anthropic, then OpenAI. Quota/cost management was the real operational bottleneck, and the `resolve_references`/`_include` cells are token-expensive on any model — itself part of the finding.
- **LLM-as-judge ground truth.** gpt-5-mini judge; *mitigated* by an independent crude deterministic re-score that reproduced the GPT-5.5 curve, so the directional conclusion is not judge-sensitive.
- **Opus raw data lost** with its torn-down box (reproducibility hole); the GPT-5.5 raw answers survive at `medplum-eval-results/runs/gptcurve/mcp.*.gpt-5.5-2026-04-23.rep.c100k.json` and carry the load. The `_scores.csv` / `_paired.json` in that dir are zeroed-out unscored artifacts — **do not cite them.**