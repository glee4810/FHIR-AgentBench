# FHIR tool-ablation eval — ALL RESULTS (authoritative numbers)

Eval: does giving a clinical agent purpose-built FHIR tools beat Medplum's single generic
`fhir_request` tool? Substrate: FHIR-AgentBench (MIMIC-IV-on-FHIR demo, single-patient retrieval QA),
held-out test split. Judge: gpt-5-mini, LLM-as-judge correctness (0/1). Paired stats: McNemar exact +
paired bootstrap on per-question deltas (arms share question_ids per sample). Accuracy = raw correctness.

The catalog (8 purpose-built read-only FHIR tools): get_patient_chart($everything), search_observations,
search_fhir, read_resource, list_search_params, resolve_references(_include), search_encounters,
search_procedures. Baselines: control=fhir_request (Medplum's exact shipped generic), control_include=
generic + _include coaching (description only), c0=fhir_request_frugal (generic + _elements coaching).
Presets: cat2=2 tools, cat4=4 tools, validated5=5 tools, arm_ref=6 (validated5+resolve_references),
arm_full8=8 (all).

---

## RUN 0 — Original result (prior session, CONFOUNDED — no rigorous controls)
Generic Medplum MCP tool (~1 tool) **≈39%** → purpose-built catalog (~5 tools) **≈50%**  (≈ **+11pp**).
This is the "1 generic tool vs 5 purpose-built tools, 5 did meaningfully better" result that motivated
everything. BUT it was measured WITHOUT: the cap-factorial (32k vs 100k context cap), paired stats,
answerable-set accounting, or a coached-generic control. So +11pp was CONFOUNDED — later shown to
fold in a context-cap/overflow artifact and to not be robust.

## RUN 1 — Opus (claude-opus-4-8), RIGOROUS, n=25 paired per cell
MEDICATION slice (questions whose answer spans MedicationRequest->Medication):
| arm | tools | accuracy |
|---|---|---|
| control | 1 generic | 0.64 |
| control_include | 1 generic + _include coaching | 0.64 |
| arm_ref | 6 (incl. resolve_references) | 0.72 |
Paired:
- coaching-lift (control -> control_include): Δ **0.00**, p=1.0 — coaching the generic did NOTHING.
- structure-lift (control_include -> arm_ref): Δ **+0.08**, 95% CI [-0.12,+0.28], p=0.69 — **NOT significant**.
- total-lift (control -> arm_ref): Δ +0.08, CI [-0.08,+0.24], p=0.625 — **NOT significant**.
- cap effect on control (32k -> 100k): 0.36 -> 0.64, Δ **+0.28**, p=**0.039** — SIGNIFICANT.
- cap effect on arm_ref (32k -> 100k): 0.16 -> 0.72, Δ **+0.56**, p=**0.0005** — HIGHLY SIGNIFICANT.
  (arm_ref @ 32k cap OVERFLOWED on 20/25 questions — its _include bundles blow the stock context cap.)
REPRESENTATIVE slice (n=25): control 0.72 -> arm_full8 (full 8-tool) 0.76, Δ +0.04, p=1.0 — NOT significant.

## RUN 2 — GPT-5.5 (gpt-5.5-2026-04-23), the NESTED dose-response staircase, rep slice n=30
| tools | arm | raw acc | answerable acc | note |
|---|---|---|---|---|
| 1 (generic) | control | **0.80** | 0.80 | generic alone already near the top |
| 2 | cat2 | 0.70 | 0.875 | 6/30 overflowed |
| 4 | cat4 | 0.767 | 0.767 | |
| 5 | validated5 | 0.767 | 0.793 | 1 overflow |
| 6 (+resolve_references) | arm_ref | **0.833** | 0.833 | |
| 8 | arm_full8 | **INCOMPLETE** | — | OpenAI quota exhausted mid-run; all 30 errored |
Paired steps (ALL NULL): 1->2 Δ-0.10 (p=0.375); 2->4 Δ+0.067 (p=0.625); 4->5 Δ0.00 (p=1.0);
5->6 Δ+0.067 (p=0.625). Curve is FLAT/noisy 0.70-0.83 with no significant climb.

---

## Cross-cutting facts for synthesis
1. The original RUN-0 +11pp (1-tool vs 5-tool, "5 meaningfully better") does NOT survive rigorous testing
   on EITHER model: opus structure-lift +8pp (NS); gpt curve flat (control(1)=0.80 >= validated5(5)=0.767,
   i.e. 5 did NOT beat 1).
2. The ONLY robust/significant effect anywhere = CONTEXT CAP (opus): 32k->100k helped both arms; the
   _include tool catastrophically overflows at 32k (0.16, 20/25 overflow). Tool "benefit" is entangled
   with context budget = the Finding-B / cap-dodging confound the methodology was built to expose.
3. GPT-5.5 with the bare generic tool (0.80) BEAT opus with the bare generic (0.72) -> gpt-5.5 is stronger
   at raw FHIR query construction, so typed tools buy it even less. (Note: an earlier gpt-5.2 smoke had
   GPT struggling on the generic; gpt-5.5 is much better than 5.2.)
4. Coaching the generic about _include (control_include) = 0 lift on opus.

## Honest conclusion
Purpose-built FHIR tools show NO robust, significant accuracy advantage over Medplum's generic tool on
single-patient retrieval, across two frontier models. The headline "tools help" from the original
confounded run was largely a context-budget/overflow artifact + chance, not a tool-design win. The real
significant lever is the context cap; reference-resolution (_include) tools are context-hungry and
overflow the default budget. For strong models the generic tool is plenty.

## Limitations / caveats (state them)
- Underpowered: n=25-30/cell; ~8pp effects can't be resolved (would need ~n>=150).
- 8-tool GPT endpoint INCOMPLETE (OpenAI quota exhausted) — can't confirm whether 8 tools specifically
  drop vs plateau (the strong form of the diminishing-returns hypothesis), but the 1-6 trend is flat.
- Single benchmark (FHIR-AgentBench), single-patient retrieval only, MIMIC-on-FHIR demo (100 patients).
- BOTH API accounts (Anthropic, then OpenAI) hit quota mid-experiment — quota/cost management was the
  real operational bottleneck, and the resolve_references _include cells are token-expensive on any model.
- Raw gptcurve answer data: medplum-eval-results/runs/gptcurve/*.json (control/cat2/cat4/validated5/arm_ref
  are real; arm_full8 is all-errors). Opus raw data lost with its torn-down box; numbers above are
  from its scoring.
