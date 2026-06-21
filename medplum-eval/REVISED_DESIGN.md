# Revised eval design (post-Codex REVISE, 2026-06-21)

> ⚠️ **SUPERSEDED — this is the pre-run DESIGN/hypothesis doc, not the result.** The "HEADLINE CLAIM"
> below (that `resolve_references` produces a paired, CI-separated lift) did **NOT** hold: the actual
> outcome was a **NULL** — no significant tool advantage on either model — with the only robust effect
> being the context-cap/`_include` overflow. See **[../REPORT.md](../REPORT.md)** for the real findings.
> Kept only as a record of the planned design.

THESIS: Medplum's shipped agentic surfaces ($ai + MCP) expose ONE generic `fhir_request` tool.
Purpose-built read-only FHIR tools beat it on FHIR-AgentBench (MIMIC-IV-on-FHIR, single-patient
retrieval QA). Goal: contribution + hire at Medplum.

CATALOG (8 purpose-built tools): get_patient_chart($everything), search_observations, search_fhir,
read_resource, list_search_params, resolve_references(_include), search_encounters, search_procedures.
BASELINES/ARMS: control=fhir_request (byte-for-byte Medplum), c0=fhir_request_frugal (coached _elements),
control_include=fhir_request_include (coached _include — the ATTRIBUTION control).

REVISED MATRIX (~$88, hard $100 cap, opus=claude-opus-4-8, gpt=gpt-5.2):
- Representative curve (opus×MCP, random N=30): control → validated5 → +resolve_references → +search_encounters → +search_full8 ; + c0
- MEDICATION SLICE (opus×MCP, stratified ~25 medication-reference questions): control vs control_include vs arm_ref  [the clean attribution test]
- 2×2 cap-factorial (opus×MCP, rep): {control, arm_ref} × {32k, 100k input cap}
- Surface validation ($ai×gpt + gpt×MCP, rep): control, arm_full8
- Paired stats: McNemar + paired bootstrap on per-question deltas (arms see same questions)
- Metrics: raw acc + answerable-set acc + by-cause failure taxonomy, reported side by side.

CODEX REVISE (already folded in): (1) coached-generic-_include arm to isolate discoverability vs capability;
(2) stratified medication slice for power (was ~7 questions, luck); (3) complete 2×2 cap-factorial;
(4) paired stats not independent Wilson; (5) don't let answerable-set hide tool-avoided failures;
(6) $ai = external validity not core proof.

HEADLINE CLAIM: resolve_references (no surveyed FHIR MCP server ships it) produces a paired,
CI-separated lift on the medication-reference slice OVER BOTH the raw generic AND the _include-coached
generic — i.e. a real affordance win, not just discoverability.
