# Expanded Ablation Catalog — FHIR-Agent Tools (FINAL SPEC)

**Substrate:** FHIR-AgentBench (MIMIC-IV-on-FHIR, single-patient retrieval QA). **Target:** Medplum `$ai`/Spaces + MCP (FHIR-purist, suggest-don't-act, AccessPolicy/AuditEvent-guarded). Every tool is a thin read-only `_fhir_get` wrapper over an existing Medplum FHIR operation — no proprietary shapes, no projections, no infra Medplum lacks.

> **Honest framing up front.** The benchmark's *answerable demand is concentrated*, not spread. Re-measuring `true_fhir_ids` over the N=2931 mappable rows: Observation 39.1%, `{MedicationRequest, Medication}` 17.0% (the **only** reference-spanning answer in the entire set), Encounter 13.4%, Procedure 3.7%, Condition 1.2%. You cannot draw a smooth 12-point dose-response curve from this — the demand supports ~4 meaningful points. Stuffing in 14 tools to "show the curve flattens" manufactures the flat region out of duplicates and below-noise slices instead of measuring it. The spec below is the **pruned, post-vet catalog**.

---

## 1. The locked ablation catalog

**Total: 8 tools** — 5 validated baseline + 3 additions. Run as **4 cumulative arms + 1 orthogonal efficiency arm**.

| Step | Tool | Medplum op (file) | Read-only | Justification (failure-mode / field-precedent) | CORE/TAIL |
|------|------|-------------------|-----------|--------------------------------------------------|-----------|
| Base | `get_patient_chart` | `Patient/{id}/$everything` (`operations/patienteverything.ts`) | ✅ | Validated baseline; broadest single read (flagged least-privilege-risky — bounded alternatives are the typed/`_include` searches below). | CORE |
| Base | `search_observations` | `GET /Observation?` typed | ✅ | Validated; addresses the 39.1% Observation slice. `code:text` matching already handles question-supplied category text. | CORE |
| Base | `search_fhir` | `GET /{type}?` generic | ✅ | Validated; generic fallback. Already covers Condition/Procedure/Encounter generically. | CORE |
| Base | `read_resource` | `GET /{type}/{id}` | ✅ | Validated; single-resource retrieval. | CORE |
| Base | `list_search_params` | `GET /metadata` (CapabilityStatement) | ✅ | Validated introspection. Field-precedent: jcafazzo `get_capability_statement` (`fhir_mcp_server.py:125`), xSoVx `fhir.capabilities` (`fhir-tools.ts:27`). | CORE |
| 1 | `resolve_references` | `GET /{type}/{id}` via `_include`-based search (already shipped, `treatment_mcp_server.py:179`) | ✅ | **The one CORE add.** Targets the diagnosed `{MedicationRequest, Medication}` failure — all 498 multi-type rows (17.0%) are exactly this pair; `medicationReference` is undereferenceable in one hop from baseline. **No surveyed server ships first-class reference resolution** (the-momentum/jcafazzo/xSoVx/wso2 all lack it) → real fix + genuine Medplum contribution. | **CORE** |
| 2 | `search_encounters` | `GET /Encounter?` typed (class/period) | ✅ | The real second typed-search worth isolating (13.4% slice). Field-precedent: the-momentum per-type routers, jcafazzo per-type searches (`fhir_mcp_server.py:121–141`). | TAIL |
| 3 | `search_procedures` | `GET /Procedure?` typed (date/encounter) | ✅ | **Deliberate curve-flattener data point** (3.7% ceiling). Same field-precedent. Included to *show* where added tools stop paying off — generic already covers it. | TAIL (optional) |

**Orthogonal arm (not stacked):** `c0` = `fhir_request_frugal` (already built, preset `c0`, `treatment_mcp_server.py:105`) — generic + default `_elements`. Run head-to-head against `control`, **not** cumulatively. Answers "could you just prompt the generic better / prune tokens?" — accuracy ~flat by design (xSoVx/wso2 bake `_elements` into every read: `fhir-tools.ts:73,99`).

---

## 2. The ablation order

```
control  →  c0  (orthogonal)                          ← efficiency axis, head-to-head
validated5  →  +resolve_references  →  +search_encounters  →  +search_procedures
   floor          CORE jump              TAIL probe           designed-flat
```

- **`control`** — Medplum's shipped single `fhir_request` tool (byte-for-byte). The honest floor; no reference resolution, no typed surface.
- **`validated5`** — current baseline (~55–60% expected floor).
- **`+resolve_references`** — **biggest and likely sole non-baseline mover.** Closes the entire 17% medication slice. If accuracy on `{MedicationRequest, Medication}` jumps while single-resource questions stay flat, the gain is clean and attributable.
- **`+search_encounters`** — modest lift bounded by the 13.4% slice and by how much the agent was losing to param-guessing on generic `search_fhir`.
- **`+search_procedures`** — bounded at 3.7%, designed to land flat.

**Where the curve flattens:** immediately after the first CORE add. `resolve_references` is the steep segment; typed-per-resource searches flatten on contact because (a) the generic already covers them and (b) their slices are ≤13%, mostly ≤4%. **The defensible POC finding is concentration, not a long smooth curve:** one tool addresses the only reference-spanning demand; everything after is sub-noise-floor or generic-redundant.

---

## 3. What we CUT and why

- **`search_with_include`** — duplicate of `resolve_references`, which is *already* `_include`-based (`treatment_mcp_server.py:179`). Two doors to one room pollutes the curve with fake "diminishing returns."
- **`search_medications` w/ baked `_include`** — same `MedicationRequest?_include=…:medication` call as `resolve_references`; merged, not duplicated.
- **`search_conditions`** — 1.2% ceiling (35 rows), below the noise floor at N=2931. A typed wrapper can't move a curve when its whole addressable slice is 35 questions.
- **`expand_valueset` / `lookup_code` (terminology)** — zero benchmark demand (codes are *given* as text, handled by `code:text`) **and** a content-loading precondition (see §4). Cut from the locked set; admissible only as a flagged right-tail probe.
- **Semantic / document search (Pinecone RAG)** — the-momentum ships it (`app/services/rag/*`); requires vector infra Medplum lacks + is projection/off-brand for a FHIR-purist substrate.
- **Cohort / aggregation** (`find_patients_with_conditions`, jcafazzo) — the aggregation wall; a *separate finding*, not a single-patient read tool.
- **Data-quality assessment** (`assess_data_quality`, jcafazzo `fhir_mcp_server.py:205`) — diagnostic, not retrieval.
- **All writes** (create/update/delete, shipped by the-momentum/xSoVx/wso2) — violate read-only; `$ai` is suggest-don't-act and MCP follows.
- **No more `$everything` variants** — the broadest read is already in the baseline; the curve's job is to test the *bounded* alternatives.

---

## 4. Terminology caveat — gates `$expand`/`$lookup`

**Partially works out of the box — verified, not assumed.**

- ✅ Medplum auto-seeds base-R4 + US-Core terminology on init (`seed.ts:37` → `rebuildR4ValueSets`; bundles in `seeds/valuesets.ts:14–25`). `$expand`/`$lookup` are real Postgres-backed SQL handlers (`operations/expand.ts`, `operations/codesystemlookup.ts`), not stubs. They work for any seeded ValueSet/CodeSystem.
- ❌ **LOINC, SNOMED CT, RxNorm — the systems MIMIC actually uses — are NOT bundled** (too large; Medplum docs say so explicitly, `docs/terminology/index.md`). Failure is **hard**: `lookup_code(system="http://loinc.org", …)` throws `OperationOutcomeError(badRequest('CodeSystem … not found'))` (`utils/terminology.ts:61,100`) → HTTP 400, not a degraded answer.

**Verdict:** terminology tools are **TAIL/optional and excluded from the locked 8.** Their flat contribution would be over-determined — both no benchmark demand *and* possibly missing content. If ever run, you MUST first either `$import` LOINC/RxNorm into the eval instance (`codesystemimport.ts` exists) or restrict to seeded US-Core value sets, and report **which case ran** — a flat result from *missing content* is a different, misleading finding versus a flat result from *no demand*. **Tools 1–8 carry no such precondition** ($everything, REST search/read, `_include`, `_elements`, `metadata` all work on stock demo data).

---

## 5. Field positioning

We are not inventing a tool surface — we are shipping the **convergent design of every serious open FHIR MCP server**, which Medplum's `$ai`/MCP demonstrably lacks (its shipped control is a *single* generic `fhir_request` tool, no typed surface, no reference resolution — verified). Of the locked additions, typed per-resource search has two independent precedents (the-momentum's `request_*_resource` routers; jcafazzo's `search_observations/conditions/medication_requests/…` at `fhir_mcp_server.py:121–141`), capability introspection is shipped by jcafazzo and xSoVx, and `_elements` frugality is xSoVx/wso2 convention. **Exactly one tool, `resolve_references`, is novel as a first-class agent tool** — the capability exists inside Medplum (`repo.readReferences()`, used by `patienteverything.ts`) but no surveyed server exposes it, and it targets the benchmark-diagnosed `MedicationRequest → Medication` failure. We deliberately omit the three non-retrieval tools these same servers ship — RAG/semantic search (the-momentum), cohort aggregation (jcafazzo), data-quality (jcafazzo) — each cuttable with a named precedent for *why* it's out of scope. The contribution is therefore both de-risked (consensus pattern, not a one-off) and pointed (one real gap closed).

**Files:** `FHIR-AgentBench/treatment_mcp_server.py` (current 5 + shipped `resolve_references`/`fhir_request_frugal`, lines 105/179/212–220); `FHIR-AgentBench/final_dataset/questions_answers_sql_fhir.csv` (`true_fhir_ids` ground truth: 498/2931 medication-spanning, Encounter 392, Procedure 108, Condition 35); `medplum/packages/server/src/fhir/operations/{patienteverything,expand,codesystemlookup}.ts`; `medplum/packages/server/src/fhir/operations/utils/terminology.ts:61,100`; `medplum/packages/server/src/seed.ts:37`; `medplum/packages/server/src/fhir/search.ts` (`_include` first-class); survey at `fhir-mcp-survey/{the-momentum_fhir-mcp-server,jcafazzo_fhir-mcp,xSoVx_fhir-mcp}`.