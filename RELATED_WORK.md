# Related work: how should an LLM agent read healthcare data?

*A cited landscape review that situates this fork's experiments
([FINDINGS.md](FINDINGS.md), [CODE_EXPERIMENT.md](CODE_EXPERIMENT.md),
[TRUSTWORTHY_REGRADE.md](TRUSTWORTHY_REGRADE.md)) in the broader field. Scope: single-patient and cohort
reads of clinical data, FHIR-centric. Sources were each fetched and fact-checked; star counts and venues are
as of 2026-06-23 and worth re-verifying before citing externally.*

---

## 0. The right axis: shape, not transport

"Reading healthcare data" conflates two questions: **(a) how the agent acquires the bytes** (a REST call, a
tool, a bulk export, a warehouse query) and **(b) what shape those bytes are in when they hit the model's
context** (raw nested FHIR JSON, a flat row, a retrieved note passage, a query result). The patterns below
differ mainly on (b), because (b) is what gates accuracy. The most-cited empirical result in the space is a
**dual bottleneck** — you need high retrieval recall, *but* flooding context with irrelevant FHIR impairs
reasoning, and FHIR's nested graph structure defeats natural-language-only reasoning [1][19]. That is a claim
about shape, not transport.

---

## 1. The six patterns

### Pattern A — Raw FHIR in context (REST, the model parses)
The agent issues live FHIR REST calls and the **raw nested resource JSON lands in context**; the model
navigates the graph itself.
- **Who uses it.** Medplum's MCP server exposes a single generic `fhir-request` tool (method/url/body),
  pushing all FHIR complexity onto the model [11][21]. HAPI FHIR is the de-facto OSS substrate this reads from
  [13]. **MedAgentBench** (Stanford, NEJM AI 2025) is the canonical benchmark of this pattern: 300
  clinician-authored tasks against a live HAPI server [2].
- **Against.** This is the pattern this fork's eval breaks: raw FHIR overflowed a 32k window on **64%
  (262/409)** of single-patient questions ([FINDINGS.md](FINDINGS.md)). FHIR-AgentBench's best agent ceilings
  at **50%**, attributed to "difficulty parsing the intricate, graph-like structure of FHIR even when the
  information is available" [1][19]. MedAgentBench v2 found model-authored raw HTTP FHIR "produced malformed
  requests" [3].

### Pattern B — Code-interpreter over fetched FHIR
Fetch the (large) blob, then let the agent **write and run code** to extract and reason.
- **Who uses it.** EHRAgent (EMNLP 2024) — code-as-action, but over relational EHRs, not FHIR [5].
  FHIR-AgentBench's best configuration is retriever + code interpreter [1]. `jmandel/health-record-mcp` (Josh
  Mandel, SMART on FHIR architect) fetches into SQLite and exposes `query_record` (read-only SQL) and
  `eval_record` (arbitrary JS) [18].
- **Against (this fork's key finding).** At matched token budget the code interpreter gives **no significant
  reasoning benefit (−3.6pp, p=0.18)**; its pooled gain is **overflow-avoidance**, not better reasoning, plus
  a reliability tax from buggy generated code ([CODE_EXPERIMENT.md](CODE_EXPERIMENT.md)). The literature's
  "code interpreter helps" is real but largely the *wrong explanation* — it keeps the blob out of context, it
  does not reason better. A code sandbox is also a governance liability (arbitrary code over PHI).

### Pattern C — SQL-on-FHIR / ViewDefinition projections (flat, typed, deterministic)
Define a **portable tabular projection** of FHIR via FHIRPath columns; the agent reads flat rows, never the
nested graph.
- **Who uses it.** HL7 **SQL-on-FHIR v2** ViewDefinition (single-resource, defers aggregation to downstream
  SQL) [14][29][30]; **FHIRPath** is an ANSI/HL7 normative standard [31]; **Pathling** (CSIRO) is the
  reference engine [15]. **FHIRPath-QA** (arXiv 2026) is the strongest single result for "synthesize an
  executable query, don't narrate": **391× fewer tokens** and failure rate 0.36→0.09 vs retrieval-first
  prompting [4]. Production analogues: Particle Health's "Flat" format [23]; Canvas Medical's read-only ORM
  read-model, never raw FHIR for reads [20].
- **Against.** A single ViewDefinition is single-resource and does not itself aggregate [14]; FHIRPath-QA
  still only hits ≤42% zero/few-shot — query *synthesis* is hard [4]. SQL-on-FHIR **explicitly leaves authz
  out of scope** ("access control left to the host") [29]; Pathling is row-level only [15].

### Pattern D — Bounded MCP / typed tools (named, schema-enforced FHIR actions)
A **fixed set of typed tools** (e.g. `searchObservations`) hides raw HTTP/FHIR formatting; the model calls
actions, not URLs. *(Deep-dive in §2 — this is the "typed tools" result.)*
- **Who uses it.** **MedAgentBench v2** (PSB 2026): bounded per-resource tools + a calculator + a typed
  `finish` tool + a plan-validate prompt lifted GPT-4.1 to **91%**, and memory to **98%** [3].
  `wso2/fhir-mcp-server` wraps a FHIR API as bounded tools + FHIRPath response filtering [16].
  `the-momentum/fhir-mcp-server` [17], `rkirkendall/medplum-mcp` (33 tools + escape hatch) [22]. FHIR-AgentEval
  (Boston Children's, 2026) uses a 5-tool CRUD MCP gateway [10]. The bounding standards: **SMART App Launch
  scopes** (a ceiling, not a grant) [32] and **SMART Backend Services** [33].
- **Against.** Still single-resource raw-FHIR underneath; the win is *input-formatting*, not a query layer; no
  GROUP BY, no aggregate ABAC [16].

### Pattern E — RAG / semantic retrieval over notes
For **unstructured text**, retrieve the relevant passages instead of dumping the chart. Myers et al. 2025:
targeted RAG over notes matches full-context at far fewer tokens (single-site preprint) [8]. **Lost in the
Middle** (TACL 2024) is the foundational *why*: long context degrades mid-context recall [9]. This pattern is
about notes, not structured FHIR.

### Pattern F — Bulk export → warehouse → governed store
Pull whole **cohorts**, normalize once, query a governed store. HL7 **Bulk Data `$export`** is the standard
cohort path [34]; Health Gorilla (TEFCA QHIN) [24]; Metriport (OSS consolidation) [26]; Innovaccer Gravity
(enterprise data fabric) [28]. Warehouse-scale and batch; aggregate-level ABAC/consent is absent in all of
them.

---

## 2. The "typed tools" result, in depth (and how it reconciles with this benchmark)

MedAgentBench v2 [3] is the cleanest "architecture, not model scale" result in the corpus, and it is worth
reading carefully because the popular one-liner ("typed tools = +21pp") over-credits it.

**What they actually changed (a bundle, plus a model swap):** v1's best was Claude 3.5 Sonnet v2 at 69.67%;
v2 ran **GPT-4.1** with five simultaneous changes — (1) bounded per-resource tools replacing raw-HTTP
construction, (2) a `calculator` tool, (3) a typed `finish` tool, (4) a plan-and-validate CoT system prompt,
(5) few-shot formatting examples — reaching 91%, then **98%** with a mem0-style memory. So "+21pp from typed
tools" **confounds the model change with a five-part bundle**; the only cleanly isolated component is
**memory (+7pp, 91→98)** on the same model and design.

**Two details that matter and are usually skipped:**

1. **Their bounded tools bound the *input*, not the *output*.** Reading their code, `fhir_observation_search`
   returns `requests.get(...).json()` — the **raw FHIR Bundle**, capped at `_count=200`. The tools remove
   malformed-HTTP and LOINC-guessing failures, but the response is still raw resources; the paper itself notes
   the 200-cap "may have [caused failures]… simply because it lacked access to all necessary information."
   Projecting the *output* (returning the rows/fields, not the Bundle) is a further step their tools do not
   take. FHIRPath-QA [4] is the evidence that output projection is where the large token/reliability wins are.

2. **Their "compute" tool is bounded arithmetic, not a code interpreter.** The `calculator` is a single
   expression evaluated against `{sum, math, datetime, Decimal}` — no loops, no statements. They aggregate
   client-side over the fetched Bundle. This is notable next to Pattern B: independent of each other, the
   strongest typed-tools system and this fork's eval both conclude *give the agent bounded deterministic
   compute, not arbitrary code over the record.*

**Reconciling with this fork's tool-ablation null.** This fork found that a **larger tool catalog did not
improve accuracy** ([REPORT.md](REPORT.md) — typed catalog vs one generic tool, null on two frontier models).
That is *not* in conflict with MedAgentBench v2:

- This fork varied **tool count / catalog richness with raw-FHIR output held constant** — adding more typed
  tools whose outputs are still raw resources. Null.
- MedAgentBench varied the **input-construction axis** (typed tool calls vs the model hand-writing HTTP) plus
  a prompt/calculator/finish bundle. Win.
- **Neither tested output projection** (returning bounded rows instead of raw resources). That axis is open.

So the honest synthesis on typed tools is: **bounding how the model *invokes* FHIR (typed calls, no
hand-written HTTP) helps; making the *catalog bigger* does not; and bounding what comes *back* (projection) is
the largely-untested axis the token-economics evidence points at.** The three claims are about three different
things and the field routinely collapses them into one.

---

## 3. How this benchmark's own findings fit

This fork contributes three results to the picture above (full detail in [FINDINGS.md](FINDINGS.md) /
[TRUSTWORTHY_REGRADE.md](TRUSTWORTHY_REGRADE.md)):

- **The raw-FHIR failure is overflow, quantified:** the no-code agent overflows 32k on 64% of single-patient
  questions. This is the same wall MedAgentBench v2 and FHIR-AgentBench hit, measured directly.
- **The celebrated code-interpreter "win" is overflow-avoidance, not reasoning** (−3.6pp at matched budget,
  p=0.18). In this FHIR-AgentBench setup, the observed code gain is overflow-driven rather than a measured
  reasoning gain.
- **Benchmark judges need auditing:** FHIR-AgentBench's default `gpt-5-mini` judge was only **61% accurate vs
  non-LLM numeric ground truth**, with a one-directional precision-punishing bias; multi-vote panels score
  98–99%. Published scores across this whole landscape deserve a skeptical eye.

---

## 4. Where the evidence agrees, and where it is contested

**Agreement.** Don't dump raw nested FHIR into context [1][3][19, this fork]. Retrieve/rank, don't stuff
[9][8]. Bounding *how* the model invokes FHIR beats free-form HTTP [3][16][32]. The single-patient ceiling is
mediocre (~50%) and improving via architecture, not scale [1][3]. In the standards and OSS systems reviewed
here, aggregate/cohort ABAC over FHIR remains unresolved [29][15][24][28].

**Contested.** Is the code interpreter load-bearing or a crutch? Literature says load-bearing [1][5]; this
fork's matched-budget result says crutch [CODE_EXPERIMENT.md]. FHIRPath vs SQL vs ViewDefinition as the query
substrate [4][29]. RAG vs long-context for notes (single-site evidence) [8][9]. Whether a new agent-native
substrate is needed (asserted, never measured) [27].

---

## 5. Open direction (not a measured claim)

The weight of evidence points toward **bounded, executable-query / projection access over FHIR** — the model
emits a query against a typed abstraction and gets back exactly the rows it needs — rather than raw-FHIR in
context or a code sandbox. Two honest caveats keep this an *open direction*, not a settled result:

1. **Output projection is largely untested for agent accuracy.** FHIRPath-QA shows the token win [4], but
   whether projected reads beat raw-Bundle bounded tools on end-task accuracy has not been measured head to
   head (it could be run on this fork's harness as a third arm — and the nearest analogue here,
   `_elements`-coaching, was a null). Projection also *relocates* difficulty to query-synthesis (≤42%
   un-tuned) [4].
2. **The governance gap is real and unfilled in this reviewed set.** SQL-on-FHIR, Pathling, and the warehouse
   vendors all explicitly leave authorization "to the host" [29][15], and I found no standard or OSS engine
   here that ships aggregate/cohort ABAC over FHIR [24][28]. Enforcing access *inside* a bounded query path is
   an open engineering problem, not solved infrastructure.

---

## Sources
[1] FHIR-AgentBench — Lee et al., ML4H 2025, arXiv 2509.19319.
[2] MedAgentBench — Stanford, NEJM AI 2025, arXiv 2501.14654.
[3] MedAgentBench v2 — Chen et al., PSB 2026 (bounded tools+calculator+finish+prompt+memory, 70→91→98% on GPT-4.1; only memory isolated). Code: github.com/ericoericochen/medagentbenchv2.
[4] FHIRPath-QA — arXiv 2026 (text-to-FHIRPath; 391× fewer tokens; ≤42% un-tuned synthesis).
[5] EHRAgent — EMNLP 2024 (code-as-action over relational EHRs), arXiv 2401.07128.
[6] EHRSQL — NeurIPS 2022 D&B, arXiv 2301.07695.
[7] EHRNoteQA — NeurIPS 2024 D&B, arXiv 2402.16040.
[8] RAG vs Long-Context over EHRs — Myers et al. 2025 (preprint), arXiv 2508.14817.
[9] Lost in the Middle — TACL 2024, arXiv 2307.03172.
[10] FHIR-AgentEval — Boston Children's, 2026, PMC12919212.
[11][12] medplum/medplum — github.com/medplum/medplum (MCP experimental, not-for-PHI).
[13] hapifhir/hapi-fhir — github.com/hapifhir/hapi-fhir.
[14][29][30] HL7 SQL-on-FHIR v2 — github.com/HL7/sql-on-fhir ; build.fhir.org/ig/FHIR/sql-on-fhir-v2/.
[15] aehrc/pathling — github.com/aehrc/pathling.
[16] wso2/fhir-mcp-server ; [17] the-momentum/fhir-mcp-server ; [18] jmandel/health-record-mcp ; [22] rkirkendall/medplum-mcp.
[19] glee4810/FHIR-AgentBench — github.com/glee4810/FHIR-AgentBench (upstream of this fork).
[20] Canvas Medical SDK Data module — docs.canvasmedical.com/sdk/data/.
[23] Particle Health — docs.particlehealth.com ; [24] Health Gorilla Patient360 ; [26] Metriport ; [28] Innovaccer Gravity.
[27] MedBeads — arXiv 2026 (theory only).
[31] FHIRPath — ANSI/HL7 Normative, hl7.org/fhirpath/.
[32] SMART App Launch v2.2.0 scopes ; [33] SMART Backend Services ; [34] FHIR Bulk Data $export v3.0.0.
